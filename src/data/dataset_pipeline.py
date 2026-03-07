import os
import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import psutil

from core import settings, features_config
from features import feature_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPipeline:

    def __init__(self):
        self.dataset_path = settings.dataset_path
        self.datadict_config_path = settings.datadict_config_path
        self.extraction_path = settings.extraction_path
        self.random_state = settings.random_state
        self.test_size = settings.test_size
        self.data_dict = self._load_data_dict()
        self.classes = list(self.data_dict.get('class_mapping', {}).keys()) or [
            'benign', 'malicious']
        self.class_mapping = self._build_class_mapping()

    def _load_data_dict(self) -> Dict[str, any]:
        data_dict_path = os.path.join(
            os.path.dirname(__file__), self.datadict_config_path)
        try:
            with open(data_dict_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(
                f"Could not load data_dict.yaml: {e}. Using defaults.")
            return {
                'fields': {
                    'url': ['url', 'input', 'target'],
                    'class': ['label', 'class', 'output', 'type']
                },
                'class_mapping': {
                    'benign': [0, 'benign', 'legitimate', 'normal'],
                    'malicious': [1, 2, 3, 'malicious', 'malware', 'phishing', 'defacement', 'redirect', 'spam']
                },
                'default_class': 'malicious'
            }

    def _build_class_mapping(self) -> Dict[str, str]:
        mapping = {}
        class_mapping = self.data_dict.get('class_mapping', {})

        for target_class, values in class_mapping.items():
            target_class_lower = target_class.lower()
            for value in values:
                if isinstance(value, int):
                    mapping[str(value)] = target_class_lower
                else:
                    mapping[str(value).lower()] = target_class_lower

        return mapping

    def _normalize_label(self, label) -> str:
        if pd.isna(label) or label is None:
            return self.data_dict.get('default_class', 'malicious')

        if isinstance(label, (int, float)):
            label = str(int(label))

        label_lower = str(label).lower().strip()
        return self.class_mapping.get(label_lower, label_lower)

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        standardized_df = pd.DataFrame()

        url_fields = self.data_dict['fields']['url']
        class_fields = self.data_dict['fields']['class']
        default_class = self.data_dict.get('default_class', 'malicious')

        url_col = None
        for field in url_fields:
            if field in df.columns:
                url_col = field
                break

        if url_col is None:
            raise ValueError(
                f"No URL column found. Expected one of: {url_fields}")

        standardized_df['url'] = df[url_col]

        class_col = None
        for field in class_fields:
            if field in df.columns:
                class_col = field
                break

        if class_col is not None:
            standardized_df['label'] = df[class_col].apply(
                self._normalize_label)
        else:
            logger.info(
                f"No class column found. Using default class: {default_class}")
            standardized_df['label'] = default_class

        return standardized_df

    def load_and_merge_datasets(self, dataset_files: List[str]) -> pd.DataFrame:
        exist_full_path = os.path.join(self.dataset_path, 'merged.csv')

        if os.path.exists(exist_full_path):
            logger.info(
                f"Loading existing merged dataset from {exist_full_path}")
            return pd.read_csv(exist_full_path)

        files_to_load = []
        for file_path in dataset_files:
            if file_path in ['dataset/raw', 'raw']:
                target_dir = self.dataset_path
                if os.path.isdir(target_dir):
                    csv_files = [f for f in os.listdir(
                        target_dir) if f.endswith('.csv')]
                    files_to_load.extend(csv_files)
                    logger.info(
                        f"Found {len(csv_files)} CSV files in directory: {target_dir}")
                else:
                    logger.warning(f"Directory not found: {target_dir}")
            else:
                files_to_load.append(file_path)

        dataframes = []
        for file_path in files_to_load:
            full_path = os.path.join(self.dataset_path, file_path)
            if not os.path.exists(full_path):
                logger.warning(f"Dataset file not found: {full_path}")
                continue

            try:
                if file_path.endswith('.csv'):
                    try:
                        df = pd.read_csv(full_path)
                        if len(df.columns) == 1:
                            df = pd.read_csv(
                                full_path, sep=';', on_bad_lines='skip')
                    except pd.errors.ParserError:
                        df = pd.read_csv(full_path, sep=';',
                                         on_bad_lines='skip')
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(full_path)
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue

                standardized_df = self._standardize_dataframe(df)
                dataframes.append(standardized_df)
                logger.info(
                    f"Loaded and standardized dataset: {file_path} with {len(standardized_df)} records")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")

        if not dataframes:
            raise ValueError('No valid datasets loaded')

        merged_df = pd.concat(dataframes, ignore_index=True)
        initial_count = len(merged_df)
        logger.info(f"Initial merged dataset size: {initial_count} records")

        label_counts = merged_df.groupby('url')['label'].nunique()
        conflicting_urls = label_counts[label_counts > 1].index

        if len(conflicting_urls) > 0:
            logger.warning(
                f"Found {len(conflicting_urls)} URLs with conflicting labels")
            sample_conflicts = list(conflicting_urls[:5])
            logger.info(
                f"Sample conflicting URLs: {sample_conflicts}{'...' if len(conflicting_urls) > 5 else ''}")

            merged_df = merged_df[~merged_df['url'].isin(conflicting_urls)]
            logger.info(
                f"Removed {len(conflicting_urls)} URLs with conflicting labels")

        merged_df = merged_df.sort_values(
            by=['url', 'label']).drop_duplicates(subset=['url'], keep='first')
        final_count = len(merged_df)

        logger.info(
            f"After deduplication: {final_count} records (removed {initial_count - final_count} duplicates)")

        merged_df = merged_df.sort_values(by='label').reset_index(drop=True)

        merged_df.to_csv(exist_full_path, index=False)
        logger.info(
            f"Saved merged dataset: {final_count} records to {exist_full_path}")

        return merged_df

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = []

        if 'url' not in df.columns:
            issues.append("Missing 'url' column")

        if 'label' not in df.columns and 'type' not in df.columns:
            issues.append("Missing 'label' or 'type' column")

        if df.isnull().sum().sum() > 0:
            null_counts = df.isnull().sum()
            issues.append(
                f"Dataset contains null values: {null_counts[null_counts > 0].to_dict()}")

        if 'label' in df.columns:
            normalized_labels = df['label'].apply(self._normalize_label)
            unique_labels = normalized_labels.unique()
            invalid_labels = [
                label for label in unique_labels if label not in self.classes]
            if invalid_labels:
                issues.append(f"Invalid labels found: {invalid_labels}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.drop_duplicates(subset=['url'], keep='first')
        logger.info(f"After removing duplicates: {len(df)} records")

        df = df.dropna(subset=['url'])

        if 'type' in df.columns and 'label' not in df.columns:
            df['label'] = df['type']

        df['label'] = df['label'].str.lower().str.strip()

        df['label'] = df['label'].apply(self._normalize_label)

        df = df[df['label'].isin(self.classes)]
        logger.info(f"After filtering valid labels: {len(df)} records")

        return df

    def _extract_single_url_features(self, url_data: Tuple[str, str]) -> Tuple[str, Dict, str]:
        url, label = url_data
        try:
            features = feature_extractor.extract(url)
            return url, features, label
        except Exception as e:
            logger.error(f"Error extracting features from URL {url}: {str(e)}")
            return url, {}, label

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info(
            'Extracting features from URLs using parallel processing...')

        cpu_cores = cpu_count()
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB

        max_processes_by_cpu = max(1, int(cpu_cores * 0.8))
        max_processes_by_memory = max(1, int(available_memory / 0.5))
        num_processes = min(max_processes_by_cpu,
                            max_processes_by_memory, len(df))

        logger.info(
            f"Using {num_processes} processes for feature extraction (CPU cores: {cpu_cores}, Available memory: {available_memory:.1f}GB)")

        url_data_list = [(row['url'], row['label'])
                         for _, row in df.iterrows()]

        with Pool(processes=num_processes) as pool:
            results = pool.map(
                self._extract_single_url_features, url_data_list)

        features_list = []
        urls_list = []
        labels_list = []

        for url, features, label in results:
            features_list.append(features)
            urls_list.append(url)
            labels_list.append(label)

        X = pd.DataFrame(features_list)
        y = pd.Series(labels_list)

        X = X.fillna(0)

        features_with_url = X.copy()
        features_with_url.insert(0, 'url', urls_list)
        features_with_url['label'] = y.values

        output_path = os.path.join(
            self.extraction_path, 'extracted_features.csv')
        features_with_url.to_csv(output_path, index=False)
        logger.info(f"Saved extracted features to {output_path}")

        extraction_dir = os.path.join(
            os.path.dirname(self.dataset_path), 'extraction')
        os.makedirs(extraction_dir, exist_ok=True)

        before_balance_path = os.path.join(
            extraction_dir, 'features_before_balance.csv')
        features_with_url.to_csv(before_balance_path, index=False)
        logger.info(
            f"Saved features before balancing to {before_balance_path}")

        logger.info(
            f"Extracted {X.shape[1]} features from {X.shape[0]} URLs using parallel processing")

        return X, y

    def detect_imbalance(self, y: pd.Series) -> Dict[str, any]:
        class_counts = Counter(y)
        total = len(y)

        class_distribution = {cls: count /
                              total for cls, count in class_counts.items()}

        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / \
            min_count if min_count > 0 else float('inf')

        if imbalance_ratio < 2.0:
            severity = 'balanced'
            is_imbalanced = False
        elif imbalance_ratio < 5.0:
            severity = 'mild'
            is_imbalanced = True
        elif imbalance_ratio < 10.0:
            severity = 'moderate'
            is_imbalanced = True
        elif imbalance_ratio < 20.0:
            severity = 'severe'
            is_imbalanced = True
        else:
            severity = 'extreme'
            is_imbalanced = True

        imbalance_info = {
            'is_imbalanced': is_imbalanced,
            'imbalance_ratio': imbalance_ratio,
            'severity': severity,
            'class_counts': dict(class_counts),
            'class_distribution': class_distribution,
            'total_samples': total,
            'min_samples': min_count,
            'max_samples': max_count
        }

        logger.info(f"Dataset imbalance analysis:")
        logger.info(
            f"  - Severity: {severity.upper()} (ratio: {imbalance_ratio:.2f})")
        logger.info(f"  - Class counts: {dict(class_counts)}")
        logger.info(f"  - Class distribution: {class_distribution}")

        return imbalance_info

    def select_balancing_method(self, imbalance_info: Dict[str, any]) -> str:
        if not imbalance_info['is_imbalanced']:
            logger.info('Dataset is balanced')
            return 'none'

        severity = imbalance_info['severity']
        imbalance_ratio = imbalance_info['imbalance_ratio']
        min_samples = imbalance_info['min_samples']
        max_samples = imbalance_info['max_samples']
        total_samples = imbalance_info['total_samples']
        class_counts = imbalance_info['class_counts']

        if min_samples < 6:
            method = 'oversampling'
            example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (เพิ่ม {max_samples - min_samples})'
        elif severity == 'mild':
            if total_samples < 1000:
                method = 'oversampling'
                example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (คัดลอกซ้ำ {max_samples - min_samples} ตัวอย่าง)'
            else:
                method = 'smote'
                example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (สร้างใหม่ {max_samples - min_samples} ตัวอย่าง)'
        elif severity in ['moderate', 'severe']:
            if min_samples < 50:
                method = 'oversampling'
                example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (คัดลอกซ้ำ {max_samples - min_samples} ตัวอย่าง)'
            else:
                method = 'smote'
                example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (สร้างใหม่ {max_samples - min_samples} ตัวอย่าง)'
        else:
            if max_samples > 10000:
                method = 'undersampling'
                example = f'Sample: คลาสมาก {max_samples} → {min_samples} ตัวอย่าง (ลดลง {max_samples - min_samples} ตัวอย่าง)'
            else:
                method = 'smote'
                example = f'Sample: คลาสน้อย {min_samples} → {max_samples} ตัวอย่าง (สร้างใหม่ {max_samples - min_samples} ตัวอย่าง)'

        logger.info(f"  - {example}")
        logger.info(f"  - Imbalance: {severity}, Ratio: {imbalance_ratio:.2f}")
        return method

    def apply_balancing(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if method == 'none':
            logger.info('No balancing applied')
            return X, y

        logger.info(f"Applying {method.upper()} balancing...")

        original_size = len(X)
        min_samples = min(Counter(y).values())
        k_neighbors = min(5, max(1, min_samples - 1))

        try:
            if method == 'smote':
                sampler = SMOTE(random_state=self.random_state,
                                k_neighbors=k_neighbors)
                X_balanced, y_balanced = sampler.fit_resample(X, y)

            elif method == 'oversampling':
                sampler = RandomOverSampler(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X, y)

            elif method == 'undersampling':
                sampler = RandomUnderSampler(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X, y)

            else:
                logger.warning(
                    f"Unknown balancing method: {method}, using oversampling")
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X, y)

        except Exception as e:
            logger.warning(f"{method.upper()} ล้มเหลว: {str(e)}")
            if method == 'smote':
                from imblearn.over_sampling import RandomOverSampler
                sampler = RandomOverSampler(random_state=self.random_state)
                X_balanced, y_balanced = sampler.fit_resample(X, y)
            else:
                raise

        new_class_counts = Counter(y_balanced)
        logger.info(
            f"Balancing complete: {original_size} -> {len(X_balanced)} samples")
        logger.info(f"New class distribution: {dict(new_class_counts)}")

        for cls, count in new_class_counts.items():
            original_count = Counter(y).get(cls, 0)
            change = count - original_count
            logger.info(
                f"  - Class '{cls}': {original_count} -> {count} ({change:+d})")

        extraction_dir = os.path.join(
            os.path.dirname(self.dataset_path), 'extraction')
        os.makedirs(extraction_dir, exist_ok=True)

        balanced_features = X_balanced.copy()
        balanced_features['label'] = y_balanced

        after_balance_path = os.path.join(
            extraction_dir, f'features_after_balance_{method}.csv')
        balanced_features.to_csv(after_balance_path, index=False)
        logger.info(f"Saved balanced features to {after_balance_path}")

        return X_balanced, y_balanced

    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def prepare_dataset(
        self,
        dataset_files: List[str],
        apply_balancing: bool = True,
        manual_balance_method: Optional[str] = None
    ) -> Dict[str, any]:
        df = self.load_and_merge_datasets(dataset_files)

        is_valid, issues = self.validate_dataset(df)
        if not is_valid:
            raise ValueError(f"Dataset validation failed: {issues}")

        df = self.preprocess_dataset(df)

        X, y = self.extract_features(df)

        imbalance_info = self.detect_imbalance(y)

        if apply_balancing:
            if manual_balance_method:
                if manual_balance_method not in settings.valid_balance_methods:
                    raise ValueError(
                        f"Invalid balance method: {manual_balance_method}. Valid options: {settings.valid_balance_methods}")
                balancing_method = manual_balance_method
                logger.info(f"Using manual balance method: {balancing_method}")
            else:
                balancing_method = self.select_balancing_method(imbalance_info)

            X, y = self.apply_balancing(X, y, balancing_method)
        else:
            balancing_method = 'none'

        X_train, X_test, y_train, y_test = self.split_dataset(X, y)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'imbalance_info': imbalance_info,
            'balancing_method': balancing_method,
            'feature_names': X.columns.tolist()
        }


dataset_pipeline = DatasetPipeline()
