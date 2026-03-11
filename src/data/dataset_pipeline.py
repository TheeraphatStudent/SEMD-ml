import os
import pandas as pd
import numpy as np
import yaml
import zipfile
import tarfile
import gzip
import shutil
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
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


    def get_dataset_files_from_store(self, store_path: str) -> List[Dict[str, str]]:
        store_dir = Path(store_path)
        dataset_files = []
        
        archive_extensions = ['.zip', '.tar.gz', '.tgz', '.gz']
        
        for file_path in store_dir.iterdir():
            if file_path.is_file():
                file_name = file_path.name
                
                is_archive = False
                for ext in archive_extensions:
                    if file_name.endswith(ext):
                        is_archive = True
                        break
                
                if is_archive:
                    dataset_name = file_name
                    for ext in archive_extensions:
                        if dataset_name.endswith(ext):
                            dataset_name = dataset_name[:-len(ext)]
                            break
                    
                    clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name).lower()
                    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
                    
                    dataset_files.append({
                        'file_path': str(file_path),
                        'file_name': file_name,
                        'dataset_name': dataset_name,
                        'clean_name': clean_name
                    })
        
        logger.info(f"Found {len(dataset_files)} dataset files in {store_path}")
        return dataset_files

    def extract_single_archive(self, archive_path: str, extract_dir: str) -> List[str]:
        """Extract a single archive file and return list of CSV files."""
        archive_path = Path(archive_path)
        extract_dir = Path(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        
        csv_files = []
        temp_dir = extract_dir / f'_temp_{archive_path.stem}'
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_dir)
            elif archive_path.suffix == '.gz':
                output_file = temp_dir / archive_path.stem
                os.makedirs(temp_dir, exist_ok=True)
                with gzip.open(archive_path, 'rb') as gz_ref:
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(gz_ref, out_ref)
            
            # Find all CSV files
            csv_files = list(temp_dir.rglob('*.csv'))
            
            # Move CSV files to extract_dir
            result_files = []
            for csv_file in csv_files:
                dest_file = extract_dir / csv_file.name
                if dest_file.exists():
                    base_name = csv_file.stem
                    counter = 1
                    while dest_file.exists():
                        dest_file = extract_dir / f"{base_name}_{counter}.csv"
                        counter += 1
                shutil.move(str(csv_file), str(dest_file))
                result_files.append(str(dest_file))
            
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            return result_files
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {str(e)}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return []

    def extract_archive_to_raw(self, archive_path: str) -> List[str]:
        """Extract archive to dataset/raw directory, replacing existing files."""
        archive_path = Path(archive_path)
        raw_dir = Path(self.dataset_path)
        os.makedirs(raw_dir, exist_ok=True)
        
        temp_dir = raw_dir / f'_temp_{archive_path.stem}'
        
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif archive_path.name.endswith('.tar.gz') or archive_path.name.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(temp_dir)
            elif archive_path.suffix == '.gz':
                output_file = temp_dir / archive_path.stem
                os.makedirs(temp_dir, exist_ok=True)
                with gzip.open(archive_path, 'rb') as gz_ref:
                    with open(output_file, 'wb') as out_ref:
                        shutil.copyfileobj(gz_ref, out_ref)
            
            csv_files = list(temp_dir.rglob('*.csv'))
            
            result_files = []
            for csv_file in csv_files:
                dest_file = raw_dir / csv_file.name
                if dest_file.exists():
                    os.remove(dest_file)
                    logger.info(f"Replacing existing file: {dest_file.name}")
                shutil.move(str(csv_file), str(dest_file))
                result_files.append(str(dest_file))
                logger.info(f"Extracted: {csv_file.name} -> {dest_file}")
            
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            return result_files
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path}: {str(e)}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            return []

    def load_single_dataset_from_archive(self, archive_info: Dict[str, str]) -> Tuple[pd.DataFrame, str]:
        archive_path = archive_info['file_path']
        dataset_name = archive_info['dataset_name']
        
        csv_files = self.extract_archive_to_raw(archive_path)
        
        if not csv_files:
            raise ValueError(f"No CSV files found in archive: {archive_path}")
        
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, on_bad_lines='skip')
                if len(df.columns) == 1:
                    df = pd.read_csv(csv_file, sep=';', on_bad_lines='skip')
                standardized_df = self._standardize_dataframe(df)
                dataframes.append(standardized_df)
                logger.info(f"Loaded {len(standardized_df)} records from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")
        
        if not dataframes:
            raise ValueError(f"No valid data loaded from archive: {archive_path}")
        
        merged_df = pd.concat(dataframes, ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['url'], keep='first')
        
        logger.info(f"Dataset '{dataset_name}': {len(merged_df)} unique URLs")
        
        return merged_df, dataset_name

    def calculate_min_class_count_across_datasets(self, store_path: str) -> Dict[str, any]:
        dataset_files = self.get_dataset_files_from_store(store_path)
        
        dataset_stats = {}
        balanced_datasets = []
        single_class_datasets = []
        all_benign_dfs = []
        
        for archive_info in dataset_files:
            try:
                df, dataset_name = self.load_single_dataset_from_archive(archive_info)
                df = self.preprocess_dataset(df)
                
                class_counts = Counter(df['label'])
                has_both_classes = 'benign' in class_counts and 'malicious' in class_counts
                
                dataset_stats[dataset_name] = {
                    'total': len(df),
                    'class_counts': dict(class_counts),
                    'clean_name': archive_info['clean_name'],
                    'has_both_classes': has_both_classes,
                    'archive_info': archive_info
                }
                
                if has_both_classes:
                    balanced_datasets.append({
                        'name': dataset_name,
                        'benign_count': class_counts.get('benign', 0),
                        'malicious_count': class_counts.get('malicious', 0),
                        'min_class': min(class_counts.get('benign', 0), class_counts.get('malicious', 0))
                    })
                    benign_df = df[df['label'] == 'benign'].copy()
                    all_benign_dfs.append(benign_df)
                else:
                    single_class_datasets.append({
                        'name': dataset_name,
                        'class': list(class_counts.keys())[0],
                        'count': list(class_counts.values())[0]
                    })
                    logger.info(f"Dataset '{dataset_name}' has single class: {list(class_counts.keys())}")
                    
            except Exception as e:
                logger.error(f"Error processing {archive_info['file_name']}: {str(e)}")
        
        benign_merge_path = None
        total_benign = 0
        if all_benign_dfs:
            merged_benign = pd.concat(all_benign_dfs, ignore_index=True)
            merged_benign = merged_benign.drop_duplicates(subset=['url'], keep='first')
            merged_benign = merged_benign.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            total_benign = len(merged_benign)
            
            benign_merge_path = os.path.join(self.dataset_path, 'benign_merge.csv')
            merged_benign.to_csv(benign_merge_path, index=False)
            logger.info(f"Created benign_merge.csv with {total_benign} shuffled benign URLs")
        
        logger.info(f"\n{'='*50}")
        logger.info("DATASET ANALYSIS SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total datasets found: {len(dataset_stats)}")
        logger.info(f"Datasets with both classes: {len(balanced_datasets)}")
        logger.info(f"Single-class datasets: {len(single_class_datasets)}")
        logger.info(f"Total benign samples merged: {total_benign}")
        
        for d in balanced_datasets:
            logger.info(f"  [balanced] {d['name']}: benign={d['benign_count']}, malicious={d['malicious_count']}")
        for d in single_class_datasets:
            logger.info(f"  [single] {d['name']}: {d['class']}={d['count']}")
        
        return {
            'dataset_stats': dataset_stats,
            'num_datasets': len(dataset_stats),
            'balanced_datasets': balanced_datasets,
            'single_class_datasets': single_class_datasets,
            'benign_merge_path': benign_merge_path,
            'total_benign': total_benign
        }

    def prepare_dataset_obo(
        self,
        archive_info: Dict[str, str],
        apply_balancing: bool = True
    ) -> Dict[str, any]:
        df, dataset_name = self.load_single_dataset_from_archive(archive_info)
        
        is_valid, issues = self.validate_dataset(df)
        if not is_valid:
            raise ValueError(f"Dataset validation failed for {dataset_name}: {issues}")
        
        df = self.preprocess_dataset(df)
        
        class_counts = Counter(df['label'])
        logger.info(f"Original class distribution for {dataset_name}: {dict(class_counts)}")
        
        has_benign = 'benign' in class_counts
        has_malicious = 'malicious' in class_counts
        
        if not (has_benign and has_malicious):
            raise ValueError(f"Dataset '{dataset_name}' missing class. Has: {list(class_counts.keys())}. Need both 'benign' and 'malicious'.")
        
        min_class_count = min(class_counts.get('benign', 0), class_counts.get('malicious', 0))
        
        sampled_dfs = []
        for cls in ['benign', 'malicious']:
            cls_df = df[df['label'] == cls]
            cls_df_sampled = cls_df.sample(n=min_class_count, random_state=self.random_state)
            sampled_dfs.append(cls_df_sampled)
        
        df_sampled = pd.concat(sampled_dfs, ignore_index=True)
        df_sampled = df_sampled.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        sampled_counts = Counter(df_sampled['label'])
        logger.info(f"Balanced class distribution for {dataset_name}: {dict(sampled_counts)}")
        logger.info(f"Total samples: {len(df_sampled)} ({min_class_count} per class)")
        
        X, y = self.extract_features(df_sampled)
        
        imbalance_info = self.detect_imbalance(y)
        balancing_method = 'undersample_to_min'
        
        final_counts = Counter(y)
        logger.info(f"Final class distribution: {dict(final_counts)}")
        
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'imbalance_info': imbalance_info,
            'balancing_method': balancing_method,
            'feature_names': X.columns.tolist(),
            'dataset_name': dataset_name,
            'clean_name': archive_info['clean_name'],
            'original_size': len(df),
            'sampled_size': len(df_sampled),
            'samples_per_class': min_class_count
        }


    def prepare_dataset_single_class(
        self,
        archive_info: Dict[str, str],
        benign_merge_path: str = None
    ) -> Dict[str, any]:
        df, dataset_name = self.load_single_dataset_from_archive(archive_info)
        
        is_valid, issues = self.validate_dataset(df)
        if not is_valid:
            raise ValueError(f"Dataset validation failed for {dataset_name}: {issues}")
        
        df = self.preprocess_dataset(df)
        
        class_counts = Counter(df['label'])
        single_class = list(class_counts.keys())[0]
        single_class_count = class_counts[single_class]
        logger.info(f"Single-class dataset {dataset_name}: {dict(class_counts)}")
        
        if benign_merge_path and os.path.exists(benign_merge_path):
            benign_df = pd.read_csv(benign_merge_path)
            benign_df = benign_df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            benign_sample = benign_df.head(single_class_count)
            logger.info(f"Adding {len(benign_sample)} benign samples from benign_merge.csv")
            
            df_combined = pd.concat([df, benign_sample], ignore_index=True)
            df_combined = df_combined.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
            
            combined_counts = Counter(df_combined['label'])
            logger.info(f"Combined class distribution for {dataset_name}: {dict(combined_counts)}")
            
            X, y = self.extract_features(df_combined)
            
            imbalance_info = self.detect_imbalance(y)
            balancing_method = 'benign_merge'
            
            final_counts = Counter(y)
            logger.info(f"Final class distribution: {dict(final_counts)}")
            
            X_train, X_test, y_train, y_test = self.split_dataset(X, y)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'imbalance_info': imbalance_info,
                'balancing_method': balancing_method,
                'feature_names': X.columns.tolist(),
                'dataset_name': dataset_name,
                'clean_name': archive_info['clean_name'],
                'original_size': len(df),
                'sampled_size': len(df_combined),
                'samples_per_class': single_class_count,
                'is_single_class': False
            }
        else:
            logger.warning(f"No benign_merge.csv found, training {dataset_name} as single-class")
            X, y = self.extract_features(df)
            
            imbalance_info = {
                'is_imbalanced': False,
                'total_samples': len(y),
                'class_distribution': dict(class_counts),
                'imbalance_ratio': 1.0
            }
            
            X_train, X_test, y_train, y_test = self.split_dataset(X, y)
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'imbalance_info': imbalance_info,
                'balancing_method': 'none',
                'feature_names': X.columns.tolist(),
                'dataset_name': dataset_name,
                'clean_name': archive_info['clean_name'],
                'original_size': len(df),
                'sampled_size': len(df),
                'samples_per_class': single_class_count,
                'is_single_class': True
            }


dataset_pipeline = DatasetPipeline()
