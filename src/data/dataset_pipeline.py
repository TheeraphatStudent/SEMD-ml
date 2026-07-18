import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from core import features_config, settings
from features import feature_extractor
from data.splitters import DatasetSplitter
from data.validators import DatasetValidator
from features.schema import build_feature_schema
from pipelines.dataset_build_pipeline import DatasetBuildPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPipeline:
    def __init__(self):
        self.datadict_config_path = settings.datadict_config_path
        self.extraction_path = settings.extraction_path
        self.random_state = settings.random_state
        self.test_size = settings.test_size
        self.validation_size = settings.validation_size
        self.data_dict = self._load_data_dict()
        self.validator = DatasetValidator(self.data_dict)
        self.classes = self.validator.classes
        self.class_mapping = self.validator.class_mapping
        self.build_pipeline = DatasetBuildPipeline(settings.dataset_path, self.validator)
        self.feature_schema = build_feature_schema(features_config)
        self.splitter = DatasetSplitter(
            random_state=self.random_state,
            test_size=self.test_size,
        )
        validation_relative_size = self.validation_size / max(1e-9, 1.0 - self.test_size)
        self.validation_splitter = DatasetSplitter(
            random_state=self.random_state,
            test_size=validation_relative_size,
        )
        self.last_validation_report: Optional[Dict[str, Any]] = None
        self.last_dataset_metadata: Optional[Dict[str, Any]] = None

    @property
    def dataset_path(self) -> str:
        return self.build_pipeline.repository.dataset_path

    @dataset_path.setter
    def dataset_path(self, value: str) -> None:
        self.build_pipeline.repository.dataset_path = value

    def _load_data_dict(self) -> Dict[str, Any]:
        data_dict_path = os.path.join(os.path.dirname(__file__), self.datadict_config_path)
        try:
            with open(data_dict_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except Exception as exc:
            logger.warning("Could not load data_dict.yaml: %s. Using defaults.", exc)
            return {
                "fields": {
                    "url": ["url", "input", "target", "text"],
                    "class": ["label", "class", "output", "type"],
                },
                "class_mapping": {
                    "benign": [0, "benign", "legitimate", "normal"],
                    "malicious": [1, 2, 3, "malicious", "malware", "phishing", "defacement", "redirect", "spam"],
                },
            }

    def load_and_merge_datasets(self, dataset_files: List[str]) -> pd.DataFrame:
        return self.build_pipeline.load_and_merge(dataset_files)

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        result = self.validator.validate(df)
        self.last_validation_report = result.to_dict()
        issues = result.errors + result.warnings
        return result.is_valid, issues

    def preprocess_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned, validation = self.validator.clean(df)
        self.last_validation_report = validation.to_dict()
        return cleaned

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        feature_rows = [feature_extractor.extract(url) for url in df["url"].tolist()]
        X = self.feature_schema.align_dataframe(pd.DataFrame(feature_rows))
        y = df["label"].reset_index(drop=True)

        features_with_metadata = X.copy()
        features_with_metadata.insert(0, "registered_domain", df["registered_domain"].reset_index(drop=True))
        features_with_metadata.insert(0, "url", df["url"].reset_index(drop=True))
        features_with_metadata["label"] = y

        os.makedirs(self.extraction_path, exist_ok=True)
        features_with_metadata.to_csv(os.path.join(self.extraction_path, "extracted_features.csv"), index=False)
        features_with_metadata.to_csv(os.path.join(self.extraction_path, "features_before_balance.csv"), index=False)
        return X, y

    def detect_imbalance(self, y: pd.Series) -> Dict[str, Any]:
        class_counts = Counter(y)
        counts = list(class_counts.values())
        if not counts:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 0.0,
                "severity": "unknown",
                "class_counts": {},
                "class_distribution": {},
                "total_samples": 0,
                "min_samples": 0,
                "max_samples": 0,
            }

        total = len(y)
        ratio = max(counts) / min(counts) if min(counts) else 0.0
        if ratio < 2.0:
            severity = "balanced"
            is_imbalanced = False
        elif ratio < 5.0:
            severity = "mild"
            is_imbalanced = True
        elif ratio < 10.0:
            severity = "moderate"
            is_imbalanced = True
        else:
            severity = "severe"
            is_imbalanced = True
        return {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": ratio,
            "severity": severity,
            "class_counts": dict(class_counts),
            "class_distribution": {key: value / total for key, value in class_counts.items()},
            "total_samples": total,
            "min_samples": min(counts),
            "max_samples": max(counts),
        }

    def select_balancing_method(self, imbalance_info: Dict[str, Any]) -> str:
        if not imbalance_info["is_imbalanced"]:
            return "none"
        if imbalance_info["min_samples"] < 6:
            return "oversampling"
        if imbalance_info["severity"] == "severe" and imbalance_info["max_samples"] > 10000:
            return "undersampling"
        return "smote" if imbalance_info["min_samples"] >= 6 else "oversampling"

    def apply_balancing(self, X: pd.DataFrame, y: pd.Series, method: str) -> Tuple[pd.DataFrame, pd.Series]:
        if method == "none":
            return X, y

        min_samples = min(Counter(y).values())
        k_neighbors = min(5, max(1, min_samples - 1))
        if method == "smote":
            sampler = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        elif method == "oversampling":
            sampler = RandomOverSampler(random_state=self.random_state)
        elif method == "undersampling":
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balance method: {method}")

        X_balanced, y_balanced = sampler.fit_resample(X, y)
        balanced = pd.DataFrame(X_balanced, columns=X.columns)
        balanced["label"] = y_balanced
        balanced.to_csv(
            os.path.join(self.extraction_path, f"features_after_balance_{method}.csv"),
            index=False,
        )
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

    def split_dataset(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
        split = self.splitter.split(X, y, groups)
        return split.X_train, split.X_test, split.y_train, split.y_test, split.strategy

    def prepare_dataset(
        self,
        dataset_files: List[str],
        apply_balancing: bool = True,
        manual_balance_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        build_result = self.build_pipeline.build(dataset_files)
        cleaned = build_result.cleaned
        dataset_metadata = build_result.dataset_metadata
        self.last_validation_report = build_result.validation_report
        self.last_dataset_metadata = dataset_metadata

        X, y = self.extract_features(cleaned)
        groups = cleaned["registered_domain"].fillna(cleaned["normalized_url"])
        train_test_split = self.splitter.split(X, y, groups)
        train_groups = groups.iloc[train_test_split.train_indices].reset_index(drop=True)
        train_val_split = self.validation_splitter.split(
            train_test_split.X_train,
            train_test_split.y_train,
            train_groups,
        )

        X_train = train_val_split.X_train
        X_val = train_val_split.X_test
        X_test = train_test_split.X_test
        y_train = train_val_split.y_train
        y_val = train_val_split.y_test
        y_test = train_test_split.y_test
        train_urls = cleaned["url"].iloc[train_test_split.train_indices].reset_index(drop=True)
        urls_train = train_urls.iloc[train_val_split.train_indices].reset_index(drop=True)
        urls_val = train_urls.iloc[train_val_split.test_indices].reset_index(drop=True)
        urls_test = cleaned["url"].iloc[train_test_split.test_indices].reset_index(drop=True)

        pre_balance = self.detect_imbalance(y_train)
        balance_method = "none"
        X_train_unbalanced = X_train.copy()
        y_train_unbalanced = y_train.copy()
        if apply_balancing:
            balance_method = manual_balance_method or self.select_balancing_method(pre_balance)
            X_train, y_train = self.apply_balancing(X_train, y_train, balance_method)
        post_balance = self.detect_imbalance(y_train)

        return {
            "X_train": X_train,
            "X_train_unbalanced": X_train_unbalanced,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_train_unbalanced": y_train_unbalanced,
            "y_val": y_val,
            "y_test": y_test,
            "groups_train": train_groups.iloc[train_val_split.train_indices].reset_index(drop=True),
            "groups_val": train_groups.iloc[train_val_split.test_indices].reset_index(drop=True),
            "groups_test": groups.iloc[train_test_split.test_indices].reset_index(drop=True),
            "urls_train": urls_train,
            "urls_val": urls_val,
            "urls_test": urls_test,
            "feature_names": self.feature_schema.feature_names,
            "feature_schema": self.feature_schema.to_dict(),
            "validation_report": self.last_validation_report,
            "dataset_metadata": dataset_metadata,
            "split_strategy": {
                "test_split": train_test_split.strategy,
                "validation_split": train_val_split.strategy,
            },
            "balance_method": balance_method,
            "train_imbalance_before": pre_balance,
            "train_imbalance_after": post_balance,
            "full_feature_frame": X,
        }


dataset_pipeline = DatasetPipeline()
