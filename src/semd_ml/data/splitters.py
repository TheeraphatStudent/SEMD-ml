from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_indices: pd.Index
    test_indices: pd.Index
    strategy: Dict[str, Any]


class DatasetSplitter:
    def __init__(self, random_state: int = 42, test_size: float = 0.3):
        self.random_state = random_state
        self.test_size = test_size

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> DatasetSplit:
        group_count = groups.nunique(dropna=True)
        label_count = y.nunique(dropna=True)

        if group_count >= 2 and label_count >= 2:
            n_splits = max(2, round(1 / self.test_size))
            n_splits = min(n_splits, group_count)
            if n_splits >= 2:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                train_idx, test_idx = next(splitter.split(X, y, groups))
                if y.iloc[train_idx].nunique(dropna=True) >= 2 and y.iloc[test_idx].nunique(dropna=True) >= 2:
                    strategy = {
                        "name": "stratified_group_k_fold_holdout",
                        "n_splits": n_splits,
                        "group_field": "registered_domain",
                    }
                    return self._build_result(X, y, train_idx, test_idx, strategy)

        train_idx, test_idx = train_test_split(
            X.index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        strategy = {
            "name": "stratified_random_split",
            "group_field": None,
        }
        return self._build_result(X, y, train_idx, test_idx, strategy)

    def _build_result(self, X, y, train_idx, test_idx, strategy):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        return DatasetSplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            train_indices=X.index[train_idx],
            test_indices=X.index[test_idx],
            strategy=strategy,
        )
