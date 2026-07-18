from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from ml.model_factory import model_factory


class TrainingPipelineBuilder:
    """Builds the scaler -> balancer -> estimator sklearn pipeline used for both
    a single fit and cross-validation, and runs the cross-validation itself.

    Scaling happens in exactly one place: `build_pipeline`'s "scaler" step. Both
    `MLPipeline.train_models` (single fit) and `cross_validate` (CV folds) go
    through this same builder, so there is no separate scaling path to drift.
    """

    def __init__(self, random_state: int, cv_folds: int) -> None:
        self.random_state = random_state
        self.cv_folds = cv_folds

    def build_pipeline(
        self,
        algorithm: str,
        balance_method: str,
        y_train: pd.Series,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> ImbPipeline:
        steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
        sampler = self._build_sampler(balance_method, y_train)
        if sampler is not None:
            steps.append(("balancer", sampler))
        estimator = model_factory.build(algorithm, overrides=hyperparameters)
        steps.append(("estimator", estimator))
        return ImbPipeline(steps)

    def _build_sampler(self, method: str, y_train: pd.Series) -> Optional[Any]:
        if method == "none":
            return None
        if method == "oversampling":
            return RandomOverSampler(random_state=self.random_state)
        if method == "undersampling":
            return RandomUnderSampler(random_state=self.random_state)
        if method == "smote":
            class_counts = y_train.value_counts()
            min_samples = int(class_counts.min()) if not class_counts.empty else 0
            k_neighbors = min(5, max(1, min_samples - 1))
            return SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
        raise ValueError(f"Unknown balance method: {method}")

    def cross_validate(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_train_decoded: pd.Series,
        groups_train: Optional[pd.Series],
        balance_method: str,
    ) -> Dict[str, Any]:
        pipeline = self.build_pipeline(
            algorithm=algorithm,
            balance_method=balance_method,
            y_train=y_train_decoded,
        )
        min_class_count = int(pd.Series(y_train).value_counts().min())
        n_splits = max(2, min(self.cv_folds, min_class_count))
        if groups_train is not None and len(pd.Series(groups_train).dropna().unique()) >= n_splits:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                groups=groups_train,
                cv=splitter,
                scoring="f1",
                n_jobs=1,
            )
            strategy = "stratified_group_k_fold"
        else:
            splitter = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            )
            scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=splitter,
                scoring="f1",
                n_jobs=1,
            )
            strategy = "stratified_k_fold"

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "n_splits": n_splits,
            "strategy": strategy,
        }
