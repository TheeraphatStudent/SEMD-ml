from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core import features_config, settings
from features import feature_extractor
from ml.model_factory import model_factory
from semd_ml.features.schema import FeatureSchema, build_feature_schema


@dataclass
class TrainingArtifact:
    run_id: str
    artifact_path: str
    algorithm: str
    metrics: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    cv_mean: float
    cv_std: float
    training_duration_seconds: float
    pipeline: ImbPipeline
    metadata: Dict[str, Any]


class MLPipeline:
    def __init__(self) -> None:
        self.random_state = settings.random_state
        self.cv_folds = settings.cv_folds
        self.models_path = settings.models_path
        self.feature_schema = build_feature_schema(features_config)
        self.runtime_feature_schema = self.feature_schema
        self.label_encoder = LabelEncoder().fit(["benign", "malicious"])
        self.loaded_artifact: Optional[Dict[str, Any]] = None
        self.best_model: Optional[ImbPipeline] = None
        self.best_algorithm: Optional[str] = None
        self.best_params: Dict[str, Any] = {}
        self.current_metadata: Dict[str, Any] = {}
        os.makedirs(self.models_path, exist_ok=True)

    def get_algorithm_configs(self) -> Dict[str, Dict[str, Any]]:
        return settings.algorithm_hyperparameters

    def available_algorithms(self) -> list[str]:
        return model_factory.identifiers()

    def generate_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"run_{timestamp}_{uuid4().hex[:8]}"

    def build_training_pipeline(
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

    def train_models(
        self,
        dataset_result: Dict[str, Any],
        algorithms: Iterable[str],
        run_id: Optional[str] = None,
        git_commit_sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        run_id = run_id or self.generate_run_id()
        X_train = dataset_result["X_train_unbalanced"]
        X_val = dataset_result["X_val"]
        X_test = dataset_result["X_test"]
        y_train = dataset_result["y_train_unbalanced"]
        y_val = dataset_result["y_val"]
        y_test = dataset_result["y_test"]
        groups_train = dataset_result.get("groups_train")
        balance_method = dataset_result["balance_method"]
        feature_schema = FeatureSchema.from_dict(dataset_result["feature_schema"])
        self.feature_schema = feature_schema
        self.runtime_feature_schema = build_feature_schema(features_config)

        y_train_encoded = pd.Series(self.label_encoder.transform(y_train), index=y_train.index)
        y_val_encoded = pd.Series(self.label_encoder.transform(y_val), index=y_val.index)
        y_test_encoded = pd.Series(self.label_encoder.transform(y_test), index=y_test.index)

        results: Dict[str, Any] = {}
        best_algorithm = None
        best_score = float("-inf")
        best_artifact: Optional[TrainingArtifact] = None

        for algorithm in algorithms:
            start = time.perf_counter()
            pipeline = self.build_training_pipeline(algorithm, balance_method, y_train)
            pipeline.fit(X_train, y_train_encoded)
            training_duration_seconds = time.perf_counter() - start

            train_metrics = self.evaluate_model(pipeline, X_train, y_train_encoded)
            validation_metrics = self.evaluate_model(pipeline, X_val, y_val_encoded)
            test_metrics = self.evaluate_model(pipeline, X_test, y_test_encoded)
            cv_stats = self.cross_validate(
                algorithm=algorithm,
                X_train=X_train,
                y_train=y_train_encoded,
                groups_train=groups_train,
                balance_method=balance_method,
            )

            metadata = self._build_artifact_metadata(
                algorithm=algorithm,
                dataset_result=dataset_result,
                metrics=test_metrics,
                validation_metrics=validation_metrics,
                configuration={
                    "random_state": self.random_state,
                    "balance_method": balance_method,
                    "hyperparameters": self.get_algorithm_configs()[algorithm],
                    "cv_folds": cv_stats["n_splits"],
                },
                git_commit_sha=git_commit_sha,
            )

            artifact_path = self._artifact_path(run_id, algorithm)
            self.save_artifact(
                artifact_path=artifact_path,
                algorithm=algorithm,
                pipeline=pipeline,
                feature_schema=feature_schema,
                metadata=metadata,
            )

            result = {
                "algorithm": algorithm,
                "run_id": run_id,
                "artifact_path": artifact_path,
                "hyperparameters": self.get_algorithm_configs()[algorithm],
                "train_metrics": train_metrics,
                "validation_metrics": validation_metrics,
                "metrics": test_metrics,
                "training_duration_seconds": training_duration_seconds,
                "prediction_latency_ms": test_metrics["prediction_latency_ms"],
                "cross_validation_mean": cv_stats["mean"],
                "cross_validation_std": cv_stats["std"],
                "cross_validation_strategy": cv_stats["strategy"],
                "metadata": metadata,
            }
            results[algorithm] = result

            selection_score = validation_metrics["malicious_f1"]
            if selection_score > best_score:
                best_score = selection_score
                best_algorithm = algorithm
                best_artifact = TrainingArtifact(
                    run_id=run_id,
                    artifact_path=artifact_path,
                    algorithm=algorithm,
                    metrics=test_metrics,
                    validation_metrics=validation_metrics,
                    cv_mean=cv_stats["mean"],
                    cv_std=cv_stats["std"],
                    training_duration_seconds=training_duration_seconds,
                    pipeline=pipeline,
                    metadata=metadata,
                )

        if best_artifact is None or best_algorithm is None:
            raise ValueError("No algorithms trained successfully")

        self.best_model = best_artifact.pipeline
        self.best_algorithm = best_algorithm
        self.best_params = self.get_algorithm_configs()[best_algorithm]
        self.current_metadata = best_artifact.metadata
        self.load_artifact(best_artifact.artifact_path, validate_runtime_schema=False)

        return {
            "run_id": run_id,
            "best_algorithm": best_algorithm,
            "best_artifact_path": best_artifact.artifact_path,
            "best_metrics": best_artifact.metrics,
            "best_validation_metrics": best_artifact.validation_metrics,
            "results": results,
        }

    def cross_validate(
        self,
        algorithm: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: Optional[pd.Series],
        balance_method: str,
    ) -> Dict[str, Any]:
        pipeline = self.build_training_pipeline(
            algorithm=algorithm,
            balance_method=balance_method,
            y_train=self.label_encoder.inverse_transform(y_train),
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

    def evaluate_model(self, pipeline: ImbPipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, Any]:
        predictions = pipeline.predict(X_eval)
        probabilities = self._predict_probabilities(pipeline, X_eval)
        cm = confusion_matrix(y_eval, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        latency_ms = self._measure_prediction_latency_ms(pipeline, X_eval)
        metrics = {
            "accuracy": float(accuracy_score(y_eval, predictions)),
            "malicious_precision": float(precision_score(y_eval, predictions, pos_label=1, zero_division=0)),
            "malicious_recall": float(recall_score(y_eval, predictions, pos_label=1, zero_division=0)),
            "malicious_f1": float(f1_score(y_eval, predictions, pos_label=1, zero_division=0)),
            "macro_precision": float(precision_score(y_eval, predictions, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_eval, predictions, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_eval, predictions, average="macro", zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "confusion_matrix": cm.tolist(),
            "prediction_latency_ms": latency_ms,
        }
        metrics["roc_auc"] = self._roc_auc(y_eval, probabilities)
        return metrics

    def _predict_probabilities(self, pipeline: ImbPipeline, X_eval: pd.DataFrame) -> Optional[np.ndarray]:
        estimator = pipeline.named_steps["estimator"]
        if hasattr(estimator, "predict_proba"):
            return pipeline.predict_proba(X_eval)
        return None

    def _roc_auc(self, y_eval: pd.Series, probabilities: Optional[np.ndarray]) -> Optional[float]:
        if probabilities is None or len(np.unique(y_eval)) < 2:
            return None
        return float(roc_auc_score(y_eval, probabilities[:, 1]))

    def _measure_prediction_latency_ms(self, pipeline: ImbPipeline, X_eval: pd.DataFrame) -> float:
        sample_count = min(10, len(X_eval))
        if sample_count == 0:
            return 0.0
        durations = []
        for idx in range(sample_count):
            start = time.perf_counter()
            pipeline.predict(X_eval.iloc[[idx]])
            durations.append((time.perf_counter() - start) * 1000.0)
        return float(np.mean(durations))

    def _artifact_path(self, run_id: str, algorithm: str) -> str:
        return os.path.join(self.models_path, f"{algorithm}_{run_id}.joblib")

    def save_artifact(
        self,
        artifact_path: str,
        algorithm: str,
        pipeline: ImbPipeline,
        feature_schema: FeatureSchema,
        metadata: Dict[str, Any],
    ) -> str:
        payload = {
            "algorithm": algorithm,
            "pipeline": pipeline,
            "feature_schema": feature_schema.to_dict(),
            "metadata": metadata,
            "label_encoder_classes": self.label_encoder.classes_.tolist(),
        }
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, artifact_path)
        return artifact_path

    def load_artifact(self, artifact_reference: str, validate_runtime_schema: bool = True) -> bool:
        artifact_path = self._resolve_artifact_path(artifact_reference)
        payload = joblib.load(artifact_path)
        runtime_schema = build_feature_schema(features_config)
        artifact_schema = FeatureSchema.from_dict(payload["feature_schema"])
        if validate_runtime_schema and runtime_schema.schema_version != artifact_schema.schema_version:
            raise ValueError(
                "Feature schema version mismatch: "
                f"runtime={runtime_schema.schema_version}, artifact={artifact_schema.schema_version}"
            )

        self.loaded_artifact = payload
        self.best_model = payload["pipeline"]
        self.best_algorithm = payload["algorithm"]
        self.feature_schema = artifact_schema
        self.runtime_feature_schema = runtime_schema
        self.current_metadata = payload["metadata"]
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(payload["label_encoder_classes"])
        return True

    def _resolve_artifact_path(self, artifact_reference: str) -> str:
        if artifact_reference in (None, "", "latest"):
            return self.latest_artifact_path()
        candidate = Path(artifact_reference)
        if candidate.exists():
            return str(candidate)

        model_dir = Path(self.models_path)
        matches = sorted(model_dir.glob(f"*{artifact_reference}*.joblib"))
        if matches:
            return str(matches[-1])
        raise FileNotFoundError(f"Artifact not found: {artifact_reference}")

    def latest_artifact_path(self) -> str:
        matches = sorted(Path(self.models_path).glob("*.joblib"))
        if not matches:
            raise FileNotFoundError("No packaged model artifacts were found")
        return str(matches[-1])

    def predict(self, url: str) -> Dict[str, Any]:
        if self.best_model is None or self.feature_schema is None:
            raise ValueError("No model loaded. Train or load a model first.")

        start = time.perf_counter()
        features = feature_extractor.extract(url)
        features = self.feature_schema.align_record(features)
        X = pd.DataFrame([features], columns=self.feature_schema.feature_names)
        probabilities = self._predict_probabilities(self.best_model, X)
        prediction = self.best_model.predict(X)[0]
        confidence = 1.0
        if probabilities is not None:
            confidence = float(np.max(probabilities[0]))
        prediction_label = self.label_encoder.inverse_transform([prediction])[0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "url": url,
            "prediction": prediction_label,
            "is_malicious": prediction_label == "malicious",
            "confidence": confidence,
            "feature_schema_version": self.feature_schema.schema_version,
            "prediction_time_ms": round(elapsed_ms, 2),
        }

    def _build_artifact_metadata(
        self,
        algorithm: str,
        dataset_result: Dict[str, Any],
        metrics: Dict[str, Any],
        validation_metrics: Dict[str, Any],
        configuration: Dict[str, Any],
        git_commit_sha: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "algorithm": algorithm,
            "dataset_version": dataset_result["dataset_metadata"]["dataset_version"],
            "dataset_hash": dataset_result["dataset_metadata"]["dataset_hash"],
            "feature_schema_version": dataset_result["feature_schema"]["schema_version"],
            "git_commit_sha": git_commit_sha,
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "validation_metrics": validation_metrics,
            "configuration": configuration,
            "feature_expectations": dataset_result["feature_schema"],
            "class_labels": self.label_encoder.classes_.tolist(),
        }


ml_pipeline = MLPipeline()
