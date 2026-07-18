from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import LabelEncoder

from core import features_config, settings
from features.schema import FeatureSchema, build_feature_schema
from ml import evaluation, inference
from ml.artifacts import ArtifactStore
from ml.model_factory import model_factory
from ml.training import TrainingPipelineBuilder


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
    """Orchestrates a training run: dataset -> per-algorithm fit/evaluate/save ->
    best-algorithm selection -> reload as the active model.

    Delegates to focused collaborators instead of doing everything inline:
    - `ml.training.TrainingPipelineBuilder` builds the scaler/balancer/estimator
      pipeline and runs cross-validation (single scaling step, shared by fit and CV).
    - `ml.evaluation` computes metrics for a fitted pipeline (pure functions).
    - `ml.artifacts.ArtifactStore` saves/loads `.joblib` artifacts (pure I/O).
    - `ml.inference` runs a single URL through the currently loaded model.
    This class owns the mutable state (best_model, label_encoder, loaded_artifact, ...)
    that those collaborators are deliberately stateless with respect to.
    """

    def __init__(self) -> None:
        self.random_state = settings.random_state
        self.cv_folds = settings.cv_folds
        self._artifact_store = ArtifactStore(settings.models_path)
        self._training_builder = TrainingPipelineBuilder(random_state=self.random_state, cv_folds=self.cv_folds)
        self.feature_schema = build_feature_schema(features_config)
        self.runtime_feature_schema = self.feature_schema
        self.label_encoder = LabelEncoder().fit(["benign", "malicious"])
        self.loaded_artifact: Optional[Dict[str, Any]] = None
        self.best_model: Optional[ImbPipeline] = None
        self.best_algorithm: Optional[str] = None
        self.best_params: Dict[str, Any] = {}
        self.current_metadata: Dict[str, Any] = {}

    @property
    def models_path(self) -> str:
        return self._artifact_store.models_path

    @models_path.setter
    def models_path(self, value: str) -> None:
        self._artifact_store.models_path = value
        os.makedirs(value, exist_ok=True)

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
        return self._training_builder.build_pipeline(algorithm, balance_method, y_train, hyperparameters)

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
        return self._training_builder.cross_validate(
            algorithm=algorithm,
            X_train=X_train,
            y_train=y_train,
            y_train_decoded=pd.Series(self.label_encoder.inverse_transform(y_train), index=y_train.index),
            groups_train=groups_train,
            balance_method=balance_method,
        )

    def evaluate_model(self, pipeline: ImbPipeline, X_eval: pd.DataFrame, y_eval: pd.Series) -> Dict[str, Any]:
        return evaluation.evaluate_model(pipeline, X_eval, y_eval)

    def _predict_probabilities(self, pipeline: ImbPipeline, X_eval: pd.DataFrame) -> Optional[np.ndarray]:
        return evaluation.predict_probabilities(pipeline, X_eval)

    def _roc_auc(self, y_eval: pd.Series, probabilities: Optional[np.ndarray]) -> Optional[float]:
        return evaluation.roc_auc(y_eval, probabilities)

    def _measure_prediction_latency_ms(self, pipeline: ImbPipeline, X_eval: pd.DataFrame) -> float:
        return evaluation.measure_prediction_latency_ms(pipeline, X_eval)

    def _artifact_path(self, run_id: str, algorithm: str) -> str:
        return self._artifact_store.artifact_path(run_id, algorithm)

    def save_artifact(
        self,
        artifact_path: str,
        algorithm: str,
        pipeline: ImbPipeline,
        feature_schema: FeatureSchema,
        metadata: Dict[str, Any],
    ) -> str:
        return self._artifact_store.save(
            artifact_path=artifact_path,
            algorithm=algorithm,
            pipeline=pipeline,
            feature_schema=feature_schema,
            metadata=metadata,
            label_encoder_classes=self.label_encoder.classes_.tolist(),
        )

    def load_artifact(self, artifact_reference: str, validate_runtime_schema: bool = True) -> bool:
        payload = self._artifact_store.load_payload(artifact_reference)
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
        return self._artifact_store.resolve_path(artifact_reference)

    def latest_artifact_path(self) -> str:
        return self._artifact_store.latest_path()

    def predict(self, url: str) -> Dict[str, Any]:
        return inference.predict(self.best_model, self.feature_schema, self.label_encoder, url)

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
