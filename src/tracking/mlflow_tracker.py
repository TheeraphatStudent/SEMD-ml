from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from core import settings
from tracking.model_registry import ModelRegistryManager

try:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except Exception:  # pragma: no cover - optional dependency
    mlflow = None
    MlflowClient = None
    MlflowException = Exception


class MLflowTracker:
    def __init__(self) -> None:
        self.active_run = None
        self.experiment_id: Optional[str] = None
        self.experiment_name = settings.mlflow_experiment_name
        self.registered_model_name = settings.mlflow_registered_model_name
        self.artifact_root = settings.mlflow_artifact_root
        self.tracking_uri = settings.mlflow_tracking_uri
        self.client = None
        self.last_error: Optional[str] = None
        self.enabled = mlflow is not None and bool(self.tracking_uri)
        if self.enabled:
            self._configure_tracking()

    def _configure_tracking(self) -> None:
        if not self.enabled:
            return
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            self.client = MlflowClient(tracking_uri=self.tracking_uri) if MlflowClient is not None else None
            self.last_error = None
        except Exception as exc:
            self.last_error = f"Unable to configure MLflow tracking URI '{self.tracking_uri}': {exc}"
            self.enabled = False
            self.client = None

    def _ensure_experiment(self) -> Optional[str]:
        if not self.enabled or self.client is None:
            return None
        if self.experiment_id is not None:
            return self.experiment_id
        try:
            experiment = self.client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                # No explicit artifact_location: let the tracking server assign one from its own
                # --default-artifact-root/--artifacts-destination. Passing a client-computed local
                # path here defeats --serve-artifacts proxying (see docker/docker-compose.yml).
                self.experiment_id = self.client.create_experiment(name=self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
            return self.experiment_id
        except Exception as exc:
            self.last_error = f"Unable to select or create experiment '{self.experiment_name}': {exc}"
            self.enabled = False
            return None

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        experiment_id = self._ensure_experiment()
        if experiment_id is None:
            return None
        run_name = run_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            self.active_run = mlflow.start_run(run_name=run_name, experiment_id=experiment_id)
            if tags:
                self.log_tags(dict(tags))
            return self.active_run.info.run_id
        except Exception as exc:
            self.last_error = f"Unable to start MLflow run '{run_name}': {exc}"
            self.active_run = None
            self.enabled = False
            return None

    def log_params(self, params: Dict[str, Any]) -> None:
        if not self.enabled or self.active_run is None:
            return
        for key, value in params.items():
            try:
                mlflow.log_param(key, self._serialize_field(value))
            except Exception:
                continue

    def log_tags(self, tags: Dict[str, Any]) -> None:
        if not self.enabled or self.active_run is None:
            return
        for key, value in tags.items():
            try:
                mlflow.set_tag(key, self._serialize_field(value))
            except Exception:
                continue

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        if not self.enabled or self.active_run is None:
            return
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            try:
                mlflow.log_metric(key, float(value))
            except Exception:
                continue

    def log_artifact(self, artifact_path: str, artifact_path_in_run: Optional[str] = None) -> None:
        if not self.enabled or self.active_run is None:
            return
        try:
            if artifact_path_in_run:
                mlflow.log_artifact(artifact_path, artifact_path_in_run)
            else:
                mlflow.log_artifact(artifact_path)
        except Exception:
            pass

    def log_artifacts(self, artifact_dir: str, artifact_path_in_run: Optional[str] = None) -> None:
        if not self.enabled or self.active_run is None:
            return
        try:
            if artifact_path_in_run:
                mlflow.log_artifacts(artifact_dir, artifact_path_in_run)
            else:
                mlflow.log_artifacts(artifact_dir)
        except Exception:
            pass

    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        if not self.enabled or self.active_run is None:
            return
        self.log_params(
            {
                "train_size": len(dataset_info.get("X_train_unbalanced", [])),
                "validation_size": len(dataset_info.get("X_val", [])),
                "test_size": len(dataset_info.get("X_test", [])),
                "num_features": len(dataset_info.get("feature_names", [])),
                "balance_method": dataset_info.get("balance_method", "none"),
            }
        )

    def log_training_results(self, results: Dict[str, Any], best_algorithm: Optional[str] = None) -> None:
        if not self.enabled or self.active_run is None:
            return
        for algorithm, result in results.items():
            metrics = result.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.log_metrics({f"{algorithm}_{key}": value})
            for split_name in ("train_metrics", "validation_metrics"):
                split_metrics = result.get(split_name, {})
                for key, value in split_metrics.items():
                    if isinstance(value, (int, float)):
                        self.log_metrics({f"{algorithm}_{split_name}_{key}": value})
            self.log_metrics(
                {
                    f"{algorithm}_cross_validation_mean": result.get("cross_validation_mean"),
                    f"{algorithm}_cross_validation_std": result.get("cross_validation_std"),
                    f"{algorithm}_training_duration_seconds": result.get("training_duration_seconds"),
                    f"{algorithm}_prediction_latency_ms": result.get("prediction_latency_ms"),
                }
            )

        if best_algorithm and best_algorithm in results:
            best_result = results[best_algorithm]
            self.log_metrics(self._flatten_best_metrics(best_result))

    def register_model(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        if not self.enabled or self.client is None:
            return None
        run_id = kwargs.get("run_id") or (args[0] if args else None)
        if not run_id:
            raise ValueError("run_id is required to register a model")
        return ModelRegistryManager(client=self.client).register_candidate(run_id)

    def evaluate_model(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.enabled or self.active_run is None:
            return
        try:
            mlflow.end_run(status=status)
        except Exception:
            pass
        self.active_run = None

    def log_error(
        self,
        error_message: str,
        error_type: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled or self.active_run is None:
            return
        self.log_params(
            {
                "error_message": error_message,
                "error_type": error_type,
                "error_context": additional_info or {},
            }
        )

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tracking_uri": self.tracking_uri,
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "registered_model_name": self.registered_model_name,
            "artifact_root": self.artifact_root,
            "aliases": {
                "candidate": settings.mlflow_alias_candidate,
                "champion": settings.mlflow_alias_champion,
                "previous_champion": settings.mlflow_alias_previous_champion,
            },
            "last_error": self.last_error,
            "autologging": "disabled",
        }

    def _flatten_best_metrics(self, result: Mapping[str, Any]) -> Dict[str, float]:
        train_metrics = result.get("train_metrics", {})
        validation_metrics = result.get("validation_metrics", {})
        test_metrics = result.get("metrics", {})
        metrics = {
            "train_accuracy": train_metrics.get("accuracy"),
            "validation_accuracy": validation_metrics.get("accuracy"),
            "test_accuracy": test_metrics.get("accuracy"),
            "malicious_precision": test_metrics.get("malicious_precision"),
            "malicious_recall": test_metrics.get("malicious_recall"),
            "malicious_f1": test_metrics.get("malicious_f1"),
            "macro_precision": test_metrics.get("macro_precision"),
            "macro_recall": test_metrics.get("macro_recall"),
            "macro_f1": test_metrics.get("macro_f1"),
            "roc_auc": test_metrics.get("roc_auc"),
            "false_positive_rate": test_metrics.get("false_positive_rate"),
            "false_negative_rate": test_metrics.get("false_negative_rate"),
            "training_duration_seconds": result.get("training_duration_seconds"),
            "prediction_latency_ms": result.get("prediction_latency_ms"),
            "cross_validation_mean": result.get("cross_validation_mean"),
            "cross_validation_std": result.get("cross_validation_std"),
        }
        return {key: value for key, value in metrics.items() if isinstance(value, (int, float))}

    def _serialize_field(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        return json.dumps(value, sort_keys=True, default=str)


mlflow_tracker = MLflowTracker()
