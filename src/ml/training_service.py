from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, classification_report

from core import settings
from data import dataset_pipeline
from ml.ml_pipeline import ml_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tracking import mlflow_tracker
except Exception:  # pragma: no cover - optional during tests
    mlflow_tracker = None


class TrainingService:
    def __init__(self) -> None:
        self.reports_path = settings.reports_path
        os.makedirs(self.reports_path, exist_ok=True)

    def execute_training(self, job_data: Dict[str, Any], run_kind: str = "training") -> Dict[str, Any]:
        os.makedirs(self.reports_path, exist_ok=True)
        dataset_files = job_data.get("dataset_files") or settings.default_dataset_files
        algorithms = job_data.get("algorithms") or settings.default_train_algorithms
        balance_method = job_data.get("balance_method")
        run_name = job_data.get("run_name") or f"{run_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = ml_pipeline.generate_run_id()
        git_sha = self._git_commit_sha()
        started_at = time.perf_counter()

        dataset_result = dataset_pipeline.prepare_dataset(
            dataset_files=dataset_files,
            apply_balancing=True,
            manual_balance_method=balance_method,
        )

        tracking_run_id = None
        tracking_status = mlflow_tracker.status() if mlflow_tracker is not None else {"enabled": False}
        if mlflow_tracker is not None:
            try:
                tracking_run_id = mlflow_tracker.start_run(
                    run_name=run_name,
                    tags=self._build_run_tags(
                        run_name=run_name,
                        run_kind=run_kind,
                        algorithms=algorithms,
                        dataset_result=dataset_result,
                        git_sha=git_sha,
                    ),
                )
                mlflow_tracker.log_params(
                    self._build_training_params(
                        dataset_files=dataset_files,
                        algorithms=algorithms,
                        dataset_result=dataset_result,
                        git_sha=git_sha,
                    )
                )
            except Exception:
                pass

        training_summary = ml_pipeline.train_models(
            dataset_result=dataset_result,
            algorithms=algorithms,
            run_id=run_id,
            git_commit_sha=git_sha,
        )
        total_duration_seconds = time.perf_counter() - started_at
        best_algorithm = training_summary["best_algorithm"]
        best_result = training_summary["results"][best_algorithm]

        result = {
            "status": "success",
            "run_id": run_id,
            "best_algorithm": best_algorithm,
            "best_artifact_path": training_summary["best_artifact_path"],
            "metrics": training_summary["best_metrics"],
            "validation_metrics": training_summary["best_validation_metrics"],
            "train_metrics": best_result["train_metrics"],
            "tracking_run_id": tracking_run_id,
            "tracking": mlflow_tracker.status() if mlflow_tracker is not None else {"enabled": False},
            "training_duration_seconds": total_duration_seconds,
            "dataset": {
                "dataset_files": dataset_files,
                "dataset_metadata": dataset_result["dataset_metadata"],
                "validation_report": dataset_result["validation_report"],
                "split_strategy": dataset_result["split_strategy"],
                "sample_size": int(dataset_result["dataset_metadata"]["total_records"]),
                "train_samples": len(dataset_result["X_train_unbalanced"]),
                "validation_samples": len(dataset_result["X_val"]),
                "test_samples": len(dataset_result["X_test"]),
                "feature_count": len(dataset_result["feature_names"]),
                "balance_method": dataset_result["balance_method"],
                "train_imbalance_before": dataset_result["train_imbalance_before"],
                "train_imbalance_after": dataset_result["train_imbalance_after"],
            },
            "results": training_summary["results"],
        }

        artifact_bundle = self._create_tracking_artifacts(
            run_id=run_id,
            run_kind=run_kind,
            dataset_result=dataset_result,
            training_summary=training_summary,
            dataset_files=dataset_files,
            git_sha=git_sha,
        )
        result["artifacts"] = artifact_bundle["paths"]

        if mlflow_tracker is not None:
            try:
                mlflow_tracker.log_training_results(training_summary["results"], best_algorithm=best_algorithm)
                mlflow_tracker.log_artifacts(artifact_bundle["directory"], "artifacts")
                mlflow_tracker.end_run(status="FINISHED")
            except Exception:
                pass
            tracking_status = mlflow_tracker.status()

        report_path = self._write_training_report(run_id, result)
        result["report_path"] = report_path
        result["tracking"] = tracking_status
        return result

    def execute_evaluation(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        evaluation = self.execute_training(job_data, run_kind="evaluation")
        return {
            "status": evaluation["status"],
            "run_id": evaluation["run_id"],
            "tracking_run_id": evaluation.get("tracking_run_id"),
            "best_algorithm": evaluation["best_algorithm"],
            "dataset": evaluation["dataset"],
            "results": evaluation["results"],
            "artifacts": evaluation.get("artifacts", {}),
            "tracking": evaluation.get("tracking", {}),
        }

    def execute_training_obo(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "failed",
            "error": "train-obo is not supported by the refactored training pipeline yet.",
        }

    def _write_training_report(self, run_id: str, payload: Dict[str, Any]) -> str:
        report_path = os.path.join(self.reports_path, f"training_report_{run_id}.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return report_path

    def _git_commit_sha(self) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
            )
            return completed.stdout.strip()
        except Exception:
            return None

    def _build_training_params(
        self,
        dataset_files: List[str],
        algorithms: List[str],
        dataset_result: Dict[str, Any],
        git_sha: Optional[str],
    ) -> Dict[str, Any]:
        metadata = dataset_result["dataset_metadata"]
        validation_report = dataset_result["validation_report"]
        class_distribution = dataset_result["train_imbalance_before"]["class_distribution"]
        return {
            "algorithm": algorithms[0] if len(algorithms) == 1 else "multi",
            "requested_algorithms": algorithms,
            "hyperparameters": {algorithm: settings.algorithm_hyperparameters[algorithm] for algorithm in algorithms},
            "random_state": settings.random_state,
            "dataset_version": metadata["dataset_version"],
            "dataset_hash": metadata["dataset_hash"],
            "sample_size": metadata["total_records"],
            "train_size": len(dataset_result["X_train_unbalanced"]),
            "validation_size": len(dataset_result["X_val"]),
            "test_size": len(dataset_result["X_test"]),
            "balancing_method": dataset_result["balance_method"],
            "scaling_method": "StandardScaler",
            "feature_schema_version": dataset_result["feature_schema"]["schema_version"],
            "feature_count": len(dataset_result["feature_names"]),
            "class_distribution": class_distribution,
            "git_commit_sha": git_sha or "unknown",
            "python_version": os.sys.version.split()[0],
            "registered_model_name": settings.mlflow_registered_model_name,
            "dataset_files": dataset_files,
            "dataset_sources": metadata["source_references"],
            "autologging": "disabled",
            "dataset_quality_summary": validation_report["stats"],
        }

    def _build_run_tags(
        self,
        run_name: str,
        run_kind: str,
        algorithms: List[str],
        dataset_result: Dict[str, Any],
        git_sha: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "run_name": run_name,
            "run_kind": run_kind,
            "project": "semd-ml",
            "dataset_version": dataset_result["dataset_metadata"]["dataset_version"],
            "dataset_hash": dataset_result["dataset_metadata"]["dataset_hash"],
            "requested_algorithms": ",".join(algorithms),
            "git_commit_sha": git_sha or "unknown",
            "autologging": "disabled",
            "registered_model_name": settings.mlflow_registered_model_name,
        }

    def _create_tracking_artifacts(
        self,
        run_id: str,
        run_kind: str,
        dataset_result: Dict[str, Any],
        training_summary: Dict[str, Any],
        dataset_files: List[str],
        git_sha: Optional[str],
    ) -> Dict[str, Any]:
        best_algorithm = training_summary["best_algorithm"]
        best_result = training_summary["results"][best_algorithm]
        pipeline = ml_pipeline.best_model
        if pipeline is None:
            raise ValueError("Best model pipeline is not available for artifact logging")

        y_test_encoded = pd.Series(
            ml_pipeline.label_encoder.transform(dataset_result["y_test"]),
            index=dataset_result["y_test"].index,
        )
        test_predictions = pipeline.predict(dataset_result["X_test"])
        probabilities = ml_pipeline._predict_probabilities(pipeline, dataset_result["X_test"])

        temp_dir = Path(tempfile.mkdtemp(prefix=f"mlflow_{run_kind}_{run_id}_", dir=self.reports_path))
        paths: Dict[str, str] = {}

        classification_report_path = temp_dir / "classification_report.json"
        classification_payload = classification_report(
            y_test_encoded,
            test_predictions,
            target_names=ml_pipeline.label_encoder.classes_.tolist(),
            output_dict=True,
            zero_division=0,
        )
        classification_report_path.write_text(json.dumps(classification_payload, indent=2), encoding="utf-8")
        paths["classification_report"] = str(classification_report_path)

        feature_schema_path = temp_dir / "feature_schema.json"
        feature_schema_path.write_text(json.dumps(dataset_result["feature_schema"], indent=2), encoding="utf-8")
        paths["feature_schema"] = str(feature_schema_path)

        training_config_path = temp_dir / "training_configuration.json"
        training_config_path.write_text(
            json.dumps(
                {
                    "run_kind": run_kind,
                    "run_id": run_id,
                    "dataset_files": dataset_files,
                    "algorithms": list(training_summary["results"].keys()),
                    "selected_algorithm": best_algorithm,
                    "random_state": settings.random_state,
                    "balancing_method": dataset_result["balance_method"],
                    "scaling_method": "StandardScaler",
                    "hyperparameters": best_result["hyperparameters"],
                    "git_commit_sha": git_sha,
                    "mlflow": {
                        "tracking_uri": settings.mlflow_tracking_uri,
                        "experiment_name": settings.mlflow_experiment_name,
                        "registered_model_name": settings.mlflow_registered_model_name,
                        "artifact_root": settings.mlflow_artifact_root,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["training_configuration"] = str(training_config_path)

        dataset_quality_path = temp_dir / "dataset_quality_report.json"
        dataset_quality_path.write_text(json.dumps(dataset_result["validation_report"], indent=2), encoding="utf-8")
        paths["dataset_quality_report"] = str(dataset_quality_path)

        dataset_metadata_path = temp_dir / "dataset_metadata.json"
        dataset_metadata_path.write_text(json.dumps(dataset_result["dataset_metadata"], indent=2), encoding="utf-8")
        paths["dataset_metadata"] = str(dataset_metadata_path)

        dependency_path = temp_dir / "requirements.txt"
        dependency_path.write_text(
            Path(Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        paths["dependency_file"] = str(dependency_path)

        sample_predictions = []
        decoded_predictions = ml_pipeline.label_encoder.inverse_transform(test_predictions)
        confidences: List[Optional[float]] = [None] * len(decoded_predictions)
        if probabilities is not None:
            confidences = [float(value) for value in probabilities.max(axis=1)]
        for idx, row_index in enumerate(dataset_result["X_test"].index[:10]):
            sample_predictions.append(
                {
                    "row_index": int(row_index),
                    "url": dataset_result["urls_test"].iloc[idx] if idx < len(dataset_result["urls_test"]) else None,
                    "actual_label": dataset_result["y_test"].iloc[idx],
                    "predicted_label": decoded_predictions[idx],
                    "confidence": confidences[idx],
                }
            )
        sample_predictions_path = temp_dir / "sample_predictions.json"
        sample_predictions_path.write_text(json.dumps(sample_predictions, indent=2), encoding="utf-8")
        paths["sample_predictions"] = str(sample_predictions_path)

        model_artifact_path = temp_dir / Path(training_summary["best_artifact_path"]).name
        model_artifact_path.write_bytes(Path(training_summary["best_artifact_path"]).read_bytes())
        paths["model_artifact"] = str(model_artifact_path)

        self._write_confusion_matrix_plot(best_result["metrics"]["confusion_matrix"], temp_dir / "confusion_matrix.png")
        paths["confusion_matrix_image"] = str(temp_dir / "confusion_matrix.png")

        self._write_roc_curve_plot(y_test_encoded, probabilities, temp_dir / "roc_curve.png")
        paths["roc_curve"] = str(temp_dir / "roc_curve.png")

        self._write_precision_recall_curve_plot(y_test_encoded, probabilities, temp_dir / "precision_recall_curve.png")
        paths["precision_recall_curve"] = str(temp_dir / "precision_recall_curve.png")

        return {"directory": str(temp_dir), "paths": paths}

    def _write_confusion_matrix_plot(self, confusion_matrix_values: List[List[int]], destination: Path) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(confusion_matrix_values, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["benign", "malicious"])
        ax.set_yticks([0, 1], labels=["benign", "malicious"])
        for row_idx, row in enumerate(confusion_matrix_values):
            for col_idx, value in enumerate(row):
                ax.text(col_idx, row_idx, value, ha="center", va="center", color="black")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(destination)
        plt.close(fig)

    def _write_roc_curve_plot(
        self,
        y_true: pd.Series,
        probabilities: Optional[Any],
        destination: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        if probabilities is not None:
            RocCurveDisplay.from_predictions(y_true, probabilities[:, 1], ax=ax)
        else:
            ax.text(0.5, 0.5, "ROC-AUC unavailable", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(destination)
        plt.close(fig)

    def _write_precision_recall_curve_plot(
        self,
        y_true: pd.Series,
        probabilities: Optional[Any],
        destination: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        if probabilities is not None:
            PrecisionRecallDisplay.from_predictions(y_true, probabilities[:, 1], ax=ax)
        else:
            ax.text(0.5, 0.5, "Precision-recall unavailable", ha="center", va="center")
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(destination)
        plt.close(fig)


training_service = TrainingService()
