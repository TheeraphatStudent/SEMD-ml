from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

try:
    import mlflow  # noqa: F401
except Exception:  # pragma: no cover - optional dependency guard
    mlflow = None

from core import config
from ml.training_service import TrainingService
from tracking.mlflow_tracker import MLflowTracker


def write_fixture_dataset(root: Path, name: str = "fixture.csv", benign: int = 12, malicious: int = 8) -> str:
    rows = []
    for idx in range(benign):
        rows.append({"url": f"https://benign{idx}.example.com/home", "label": "benign"})
    for idx in range(malicious):
        rows.append({"url": f"http://secure-login{idx}.bad-example.net/verify?token={idx}", "label": "malicious"})
    pd.DataFrame(rows).to_csv(root / name, index=False)
    return name


@unittest.skipIf(mlflow is None, "mlflow is not installed")
class MLflowTrackingIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        self.dataset_name = write_fixture_dataset(self.dataset_dir)
        self.reports_dir = self.root / "reports"
        self.models_dir = self.root / "models"
        self.extraction_dir = self.root / "extraction"

        self.original_settings = {
            "mlflow_tracking_uri": config.settings.mlflow_tracking_uri,
            "mlflow_experiment_name": config.settings.mlflow_experiment_name,
            "mlflow_registered_model_name": config.settings.mlflow_registered_model_name,
            "mlflow_artifact_root": config.settings.mlflow_artifact_root,
        }

        self.training_module = importlib.import_module("ml.training_service")
        self.original_tracker = self.training_module.mlflow_tracker
        self.original_dataset_path = self.training_module.dataset_pipeline.dataset_path
        self.original_extraction_path = self.training_module.dataset_pipeline.extraction_path
        self.original_models_path = self.training_module.ml_pipeline.models_path

    def tearDown(self) -> None:
        self.training_module.mlflow_tracker = self.original_tracker
        self.training_module.dataset_pipeline.dataset_path = self.original_dataset_path
        self.training_module.dataset_pipeline.extraction_path = self.original_extraction_path
        self.training_module.ml_pipeline.models_path = self.original_models_path

        config.settings.mlflow_tracking_uri = self.original_settings["mlflow_tracking_uri"]
        config.settings.mlflow_experiment_name = self.original_settings["mlflow_experiment_name"]
        config.settings.mlflow_registered_model_name = self.original_settings["mlflow_registered_model_name"]
        config.settings.mlflow_artifact_root = self.original_settings["mlflow_artifact_root"]
        self.temp_dir.cleanup()

    def test_mlflow_training_run_logs_required_metadata_and_artifacts(self) -> None:
        tracker = self._build_tracker(
            tracking_uri=f"sqlite:///{self.root / 'tracking.db'}",
            experiment_name="semd-url-classification-test",
            artifact_root=str(self.root / "artifacts" / "mlflow"),
        )
        training = self._build_training_service(tracker)

        result = training.execute_training(
            {
                "dataset_files": [self.dataset_name],
                "algorithms": ["random_forest"],
                "run_name": "integration-mlflow-run",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertIsNotNone(result["tracking_run_id"])

        experiment = tracker.client.get_experiment_by_name(tracker.experiment_name)
        self.assertIsNotNone(experiment)

        runs = tracker.client.search_runs([experiment.experiment_id])
        self.assertEqual(len(runs), 1)
        run = runs[0]

        self.assertEqual(run.data.params["algorithm"], "random_forest")
        self.assertIn("dataset_version", run.data.params)
        self.assertIn("dataset_hash", run.data.params)
        self.assertIn("feature_schema_version", run.data.params)
        self.assertIn("balancing_method", run.data.params)
        self.assertIn("python_version", run.data.params)

        for metric_name in (
            "train_accuracy",
            "validation_accuracy",
            "test_accuracy",
            "malicious_f1",
            "cross_validation_mean",
            "cross_validation_std",
        ):
            self.assertIn(metric_name, run.data.metrics)

        artifacts = tracker.client.list_artifacts(run.info.run_id, "artifacts")
        artifact_names = {item.path.split("/")[-1] for item in artifacts}
        self.assertTrue(
            {
                "classification_report.json",
                "confusion_matrix.png",
                "roc_curve.png",
                "precision_recall_curve.png",
                "feature_schema.json",
                "training_configuration.json",
                "dataset_quality_report.json",
                "dataset_metadata.json",
                "requirements.txt",
                "sample_predictions.json",
            }.issubset(artifact_names)
        )

        model_items = [item for item in artifacts if item.path.endswith(".joblib")]
        self.assertTrue(model_items)
        downloaded_model = tracker.client.download_artifacts(run.info.run_id, model_items[0].path)
        payload = joblib.load(downloaded_model)
        self.assertEqual(payload["algorithm"], "random_forest")

    def test_mlflow_unavailable_does_not_break_training(self) -> None:
        tracker = self._build_tracker(
            tracking_uri=f"sqlite:///{self.root / 'tracking-unavailable.db'}",
            experiment_name="semd-url-classification-unavailable",
            artifact_root=str(self.root / "artifacts" / "mlflow"),
        )
        tracker.start_run = self._failed_start_run(tracker)
        training = self._build_training_service(tracker)

        result = training.execute_training(
            {
                "dataset_files": [self.dataset_name],
                "algorithms": ["random_forest"],
                "run_name": "integration-mlflow-unavailable",
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertIsNone(result["tracking_run_id"])
        self.assertFalse(result["tracking"]["enabled"])
        self.assertTrue(tracker.last_error)

    def _build_tracker(self, tracking_uri: str, experiment_name: str, artifact_root: str) -> MLflowTracker:
        config.settings.mlflow_tracking_uri = tracking_uri
        config.settings.mlflow_experiment_name = experiment_name
        config.settings.mlflow_registered_model_name = "semd-malicious-url-detector"
        config.settings.mlflow_artifact_root = artifact_root
        return MLflowTracker()

    def _build_training_service(self, tracker: MLflowTracker) -> TrainingService:
        training = TrainingService()
        training.reports_path = str(self.reports_dir)
        os.makedirs(training.reports_path, exist_ok=True)

        self.training_module.mlflow_tracker = tracker
        self.training_module.dataset_pipeline.dataset_path = str(self.dataset_dir)
        self.training_module.dataset_pipeline.extraction_path = str(self.extraction_dir)
        self.training_module.ml_pipeline.models_path = str(self.models_dir)
        return training

    def _failed_start_run(self, tracker: MLflowTracker):
        def _inner(*_args, **_kwargs):
            tracker.last_error = "MLflow unavailable"
            tracker.enabled = False
            return None

        return _inner


if __name__ == "__main__":
    unittest.main()
