import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from data.dataset_pipeline import DatasetPipeline
from ml.ml_pipeline import MLPipeline
from ml.model_factory import model_factory


def write_fixture_dataset(root: Path, name: str = "fixture.csv", benign: int = 12, malicious: int = 8) -> str:
    rows = []
    for idx in range(benign):
        rows.append({"url": f"https://benign{idx}.example.com/home", "label": "benign"})
    for idx in range(malicious):
        rows.append({"url": f"http://secure-login{idx}.bad-example.net/verify?token={idx}", "label": "malicious"})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / name, index=False)
    return name


class ModelFactoryTests(unittest.TestCase):
    def test_supported_identifiers_are_consistent(self):
        identifiers = model_factory.identifiers()
        self.assertIn("svm", identifiers)
        self.assertIn("random_forest", identifiers)
        self.assertIn("gradient_boosting", identifiers)
        self.assertNotIn("decision_tree", identifiers)


class TrainingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dataset_name = write_fixture_dataset(self.root)
        self.dataset_pipeline = DatasetPipeline()
        self.dataset_pipeline.dataset_path = str(self.root)
        self.dataset_pipeline.extraction_path = str(self.root / "extraction")
        self.ml_pipeline = MLPipeline()
        self.ml_pipeline.models_path = str(self.root / "models")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_training_with_fixture_data_runs_two_algorithms(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest", "gradient_boosting"], run_id="fixture-run")
        self.assertEqual(set(result["results"].keys()), {"random_forest", "gradient_boosting"})

    def test_balancing_only_changes_train_split(self):
        dataset = self.dataset_pipeline.prepare_dataset(
            [self.dataset_name],
            apply_balancing=True,
            manual_balance_method="oversampling",
        )
        self.assertGreater(len(dataset["y_train"]), len(dataset["y_train_unbalanced"]))
        self.assertLessEqual(len(dataset["y_val"].unique()), 2)
        self.assertEqual(len(dataset["X_test"]), len(dataset["y_test"]))

    def test_scaling_is_fit_on_training_data_only(self):
        dataset = self.dataset_pipeline.prepare_dataset(
            [self.dataset_name],
            apply_balancing=False,
        )
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="scale-run")
        artifact_path = result["best_artifact_path"]
        payload = joblib.load(artifact_path)
        scaler = payload["pipeline"].named_steps["scaler"]
        expected_means = dataset["X_train_unbalanced"].mean().to_numpy()
        self.assertEqual(len(scaler.mean_), len(expected_means))
        self.assertTrue(((scaler.mean_ - expected_means).round(10) == 0).all())

    def test_cross_validation_metrics_are_reported(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="cv-run")
        metrics = result["results"]["random_forest"]
        self.assertIn("cross_validation_mean", metrics)
        self.assertIn("cross_validation_std", metrics)
        self.assertIn("fold", metrics["cross_validation_strategy"])

    def test_metrics_are_complete(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="metrics-run")
        metrics = result["results"]["random_forest"]["metrics"]
        for key in [
            "accuracy",
            "malicious_precision",
            "malicious_recall",
            "malicious_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "false_positive_rate",
            "false_negative_rate",
            "confusion_matrix",
            "prediction_latency_ms",
        ]:
            self.assertIn(key, metrics)

    def test_save_load_round_trip_is_consistent(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="roundtrip-run")
        url = "https://roundtrip.example.com"
        before = self.ml_pipeline.predict(url)
        reloaded = MLPipeline()
        reloaded.load_artifact(result["best_artifact_path"])
        after = reloaded.predict(url)
        self.assertEqual(before["prediction"], after["prediction"])
        self.assertEqual(before["feature_schema_version"], after["feature_schema_version"])

    def test_prediction_output_shape(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="predict-run")
        loaded = MLPipeline()
        loaded.load_artifact(result["best_artifact_path"])
        prediction = loaded.predict("https://example.com")
        self.assertEqual(
            set(prediction.keys()),
            {"url", "prediction", "is_malicious", "confidence", "feature_schema_version", "prediction_time_ms"},
        )

    def test_feature_schema_incompatibility_raises(self):
        dataset = self.dataset_pipeline.prepare_dataset([self.dataset_name], apply_balancing=True)
        result = self.ml_pipeline.train_models(dataset, ["random_forest"], run_id="schema-run")
        incompatible_path = self.root / "models" / "schema-incompatible.joblib"
        payload = joblib.load(result["best_artifact_path"])
        payload["feature_schema"]["schema_version"] = "999.0.0"
        joblib.dump(payload, incompatible_path)
        with self.assertRaises(ValueError):
            self.ml_pipeline.load_artifact(str(incompatible_path))


if __name__ == "__main__":
    unittest.main()
