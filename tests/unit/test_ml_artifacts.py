"""T06 (docs/refactoring-plan.md): unit coverage for ml/artifacts.py's ArtifactStore,
extracted out of MLPipeline as a pure filesystem I/O collaborator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from core import features_config
from features.schema import build_feature_schema
from ml.artifacts import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = ArtifactStore(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_init_creates_models_dir(self):
        nested = Path(self.tmpdir.name) / "nested" / "models"
        ArtifactStore(str(nested))
        self.assertTrue(nested.exists())

    def test_artifact_path_uses_algorithm_and_run_id(self):
        path = self.store.artifact_path("run-123", "svm")
        self.assertEqual(Path(path).name, "svm_run-123.joblib")

    def test_resolve_path_latest_raises_when_no_artifacts(self):
        with self.assertRaises(FileNotFoundError):
            self.store.latest_path()

    def test_resolve_path_unknown_reference_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.store.resolve_path("does-not-exist")

    def test_save_then_load_payload_round_trip(self):
        feature_schema = build_feature_schema(features_config)
        pipeline = ImbPipeline([("scaler", StandardScaler()), ("estimator", DecisionTreeClassifier())])
        artifact_path = self.store.artifact_path("run-abc", "svm")
        saved_path = self.store.save(
            artifact_path=artifact_path,
            algorithm="svm",
            pipeline=pipeline,
            feature_schema=feature_schema,
            metadata={"algorithm": "svm"},
            label_encoder_classes=["benign", "malicious"],
        )
        self.assertEqual(saved_path, artifact_path)
        self.assertTrue(Path(artifact_path).exists())

        payload = self.store.load_payload("run-abc")
        self.assertEqual(payload["algorithm"], "svm")
        self.assertEqual(payload["label_encoder_classes"], ["benign", "malicious"])
        self.assertEqual(self.store.latest_path(), artifact_path)


if __name__ == "__main__":
    unittest.main()
