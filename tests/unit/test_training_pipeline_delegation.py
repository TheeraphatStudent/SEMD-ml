"""T07 (docs/refactoring-plan.md): unit coverage for pipelines/training_pipeline.py.

Verifies TrainingPipeline.prepare_dataset/train delegate to dataset_pipeline and
ml_pipeline with the right arguments -- mocked, so this stays a fast unit test.
The real dataset-prep -> train -> evaluate sequence is already covered end-to-end
by tests/unit/test_training_pipeline.py (MLPipeline directly) and this session's
live train/register/predict verification against real infra.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipelines.training_pipeline import TrainingPipeline


class TrainingPipelineDelegationTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = TrainingPipeline()

    def test_prepare_dataset_delegates_with_balancing_enabled(self):
        with patch("pipelines.training_pipeline.dataset_pipeline") as mock_dataset_pipeline:
            mock_dataset_pipeline.prepare_dataset.return_value = {"ok": True}
            result = self.pipeline.prepare_dataset(["a.csv"], balance_method="smote")

        mock_dataset_pipeline.prepare_dataset.assert_called_once_with(
            dataset_files=["a.csv"], apply_balancing=True, manual_balance_method="smote",
        )
        self.assertEqual(result, {"ok": True})

    def test_train_delegates_to_ml_pipeline_train_models(self):
        dataset_result = {"feature_schema": {}}
        with patch("ml.ml_pipeline.ml_pipeline") as mock_ml_pipeline:
            mock_ml_pipeline.train_models.return_value = {"best_algorithm": "svm"}
            result = self.pipeline.train(dataset_result, ["svm"], run_id="run-1", git_commit_sha="abc123")

        mock_ml_pipeline.train_models.assert_called_once_with(
            dataset_result=dataset_result, algorithms=["svm"], run_id="run-1", git_commit_sha="abc123",
        )
        self.assertEqual(result, {"best_algorithm": "svm"})


if __name__ == "__main__":
    unittest.main()
