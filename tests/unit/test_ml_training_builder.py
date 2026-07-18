"""T06 (docs/refactoring-plan.md): unit coverage for ml/training.py's
TrainingPipelineBuilder, extracted out of MLPipeline. Covers the "scaling
happens in exactly one place" acceptance criterion directly: both a plain
build and cross_validate must go through the same scaler step."""

from __future__ import annotations

import unittest

import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.training import TrainingPipelineBuilder


class TrainingPipelineBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = TrainingPipelineBuilder(random_state=0, cv_folds=3)
        self.y_train = pd.Series(["benign"] * 10 + ["malicious"] * 10)

    def test_build_pipeline_none_balance_has_no_balancer_step(self):
        pipeline = self.builder.build_pipeline("random_forest", "none", self.y_train)
        self.assertNotIn("balancer", pipeline.named_steps)
        self.assertIsInstance(pipeline.named_steps["scaler"], StandardScaler)

    def test_build_pipeline_smote_adds_balancer_step(self):
        pipeline = self.builder.build_pipeline("random_forest", "smote", self.y_train)
        self.assertIn("balancer", pipeline.named_steps)

    def test_build_pipeline_unknown_balance_method_raises(self):
        with self.assertRaises(ValueError):
            self.builder.build_pipeline("random_forest", "not-a-method", self.y_train)

    def test_scaling_happens_in_exactly_one_place(self):
        # T06 acceptance criterion: the plain-fit pipeline and the cross-validation
        # pipeline are built by the same build_pipeline() call -- one scaler step,
        # not a second independent scaling path for CV.
        fit_pipeline = self.builder.build_pipeline("random_forest", "none", self.y_train)
        cv_pipeline = self.builder.build_pipeline("random_forest", "none", self.y_train)
        self.assertEqual(list(fit_pipeline.named_steps.keys()), list(cv_pipeline.named_steps.keys()))
        self.assertEqual(list(fit_pipeline.named_steps.keys()).count("scaler"), 1)

    def test_cross_validate_returns_locked_keys(self):
        X_train = pd.DataFrame({
            "a": list(range(20)),
            "b": list(range(20, 40)),
        })
        y_train_encoded = pd.Series([0] * 10 + [1] * 10)
        y_train_decoded = pd.Series(["benign"] * 10 + ["malicious"] * 10)
        result = self.builder.cross_validate(
            algorithm="random_forest",
            X_train=X_train,
            y_train=y_train_encoded,
            y_train_decoded=y_train_decoded,
            groups_train=None,
            balance_method="none",
        )
        self.assertEqual(set(result.keys()), {"mean", "std", "n_splits", "strategy"})
        self.assertEqual(result["strategy"], "stratified_k_fold")


if __name__ == "__main__":
    unittest.main()
