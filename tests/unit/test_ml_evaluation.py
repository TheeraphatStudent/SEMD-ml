"""T06 (docs/refactoring-plan.md): unit coverage for ml/evaluation.py, extracted
out of MLPipeline as pure functions (no self/instance state involved)."""

from __future__ import annotations

import unittest

import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.evaluation import evaluate_model, measure_prediction_latency_ms, predict_probabilities, roc_auc


def _fitted_pipeline():
    X = pd.DataFrame({"a": [0, 0, 1, 1, 0, 1, 1, 0], "b": [1, 0, 1, 0, 1, 0, 1, 0]})
    y = pd.Series([0, 0, 1, 1, 0, 1, 1, 0])
    pipeline = ImbPipeline([("scaler", StandardScaler()), ("estimator", DecisionTreeClassifier(random_state=0))])
    pipeline.fit(X, y)
    return pipeline, X, y


class EvaluationTests(unittest.TestCase):
    def test_predict_probabilities_returns_array_for_proba_capable_estimator(self):
        pipeline, X, _ = _fitted_pipeline()
        probabilities = predict_probabilities(pipeline, X)
        self.assertIsNotNone(probabilities)
        self.assertEqual(probabilities.shape[0], len(X))

    def test_roc_auc_none_when_single_class_present(self):
        self.assertIsNone(roc_auc(pd.Series([0, 0, 0]), None))
        self.assertIsNone(roc_auc(pd.Series([1, 1, 1]), None))

    def test_measure_prediction_latency_ms_zero_for_empty_input(self):
        pipeline, X, _ = _fitted_pipeline()
        self.assertEqual(measure_prediction_latency_ms(pipeline, X.iloc[0:0]), 0.0)

    def test_evaluate_model_returns_locked_metric_keys(self):
        pipeline, X, y = _fitted_pipeline()
        metrics = evaluate_model(pipeline, X, y)
        expected_keys = {
            "accuracy", "malicious_precision", "malicious_recall", "malicious_f1",
            "macro_precision", "macro_recall", "macro_f1", "false_positive_rate",
            "false_negative_rate", "confusion_matrix", "prediction_latency_ms", "roc_auc",
        }
        self.assertEqual(set(metrics.keys()), expected_keys)
        self.assertEqual(metrics["accuracy"], 1.0)  # perfectly separable toy data


if __name__ == "__main__":
    unittest.main()
