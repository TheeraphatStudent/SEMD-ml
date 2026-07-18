"""T07 (docs/refactoring-plan.md): unit coverage for pipelines/prediction_pipeline.py,
extracted out of PredictionService. Covers the model-resolution decision
(reuse loaded model vs. reload vs. fall back to champion) independently of
monitoring-event recording and batch orchestration, which stay in
ml/prediction_service.py.
"""

from __future__ import annotations

import unittest

from pipelines.prediction_pipeline import PredictionPipeline


class FakeModelLoader:
    def __init__(self):
        self.load_calls = []
        self.predict_calls = []

    def load(self, selector=None):
        self.load_calls.append(selector)

    def predict(self, url, selector=None):
        self.predict_calls.append((url, selector))
        return {"url": url, "prediction": "benign", "model_alias": selector}


class PredictionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.loader = FakeModelLoader()
        self.pipeline = PredictionPipeline(model_loader=self.loader)

    def test_no_url_raises(self):
        with self.assertRaises(ValueError):
            self.pipeline.predict("")

    def test_first_call_with_no_model_id_loads_champion(self):
        self.pipeline.predict("https://example.com")
        self.assertEqual(self.loader.load_calls, ["champion"])
        self.assertEqual(self.pipeline.current_model_id, "champion")

    def test_second_call_with_no_model_id_reuses_loaded_model(self):
        self.pipeline.predict("https://a.example.com")
        self.pipeline.predict("https://b.example.com")
        self.assertEqual(self.loader.load_calls, ["champion"])  # only loaded once

    def test_explicit_model_id_different_from_current_triggers_reload(self):
        self.pipeline.predict("https://a.example.com")  # loads champion
        self.pipeline.predict("https://b.example.com", model_id="run-42")
        self.assertEqual(self.loader.load_calls, ["champion", "run-42"])
        self.assertEqual(self.pipeline.current_model_id, "run-42")

    def test_explicit_model_id_matching_current_does_not_reload(self):
        self.pipeline.predict("https://a.example.com", model_id="run-42")
        self.pipeline.predict("https://b.example.com", model_id="run-42")
        self.assertEqual(self.loader.load_calls, ["run-42"])  # not reloaded

    def test_predict_passes_resolved_selector_to_loader(self):
        self.pipeline.predict("https://a.example.com", model_id="run-42")
        self.assertEqual(self.loader.predict_calls, [("https://a.example.com", "run-42")])


if __name__ == "__main__":
    unittest.main()
