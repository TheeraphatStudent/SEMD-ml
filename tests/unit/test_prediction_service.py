from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from ml.prediction_service import PredictionService
from monitoring.store import MonitoringStore


class FakeModelLoader:
    def __init__(self, response):
        self.response = response
        self.load_calls = []
        self.predict_calls = []

    def load(self, selector=None):
        self.load_calls.append(selector)
        return {}

    def predict(self, url, selector=None):
        self.predict_calls.append((url, selector))
        return dict(self.response, url=url)


class PredictionServiceMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = MonitoringStore(db_path=os.path.join(self.tmpdir.name, "monitoring.db"))
        self.response = {
            "prediction": "malicious",
            "is_malicious": True,
            "confidence": 0.87,
            "feature_schema_version": "2.1.0",
            "prediction_time_ms": 5.0,
            "model_name": "semd-malicious-url-detector",
            "model_version": "3",
            "model_alias": "champion",
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def _service(self):
        service = PredictionService()
        service.model_loader = FakeModelLoader(self.response)
        service.current_model_id = "champion"
        return service

    def test_execute_prediction_returns_prediction_id_and_records_event(self):
        service = self._service()
        with patch("ml.prediction_service.monitoring_store", self.store):
            result = service.execute_prediction({"url": "http://bad.example.com"}, input_source="cli")

        self.assertIn("prediction_id", result)
        self.assertIsNotNone(result["prediction_id"])
        event = self.store.get_event(result["prediction_id"])
        self.assertIsNotNone(event)
        self.assertEqual(event["url"], "http://bad.example.com")
        self.assertEqual(event["prediction"], "malicious")
        self.assertEqual(event["model_version"], "3")
        self.assertEqual(event["input_source"], "cli")

    def test_batch_predict_records_one_event_per_url_with_shared_input_source(self):
        service = self._service()
        urls = ["https://a.example.com", "https://b.example.com"]
        with patch("ml.prediction_service.monitoring_store", self.store):
            batch = service.batch_predict({"urls": urls}, input_source="queue")

        prediction_ids = [item["prediction_id"] for item in batch["predictions"]]
        self.assertEqual(len(prediction_ids), 2)
        self.assertTrue(all(prediction_ids))
        events = self.store.get_events()
        self.assertEqual(len(events), 2)
        self.assertTrue(all(event["input_source"] == "queue" for event in events))

    def test_monitoring_store_failure_does_not_break_prediction_response(self):
        service = self._service()

        class BoomStore:
            def record_event(self, **kwargs):
                raise RuntimeError("store unavailable")

        with patch("ml.prediction_service.monitoring_store", BoomStore()):
            result = service.execute_prediction({"url": "http://bad.example.com"}, input_source="cli")

        self.assertEqual(result["prediction"], "malicious")
        self.assertIsNone(result["prediction_id"])


if __name__ == "__main__":
    unittest.main()
