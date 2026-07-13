from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from monitoring.retraining import build_feedback_dataset
from monitoring.store import MonitoringStore


class RetrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = MonitoringStore(db_path=os.path.join(self.tmpdir.name, "monitoring.db"))
        self.output_dir = os.path.join(self.tmpdir.name, "datasets")

    def tearDown(self):
        self.tmpdir.cleanup()

    def _record_and_review(self, url, prediction, admin_label):
        prediction_id = self.store.record_event(
            url=url,
            prediction=prediction,
            confidence=0.9,
            model_version="1",
            model_alias="champion",
            feature_schema_version="2.1.0",
            prediction_latency_ms=10.0,
            input_source="cli",
        )
        self.store.set_admin_review(prediction_id, admin_label)
        return prediction_id

    def test_no_reviewed_events_returns_empty_result(self):
        self.store.record_event(
            url="https://unreviewed.example.com",
            prediction="benign",
            confidence=0.9,
            model_version="1",
            model_alias="champion",
            feature_schema_version="2.1.0",
            prediction_latency_ms=10.0,
            input_source="cli",
        )
        result = build_feedback_dataset(store=self.store, output_dir=self.output_dir)
        self.assertIsNone(result["path"])
        self.assertEqual(result["record_count"], 0)

    def test_only_admin_reviewed_events_are_included_using_reviewed_label_as_ground_truth(self):
        self._record_and_review("https://benign.example.com", prediction="benign", admin_label="benign")
        self._record_and_review("https://corrected.example.com", prediction="benign", admin_label="malicious")
        self.store.record_event(
            url="https://unreviewed.example.com",
            prediction="malicious",
            confidence=0.9,
            model_version="1",
            model_alias="champion",
            feature_schema_version="2.1.0",
            prediction_latency_ms=10.0,
            input_source="cli",
        )

        result = build_feedback_dataset(store=self.store, output_dir=self.output_dir)
        self.assertEqual(result["record_count"], 2)
        frame = pd.read_csv(result["path"])
        self.assertEqual(set(frame.columns), {"url", "label"})
        self.assertEqual(len(frame), 2)
        self.assertNotIn("https://unreviewed.example.com", frame["url"].tolist())
        corrected_row = frame[frame["url"] == "https://corrected.example.com"].iloc[0]
        self.assertEqual(corrected_row["label"], "malicious")

    def test_duplicate_urls_keep_latest_review_only(self):
        url = "https://flip-flopped.example.com"
        self._record_and_review(url, prediction="benign", admin_label="benign")
        self._record_and_review(url, prediction="benign", admin_label="malicious")

        result = build_feedback_dataset(store=self.store, output_dir=self.output_dir)
        frame = pd.read_csv(result["path"])
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["label"], "malicious")


if __name__ == "__main__":
    unittest.main()
