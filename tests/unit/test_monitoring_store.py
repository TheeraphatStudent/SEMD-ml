from __future__ import annotations

import os
import tempfile
import unittest

from monitoring.store import MonitoringStore, hash_url


class MonitoringStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = MonitoringStore(db_path=os.path.join(self.tmpdir.name, "monitoring.db"))

    def tearDown(self):
        self.tmpdir.cleanup()

    def _record(self, url="https://example.com/login", prediction="malicious", **overrides):
        payload = dict(
            url=url,
            prediction=prediction,
            confidence=0.9,
            model_version="1",
            model_alias="champion",
            feature_schema_version="2.1.0",
            prediction_latency_ms=12.5,
            input_source="cli",
        )
        payload.update(overrides)
        return self.store.record_event(**payload)

    def test_record_event_persists_all_required_fields(self):
        prediction_id = self._record()
        event = self.store.get_event(prediction_id)
        self.assertIsNotNone(event)
        for field in (
            "prediction_id",
            "url",
            "url_hash",
            "prediction",
            "confidence",
            "model_version",
            "feature_schema_version",
            "prediction_latency_ms",
            "input_source",
            "created_at",
        ):
            self.assertIn(field, event)
        self.assertIsNone(event["user_feedback"])
        self.assertIsNone(event["admin_reviewed_label"])
        self.assertEqual(event["url_hash"], hash_url(event["url"]))

    def test_url_hash_is_stable_and_case_insensitive(self):
        first = self._record(url="HTTPS://Example.com/Path")
        second = self._record(url="https://example.com/path")
        event1 = self.store.get_event(first)
        event2 = self.store.get_event(second)
        self.assertEqual(event1["url_hash"], event2["url_hash"])

    def test_set_user_feedback_updates_row_and_rejects_unknown_values(self):
        prediction_id = self._record()
        self.assertTrue(self.store.set_user_feedback(prediction_id, "reported_incorrect"))
        self.assertEqual(self.store.get_event(prediction_id)["user_feedback"], "reported_incorrect")
        with self.assertRaises(ValueError):
            self.store.set_user_feedback(prediction_id, "not-a-real-status")

    def test_set_user_feedback_on_unknown_prediction_id_returns_false(self):
        self.assertFalse(self.store.set_user_feedback("does-not-exist", "reported_incorrect"))

    def test_set_admin_review_updates_label_and_timestamp(self):
        prediction_id = self._record(prediction="benign")
        self.assertTrue(self.store.set_admin_review(prediction_id, "malicious"))
        event = self.store.get_event(prediction_id)
        self.assertEqual(event["admin_reviewed_label"], "malicious")
        self.assertIsNotNone(event["admin_reviewed_at"])
        with self.assertRaises(ValueError):
            self.store.set_admin_review(prediction_id, "phishing")

    def test_get_events_reviewed_only_filters_unreviewed_rows(self):
        reviewed = self._record()
        unreviewed = self._record(url="https://example.com/other")
        self.store.set_admin_review(reviewed, "malicious")

        all_events = self.store.get_events()
        reviewed_events = self.store.get_events(reviewed_only=True)

        self.assertEqual({event["prediction_id"] for event in all_events}, {reviewed, unreviewed})
        self.assertEqual([event["prediction_id"] for event in reviewed_events], [reviewed])

    def test_get_events_since_filters_by_created_at(self):
        prediction_id = self._record()
        future = "2999-01-01T00:00:00+00:00"
        self.assertEqual(self.store.get_events(since=future), [])
        recent_events = self.store.get_events(since="2000-01-01")
        self.assertEqual([event["prediction_id"] for event in recent_events], [prediction_id])


if __name__ == "__main__":
    unittest.main()
