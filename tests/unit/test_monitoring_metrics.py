from __future__ import annotations

import unittest

from monitoring.metrics import compute_monitoring_metrics


def _event(prediction, confidence, latency_ms, user_feedback=None, admin_reviewed_label=None):
    return {
        "prediction": prediction,
        "confidence": confidence,
        "prediction_latency_ms": latency_ms,
        "user_feedback": user_feedback,
        "admin_reviewed_label": admin_reviewed_label,
    }


class MonitoringMetricsTests(unittest.TestCase):
    def test_empty_events_returns_zeroed_metrics(self):
        metrics = compute_monitoring_metrics([])
        self.assertEqual(metrics["prediction_count"], 0)
        self.assertIsNone(metrics["malicious_ratio"])
        self.assertIsNone(metrics["mean_confidence"])
        self.assertEqual(metrics["latency_percentiles_ms"], {"p50": None, "p90": None, "p99": None})
        self.assertEqual(metrics["reviewed_count"], 0)

    def test_basic_aggregate_counts_and_ratios(self):
        events = [
            _event("malicious", 0.9, 10.0),
            _event("malicious", 0.8, 20.0),
            _event("benign", 0.99, 15.0),
            _event("benign", 0.95, 25.0),
        ]
        metrics = compute_monitoring_metrics(events)
        self.assertEqual(metrics["prediction_count"], 4)
        self.assertEqual(metrics["malicious_ratio"], 0.5)
        self.assertAlmostEqual(metrics["mean_confidence"], (0.9 + 0.8 + 0.99 + 0.95) / 4)
        self.assertEqual(metrics["latency_percentiles_ms"]["p50"], 17.5)

    def test_user_report_count_only_counts_reported_incorrect(self):
        events = [
            _event("malicious", 0.9, 10.0, user_feedback="reported_incorrect"),
            _event("benign", 0.9, 10.0, user_feedback="confirmed_correct"),
            _event("benign", 0.9, 10.0, user_feedback=None),
        ]
        metrics = compute_monitoring_metrics(events)
        self.assertEqual(metrics["user_report_count"], 1)

    def test_admin_correction_count_only_counts_label_mismatches(self):
        events = [
            _event("malicious", 0.9, 10.0, admin_reviewed_label="malicious"),  # agrees, not a correction
            _event("benign", 0.9, 10.0, admin_reviewed_label="malicious"),  # corrected
            _event("benign", 0.9, 10.0, admin_reviewed_label=None),  # unreviewed
        ]
        metrics = compute_monitoring_metrics(events)
        self.assertEqual(metrics["reviewed_count"], 2)
        self.assertEqual(metrics["admin_correction_count"], 1)

    def test_estimated_fpr_and_fnr_computed_over_reviewed_subset_only(self):
        events = [
            # predicted malicious, admin says benign -> false positive
            _event("malicious", 0.9, 10.0, admin_reviewed_label="benign"),
            # predicted malicious, admin agrees -> not a false positive
            _event("malicious", 0.9, 10.0, admin_reviewed_label="malicious"),
            # predicted benign, admin says malicious -> false negative
            _event("benign", 0.9, 10.0, admin_reviewed_label="malicious"),
            # predicted benign, admin agrees -> not a false negative
            _event("benign", 0.9, 10.0, admin_reviewed_label="benign"),
            # unreviewed, must not affect the rate at all
            _event("malicious", 0.9, 10.0, admin_reviewed_label=None),
        ]
        metrics = compute_monitoring_metrics(events)
        self.assertEqual(metrics["reviewed_count"], 4)
        self.assertEqual(metrics["estimated_false_positive_rate"], 0.5)
        self.assertEqual(metrics["estimated_false_negative_rate"], 0.5)

    def test_fpr_and_fnr_are_none_when_no_reviewed_events_of_that_predicted_class(self):
        events = [_event("malicious", 0.9, 10.0)]
        metrics = compute_monitoring_metrics(events)
        self.assertIsNone(metrics["estimated_false_positive_rate"])
        self.assertIsNone(metrics["estimated_false_negative_rate"])


if __name__ == "__main__":
    unittest.main()
