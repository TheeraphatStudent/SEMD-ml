from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    lower_weight = sorted_values[lower] * (upper - rank)
    upper_weight = sorted_values[upper] * (rank - lower)
    return float(lower_weight + upper_weight)


def _empty_metrics() -> Dict[str, Any]:
    return {
        "prediction_count": 0,
        "malicious_ratio": None,
        "mean_confidence": None,
        "latency_percentiles_ms": {"p50": None, "p90": None, "p99": None},
        "user_report_count": 0,
        "admin_correction_count": 0,
        "reviewed_count": 0,
        "estimated_false_positive_rate": None,
        "estimated_false_negative_rate": None,
    }


def compute_monitoring_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate prediction-event telemetry.

    `estimated_false_positive_rate` / `estimated_false_negative_rate` are computed only over the
    admin-reviewed subset (`reviewed_count`), not the full `prediction_count` — reviewed predictions
    are not a random sample (admins tend to review reported/suspicious ones), so these are directional
    signals, not population error rates.
    """
    prediction_count = len(events)
    if prediction_count == 0:
        return _empty_metrics()

    malicious_count = sum(1 for event in events if event.get("prediction") == "malicious")
    confidences = [event["confidence"] for event in events if event.get("confidence") is not None]
    latencies = sorted(
        event["prediction_latency_ms"] for event in events if event.get("prediction_latency_ms") is not None
    )

    user_report_count = sum(1 for event in events if event.get("user_feedback") == "reported_incorrect")

    reviewed = [event for event in events if event.get("admin_reviewed_label") is not None]
    admin_correction_count = sum(1 for event in reviewed if event["admin_reviewed_label"] != event.get("prediction"))

    # False positive: model predicted malicious, admin-reviewed ground truth is benign.
    # False negative: model predicted benign, admin-reviewed ground truth is malicious.
    reviewed_predicted_malicious = [event for event in reviewed if event.get("prediction") == "malicious"]
    reviewed_predicted_benign = [event for event in reviewed if event.get("prediction") == "benign"]
    false_positives = sum(1 for event in reviewed_predicted_malicious if event["admin_reviewed_label"] == "benign")
    false_negatives = sum(1 for event in reviewed_predicted_benign if event["admin_reviewed_label"] == "malicious")

    estimated_fpr: Optional[float] = (
        false_positives / len(reviewed_predicted_malicious) if reviewed_predicted_malicious else None
    )
    estimated_fnr: Optional[float] = (
        false_negatives / len(reviewed_predicted_benign) if reviewed_predicted_benign else None
    )

    return {
        "prediction_count": prediction_count,
        "malicious_ratio": malicious_count / prediction_count,
        "mean_confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "latency_percentiles_ms": (
            {"p50": _percentile(latencies, 50), "p90": _percentile(latencies, 90), "p99": _percentile(latencies, 99)}
            if latencies
            else {"p50": None, "p90": None, "p99": None}
        ),
        "user_report_count": user_report_count,
        "admin_correction_count": admin_correction_count,
        "reviewed_count": len(reviewed),
        "estimated_false_positive_rate": estimated_fpr,
        "estimated_false_negative_rate": estimated_fnr,
        "estimate_note": (
            f"false-positive/false-negative rates are estimated over the admin-reviewed subset only "
            f"(N={len(reviewed)} of {prediction_count} total predictions) and are not population rates."
        ),
    }
