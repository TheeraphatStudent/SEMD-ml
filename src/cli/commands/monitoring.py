from __future__ import annotations

from typing import Any

from core import get_logger
from monitoring.metrics import compute_monitoring_metrics
from monitoring.store import monitoring_store

from ..common import emit_result

logger = get_logger(__name__)


def cmd_feedback(args: Any) -> int:
    updated = monitoring_store.set_user_feedback(args.prediction_id, args.status)
    if not updated:
        logger.error("No prediction event found for prediction_id=%s", args.prediction_id)
        return 1
    emit_result(
        {"prediction_id": args.prediction_id, "user_feedback": args.status, "updated": True},
        getattr(args, "output", None),
    )
    return 0


def cmd_review(args: Any) -> int:
    updated = monitoring_store.set_admin_review(args.prediction_id, args.label)
    if not updated:
        logger.error("No prediction event found for prediction_id=%s", args.prediction_id)
        return 1
    emit_result(
        {"prediction_id": args.prediction_id, "admin_reviewed_label": args.label, "updated": True},
        getattr(args, "output", None),
    )
    return 0


def cmd_monitor_report(args: Any) -> int:
    since = getattr(args, "since", None)
    events = monitoring_store.get_events(since=since)
    metrics = compute_monitoring_metrics(events)
    payload = {"since": since, **metrics}
    emit_result(payload, getattr(args, "output", None))
    return 0
