from __future__ import annotations

from typing import Any

from core import get_logger, settings
from ml import training_service
from monitoring.retraining import build_feedback_dataset

from ..common import emit_result, normalize_requested_algorithms, validate_algorithms

logger = get_logger(__name__)


def cmd_retrain_from_feedback(args: Any) -> int:
    """Manual retraining entrypoint: approved feedback -> new dataset version -> validation ->
    candidate training -> MLflow tracking. Stops there — registration and promotion are always a
    separate, explicit `register` / `gate-check` / `promote` invocation (see docs/retraining.md)."""
    algorithms = normalize_requested_algorithms(getattr(args, "model", None), getattr(args, "algorithms", None))
    algorithms = algorithms or settings.default_train_algorithms
    if not validate_algorithms(algorithms):
        return 1

    feedback = build_feedback_dataset(since=getattr(args, "since", None))
    if feedback["record_count"] == 0:
        logger.error(
            "No approved feedback available (no prediction event has an admin_reviewed_label set yet). "
            "Nothing to retrain — review predictions with `uv run semd-ml review` first."
        )
        emit_result({"status": "skipped", "reason": "no_approved_feedback"}, getattr(args, "output", None))
        return 1

    base_dataset_files = getattr(args, "dataset_files", None) or settings.default_dataset_files
    dataset_files = [*base_dataset_files, feedback["path"]]
    run_name = getattr(args, "run_name", None) or f"retrain_from_feedback_{feedback['record_count']}rows"

    result = training_service.execute_training(
        {
            "dataset_files": dataset_files,
            "algorithms": algorithms,
            "balance_method": getattr(args, "balance", None),
            "run_name": run_name,
        },
        run_kind="retraining",
    )
    result["feedback_dataset"] = feedback
    result["next_steps"] = (
        "Candidate trained and tracked in MLflow; nothing was registered or promoted automatically. "
        f"Review the run, then: uv run semd-ml register --run-id {result.get('tracking_run_id')}, "
        "uv run semd-ml gate-check, and uv run semd-ml promote once you approve it."
    )
    emit_result(result, getattr(args, "output", None))
    return 0 if result["status"] == "success" else 1
