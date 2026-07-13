from __future__ import annotations

from typing import Any

from core import get_logger
from tracking.model_registry import ModelRegistryManager

from ..common import emit_result

logger = get_logger(__name__)


def cmd_register_model(args: Any) -> int:
    manager = ModelRegistryManager()
    payload = manager.register_candidate(args.run_id)
    emit_result(payload, getattr(args, "output", None))
    return 0


def cmd_promote_model(args: Any) -> int:
    manager = ModelRegistryManager()
    payload = manager.promote_candidate(args.model_version)
    emit_result(payload, getattr(args, "output", None))
    return 0


def cmd_rollback_model(args: Any) -> int:
    manager = ModelRegistryManager()
    payload = manager.rollback_to_previous_champion()
    emit_result(payload, getattr(args, "output", None))
    return 0


def cmd_gate_check(args: Any) -> int:
    """Preview promotion-gate evaluation without mutating any alias — a dry run of `promote`."""
    manager = ModelRegistryManager()
    payload = manager.validate_candidate(args.model_version)
    emit_result(payload, getattr(args, "output", None))
    return 0 if payload["passed"] else 1
