from typing import Any

from core import get_logger, settings
from ml import training_service

from ..common import emit_result, normalize_requested_algorithms, validate_algorithms

logger = get_logger(__name__)


def cmd_evaluate(args: Any) -> int:
    algorithms = normalize_requested_algorithms(args.model, args.algorithms)
    algorithms = algorithms or settings.available_algorithms
    if not validate_algorithms(algorithms):
        return 1

    result = training_service.execute_evaluation(
        {
            "dataset_files": getattr(args, "dataset_files", None) or settings.default_dataset_files,
            "algorithms": algorithms,
            "run_name": getattr(args, "run_name", None),
            "balance_method": getattr(args, "balance", None),
        }
    )
    emit_result(result, getattr(args, "output", None))
    return 0 if result["status"] == "success" else 1
