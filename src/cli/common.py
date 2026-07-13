from typing import Any

from core import get_logger, settings
from core.reporting import emit_json_result
from ml import ml_pipeline

logger = get_logger(__name__)


def emit_result(data: Any, output: str = None) -> None:
    emit_json_result(data, output, settings.reports_path, logger)


def validate_algorithms(algorithms: list) -> bool:
    if not algorithms:
        return True

    available_algorithms = list(ml_pipeline.get_algorithm_configs().keys())
    invalid_algorithms = [
        alg for alg in algorithms if alg not in available_algorithms]

    if invalid_algorithms:
        logger.error(f"Invalid algorithm(s): {', '.join(invalid_algorithms)}")
        logger.error(
            f"Available algorithms: {', '.join(available_algorithms)}")
        return False

    return True


def normalize_requested_algorithms(model: str | None, algorithms: list | None) -> list | None:
    if model and algorithms:
        return [model, *algorithms]
    if model:
        return [model]
    return algorithms
