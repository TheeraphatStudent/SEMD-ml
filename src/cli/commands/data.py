from typing import Any

from core import get_logger, settings
from data import dataset_pipeline

from ..common import emit_result

logger = get_logger(__name__)


def cmd_data_validate(args: Any) -> int:
    dataset_files = getattr(args, "dataset_files", None) or settings.default_dataset_files
    merged = dataset_pipeline.load_and_merge_datasets(dataset_files)
    is_valid, issues = dataset_pipeline.validate_dataset(merged)
    cleaned = dataset_pipeline.preprocess_dataset(merged)
    payload = {
        "status": "success" if is_valid else "failed",
        "dataset_files": dataset_files,
        "is_valid": is_valid,
        "issues": issues,
        "validation_report": dataset_pipeline.last_validation_report,
        "cleaned_records": len(cleaned),
    }
    emit_result(payload, getattr(args, "output", None))
    return 0 if is_valid else 1
