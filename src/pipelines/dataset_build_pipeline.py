from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from data.loaders import load_dataset_file
from data.repositories import DatasetRepository
from data.validators import DatasetValidator
from data.versioning import build_dataset_metadata, compute_dataset_hash

logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildResult:
    merged: pd.DataFrame
    cleaned: pd.DataFrame
    dataset_hash: str
    dataset_metadata: Dict[str, Any]
    validation_report: Dict[str, Any]


class DatasetBuildPipeline:
    """Loads, merges, validates, cleans, and fingerprints a raw dataset selection.

    This is the load -> validate -> clean -> version slice of dataset
    preparation. Feature extraction, splitting, and class balancing stay in
    the caller since they depend on the feature schema and training config,
    not on dataset construction itself.
    """

    def __init__(self, dataset_path: str, validator: DatasetValidator):
        self.repository = DatasetRepository(dataset_path)
        self.validator = validator

    def load_and_merge(self, dataset_files: List[str]) -> pd.DataFrame:
        standardized_frames = []
        source_references = []
        for file_path in self.repository.resolve_file_list(dataset_files):
            full_path = self.repository.full_path(file_path)
            if not self.repository.exists(file_path):
                logger.warning("Dataset file not found: %s", full_path)
                continue
            try:
                frame = load_dataset_file(full_path)
            except Exception as exc:
                logger.error("Error loading %s: %s", file_path, exc)
                continue
            standardized = self.validator.standardize_dataframe(frame, file_path)
            standardized_frames.append(standardized)
            source_references.append({"source": file_path, "path": full_path, "records": len(standardized)})

        if not standardized_frames:
            raise ValueError("No valid datasets loaded")

        merged = pd.concat(standardized_frames, ignore_index=True)
        self.repository.save_merged(merged, source_references)
        return merged

    def build(self, dataset_files: List[str]) -> DatasetBuildResult:
        merged = self.load_and_merge(dataset_files)

        validation = self.validator.validate(merged)
        if validation.errors:
            logger.warning("Dataset validation reported issues prior to cleaning: %s", validation.errors)

        cleaned, validation = self.validator.clean(merged)
        if cleaned.empty:
            raise ValueError(f"Dataset cleaning removed all rows. Validation errors: {validation.errors}")

        dataset_hash = compute_dataset_hash(cleaned, validation.stats["source_metadata"])
        dataset_metadata = build_dataset_metadata(
            cleaned_df=cleaned,
            validation_stats=validation.stats,
            dataset_hash=dataset_hash,
            source_references=validation.stats["source_metadata"],
        )

        return DatasetBuildResult(
            merged=merged,
            cleaned=cleaned,
            dataset_hash=dataset_hash,
            dataset_metadata=dataset_metadata,
            validation_report=validation.to_dict(),
        )
