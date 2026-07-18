from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd

RAW_DIR_ALIASES = {"dataset/raw", "raw"}


class DatasetRepository:
    """Filesystem access for raw dataset files and the merged-dataset cache."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def resolve_file_list(self, dataset_files: List[str]) -> List[str]:
        """Expand raw-dir aliases (e.g. 'dataset/raw') into filenames on disk."""
        resolved: List[str] = []
        for file_path in dataset_files:
            if file_path in RAW_DIR_ALIASES:
                if os.path.isdir(self.dataset_path):
                    resolved.extend(
                        sorted(
                            name
                            for name in os.listdir(self.dataset_path)
                            if name.endswith((".csv", ".xlsx")) and name != "merged.csv"
                        )
                    )
            else:
                resolved.append(file_path)
        return resolved

    def full_path(self, file_path: str) -> str:
        return os.path.join(self.dataset_path, file_path)

    def exists(self, file_path: str) -> bool:
        return os.path.exists(self.full_path(file_path))

    def save_merged(self, merged: pd.DataFrame, source_references: List[Dict[str, Any]]) -> str:
        merged_path = os.path.join(self.dataset_path, "merged.csv")
        merged.to_csv(merged_path, index=False)
        metadata_path = os.path.join(self.dataset_path, "merged.metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"source_references": source_references, "total_records": len(merged)},
                handle,
                indent=2,
            )
        return merged_path
