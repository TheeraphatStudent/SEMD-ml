from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

DATASET_VERSION = "1.0.0"


def compute_dataset_hash(df: pd.DataFrame, source_references: List[Dict[str, Any]] | None = None) -> str:
    payload = {
        "records": sorted(
            df.loc[:, [column for column in ["normalized_url", "label", "source"] if column in df.columns]]
            .fillna("")
            .astype(str)
            .to_dict(orient="records"),
            key=lambda item: (item.get("normalized_url", ""), item.get("label", ""), item.get("source", "")),
        ),
        "sources": sorted(source_references or [], key=lambda item: item.get("source", "")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dataset_metadata(
    cleaned_df: pd.DataFrame,
    validation_stats: Dict[str, Any],
    dataset_hash: str,
    source_references: List[Dict[str, Any]],
    dataset_version: str = DATASET_VERSION,
) -> Dict[str, Any]:
    benign_count = int((cleaned_df["label"] == "benign").sum()) if "label" in cleaned_df.columns else 0
    malicious_count = int((cleaned_df["label"] == "malicious").sum()) if "label" in cleaned_df.columns else 0
    return {
        "dataset_version": dataset_version,
        "dataset_hash": dataset_hash,
        "total_records": int(validation_stats.get("total_records", len(cleaned_df))),
        "valid_records": int(len(cleaned_df)),
        "invalid_records": int(validation_stats.get("total_records", len(cleaned_df)) - len(cleaned_df)),
        "duplicate_count": int(validation_stats.get("duplicate_count", 0)),
        "conflicting_label_count": int(validation_stats.get("conflicting_label_count", 0)),
        "benign_count": benign_count,
        "malicious_count": malicious_count,
        "unique_domains": int(cleaned_df["registered_domain"].dropna().nunique()) if "registered_domain" in cleaned_df.columns else 0,
        "created_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_references": source_references,
    }
