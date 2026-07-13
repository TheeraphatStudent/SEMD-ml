from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from semd_ml.features.url_normalizer import extract_registered_domain, normalize_url


@dataclass
class DatasetValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetValidator:
    def __init__(self, data_dict: Dict[str, Any]):
        self.data_dict = data_dict
        self.classes = list(self.data_dict.get("class_mapping", {}).keys()) or ["benign", "malicious"]
        self.class_mapping = self._build_class_mapping()

    def standardize_dataframe(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        standardized = pd.DataFrame()
        url_fields = self.data_dict["fields"]["url"]
        class_fields = self.data_dict["fields"]["class"]

        url_col = next((field for field in url_fields if field in df.columns), None)
        if url_col is None:
            raise ValueError(f"{source_name}: missing URL column, expected one of {url_fields}")

        class_col = next((field for field in class_fields if field in df.columns), None)
        standardized["url"] = df[url_col]
        standardized["raw_label"] = df[class_col] if class_col else None
        standardized["source"] = source_name
        standardized["source_row"] = range(1, len(df) + 1)
        standardized["source_url_column"] = url_col
        standardized["source_label_column"] = class_col or ""
        standardized["type"] = df["type"] if "type" in df.columns else None
        return standardized

    def validate(self, df: pd.DataFrame) -> DatasetValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        if "url" not in df.columns:
            return DatasetValidationResult(False, ["Missing 'url' column"], warnings, {})
        if "raw_label" not in df.columns and "label" not in df.columns:
            return DatasetValidationResult(False, ["Missing label column"], warnings, {})

        working = df.copy()
        label_col = "label" if "label" in working.columns else "raw_label"
        working["raw_label"] = working[label_col]
        working["normalized_label"] = working["raw_label"].apply(self.normalize_label)
        working["raw_url"] = working["url"]
        working["url_missing"] = working["raw_url"].isna()
        working["url_empty"] = working["raw_url"].fillna("").astype(str).str.strip().eq("")
        normalized = working["raw_url"].apply(normalize_url)
        working["normalized_url"] = normalized.map(lambda item: item.normalized_url)
        working["registered_domain"] = normalized.map(lambda item: item.registered_domain)
        working["url_valid"] = normalized.map(lambda item: item.is_valid)
        working["normalization_error"] = normalized.map(lambda item: item.error)
        working["label_missing"] = working["raw_label"].isna() | working["raw_label"].astype(str).str.strip().eq("")
        working["label_invalid"] = ~working["label_missing"] & ~working["normalized_label"].isin(self.classes)
        working["empty_url_type"] = False
        if "type" in working.columns:
            working["empty_url_type"] = working["type"].isna() | working["type"].astype(str).str.strip().eq("")

        valid_rows = working[
            ~working["url_missing"]
            & ~working["url_empty"]
            & working["url_valid"]
            & ~working["label_missing"]
            & ~working["label_invalid"]
        ].copy()

        duplicate_count = int(valid_rows.duplicated(subset=["normalized_url"], keep="first").sum())
        conflicting = (
            valid_rows.groupby("normalized_url")["normalized_label"].nunique().reset_index(name="label_count")
        )
        conflicting_urls = conflicting.loc[conflicting["label_count"] > 1, "normalized_url"].tolist()
        conflicting_label_count = len(conflicting_urls)

        class_counts = valid_rows.loc[
            ~valid_rows["normalized_url"].isin(conflicting_urls), "normalized_label"
        ].value_counts().to_dict()
        stats = {
            "total_records": int(len(working)),
            "missing_url_count": int(working["url_missing"].sum()),
            "empty_url_count": int(working["url_empty"].sum()),
            "invalid_url_count": int((~working["url_valid"] & ~working["url_empty"] & ~working["url_missing"]).sum()),
            "duplicate_count": duplicate_count,
            "conflicting_label_count": conflicting_label_count,
            "missing_label_count": int(working["label_missing"].sum()),
            "invalid_label_count": int(working["label_invalid"].sum()),
            "empty_url_type_count": int(working["empty_url_type"].sum()),
            "unique_domain_count": int(valid_rows["registered_domain"].dropna().nunique()),
            "class_counts": class_counts,
            "class_imbalance": self._build_class_imbalance(class_counts),
            "source_metadata": self._build_source_metadata(working),
            "invalid_url_examples": sorted(
                set(
                    working.loc[
                        ~working["url_valid"] & ~working["url_empty"] & ~working["url_missing"],
                        "raw_url",
                    ].astype(str).head(5).tolist()
                )
            ),
            "conflicting_urls": conflicting_urls[:5],
        }

        if stats["missing_url_count"]:
            errors.append(f"Missing URLs: {stats['missing_url_count']}")
        if stats["empty_url_count"]:
            errors.append(f"Empty URLs: {stats['empty_url_count']}")
        if stats["invalid_url_count"]:
            errors.append(f"Invalid URLs: {stats['invalid_url_count']}")
        if stats["missing_label_count"]:
            errors.append(f"Missing labels: {stats['missing_label_count']}")
        if stats["invalid_label_count"]:
            errors.append(f"Invalid labels: {stats['invalid_label_count']}")
        if stats["conflicting_label_count"]:
            errors.append(f"Conflicting labels: {stats['conflicting_label_count']}")
        if stats["duplicate_count"]:
            warnings.append(f"Duplicate normalized URLs: {stats['duplicate_count']}")
        if stats["empty_url_type_count"]:
            warnings.append(f"Empty URL types: {stats['empty_url_type_count']}")

        return DatasetValidationResult(is_valid=not errors, errors=errors, warnings=warnings, stats=stats)

    def clean(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DatasetValidationResult]:
        validation = self.validate(df)
        working = df.copy()
        label_col = "label" if "label" in working.columns else "raw_label"
        working["label"] = working[label_col].apply(self.normalize_label)
        normalized = working["url"].apply(normalize_url)
        working["normalized_url"] = normalized.map(lambda item: item.normalized_url)
        working["registered_domain"] = normalized.map(lambda item: item.registered_domain)
        working["is_valid_url"] = normalized.map(lambda item: item.is_valid)
        working["url"] = working["url"].fillna("").astype(str).str.strip()

        cleaned = working[
            working["is_valid_url"]
            & working["label"].isin(self.classes)
            & working["url"].ne("")
        ].copy()

        conflicting = cleaned.groupby("normalized_url")["label"].nunique()
        conflicting_urls = conflicting[conflicting > 1].index.tolist()
        if conflicting_urls:
            cleaned = cleaned.loc[~cleaned["normalized_url"].isin(conflicting_urls)].copy()

        cleaned = cleaned.sort_values(["normalized_url", "label", "source", "source_row"])
        cleaned = cleaned.drop_duplicates(subset=["normalized_url"], keep="first").reset_index(drop=True)
        cleaned["url"] = cleaned["normalized_url"]
        cleaned["registered_domain"] = cleaned["registered_domain"].fillna(
            cleaned["normalized_url"].map(lambda value: extract_registered_domain(None if value is None else value.split("/")[2]))
        )
        return cleaned[["url", "label", "normalized_url", "registered_domain", "source", "source_row"]], validation

    def normalize_label(self, label: object) -> str:
        if pd.isna(label) or label is None:
            return ""
        value = str(int(label)) if isinstance(label, float) and label.is_integer() else str(label)
        normalized = value.lower().strip()
        return self.class_mapping.get(normalized, normalized)

    def _build_class_mapping(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for target_class, values in self.data_dict.get("class_mapping", {}).items():
            for value in values:
                mapping[str(value).lower()] = target_class.lower()
        return mapping

    def _build_source_metadata(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        rows = []
        for source_name, group in df.groupby("source", dropna=False):
            rows.append(
                {
                    "source": source_name,
                    "records": int(len(group)),
                    "url_column": group["source_url_column"].iloc[0] if "source_url_column" in group else "",
                    "label_column": group["source_label_column"].iloc[0] if "source_label_column" in group else "",
                }
            )
        return rows

    def _build_class_imbalance(self, class_counts: Dict[str, int]) -> Dict[str, Any]:
        if not class_counts:
            return {"is_imbalanced": False, "severity": "unknown", "ratio": 0.0}
        counts = list(class_counts.values())
        ratio = max(counts) / min(counts) if min(counts) else 0.0
        if ratio < 2.0:
            severity = "balanced"
            is_imbalanced = False
        elif ratio < 5.0:
            severity = "mild"
            is_imbalanced = True
        elif ratio < 10.0:
            severity = "moderate"
            is_imbalanced = True
        else:
            severity = "severe"
            is_imbalanced = True
        return {"is_imbalanced": is_imbalanced, "severity": severity, "ratio": ratio}
