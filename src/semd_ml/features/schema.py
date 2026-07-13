from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


SCHEMA_VERSION = "2.1.0"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    description: str
    expected_range: Optional[List[float]]
    default_value: float
    schema_version: str = SCHEMA_VERSION


class FeatureSchema:
    def __init__(self, features: Iterable[FeatureSpec], schema_version: str = SCHEMA_VERSION):
        self.features = list(features)
        self.schema_version = schema_version
        self._defaults = {feature.name: feature.default_value for feature in self.features}
        self._dtypes = {feature.name: feature.dtype for feature in self.features}

    @property
    def feature_names(self) -> List[str]:
        return [feature.name for feature in self.features]

    def align_record(self, record: Dict[str, Any]) -> Dict[str, float]:
        aligned: Dict[str, float] = {}
        for feature in self.features:
            value = record.get(feature.name, feature.default_value)
            if value is None:
                value = feature.default_value
            aligned[feature.name] = float(value)
        return aligned

    def align_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        aligned = frame.copy()
        for feature in self.features:
            if feature.name not in aligned.columns:
                aligned[feature.name] = feature.default_value
        aligned = aligned[self.feature_names]
        return aligned.fillna(self._defaults).astype(float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "features": [asdict(feature) for feature in self.features],
        }

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FeatureSchema":
        features = [FeatureSpec(**item) for item in payload["features"]]
        return cls(features=features, schema_version=payload["schema_version"])

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureSchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def build_feature_schema(features_config: Any) -> FeatureSchema:
    if getattr(features_config, "features", None):
        configured = features_config.features
    else:
        configured = []
        for group in features_config.feature_groups.values():
            configured.extend(group.get("features", []))

    feature_specs: List[FeatureSpec] = []
    for item in configured:
        if isinstance(item, str):
            item = {"name": item}
        dtype = item.get("type", "numeric")
        expected_range = _expected_range_for_type(dtype)
        feature_specs.append(
            FeatureSpec(
                name=item["name"],
                dtype=dtype,
                description=item.get("description", ""),
                expected_range=expected_range,
                default_value=float(item.get("default_value", 0.0)),
            )
        )

    return FeatureSchema(feature_specs)


def _expected_range_for_type(dtype: str) -> Optional[List[float]]:
    if dtype == "binary":
        return [0.0, 1.0]
    if dtype == "ratio":
        return [0.0, 1.0]
    return None
