from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReferenceLoadResult:
    """Diagnostic record for one reference CSV load attempt."""

    name: str
    path: Path
    loaded: bool
    used_default: bool
    value_count: int
    error: Optional[str] = None


class ReferenceStore:
    """Loads feature-reference CSVs (brand keywords, suspicious TLDs, ...) from a directory.

    Falls back to a caller-supplied default set on any load failure, but records
    the failure in `diagnostics` and logs a warning instead of failing silently.
    """

    def __init__(self, root: Path):
        self.root = root
        self.diagnostics: Dict[str, ReferenceLoadResult] = {}

    def load(
        self,
        name: str,
        default: List[str],
        transform: Optional[Callable[[str], str]] = None,
    ) -> Set[str]:
        csv_path = self.root / f"{name}.csv"
        values: Optional[List[str]] = None
        error: Optional[str] = None

        if not csv_path.exists():
            error = f"reference file not found: {csv_path}"
        else:
            try:
                frame = pd.read_csv(csv_path)
                if "value" in frame.columns:
                    values = frame["value"].dropna().astype(str).str.strip().tolist()
                else:
                    error = f"missing 'value' column in {csv_path}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

        used_default = values is None
        raw_values = default if used_default else values
        if transform:
            raw_values = [transform(value) for value in raw_values]
        result = {value.lower() for value in raw_values if value}

        if error:
            logger.warning("Reference store: '%s' falling back to defaults (%s)", name, error)

        self.diagnostics[name] = ReferenceLoadResult(
            name=name,
            path=csv_path,
            loaded=not used_default,
            used_default=used_default,
            value_count=len(result),
            error=error,
        )
        return result

    def has_failures(self) -> bool:
        return any(result.error for result in self.diagnostics.values())
