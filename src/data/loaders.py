from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


class UnsupportedDatasetFormatError(ValueError):
    pass


def load_dataset_file(path: Union[str, Path]) -> pd.DataFrame:
    """Read a single raw dataset file (CSV or XLSX) into a DataFrame.

    CSVs that parse down to a single column are retried with a `;` separator —
    several source datasets in this project ship semicolon-delimited.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
            if len(frame.columns) == 1:
                frame = pd.read_csv(path, sep=";", on_bad_lines="skip")
        except pd.errors.ParserError:
            frame = pd.read_csv(path, sep=";", on_bad_lines="skip")
        return frame

    if suffix == ".xlsx":
        return pd.read_excel(path)

    raise UnsupportedDatasetFormatError(f"Unsupported file format: {path}")
