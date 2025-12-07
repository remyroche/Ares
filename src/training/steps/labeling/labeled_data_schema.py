from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


LABELED_DATA_SCHEMA_VERSION = "1.0"

_BASE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "realized_return",
)


def get_required_labeled_data_columns(extra: Iterable[str] | None = None) -> list[str]:
    cols: list[str] = list(_BASE_REQUIRED_COLUMNS)
    if extra is not None:
        for name in extra:
            if name not in cols:
                cols.append(name)
    return cols


def validate_labeled_data_schema(
    df: pd.DataFrame,
    required_cols: Sequence[str] | None = None,
    context: str | None = None,
) -> None:
    if required_cols is None:
        required_cols = _BASE_REQUIRED_COLUMNS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        prefix = f"{context}: " if context else ""
        raise ValueError(prefix + "labeled_data missing required columns: " + ", ".join(missing))
