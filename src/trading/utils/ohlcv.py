"""Utilities for working with OHLCV market data frames."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

_DEFAULT_OHLCV_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]


def ensure_ohlcv_dataframe(
    frame: Any,
    required_columns: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    allow_empty: bool = True,
) -> Optional[pd.DataFrame]:
    """Normalise a frame-like object into a datetime-indexed OHLCV DataFrame.

    Args:
        frame: The candidate data structure (DataFrame, list of rows, dict of columns).
        required_columns: Iterable with the columns that must be present/order enforced.
        limit: Optional maximum number of rows to retain from the tail of the frame.
        allow_empty: When ``True`` return an empty DataFrame if the input has no rows;
            otherwise return ``None`` when the frame cannot be converted.

    Returns:
        A DataFrame containing the requested columns indexed by timestamp, or
        ``None`` if the payload could not be converted and ``allow_empty`` is
        ``False``.
    """

    columns = list(required_columns) if required_columns is not None else list(_DEFAULT_OHLCV_COLUMNS)

    if isinstance(frame, pd.DataFrame):
        df = frame.copy()
    elif isinstance(frame, (list, tuple)):
        df = pd.DataFrame(frame)
    elif isinstance(frame, dict):
        df = pd.DataFrame(frame)
    elif frame is None:
        return pd.DataFrame(columns=columns) if allow_empty else None
    else:
        return None

    if df.empty:
        empty_df = pd.DataFrame(columns=columns)
        empty_df.index = pd.DatetimeIndex([], name="timestamp")
        return empty_df if allow_empty else None

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
            if timestamps.notna().any():
                df = df.assign(_timestamp=timestamps).set_index("_timestamp")
                df.index.name = "timestamp"
                df = df.drop(columns=["timestamp"], errors="ignore")
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

    df = df.sort_index()
    df = df[~df.index.isna()]

    normalized = pd.DataFrame(index=df.index)

    for column in columns:
        if column in df.columns:
            normalized[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            normalized[column] = np.nan

    normalized = normalized[columns]

    if limit is not None and limit > 0 and len(normalized) > limit:
        normalized = normalized.tail(limit)

    return normalized
