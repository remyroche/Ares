from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SIDE_LONG = np.int8(1)
SIDE_SHORT = np.int8(-1)


def _aligned_series(value: Any, n: int, *, name: str) -> pd.Series:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return pd.Series([value] * int(n), dtype=object)
    ser = pd.Series(value, dtype=object).reset_index(drop=True)
    if len(ser) != int(n):
        raise ValueError(f"{name} length mismatch: got {len(ser)}, expected {int(n)}")
    return ser


def normalise_side_value(side: Any) -> np.int8:
    """Return the canonical numeric candidate side: +1 for long, -1 for short."""

    if isinstance(side, str):
        text = side.strip().lower()
        if text in {"long", "buy", "+1", "1"}:
            return SIDE_LONG
        if text in {"short", "sell", "-1"}:
            return SIDE_SHORT
    try:
        value = float(side)
    except Exception as exc:
        raise ValueError(f"Unsupported side value: {side!r}") from exc
    if value > 0:
        return SIDE_LONG
    if value < 0:
        return SIDE_SHORT
    raise ValueError(f"Unsupported flat side value: {side!r}")


def normalise_side_array(side: Any, n: int | None = None) -> np.ndarray:
    """Vectorize side normalization for scalars or row-level side columns."""

    if isinstance(side, pd.Series):
        raw = side.to_numpy()
    else:
        raw = np.asarray(side)
    if raw.ndim == 0:
        if n is None:
            n = 1
        return np.full(int(n), normalise_side_value(raw.item()), dtype=np.int8)
    out = np.array([normalise_side_value(value) for value in raw], dtype=np.int8)
    if n is not None and len(out) != int(n):
        raise ValueError(f"Side length mismatch: got {len(out)}, expected {int(n)}")
    return out


def side_name(side: Any) -> str:
    return "short" if int(normalise_side_value(side)) < 0 else "long"


def side_name_array(side: Any, n: int | None = None) -> np.ndarray:
    vals = normalise_side_array(side, n=n)
    return np.where(vals < 0, "short", "long").astype(object)


def side_adjust_return(raw_future_return: Any, side: Any) -> np.ndarray:
    """Make positive return mean the candidate side made money before costs."""

    returns = np.asarray(raw_future_return, dtype=np.float64)
    sides = normalise_side_array(side, n=returns.size).reshape(returns.shape)
    return (returns * sides).astype(np.float32, copy=False)


def side_aware_path_metrics(
    *,
    entry_price: Any,
    future_high: Any,
    future_low: Any,
    future_close: Any,
    side: Any,
) -> pd.DataFrame:
    """Compute side-normalized future return plus adverse/favorable excursions.

    The inputs are row-aligned arrays. `future_high` and `future_low` should be
    the max/min over the future path used by the target horizon.
    """

    entry = np.asarray(entry_price, dtype=np.float64)
    high = np.asarray(future_high, dtype=np.float64)
    low = np.asarray(future_low, dtype=np.float64)
    close = np.asarray(future_close, dtype=np.float64)
    if not (entry.shape == high.shape == low.shape == close.shape):
        raise ValueError("entry_price, future_high, future_low, and future_close must align")
    sides = normalise_side_array(side, n=entry.size).reshape(entry.shape)
    denom = np.maximum(entry, 1e-12)
    raw_future_return = close / denom - 1.0
    long_adverse = np.maximum(entry / np.maximum(low, 1e-12) - 1.0, 0.0)
    long_favorable = np.maximum(high / denom - 1.0, 0.0)
    short_adverse = np.maximum(high / denom - 1.0, 0.0)
    short_favorable = np.maximum(entry / np.maximum(low, 1e-12) - 1.0, 0.0)
    is_short = sides < 0
    adverse = np.where(is_short, short_adverse, long_adverse)
    favorable = np.where(is_short, short_favorable, long_favorable)
    return pd.DataFrame(
        {
            "side": sides.astype(np.int8, copy=False).ravel(),
            "raw_future_return": raw_future_return.astype(np.float32, copy=False).ravel(),
            "side_adjusted_return": (raw_future_return * sides)
            .astype(np.float32, copy=False)
            .ravel(),
            "adverse_excursion": adverse.astype(np.float32, copy=False).ravel(),
            "favorable_excursion": favorable.astype(np.float32, copy=False).ravel(),
        }
    )


def candidate_id_series(
    timestamp: Any,
    asset: Any,
    timeframe: Any,
    side: Any,
) -> pd.Series:
    assets = pd.Series(asset, dtype=object).astype(str).reset_index(drop=True)
    n = len(assets)
    ts = pd.to_datetime(
        _aligned_series(timestamp, n, name="timestamp"),
        utc=True,
        errors="coerce",
    )
    sides = side_name_array(side, n=n)
    timeframes = _aligned_series(timeframe, n, name="timeframe").astype(str)
    ts_key = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("NaT").reset_index(drop=True)
    return assets + "|" + ts_key + "|" + timeframes + "|" + pd.Series(sides)


def add_side_contract_columns(
    frame: pd.DataFrame,
    *,
    side: Any,
    timestamp_col: str,
    asset_col: str,
    timeframe: Any,
    copy: bool = True,
) -> pd.DataFrame:
    """Attach the row-level side/candidate-id contract to a candidate table."""

    if timestamp_col not in frame.columns or asset_col not in frame.columns:
        raise KeyError(f"Missing timestamp/asset columns: {timestamp_col!r}, {asset_col!r}")
    out = frame.copy() if copy else frame
    sides = normalise_side_array(side, n=len(out))
    out["side"] = sides.astype(np.int8, copy=False)
    out["side_name"] = side_name_array(sides, n=len(out))
    out["timeframe"] = _aligned_series(timeframe, len(out), name="timeframe").astype(str).to_numpy()
    out["candidate_id"] = candidate_id_series(
        out[timestamp_col],
        out[asset_col],
        out["timeframe"],
        sides,
    ).to_numpy(dtype=object)
    return out


def expand_side_candidates(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    asset_col: str,
    timeframe: Any,
    sides: tuple[Any, ...] = (SIDE_LONG, SIDE_SHORT),
) -> pd.DataFrame:
    """Duplicate each timestamp/asset row into explicit side-aware candidates."""

    if timestamp_col not in frame.columns or asset_col not in frame.columns:
        raise KeyError(f"Missing timestamp/asset columns: {timestamp_col!r}, {asset_col!r}")
    pieces = [
        add_side_contract_columns(
            frame,
            side=side,
            timestamp_col=timestamp_col,
            asset_col=asset_col,
            timeframe=timeframe,
            copy=True,
        )
        for side in sides
    ]
    out = pd.concat(pieces, axis=0, ignore_index=True)
    validate_side_candidate_contract(out)
    return out


def validate_side_candidate_contract(
    frame: pd.DataFrame,
    *,
    require_candidate_id: bool = True,
    require_unique_candidate_id: bool = True,
) -> dict[str, Any]:
    missing = [col for col in ("side",) if col not in frame.columns]
    if require_candidate_id and "candidate_id" not in frame.columns:
        missing.append("candidate_id")
    if missing:
        raise ValueError(f"Missing side-aware candidate columns: {missing}")
    sides = normalise_side_array(frame["side"], n=len(frame))
    unique_sides = sorted({int(value) for value in sides})
    duplicate_candidate_ids = 0
    if require_candidate_id and "candidate_id" in frame.columns:
        ids = frame["candidate_id"].astype(str)
        duplicate_candidate_ids = int(ids.duplicated().sum())
        if require_unique_candidate_id and duplicate_candidate_ids:
            raise ValueError(f"candidate_id is not unique: duplicates={duplicate_candidate_ids}")
    return {
        "rows": int(len(frame)),
        "long_rows": int(np.sum(sides > 0)),
        "short_rows": int(np.sum(sides < 0)),
        "unique_sides": unique_sides,
        "duplicate_candidate_ids": duplicate_candidate_ids,
    }
