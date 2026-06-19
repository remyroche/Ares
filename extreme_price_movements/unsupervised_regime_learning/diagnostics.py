"""Quality, dynamics diagnostics, and correlation pruning for regime features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DYNAMICS_COMPONENT_WEIGHTS = {
    "mean_instability": 0.20,
    "variance_instability": 0.20,
    "quantile_instability": 0.15,
    "distribution_shift": 0.20,
    "autocorr_change": 0.15,
    "tail_change": 0.10,
}


@dataclass(frozen=True)
class FeatureSelectionResult:
    selected_features: list[str]
    diagnostics: pd.DataFrame
    quality_report: pd.DataFrame
    spearman_threshold: float
    candidate_features: list[str]


@dataclass(frozen=True)
class PreparedFrameContext:
    n_rows: int
    symbol_col: str
    timestamp_col: str
    has_symbol: bool
    has_timestamp: bool
    order: np.ndarray
    inverse_order: np.ndarray
    order_is_identity: bool
    timestamp_ns: np.ndarray
    timestamp_valid: np.ndarray
    sorted_timestamp_ns: np.ndarray
    sorted_timestamp_valid: np.ndarray
    symbols: np.ndarray
    sorted_symbols: np.ndarray
    sorted_group_starts: np.ndarray
    sorted_group_ends: np.ndarray


def numeric_feature_frame(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    dtype: object = np.float32,
) -> pd.DataFrame:
    """Return a numeric frame with missing features materialized as NaN."""

    data: dict[str, np.ndarray] = {}
    for col in dict.fromkeys(str(c) for c in feature_columns if str(c)):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").to_numpy(
                dtype=dtype,
                copy=False,
            )
        else:
            values = np.full(len(frame), np.nan, dtype=dtype)
        values = np.asarray(values, dtype=dtype)
        if np.issubdtype(values.dtype, np.floating):
            values = np.where(np.isfinite(values), values, np.nan).astype(
                dtype,
                copy=False,
            )
        data[col] = values
    return pd.DataFrame(data, index=frame.index, dtype=dtype)


def _contiguous_group_bounds(
    values: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_rows <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    if values.size != n_rows:
        return np.array([0], dtype=np.int64), np.array([n_rows], dtype=np.int64)
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(
        [np.array([0], dtype=np.int64), changes.astype(np.int64)]
    )
    ends = np.concatenate(
        [changes.astype(np.int64), np.array([n_rows], dtype=np.int64)]
    )
    return starts, ends


def _valid_context(
    context: PreparedFrameContext | None,
    frame: pd.DataFrame,
    *,
    symbol_col: str,
    timestamp_col: str,
) -> PreparedFrameContext | None:
    if context is None:
        return None
    if context.n_rows != len(frame):
        return None
    if context.symbol_col != str(symbol_col) or context.timestamp_col != str(
        timestamp_col
    ):
        return None
    return context


def prepare_frame_context(
    frame: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> PreparedFrameContext:
    """Precompute reusable row-order, timestamp, and symbol-group metadata."""

    n = len(frame)
    order = time_sort_order(
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    ).astype(np.int64, copy=False)
    inverse_order = np.empty(n, dtype=np.int64)
    if n:
        inverse_order[order] = np.arange(n, dtype=np.int64)
    has_timestamp = timestamp_col in frame.columns
    if has_timestamp:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        timestamp_valid = ts.notna().to_numpy(dtype=bool)
        timestamp_ns = ts.to_numpy(dtype="datetime64[ns]").astype(
            "int64",
            copy=False,
        )
    else:
        timestamp_valid = np.zeros(n, dtype=bool)
        timestamp_ns = np.zeros(n, dtype=np.int64)
    has_symbol = symbol_col in frame.columns
    if has_symbol:
        symbols = frame[symbol_col].astype(str).to_numpy()
    else:
        symbols = np.repeat("__all__", n)
    sorted_symbols = symbols[order] if n else symbols
    starts, ends = _contiguous_group_bounds(sorted_symbols, n)
    return PreparedFrameContext(
        n_rows=n,
        symbol_col=str(symbol_col),
        timestamp_col=str(timestamp_col),
        has_symbol=has_symbol,
        has_timestamp=has_timestamp,
        order=order,
        inverse_order=inverse_order,
        order_is_identity=bool(
            np.array_equal(order, np.arange(n, dtype=np.int64))
        ),
        timestamp_ns=timestamp_ns.astype(np.int64, copy=False),
        timestamp_valid=timestamp_valid,
        sorted_timestamp_ns=(
            timestamp_ns[order].astype(np.int64, copy=False) if n else timestamp_ns
        ),
        sorted_timestamp_valid=timestamp_valid[order] if n else timestamp_valid,
        symbols=symbols,
        sorted_symbols=sorted_symbols,
        sorted_group_starts=starts,
        sorted_group_ends=ends,
    )


def _sorted_context(context: PreparedFrameContext | None) -> PreparedFrameContext | None:
    if context is None:
        return None
    n = int(context.n_rows)
    order = np.arange(n, dtype=np.int64)
    return PreparedFrameContext(
        n_rows=n,
        symbol_col=context.symbol_col,
        timestamp_col=context.timestamp_col,
        has_symbol=context.has_symbol,
        has_timestamp=context.has_timestamp,
        order=order,
        inverse_order=order,
        order_is_identity=True,
        timestamp_ns=context.sorted_timestamp_ns,
        timestamp_valid=context.sorted_timestamp_valid,
        sorted_timestamp_ns=context.sorted_timestamp_ns,
        sorted_timestamp_valid=context.sorted_timestamp_valid,
        symbols=context.sorted_symbols,
        sorted_symbols=context.sorted_symbols,
        sorted_group_starts=context.sorted_group_starts,
        sorted_group_ends=context.sorted_group_ends,
    )


def time_sort_order(
    frame: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
) -> np.ndarray:
    """Stable positional order by symbol and observable timestamp when available."""

    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=int)
    if timestamp_col not in frame.columns:
        return np.arange(n, dtype=int)
    sorter = pd.DataFrame({"__pos__": np.arange(n, dtype=int)})
    sorter["__timestamp__"] = pd.to_datetime(
        frame[timestamp_col], utc=True, errors="coerce"
    ).to_numpy()
    sort_cols: list[str] = []
    if symbol_col in frame.columns:
        sorter["__symbol__"] = frame[symbol_col].to_numpy()
        sort_cols.append("__symbol__")
    sort_cols.extend(["__timestamp__", "__pos__"])
    return sorter.sort_values(sort_cols, kind="mergesort")["__pos__"].to_numpy(
        dtype=int
    )


def per_symbol_warmup_mask(
    frame: pd.DataFrame,
    *,
    warmup_rows: int = 0,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    context: PreparedFrameContext | None = None,
) -> np.ndarray:
    """Rows eligible after dropping the first ``warmup_rows`` per symbol."""

    n = len(frame)
    if n == 0:
        return np.zeros(0, dtype=bool)
    warmup = max(0, int(warmup_rows or 0))
    if warmup <= 0:
        return np.ones(n, dtype=bool)
    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    if ctx is not None:
        ordered_mask = np.zeros(n, dtype=bool)
        for start, end in zip(ctx.sorted_group_starts, ctx.sorted_group_ends):
            ordered_mask[start:end] = (
                np.arange(end - start, dtype=np.int64) >= warmup
            )
        mask = np.zeros(n, dtype=bool)
        mask[ctx.order] = ordered_mask
        return mask
    order = time_sort_order(frame, symbol_col=symbol_col, timestamp_col=timestamp_col)
    if symbol_col in frame.columns:
        ordered_symbols = frame.iloc[order][symbol_col]
        positions = ordered_symbols.groupby(ordered_symbols, sort=False).cumcount()
        ordered_mask = positions.to_numpy(dtype=int) >= warmup
    else:
        ordered_mask = np.arange(n, dtype=int) >= warmup
    mask = np.zeros(n, dtype=bool)
    mask[order] = ordered_mask
    return mask


def time_sorted_frame_and_features(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    context: PreparedFrameContext | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """Sort frame/features for causal rolling work and return original positions."""

    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    order = ctx.order if ctx is not None else time_sort_order(
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    if ctx is not None and ctx.order_is_identity:
        return frame, features, None
    if np.array_equal(order, np.arange(len(frame), dtype=int)):
        return frame, features, None
    sorted_frame = frame.iloc[order].reset_index(drop=True)
    sorted_features = features.iloc[order].reset_index(drop=True)
    return sorted_frame, sorted_features, order


def restore_time_sorted_frame(
    sorted_values: pd.DataFrame,
    *,
    original_index: pd.Index,
    order: np.ndarray | None,
) -> pd.DataFrame:
    """Restore a DataFrame computed in sorted position order to input row order."""

    if order is None:
        return sorted_values
    values = sorted_values.to_numpy(copy=True)
    restored = np.empty(values.shape, dtype=values.dtype)
    restored[order] = values
    return pd.DataFrame(restored, index=original_index, columns=sorted_values.columns)


def restore_time_sorted_series(
    sorted_values: pd.Series,
    *,
    original_index: pd.Index,
    order: np.ndarray | None,
) -> pd.Series:
    """Restore a Series computed in sorted position order to input row order."""

    if order is None:
        return sorted_values
    values = sorted_values.to_numpy(copy=True)
    restored = np.empty(values.shape, dtype=values.dtype)
    restored[order] = values
    return pd.Series(restored, index=original_index, name=sorted_values.name)


def low_quality_mask(
    values: np.ndarray,
    *,
    treat_zero_as_low_quality: bool = True,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(arr)
    if treat_zero_as_low_quality:
        bad |= arr == 0.0
    return bad


def robust_std(values: np.ndarray, *, eps: float = 1e-12) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    med = float(np.nanmedian(vals))
    mad = float(np.nanmedian(np.abs(vals - med)))
    scale = 1.4826 * mad
    if np.isfinite(scale) and scale > eps:
        return float(scale)
    q25, q75 = np.nanpercentile(vals, [25.0, 75.0])
    scale = float((q75 - q25) / 1.349)
    if np.isfinite(scale) and scale > eps:
        return float(scale)
    scale = float(np.nanstd(vals))
    if np.isfinite(scale) and scale > eps:
        return float(scale)
    return 0.0


def compute_quality_report(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    warmup_rows: int = 0,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    min_good_row_fraction: float = 0.90,
    treat_zero_as_low_quality: bool = True,
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Score feature row quality after the per-symbol warm-up period.

    A row is low quality when it is missing, non-finite, NaN, or zero when
    ``treat_zero_as_low_quality`` is enabled. Features are kept when at least
    ``min_good_row_fraction`` of post-warm-up rows are high quality.
    """

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    x = numeric_feature_frame(frame, features)
    mask = per_symbol_warmup_mask(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    rows: list[dict[str, object]] = []
    observed_rows = int(mask.sum())
    min_good = float(min_good_row_fraction)
    values = x.loc[mask, features].to_numpy(dtype=np.float32, copy=False)
    bad = ~np.isfinite(values)
    if treat_zero_as_low_quality:
        bad |= values == 0.0
    low_quality_counts = bad.sum(axis=0).astype(np.int64, copy=False)
    total_rows = int(values.shape[0])
    good_counts = total_rows - low_quality_counts
    for i, feature in enumerate(features):
        low_quality_rows = int(low_quality_counts[i])
        good_rows = int(good_counts[i])
        good_fraction = float(good_rows / total_rows) if total_rows else 0.0
        low_quality_fraction = (
            float(low_quality_rows / total_rows) if total_rows else 1.0
        )
        rows.append(
            {
                "feature": feature,
                "observed_rows": observed_rows,
                "good_rows": good_rows,
                "low_quality_rows": low_quality_rows,
                "good_fraction": good_fraction,
                "low_quality_fraction": low_quality_fraction,
                "keep": bool(good_fraction >= min_good),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "observed_rows",
                "good_rows",
                "low_quality_rows",
                "good_fraction",
                "low_quality_fraction",
                "keep",
            ]
        )
    out = pd.DataFrame(rows).set_index("feature")
    for col in ["observed_rows", "good_rows", "low_quality_rows"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int32)
    for col in ["good_fraction", "low_quality_fraction"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    out["keep"] = out["keep"].astype(bool)
    return out


def filter_quality_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    **kwargs: object,
) -> tuple[list[str], pd.DataFrame]:
    report = compute_quality_report(frame, feature_columns, **kwargs)
    kept = (
        report.index[report["keep"].to_numpy(dtype=bool, copy=False)]
        .astype(str)
        .tolist()
    )
    return kept, report


def _rank01(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.size, 0.5, dtype=np.float64)
    fill = float(np.nanmedian(arr[finite]))
    arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(arr.size, dtype=np.float64)
    ranks[order] = np.arange(arr.size, dtype=np.float64)
    if arr.size > 1:
        ranks /= float(arr.size - 1)
    else:
        ranks[:] = 1.0
    return ranks


def _evenly_spaced_positions(positions: np.ndarray, limit: int) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    if positions.size <= int(limit):
        return positions
    if int(limit) <= 0:
        return np.zeros(0, dtype=int)
    take = np.linspace(0, positions.size - 1, int(limit), dtype=int)
    return positions[take]


def _stratified_symbol_sample(
    frame: pd.DataFrame,
    positions: np.ndarray,
    *,
    max_rows: int,
    symbol_col: str,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=int)
    limit = int(max_rows)
    if positions.size <= limit:
        return positions
    if limit <= 0:
        return np.zeros(0, dtype=int)
    if symbol_col not in frame.columns:
        return _evenly_spaced_positions(positions, limit)
    symbols = frame[symbol_col].to_numpy()[positions]
    groups = [positions[symbols == symbol] for symbol in pd.unique(symbols)]
    groups = [group for group in groups if group.size]
    if not groups:
        return _evenly_spaced_positions(positions, limit)
    quota_base = max(1, limit // len(groups))
    selected: list[np.ndarray] = []
    for group in groups:
        selected.append(_evenly_spaced_positions(group, min(quota_base, group.size)))
    out = np.unique(np.concatenate(selected)) if selected else np.zeros(0, dtype=int)
    if out.size < limit:
        remaining = np.setdiff1d(positions, out, assume_unique=False)
        fill = _evenly_spaced_positions(remaining, limit - out.size)
        out = np.unique(np.concatenate([out, fill]))
    return out[:limit]


def stratified_period_sample_positions(
    frame: pd.DataFrame,
    positions: Sequence[int] | np.ndarray,
    *,
    max_rows: int | None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    n_periods: int = 24,
    context: PreparedFrameContext | None = None,
) -> np.ndarray:
    """Deterministically sample positions across time periods and symbols."""

    pos = np.asarray(positions, dtype=int)
    if pos.size == 0:
        return pos
    limit = int(max_rows or 0)
    if limit <= 0 or pos.size <= limit:
        return np.sort(pos)
    period_count = max(1, min(int(n_periods or 1), limit, pos.size))
    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    if ctx is not None and ctx.has_timestamp:
        valid = ctx.timestamp_valid
        valid_pos = pos[valid[pos]]
        if valid_pos.size:
            ts_ns = ctx.timestamp_ns
            order = np.argsort(ts_ns[valid_pos], kind="mergesort")
            sorted_pos = valid_pos[order]
        else:
            sorted_pos = np.sort(pos)
    elif timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        valid = ts.notna().to_numpy(dtype=bool)
        valid_pos = pos[valid[pos]]
        if valid_pos.size:
            ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
            order = np.argsort(ts_ns[valid_pos], kind="mergesort")
            sorted_pos = valid_pos[order]
        else:
            sorted_pos = np.sort(pos)
    else:
        sorted_pos = np.sort(pos)
    bins = [bucket for bucket in np.array_split(sorted_pos, period_count) if bucket.size]
    if not bins:
        return _evenly_spaced_positions(np.sort(pos), limit)
    base = max(1, limit // len(bins))
    extra = max(0, limit - base * len(bins))
    selected: list[np.ndarray] = []
    for i, bucket in enumerate(bins):
        quota = base + (1 if i < extra else 0)
        selected.append(
            _stratified_symbol_sample(
                frame,
                bucket,
                max_rows=min(quota, bucket.size),
                symbol_col=symbol_col,
            )
        )
    out = np.unique(np.concatenate(selected)) if selected else np.zeros(0, dtype=int)
    if out.size < limit:
        remaining = np.setdiff1d(sorted_pos, out, assume_unique=False)
        fill = _evenly_spaced_positions(remaining, limit - out.size)
        out = np.unique(np.concatenate([out, fill]))
    if ctx is not None and ctx.has_timestamp and out.size:
        ts_ns = ctx.timestamp_ns
        return out[np.argsort(ts_ns[out], kind="mergesort")][:limit]
    if timestamp_col in frame.columns and out.size:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
        return out[np.argsort(ts_ns[out], kind="mergesort")][:limit]
    return np.sort(out)[:limit]


def _eligible_positions(
    frame: pd.DataFrame,
    *,
    warmup_rows: int,
    symbol_col: str,
    timestamp_col: str,
    context: PreparedFrameContext | None = None,
) -> np.ndarray:
    mask = per_symbol_warmup_mask(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    if ctx is not None and ctx.has_timestamp:
        mask &= ctx.timestamp_valid
    elif timestamp_col in frame.columns:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        mask &= ts.notna().to_numpy(dtype=bool)
    return np.flatnonzero(mask)


def _make_blocks(
    frame: pd.DataFrame,
    *,
    warmup_rows: int,
    symbol_col: str,
    timestamp_col: str,
    block_hours: int = 24 * 7,
    n_blocks: int = 12,
    min_block_rows: int = 48,
    context: PreparedFrameContext | None = None,
) -> list[np.ndarray]:
    positions = _eligible_positions(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    if len(positions) == 0:
        return []
    min_rows = max(1, int(min_block_rows or 1))
    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    if ctx is not None and ctx.has_timestamp and int(block_hours or 0) > 0:
        ts_ns = ctx.timestamp_ns
        valid_ts = ctx.timestamp_valid
        positions = positions[valid_ts[positions]]
        if len(positions) == 0:
            return []
        start_ns = int(np.nanmin(ts_ns[positions]))
        block_ns = int(block_hours) * 3600 * 1_000_000_000
        ids = (ts_ns[positions] - start_ns) // max(block_ns, 1)
        blocks = [
            positions[ids == block_id]
            for block_id in np.unique(ids)
            if int(np.sum(ids == block_id)) >= min_rows
        ]
        if len(blocks) >= 2:
            return blocks
    elif timestamp_col in frame.columns and int(block_hours or 0) > 0:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
        valid_ts = ts.notna().to_numpy(dtype=bool)
        positions = positions[valid_ts[positions]]
        if len(positions) == 0:
            return []
        start_ns = int(np.nanmin(ts_ns[positions]))
        block_ns = int(block_hours) * 3600 * 1_000_000_000
        ids = (ts_ns[positions] - start_ns) // max(block_ns, 1)
        blocks = [
            positions[ids == block_id]
            for block_id in np.unique(ids)
            if int(np.sum(ids == block_id)) >= min_rows
        ]
        if len(blocks) >= 2:
            return blocks
    split_count = max(2, min(int(n_blocks or 12), len(positions) // min_rows))
    if split_count <= 1:
        return [positions] if len(positions) >= min_rows else []
    return [block for block in np.array_split(positions, split_count) if len(block) >= min_rows]


def _valid_values(
    values: np.ndarray,
    positions: np.ndarray,
    *,
    treat_zero_as_low_quality: bool,
) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)[positions]
    bad = low_quality_mask(vals, treat_zero_as_low_quality=treat_zero_as_low_quality)
    return vals[~bad]


def _ks_distance(sample: np.ndarray, reference: np.ndarray, *, max_sample: int) -> float:
    if sample.size < 2 or reference.size < 2:
        return np.nan
    left = sample
    right = reference
    limit = max(2, int(max_sample or 0))
    if limit and left.size > limit:
        left = left[np.linspace(0, left.size - 1, limit, dtype=int)]
    if limit and right.size > limit:
        right = right[np.linspace(0, right.size - 1, limit, dtype=int)]
    return float(ks_2samp(left, right, mode="asymp").statistic)


def _acf(values: np.ndarray, lag: int) -> tuple[float, int]:
    lag = max(1, int(lag or 1))
    vals = np.asarray(values, dtype=np.float64)
    if vals.size <= lag:
        return np.nan, 0
    left = vals[:-lag]
    right = vals[lag:]
    ok = np.isfinite(left) & np.isfinite(right)
    if int(ok.sum()) < 3:
        return np.nan, int(ok.sum())
    left = left[ok]
    right = right[ok]
    if float(np.nanstd(left)) <= 1e-12 or float(np.nanstd(right)) <= 1e-12:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(left, right)[0, 1]), int(ok.sum())


def _block_acf(
    frame: pd.DataFrame,
    values: np.ndarray,
    positions: np.ndarray,
    *,
    symbol_col: str,
    lag: int,
    context: PreparedFrameContext | None = None,
) -> float:
    ctx = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=context.timestamp_col if context is not None else "timestamp",
    )
    if ctx is not None and ctx.has_symbol:
        symbols = ctx.symbols[positions]
    elif symbol_col in frame.columns:
        symbols = frame[symbol_col].to_numpy()[positions]
    else:
        value, _count = _acf(values[positions], lag)
        return value
    acfs: list[float] = []
    weights: list[int] = []
    for symbol in pd.unique(symbols):
        symbol_positions = positions[symbols == symbol]
        value, count = _acf(values[symbol_positions], lag)
        if np.isfinite(value) and count > 0:
            acfs.append(value)
            weights.append(count)
    if not acfs:
        return np.nan
    return float(np.average(np.asarray(acfs), weights=np.asarray(weights)))


def compute_feature_diagnostics(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    quality_report: pd.DataFrame | None = None,
    warmup_rows: int = 0,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    block_hours: int = 24 * 7,
    n_blocks: int = 12,
    min_block_rows: int = 48,
    autocorr_lag: int = 1,
    treat_zero_as_low_quality: bool = True,
    max_ks_sample: int = 5000,
    eps: float = 1e-12,
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Compute cheap dynamics diagnostics for primitive or derived features."""

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    original_context = _valid_context(
        context,
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    x = numeric_feature_frame(frame, features)
    frame, x, _order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=original_context,
    )
    sorted_context = _sorted_context(original_context)
    quality = quality_report
    if quality is None:
        quality = compute_quality_report(
            frame,
            features,
            warmup_rows=warmup_rows,
            symbol_col=symbol_col,
            timestamp_col=timestamp_col,
            treat_zero_as_low_quality=treat_zero_as_low_quality,
            context=sorted_context,
        )
    eligible = _eligible_positions(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=sorted_context,
    )
    blocks = _make_blocks(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        block_hours=block_hours,
        n_blocks=n_blocks,
        min_block_rows=min_block_rows,
        context=sorted_context,
    )
    rows: list[dict[str, object]] = []
    for feature in features:
        values = x[feature].to_numpy(dtype=np.float64, copy=False)
        reference = _valid_values(
            values,
            eligible,
            treat_zero_as_low_quality=treat_zero_as_low_quality,
        )
        scale = robust_std(reference, eps=eps)
        block_means: list[float] = []
        block_log_vars: list[float] = []
        block_quantiles: list[np.ndarray] = []
        block_ks: list[float] = []
        block_acfs: list[float] = []
        block_tail_widths: list[float] = []
        for block_positions in blocks:
            block_values = _valid_values(
                values,
                block_positions,
                treat_zero_as_low_quality=treat_zero_as_low_quality,
            )
            if block_values.size < 3:
                continue
            block_means.append(float(np.nanmean(block_values)))
            block_log_vars.append(float(np.log(float(np.nanvar(block_values)) + eps)))
            q = np.nanpercentile(block_values, [5.0, 25.0, 50.0, 75.0, 95.0])
            block_quantiles.append(np.asarray(q, dtype=np.float64))
            block_tail_widths.append(float(q[-1] - q[0]))
            block_ks.append(
                _ks_distance(block_values, reference, max_sample=max_ks_sample)
            )
            block_acfs.append(
                _block_acf(
                    frame,
                    values,
                    block_positions,
                    symbol_col=symbol_col,
                    lag=autocorr_lag,
                    context=sorted_context,
                )
            )
        quantile_instability = np.nan
        if len(block_quantiles) >= 2:
            distances: list[float] = []
            q_arr = np.vstack(block_quantiles)
            for i in range(len(q_arr)):
                for j in range(i + 1, len(q_arr)):
                    distances.append(
                        float(np.sqrt(np.nanmean((q_arr[i] - q_arr[j]) ** 2)))
                    )
            if distances:
                denom = scale if scale > eps else 1.0
                quantile_instability = float(np.nanmean(distances) / denom)
        tail_widths = np.asarray(block_tail_widths, dtype=np.float64)
        median_tail = float(np.nanmedian(tail_widths)) if tail_widths.size else np.nan
        tail_change = (
            float(np.nanstd(tail_widths) / max(abs(median_tail), eps))
            if np.isfinite(median_tail)
            else np.nan
        )
        denom = scale if scale > eps else 1.0
        rows.append(
            {
                "feature": feature,
                "observations": int(reference.size),
                "blocks_used": int(len(block_quantiles)),
                "robust_std": float(scale),
                "mean_instability": float(np.nanstd(block_means) / denom)
                if block_means
                else np.nan,
                "variance_instability": float(np.nanstd(block_log_vars))
                if block_log_vars
                else np.nan,
                "quantile_instability": quantile_instability,
                "distribution_shift": float(np.nanmean(block_ks))
                if block_ks
                else np.nan,
                "autocorr_change": float(np.nanstd(block_acfs))
                if block_acfs
                else np.nan,
                "tail_change": tail_change,
                "good_fraction": float(
                    quality.loc[feature, "good_fraction"]
                    if feature in quality.index
                    else np.nan
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).set_index("feature")
    for component in DYNAMICS_COMPONENT_WEIGHTS:
        out[f"{component}_norm"] = _rank01(out[component].to_numpy(dtype=np.float64))
    score = np.zeros(len(out), dtype=np.float64)
    for component, weight in DYNAMICS_COMPONENT_WEIGHTS.items():
        score += float(weight) * out[f"{component}_norm"].to_numpy(dtype=np.float64)
    out["dynamics_score"] = score
    out["quality_dynamics_score"] = (
        out["good_fraction"].fillna(0.0).clip(0.0, 1.0) * out["dynamics_score"]
    )
    for col in ["observations", "blocks_used"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.int32)
    for col in out.columns.difference(["observations", "blocks_used"]):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    return out.sort_values("dynamics_score", ascending=False)


def _rank_tie_average(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    out = np.full(arr.size, np.nan, dtype=np.float32)
    finite = np.isfinite(arr)
    count = int(finite.sum())
    if count <= 1:
        return out
    finite_values = arr[finite]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_values = finite_values[order]
    ranks = np.empty(count, dtype=np.float32)
    start = 0
    while start < count:
        end = start + 1
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        ranks[order[start:end]] = avg_rank
        start = end
    ranks /= float(count - 1)
    out[np.flatnonzero(finite)] = ranks
    return out


def _standardized_spearman_ranks(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    warmup_rows: int,
    symbol_col: str,
    timestamp_col: str,
    max_rows: int | None,
    time_bins: int,
    context: PreparedFrameContext | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    x = numeric_feature_frame(frame, features)
    positions = _eligible_positions(
        frame,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    positions = stratified_period_sample_positions(
        frame,
        positions,
        max_rows=max_rows,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        n_periods=time_bins,
        context=context,
    )
    values = x.iloc[positions][features].to_numpy(dtype=np.float32, copy=False)
    ranks = np.empty(values.shape, dtype=np.float32)
    for col_idx in range(values.shape[1]):
        ranks[:, col_idx] = _rank_tie_average(values[:, col_idx])
    means = np.nanmean(ranks, axis=0)
    means = np.where(np.isfinite(means), means, 0.5).astype(np.float32)
    centered = ranks - means.reshape(1, -1)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.sqrt(np.sum(centered * centered, axis=0)).astype(np.float32)
    ok = norms > 1e-12
    centered[:, ok] /= norms[ok].reshape(1, -1)
    centered[:, ~ok] = 0.0
    return centered.astype(np.float32, copy=False), {
        feature: i for i, feature in enumerate(features)
    }


def _greedy_prune_rank_matrix(
    ranked_features: Sequence[str],
    rank_matrix: np.ndarray,
    feature_pos: Mapping[str, int],
    *,
    threshold: float,
    max_features: int | None,
) -> list[str]:
    selected: list[str] = []
    selected_pos: list[int] = []
    for feature in ranked_features:
        if feature not in feature_pos:
            continue
        pos = int(feature_pos[feature])
        if selected_pos:
            corr = np.abs(rank_matrix[:, selected_pos].T @ rank_matrix[:, pos])
            max_corr = float(np.max(corr)) if corr.size else 0.0
            if max_corr >= float(threshold):
                continue
        selected.append(str(feature))
        selected_pos.append(pos)
        if max_features is not None and len(selected) >= int(max_features):
            break
    return selected


def select_representatives_by_spearman(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    scores: Mapping[str, float] | pd.Series,
    *,
    target_features: int = 100,
    initial_threshold: float = 0.96,
    threshold_step: float = 0.005,
    max_threshold: float = 0.999,
    warmup_rows: int = 0,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    max_corr_rows: int | None = None,
    corr_time_bins: int = 24,
    max_candidates: int | None = None,
    context: PreparedFrameContext | None = None,
) -> tuple[list[str], float]:
    """Greedy representative selection with a widening Spearman threshold."""

    candidates = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    if not candidates:
        return [], float(initial_threshold)
    score_series = pd.Series(scores, dtype=float).reindex(candidates).fillna(0.0)
    ranked = [
        str(idx)
        for idx in score_series.sort_values(ascending=False, kind="mergesort").index
    ]
    if max_candidates is not None and int(max_candidates) > 0:
        ranked = ranked[: int(max_candidates)]
    target = max(1, int(target_features or len(ranked)))
    if len(ranked) <= target:
        return ranked, float(initial_threshold)
    rank_matrix, feature_pos = _standardized_spearman_ranks(
        frame,
        ranked,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        max_rows=max_corr_rows,
        time_bins=corr_time_bins,
        context=context,
    )
    best: list[str] = []
    best_threshold = float(initial_threshold)
    threshold = float(initial_threshold)
    while threshold <= float(max_threshold) + 1e-12:
        selected = _greedy_prune_rank_matrix(
            ranked,
            rank_matrix,
            feature_pos,
            threshold=threshold,
            max_features=target,
        )
        if len(selected) > len(best):
            best = selected
            best_threshold = threshold
        if len(selected) >= target:
            return selected[:target], float(threshold)
        next_threshold = min(float(max_threshold), threshold + float(threshold_step))
        if next_threshold <= threshold:
            break
        threshold = next_threshold
    if not best:
        best = ranked[:target]
    return best[:target], float(best_threshold)


def select_primitive_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    target_features: int = 100,
    min_good_row_fraction: float = 0.90,
    warmup_rows: int = 0,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    initial_spearman_threshold: float = 0.96,
    threshold_step: float = 0.005,
    max_spearman_threshold: float = 0.999,
    block_hours: int = 24 * 7,
    min_block_rows: int = 48,
    autocorr_lag: int = 1,
    treat_zero_as_low_quality: bool = True,
    spearman_max_corr_rows: int | None = None,
    spearman_corr_time_bins: int = 24,
    spearman_max_candidates: int | None = None,
    context: PreparedFrameContext | None = None,
) -> FeatureSelectionResult:
    """Apply quality filtering, dynamics scoring, and primitive pruning."""

    candidates, quality = filter_quality_features(
        frame,
        feature_columns,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        min_good_row_fraction=min_good_row_fraction,
        treat_zero_as_low_quality=treat_zero_as_low_quality,
        context=context,
    )
    diagnostics = compute_feature_diagnostics(
        frame,
        candidates,
        quality_report=quality,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        block_hours=block_hours,
        min_block_rows=min_block_rows,
        autocorr_lag=autocorr_lag,
        treat_zero_as_low_quality=treat_zero_as_low_quality,
        context=context,
    )
    selected, threshold = select_representatives_by_spearman(
        frame,
        diagnostics.index.tolist(),
        diagnostics["dynamics_score"] if not diagnostics.empty else {},
        target_features=target_features,
        initial_threshold=initial_spearman_threshold,
        threshold_step=threshold_step,
        max_threshold=max_spearman_threshold,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        max_corr_rows=spearman_max_corr_rows,
        corr_time_bins=spearman_corr_time_bins,
        max_candidates=spearman_max_candidates,
        context=context,
    )
    return FeatureSelectionResult(
        selected_features=selected,
        diagnostics=diagnostics,
        quality_report=quality,
        spearman_threshold=threshold,
        candidate_features=candidates,
    )
