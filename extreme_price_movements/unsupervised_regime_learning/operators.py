"""Derived operators for unsupervised regime learning."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLasso
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import NearestNeighbors

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is unavailable
    njit = None
    _NUMBA_AVAILABLE = False

from extreme_price_movements.unsupervised_regime_learning.diagnostics import (
    PreparedFrameContext,
    compute_quality_report,
    numeric_feature_frame,
    restore_time_sorted_frame,
    stratified_period_sample_positions,
    time_sorted_frame_and_features,
    time_sort_order,
)


@dataclass(frozen=True)
class PairCandidateScore:
    feature_i: str
    feature_j: str
    pair_score: float
    rho_variation: float
    rho_persistence: float
    reliability: float
    graph_edge_stability: float
    graph_edge_strength: float
    sparse_graph_score: float
    mechanism_i: str
    mechanism_j: str


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
    )


if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _insert_sorted_value_numba(
        sorted_values: np.ndarray,
        count: int,
        value: float,
    ) -> int:
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_values[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        for i in range(count, lo, -1):
            sorted_values[i] = sorted_values[i - 1]
        sorted_values[lo] = value
        return count + 1


    @njit(cache=True)
    def _remove_sorted_value_numba(
        sorted_values: np.ndarray,
        count: int,
        value: float,
    ) -> int:
        remove_at = -1
        for i in range(count):
            if sorted_values[i] == value:
                remove_at = i
                break
        if remove_at < 0:
            return count
        for i in range(remove_at, count - 1):
            sorted_values[i] = sorted_values[i + 1]
        return count - 1


    @njit(cache=True)
    def _upper_bound_sorted_value_numba(
        sorted_values: np.ndarray,
        count: int,
        value: float,
    ) -> int:
        lo = 0
        hi = count
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_values[mid] <= value:
                lo = mid + 1
            else:
                hi = mid
        return lo


    @njit(cache=True)
    def _sorted_quantile_numba(
        sorted_values: np.ndarray,
        count: int,
        q: float,
    ) -> float:
        if count <= 0:
            return np.nan
        if count == 1:
            return sorted_values[0]
        pos = q * (count - 1)
        lo = int(np.floor(pos))
        hi = lo + 1
        if hi >= count:
            return sorted_values[lo]
        weight = pos - lo
        return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


    @njit(cache=True)
    def _rolling_quantiles_1d_numba(
        values: np.ndarray,
        window: int,
        min_periods: int,
    ) -> np.ndarray:
        n = values.shape[0]
        out = np.empty((n, 6), dtype=np.float32)
        for i in range(n):
            for j in range(6):
                out[i, j] = np.nan
        if window <= 0:
            return out
        sorted_values = np.empty(window, dtype=np.float64)
        count = 0
        for i in range(n):
            stale_idx = i - window
            if stale_idx >= 0:
                stale = values[stale_idx]
                if np.isfinite(stale):
                    count = _remove_sorted_value_numba(sorted_values, count, stale)
            current = values[i]
            if np.isfinite(current):
                count = _insert_sorted_value_numba(sorted_values, count, current)
            if count >= min_periods:
                out[i, 0] = _sorted_quantile_numba(sorted_values, count, 0.05)
                out[i, 1] = _sorted_quantile_numba(sorted_values, count, 0.25)
                out[i, 2] = _sorted_quantile_numba(sorted_values, count, 0.50)
                out[i, 3] = _sorted_quantile_numba(sorted_values, count, 0.75)
                out[i, 4] = _sorted_quantile_numba(sorted_values, count, 0.95)
                if np.isfinite(current):
                    rank = _upper_bound_sorted_value_numba(
                        sorted_values,
                        count,
                        current,
                    )
                    out[i, 5] = rank / count
        return out


    @njit(cache=True)
    def _rolling_autocorr_1d_numba(
        values: np.ndarray,
        window: int,
        min_periods: int,
        lag: int,
    ) -> np.ndarray:
        n = values.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(n):
            out[i] = np.nan
            start = i - window + 1
            if start < 0:
                start = 0
            count = 0
            sx = 0.0
            sy = 0.0
            sx2 = 0.0
            sy2 = 0.0
            sxy = 0.0
            first = start + lag
            if first > i:
                continue
            for j in range(first, i + 1):
                x = values[j - lag]
                y = values[j]
                if np.isfinite(x) and np.isfinite(y):
                    count += 1
                    sx += x
                    sy += y
                    sx2 += x * x
                    sy2 += y * y
                    sxy += x * y
            if count >= min_periods:
                vx = count * sx2 - sx * sx
                vy = count * sy2 - sy * sy
                denom = np.sqrt(max(vx * vy, 0.0))
                if denom > 1e-12:
                    out[i] = (count * sxy - sx * sy) / denom
        return out


    @njit(cache=True)
    def _rolling_pair_values_1d_numba(
        left: np.ndarray,
        right: np.ndarray,
        window: int,
        min_periods: int,
    ) -> np.ndarray:
        n = left.shape[0]
        out = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            out[i, 0] = np.nan
            out[i, 1] = np.nan
        if window <= 0:
            return out
        count = 0
        sx = 0.0
        sy = 0.0
        sx2 = 0.0
        sy2 = 0.0
        sxy = 0.0
        for i in range(n):
            stale_idx = i - window
            if stale_idx >= 0:
                stale_x = left[stale_idx]
                stale_y = right[stale_idx]
                if np.isfinite(stale_x) and np.isfinite(stale_y):
                    count -= 1
                    sx -= stale_x
                    sy -= stale_y
                    sx2 -= stale_x * stale_x
                    sy2 -= stale_y * stale_y
                    sxy -= stale_x * stale_y
            x = left[i]
            y = right[i]
            if np.isfinite(x) and np.isfinite(y):
                count += 1
                sx += x
                sy += y
                sx2 += x * x
                sy2 += y * y
                sxy += x * y
            if count >= min_periods and count > 1:
                cov_num = sxy - (sx * sy / count)
                out[i, 0] = cov_num / (count - 1)
                vx = sx2 - (sx * sx / count)
                vy = sy2 - (sy * sy / count)
                denom = np.sqrt(max(vx * vy, 0.0))
                if denom > 1e-12:
                    out[i, 1] = cov_num / denom
        return out


    @njit(cache=True)
    def _rolling_corr_pair_summaries_numba(
        arr: np.ndarray,
        left_idx: np.ndarray,
        right_idx: np.ndarray,
        group_starts: np.ndarray,
        group_ends: np.ndarray,
        window: int,
        min_periods: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_pairs = left_idx.shape[0]
        n_rows = arr.shape[0]
        variation = np.zeros(n_pairs, dtype=np.float64)
        persistence = np.zeros(n_pairs, dtype=np.float64)
        finite_fraction = np.zeros(n_pairs, dtype=np.float64)
        for p in range(n_pairs):
            li = left_idx[p]
            ri = right_idx[p]
            finite_count = 0
            sum_r = 0.0
            sum_r2 = 0.0
            sum_abs = 0.0
            for g in range(group_starts.shape[0]):
                start = group_starts[g]
                end = group_ends[g]
                count = 0
                sx = 0.0
                sy = 0.0
                sx2 = 0.0
                sy2 = 0.0
                sxy = 0.0
                for i in range(start, end):
                    stale_idx = i - window
                    if stale_idx >= start:
                        stale_x = arr[stale_idx, li]
                        stale_y = arr[stale_idx, ri]
                        if np.isfinite(stale_x) and np.isfinite(stale_y):
                            count -= 1
                            sx -= stale_x
                            sy -= stale_y
                            sx2 -= stale_x * stale_x
                            sy2 -= stale_y * stale_y
                            sxy -= stale_x * stale_y
                    x = arr[i, li]
                    y = arr[i, ri]
                    if np.isfinite(x) and np.isfinite(y):
                        count += 1
                        sx += x
                        sy += y
                        sx2 += x * x
                        sy2 += y * y
                        sxy += x * y
                    if count >= min_periods and count > 1:
                        vx = count * sx2 - sx * sx
                        vy = count * sy2 - sy * sy
                        denom = np.sqrt(max(vx * vy, 0.0))
                        if denom > 1e-12:
                            r = (count * sxy - sx * sy) / denom
                            finite_count += 1
                            sum_r += r
                            sum_r2 += r * r
                            if r >= 0.0:
                                sum_abs += r
                            else:
                                sum_abs -= r
            if finite_count > 0:
                mean_r = sum_r / finite_count
                variance = max((sum_r2 / finite_count) - mean_r * mean_r, 0.0)
                variation[p] = np.sqrt(variance)
                persistence[p] = sum_abs / finite_count
                finite_fraction[p] = finite_count / max(n_rows, 1)
        return variation, persistence, finite_fraction


def _rolling_by_symbol(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
    fn: str,
    quantile: float | None = None,
) -> pd.Series:
    if symbol_col not in frame.columns:
        roller = series.rolling(window=window, min_periods=min_periods)
        if fn == "quantile":
            return roller.quantile(float(quantile))
        if fn == "median":
            return roller.median()
        return roller.mean()
    grouped = series.groupby(frame[symbol_col], sort=False)
    if fn == "quantile":
        return grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).quantile(
                float(quantile)
            )
        )
    if fn == "median":
        return grouped.transform(
            lambda s: s.rolling(window=window, min_periods=min_periods).median()
        )
    return grouped.transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).mean()
    )


def _rolling_apply_by_symbol(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
    func,
) -> pd.Series:
    if symbol_col not in frame.columns:
        return series.rolling(window=window, min_periods=min_periods).apply(
            func, raw=True
        )
    return series.groupby(frame[symbol_col], sort=False).transform(
        lambda s: s.rolling(window=window, min_periods=min_periods).apply(
            func, raw=True
        )
    )


def _group_bounds(frame: pd.DataFrame, *, symbol_col: str) -> tuple[np.ndarray, np.ndarray]:
    if symbol_col not in frame.columns:
        return np.array([0], dtype=np.int64), np.array([len(frame)], dtype=np.int64)
    starts: list[int] = []
    ends: list[int] = []
    for positions in frame.groupby(symbol_col, sort=False).indices.values():
        pos = np.asarray(positions, dtype=np.int64)
        if pos.size == 0:
            continue
        starts.append(int(pos.min()))
        ends.append(int(pos.max()) + 1)
    return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _group_bounds_from_context_or_frame(
    frame: pd.DataFrame,
    *,
    symbol_col: str,
    context: PreparedFrameContext | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        context is not None
        and context.n_rows == len(frame)
        and context.symbol_col == str(symbol_col)
        and context.order_is_identity
    ):
        return context.sorted_group_starts, context.sorted_group_ends
    return _group_bounds(frame, symbol_col=symbol_col)


def _sorted_operator_context(
    context: PreparedFrameContext | None,
    n_rows: int,
) -> PreparedFrameContext | None:
    if context is None or context.n_rows != int(n_rows):
        return None
    order = np.arange(context.n_rows, dtype=np.int64)
    return PreparedFrameContext(
        n_rows=context.n_rows,
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


def _rolling_quantiles_by_symbol(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
    group_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    if not _NUMBA_AVAILABLE:
        quantiles = [
            _rolling_by_symbol(
                frame,
                series,
                symbol_col=symbol_col,
                window=window,
                min_periods=min_periods,
                fn="quantile",
                quantile=q,
            ).to_numpy(dtype=np.float32, copy=False)
            for q in (0.05, 0.25, 0.50, 0.75, 0.95)
        ]
        quantiles.append(
            _rolling_apply_by_symbol(
                frame,
                series,
                symbol_col=symbol_col,
                window=window,
                min_periods=min_periods,
                func=_percentile_rank_current,
            ).to_numpy(dtype=np.float32, copy=False)
        )
        return np.column_stack(quantiles).astype(np.float32, copy=False)
    out = np.full((len(series), 6), np.nan, dtype=np.float32)
    values = series.to_numpy(dtype=np.float32, copy=False)
    starts, ends = (
        group_bounds
        if group_bounds is not None
        else _group_bounds(frame, symbol_col=symbol_col)
    )
    for start, end in zip(starts, ends):
        out[start:end] = _rolling_quantiles_1d_numba(
            values[start:end],
            int(window),
            int(min_periods),
        )
    return out


def _rolling_autocorr_by_symbol(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
    lag: int,
    group_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> pd.Series:
    if not _NUMBA_AVAILABLE:
        return _rolling_apply_by_symbol(
            frame,
            series,
            symbol_col=symbol_col,
            window=window,
            min_periods=min_periods,
            func=lambda values, lag=lag: _rolling_autocorr(values, lag),
        )
    out = np.full(len(series), np.nan, dtype=np.float32)
    values = series.to_numpy(dtype=np.float32, copy=False)
    starts, ends = (
        group_bounds
        if group_bounds is not None
        else _group_bounds(frame, symbol_col=symbol_col)
    )
    for start, end in zip(starts, ends):
        out[start:end] = _rolling_autocorr_1d_numba(
            values[start:end],
            int(window),
            int(min_periods),
            int(lag),
        )
    return pd.Series(out, index=frame.index, dtype=np.float32)


def _rolling_pair_by_symbol(
    frame: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
    method: str,
) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if symbol_col not in frame.columns:
        if method == "corr":
            return left.rolling(window=window, min_periods=min_periods).corr(right)
        return left.rolling(window=window, min_periods=min_periods).cov(right)
    for _, positions in frame.groupby(symbol_col, sort=False).indices.items():
        l = left.iloc[positions]
        r = right.iloc[positions]
        if method == "corr":
            values = l.rolling(window=window, min_periods=min_periods).corr(r)
        else:
            values = l.rolling(window=window, min_periods=min_periods).cov(r)
        out.iloc[positions] = values.to_numpy(dtype=float)
    return out


def _rolling_pair_values_by_symbol(
    frame: pd.DataFrame,
    left: pd.Series,
    right: pd.Series,
    *,
    symbol_col: str,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = _group_bounds(frame, symbol_col=symbol_col)
    return _rolling_pair_values_from_arrays(
        left.to_numpy(dtype=np.float32, copy=False),
        right.to_numpy(dtype=np.float32, copy=False),
        starts,
        ends,
        window=window,
        min_periods=min_periods,
    )


def _rolling_pair_values_from_arrays(
    left_values: np.ndarray,
    right_values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    window: int,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left_values, dtype=np.float32)
    right_values = np.asarray(right_values, dtype=np.float32)
    n = int(left_values.shape[0])
    cov = np.full(n, np.nan, dtype=np.float32)
    corr = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return cov, corr
    if _NUMBA_AVAILABLE:
        for start, end in zip(starts, ends):
            values = _rolling_pair_values_1d_numba(
                left_values[start:end],
                right_values[start:end],
                int(window),
                int(min_periods),
            )
            cov[start:end] = values[:, 0]
            corr[start:end] = values[:, 1]
        return cov, corr
    win = max(1, int(window or 1))
    minp = max(1, int(min_periods or 1))
    for start, end in zip(starts, ends):
        left_block = np.asarray(left_values[start:end], dtype=np.float64)
        right_block = np.asarray(right_values[start:end], dtype=np.float64)
        m = int(left_block.size)
        if m == 0:
            continue
        finite = np.isfinite(left_block) & np.isfinite(right_block)
        x = np.where(finite, left_block, 0.0)
        y = np.where(finite, right_block, 0.0)
        count = np.concatenate([[0.0], np.cumsum(finite.astype(np.float64))])
        sx = np.concatenate([[0.0], np.cumsum(x)])
        sy = np.concatenate([[0.0], np.cumsum(y)])
        sx2 = np.concatenate([[0.0], np.cumsum(x * x)])
        sy2 = np.concatenate([[0.0], np.cumsum(y * y)])
        sxy = np.concatenate([[0.0], np.cumsum(x * y)])
        right_idx = np.arange(1, m + 1)
        left_idx = np.maximum(0, right_idx - win)
        cnt = count[right_idx] - count[left_idx]
        wx = sx[right_idx] - sx[left_idx]
        wy = sy[right_idx] - sy[left_idx]
        wx2 = sx2[right_idx] - sx2[left_idx]
        wy2 = sy2[right_idx] - sy2[left_idx]
        wxy = sxy[right_idx] - sxy[left_idx]
        valid = (cnt >= float(minp)) & (cnt > 1.0)
        if not bool(np.any(valid)):
            continue
        cov_num = wxy - (wx * wy / np.maximum(cnt, 1.0))
        cov_block = np.full(m, np.nan, dtype=np.float32)
        corr_block = np.full(m, np.nan, dtype=np.float32)
        cov_block[valid] = (cov_num[valid] / (cnt[valid] - 1.0)).astype(
            np.float32
        )
        vx = wx2 - (wx * wx / np.maximum(cnt, 1.0))
        vy = wy2 - (wy * wy / np.maximum(cnt, 1.0))
        denom = np.sqrt(np.maximum(vx * vy, 0.0))
        corr_valid = valid & (denom > 1e-12)
        corr_block[corr_valid] = (
            cov_num[corr_valid] / denom[corr_valid]
        ).astype(np.float32)
        cov[start:end] = cov_block
        corr[start:end] = corr_block
    return cov, corr


def _time_blocks(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    block_hours: int,
    min_block_rows: int,
) -> list[np.ndarray]:
    n = len(frame)
    if n == 0:
        return []
    min_rows = max(3, int(min_block_rows or 3))
    if timestamp_col in frame.columns and int(block_hours or 0) > 0:
        ts = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        valid = ts.notna().to_numpy(dtype=bool)
        if valid.any():
            ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
            start_ns = int(np.nanmin(ts_ns[valid]))
            block_ns = max(1, int(block_hours) * 3600 * 1_000_000_000)
            block_ids = (ts_ns - start_ns) // block_ns
            return [
                np.flatnonzero(valid & (block_ids == block_id))
                for block_id in np.unique(block_ids[valid])
                if int(np.sum(valid & (block_ids == block_id))) >= min_rows
            ]
    block_count = max(1, n // min_rows)
    return [block for block in np.array_split(np.arange(n), block_count) if len(block) >= min_rows]


def _standardized_block_matrix(block: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = list(block.columns)
    arr = block.to_numpy(dtype=np.float64, copy=True)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float32), []
    med = np.nanmedian(arr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(med, np.where(missing)[1])
    q25 = np.nanpercentile(arr, 25.0, axis=0)
    q75 = np.nanpercentile(arr, 75.0, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    keep = np.isfinite(scale) & (scale > 1e-8)
    if int(keep.sum()) < 2:
        return np.empty((len(arr), 0), dtype=np.float32), []
    arr = arr[:, keep]
    scale = scale[keep]
    med = med[keep]
    kept_cols = [col for col, ok in zip(cols, keep) if bool(ok)]
    arr = (arr - med.reshape(1, -1)) / scale.reshape(1, -1)
    return np.clip(arr, -8.0, 8.0).astype(np.float32), kept_cols


def sparse_dependency_graph_edge_scores(
    frame: pd.DataFrame,
    x: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    timestamp_col: str = "timestamp",
    block_hours: int = 24 * 7,
    min_block_rows: int = 48,
    alpha: float = 0.05,
    partial_corr_threshold: float = 1e-4,
    max_iter: int = 100,
) -> dict[tuple[str, str], dict[str, float]]:
    """Estimate stable sparse precision-graph edges across timestamp blocks."""

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c) in x.columns))
    if len(features) < 2:
        return {}
    edge_hits: dict[tuple[str, str], int] = {}
    edge_strength_sums: dict[tuple[str, str], float] = {}
    edge_strength_counts: dict[tuple[str, str], int] = {}
    blocks_used = 0
    for positions in _time_blocks(
        frame,
        timestamp_col=timestamp_col,
        block_hours=block_hours,
        min_block_rows=min_block_rows,
    ):
        arr, cols = _standardized_block_matrix(x.iloc[positions][features])
        if arr.shape[1] < 2 or arr.shape[0] < max(3, min_block_rows):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = GraphicalLasso(
                    alpha=max(float(alpha), 1e-6),
                    max_iter=max(10, int(max_iter)),
                    assume_centered=False,
                )
                model.fit(arr)
        except Exception:
            continue
        precision = np.asarray(model.precision_, dtype=np.float64)
        diag = np.diag(precision)
        denom = np.sqrt(np.maximum(diag[:, None] * diag[None, :], 1e-12))
        partial = -precision / denom
        np.fill_diagonal(partial, 0.0)
        blocks_used += 1
        tri_i, tri_j = np.triu_indices(len(cols), k=1)
        strengths = np.abs(partial[tri_i, tri_j])
        hits = strengths >= float(partial_corr_threshold)
        for idx, strength in enumerate(strengths):
            left = str(cols[int(tri_i[idx])])
            right = str(cols[int(tri_j[idx])])
            if right < left:
                left, right = right, left
            key = (left, right)
            edge_strength_sums[key] = edge_strength_sums.get(key, 0.0) + float(
                strength
            )
            edge_strength_counts[key] = edge_strength_counts.get(key, 0) + 1
            if bool(hits[idx]):
                edge_hits[key] = edge_hits.get(key, 0) + 1
    if blocks_used <= 0:
        return {}
    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, strength_sum in edge_strength_sums.items():
        stability = float(edge_hits.get(key, 0) / blocks_used)
        strength_mean = float(
            strength_sum / max(edge_strength_counts.get(key, 0), 1)
        )
        out[key] = {
            "graph_edge_stability": stability,
            "graph_edge_strength": strength_mean,
            "sparse_graph_score": stability * strength_mean,
            "sparse_graph_blocks": float(blocks_used),
        }
    return out


def _percentile_rank_current(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64)
    current = vals[-1] if vals.size else np.nan
    vals = vals[np.isfinite(vals)]
    if vals.size == 0 or not np.isfinite(current):
        return np.nan
    return float(np.mean(vals <= current))


def generate_quantile_operator_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window: int = 168,
    min_periods: int | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Rolling IQR, tail-width, tail-asymmetry, and percentile-rank operators."""

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    x = numeric_feature_frame(frame, features)
    original_index = frame.index
    frame, x, order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    sorted_context = _sorted_operator_context(context, len(frame))
    win = max(2, int(window or 2))
    minp = max(2, int(min_periods or max(3, win // 4)))
    columns: dict[str, object] = {}
    group_bounds = _group_bounds_from_context_or_frame(
        frame,
        symbol_col=symbol_col,
        context=sorted_context,
    )
    for feature in features:
        series = x[feature]
        quantiles = _rolling_quantiles_by_symbol(
            frame,
            series,
            symbol_col=symbol_col,
            window=win,
            min_periods=minp,
            group_bounds=group_bounds,
        )
        q05 = quantiles[:, 0]
        q25 = quantiles[:, 1]
        q50 = quantiles[:, 2]
        q75 = quantiles[:, 3]
        q95 = quantiles[:, 4]
        percentile_rank = quantiles[:, 5]
        name = _safe_name(feature)
        columns[f"q_iqr__{name}"] = q75 - q25
        columns[f"q_tail_width__{name}"] = q95 - q05
        columns[f"q_upper_tail__{name}"] = q95 - q50
        columns[f"q_lower_tail__{name}"] = q50 - q05
        columns[f"q_tail_asym__{name}"] = (q95 - q50) - (q50 - q05)
        columns[f"q_percentile_rank__{name}"] = percentile_rank
    out = pd.DataFrame(columns, index=frame.index, dtype=np.float32)
    return restore_time_sorted_frame(
        out,
        original_index=original_index,
        order=order,
    )


def _rolling_autocorr(values: np.ndarray, lag: int) -> float:
    vals = np.asarray(values, dtype=np.float64)
    lag = max(1, int(lag or 1))
    if vals.size <= lag:
        return np.nan
    left = vals[:-lag]
    right = vals[lag:]
    ok = np.isfinite(left) & np.isfinite(right)
    if int(ok.sum()) < 3:
        return np.nan
    left = left[ok]
    right = right[ok]
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _is_rolling_feature(name: str) -> bool:
    lower = str(name).lower()
    return (
        lower.startswith(("q_", "cov_", "corr_", "eig_", "svd"))
        or "rolling" in lower
        or "autocorr" in lower
    )


def generate_autocorr_operator_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    window: int = 168,
    lag: int = 1,
    min_periods: int | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Rolling autocorrelation operators, excluding already rolling features."""

    features = [
        str(c)
        for c in dict.fromkeys(str(c) for c in feature_columns if str(c))
        if not _is_rolling_feature(str(c))
    ]
    x = numeric_feature_frame(frame, features)
    original_index = frame.index
    frame, x, order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    sorted_context = _sorted_operator_context(context, len(frame))
    win = max(int(lag) + 2, int(window or 2))
    minp = max(int(lag) + 2, int(min_periods or max(4, win // 4)))
    columns: dict[str, pd.Series] = {}
    group_bounds = _group_bounds_from_context_or_frame(
        frame,
        symbol_col=symbol_col,
        context=sorted_context,
    )
    for feature in features:
        name = _safe_name(feature)
        columns[
            f"autocorr_lag{int(lag)}_w{win}__{name}"
        ] = _rolling_autocorr_by_symbol(
            frame,
            x[feature],
            symbol_col=symbol_col,
            window=win,
            min_periods=minp,
            lag=int(lag),
            group_bounds=group_bounds,
        )
    out = pd.DataFrame(columns, index=frame.index, dtype=np.float32)
    return restore_time_sorted_frame(out, original_index=original_index, order=order)


def score_pair_candidates(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    mechanisms: Mapping[str, str] | None = None,
    rolling_window: int = 168,
    min_periods: int | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    warmup_rows: int = 0,
    min_good_row_fraction: float = 0.0,
    treat_zero_as_low_quality: bool = True,
    max_pairs: int | None = None,
    prefer_different_mechanisms: bool = True,
    sparse_graph_enabled: bool = True,
    sparse_graph_block_hours: int | None = None,
    sparse_graph_min_block_rows: int | None = None,
    sparse_graph_alpha: float = 0.05,
    sparse_graph_partial_corr_threshold: float = 1e-4,
    sparse_graph_max_iter: int = 100,
    sparse_graph_weight: float = 0.50,
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Score rolling correlation/covariance pair candidates.

    ``pair_score = variation(rho_t) * persistence(rho_t) * reliability``.
    Reliability combines row coverage and rolling-correlation finite coverage.
    """

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    x = numeric_feature_frame(frame, features)
    original_frame = frame
    frame, x, _order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    sorted_context = _sorted_operator_context(context, len(frame))
    quality = compute_quality_report(
        original_frame,
        features,
        warmup_rows=warmup_rows,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        min_good_row_fraction=min_good_row_fraction,
        treat_zero_as_low_quality=treat_zero_as_low_quality,
        context=context,
    )
    pairs = list(combinations(features, 2))
    if prefer_different_mechanisms and mechanisms:
        diff_pairs = [
            pair
            for pair in pairs
            if str(mechanisms.get(pair[0], "unknown"))
            != str(mechanisms.get(pair[1], "unknown"))
        ]
        if diff_pairs:
            pairs = diff_pairs
    win = max(3, int(rolling_window or 3))
    minp = max(3, int(min_periods or max(4, win // 4)))
    graph_scores: dict[tuple[str, str], dict[str, float]] = {}
    if sparse_graph_enabled:
        graph_scores = sparse_dependency_graph_edge_scores(
            frame,
            x,
            features,
            timestamp_col=timestamp_col,
            block_hours=int(sparse_graph_block_hours or rolling_window or 24 * 7),
            min_block_rows=int(sparse_graph_min_block_rows or minp),
            alpha=float(sparse_graph_alpha),
            partial_corr_threshold=float(sparse_graph_partial_corr_threshold),
            max_iter=int(sparse_graph_max_iter),
        )
    pair_summary: dict[tuple[str, str], tuple[float, float, float]] = {}
    if _NUMBA_AVAILABLE and pairs:
        feature_pos = {feature: i for i, feature in enumerate(features)}
        left_idx = np.asarray([feature_pos[left] for left, _right in pairs], dtype=np.int64)
        right_idx = np.asarray([feature_pos[right] for _left, right in pairs], dtype=np.int64)
        starts, ends = _group_bounds_from_context_or_frame(
            frame,
            symbol_col=symbol_col,
            context=sorted_context,
        )
        variation_arr, persistence_arr, finite_fraction_arr = (
            _rolling_corr_pair_summaries_numba(
                x[features].to_numpy(dtype=np.float32, copy=False),
                left_idx,
                right_idx,
                starts,
                ends,
                int(win),
                int(minp),
            )
        )
        pair_summary = {
            pair: (
                float(variation_arr[i]),
                float(persistence_arr[i]),
                float(finite_fraction_arr[i]),
            )
            for i, pair in enumerate(pairs)
        }
    rows: list[dict[str, object]] = []
    for left_name, right_name in pairs:
        summary = pair_summary.get((left_name, right_name))
        if summary is None:
            rho = _rolling_pair_by_symbol(
                frame,
                x[left_name],
                x[right_name],
                symbol_col=symbol_col,
                window=win,
                min_periods=minp,
                method="corr",
            ).replace([np.inf, -np.inf], np.nan)
            rho_values = rho.to_numpy(dtype=np.float64)
            finite = np.isfinite(rho_values)
            if int(finite.sum()) < 3:
                continue
            rho_finite = rho_values[finite]
            variation = float(np.nanstd(rho_finite))
            persistence = float(np.nanmean(np.abs(rho_finite)))
            finite_fraction = float(finite.mean())
        else:
            variation, persistence, finite_fraction = summary
            if int(round(finite_fraction * len(frame))) < 3:
                continue
        q_left = (
            float(quality.loc[left_name, "good_fraction"])
            if left_name in quality.index
            else 0.0
        )
        q_right = (
            float(quality.loc[right_name, "good_fraction"])
            if right_name in quality.index
            else 0.0
        )
        reliability = float(np.sqrt(max(q_left, 0.0) * max(q_right, 0.0)))
        reliability *= finite_fraction
        graph_key = tuple(sorted((left_name, right_name)))
        graph = graph_scores.get(
            graph_key,
            {
                "graph_edge_stability": 0.0,
                "graph_edge_strength": 0.0,
                "sparse_graph_score": 0.0,
                "sparse_graph_blocks": 0.0,
            },
        )
        sparse_graph_score = float(graph.get("sparse_graph_score", 0.0))
        rho_score = variation * persistence
        pair_score = reliability * (
            rho_score + float(sparse_graph_weight) * sparse_graph_score
        )
        rows.append(
            {
                "feature_i": left_name,
                "feature_j": right_name,
                "pair_score": float(pair_score),
                "rho_variation": variation,
                "rho_persistence": persistence,
                "reliability": reliability,
                "graph_edge_stability": float(graph.get("graph_edge_stability", 0.0)),
                "graph_edge_strength": float(graph.get("graph_edge_strength", 0.0)),
                "sparse_graph_score": sparse_graph_score,
                "sparse_graph_blocks": float(graph.get("sparse_graph_blocks", 0.0)),
                "mechanism_i": str((mechanisms or {}).get(left_name, "unknown")),
                "mechanism_j": str((mechanisms or {}).get(right_name, "unknown")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_i",
                "feature_j",
                "pair_score",
                "rho_variation",
                "rho_persistence",
                "reliability",
                "graph_edge_stability",
                "graph_edge_strength",
                "sparse_graph_score",
                "sparse_graph_blocks",
                "mechanism_i",
                "mechanism_j",
            ]
        )
    out = pd.DataFrame(rows).sort_values(
        "pair_score", ascending=False, kind="mergesort"
    )
    if max_pairs is not None and int(max_pairs) > 0:
        out = out.head(int(max_pairs))
    for col in [
        "pair_score",
        "rho_variation",
        "rho_persistence",
        "reliability",
        "graph_edge_stability",
        "graph_edge_strength",
        "sparse_graph_score",
        "sparse_graph_blocks",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(np.float32)
    return out.reset_index(drop=True)


def generate_pair_operator_features(
    frame: pd.DataFrame,
    pair_scores: pd.DataFrame,
    *,
    window: int = 168,
    min_periods: int | None = None,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    max_pairs: int | None = None,
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Generate rolling covariance and correlation columns for scored pairs."""

    pairs = pair_scores
    if max_pairs is not None and int(max_pairs) > 0:
        pairs = pairs.head(int(max_pairs))
    pair_features = list(
        dict.fromkeys(
            [str(v) for v in pairs.get("feature_i", [])]
            + [str(v) for v in pairs.get("feature_j", [])]
        )
    )
    x = numeric_feature_frame(frame, pair_features)
    original_index = frame.index
    frame, x, order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    sorted_context = _sorted_operator_context(context, len(frame))
    win = max(3, int(window or 3))
    minp = max(3, int(min_periods or max(4, win // 4)))
    columns: dict[str, object] = {}
    if not pair_features:
        return pd.DataFrame(index=original_index, dtype=np.float32)
    values = x[pair_features].to_numpy(dtype=np.float32, copy=False)
    feature_pos = {feature: i for i, feature in enumerate(pair_features)}
    group_bounds = _group_bounds_from_context_or_frame(
        frame,
        symbol_col=symbol_col,
        context=sorted_context,
    )
    for row in pairs.itertuples(index=False):
        left_name = str(row.feature_i)
        right_name = str(row.feature_j)
        if left_name not in feature_pos or right_name not in feature_pos:
            continue
        suffix = f"{_safe_name(left_name)}__{_safe_name(right_name)}"
        cov_values, corr_values = _rolling_pair_values_from_arrays(
            values[:, feature_pos[left_name]],
            values[:, feature_pos[right_name]],
            group_bounds[0],
            group_bounds[1],
            window=win,
            min_periods=minp,
        )
        columns[f"cov_w{win}__{suffix}"] = cov_values
        columns[f"corr_w{win}__{suffix}"] = corr_values
    out = pd.DataFrame(columns, index=frame.index, dtype=np.float32)
    return restore_time_sorted_frame(out, original_index=original_index, order=order)


def make_mechanism_feature_groups(
    feature_columns: Sequence[str],
    mechanisms: Mapping[str, str] | None,
    *,
    min_group_size: int = 2,
    max_group_size: int = 32,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in dict.fromkeys(str(c) for c in feature_columns if str(c)):
        group = str((mechanisms or {}).get(feature, "ungrouped"))
        groups.setdefault(group, []).append(feature)
    out: dict[str, list[str]] = {}
    for group, features in groups.items():
        if len(features) < int(min_group_size):
            continue
        for start in range(0, len(features), int(max_group_size)):
            chunk = features[start : start + int(max_group_size)]
            if len(chunk) >= int(min_group_size):
                key = group if start == 0 else f"{group}_{start // int(max_group_size) + 1}"
                out[key] = chunk
    return out


def _eigen_summary(matrix: np.ndarray, top_k: int) -> tuple[float, float, float, float, np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 2:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    col_medians = np.nanmedian(arr, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    inds = np.where(~np.isfinite(arr))
    if inds[0].size:
        arr = arr.copy()
        arr[inds] = np.take(col_medians, inds[1])
    arr = arr - np.nanmean(arr, axis=0, keepdims=True)
    cov = np.cov(arr, rowvar=False)
    if cov.ndim != 2 or not np.isfinite(cov).any():
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    vals, vecs = np.linalg.eigh(np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0))
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    total = float(np.sum(vals))
    if total <= 1e-12:
        return np.nan, np.nan, np.nan, np.nan, np.array([], dtype=np.float64)
    shares = vals / total
    top_share = float(np.sum(shares[: max(1, int(top_k))]))
    positive = shares[shares > 1e-12]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation = float((np.sum(vals) ** 2) / max(np.sum(vals * vals), 1e-12))
    return float(shares[0]), top_share, effective_rank, participation, vecs[:, 0]


def generate_eigenvalue_summary_features(
    frame: pd.DataFrame,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    window: int = 168,
    min_periods: int | None = None,
    top_k: int = 3,
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    context: PreparedFrameContext | None = None,
) -> pd.DataFrame:
    """Rolling covariance eigen summaries per feature mechanism group."""

    all_features = list(
        dict.fromkeys(
            feature
            for group_features in feature_groups.values()
            for feature in group_features
        )
    )
    x = numeric_feature_frame(frame, all_features)
    original_index = frame.index
    frame, x, order = time_sorted_frame_and_features(
        frame,
        x,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
        context=context,
    )
    sorted_context = _sorted_operator_context(context, len(frame))
    win = max(3, int(window or 3))
    minp = max(3, int(min_periods or max(4, win // 4)))
    columns: dict[str, np.ndarray] = {}
    group_bounds = _group_bounds_from_context_or_frame(
        frame,
        symbol_col=symbol_col,
        context=sorted_context,
    )
    for group_name, group_features_raw in feature_groups.items():
        group_features = [str(c) for c in group_features_raw if str(c) in x.columns]
        if len(group_features) < 2:
            continue
        safe_group = _safe_name(str(group_name))
        cols = [
            f"eig_largest_share__{safe_group}",
            f"eig_top{int(top_k)}_share__{safe_group}",
            f"eig_effective_rank__{safe_group}",
            f"eig_participation_ratio__{safe_group}",
            f"eig_turnover__{safe_group}",
        ]
        for col in cols:
            columns[col] = np.full(len(frame), np.nan, dtype=np.float32)
        group_values = x[group_features].to_numpy(dtype=np.float32, copy=False)
        for group_start, group_end in zip(group_bounds[0], group_bounds[1]):
            positions = np.arange(int(group_start), int(group_end), dtype=np.int64)
            if positions.size == 0:
                continue
            prev_vec: np.ndarray | None = None
            values = group_values[positions].astype(np.float64, copy=False)
            for local_idx in range(len(positions)):
                start = max(0, local_idx - win + 1)
                sample = values[start : local_idx + 1]
                finite_rows = np.isfinite(sample).any(axis=1)
                if int(finite_rows.sum()) < minp:
                    continue
                (
                    largest_share,
                    top_share,
                    effective_rank,
                    participation,
                    vec,
                ) = _eigen_summary(sample[finite_rows], top_k=top_k)
                turnover = np.nan
                if prev_vec is not None and vec.size == prev_vec.size and vec.size:
                    turnover = float(1.0 - abs(float(np.dot(prev_vec, vec))))
                if vec.size:
                    prev_vec = vec
                row_pos = int(positions[local_idx])
                columns[cols[0]][row_pos] = largest_share
                columns[cols[1]][row_pos] = top_share
                columns[cols[2]][row_pos] = effective_rank
                columns[cols[3]][row_pos] = participation
                columns[cols[4]][row_pos] = turnover
    out = pd.DataFrame(columns, index=frame.index, dtype=np.float32)
    return restore_time_sorted_frame(
        out,
        original_index=original_index,
        order=order,
    )


def _robust_scale_matrix(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float64, copy=True)
    center = np.nanmedian(arr, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    q25 = np.nanpercentile(arr, 25.0, axis=0)
    q75 = np.nanpercentile(arr, 75.0, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(arr, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(center, np.where(missing)[1])
    scaled = (arr - center.reshape(1, -1)) / scale.reshape(1, -1)
    return np.clip(scaled, -8.0, 8.0).astype(np.float32), center, scale


def _apply_scale_matrix(
    x: pd.DataFrame,
    center: Sequence[float],
    scale: Sequence[float],
) -> np.ndarray:
    arr = x.to_numpy(dtype=np.float64, copy=True)
    center_arr = np.asarray(center, dtype=np.float64)
    scale_arr = np.asarray(scale, dtype=np.float64)
    if center_arr.size != arr.shape[1]:
        center_arr = np.zeros(arr.shape[1], dtype=np.float64)
    if scale_arr.size != arr.shape[1]:
        scale_arr = np.ones(arr.shape[1], dtype=np.float64)
    scale_arr = np.where(np.isfinite(scale_arr) & (scale_arr > 1e-8), scale_arr, 1.0)
    missing = ~np.isfinite(arr)
    if missing.any():
        arr[missing] = np.take(center_arr, np.where(missing)[1])
    scaled = (arr - center_arr.reshape(1, -1)) / scale_arr.reshape(1, -1)
    return np.clip(scaled, -8.0, 8.0).astype(np.float32)


def _svd_knn_columns(
    *,
    svd_components: Sequence[int],
    knn_svd_components: int,
) -> list[str]:
    cols: list[str] = []
    requested = list(dict.fromkeys(int(c) for c in svd_components))
    for requested_components in requested:
        for i in range(int(requested_components)):
            cols.append(f"svd{int(requested_components)}_{i:02d}")
    if int(knn_svd_components) in set(requested):
        cols.extend(
            [
                "svd16_knn_dist_mean",
                "svd16_knn_dist_p50",
                "svd16_knn_radius",
                "svd16_knn_density",
            ]
        )
    return cols


def fit_svd_knn_state(
    reference_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    svd_components: Sequence[int] = (8, 16, 32),
    knn_svd_components: int = 16,
    knn_neighbors: int = 25,
    random_state: int = 42,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    max_reference_rows: int | None = None,
    knn_max_reference_rows: int | None = None,
    sample_time_bins: int = 24,
) -> dict[str, object]:
    """Fit robust scaling, SVD components, and a KNN reference on reference rows."""

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    original_reference_rows = len(reference_frame)
    if max_reference_rows is not None and int(max_reference_rows) > 0:
        sample_positions = stratified_period_sample_positions(
            reference_frame,
            np.arange(len(reference_frame), dtype=int),
            max_rows=int(max_reference_rows),
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            n_periods=int(sample_time_bins),
        )
        if len(sample_positions) < len(reference_frame):
            reference_frame = reference_frame.iloc[sample_positions].reset_index(
                drop=True
            )
    x = numeric_feature_frame(reference_frame, features)
    if not features or len(reference_frame) < 2:
        return {
            "enabled": False,
            "reason": "insufficient_reference_rows_or_features",
            "feature_columns": features,
            "svd_components": list(dict.fromkeys(int(c) for c in svd_components)),
            "knn_svd_components": int(knn_svd_components),
            "knn_neighbors": int(knn_neighbors),
            "original_reference_rows": int(original_reference_rows),
            "sampled_reference_rows": int(len(reference_frame)),
        }
    arr, center, scale = _robust_scale_matrix(x)
    state: dict[str, object] = {
        "enabled": True,
        "feature_columns": features,
        "center": center.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "svd_components": list(dict.fromkeys(int(c) for c in svd_components)),
        "knn_svd_components": int(knn_svd_components),
        "knn_neighbors": int(knn_neighbors),
        "original_reference_rows": int(original_reference_rows),
        "sampled_reference_rows": int(len(reference_frame)),
        "svd": {},
    }
    max_rank = max(1, min(arr.shape[0] - 1, arr.shape[1]))
    embeddings: dict[int, np.ndarray] = {}
    for requested_components in state["svd_components"]:
        n_components = max(1, min(int(requested_components), max_rank))
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        z = svd.fit_transform(arr).astype(np.float32)
        padded = np.zeros((len(reference_frame), int(requested_components)), dtype=np.float32)
        padded[:, :n_components] = z[:, :n_components]
        embeddings[int(requested_components)] = padded
        state["svd"][int(requested_components)] = {
            "n_components_fit": int(n_components),
            "components": svd.components_.astype(float).tolist(),
            "explained_variance_ratio": svd.explained_variance_ratio_.astype(
                float
            ).tolist(),
        }
    knn_key = int(knn_svd_components)
    z_knn = embeddings.get(knn_key)
    if z_knn is not None and len(z_knn) >= 2:
        knn_ref = z_knn
        if knn_max_reference_rows is not None and int(knn_max_reference_rows) > 0:
            knn_positions = stratified_period_sample_positions(
                reference_frame,
                np.arange(len(reference_frame), dtype=int),
                max_rows=int(knn_max_reference_rows),
                timestamp_col=timestamp_col,
                symbol_col=symbol_col,
                n_periods=int(sample_time_bins),
            )
            knn_ref = z_knn[knn_positions]
        state["knn"] = {
            "enabled": True,
            "svd_components": knn_key,
            "neighbors": int(max(1, min(int(knn_neighbors), len(knn_ref)))),
            "reference_rows": int(len(knn_ref)),
            "reference_embedding": knn_ref.astype(float).tolist(),
        }
    else:
        state["knn"] = {"enabled": False, "reason": "svd_reference_unavailable"}
    return state


def transform_svd_knn_features(
    frame: pd.DataFrame,
    state: Mapping[str, object],
) -> pd.DataFrame:
    """Apply a fitted SVD/KNN state to rows without refitting on those rows."""

    features = [str(c) for c in state.get("feature_columns", [])]
    svd_components = [int(c) for c in state.get("svd_components", [8, 16, 32])]
    knn_svd_components = int(state.get("knn_svd_components", 16))
    out = pd.DataFrame(
        np.nan,
        index=frame.index,
        columns=_svd_knn_columns(
            svd_components=svd_components,
            knn_svd_components=knn_svd_components,
        ),
        dtype=np.float32,
    )
    if not bool(state.get("enabled", False)) or not features:
        return out
    x = numeric_feature_frame(frame, features)
    arr = _apply_scale_matrix(
        x,
        state.get("center", []),
        state.get("scale", []),
    )
    embeddings: dict[int, np.ndarray] = {}
    svd_state = state.get("svd", {})
    if not isinstance(svd_state, Mapping):
        return out
    for requested_components in svd_components:
        spec = svd_state.get(int(requested_components), {})
        if not isinstance(spec, Mapping):
            continue
        components = np.asarray(spec.get("components", []), dtype=np.float32)
        if components.ndim != 2 or components.shape[1] != arr.shape[1]:
            continue
        z = (arr @ components.T).astype(np.float32)
        padded = np.zeros((len(frame), int(requested_components)), dtype=np.float32)
        dim = min(padded.shape[1], z.shape[1])
        padded[:, :dim] = z[:, :dim]
        embeddings[int(requested_components)] = padded
        for i in range(int(requested_components)):
            out[f"svd{int(requested_components)}_{i:02d}"] = padded[:, i]
    knn = state.get("knn", {})
    if isinstance(knn, Mapping) and bool(knn.get("enabled", False)):
        z_knn = embeddings.get(knn_svd_components)
        ref = np.asarray(knn.get("reference_embedding", []), dtype=np.float32)
        if z_knn is not None and ref.ndim == 2 and len(ref) > 0:
            n_neighbors = max(1, min(int(knn.get("neighbors", 1)), len(ref)))
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
            nn.fit(ref)
            dist, _idx = nn.kneighbors(z_knn, return_distance=True)
            mean_dist = np.mean(dist, axis=1)
            out["svd16_knn_dist_mean"] = mean_dist.astype(np.float32)
            out["svd16_knn_dist_p50"] = np.median(dist, axis=1).astype(np.float32)
            out["svd16_knn_radius"] = np.max(dist, axis=1).astype(np.float32)
            out["svd16_knn_density"] = (1.0 / (1.0 + mean_dist)).astype(np.float32)
    return out.astype(np.float32)


def fit_transform_svd_knn_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    reference_frame: pd.DataFrame | None = None,
    fit_mask: Sequence[bool] | None = None,
    svd_components: Sequence[int] = (8, 16, 32),
    knn_svd_components: int = 16,
    knn_neighbors: int = 25,
    random_state: int = 42,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    max_reference_rows: int | None = None,
    knn_max_reference_rows: int | None = None,
    sample_time_bins: int = 24,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit on reference rows, then transform ``frame`` without refitting."""

    if reference_frame is None and fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=bool)
        if len(mask) != len(frame):
            raise ValueError("fit_mask must have the same length as frame")
        reference_frame = frame.loc[mask]
    if reference_frame is None:
        reference_frame = frame
    state = fit_svd_knn_state(
        reference_frame,
        feature_columns,
        svd_components=svd_components,
        knn_svd_components=knn_svd_components,
        knn_neighbors=knn_neighbors,
        random_state=random_state,
        timestamp_col=timestamp_col,
        symbol_col=symbol_col,
        max_reference_rows=max_reference_rows,
        knn_max_reference_rows=knn_max_reference_rows,
        sample_time_bins=sample_time_bins,
    )
    state["fit_scope"] = "reference_frame"
    return transform_svd_knn_features(frame, state), state


def fit_transform_svd_knn_features_walk_forward(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    svd_components: Sequence[int] = (8, 16, 32),
    knn_svd_components: int = 16,
    knn_neighbors: int = 25,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    block_hours: int = 24 * 7,
    min_prior_rows: int = 500,
    random_state: int = 42,
    max_reference_rows: int | None = None,
    knn_max_reference_rows: int | None = None,
    sample_time_bins: int = 24,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prior-only SVD/KNN features by timestamp block.

    Each block is transformed with SVD/KNN state fitted on rows strictly before
    that block's first timestamp. Early blocks with insufficient prior rows stay
    NaN instead of fitting on their own future.
    """

    features = list(dict.fromkeys(str(c) for c in feature_columns if str(c)))
    columns = _svd_knn_columns(
        svd_components=svd_components,
        knn_svd_components=knn_svd_components,
    )
    out = pd.DataFrame(np.nan, index=frame.index, columns=columns, dtype=np.float32)
    if not features or len(frame) == 0 or timestamp_col not in frame.columns:
        state = {
            "enabled": False,
            "reason": "missing_features_or_timestamp",
            "mode": "walk_forward_prior_only",
            "feature_columns": features,
        }
        return out, state
    order = time_sort_order(
        frame,
        symbol_col=symbol_col,
        timestamp_col=timestamp_col,
    )
    sorted_frame = frame.iloc[order].reset_index(drop=True)
    ts = pd.to_datetime(sorted_frame[timestamp_col], utc=True, errors="coerce")
    valid_ts = ts.notna().to_numpy(dtype=bool)
    if not valid_ts.any():
        state = {
            "enabled": False,
            "reason": "no_valid_timestamps",
            "mode": "walk_forward_prior_only",
            "feature_columns": features,
        }
        return out, state
    ts_ns = ts.to_numpy(dtype="datetime64[ns]").astype("int64")
    start_ns = int(np.nanmin(ts_ns[valid_ts]))
    block_ns = max(1, int(block_hours or 1) * 3600 * 1_000_000_000)
    block_ids = (ts_ns - start_ns) // block_ns
    sorted_out = pd.DataFrame(
        np.nan,
        index=sorted_frame.index,
        columns=columns,
        dtype=np.float32,
    )
    enabled_blocks = 0
    skipped_blocks = 0
    fit_rows: list[int] = []
    sampled_fit_rows: list[int] = []
    knn_reference_rows: list[int] = []
    for block_id in np.unique(block_ids[valid_ts]):
        current_pos = np.flatnonzero(valid_ts & (block_ids == block_id))
        if len(current_pos) == 0:
            continue
        block_start_ns = int(np.nanmin(ts_ns[current_pos]))
        reference_pos = np.flatnonzero(valid_ts & (ts_ns < block_start_ns))
        if len(reference_pos) < int(min_prior_rows):
            skipped_blocks += 1
            continue
        state = fit_svd_knn_state(
            sorted_frame.iloc[reference_pos],
            features,
            svd_components=svd_components,
            knn_svd_components=knn_svd_components,
            knn_neighbors=knn_neighbors,
            random_state=random_state,
            timestamp_col=timestamp_col,
            symbol_col=symbol_col,
            max_reference_rows=max_reference_rows,
            knn_max_reference_rows=knn_max_reference_rows,
            sample_time_bins=sample_time_bins,
        )
        transformed = transform_svd_knn_features(sorted_frame.iloc[current_pos], state)
        sorted_out.iloc[current_pos, :] = transformed.reindex(columns=columns).to_numpy(
            dtype=np.float32
        )
        enabled_blocks += 1
        fit_rows.append(int(len(reference_pos)))
        sampled_fit_rows.append(int(state.get("sampled_reference_rows", 0) or 0))
        knn_state = state.get("knn", {})
        if isinstance(knn_state, Mapping):
            knn_reference_rows.append(int(knn_state.get("reference_rows", 0) or 0))
    restored = restore_time_sorted_frame(
        sorted_out,
        original_index=frame.index,
        order=order,
    )
    state = {
        "enabled": enabled_blocks > 0,
        "mode": "walk_forward_prior_only",
        "feature_columns": features,
        "svd_components": list(dict.fromkeys(int(c) for c in svd_components)),
        "knn_svd_components": int(knn_svd_components),
        "knn_neighbors": int(knn_neighbors),
        "block_hours": int(block_hours),
        "min_prior_rows": int(min_prior_rows),
        "enabled_blocks": int(enabled_blocks),
        "skipped_blocks": int(skipped_blocks),
        "min_fit_rows": int(min(fit_rows)) if fit_rows else 0,
        "max_fit_rows": int(max(fit_rows)) if fit_rows else 0,
        "max_reference_rows": int(max_reference_rows or 0),
        "knn_max_reference_rows": int(knn_max_reference_rows or 0),
        "min_sampled_fit_rows": int(min(sampled_fit_rows)) if sampled_fit_rows else 0,
        "max_sampled_fit_rows": int(max(sampled_fit_rows)) if sampled_fit_rows else 0,
        "max_knn_reference_rows_used": int(max(knn_reference_rows))
        if knn_reference_rows
        else 0,
    }
    return restored.astype(np.float32), state
