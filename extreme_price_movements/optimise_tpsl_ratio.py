"""
Production-grade TP/SL Grid Selection (ATR-Scaled Triple Barrier) — High Throughput

Key optimizations implemented:

A) Normalized return tensors (no per-grid price-level recompute)
   Precompute:
     rH = high_window / entry_px - 1
     rL = low_window  / entry_px - 1
     rC_end = close_end / entry_px - 1
   Then, for each (tp_mult, sl_mult):
     tp_thr = tp_mult * barrier_pct
     sl_thr = sl_mult * barrier_pct
     PT hit: rH >= tp_thr
     SL hit: rL <= -sl_thr

B) Fast weighted ridge via Cholesky (no sklearn in inner loops)
   Solve (X^T W X + αI) w = X^T W y  (y in {-1,+1})
   - Adds intercept by augmenting X with a column of ones
   - Does NOT regularize intercept (configurable)

C) Sample weights: AE until exit, precomputed once
   - Compute AE% until exit using prefix-min lows up to exit index:
       exit_t = min(pt_t, sl_t, horizon-1) with sentinels handled
       min_low_until_exit = prefix_min_low[range(m), exit_t]
       ae_pct = max(0, (entry - min_low_until_exit)/entry)
   - Inside loop: only scalar division by (sl_mult * barrier_pct) to scale AE
   - Map to weight multiplier in [0.5, 2.0], lower AE => higher weight
   - Also incorporates class_weight='balanced' behavior via per-sample weights

D) No redundant window builds / gathers
   - Event windows (H/L/C) gathered once
   - Prefix minima computed once

Assumptions:
- Single instrument arrays, time-aligned.
- Event indices refer to signal time t.
- Window is [t+1 ... t+horizon] (horizon bars).
- Entry is open[t+1] (default) or close[t].

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import scipy.linalg as la
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from extreme_price_movements.utils import tprint

# Optional rolling speedups
try:
    import bottleneck as bn  # type: ignore
    _HAS_BN = True
except Exception:
    bn = None
    _HAS_BN = False

try:
    import pandas as pd  # type: ignore
    _HAS_PD = True
except Exception:
    pd = None
    _HAS_PD = False


# --------------------------
# Walk-Forward CV (expanding window, train > test)
# --------------------------
class WalkForwardCV:
    """Expanding-window walk-forward cross-validation.
    
    Events MUST be sorted by time before calling split().
    Each fold: train = [0 .. split_point - purge], test = [split_point .. split_point + test_size].
    Train always >= 2x test. Purge gap between train end and test start.
    """
    def __init__(self, n_splits: int = 3, purge: int = 5, test_fraction: float = 0.15,
                 min_train_size: int = 50):
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.test_fraction = float(test_fraction)
        self.min_train_size = int(min_train_size)

    def split(self, X) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """X = event_idx values (must be sorted ascending = temporal order)."""
        vals = np.asarray(X)
        n = len(vals)
        
        # Assert monotonicity (caller must sort by time)
        if n > 1 and np.any(np.diff(vals) < 0):
            bad = np.where(np.diff(vals) < 0)[0]
            pos = int(bad[0]) if bad.size else -1
            raise ValueError(f"WalkForwardCV: event values are not sorted! "
                             f"First violation at position {pos}, val[{pos}]={vals[pos]}, val[{pos+1}]={vals[pos+1]}. "
                             f"Sort events by time before calling.")
        
        test_size = max(10, int(n * self.test_fraction))
        # Place test windows at evenly-spaced positions in the latter part of the data
        # First test window starts after we have enough training data
        min_start = max(self.min_train_size + self.purge, int(n * 0.3))
        max_start = n - test_size
        
        if max_start <= min_start:
            # Not enough data for even 1 fold — try with smaller test
            test_size = max(5, n // 5)
            min_start = max(self.min_train_size, int(n * 0.3))
            max_start = n - test_size
            if max_start <= min_start:
                return  # truly not enough data
        
        # Generate fold start positions
        if self.n_splits == 1:
            starts = [max_start]
        else:
            starts = np.linspace(min_start, max_start, self.n_splits, dtype=int).tolist()
        
        idx = np.arange(n, dtype=np.int32)
        seen_starts = set()
        
        for test_start in starts:
            test_start = int(test_start)
            if test_start in seen_starts:
                continue
            seen_starts.add(test_start)
            
            test_end = min(test_start + test_size, n)
            # Purge: gap between train end and test start (in value space)
            test_min_val = vals[test_start]
            train_mask = vals < (test_min_val - self.purge)
            train = idx[train_mask]
            test = idx[test_start:test_end]
            
            if train.size < self.min_train_size or test.size == 0:
                continue
            
            yield train, test


# Keep PurgedKFold for backward compatibility (inner CV)
class PurgedKFold:
    def __init__(self, n_splits: int = 5, purge: int = 5, embargo: int = 0, min_train_size: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be >=2")
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.embargo = int(embargo)
        self.min_train_size = None if min_train_size is None else int(min_train_size)

    def split(self, X) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        vals = np.asarray(X)
        n = len(vals)
        idx = np.arange(n, dtype=np.int32)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int32)
        fold_sizes[: n % self.n_splits] += 1
        bounds = np.r_[0, fold_sizes.cumsum()]

        for k in range(self.n_splits):
            test_indices = idx[bounds[k]:bounds[k+1]]
            if test_indices.size == 0:
                continue
            test_vals = vals[test_indices]
            t_min, t_max = test_vals.min(), test_vals.max()

            train_mask = (vals < t_min - self.purge) | (vals > t_max + self.embargo)
            train = idx[train_mask]
            test = test_indices

            if self.min_train_size is not None and train.size < self.min_train_size:
                continue
            if train.size == 0 or test.size == 0:
                continue

            yield train, test


# --------------------------
# Rolling utils (efficient)
# --------------------------
def _f32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

def rolling_median_1d(x: np.ndarray, window: int) -> np.ndarray:
    x = _f32(x)
    w = int(window)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float32)
    if w <= 1 or n < w:
        return out
    if _HAS_BN:
        return bn.move_median(x, window=w, min_count=w).astype(np.float32, copy=False)
    if _HAS_PD:
        s = pd.Series(x)
        return s.rolling(w, min_periods=w).median().to_numpy(dtype=np.float32, na_value=np.nan)
    # fallback (last resort)
    s0 = x.strides[0]
    Xw = np.lib.stride_tricks.as_strided(x, shape=(n - w + 1, w), strides=(s0, s0))
    out[w - 1:] = np.median(Xw, axis=1).astype(np.float32)
    return out

def rolling_quantile_1d(x: np.ndarray, window: int, q: float) -> np.ndarray:
    x = _f32(x)
    w = int(window)
    n = x.size
    out = np.full(n, np.nan, dtype=np.float32)
    if w <= 1 or n < w:
        return out
    if _HAS_PD:
        s = pd.Series(x)
        return s.rolling(w, min_periods=w).quantile(q).to_numpy(dtype=np.float32, na_value=np.nan)
    # fallback (last resort)
    s0 = x.strides[0]
    Xw = np.lib.stride_tricks.as_strided(x, shape=(n - w + 1, w), strides=(s0, s0))
    out[w - 1:] = np.quantile(Xw, q, axis=1).astype(np.float32)
    return out

def rolling_mad_1d(x: np.ndarray, window: int, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    x = _f32(x)
    med = rolling_median_1d(x, window)
    w = int(window)
    n = x.size
    mad = np.full(n, np.nan, dtype=np.float32)
    if w <= 1 or n < w:
        return med, mad
    if _HAS_PD:
        s = pd.Series(x)
        med_s = pd.Series(med)
        abs_dev = (s - med_s).abs()
        mad = abs_dev.rolling(w, min_periods=w).median().to_numpy(dtype=np.float32, na_value=np.nan)
        mad = np.maximum(mad, eps).astype(np.float32, copy=False)
        return med.astype(np.float32, copy=False), mad
    # fallback (last resort)
    s0 = x.strides[0]
    Xw = np.lib.stride_tricks.as_strided(x, shape=(n - w + 1, w), strides=(s0, s0))
    Mw = np.lib.stride_tricks.as_strided(med[w - 1:], shape=(n - w + 1, 1), strides=(med.strides[0], 0))
    mad[w - 1:] = np.median(np.abs(Xw - Mw), axis=1).astype(np.float32)
    mad = np.maximum(mad, eps).astype(np.float32, copy=False)
    return med.astype(np.float32, copy=False), mad

def calibrate_atr_base_pct(
    atr_pct: np.ndarray,
    window: int,
    method: str = "percentile",
    q: float = 0.30,
    k: float = 0.5,
    eps: float = 1e-12,
) -> np.ndarray:
    atr_pct = _f32(atr_pct)
    if method == "percentile":
        base = rolling_quantile_1d(atr_pct, window, q=q)
        return np.maximum(base, eps).astype(np.float32, copy=False)
    if method == "mad":
        med, mad = rolling_mad_1d(atr_pct, window, eps=eps)
        base = (med - k * (1.4826 * mad)).astype(np.float32)
        return np.maximum(base, eps).astype(np.float32, copy=False)
    raise ValueError(f"Unknown method={method}")

def compute_vol_z_log_mad(atr_pct: np.ndarray, window: int, eps: float = 1e-12) -> np.ndarray:
    atr_pct = _f32(atr_pct)
    x = np.log(np.maximum(atr_pct, eps)).astype(np.float32, copy=False)
    med, mad = rolling_mad_1d(x, window, eps=eps)
    scale = (1.4826 * mad).astype(np.float32, copy=False)
    z = ((x - med) / np.maximum(scale, eps)).astype(np.float32, copy=False)
    return z


# --------------------------
# ATR scaling — canonical implementation (used by both optimizer and training)
# --------------------------
def scaled_atr_pct(
    atr_pct,
    z,
    atr_base_pct,
    *,
    z_max: float = 3.0,
    lo: float = 0.03,
    hi: float = 0.06,
    eps: float = 1e-12,
):
    """
    ATR-informed, shock-scaled, bounded barrier percent.
    Works with both scalars and numpy arrays (float32-safe).
    """
    is_array = isinstance(atr_pct, np.ndarray)
    if is_array:
        atr_pct = _f32(atr_pct)
        z = _f32(z)
        atr_base_pct = _f32(atr_base_pct)

    shock = np.clip(z, 0.0, z_max)
    if is_array:
        shock = shock.astype(np.float32, copy=False)

    a = (hi / np.maximum(atr_base_pct, eps) - 1.0) / z_max
    if is_array:
        a = a.astype(np.float32, copy=False)

    raw = atr_pct * (1.0 + a * shock)
    result = np.clip(raw, lo, hi)
    if is_array:
        result = result.astype(np.float32, copy=False)
    return result


# Backward-compatible alias
scaled_atr_pct_dynamic_a = scaled_atr_pct


# --------------------------
# Event cache with normalized return tensors
# --------------------------
@dataclass
class EventCache:
    event_idx: np.ndarray   # valid event indices (signal time t)
    entry_px: np.ndarray    # (m,)
    rH: np.ndarray          # (m, horizon) high normalized return: H/entry - 1
    rL: np.ndarray          # (m, horizon) low  normalized return: L/entry - 1
    rC_end: np.ndarray      # (m,) close normalized return at horizon end: C_end/entry - 1
    rL_prefix_min: np.ndarray # (m, horizon) normalized low prefix-min for AE (long)
    rH_prefix_max: np.ndarray # (m, horizon) normalized high prefix-max for AE (short)
    horizon: int
    side: str = "long"


def build_event_cache(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    event_idx: np.ndarray,
    horizon: int,
    entry_mode: str = "next_open",
    side: str = "long",
    eps: float = 1e-12,
) -> EventCache:
    open_ = _f32(open_)
    high = _f32(high)
    low = _f32(low)
    close = _f32(close)

    n = close.size
    event_idx = np.asarray(event_idx, dtype=np.int32)
    HN = int(horizon)

    if entry_mode == "next_open":
        valid = (event_idx + 1 + HN) < n
        e = event_idx[valid]
        entry_px = open_[e + 1]
        start = e + 1
    elif entry_mode == "close":
        valid = (event_idx + HN) < n
        e = event_idx[valid]
        entry_px = close[e]
        start = e + 1
    else:
        raise ValueError("entry_mode must be 'next_open' or 'close'")

    if e.size == 0:
        z = np.zeros((0, HN), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z,
            rH_prefix_max=z,
            horizon=HN,
            side=side
        )

    offs = np.arange(HN, dtype=np.int32)[None, :]
    widx = start[:, None] + offs  # (m, H)

    H = high[widx]  # (m,H)
    L = low[widx]
    C_end = close[widx[:, -1]]  # (m,)

    denom = np.maximum(entry_px, eps).astype(np.float32, copy=False)
    rH = (H / denom[:, None]) - 1.0
    rL = (L / denom[:, None]) - 1.0
    rC_end = (C_end / denom) - 1.0

    # Memory-optimized Prefix calculation on normalized returns (Fix #9)
    rL_prefix_min = np.minimum.accumulate(rL, axis=1).astype(np.float32, copy=False)
    rH_prefix_max = np.maximum.accumulate(rH, axis=1).astype(np.float32, copy=False)

    return EventCache(
        event_idx=e.astype(np.int32, copy=False),
        entry_px=entry_px.astype(np.float32, copy=False),
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN,
        side=side
    )


def build_event_cache_15m(
    open_15m: np.ndarray,
    high_15m: np.ndarray,
    low_15m: np.ndarray,
    close_15m: np.ndarray,
    event_idx_1h: np.ndarray,
    horizon_1h: int,
    entry_mode: str = "next_open",
    side: str = "long",
    eps: float = 1e-12,
) -> EventCache:
    """Build event cache from 15m OHLCV data.

    event_idx_1h: indices into the 1h timeline (signal bar).
    Each 1h bar maps to 4 consecutive 15m bars: 1h_idx * 4 .. 1h_idx * 4 + 3.
    The horizon in 15m bars is horizon_1h * 4.
    Entry price is the open of the first 15m bar after the signal hour.
    """
    open_15m = _f32(open_15m)
    high_15m = _f32(high_15m)
    low_15m = _f32(low_15m)
    close_15m = _f32(close_15m)

    n_15m = close_15m.size
    event_idx_1h = np.asarray(event_idx_1h, dtype=np.int32)
    HN_15m = int(horizon_1h) * 4  # 15m resolution

    # Map 1h indices to 15m indices
    event_idx_15m = event_idx_1h * 4  # start of the signal hour in 15m

    if entry_mode == "next_open":
        # Entry = open of first 15m bar of the NEXT hour = (event_1h + 1) * 4
        entry_start_15m = event_idx_15m + 4
        valid = (entry_start_15m + HN_15m) < n_15m
        e_1h = event_idx_1h[valid]
        start = entry_start_15m[valid]
        entry_px = open_15m[start]
    elif entry_mode == "close":
        # Entry = close of last 15m bar of signal hour = event_15m + 3
        entry_close_15m = event_idx_15m + 3
        valid = (entry_close_15m + 1 + HN_15m) < n_15m
        e_1h = event_idx_1h[valid]
        entry_px = close_15m[entry_close_15m[valid]]
        start = entry_close_15m[valid] + 1
    else:
        raise ValueError("entry_mode must be 'next_open' or 'close'")

    if e_1h.size == 0:
        z = np.zeros((0, HN_15m), dtype=np.float32)
        return EventCache(
            event_idx=np.zeros(0, dtype=np.int32),
            entry_px=np.zeros(0, dtype=np.float32),
            rH=z, rL=z, rC_end=np.zeros(0, dtype=np.float32),
            rL_prefix_min=z, rH_prefix_max=z,
            horizon=HN_15m, side=side
        )

    offs = np.arange(HN_15m, dtype=np.int32)[None, :]
    widx = start[:, None] + offs  # (m, HN_15m)

    H = high_15m[widx]
    L = low_15m[widx]
    C_end = close_15m[widx[:, -1]]

    denom = np.maximum(entry_px, eps).astype(np.float32, copy=False)
    rH = (H / denom[:, None]) - 1.0
    rL = (L / denom[:, None]) - 1.0
    rC_end = (C_end / denom) - 1.0

    rL_prefix_min = np.minimum.accumulate(rL, axis=1).astype(np.float32, copy=False)
    rH_prefix_max = np.maximum.accumulate(rH, axis=1).astype(np.float32, copy=False)

    return EventCache(
        event_idx=e_1h.astype(np.int32, copy=False),
        entry_px=entry_px.astype(np.float32, copy=False),
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        rL_prefix_min=rL_prefix_min,
        rH_prefix_max=rH_prefix_max,
        horizon=HN_15m,
        side=side
    )


# --------------------------
# Labeling: first-touch on normalized returns
# --------------------------
@dataclass
class GridLabels:
    y_bin: np.ndarray      # (m,) uint8
    y_ret: np.ndarray      # (m,) float32
    pt_t: np.ndarray       # (m,) int32
    sl_t: np.ndarray       # (m,) int32
    exit_kind: np.ndarray  # (m,) int8 (1=PT,-1=SL,0=time,2=ambig)


def label_from_cache(
    cache: EventCache,
    barrier_pct: np.ndarray,  # (m,)
    tp_mult: float,
    sl_mult: float,
    trail_mult: float = 0.25,
) -> GridLabels:
    """
    Triple-barrier labeling with trailing-stop-aware PT returns.

    Independent grids:
      tp_thr = tp_mult * barrier_pct   (activation threshold)
      sl_thr = sl_mult * barrier_pct
    Conditions:
      Long: PT = rH >= tp_thr, SL = rL <= -sl_thr
      Short: PT = rL <= -tp_thr, SL = rH >= sl_thr
    Pessimistic: ambiguous => SL hit

    Trailing-aware PT return:
      When TP is touched at bar pt_t, a trailing stop activates.
      trail_dist = trail_mult * barrier_pct.
      The return is: peak_MFE_after_activation - trail_dist,
      floored at tp_thr (can't exit below activation level).
      This bridges the gap between instant-exit triple-barrier
      and the engine's trailing-stop execution.
    """
    m = cache.entry_px.size
    HN = cache.horizon
    barrier_pct = _f32(barrier_pct)

    tp_thr = (float(tp_mult) * barrier_pct).astype(np.float32, copy=False)
    sl_thr = (float(sl_mult) * barrier_pct).astype(np.float32, copy=False)
    trail_dist = (float(trail_mult) * barrier_pct).astype(np.float32, copy=False)

    if cache.side == "long":
        hit_pt = cache.rH >= tp_thr[:, None]
        hit_sl = cache.rL <= (-sl_thr[:, None])
    else: # short
        hit_pt = cache.rL <= (-tp_thr[:, None])
        hit_sl = cache.rH >= sl_thr[:, None]

    pt_any = hit_pt.any(axis=1)
    sl_any = hit_sl.any(axis=1)

    sentinel = HN + 1
    pt_t = np.where(pt_any, hit_pt.argmax(axis=1), sentinel).astype(np.int32, copy=False)
    sl_t = np.where(sl_any, hit_sl.argmax(axis=1), sentinel).astype(np.int32, copy=False)

    # Pessimistic ambiguity resolution: if same bar, SL wins
    ambiguous = (pt_t == sl_t) & (pt_t <= HN - 1)
    pt_first = (pt_t < sl_t)
    sl_first = (sl_t < pt_t)
    # Explicitly pull out ambiguous for diagnostics
    # Note: pt_first and sl_first don't include ambiguous (which is pt_t == sl_t)

    exit_kind = np.zeros(m, dtype=np.int8)
    exit_kind[pt_first] = 1
    # Consistent Ambiguity Resolution (Fix #2, #5, #10):
    # Treat ambiguous (same-bar PT/SL) as SL (-1) for both y_bin and y_ret.
    # We set exit_kind=2 explicitly so we can mask it in diagnostics, 
    # but the implementation below treats it as SL.
    exit_kind[sl_first] = -1
    exit_kind[ambiguous] = 2

    # For labels/returns, resolve pessimistically (ambiguous => SL wins)
    sl_pessimistic = sl_first | ambiguous

    y_ret = np.zeros(m, dtype=np.float32)

    # Trailing-aware PT return: peak MFE after activation minus trail distance
    if np.any(pt_first):
        pt_idx = np.where(pt_first)[0]
        pt_bars = pt_t[pt_idx]  # bar of TP activation
        # Peak MFE from activation to horizon end (using prefix max arrays)
        # rH_prefix_max[:, t] = max(rH[:, 0:t+1]) for longs
        # For the peak after activation, we take prefix_max at horizon end
        last_bar = HN - 1
        if cache.side == "long":
            peak_mfe = cache.rH_prefix_max[pt_idx, last_bar]
        else:
            peak_mfe = -cache.rL_prefix_min[pt_idx, last_bar]
        # Trailing exit return = peak - trail_dist, floored at activation level
        trail_ret = np.maximum(peak_mfe - trail_dist[pt_idx], tp_thr[pt_idx])
        y_ret[pt_first] = trail_ret.astype(np.float32, copy=False)

    y_ret[sl_pessimistic] = -sl_thr[sl_pessimistic]
    
    time_mask = ~(pt_first | sl_pessimistic)
    if np.any(time_mask):
        if cache.side == "long":
            y_ret[time_mask] = cache.rC_end[time_mask]
        else:
            y_ret[time_mask] = -cache.rC_end[time_mask]

    y_bin = np.zeros(m, dtype=np.uint8)
    y_bin[pt_first] = 1

    return GridLabels(y_bin=y_bin, y_ret=y_ret, pt_t=pt_t, sl_t=sl_t, exit_kind=exit_kind)


# --------------------------
# AE until exit (precompute once) + fast weight scaling per grid
# --------------------------
def compute_ae_until_exit_pct(
    cache: EventCache,
    pt_t: np.ndarray,
    sl_t: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    AE until exit for LONG:
      exit_t = min(pt_t, sl_t, horizon-1) with sentinels handled:
        - if pt_t or sl_t is sentinel (HN+1), treat as horizon-1
      min_low_until_exit = prefix_min_low[:, exit_t]
      ae_pct = max(0, (entry - min_low)/entry)

    This is "more correct" than AE over full horizon.
    """
    m = cache.entry_px.size
    HN = cache.horizon
    sentinel = HN + 1

    pt = np.asarray(pt_t, dtype=np.int32)
    sl = np.asarray(sl_t, dtype=np.int32)

    pt_eff = np.where(pt == sentinel, HN - 1, pt)
    sl_eff = np.where(sl == sentinel, HN - 1, sl)
    exit_t = np.minimum(pt_eff, sl_eff).astype(np.int32, copy=False)

    rows = np.arange(m, dtype=np.int32)
    
    if cache.side == "long":
        # AE = (entry - min_low)/entry = 1 - min_low/entry = -(min_low/entry - 1) = -rL_prefix_min
        rL_min = cache.rL_prefix_min[rows, exit_t].astype(np.float32, copy=False)
        ae_pct = np.maximum(0.0, -rL_min)
    else: # short
        # AE = (max_high - entry)/entry = max_high/entry - 1 = rH_prefix_max
        rH_max = cache.rH_prefix_max[rows, exit_t].astype(np.float32, copy=False)
        ae_pct = np.maximum(0.0, rH_max)
        
    return ae_pct.astype(np.float32, copy=False)


def compute_empirical_mfe_stats(cache: EventCache) -> Dict[str, float]:
    """Compute empirical MFE (max favorable excursion) statistics from event cache.
    
    Returns dict with MFE quantiles as fraction of entry price (e.g., 0.02 = 2%).
    These are used to anchor profit-protection thresholds to actual trade behavior.
    """
    _defaults = {"mfe_median": 0.01, "mfe_p25": 0.005, "mfe_p75": 0.02, "mfe_p90": 0.03,
                 "mae_median": 0.01, "mae_p75": 0.02, "n_events": 0}
    m = cache.entry_px.size
    if m == 0:
        return _defaults
    
    # Extract raw MFE/MAE from prefix tensors; use nanmax/nanmin to handle NaN windows
    rH_max = np.nanmax(cache.rH_prefix_max, axis=1)  # max high return over full horizon
    rL_min = np.nanmin(cache.rL_prefix_min, axis=1)   # min low return over full horizon
    
    if cache.side == "long":
        mfe_raw = rH_max    # MFE = max high return
        mae_raw = -rL_min   # MAE = max adverse (low) excursion
    else:
        mfe_raw = -rL_min   # Short MFE = max drop (favorable)
        mae_raw = rH_max    # Short MAE = max rise (adverse)
    
    # Filter out NaN events (windows with missing OHLC data)
    valid = np.isfinite(mfe_raw) & np.isfinite(mae_raw)
    mfe = np.maximum(0.0, mfe_raw[valid])
    mae = np.maximum(0.0, mae_raw[valid])
    
    if len(mfe) < 3:
        return _defaults
    
    # Filter out zero-MFE events (no favorable movement at all)
    mfe_pos = mfe[mfe > 0.001]  # > 0.1% MFE
    if len(mfe_pos) < 5:
        mfe_pos = mfe[mfe > 0]
    if len(mfe_pos) < 3:
        mfe_pos = mfe  # use all
    
    return {
        "mfe_median": float(np.nanmedian(mfe_pos)) if len(mfe_pos) > 0 else 0.01,
        "mfe_p25": float(np.nanpercentile(mfe_pos, 25)) if len(mfe_pos) > 0 else 0.005,
        "mfe_p75": float(np.nanpercentile(mfe_pos, 75)) if len(mfe_pos) > 0 else 0.02,
        "mfe_p90": float(np.nanpercentile(mfe_pos, 90)) if len(mfe_pos) > 0 else 0.03,
        "mae_median": float(np.nanmedian(mae)) if len(mae) > 0 else 0.01,
        "mae_p75": float(np.nanpercentile(mae, 75)) if len(mae) > 0 else 0.02,
        "n_events": int(valid.sum()),
    }


def ae_weight_multiplier(
    ae_pct: np.ndarray,
    sl_pct: np.ndarray,
    *,
    w_min: float = 0.5,
    w_max: float = 2.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Lower AE => higher weight. Scale AE by sl_pct (risk unit).
      r = clip(ae_pct / sl_pct, 0, 1)
      w = w_max - (w_max - w_min)*r
    """
    ae = _f32(ae_pct)
    sl = np.maximum(_f32(sl_pct), eps)
    r = np.clip(ae / sl, 0.0, 1.0).astype(np.float32, copy=False)
    w = (w_max - (w_max - w_min) * r).astype(np.float32, copy=False)
    return np.clip(w, w_min, w_max).astype(np.float32, copy=False)


def class_weight_balanced(y01: np.ndarray) -> np.ndarray:
    """
    Equivalent of sklearn class_weight='balanced' for binary labels in {0,1}.
    w_c = n_samples / (n_classes * count_c)
    """
    y = np.asarray(y01, dtype=np.int32)
    n = y.size
    c0 = max(1, int((y == 0).sum()))
    c1 = max(1, int((y == 1).sum()))
    w0 = n / (2.0 * c0)
    w1 = n / (2.0 * c1)
    out = np.where(y == 1, w1, w0).astype(np.float32, copy=False)
    return out


# --------------------------
# Fast weighted ridge (Cholesky)
# --------------------------
def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    return np.concatenate([X, ones], axis=1)

def fast_ridge_fit_cholesky(
    X: np.ndarray,           # (n,d) already with intercept if desired
    y_pm1: np.ndarray,       # (n,) in {-1,+1}
    sw: np.ndarray,          # (n,) sample weights (already includes class balancing, AE weights, etc.)
    alpha: float,
    regularize_intercept: bool = False,
) -> np.ndarray:
    """
    Solve (X^T W X + αI) w = X^T W y using Cholesky.
    If regularize_intercept=False, last coefficient (intercept) is not regularized.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y_pm1, dtype=np.float32).ravel()
    sw = np.asarray(sw, dtype=np.float32).ravel()

    # Weighted normal equations without forming diag(W):
    # XTWX = X.T @ (X * sw[:,None])
    Xw = X * sw[:, None]  # (n,d)
    xtwx = (X.T @ Xw).astype(np.float64, copy=False)  # (d,d) float64 for numeric stability
    xtwy = (X.T @ (y * sw)).astype(np.float64, copy=False)  # (d,)

    d = xtwx.shape[0]
    if alpha > 0:
        if regularize_intercept:
            idx = np.diag_indices(d)
            xtwx[idx] += alpha
        else:
            # don't regularize last column (intercept)
            idx = np.diag_indices(d)
            xtwx[idx] += alpha
            xtwx[-1, -1] -= alpha

    try:
        # Stabilization jitter (Fix #7)
        jitter = 1e-9 * np.mean(np.diag(xtwx))
        if jitter > 0:
            idx = np.diag_indices(d)
            xtwx[idx] += jitter
            
        c, low = la.cho_factor(xtwx, lower=True, check_finite=False)
        w = la.cho_solve((c, low), xtwy, check_finite=False)
        return w.astype(np.float32, copy=False)
    except (la.LinAlgError, ValueError):
        # fallback
        w = la.lstsq(xtwx, xtwy, check_finite=False)[0]
        return w.astype(np.float32, copy=False)

def fast_ridge_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (X @ w).astype(np.float32, copy=False)


# --------------------------
# Metrics
# --------------------------
def _auc_safe(y_bin: np.ndarray, scores: np.ndarray) -> float:
    yb = np.asarray(y_bin, dtype=np.int32)
    if np.unique(yb).size < 2:
        return 0.5
    try:
        return float(roc_auc_score(yb, scores))
    except Exception:
        return 0.5

def _spearman_ic(scores: np.ndarray, y_ret: np.ndarray) -> float:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(y_ret, dtype=np.float64)
    if s.size < 5:
        return 0.0
    if np.all(s == s[0]) or np.all(r == r[0]):
        return 0.0
    ic = spearmanr(s, r).correlation
    return 0.0 if (ic is None or np.isnan(ic)) else float(ic)

def _pnl_proxy(scores: np.ndarray, y_ret: np.ndarray) -> float:
    pos = np.sign(scores).astype(np.float32)
    return float(np.mean(pos * y_ret))

def _calculate_strategy_metrics(
    scores: np.ndarray, 
    returns: np.ndarray, 
    labels: np.ndarray, 
    exit_kinds: np.ndarray,
    threshold_p: float = 0.5,
    fee_bps: float = 25.0,
) -> Dict[str, float]:
    """
    Net-after-fee strategy performance metrics.
    1. Gating/Masking: Top-threshold scores, EXCLUDE ambiguous (exit_kind=2).
    2. Fee model: round-trip fee = 2 * fee_bps/10000 per trade.
    3. Metrics: Net PnL, Net Profit Factor, T-stat, Win Rate.
    """
    _empty = {"t_stat": 0.0, "pnl": 0.0, "net_pnl": 0.0, "net_pf": 0.0,
              "payoff": 0.0, "wr": 0.0, "tp_p": 0.0, "sl_p": 0.0, "to_p": 0.0,
              "n_active": 0, "ev": 0.0}
    scores = np.asarray(scores, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    
    # Filter out NaN inputs
    finite_mask = np.isfinite(scores) & np.isfinite(returns)
    if not np.all(finite_mask):
        scores = scores[finite_mask]
        returns = returns[finite_mask]
        labels = labels[finite_mask]
        exit_kinds = exit_kinds[finite_mask]
    
    n = len(scores)
    if n < 5:
        return _empty

    # Gate: threshold by percentile
    abs_scores = np.abs(scores)
    thresh = np.percentile(abs_scores, 100.0 * (1.0 - threshold_p))
    mask = abs_scores >= thresh
    
    # EXCLUDE ambiguous (exit_kind=2)
    mask = mask & (exit_kinds != 2)

    n_active = int(mask.sum())
    if n_active < 2:
        return _empty

    m_scores = abs_scores[mask]
    pos = np.sign(scores[mask])
    r_sub = returns[mask]
    l_sub = labels[mask]
    k_sub = exit_kinds[mask]

    # Directional gross returns
    gross_rets = pos * r_sub
    
    # Net returns after round-trip fees
    rt_fee = 2.0 * fee_bps / 10000.0
    net_rets = gross_rets - rt_fee

    # Net PnL and Profit Factor
    net_pnl = float(np.sum(net_rets))
    gross_wins = float(np.sum(net_rets[net_rets > 0]))
    gross_losses = float(np.abs(np.sum(net_rets[net_rets <= 0])))
    net_pf = gross_wins / max(gross_losses, 1e-9)

    mu = float(np.mean(net_rets))
    
    # Robust T-stat with floor
    std = np.std(net_rets, ddof=1)
    std_floor = 0.25 * float(np.median(np.abs(net_rets)))
    std = max(float(std), float(std_floor), 1e-6)
    t_stat = mu / (std / np.sqrt(n_active) + 1e-12)
    
    wr = float((net_rets > 0).mean())
    tp_p = float((k_sub == 1).mean())
    sl_p = float((k_sub == -1).mean())
    to_p = float((k_sub == 0).mean())
    
    # Payoff ratio: avg_win / avg_loss (higher = better asymmetry)
    wins = net_rets[net_rets > 0]
    losses = net_rets[net_rets <= 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(np.abs(losses.mean())) if len(losses) > 0 else 1e-9
    payoff = avg_win / max(avg_loss, 1e-9)
    
    return {
        "t_stat": float(t_stat),
        "pnl": net_pnl,
        "net_pnl": net_pnl,
        "net_pf": net_pf,
        "payoff": payoff,
        "wr": wr,
        "tp_p": tp_p,
        "sl_p": sl_p,
        "to_p": to_p,
        "n_active": n_active,
        "ev": mu,
    }

def _rank01(x: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    if x.size > 1:
        xs = x[order]
        i = 0
        while i < xs.size:
            j = i + 1
            while j < xs.size and xs[j] == xs[i]:
                j += 1
            if j - i > 1:
                avg = ranks[order[i:j]].mean()
                ranks[order[i:j]] = avg
            i = j
    pct = (ranks - 1.0) / max(1.0, (x.size - 1.0))
    return pct if higher_is_better else (1.0 - pct)


# --------------------------
# Feature selection (fast ridge on reference labels)
# --------------------------
def select_top_features_fast_ridge(
    X: np.ndarray,         # (n,p_full)
    y_bin: np.ndarray,     # (n,)
    sw: np.ndarray,        # (n,)
    alpha: float,
    top_k: int,
    candidate_idx: Optional[np.ndarray] = None, # Fix #1
) -> np.ndarray:
    """
    Fit fast weighted ridge on y in {-1,+1}, return top_k by |coef|.
    If candidate_idx is provided, ONLY consider these features but return indices relative to full X.
    """
    if candidate_idx is not None:
        X_sub = X[:, candidate_idx]
        Xb = add_intercept(X_sub)
    else:
        Xb = add_intercept(X)

    y_pm1 = (2 * np.asarray(y_bin, dtype=np.int32) - 1).astype(np.float32)
    w = fast_ridge_fit_cholesky(Xb, y_pm1, sw, alpha=alpha, regularize_intercept=False)
    
    coef = w[:-1]  # exclude intercept
    local_idx = np.argsort(np.abs(coef))[::-1][:top_k]
    
    if candidate_idx is not None:
        return candidate_idx[local_idx].astype(np.int32, copy=False)
    return local_idx.astype(np.int32, copy=False)


# --------------------------
# Nested CV selection
# --------------------------
@dataclass
class GridResult:
    tp_mult: float
    sl_mult: float
    trail_mult: float
    lo: float
    hi: float
    z_max: float
    threshold_p: float
    inner_score: float
    inner_auc: float
    inner_ic: float
    inner_pnl: float
    win_rate: float
    trades: int
    tp_pct: float = 0.0
    sl_pct: float = 0.0
    timeout_pct: float = 0.0
    trades_per_month: float = 0.0
    t_stat: float = 0.0
    strategy_pnl: float = 0.0
    ev: float = 0.0
    net_pf: float = 0.0
    payoff: float = 0.0

@dataclass
class OuterFoldResult:
    fold: int
    chosen_tp_mult: float
    chosen_sl_mult: float
    chosen_trail_mult: float
    chosen_lo: float
    chosen_hi: float
    chosen_z_max: float
    chosen_threshold_p: float
    test_score: float
    test_auc: float
    test_ic: float
    test_pnl: float

@dataclass
class SelectionSummary:
    chosen_configs: List[Tuple[float, float, float, float, float, float, float]] # tp, sl, trail, lo, hi, z_max, thr_p
    outer_results: List[OuterFoldResult]
    final_tp_mult: float
    final_sl_mult: float
    final_trail_mult: float
    final_lo: float
    final_hi: float
    final_z_max: float
    final_threshold_p: float
    empirical_mfe_stats: Optional[Dict[str, float]] = None


def run_tp_sl_selection_fast(
    X: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    z: np.ndarray,
    atr_base_pct: np.ndarray,
    event_idx: np.ndarray,
    *,
    horizon: int,
    tp_mult_grid: Iterable[float] = (0.6, 0.8, 1.0, 1.25, 1.5),
    sl_mult_grid: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    trail_mult_grid: Iterable[float] = (0.05,),  # Tight trail (5% of barrier) — aligns backtest with triple-barrier assumption
    max_events: int = 5000,
    ridge_alpha: float = 0.5,
    top_k_features: int = 40,
    random_state: int = 42,
    lo: float = 0.03,
    hi: float = 0.06,
    z_max: float = 3.0,
    side: str = "long",
    entry_mode: str = "next_open",
    # AE weight bounds
    w_min: float = 0.5,
    w_max: float = 2.0,
    # Optional 15m OHLCV for higher-resolution event windows
    open_15m: Optional[np.ndarray] = None,
    high_15m: Optional[np.ndarray] = None,
    low_15m: Optional[np.ndarray] = None,
    close_15m: Optional[np.ndarray] = None,
    # New grids
    lo_grid: Optional[Iterable[float]] = None,
    hi_grid: Optional[Iterable[float]] = None,
    z_max_grid: Optional[Iterable[float]] = None,
    threshold_p_grid: Iterable[float] = (0.3, 0.4, 0.5),
    # Context for frequency logging
    n_assets: int = 1,
    n_months: float = 3.0,
    # Parallel time index for proper temporal sorting (same length as event_idx)
    # If None, falls back to event_idx (only correct if event_idx is time-monotonic)
    event_time_idx: Optional[np.ndarray] = None,
) -> SelectionSummary:
    """
    High-throughput variant:
      - Build event cache once (normalized return tensors)
      - Precompute barrier_pct once per (lo, hi, z_max) tuple
      - Per-grid: label once, precompute AE-until-exit once per grid (needed because exit depends on thresholds)
        (Note: AE itself is computed from prefix-min lows; cheap.)
      - Fit via Cholesky ridge (no sklearn model.fit overhead)
      - Class balancing is done via per-sample weights (balanced weights * AE weights)

    If you want to avoid recomputing AE-until-exit for every grid point:
      - you *can* approximate with AE over full horizon (but you explicitly asked for AE-until-exit correctness)
    """
    rng = np.random.default_rng(random_state)
    final_events = event_idx
    # Parallel time index: if provided, use for temporal sorting; else fall back to event_idx
    time_idx = np.asarray(event_time_idx, dtype=np.int64) if event_time_idx is not None else final_events.copy()
    if final_events.size > max_events:
        sel = rng.choice(final_events.size, size=max_events, replace=False)
        final_events = final_events[sel]
        time_idx = time_idx[sel]

    # Build event cache: use 15m data if available for higher-resolution labeling
    if open_15m is not None and high_15m is not None and low_15m is not None and close_15m is not None:
        tprint(f"  Using 15m precision for event cache (horizon={horizon}h -> {horizon*4} bars)")
        cache = build_event_cache_15m(
            open_15m=open_15m, high_15m=high_15m, low_15m=low_15m, close_15m=close_15m,
            event_idx_1h=final_events, horizon_1h=horizon, entry_mode=entry_mode, side=side
        )
    else:
        cache = build_event_cache(open_=open_, high=high, low=low, close=close,
                                  event_idx=final_events, horizon=horizon, entry_mode=entry_mode, side=side)

    e = cache.event_idx
    if e.size == 0:
        return SelectionSummary([], [], 1.0, 1.0, 0.5, lo, hi, z_max, 0.5, None)

    # build_event_cache may filter out invalid events (e.g. near array boundary).
    # We must apply the same filter to time_idx.
    # cache.event_idx is a subset of final_events; find which survived.
    # Since build_event_cache filters by `valid = (event_idx + 1 + HN) < n`,
    # the surviving indices are a prefix-mask of final_events. Reconstruct:
    if e.size < final_events.size:
        # Find which positions in final_events survived into cache
        survived = np.isin(final_events, e)
        time_idx = time_idx[survived]

    # CRITICAL: sort events by TIME (not by flat_idx which interleaves assets).
    # flat_idx = asset_offset + time_idx, so sorting by flat_idx groups by asset.
    # We must sort by time_idx to get proper temporal order for CV splitting.
    sort_order = np.argsort(time_idx)
    e = e[sort_order]
    time_idx = time_idx[sort_order]
    # Reorder ALL cache arrays to match sorted temporal order
    cache = EventCache(
        event_idx=e,
        entry_px=cache.entry_px[sort_order],
        rH=cache.rH[sort_order],
        rL=cache.rL[sort_order],
        rC_end=cache.rC_end[sort_order],
        rL_prefix_min=cache.rL_prefix_min[sort_order],
        rH_prefix_max=cache.rH_prefix_max[sort_order],
        horizon=cache.horizon,
        side=cache.side,
    )

    # Compute empirical MFE/MAE stats from the cache (before any labeling)
    mfe_stats = compute_empirical_mfe_stats(cache)
    tprint(f"  Empirical MFE: med={mfe_stats['mfe_median']*100:.2f}% p25={mfe_stats['mfe_p25']*100:.2f}% "
           f"p75={mfe_stats['mfe_p75']*100:.2f}% p90={mfe_stats['mfe_p90']*100:.2f}% | "
           f"MAE: med={mfe_stats['mae_median']*100:.2f}% p75={mfe_stats['mae_p75']*100:.2f}% (n={mfe_stats.get('n_events', 0)})")

    # trail_mult IS now searched: affects PT return via trailing-stop-aware labeling
    trail_mult_grid = list(trail_mult_grid)

    X_e_full = X[e].astype(np.float32, copy=False)

    atr_pct = _f32(atr_pct)
    z = _f32(z)
    atr_base_pct = _f32(atr_base_pct)

    # Defaults for grids if not provided
    if lo_grid is None: lo_grid = [lo]
    if hi_grid is None: hi_grid = [hi]
    if z_max_grid is None: z_max_grid = [z_max]

    # Deduplicate grids
    lo_grid = sorted(list(set(lo_grid)))
    hi_grid = sorted(list(set(hi_grid)))
    z_max_grid = sorted(list(set(z_max_grid)))
    tp_mult_grid = sorted(list(set(tp_mult_grid)))
    sl_mult_grid = sorted(list(set(sl_mult_grid)))
    threshold_p_grid = sorted(list(set(threshold_p_grid)))

    # Purge size: horizon hours of temporal gap between train and test
    safe_purge = int(horizon) + 24  # 48 bar-indices ≈ 2 days

    n_events = e.size
    # Outer CV: walk-forward expanding window (train always > test)
    # Split by time_idx (true temporal coordinate), not by e (flat_idx)
    outer_cv = WalkForwardCV(n_splits=3, purge=safe_purge, test_fraction=0.15,
                             min_train_size=max(50, n_events // 5))
    # Inner CV: PurgedKFold on the outer-train subset (sorted by time)
    inner_n_splits = 2 if n_events < 400 else 3
    inner_cv = PurgedKFold(n_splits=inner_n_splits, purge=safe_purge, embargo=2)

    outer_splits = list(outer_cv.split(time_idx))
    tprint(f"  CV setup: {n_events} events (sorted by time), outer=WalkForward({len(outer_splits)} folds, test~15%), "
           f"inner={inner_n_splits}-fold, purge={safe_purge}")
    for fi, (tr_i, te_i) in enumerate(outer_splits):
        tprint(f"    Fold {fi}: train={len(tr_i)} test={len(te_i)} "
               f"time_range=[{int(time_idx[tr_i[0]])}..{int(time_idx[tr_i[-1]])}] "
               f"test_time=[{int(time_idx[te_i[0]])}..{int(time_idx[te_i[-1]])}]")

    if not outer_splits:
        tprint(f"  WARNING: No valid outer folds for {n_events} events. Using conservative defaults.")
        return SelectionSummary([], [], 1.0, 1.0, 0.5, lo, hi, z_max, 0.5, mfe_stats)

    outer_results: List[OuterFoldResult] = []
    chosen_configs: List[Tuple[float, float, float, float, float, float]] = []

    # Pre-compute median barrier_pct for absolute value logging (issue #8: use actual scale, not raw ATR)
    _ref_barrier = scaled_atr_pct_dynamic_a(atr_pct=atr_pct[e], z=z[e], atr_base_pct=atr_base_pct[e],
                                             z_max=z_max, lo=lo, hi=hi)
    mean_atr = float(np.nanmedian(_ref_barrier)) if _ref_barrier.size > 0 else 0.03
    if np.isnan(mean_atr) or mean_atr <= 0:
        mean_atr = 0.03

    for ofold, (tr, te) in enumerate(outer_splits):
        X_tr_full = X_e_full[tr]
        X_te_full = X_e_full[te]

        # Reference labels for feature selection mixture (Consistency fix #8)
        # Using a union of (1.0, 1.0) and (0.8, 1.5) to pick features robust to different TP/SL
        feat_indices = []
        for (tp_r, sl_r) in [(1.0, 1.0), (0.8, 1.5)]:
            barrier_ref = scaled_atr_pct_dynamic_a(
                atr_pct=atr_pct[e], z=z[e], atr_base_pct=atr_base_pct[e],
                z_max=z_max, lo=lo, hi=hi
            )
            lab_ref = label_from_cache(cache, barrier_ref, tp_mult=tp_r, sl_mult=sl_r)
            ae_ref = compute_ae_until_exit_pct(cache, lab_ref.pt_t, lab_ref.sl_t)
            sw_ref = (ae_weight_multiplier(ae_ref, (sl_r * barrier_ref), w_min=w_min, w_max=w_max)[tr] * 
                      class_weight_balanced(lab_ref.y_bin[tr])).astype(np.float32, copy=False)

            feat_indices.append(select_top_features_fast_ridge(
                X=X_tr_full, y_bin=lab_ref.y_bin[tr], sw=sw_ref,
                alpha=ridge_alpha, top_k=top_k_features
            ))
        
        # Union the features (Consistency fix #4)
        feat_idx = np.unique(np.concatenate(feat_indices))
        if len(feat_idx) > top_k_features:
            # Re-select top_k from the union using the first reference label to maintain performance
            # We already have sw_ref for the last lab_ref. We'll use (1.0, 1.0) specifically if we can.
            # For simplicity, we'll just take the top_k by sorting the union or re-scoring.
            # Re-scoring is safer.
            feat_idx = select_top_features_fast_ridge(
                X=X_tr_full, y_bin=lab_ref.y_bin[tr], sw=sw_ref,
                alpha=ridge_alpha, top_k=top_k_features, candidate_idx=feat_idx
            )


        Xtr = X_tr_full[:, feat_idx]
        Xte = X_te_full[:, feat_idx]

        # Add intercept once (saves repeated concat)
        Xtr_b = add_intercept(Xtr)
        Xte_b = add_intercept(Xte)

        inner_splits = list(inner_cv.split(time_idx[tr])) # use TIME indices for temporal purging
        if not inner_splits:
            tprint(f"  Outer fold {ofold}: no valid inner splits (n_train={len(tr)}, n_test={len(te)}). Skipping.")
            continue
        
        # Adaptive min-N: scale with available OOF data
        # With threshold_p gating, only ~threshold_p fraction of OOF events are "active"
        # So min-N should be proportional to expected active count
        n_oof_expected = len(tr)  # OOF covers all outer-train events
        adaptive_min_n = max(10, min(30, int(n_oof_expected * 0.08)))  # ~8% of OOF, floor 10, cap 30

        grid_metrics: List[GridResult] = []
        seen_cfg5 = set()  # (z, lo, hi, tp, sl, trail) dedup for labeling
        seen_cfg6 = set()  # (z, lo, hi, tp, sl, trail, thr_p) dedup for scoring
        _diag = {"total_cfgs": 0, "skip_min_n": 0, "skip_sl_cap": 0, "passed": 0,
                 "max_trades_seen": 0, "min_sl_seen": 1.0}

        # Grid Search: tp_mult × sl_mult × trail_mult × vol params × threshold_p
        for z_val in z_max_grid:
            for lo_val in lo_grid:
                for hi_val in hi_grid:
                    # Recompute barrier pct for this config
                    barrier_pct = scaled_atr_pct_dynamic_a(
                        atr_pct=atr_pct[e],
                        z=z[e],
                        atr_base_pct=atr_base_pct[e],
                        z_max=z_val, lo=lo_val, hi=hi_val
                    )
                    # Compute mean barrier_pct for this vol config to enforce absolute constraints
                    _mean_bp = float(np.nanmedian(barrier_pct)) if barrier_pct.size > 0 else 0.04
                    if _mean_bp <= 0 or np.isnan(_mean_bp):
                        _mean_bp = 0.04

                    for tp_mult in tp_mult_grid:
                        # Constraint: absolute TP must be >= 2%
                        abs_tp = float(tp_mult) * _mean_bp
                        if abs_tp < 0.02:
                            continue

                        for sl_mult in sl_mult_grid:
                            # Constraint 1: Asymmetric TP > SL for positive expectancy
                            if float(tp_mult) <= float(sl_mult):
                                continue
                            
                            # Constraint 2: TP:SL ratio must be >= 1.5 (hard floor)
                            tp_sl_ratio = float(tp_mult) / max(float(sl_mult), 0.01)
                            if tp_sl_ratio < 1.5 or tp_sl_ratio > 5.0:
                                continue

                            for trail_val in trail_mult_grid:
                                # Deduplicate labeling (same labels for same tp/sl/trail/vol)
                                cfg_key_lab = (float(z_val), float(lo_val), float(hi_val), float(tp_mult), float(sl_mult), float(trail_val))
                                if cfg_key_lab in seen_cfg5:
                                    continue
                                seen_cfg5.add(cfg_key_lab)

                                # Label events for this TP/SL/Trail configuration
                                lab = label_from_cache(cache, barrier_pct, tp_mult=float(tp_mult), sl_mult=float(sl_mult), trail_mult=float(trail_val))

                                # AE until exit for this grid
                                ae = compute_ae_until_exit_pct(cache, lab.pt_t, lab.sl_t)

                                sl_pct = (float(sl_mult) * barrier_pct).astype(np.float32, copy=False)
                                w_ae = ae_weight_multiplier(ae, sl_pct, w_min=w_min, w_max=w_max)

                                # Strict OOF scoring on outer-train: collect inner-fold predictions
                                # and score each TP/SL grid point on combined OOF outputs.
                                oof_scores = np.full(tr.shape[0], np.nan, dtype=np.float32)
                                y_oof_all = np.full(tr.shape[0], -1, dtype=np.int32)
                                yr_oof_all = np.full(tr.shape[0], np.nan, dtype=np.float32)
                                k_oof_all = np.full(tr.shape[0], 0, dtype=np.int8)

                                for itr, ite in inner_splits:
                                    y_tr = lab.y_bin[tr][itr]
                                    # class balance on THIS inner-train
                                    w_cls = class_weight_balanced(y_tr)
                                    sw = (w_ae[tr][itr] * w_cls).astype(np.float32, copy=False)

                                    y_pm1 = (2 * y_tr.astype(np.int32) - 1).astype(np.float32, copy=False)

                                    w = fast_ridge_fit_cholesky(
                                        X=Xtr_b[itr],
                                        y_pm1=y_pm1,
                                        sw=sw,
                                        alpha=ridge_alpha,
                                        regularize_intercept=False,
                                    )

                                    oof_scores[ite] = fast_ridge_scores(Xtr_b[ite], w)
                                    y_oof_all[ite] = lab.y_bin[tr][ite]
                                    yr_oof_all[ite] = lab.y_ret[tr][ite]
                                    k_oof_all[ite] = lab.exit_kind[tr][ite]

                                valid_oof = np.isfinite(oof_scores) & np.isfinite(yr_oof_all)
                                if not np.any(valid_oof):
                                    continue
                                
                                y_oof = y_oof_all[valid_oof]
                                yr_oof = yr_oof_all[valid_oof]
                                s_oof = oof_scores[valid_oof]
                                k_oof = k_oof_all[valid_oof]
                                inner_auc = _auc_safe(y_oof, s_oof)
                                inner_ic = _spearman_ic(s_oof, yr_oof)
                                    
                                for thr_p in threshold_p_grid:
                                    cfg_key_score = (float(z_val), float(lo_val), float(hi_val), float(tp_mult), float(sl_mult), float(trail_val), float(thr_p))
                                    if cfg_key_score in seen_cfg6: continue
                                    seen_cfg6.add(cfg_key_score)

                                    # Net-after-fee Strategy Metrics
                                    m = _calculate_strategy_metrics(
                                        scores=s_oof,
                                        returns=yr_oof,
                                        labels=y_oof,
                                        exit_kinds=k_oof,
                                        threshold_p=float(thr_p),
                                        fee_bps=25.0,
                                    )
                                    t_stat = m["t_stat"]
                                    inner_pnl = m["net_pnl"]
                                    win_rate = m["wr"]
                                    trades = m["n_active"]
                                    tp_p = m["tp_p"]
                                    sl_p = m["sl_p"]
                                    to_p = m["to_p"]
                                    ev = m["ev"]
                                    net_pf = m["net_pf"]

                                    # Hard constraints: skip garbage configs
                                    _diag["total_cfgs"] += 1
                                    _diag["max_trades_seen"] = max(_diag["max_trades_seen"], trades)
                                    _diag["min_sl_seen"] = min(_diag["min_sl_seen"], sl_p)
                                    if trades < adaptive_min_n:
                                        _diag["skip_min_n"] += 1
                                        continue  # min-N constraint (adaptive)
                                    if sl_p > 0.70:
                                        _diag["skip_sl_cap"] += 1
                                        continue  # SL% cap: reject configs with >70% stop-loss rate
                                    _diag["passed"] += 1

                                    grid_metrics.append(GridResult(
                                        tp_mult=float(tp_mult),
                                        sl_mult=float(sl_mult),
                                        trail_mult=float(trail_val),
                                        lo=float(lo_val),
                                        hi=float(hi_val),
                                        z_max=float(z_val),
                                        threshold_p=float(thr_p),
                                        inner_score=0.0,
                                        inner_auc=float(inner_auc),
                                        inner_ic=float(inner_ic),
                                        inner_pnl=float(inner_pnl),
                                        win_rate=float(win_rate),
                                        trades=int(trades),
                                        tp_pct=float(tp_p),
                                        sl_pct=float(sl_p),
                                        timeout_pct=float(to_p),
                                        trades_per_month=float(trades) / (n_assets * n_months) if (n_assets * n_months) > 0 else 0.0,
                                        t_stat=float(t_stat),
                                        strategy_pnl=float(inner_pnl),
                                        ev=float(ev),
                                        net_pf=float(net_pf),
                                        payoff=float(m.get("payoff", 0.0)),
                                    ))


        if not grid_metrics:
            tprint(f"  Outer fold {ofold}: no configs passed hard constraints (min_n={adaptive_min_n}). "
                   f"Evaluated={_diag['total_cfgs']}, skip_min_n={_diag['skip_min_n']}, skip_sl_cap={_diag['skip_sl_cap']}, "
                   f"max_trades_seen={_diag['max_trades_seen']}, min_sl_seen={_diag['min_sl_seen']:.2f}. Skipping.")
            continue

        # Rank + composite score: proven scoring that finds positive-gross configs
        # Weights: 0.5 T-stat + 0.3 EV + 0.2 IC (original scoring that worked)
        # Hard constraints (min-N, SL% cap) already filter garbage configs above
        t_stats = np.array([g.t_stat for g in grid_metrics])
        ics = np.array([g.inner_ic for g in grid_metrics])
        evs = np.array([g.ev for g in grid_metrics])
        
        r_tval = _rank01(t_stats, True)
        r_ic = _rank01(ics, True)
        # Robust scaling for EV
        ev_scale = float(np.median(np.abs(evs)) + 1e-6)
        r_ev = _rank01(np.tanh(evs / ev_scale), True)
        
        comp = 0.5*r_tval + 0.3*r_ev + 0.2*r_ic
        for idx, res in enumerate(grid_metrics):
            res.inner_score = float(comp[idx])

        # Log top performers
        indices = np.argsort(comp)[::-1]
        tprint(f"[Fold {ofold}] Top Grid Results (sorted by Score):")
        tprint(f"  {'Rank':<4} {'Config (TP/SL/Trail/Thr)':<35} {'NetPnL':>9} {'AvgPnL':>9} {'PF':>5} {'WR':>6} {'Payoff':>6} {'AUC':>6} {'IC':>7} {'N':>4}")

        for k in range(min(5, len(indices))):
            idx = indices[k]
            res = grid_metrics[idx]
            # Calculate absolute avg distances for context
            abs_tp = res.tp_mult * mean_atr
            abs_sl = res.sl_mult * mean_atr

            cfg_str = f"TP={res.tp_mult:.2f} SL={res.sl_mult:.2f} Tr={res.trail_mult:.2f} Th={res.threshold_p:.2f}"

            # Detailed breakdown log
            # We want to show: NetPnL (total), AvgPnL (EV), Payoff (AvgWin/AvgLoss), WR, PF, AUC, IC
            tprint(f"  #{k+1:<3} {cfg_str:<35} {res.strategy_pnl:>+9.4f} {res.ev:>+9.6f} {res.net_pf:>5.2f} {res.win_rate:>6.2%} {res.payoff:>6.2f} {res.inner_auc:>6.3f} {res.inner_ic:>+7.4f} {res.trades:>4}")

            # Second line for TP/SL details if verbose or just top 1?
            if k == 0:
                tprint(f"       -> Abs: TP~{abs_tp:.2%} SL~{abs_sl:.2%} | Dist: TP={res.tp_pct:.0%} SL={res.sl_pct:.0%} TO={res.timeout_pct:.0%} | Score={comp[idx]:.4f}")

        best_i = int(np.argmax(comp))
        best_g = grid_metrics[best_i]
        best_inner_score = float(comp[best_i])
        tprint(f"[Fold {ofold}] Selected: TP={best_g.tp_mult:.2f} SL={best_g.sl_mult:.2f} Trail={best_g.trail_mult:.2f} "
               f"Lo={best_g.lo:.2f} Hi={best_g.hi:.2f} Thr={best_g.threshold_p:.2f} | InnerScore={best_inner_score:.4f}")
        chosen_configs.append((best_g.tp_mult, best_g.sl_mult, best_g.trail_mult, best_g.lo, best_g.hi, best_g.z_max, best_g.threshold_p))

        # Outer test evaluation with chosen grid
        barrier_pct_best = scaled_atr_pct_dynamic_a(
            atr_pct=atr_pct[e],
            z=z[e],
            atr_base_pct=atr_base_pct[e],
            z_max=best_g.z_max, lo=best_g.lo, hi=best_g.hi
        )
        lab_best = label_from_cache(cache, barrier_pct_best, best_g.tp_mult, best_g.sl_mult, trail_mult=best_g.trail_mult)
        ae_best = compute_ae_until_exit_pct(cache, lab_best.pt_t, lab_best.sl_t)

        sl_best = (best_g.sl_mult * barrier_pct_best).astype(np.float32, copy=False)
        w_ae_best = ae_weight_multiplier(ae_best, sl_best, w_min=w_min, w_max=w_max)

        # Class balance on outer-train for final fit
        w_cls_best = class_weight_balanced(lab_best.y_bin[tr])
        sw_best_tr = (w_ae_best[tr] * w_cls_best).astype(np.float32, copy=False)
        y_pm1_tr = (2 * lab_best.y_bin[tr].astype(np.int32) - 1).astype(np.float32, copy=False)

        w_final = fast_ridge_fit_cholesky(
            X=Xtr_b,
            y_pm1=y_pm1_tr,
            sw=sw_best_tr,
            alpha=ridge_alpha,
            regularize_intercept=False,
        )
        scores_te = fast_ridge_scores(Xte_b, w_final)

        yb_te = lab_best.y_bin[te]
        yr_te = lab_best.y_ret[te]
        
        # Guard inputs: NaN in returns or scores will poison metrics
        nan_scores = int(np.isnan(scores_te).sum())
        nan_rets = int(np.isnan(yr_te).sum())
        if nan_scores > 0 or nan_rets > 0:
            tprint(f"  [Fold {ofold}] Outer test has NaN inputs: {nan_scores} NaN scores, {nan_rets} NaN returns (of {len(yr_te)}). Cleaning.")
            valid_mask = ~(np.isnan(scores_te) | np.isnan(yr_te))
            scores_te = scores_te[valid_mask]
            yr_te = yr_te[valid_mask]
            yb_te = yb_te[valid_mask]
            exit_kinds_te = lab_best.exit_kind[te][valid_mask]
        else:
            exit_kinds_te = lab_best.exit_kind[te]
        
        test_auc = _auc_safe(yb_te, scores_te)
        test_ic = _spearman_ic(scores_te, yr_te)
        
        # Outer test metrics — use same fee-aware calculation
        tm = _calculate_strategy_metrics(
            scores=scores_te,
            returns=yr_te,
            labels=yb_te,
            exit_kinds=exit_kinds_te,
            threshold_p=best_g.threshold_p,
            fee_bps=25.0,
        )
        
        # Guard against NaN — if metrics are invalid, skip this fold
        test_net_pnl = tm["net_pnl"]
        test_ev = tm["ev"]
        test_n = tm["n_active"]
        if np.isnan(test_net_pnl) or np.isnan(test_ev) or test_n < 5:
            tprint(f"  [Fold {ofold}] Outer test invalid (NetPnL={test_net_pnl}, EV={test_ev}, N={test_n}). Skipping.")
            continue
        
        # Outer test score = the SAME scalar used for inner ranking
        # Use net_pnl directly (the thing we actually care about)
        test_score = float(test_net_pnl)
        
        tprint(f"  [Fold {ofold}] Outer test: AUC={test_auc:.4f} IC={test_ic:.4f} "
               f"NetPnL={test_net_pnl:+.4f} EV={test_ev:+.6f} N={test_n} | TestScore={test_score:+.4f}")

        outer_results.append(OuterFoldResult(
            fold=ofold,
            chosen_tp_mult=best_g.tp_mult,
            chosen_sl_mult=best_g.sl_mult,
            chosen_trail_mult=best_g.trail_mult,
            chosen_lo=best_g.lo,
            chosen_hi=best_g.hi,
            chosen_z_max=best_g.z_max,
            chosen_threshold_p=best_g.threshold_p,
            test_score=test_score,
            test_auc=float(test_auc),
            test_ic=float(test_ic),
            test_pnl=float(test_ev),
        ))

    # --- AGGREGATION ---
    if not outer_results:
        tprint("  WARNING: No valid outer folds. Using conservative defaults.")
        return SelectionSummary([], [], 1.0, 1.0, 0.5, lo, hi, z_max, 0.5, mfe_stats)

    # Debug table: show all configs and their per-fold scores
    tprint(f"\n  --- Aggregation Debug ({len(outer_results)} valid folds) ---")
    config_data: Dict[Tuple, List[Tuple[int, float]]] = {}  # cfg -> [(fold, score), ...]
    for r in outer_results:
        cfg = (float(r.chosen_tp_mult), float(r.chosen_sl_mult), float(r.chosen_trail_mult),
               float(r.chosen_lo), float(r.chosen_hi), float(r.chosen_z_max), float(r.chosen_threshold_p))
        if cfg not in config_data:
            config_data[cfg] = []
        config_data[cfg].append((r.fold, r.test_score))
    
    for cfg, fold_scores in config_data.items():
        scores_list = [s for _, s in fold_scores]
        folds_str = ", ".join([f"F{f}={s:+.4f}" for f, s in fold_scores])
        tprint(f"  Config TP={cfg[0]:.2f} SL={cfg[1]:.2f} Trail={cfg[2]:.2f} Lo={cfg[3]:.2f} Hi={cfg[4]:.2f}: "
               f"{folds_str} | mean={np.mean(scores_list):+.4f} n_folds={len(scores_list)}")

    # Selection strategy:
    # 1. If any config appears in >= 2 folds, pick the one with best mean score (stability)
    # 2. Otherwise, pick the config with the best single-fold test score
    multi_fold_cfgs = {cfg: fs for cfg, fs in config_data.items() if len(fs) >= 2}
    
    if multi_fold_cfgs:
        # Pick most stable config (appears in most folds, tiebreak by mean score)
        best_cfg = max(multi_fold_cfgs.keys(),
                       key=lambda c: (len(multi_fold_cfgs[c]), np.mean([s for _, s in multi_fold_cfgs[c]])))
        best_mean = float(np.mean([s for _, s in multi_fold_cfgs[best_cfg]]))
        tprint(f"  Selection: stable config (appears in {len(multi_fold_cfgs[best_cfg])} folds, mean={best_mean:+.4f})")
    else:
        # No config appears in multiple folds — pick best single-fold score
        best_r = max(outer_results, key=lambda r: r.test_score)
        best_cfg = (float(best_r.chosen_tp_mult), float(best_r.chosen_sl_mult), float(best_r.chosen_trail_mult),
                    float(best_r.chosen_lo), float(best_r.chosen_hi), float(best_r.chosen_z_max), float(best_r.chosen_threshold_p))
        best_mean = best_r.test_score
        tprint(f"  Selection: best single-fold (fold {best_r.fold}, score={best_mean:+.4f})")

    tprint(f"\nFinal Combined Selection (Score={best_mean:+.4f}):")
    tprint(f"  TP={best_cfg[0]:.2f}, SL={best_cfg[1]:.2f}, Trail={best_cfg[2]:.2f}, "
           f"Lo={best_cfg[3]:.2f}, Hi={best_cfg[4]:.2f}, Z={best_cfg[5]:.1f}, Thr={best_cfg[6]:.2f}")

    return SelectionSummary(
        chosen_configs=chosen_configs,
        outer_results=outer_results,
        final_tp_mult=best_cfg[0],
        final_sl_mult=best_cfg[1],
        final_trail_mult=best_cfg[2],
        final_lo=best_cfg[3],
        final_hi=best_cfg[4],
        final_z_max=best_cfg[5],
        final_threshold_p=best_cfg[6],
        empirical_mfe_stats=mfe_stats,
    )
