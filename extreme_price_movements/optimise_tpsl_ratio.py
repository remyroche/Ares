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
# Purged K-Fold (counts-based)
# --------------------------
class PurgedKFold:
    def __init__(self, n_splits: int = 5, purge: int = 5, embargo: int = 0, min_train_size: Optional[int] = None):
        if n_splits < 2:
            raise ValueError("n_splits must be >=2")
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.embargo = int(embargo)
        self.min_train_size = None if min_train_size is None else int(min_train_size)

    def split(self, X) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        idx = np.arange(n, dtype=np.int32)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int32)
        fold_sizes[: n % self.n_splits] += 1
        bounds = np.r_[0, fold_sizes.cumsum()]

        for k in range(self.n_splits):
            test_start = int(bounds[k])
            test_end = int(bounds[k + 1])

            pre_end = max(0, test_start - self.purge)
            post_start = min(n, test_end + self.embargo)

            train = np.r_[0:pre_end, post_start:n].astype(np.int32, copy=False)
            test = idx[test_start:test_end]

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
# ATR scaling (dynamic a)
# --------------------------
def scaled_atr_pct_dynamic_a(
    atr_pct: np.ndarray,
    z: np.ndarray,
    atr_base_pct: np.ndarray,
    *,
    z_max: float = 3.0,
    lo: float = 0.03,
    hi: float = 0.06,
    eps: float = 1e-12,
) -> np.ndarray:
    atr_pct = _f32(atr_pct)
    z = _f32(z)
    atr_base_pct = _f32(atr_base_pct)

    shock = np.clip(z, 0.0, z_max).astype(np.float32, copy=False)
    a = ((hi / np.maximum(atr_base_pct, eps)) - 1.0) / z_max
    a = a.astype(np.float32, copy=False)
    raw = atr_pct * (1.0 + a * shock)
    return np.clip(raw, lo, hi).astype(np.float32, copy=False)


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
    L_prefix_min: np.ndarray # (m, horizon) prefix-min of LOW prices (not returns) for AE
    horizon: int


def build_event_cache(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    event_idx: np.ndarray,
    horizon: int,
    entry_mode: str = "next_open",
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
            L_prefix_min=z,
            horizon=HN,
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

    # For AE-until-exit, we want prefix min of LOW prices (not returns) to compute AE precisely
    L_prefix_min = np.minimum.accumulate(L, axis=1).astype(np.float32, copy=False)

    return EventCache(
        event_idx=e.astype(np.int32, copy=False),
        entry_px=entry_px.astype(np.float32, copy=False),
        rH=rH.astype(np.float32, copy=False),
        rL=rL.astype(np.float32, copy=False),
        rC_end=rC_end.astype(np.float32, copy=False),
        L_prefix_min=L_prefix_min,
        horizon=HN,
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
) -> GridLabels:
    """
    Trailing Profit with Floor logic:
      Activation = tp_mult * barrier_pct
      Dist = sl_mult * barrier_pct

      Stop[t] depends on MaxHigh[0...t]:
        if MaxHigh >= Activation:
             Stop[t] = max(MaxHigh - Dist, Activation)
        else:
             Stop[t] = -Dist

      Exit if rL[t] <= Stop[t].

    Returns:
      y_bin = 1 if result > 0 (Profit)
      y_ret = realized return (Stop Price or Time Exit)
      pt_t = sentinel (unused/proxy for "Win" exit time)
      sl_t = Exit Time
    """
    m = cache.entry_px.size
    HN = cache.horizon
    barrier_pct = _f32(barrier_pct)

    tp_thr = (float(tp_mult) * barrier_pct).astype(np.float32, copy=False)
    sl_thr = (float(sl_mult) * barrier_pct).astype(np.float32, copy=False)

    # Pre-allocate output arrays
    sentinel = HN + 1

    # 1. Cumulative Max High (Long-like logic)
    # cache.rH is (m, H) normalized high returns
    max_h = np.maximum.accumulate(cache.rH, axis=1)

    # 2. Activation mask
    activated = max_h >= tp_thr[:, None]

    # 3. Calculate Stop Curve
    # Base trailing stop: MaxH - Dist
    trail_stop = max_h - sl_thr[:, None]

    # Apply logic:
    # If activated: max(trail_stop, floor=tp_thr)
    # If not activated: -sl_thr
    # We can use np.where

    stop_curve = np.where(
        activated,
        np.maximum(trail_stop, tp_thr[:, None]),
        -sl_thr[:, None]
    )

    # 4. Check Exit (Low <= Stop)
    hit_exit = cache.rL <= stop_curve

    # 5. Find First Exit
    exit_any = hit_exit.any(axis=1)
    exit_t = np.where(exit_any, hit_exit.argmax(axis=1), sentinel).astype(np.int32, copy=False)

    # 6. Determine Return
    # If exited: return is Stop Level at exit_t
    # If time exit: return is Close at end

    y_ret = np.zeros(m, dtype=np.float32)

    # Vectorized indexing for exited trades
    exited_mask = (exit_t < sentinel)
    if np.any(exited_mask):
        # We need to pick stop_curve[i, exit_t[i]]
        # Use simple indexing
        rows = np.where(exited_mask)[0]
        cols = exit_t[rows]

        realized_stops = stop_curve[rows, cols]
        y_ret[rows] = realized_stops

    # Time Exits
    time_mask = ~exited_mask
    if np.any(time_mask):
        y_ret[time_mask] = cache.rC_end[time_mask]

    # 7. Classify Outcome
    # Profit > 0 is a Win (y_bin=1)
    y_bin = (y_ret > 0).astype(np.uint8)

    # Construct "virtual" pt_t / sl_t for compatibility
    # If Win -> pt_t = exit_t, sl_t = sentinel
    # If Loss -> pt_t = sentinel, sl_t = exit_t

    pt_t = np.full(m, sentinel, dtype=np.int32)
    sl_t = np.full(m, sentinel, dtype=np.int32)

    wins = (y_bin == 1) & exited_mask
    losses = (y_bin == 0) & exited_mask

    pt_t[wins] = exit_t[wins]
    sl_t[losses] = exit_t[losses]

    exit_kind = np.zeros(m, dtype=np.int8)
    exit_kind[wins] = 1
    exit_kind[losses] = -1

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
    min_low = cache.L_prefix_min[rows, exit_t].astype(np.float32, copy=False)
    entry = cache.entry_px

    ae_pct = np.maximum(0.0, (entry - min_low) / np.maximum(entry, eps)).astype(np.float32, copy=False)
    return ae_pct


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
        c, low = la.cho_factor(xtwx, lower=True, check_finite=False)
        w = la.cho_solve((c, low), xtwy, check_finite=False)
        return w.astype(np.float32, copy=False)
    except la.LinAlgError:
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
    X: np.ndarray,         # (n,p)
    y_bin: np.ndarray,     # (n,)
    sw: np.ndarray,        # (n,)
    alpha: float,
    top_k: int,
) -> np.ndarray:
    """
    Fit fast weighted ridge on y in {-1,+1}, return top_k by |coef| (excluding intercept).
    """
    Xb = add_intercept(X)
    y_pm1 = (2 * np.asarray(y_bin, dtype=np.int32) - 1).astype(np.float32)
    w = fast_ridge_fit_cholesky(Xb, y_pm1, sw, alpha=alpha, regularize_intercept=False)
    coef = w[:-1]  # exclude intercept
    idx = np.argsort(np.abs(coef))[::-1][:top_k]
    return idx.astype(np.int32, copy=False)


# --------------------------
# Nested CV selection
# --------------------------
@dataclass
class GridResult:
    tp_mult: float
    sl_mult: float
    inner_score: float
    inner_auc: float
    inner_ic: float
    inner_pnl: float

@dataclass
class OuterFoldResult:
    fold: int
    chosen_tp_mult: float
    chosen_sl_mult: float
    test_score: float
    test_auc: float
    test_ic: float
    test_pnl: float

@dataclass
class SelectionSummary:
    chosen_pairs: List[Tuple[float, float]]
    outer_results: List[OuterFoldResult]
    final_tp_mult: float
    final_sl_mult: float


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
    max_events: int = 5000,
    ridge_alpha: float = 0.5,
    top_k_features: int = 40,
    outer_cv: Optional[PurgedKFold] = None,
    inner_cv: Optional[PurgedKFold] = None,
    random_state: int = 42,
    z_max: float = 3.0,
    lo: float = 0.03,
    hi: float = 0.06,
    entry_mode: str = "next_open",
    # AE weight bounds
    w_min: float = 0.5,
    w_max: float = 2.0,
) -> SelectionSummary:
    """
    High-throughput variant:
      - Build event cache once (normalized return tensors)
      - Precompute barrier_pct once
      - Per-grid: label once, precompute AE-until-exit once per grid (needed because exit depends on thresholds)
        (Note: AE itself is computed from prefix-min lows; cheap.)
      - Fit via Cholesky ridge (no sklearn model.fit overhead)
      - Class balancing is done via per-sample weights (balanced weights * AE weights)

    If you want to avoid recomputing AE-until-exit for every grid point:
      - you *can* approximate with AE over full horizon (but you explicitly asked for AE-until-exit correctness)
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=np.float32)
    event_idx = np.asarray(event_idx, dtype=np.int32)

    if event_idx.size > max_events:
        sel = rng.choice(event_idx.size, size=max_events, replace=False)
        event_idx = np.sort(event_idx[sel])

    cache = build_event_cache(open_=open_, high=high, low=low, close=close,
                              event_idx=event_idx, horizon=horizon, entry_mode=entry_mode)

    e = cache.event_idx
    if e.size == 0:
        return SelectionSummary([], [], 1.0, 1.0)

    X_e_full = X[e].astype(np.float32, copy=False)

    atr_pct = _f32(atr_pct)
    z = _f32(z)
    atr_base_pct = _f32(atr_base_pct)
    barrier_pct = scaled_atr_pct_dynamic_a(
        atr_pct=atr_pct[e],
        z=z[e],
        atr_base_pct=atr_base_pct[e],
        z_max=z_max, lo=lo, hi=hi
    )

    if outer_cv is None:
        outer_cv = PurgedKFold(n_splits=3, purge=5, embargo=2)
    if inner_cv is None:
        inner_cv = PurgedKFold(n_splits=3, purge=5, embargo=2)

    outer_results: List[OuterFoldResult] = []
    chosen_pairs: List[Tuple[float, float]] = []

    outer_splits = list(outer_cv.split(X_e_full))

    for ofold, (tr, te) in enumerate(outer_splits):
        X_tr_full = X_e_full[tr]
        X_te_full = X_e_full[te]

        # Reference labels for feature selection (tp=1, sl=1)
        ref = label_from_cache(cache, barrier_pct, tp_mult=1.0, sl_mult=1.0)

        # AE until exit for ref (computed once here)
        ae_ref = compute_ae_until_exit_pct(cache, ref.pt_t, ref.sl_t)

        sl_ref = (1.0 * barrier_pct).astype(np.float32, copy=False)
        w_ae_ref = ae_weight_multiplier(ae_ref, sl_ref, w_min=w_min, w_max=w_max)

        # class balanced weights on outer-train subset
        w_cls_ref = class_weight_balanced(ref.y_bin[tr])

        sw_ref_tr = (w_ae_ref[tr] * w_cls_ref).astype(np.float32, copy=False)

        # Feature selection (fast ridge), done on outer-train only
        feat_idx = select_top_features_fast_ridge(
            X=X_tr_full,
            y_bin=ref.y_bin[tr],
            sw=sw_ref_tr,
            alpha=ridge_alpha,
            top_k=top_k_features,
        )

        Xtr = X_tr_full[:, feat_idx]
        Xte = X_te_full[:, feat_idx]

        # Add intercept once (saves repeated concat)
        Xtr_b = add_intercept(Xtr)
        Xte_b = add_intercept(Xte)

        inner_splits = list(inner_cv.split(Xtr_b))

        grid_metrics: List[GridResult] = []
        # Cache labels per grid to avoid recomputing across inner folds
        # (Still per outer fold; memory ok for small grid)
        for tp_mult in tp_mult_grid:
            for sl_mult in sl_mult_grid:
                lab = label_from_cache(cache, barrier_pct, tp_mult=float(tp_mult), sl_mult=float(sl_mult))

                # AE until exit for this grid (uses prefix-min + exit indices)
                ae = compute_ae_until_exit_pct(cache, lab.pt_t, lab.sl_t)

                sl_pct = (float(sl_mult) * barrier_pct).astype(np.float32, copy=False)
                w_ae = ae_weight_multiplier(ae, sl_pct, w_min=w_min, w_max=w_max)

                aucs, ics, pnls = [], [], []
                for itr, ite in inner_splits:
                    y_tr = lab.y_bin[tr][itr]
                    y_te = lab.y_bin[tr][ite]
                    yr_te = lab.y_ret[tr][ite]

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

                    scores = fast_ridge_scores(Xtr_b[ite], w)
                    aucs.append(_auc_safe(y_te, scores))
                    ics.append(_spearman_ic(scores, yr_te))
                    pnls.append(_pnl_proxy(scores, yr_te))

                grid_metrics.append(GridResult(
                    tp_mult=float(tp_mult),
                    sl_mult=float(sl_mult),
                    inner_score=0.0,
                    inner_auc=float(np.mean(aucs)),
                    inner_ic=float(np.mean(ics)),
                    inner_pnl=float(np.mean(pnls)),
                ))

        # Rank + composite score across grid
        auc_arr = np.array([g.inner_auc for g in grid_metrics], dtype=np.float64)
        ic_arr = np.array([g.inner_ic for g in grid_metrics], dtype=np.float64)
        pnl_arr = np.array([g.inner_pnl for g in grid_metrics], dtype=np.float64)

        comp = 0.5 * _rank01(pnl_arr, True) + 0.3 * _rank01(ic_arr, True) + 0.2 * _rank01(auc_arr, True)
        best_i = int(np.argmax(comp))
        best_tp = float(grid_metrics[best_i].tp_mult)
        best_sl = float(grid_metrics[best_i].sl_mult)
        chosen_pairs.append((best_tp, best_sl))

        # Outer test evaluation with chosen grid
        lab_best = label_from_cache(cache, barrier_pct, best_tp, best_sl)
        ae_best = compute_ae_until_exit_pct(cache, lab_best.pt_t, lab_best.sl_t)

        sl_best = (best_sl * barrier_pct).astype(np.float32, copy=False)
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
        test_auc = _auc_safe(yb_te, scores_te)
        test_ic = _spearman_ic(scores_te, yr_te)
        test_pnl = _pnl_proxy(scores_te, yr_te)

        pnl_s = np.tanh(test_pnl)
        ic_s = 0.5 * (np.clip(test_ic, -1.0, 1.0) + 1.0)
        auc_s = np.clip(test_auc, 0.0, 1.0)
        test_score = float(0.5 * (0.5 * (pnl_s + 1.0)) + 0.3 * ic_s + 0.2 * auc_s)

        outer_results.append(OuterFoldResult(
            fold=ofold,
            chosen_tp_mult=best_tp,
            chosen_sl_mult=best_sl,
            test_score=test_score,
            test_auc=float(test_auc),
            test_ic=float(test_ic),
            test_pnl=float(test_pnl),
        ))

    # Mode over pairs (tuple-mode)
    pairs = np.array(chosen_pairs, dtype=np.float32)
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    best = uniq[np.argmax(counts)]
    final_tp, final_sl = float(best[0]), float(best[1])

    return SelectionSummary(
        chosen_pairs=chosen_pairs,
        outer_results=outer_results,
        final_tp_mult=final_tp,
        final_sl_mult=final_sl,
    )
