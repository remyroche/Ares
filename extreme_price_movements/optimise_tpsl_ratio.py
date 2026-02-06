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
        # X is usually event_idx values for temporal purging
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        idx = np.arange(n, dtype=np.int32)
        vals = np.asarray(X)

        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=np.int32)
        fold_sizes[: n % self.n_splits] += 1
        bounds = np.r_[0, fold_sizes.cumsum()]

        for k in range(self.n_splits):
            test_indices = idx[bounds[k]:bounds[k+1]]
            test_vals = vals[test_indices]
            
            t_min, t_max = test_vals.min(), test_vals.max()

            # Temporal purge: drop training events that overlap in time
            # purge/embargo are bar-distances from the test window
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
    Independent grids:
      tp_thr = tp_mult * barrier_pct
      sl_thr = sl_mult * barrier_pct
    Conditions:
      Long: PT = rH >= tp_thr, SL = rL <= -sl_thr
      Short: PT = rL <= -tp_thr, SL = rH >= sl_thr
    Pessimistic: ambiguous => SL hit
    """
    m = cache.entry_px.size
    HN = cache.horizon
    barrier_pct = _f32(barrier_pct)

    tp_thr = (float(tp_mult) * barrier_pct).astype(np.float32, copy=False)
    sl_thr = (float(sl_mult) * barrier_pct).astype(np.float32, copy=False)

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
    y_ret[pt_first] = tp_thr[pt_first]
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
    cost_linear: float = 0.0010, # 10bps base
    cost_quad: float = 0.0005,   # scaling with aggressiveness
) -> Dict[str, float]:
    """
    Robust strategy performance:
    1. Gating/Masking: Top-threshold scores, EXCLUDE ambiguous (exit_kind=2).
    2. Cost Model: impact = L * size + Q * size^2 (size normalized to traded signals).
    3. Metrics: T-stat and Expected Value (EV) net of costs.
    """
    scores = np.asarray(scores, dtype=np.float32)
    returns = np.asarray(returns, dtype=np.float32)
    
    n = len(scores)
    if n < 5:
        return {"t_stat": 0.0, "pnl": 0.0, "wr": 0.0, "tp_p": 0.0, "sl_p": 0.0, "to_p": 0.0, "n_active": 0, "ev": 0.0}

    # Gate: threshold by percentile
    abs_scores = np.abs(scores)
    thresh = np.percentile(abs_scores, 100.0 * (1.0 - threshold_p))
    mask = abs_scores >= thresh
    
    # EXCLUDE ambiguous (exit_kind=2) if diagnostic separation is desired (Fix #10)
    mask = mask & (exit_kinds != 2)

    n_active = int(mask.sum())
    if n_active < 2:
        return {"t_stat": 0.0, "pnl": 0.0, "wr": 0.0, "tp_p": 0.0, "sl_p": 0.0, "to_p": 0.0, "n_active": 0, "ev": 0.0}

    m_scores = abs_scores[mask]
    pos = np.sign(scores[mask])
    r_sub = returns[mask]
    l_sub = labels[mask]
    k_sub = exit_kinds[mask]

    # Realistic Sizing (Conceptual fix #6)
    # exposure scales with normalized score; costs also apply.
    s_min = m_scores.min()
    s_ptp = m_scores.ptp()
    size = (m_scores - s_min) / (s_ptp + 1e-12)
    impact = cost_linear * size + cost_quad * (size**2)
    
    # exposure is (pos * size)
    r_strat = (pos * size) * r_sub - impact

    mu = float(np.mean(r_strat))
    
    # Robust T-stat (Conceptual fix #8)
    # Add floor to std to avoid spurious huge t-stats at tiny horizons
    std = np.std(r_strat, ddof=1)
    std_floor = 0.25 * float(np.median(np.abs(r_strat)))
    std = max(float(std), float(std_floor), 1e-6)

    t_stat = mu / (std / np.sqrt(n_active) + 1e-12)
    pnl = mu * n_active # Total return across subset
    
    wr = float(l_sub.mean())
    tp_p = (k_sub == 1).mean()
    sl_p = (k_sub == -1).mean()
    to_p = (k_sub == 0).mean()
    
    return {
        "t_stat": float(t_stat),
        "pnl": float(pnl),
        "wr": float(wr),
        "tp_p": float(tp_p),
        "sl_p": float(sl_p),
        "to_p": float(to_p),
        "n_active": n_active,
        "ev": float(mu) # Expected Value net of cost
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

@dataclass
class OuterFoldResult:
    fold: int
    chosen_tp_mult: float
    chosen_sl_mult: float
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
    chosen_configs: List[Tuple[float, float, float, float, float, float]] # tp, sl, lo, hi, z_max, thr_p
    outer_results: List[OuterFoldResult]
    final_tp_mult: float
    final_sl_mult: float
    final_lo: float
    final_hi: float
    final_z_max: float
    final_threshold_p: float


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
    lo: float = 0.03,
    hi: float = 0.06,
    z_max: float = 3.0,
    side: str = "long",
    entry_mode: str = "next_open",
    # AE weight bounds
    w_min: float = 0.5,
    w_max: float = 2.0,
    # New grids
    lo_grid: Optional[Iterable[float]] = None,
    hi_grid: Optional[Iterable[float]] = None,
    z_max_grid: Optional[Iterable[float]] = None,
    threshold_p_grid: Iterable[float] = (0.2, 0.3, 0.4),
    # Context for frequency logging
    n_assets: int = 1,
    n_months: float = 3.0,
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
    if final_events.size > max_events:
        sel = rng.choice(final_events.size, size=max_events, replace=False)
        final_events = np.sort(final_events[sel])

    cache = build_event_cache(open_=open_, high=high, low=low, close=close,
                              event_idx=final_events, horizon=horizon, entry_mode=entry_mode, side=side)

    e = cache.event_idx
    if e.size == 0:
        return SelectionSummary([], [], 1.0, 1.0, lo, hi, z_max, 0.5)

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

    # Compute safe purge size approx based on horizon (assuming worst case 1 sample = 1 bar)
    # If events are sparse, this is conservative (good).
    safe_purge = int(horizon) + 24 # 24h buffer safety

    outer_cv = PurgedKFold(n_splits=3, purge=safe_purge, embargo=2)
    inner_cv = PurgedKFold(n_splits=3, purge=safe_purge, embargo=2)

    outer_results: List[OuterFoldResult] = []
    chosen_configs: List[Tuple[float, float, float, float, float, float]] = []

    # Important: PurgedKFold.split now uses event_idx for time-based purging (Fix #3)
    outer_splits = list(outer_cv.split(e)) # use event indices for temporal purging

    # Pre-compute mean ATR for absolute value logging
    # Using full event set `e` to get global average context
    mean_atr = np.mean(atr_pct) if len(atr_pct) > 0 else 0.01

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

        inner_splits = list(inner_cv.split(e[tr])) # use event indices for temporal purging

        grid_metrics: List[GridResult] = []
        unique_grid5 = set()
        unique_grid6 = set()

        # Extended Grid Search
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

                    for tp_mult in tp_mult_grid:
                        for sl_mult in sl_mult_grid:
                            
                            # Deduplicate check (if floating point drift makes them look different)
                            cfg_key5 = (float(z_val), float(lo_val), float(hi_val), float(tp_mult), float(sl_mult))
                            if cfg_key5 in unique_grid5:
                                continue
                            unique_grid5.add(cfg_key5)

                            lab = label_from_cache(cache, barrier_pct, tp_mult=float(tp_mult), sl_mult=float(sl_mult))

                            # AE until exit for this grid (uses prefix-min + exit indices)
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

                            valid_oof = np.isfinite(oof_scores)
                            if not np.any(valid_oof):
                                continue # Correctness fix #3: avoid fall-through to threshold loop with uninitialized vars
                            
                            y_oof = y_oof_all[valid_oof]
                            yr_oof = yr_oof_all[valid_oof]
                            s_oof = oof_scores[valid_oof]
                            k_oof = k_oof_all[valid_oof]
                            inner_auc = _auc_safe(y_oof, s_oof)
                            inner_ic = _spearman_ic(s_oof, yr_oof)
                                
                            for thr_p in threshold_p_grid:
                                # Deduplicate evaluated configs (Correctness fix #2)
                                cfg_key6 = (float(z_val), float(lo_val), float(hi_val), float(tp_mult), float(sl_mult), float(thr_p))
                                if cfg_key6 in unique_grid6: continue
                                unique_grid6.add(cfg_key6)

                                # Robust Strategy Metrics
                                m = _calculate_strategy_metrics(
                                    scores=s_oof,
                                    returns=yr_oof,
                                    labels=y_oof,
                                    exit_kinds=k_oof,
                                    threshold_p=float(thr_p)
                                )
                                t_stat = m["t_stat"]
                                inner_pnl = m["pnl"]
                                win_rate = m["wr"]
                                trades = m["n_active"]
                                tp_p = m["tp_p"]
                                sl_p = m["sl_p"]
                                to_p = m["to_p"]
                                ev = m["ev"]

                                grid_metrics.append(GridResult(
                                    tp_mult=float(tp_mult),
                                    sl_mult=float(sl_mult),
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
                                ))
                                grid_metrics[-1].ev = float(ev)


        # Rank + composite score across grid (Statistical Validity fix #6)
        # Weights: 0.5 T-stat, 0.3 Expected Value (EV), 0.2 IC
        t_stats = np.array([g.t_stat for g in grid_metrics])
        ics = np.array([g.inner_ic for g in grid_metrics])
        evs = np.array([g.ev for g in grid_metrics])
        
        r_tval = _rank01(t_stats, True)
        r_ic = _rank01(ics, True)
        
        # Robust scaling for EV (Conceptual fix #7)
        # Scale EV by a robust volatility estimate to make it dimensionless
        # and bound with tanh to prevent it from over-dominating.
        ev_scale = float(np.median(np.abs(evs)) + 1e-6)
        r_ev = _rank01(np.tanh(evs / ev_scale), True)
        
        comp = 0.5*r_tval + 0.3*r_ev + 0.2*r_ic
        for idx, res in enumerate(grid_metrics):
            res.inner_score = float(comp[idx])

        # Log top performers
        indices = np.argsort(comp)[::-1]
        tprint(f"[Fold {ofold}] Top Grid Results (sorted by Score):")
        for k in range(min(5, len(indices))):
            idx = indices[k]
            res = grid_metrics[idx]
            # Calculate absolute avg distances
            abs_tp = res.tp_mult * mean_atr
            abs_sl = res.sl_mult * mean_atr
            tprint(f"  #{k+1}: TP={res.tp_mult:.2f} ({abs_tp:.2%}) SL={res.sl_mult:.2f} ({abs_sl:.2%}) Lo={res.lo:.2f} Hi={res.hi:.2f} Z={res.z_max:.1f} Thr={res.threshold_p:.2f} | "
                   f"AUC={res.inner_auc:.4f} IC={res.inner_ic:.4f} EV={res.ev:.4f} T={res.t_stat:.2f} WR={res.win_rate:.2%} N={res.trades} | "
                   f"TP:{res.tp_pct:.1%}|SL:{res.sl_pct:.1%}|TO:{res.timeout_pct:.1%} | N/Mo:{res.trades_per_month:.1f} | Score={comp[idx]:.4f}")

        best_i = int(np.argmax(comp))
        best_g = grid_metrics[best_i]
        tprint(f"[Fold {ofold}] Selected: TP={best_g.tp_mult:.2f} SL={best_g.sl_mult:.2f} Lo={best_g.lo:.2f} Hi={best_g.hi:.2f} Thr={best_g.threshold_p:.2f}")
        chosen_configs.append((best_g.tp_mult, best_g.sl_mult, best_g.lo, best_g.hi, best_g.z_max, best_g.threshold_p))

        # Outer test evaluation with chosen grid
        barrier_pct_best = scaled_atr_pct_dynamic_a(
            atr_pct=atr_pct[e],
            z=z[e],
            atr_base_pct=atr_base_pct[e],
            z_max=best_g.z_max, lo=best_g.lo, hi=best_g.hi
        )
        lab_best = label_from_cache(cache, barrier_pct_best, best_g.tp_mult, best_g.sl_mult)
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
        test_auc = _auc_safe(yb_te, scores_te)
        test_ic = _spearman_ic(scores_te, yr_te)
        
        # Aligned Outer Metrics (Correctness fix #3: use selected best_g.threshold_p)
        tm = _calculate_strategy_metrics(
            scores=scores_te,
            returns=yr_te,
            labels=yb_te,
            exit_kinds=lab_best.exit_kind[te],
            threshold_p=best_g.threshold_p
        )
        test_tval = tm["t_stat"]
        test_ev = tm["ev"]
        
        # Outer score tracks inner logic: 0.5*tanh(t) + 0.3*tanh(ev*100) + 0.2*IC
        test_score = float(0.5 * np.tanh(test_tval) + 0.3 * np.tanh(test_ev * 100) + 0.2 * test_ic)

        outer_results.append(OuterFoldResult(
            fold=ofold,
            chosen_tp_mult=best_g.tp_mult,
            chosen_sl_mult=best_g.sl_mult,
            chosen_lo=best_g.lo,
            chosen_hi=best_g.hi,
            chosen_z_max=best_g.z_max,
            chosen_threshold_p=best_g.threshold_p,
            test_score=test_score,
            test_auc=float(test_auc),
            test_ic=float(test_ic),
            test_pnl=float(test_ev),
        ))

    # --- AGGREGATION: Maximize mean test_score across folds ---
    if not outer_results:
        return SelectionSummary([], [], 1.0, 1.0, lo, hi, z_max, 0.5)

    # Group results by config tuple
    config_scores: Dict[Tuple[float, float, float, float, float, float], List[float]] = {}
    for r in outer_results:
        cfg = (float(r.chosen_tp_mult), float(r.chosen_sl_mult), float(r.chosen_lo), float(r.chosen_hi), float(r.chosen_z_max), float(r.chosen_threshold_p))
        if cfg not in config_scores:
            config_scores[cfg] = []
        config_scores[cfg].append(r.test_score)
    
    # Pick cfg with max mean test_score
    best_cfg = (1.0, 1.0, lo, hi, z_max, 0.5)
    max_mean = -999.0
    for cfg, f_scores in config_scores.items():
        m_score = float(np.mean(f_scores))
        if m_score > max_mean:
            max_mean = m_score
            best_cfg = cfg

    tprint(f"\nFinal Combined Selection (Max Mean Test Score={max_mean:.4f}):")
    tprint(f"  TP={best_cfg[0]:.2f}, SL={best_cfg[1]:.2f}, Lo={best_cfg[2]:.2f}, Hi={best_cfg[3]:.2f}, Z={best_cfg[4]:.1f}, Thr={best_cfg[5]:.2f}")

    return SelectionSummary(
        chosen_configs=chosen_configs,
        outer_results=outer_results,
        final_tp_mult=best_cfg[0],
        final_sl_mult=best_cfg[1],
        final_lo=best_cfg[2],
        final_hi=best_cfg[3],
        final_z_max=best_cfg[4],
        final_threshold_p=best_cfg[5]
    )
