import math
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np


def alpha_objective_logloss(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None, eps: float = 1e-12) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    loss = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if w is None:
        return float(np.mean(loss))
    ww = np.asarray(w, dtype=float)
    s = ww.sum()
    return float(np.dot(loss, ww) / s) if s > 0 else float('nan')


def topk_mask(scores: np.ndarray, k_frac: float, groups: Optional[np.ndarray] = None) -> np.ndarray:
    scores = np.asarray(scores)
    n = scores.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    k_frac = float(k_frac)
    if not (0.0 < k_frac <= 1.0):
        raise ValueError("k_frac must be in (0,1]")
    if groups is None:
        k = max(1, int(math.ceil(k_frac * n)))
        idx = np.argpartition(scores, -k)[-k:]
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return m

    groups = np.asarray(groups)
    m = np.zeros(n, dtype=bool)
    for g in np.unique(groups):
        idxg = np.where(groups == g)[0]
        if idxg.size == 0:
            continue
        k = max(1, int(math.ceil(k_frac * idxg.size)))
        top_local = idxg[np.argpartition(scores[idxg], -k)[-k:]]
        m[top_local] = True
    return m


def precision_at_k(y: np.ndarray, scores: np.ndarray, k_frac: float, groups: Optional[np.ndarray] = None) -> float:
    y = (np.asarray(y) >= 0.5).astype(int)
    m = topk_mask(scores, k_frac, groups=groups)
    return float(y[m].mean()) if m.any() else float('nan')


def brier_at_mask(y: np.ndarray, p: np.ndarray, mask: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float('nan')
    e = (p[m] - y[m]) ** 2
    if w is None:
        return float(np.mean(e))
    ww = np.asarray(w, dtype=float)[m]
    s = ww.sum()
    return float(np.dot(e, ww) / s) if s > 0 else float('nan')


def ece_at_mask(y: np.ndarray, p: np.ndarray, mask: np.ndarray, n_bins: int = 10, w: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return float('nan')
    yy = y[m]
    pp = np.clip(p[m], 0.0, 1.0)
    ww = None if w is None else np.asarray(w, dtype=float)[m]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_w = float(ww.sum()) if ww is not None else float(pp.size)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (pp >= lo) & (pp < hi if i < n_bins - 1 else pp <= hi)
        if not np.any(in_bin):
            continue
        if ww is None:
            bw = float(np.sum(in_bin))
            conf = float(np.mean(pp[in_bin]))
            acc = float(np.mean(yy[in_bin]))
        else:
            wbin = ww[in_bin]
            bw = float(np.sum(wbin))
            if bw <= 0:
                continue
            conf = float(np.dot(pp[in_bin], wbin) / bw)
            acc = float(np.dot(yy[in_bin], wbin) / bw)
        ece += (bw / total_w) * abs(acc - conf)
    return float(ece)


def calibration_curve_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if not np.any(m):
            continue
        out.append({
            "bin": i,
            "n": int(np.sum(m)),
            "pred_mean": float(np.mean(p[m])),
            "actual_rate": float(np.mean(y[m])),
        })
    return out


def calibration_profile(curve_bins: List[Dict[str, float]], tol: float = 0.01) -> str:
    if len(curve_bins) < 2:
        return "flat"
    pred = np.array([b["pred_mean"] for b in curve_bins], dtype=float)
    act = np.array([b["actual_rate"] for b in curve_bins], dtype=float)
    if np.std(act) < 1e-3:
        return "flat"
    diff = float(np.mean(act - pred))
    if diff < -tol:
        return "overconfident"
    if diff > tol:
        return "underconfident/conservative"
    return "well-calibrated"


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt(np.sum(ra * ra) * np.sum(rb * rb))
    return float(np.sum(ra * rb) / denom) if denom > 0 else float('nan')


def ic_cross_sectional(scores: np.ndarray, rets: np.ndarray, groups: Optional[np.ndarray] = None) -> float:
    if groups is None:
        return spearman_corr(scores, rets)
    groups = np.asarray(groups)
    vals = []
    for g in np.unique(groups):
        idx = groups == g
        if np.sum(idx) >= 3:
            vals.append(spearman_corr(scores[idx], rets[idx]))
    return float(np.nanmean(vals)) if vals else float('nan')


def effective_sample_ratio(w: np.ndarray) -> float:
    ww = np.maximum(np.asarray(w, dtype=float), 0.0)
    if ww.size == 0:
        return float('nan')
    s = ww.sum()
    if s <= 0:
        return 0.0
    wn = ww / s
    neff = 1.0 / np.sum(np.square(wn))
    return float(neff / ww.size)


@dataclass(frozen=True)
class AlphaRankConfig:
    k_frac: float = 0.10
    cal_metric: str = "brier"
    cal_bins: int = 10
    neff_floor: float = 0.60
    w_ic: float = 0.45
    w_prec: float = 0.45
    w_cal: float = 0.05
    w_std: float = 0.05
    w_neff_pen: float = 0.10


def alpha_rank_components(y: np.ndarray, p: np.ndarray, rets: np.ndarray, w: Optional[np.ndarray], groups: Optional[np.ndarray], cfg: AlphaRankConfig) -> Dict[str, float]:
    ic = ic_cross_sectional(p, rets, groups=groups)
    prec = precision_at_k(y, p, cfg.k_frac, groups=groups)
    m = topk_mask(p, cfg.k_frac, groups=groups)
    cal = brier_at_mask(y, p, m, w=w) if cfg.cal_metric == "brier" else ece_at_mask(y, p, m, n_bins=cfg.cal_bins, w=w)
    if groups is None:
        std_ic = float('nan')
    else:
        vals = []
        gg = np.asarray(groups)
        for g in np.unique(gg):
            idx = gg == g
            if np.sum(idx) >= 3:
                vals.append(spearman_corr(p[idx], rets[idx]))
        std_ic = float(np.nanstd(vals)) if vals else float('nan')
    neff_ratio = effective_sample_ratio(w) if w is not None else float('nan')
    neff_pen = max(0.0, cfg.neff_floor - neff_ratio) if np.isfinite(neff_ratio) else 0.0
    return {"IC": ic, "Prec@K": prec, "Cal@K": cal, "StdIC": std_ic, "n_eff_pen": neff_pen}


def meta_objective_huber(y_util: np.ndarray, yhat: np.ndarray, w: Optional[np.ndarray] = None, delta: float = 1.0) -> float:
    y_util = np.asarray(y_util, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    r = y_util - yhat
    a = np.abs(r)
    quad = np.minimum(a, delta)
    lin = a - quad
    loss = 0.5 * quad**2 + delta * lin
    if w is None:
        return float(np.mean(loss))
    ww = np.asarray(w, dtype=float)
    s = ww.sum()
    return float(np.dot(loss, ww) / s) if s > 0 else float('nan')


def avg_trades_per_day(scores: np.ndarray, k_frac: float, timestamps: np.ndarray) -> float:
    ts = np.asarray(timestamps)
    m = topk_mask(scores, k_frac, groups=ts)
    if not np.any(m):
        return 0.0
    days = np.array([np.datetime64(t, 'D') for t in ts[m]])
    _, cnt = np.unique(days, return_counts=True)
    return float(np.mean(cnt)) if cnt.size else 0.0
