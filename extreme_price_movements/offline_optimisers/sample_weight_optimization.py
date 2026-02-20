"""Sample-weight optimization utilities for extreme_price_movements training."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import csv
import resource

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

from ..purged_cv import IntervalPurgedKFold
from ..utils import tprint
from ..config import CFG, TEST_FEATURE_KEYS
from ..training_defaults import get_sample_weight_opt_defaults, get_sample_weight_eval_model_defaults


EPS = 1e-8
KNOWN_BUCKETS = {"MR_long", "MR_short", "TF_long", "TF_short"}


def _process_memory_mb() -> float:
    """Best-effort current process RSS in MB."""
    try:
        rss_kb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # ru_maxrss is in bytes on macOS, KB on Linux.
        if rss_kb > 10_000_000:
            return rss_kb / (1024.0 * 1024.0)
        return rss_kb / 1024.0
    except Exception:
        return float("nan")


def _tprint_metrics(prefix: str, **metrics: float | int | str) -> None:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.6f}")
        else:
            parts.append(f"{k}={v}")
    tprint(f"{prefix} | {' '.join(parts)}")


def select_test_feature_frame(X_frame: pd.DataFrame) -> pd.DataFrame:
    """Return X_frame filtered to configured learnability-test features when available."""
    if not isinstance(X_frame, pd.DataFrame) or X_frame.empty:
        return X_frame
    test_keys = CFG.get("test_feature_keys", TEST_FEATURE_KEYS)
    keep = [c for c in test_keys if c in X_frame.columns]
    if keep:
        return X_frame.loc[:, keep]
    return X_frame


def _select_test_feature_matrix(
    X: np.ndarray | pd.DataFrame,
    feature_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Filter X to configured test_feature_keys when column names are available."""
    if isinstance(X, pd.DataFrame):
        return np.asarray(select_test_feature_frame(X), dtype=np.float32)

    X_arr = np.asarray(X, dtype=np.float32)
    if feature_names is None:
        return X_arr

    names = [str(c) for c in feature_names]
    test_keys = CFG.get("test_feature_keys", TEST_FEATURE_KEYS)
    keep_idx = [i for i, c in enumerate(names) if c in test_keys]
    if not keep_idx:
        return X_arr
    return X_arr[:, keep_idx]


def _standardize_feature_matrix(X: np.ndarray) -> np.ndarray:
    """In-place standardisation (zero mean / unit std) to avoid per-fold copies."""
    X_arr = np.asarray(X, dtype=np.float32)
    if X_arr.ndim != 2 or X_arr.size == 0:
        return X_arr

    mean = np.mean(X_arr, axis=0, keepdims=True, dtype=np.float32)
    std = np.std(X_arr, axis=0, keepdims=True, dtype=np.float32)
    std = np.maximum(std, np.float32(1e-6))
    X_arr -= mean
    X_arr /= std
    return X_arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def sample_weight_tp_classifier(
    atr_pct_past: np.ndarray,
    fee_rt: float = 0.002,
    k: float = 1.5,
    s: float | None = None,
    w_min: float = 0.4,
    dtype=np.float32,
) -> np.ndarray:
    m = np.asarray(atr_pct_past, dtype=np.float64)
    if s is None:
        s = 0.5 * float(fee_rt)
    s = max(float(s), 1e-12)
    gate = _sigmoid((m - (k * float(fee_rt))) / s)
    w = float(w_min) + (1.0 - float(w_min)) * gate
    return np.asarray(w, dtype=dtype)


def compute_alpha_from_train_fold(y_train: np.ndarray, q: float = 0.50) -> float:
    y = np.asarray(y_train, dtype=np.float64)
    abs_y = np.abs(y)
    alpha = float(np.quantile(abs_y, q)) if abs_y.size else 1e-12
    return max(alpha, 1e-12)


def sample_weight_meta_regression(
    y_ret_net: np.ndarray,
    atr_pct_past: np.ndarray | None = None,
    fee_rt: float = 0.002,
    k: float = 1.5,
    s: float | None = None,
    w_min: float = 0.4,
    alpha: float | None = None,
    alpha_quantile: float = 0.50,
    dtype=np.float32,
) -> np.ndarray:
    y = np.asarray(y_ret_net, dtype=np.float64)
    if s is None:
        s = 0.5 * float(fee_rt)
    s = max(float(s), 1e-12)

    if atr_pct_past is None:
        w_opp = np.ones(len(y), dtype=np.float64)
    else:
        m = np.asarray(atr_pct_past, dtype=np.float64)
        w_opp = _sigmoid((m - (k * float(fee_rt))) / s)

    abs_y = np.abs(y)
    if alpha is None:
        alpha = compute_alpha_from_train_fold(abs_y, q=float(alpha_quantile))
    alpha = max(float(alpha), 1e-12)
    w_tail = np.tanh(abs_y / alpha)

    w = float(w_min) + (1.0 - float(w_min)) * (w_opp * w_tail)
    return np.asarray(w, dtype=dtype)


def compute_n_eff(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0:
        return 0.0
    return (sw * sw) / max(float(np.sum(w * w)), EPS)


def combine_weights_safely(
    components: Dict[str, np.ndarray],
    component_weights: Dict[str, float],
    min_n_eff_ratio: float = 0.30,
    eps: float = 1e-6,
) -> np.ndarray:
    """Robustly combine weight components using clipped log-space geometric blending."""
    if not components:
        raise ValueError("components must not be empty")

    local_weights = component_weights.copy()
    clipped: Dict[str, np.ndarray] = {}
    template = None

    for name, w in components.items():
        arr = np.asarray(w, dtype=np.float32)
        p5 = float(np.nanpercentile(arr, 5))
        p95 = float(np.nanpercentile(arr, 95))
        span = p95 - p5
        ratio = p95 / (p5 + 1e-12)

        if not np.isfinite(span) or span < 1e-6 or ratio < 1.05:
            clipped[name] = np.ones_like(arr, dtype=np.float32)
            local_weights[name] = 0.0
        else:
            base = np.maximum(arr, eps, dtype=np.float32)
            clipped[name] = np.clip(base, p5, p95).astype(np.float32, copy=False)
        if template is None:
            template = clipped[name]

    log_w = np.zeros_like(template, dtype=np.float32)
    for name, w in clipped.items():
        alpha = float(local_weights.get(name, 1.0))
        if alpha == 0.0:
            continue
        log_w += np.float32(alpha) * np.log(w + eps, dtype=np.float32)

    w_final = np.exp(log_w, dtype=np.float32)
    if not np.all(np.isfinite(w_final)) or np.sum(w_final, dtype=np.float64) <= 0:
        w_final = np.ones_like(w_final, dtype=np.float32)

    target_n_eff = min_n_eff_ratio * len(w_final)
    n_eff = compute_n_eff(w_final)

    if n_eff < target_n_eff:
        # temperature flattening: w^(1/T), T>=1.0
        lo, hi = 1.0, 8.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            w_t = np.power(np.maximum(w_final, eps), 1.0 / mid, dtype=np.float32)
            w_t /= max(float(np.mean(w_t)), eps)
            if compute_n_eff(w_t) >= target_n_eff:
                hi = mid
            else:
                lo = mid
        w_final = np.power(np.maximum(w_final, eps), 1.0 / hi, dtype=np.float32)

    w_final /= max(float(np.mean(w_final, dtype=np.float64)), eps)
    return w_final


def _nanmedian_group(values: np.ndarray, inv: np.ndarray, group_count: int) -> np.ndarray:
    """Vectorized nanmedian per group using partial sorting for moderate group sizes."""
    if values.size == 0 or group_count == 0:
        return np.ones(group_count, dtype=np.float32)

    df = pd.DataFrame({"v": values, "g": inv})
    grouped = df.groupby("g")["v"].median()
    medians = grouped.reindex(range(group_count)).values

    # Fill NaNs/inf/<=0 with 1.0
    mask = np.isfinite(medians) & (medians > 0)
    medians = np.where(mask, medians, 1.0)
    return medians.astype(np.float32)


def compute_vol_weights(
    past_vol: np.ndarray,
    timestamps: np.ndarray,
    direction: str = "downweight_high",
    power: float = 0.5,
    min_group_size: int = 20,
) -> np.ndarray:
    vol = np.asarray(past_vol, dtype=np.float32)
    ts = pd.to_datetime(timestamps).values.astype("datetime64[ns]")
    if vol.size == 0:
        return np.ones(0, dtype=np.float32)

    ts_int = ts.astype("int64")
    unique_ts, inv = np.unique(ts_int, return_inverse=True)
    if unique_ts.size and min_group_size > 1:
        medians = _nanmedian_group(vol, inv, unique_ts.size)
    else:
        medians = np.ones(unique_ts.size, dtype=np.float32)

    denom = vol / np.maximum(medians[inv], np.float32(EPS))
    denom = np.maximum(denom, np.float32(EPS))
    if direction == "downweight_high":
        w = np.power(denom, -power, dtype=np.float32)
    else:
        w = np.power(denom, power, dtype=np.float32)
    return w / max(float(np.mean(w, dtype=np.float64)), EPS)


def compute_liquidity_weights(
    adv: np.ndarray,
    spread: np.ndarray | None = None,
    clip_range: tuple[float, float] = (0.7, 1.3),
) -> np.ndarray:
    if spread is not None:
        w = 1.0 / (np.asarray(spread, dtype=float) + EPS)
    else:
        w = np.log1p(np.maximum(np.asarray(adv, dtype=float), 0.0))
    w = np.clip(w, clip_range[0], clip_range[1])
    return w / max(float(np.mean(w)), EPS)


def compute_distance_to_barrier_weights(
    entry_prices: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    atr_past: np.ndarray,
    k: float = 0.5,
    min_dist: float = 0.5,
    form: str = "inverse",
) -> np.ndarray:
    entry = np.asarray(entry_prices, dtype=float)
    up = np.asarray(upper_barriers, dtype=float)
    lo = np.asarray(lower_barriers, dtype=float)
    atr = np.maximum(np.asarray(atr_past, dtype=float), EPS)

    dist_up = (up - entry) / atr
    dist_dn = (entry - lo) / atr
    dist_nearest = np.maximum(np.minimum(dist_up, dist_dn), min_dist)

    if form == "exp":
        w = np.exp(-k * dist_nearest)
    else:
        w = 1.0 / (dist_nearest + k)
    w = np.clip(w, 0.5, 2.0)
    return w / max(float(np.mean(w)), EPS)


def compute_recency_weights(
    bar_indices: np.ndarray,
    era_indices: np.ndarray,
    half_life_bars: int = 50,
    clip_range: tuple[float, float] = (0.5, 2.0),
    min_era_neff_ratio: float = 0.2,
) -> np.ndarray:
    bar_idx = np.asarray(bar_indices, dtype=float)
    max_idx = float(np.max(bar_idx)) if len(bar_idx) else 0.0
    age_bars = max_idx - bar_idx
    w = np.power(2.0, -age_bars / max(float(half_life_bars), 1.0))
    w = np.clip(w, clip_range[0], clip_range[1])

    eras = np.asarray(era_indices)
    if eras.size:
        _, inv = np.unique(eras, return_inverse=True)
        w_sum = np.bincount(inv, weights=w)
        w_sq_sum = np.bincount(inv, weights=w * w)
        n_samples_per_era = np.bincount(inv)
        era_neff = (w_sum ** 2) / np.maximum(w_sq_sum, EPS)
        min_expected = float(np.min(n_samples_per_era * min_era_neff_ratio)) if n_samples_per_era.size else 0.0
        if era_neff.size and float(np.min(era_neff)) < min_expected:
            w = np.power(w, 0.5)

    return w / max(float(np.mean(w)), EPS)


def check_component_redundancy(
    components: Dict[str, np.ndarray],
    threshold: float = 0.85,
) -> Dict[str, Any]:
    names = list(components.keys())
    arrays = [np.asarray(components[n], dtype=float) for n in names]
    redundant: list[tuple[str, str, float]] = []
    max_corr = 0.0

    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            r, _ = spearmanr(arrays[i], arrays[j])
            r = float(0.0 if not np.isfinite(r) else r)
            max_corr = max(max_corr, abs(r))
            if abs(r) > threshold:
                redundant.append((names[i], names[j], r))

    return {"pairs": redundant, "max_corr": max_corr}


def log_weight_statistics(weights: np.ndarray, era_indices: np.ndarray, name: str) -> Dict[str, float]:
    w = np.asarray(weights, dtype=float)
    eras = np.asarray(era_indices)
    if eras.size:
        _, inv = np.unique(eras, return_inverse=True)
        w_sum = np.bincount(inv, weights=w)
        w_sq_sum = np.bincount(inv, weights=w * w)
        era_neff = (w_sum ** 2) / np.maximum(w_sq_sum, EPS)
        era_neff_min = float(np.min(era_neff)) if era_neff.size else 0.0
        era_neff_mean = float(np.mean(era_neff)) if era_neff.size else 0.0
    else:
        era_neff_min = 0.0
        era_neff_mean = 0.0

    top1_k = max(1, int(len(w) * 0.01))
    total = max(np.sum(w), EPS)
    stats = {
        "mean": float(np.mean(w)),
        "std": float(np.std(w)),
        "p1": float(np.percentile(w, 1)),
        "p5": float(np.percentile(w, 5)),
        "p50": float(np.percentile(w, 50)),
        "p95": float(np.percentile(w, 95)),
        "p99": float(np.percentile(w, 99)),
        "max": float(np.max(w)),
        "n_eff": float(compute_n_eff(w)),
        "top1pct_share": float(np.sort(w)[-top1_k:].sum() / total),
        "era_neff_min": era_neff_min,
        "era_neff_mean": era_neff_mean,
    }
    tprint(f"{name} weights | " + " ".join([f"{k}={v:.4f}" for k, v in stats.items()]))
    return stats


def _make_model(model_family: str, random_state: int, cfg_runtime: Optional[Dict[str, Any]] = None):
    fam = model_family.lower()
    model_defaults = get_sample_weight_eval_model_defaults(cfg_runtime if cfg_runtime is not None else CFG)
    et_defaults = dict(model_defaults.get("extratrees", {}))
    rf_defaults = dict(model_defaults.get("randomforest", {}))

    if fam == "randomforest":
        return RandomForestRegressor(**rf_defaults, random_state=random_state)

    # ExtraTrees is the canonical fallback to match training-side optimizer defaults.
    return ExtraTreesRegressor(**et_defaults, random_state=random_state)


def _safe_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ic, _ = spearmanr(y_true, y_pred)
    if not np.isfinite(ic):
        return 0.0
    return float(ic)


def _decile_spread(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    m = np.isfinite(y) & np.isfinite(s)
    if int(np.sum(m)) < 20:
        return 0.0
    y = y[m]
    s = s[m]
    try:
        dec = pd.qcut(pd.Series(s), 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"d": dec, "y": y}).groupby("d")["y"].mean()
        if len(g) < 2:
            return 0.0
        return float(g.iloc[-1] - g.iloc[0])
    except Exception:
        return 0.0


def _run_cv_ic(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    label_intervals: np.ndarray,
    model_family: str,
    n_splits: int = 2,  # 2 for stage1, 3 for stage2, 4 for stage3
    embargo_bars: int = 10,  # Default value added
    seed: int = 42,  # Default value added
    cfg_runtime: Optional[Dict[str, Any]] = None,
    bucket_codes: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = IntervalPurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    fold_ics = []
    fold_spreads = []
    Xs = np.asarray(X, dtype=np.float32)
    yv = np.asarray(y, dtype=float)
    _yv_finite = np.isfinite(yv)
    if not _yv_finite.all():
        _yv_fill = float(np.nanmedian(yv[_yv_finite])) if _yv_finite.any() else 0.0
        yv = np.where(_yv_finite, yv, _yv_fill)
    wv = np.asarray(sample_weight, dtype=float)
    bv = None if bucket_codes is None else np.asarray(bucket_codes)

    base_model = _make_model(model_family, random_state=seed, cfg_runtime=cfg_runtime)
    for fold_idx, (tr, va) in enumerate(splitter.split(Xs, label_intervals=label_intervals), start=1):
        if len(tr) < 64 or len(va) < 32:
            _tprint_metrics(
                "CV fold skipped",
                fold=fold_idx,
                train_size=len(tr),
                valid_size=len(va),
                mem_mb=_process_memory_mb(),
            )
            continue
        if bv is None:
            model = clone(base_model)
            model.fit(Xs[tr], yv[tr], sample_weight=wv[tr])
            pred = model.predict(Xs[va])
            fold_ic = _safe_ic(yv[va], pred)
            fold_spread = _decile_spread(yv[va], pred)
            buckets_scored = 1
        else:
            pred_full = np.full(len(va), np.nan, dtype=float)
            va_b = bv[va]
            tr_b = bv[tr]
            buckets_scored = 0
            for b in np.unique(tr_b):
                tr_mask = tr_b == b
                va_mask = va_b == b
                tr_idx = tr[tr_mask]
                va_idx = va[va_mask]
                if len(tr_idx) < 64 or len(va_idx) < 32:
                    continue
                model = clone(base_model)
                model.fit(Xs[tr_idx], yv[tr_idx], sample_weight=wv[tr_idx])
                pred_full[va_mask] = model.predict(Xs[va_idx])
                buckets_scored += 1
            valid_mask = np.isfinite(pred_full)
            if int(np.sum(valid_mask)) < 32:
                _tprint_metrics(
                    "CV fold skipped",
                    fold=fold_idx,
                    reason="insufficient_bucket_predictions",
                    train_size=len(tr),
                    valid_size=len(va),
                    buckets_scored=buckets_scored,
                    mem_mb=_process_memory_mb(),
                )
                continue
            pred = pred_full[valid_mask]
            fold_ic = _safe_ic(yv[va][valid_mask], pred)
            fold_spread = _decile_spread(yv[va][valid_mask], pred)
        fold_ics.append(fold_ic)
        fold_spreads.append(fold_spread)
        _tprint_metrics(
            "CV fold complete",
            fold=fold_idx,
            seed=seed,
            buckets_scored=buckets_scored,
            train_size=len(tr),
            valid_size=len(va),
            ic=fold_ic,
            decile_spread=fold_spread,
            pred_mean=float(np.mean(pred)),
            pred_std=float(np.std(pred)),
            valid_ret_mean=float(np.mean(yv[va])),
            weighted_valid_ret_mean=float(np.average(yv[va], weights=wv[va])) if len(va) else 0.0,
            mem_mb=_process_memory_mb(),
        )

    return np.asarray(fold_ics, dtype=float), np.asarray(fold_spreads, dtype=float)


def constrained_objective(
    w: np.ndarray,
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    model_family: str,
    n_splits: int = 2,  # 2 for stage1, 3 for stage2, 4 for stage3
    embargo_bars: int = 10,  # Default value added
    seeds: Iterable[int] = (42, 123),
    min_n_eff_ratio: float = 0.30,
    max_top1pct: float = 0.10,
    cfg_runtime: Optional[Dict[str, Any]] = None,
    bucket_codes: Optional[np.ndarray] = None,
) -> float:
    n_eff = compute_n_eff(w)
    min_n_eff = min_n_eff_ratio * len(w)
    k = max(1, int(np.ceil(len(w) * 0.01)))
    top1pct_share = float(np.sort(w)[-k:].sum() / max(float(np.sum(w)), EPS))

    if n_eff < min_n_eff or top1pct_share > max_top1pct:
        _tprint_metrics(
            "Objective rejected",
            n_eff=n_eff,
            min_n_eff=min_n_eff,
            n_eff_ratio=n_eff / max(len(w), 1),
            top1pct_share=top1pct_share,
            max_top1pct=max_top1pct,
            weighted_return_mean=float(np.average(y_ret, weights=w)) if len(y_ret) else 0.0,
            mem_mb=_process_memory_mb(),
        )
        return -10.0

    fold_scores = []
    spread_scores = []
    for seed in seeds:
        fold_ics, fold_spreads = _run_cv_ic(
            X, y_ret, w, label_intervals, model_family=model_family,
            n_splits=n_splits, embargo_bars=embargo_bars, seed=int(seed), cfg_runtime=cfg_runtime,
            bucket_codes=bucket_codes,
        )
        if fold_ics.size:
            fold_scores.append(float(np.mean(fold_ics)))
        if fold_spreads.size:
            spread_scores.append(float(np.mean(fold_spreads)))
    if not fold_scores:
        _tprint_metrics("Objective invalid", reason="no_fold_scores", mem_mb=_process_memory_mb())
        return -10.0
    ic_mean = float(np.mean(fold_scores))
    ic_std = float(np.std(fold_scores))
    spread_mean = float(np.mean(spread_scores)) if spread_scores else 0.0
    objective = ic_mean - 0.5 * ic_std
    _tprint_metrics(
        "Objective evaluated",
        n_eff=n_eff,
        n_eff_ratio=n_eff / max(len(w), 1),
        top1pct_share=top1pct_share,
        ic_mean=ic_mean,
        ic_std=ic_std,
        decile_spread_mean=spread_mean,
        objective=objective,
        weighted_return_mean=float(np.average(y_ret, weights=w)) if len(y_ret) else 0.0,
        mem_mb=_process_memory_mb(),
    )
    return objective


@dataclass
class WeightOptimizationResult:
    optimized_weights: np.ndarray
    component_alphas: Dict[str, float]
    objective_value: float
    diagnostics: Dict[str, Any]


def optimize_component_weights(
    X: np.ndarray | pd.DataFrame,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    components: Dict[str, np.ndarray],
    production_model: str = "ExtraTrees",
    n_trials: int = 20,
    n_splits: int = 2,  # 2 for stage1, 3 for stage2, 4 for stage3
    embargo_bars: int = 10,
    min_n_eff_ratio: float = 0.30,
    max_top1pct: float = 0.10,
    random_state: int = 42,
    feature_names: Optional[Iterable[str]] = None,
    cfg_runtime: Optional[Dict[str, Any]] = None,
    bucket_codes: Optional[np.ndarray] = None,
) -> WeightOptimizationResult:
    if not components:
        ones = np.ones(len(y_ret), dtype=float)
        return WeightOptimizationResult(ones, {}, -10.0, {"reason": "no_components"})

    X = _select_test_feature_matrix(X, feature_names=feature_names)
    X = _standardize_feature_matrix(X)
    names = list(components.keys())
    _tprint_metrics(
        "Weight optimisation started",
        n_samples=len(y_ret),
        n_features=int(X.shape[1]) if X.ndim == 2 else 0,
        n_components=len(names),
        n_trials=n_trials,
        model=production_model,
        mem_mb=_process_memory_mb(),
    )
    if optuna is None or n_trials <= 0:
        alphas = {k: 1.0 for k in names}
        w = combine_weights_safely(components, alphas, min_n_eff_ratio=min_n_eff_ratio)
        val = constrained_objective(
            w, X, y_ret, label_intervals,
            model_family=production_model,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            min_n_eff_ratio=min_n_eff_ratio,
            max_top1pct=max_top1pct,
            cfg_runtime=cfg_runtime,
            bucket_codes=bucket_codes,
        )
        return WeightOptimizationResult(w, alphas, val, {"fallback": "optuna_unavailable_or_disabled"})

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _obj(trial: "optuna.trial.Trial") -> float:
        alphas = {n: trial.suggest_float(f"alpha_{n}", 0.0, 1.5) for n in names}
        
        # Determine constraints: allow Optuna to dynamically search these if they aren't strictly locked
        trial_n_eff = trial.suggest_float("min_n_eff_ratio", 0.10, 0.90) if cfg_runtime is None or "sample_weight_opt_min_n_eff_ratio" not in cfg_runtime else min_n_eff_ratio
        trial_top1pct = trial.suggest_float("max_top1pct", 0.05, 0.20) if cfg_runtime is None or "sample_weight_opt_max_top1pct" not in cfg_runtime else max_top1pct
        
        w = combine_weights_safely(components, alphas, min_n_eff_ratio=trial_n_eff)
        score = constrained_objective(
            w, X, y_ret, label_intervals,
            model_family=production_model,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            min_n_eff_ratio=trial_n_eff,
            max_top1pct=trial_top1pct,
            cfg_runtime=cfg_runtime,
            bucket_codes=bucket_codes,
        )
        learnability = _weight_learnability_metrics(w, y_ret)
        _tprint_metrics(
            "Optuna trial",
            trial=trial.number,
            score=score,
            n_eff=learnability["n_eff"],
            n_eff_ratio=learnability["n_eff_ratio"],
            top1pct_weight_share=learnability["top1pct_weight_share"],
            weighted_return_mean=learnability["weighted_return_mean"],
            weight_std=learnability["weight_std"],
            mem_mb=_process_memory_mb(),
        )
        return float(score)

    study.optimize(_obj, n_trials=int(n_trials), show_progress_bar=False)

    best_params = study.best_params if len(study.trials) else {}
    alphas = {n: float(best_params.get(f"alpha_{n}", 1.0)) for n in names}
    best_n_eff = float(best_params.get("min_n_eff_ratio", min_n_eff_ratio))
    optimized = combine_weights_safely(components, alphas, min_n_eff_ratio=best_n_eff)

    diagnostics = {
        "best_trial": int(study.best_trial.number) if study.best_trial is not None else -1,
        "n_trials": len(study.trials),
        "redundancy": check_component_redundancy(components),
    }
    _tprint_metrics(
        "Weight optimisation complete",
        best_trial=diagnostics["best_trial"],
        n_trials=diagnostics["n_trials"],
        best_value=float(study.best_value),
        mem_mb=_process_memory_mb(),
    )
    return WeightOptimizationResult(optimized, alphas, float(study.best_value), diagnostics)


def run_ablation(
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    components: Dict[str, np.ndarray],
    baseline_weights: np.ndarray,
    production_model: str = "ExtraTrees",
    n_splits: int = 3,  # Reduced from 5 for memory efficiency
    embargo_bars: int = 10,
    cfg_runtime: Optional[Dict[str, Any]] = None,
    enable: bool = True,
    bucket_codes: Optional[np.ndarray] = None,
) -> list[tuple[str, float]]:
    _tprint_metrics("Ablation started", n_components=len(components), mem_mb=_process_memory_mb())
    if not enable or not components:
        return [("baseline", float("nan"))]
    X_std = _standardize_feature_matrix(X)
    results = []
    base_score = constrained_objective(
        baseline_weights, X_std, y_ret, label_intervals,
        model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
        cfg_runtime=cfg_runtime, bucket_codes=bucket_codes,
    )
    results.append(("baseline", float(base_score)))
    _tprint_metrics("Ablation baseline", score=float(base_score), mem_mb=_process_memory_mb())

    baseline_arr = np.asarray(baseline_weights, dtype=float)
    scratch = baseline_arr.copy()
    comp_arrays = {k: np.asarray(v, dtype=float) for k, v in components.items()}

    for name, w_comp in comp_arrays.items():
        np.multiply(baseline_arr, w_comp, out=scratch)
        scratch /= max(float(np.mean(scratch)), EPS)
        score = constrained_objective(
            scratch, X_std, y_ret, label_intervals,
            model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
            cfg_runtime=cfg_runtime, bucket_codes=bucket_codes,
        )
        results.append((name, float(score)))
        _tprint_metrics("Ablation single component", component=name, score=float(score), mem_mb=_process_memory_mb())
        scratch[:] = baseline_arr

    for n1, n2 in combinations(comp_arrays.keys(), 2):
        np.multiply(baseline_arr, comp_arrays[n1], out=scratch)
        np.multiply(scratch, comp_arrays[n2], out=scratch)
        scratch /= max(float(np.mean(scratch)), EPS)
        score = constrained_objective(
            scratch, X_std, y_ret, label_intervals,
            model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
            cfg_runtime=cfg_runtime, bucket_codes=bucket_codes,
        )
        results.append((f"{n1}+{n2}", float(score)))
        _tprint_metrics("Ablation pair", component=f"{n1}+{n2}", score=float(score), mem_mb=_process_memory_mb())
        scratch[:] = baseline_arr

    return sorted(results, key=lambda x: x[1], reverse=True)


def _default_sample_weight_best_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    sw_defaults = get_sample_weight_opt_defaults(cfg)
    default_alphas = dict(cfg.get("sample_weight_component_alphas", {})) if isinstance(cfg.get("sample_weight_component_alphas"), dict) else {}
    return {
        "sample_weight_vol_power": float(sw_defaults["sample_weight_vol_power"]),
        "sample_weight_distance_k": float(sw_defaults["sample_weight_distance_k"]),
        "sample_weight_distance_min_dist": float(sw_defaults["sample_weight_distance_min_dist"]),
        "sample_weight_recency_half_life_bars": int(sw_defaults["sample_weight_recency_half_life_bars"]),
        "component_alphas": default_alphas,
        "component_alphas_base": dict(cfg.get("sample_weight_component_alphas_base", default_alphas)) if isinstance(cfg.get("sample_weight_component_alphas_base", default_alphas), dict) else default_alphas,
        "component_alphas_meta": dict(cfg.get("sample_weight_component_alphas_meta", default_alphas)) if isinstance(cfg.get("sample_weight_component_alphas_meta", default_alphas), dict) else default_alphas,
    }


def _detect_stage_column(df: pd.DataFrame) -> str | None:
    for c in ("stage", "model_stage", "split", "dataset"):
        if c in df.columns:
            return c
    return None


def _detect_bucket_column(df: pd.DataFrame) -> str | None:
    for c in ("bucket", "slice_bucket", "model_bucket"):
        if c in df.columns:
            return c
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(float) > 0.0
    as_str = s.astype(str).str.strip().str.lower()
    return as_str.isin({"1", "true", "t", "yes", "y"})


def _apply_hard_candidate_prefilter(df: pd.DataFrame, bucket_col: str | None = None) -> pd.DataFrame:
    for c in ("candidate_mask", "is_candidate", "candidate", "is_trade_candidate"):
        if c in df.columns:
            mask = _to_bool_series(df[c])
            out = df.loc[mask].copy()
            _tprint_metrics(
                "Hard candidate prefilter applied",
                source_col=c,
                kept=len(out),
                total=len(df),
                keep_rate=float(len(out) / max(len(df), 1)),
                mem_mb=_process_memory_mb(),
            )
            return out

    if bucket_col and bucket_col in df.columns:
        mask = df[bucket_col].astype(str).isin(KNOWN_BUCKETS)
        out = df.loc[mask].copy()
        _tprint_metrics(
            "Hard candidate prefilter applied",
            source_col=bucket_col,
            kept=len(out),
            total=len(df),
            keep_rate=float(len(out) / max(len(df), 1)),
            mem_mb=_process_memory_mb(),
        )
        return out

    _tprint_metrics(
        "Hard candidate prefilter unavailable",
        reason="no_candidate_or_bucket_column",
        rows=len(df),
        mem_mb=_process_memory_mb(),
    )
    return df


def _is_meta_stage(stage_val: Any) -> bool:
    s = str(stage_val).lower()
    return "meta" in s


def _weight_learnability_metrics(weights: np.ndarray, y_ret: np.ndarray) -> Dict[str, float]:
    w = np.asarray(weights, dtype=float)
    y = np.asarray(y_ret, dtype=float)
    s = max(float(np.sum(w)), EPS)
    k = max(1, int(np.ceil(0.01 * len(w))))
    return {
        "n_samples": float(len(w)),
        "n_eff": float(compute_n_eff(w)),
        "n_eff_ratio": float(compute_n_eff(w) / max(len(w), 1)),
        "top1pct_weight_share": float(np.sort(w)[-k:].sum() / s),
        "weight_mean": float(np.mean(w)),
        "weight_std": float(np.std(w)),
        "weight_p05": float(np.quantile(w, 0.05)),
        "weight_p50": float(np.quantile(w, 0.50)),
        "weight_p95": float(np.quantile(w, 0.95)),
        "weighted_return_mean": float(np.average(y, weights=w)) if len(y) else 0.0,
    }


def _build_stage_learnability_rows(
    stage_name: str,
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    components: Dict[str, np.ndarray],
    optimized_weights: np.ndarray,
    objective_value: float,
    cfg_runtime: Dict[str, Any],
    bucket_codes: Optional[np.ndarray] = None,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    metrics = _weight_learnability_metrics(optimized_weights, y_ret)
    redundancy = check_component_redundancy(components)
    sw_defaults = get_sample_weight_opt_defaults(cfg_runtime)
    ablation = run_ablation(
        X=np.asarray(X, dtype=np.float32),
        y_ret=np.asarray(y_ret, dtype=float),
        label_intervals=np.asarray(label_intervals),
        components=components,
        baseline_weights=np.asarray(optimized_weights, dtype=float),
        production_model=sw_defaults["sample_weight_opt_model_family"],
        n_splits=int(sw_defaults["sample_weight_opt_n_splits"]),
        embargo_bars=int(sw_defaults["sample_weight_opt_embargo_bars"]),
        cfg_runtime=cfg_runtime,
        enable=bool(cfg_runtime.get("sample_weight_opt_enable_ablation", True)) if isinstance(cfg_runtime, dict) else True,
        bucket_codes=bucket_codes,
    )
    ablation_rank = {name: i + 1 for i, (name, _) in enumerate(ablation)}
    ablation_score = {name: float(score) for name, score in ablation}

    for comp_name in sorted(components.keys()):
        comp_arr = np.asarray(components[comp_name], dtype=float)
        corr = np.nan
        try:
            corr = float(np.corrcoef(comp_arr, optimized_weights)[0, 1])
        except Exception:
            corr = np.nan
        rows.append(
            {
                "stage": stage_name,
                "row_type": "component",
                "component": comp_name,
                "objective_value": float(objective_value),
                **metrics,
                "component_mean": float(np.mean(comp_arr)),
                "component_std": float(np.std(comp_arr)),
                "component_p05": float(np.quantile(comp_arr, 0.05)),
                "component_p95": float(np.quantile(comp_arr, 0.95)),
                "corr_component_vs_optimized": corr,
                "ablation_score": ablation_score.get(comp_name, np.nan),
                "ablation_rank": float(ablation_rank.get(comp_name, np.nan)),
                "redundancy_max_corr": float(redundancy.get("max_corr", np.nan)) if isinstance(redundancy, dict) else np.nan,
            }
        )

    # one summary row per stage
    rows.append(
        {
            "stage": stage_name,
            "row_type": "summary",
            "component": "ALL",
            "objective_value": float(objective_value),
            **metrics,
            "component_mean": np.nan,
            "component_std": np.nan,
            "component_p05": np.nan,
            "component_p95": np.nan,
            "corr_component_vs_optimized": np.nan,
            "ablation_score": ablation_score.get("baseline", np.nan),
            "ablation_rank": float(ablation_rank.get("baseline", np.nan)),
            "redundancy_max_corr": float(redundancy.get("max_corr", np.nan)) if isinstance(redundancy, dict) else np.nan,
        }
    )
    return rows


def run_offline_sample_weight_optimisation(
    component_csv: str | None = None,
    output_csv: str | None = None,
    n_trials: int = 20,
) -> Dict[str, Any]:
    """Persist sample-weight optimiser best params for training-pipeline consumption."""
    from .params_store import (
        REPORTS_DIR,
        SAMPLE_WEIGHT_BEST_PARAMS_CSV,
        save_best_params_csv,
        apply_offline_optimizer_best_params,
    )

    cfg_runtime = apply_offline_optimizer_best_params(dict(CFG))
    best_params = _default_sample_weight_best_params(cfg_runtime)
    _tprint_metrics(
        "Offline sample-weight optimisation started",
        component_csv=component_csv or "",
        output_csv=output_csv or "",
        n_trials=n_trials,
        mem_mb=_process_memory_mb(),
    )

    report_rows: list[dict[str, Any]] = []
    if component_csv:
        # Pre-scan header to determine required columns
        header_df = pd.read_csv(component_csv, nrows=0)
        all_cols = set(header_df.columns)
        test_keys = set(CFG.get("test_feature_keys", TEST_FEATURE_KEYS))

        # Determine columns to load
        keep_cols = {c for c in all_cols if c in ("y_ret", "t_start", "t_end") or c.startswith("comp_")}
        stage_col = _detect_stage_column(header_df)
        if stage_col:
            keep_cols.add(stage_col)
        bucket_col = _detect_bucket_column(header_df)
        if bucket_col:
            keep_cols.add(bucket_col)
        for c in ("candidate_mask", "is_candidate", "candidate", "is_trade_candidate"):
            if c in all_cols:
                keep_cols.add(c)

        feature_cols = []
        for c in all_cols:
            norm_c = c[2:] if c.startswith("x_") else c
            if norm_c in test_keys:
                feature_cols.append(c)
                keep_cols.add(c)

        df = pd.read_csv(component_csv, usecols=list(keep_cols))
        _tprint_metrics(
            "Loaded component CSV",
            rows=len(df),
            columns=len(df.columns),
            features_loaded=len(feature_cols),
            mem_mb=_process_memory_mb(),
        )
        comp_cols = [c for c in df.columns if c.startswith("comp_")]

        # Verify essential columns
        if {"y_ret", "t_start", "t_end"}.issubset(df.columns) and comp_cols:
            if stage_col is None:
                # If stage_col was not detected, create dummy column
                df = df.copy()
                df["stage"] = "base"
                stage_col = "stage"

            stage_frames = {
                "base": df.loc[~df[stage_col].map(_is_meta_stage)].copy(),
                "meta": df.loc[df[stage_col].map(_is_meta_stage)].copy(),
            }

            sw_defaults = get_sample_weight_opt_defaults(cfg_runtime)
            for stage_name, sdf in stage_frames.items():
                if sdf.empty:
                    continue
                sdf = _apply_hard_candidate_prefilter(sdf, bucket_col=bucket_col)
                if sdf.empty:
                    _tprint_metrics("Stage skipped", stage=stage_name, reason="empty_after_candidate_prefilter", mem_mb=_process_memory_mb())
                    continue
                _tprint_metrics(
                    "Stage optimisation started",
                    stage=stage_name,
                    rows=len(sdf),
                    components=len(comp_cols),
                    y_ret_mean=float(np.mean(sdf["y_ret"].astype(float).values)),
                    y_ret_std=float(np.std(sdf["y_ret"].astype(float).values)),
                    mem_mb=_process_memory_mb(),
                )
                components = {c.replace("comp_", ""): sdf[c].astype(float).values for c in comp_cols}

                current_feature_cols = [c for c in feature_cols if c in sdf.columns]
                if current_feature_cols:
                    X = sdf[current_feature_cols]
                else:
                    X = np.zeros((len(sdf), 1), dtype=float)
                if bucket_col and bucket_col in sdf.columns:
                    bucket_codes = sdf[bucket_col].astype(str).values
                else:
                    bucket_codes = np.full(len(sdf), "Global", dtype=object)
                _tprint_metrics(
                    "Stage bucket training setup",
                    stage=stage_name,
                    n_buckets=len(pd.unique(bucket_codes)),
                    mem_mb=_process_memory_mb(),
                )
                label_intervals = np.column_stack([
                    pd.to_datetime(sdf["t_start"]).values.astype("datetime64[ns]"),
                    pd.to_datetime(sdf["t_end"]).values.astype("datetime64[ns]"),
                ])
                res = optimize_component_weights(
                    X=X,
                    y_ret=sdf["y_ret"].astype(float).values,
                    label_intervals=label_intervals,
                    components=components,
                    production_model=sw_defaults["sample_weight_opt_model_family"],
                    n_trials=int(n_trials),
                    n_splits=int(sw_defaults["sample_weight_opt_n_splits"]),
                    embargo_bars=int(sw_defaults["sample_weight_opt_embargo_bars"]),
                    min_n_eff_ratio=float(sw_defaults["sample_weight_opt_min_n_eff_ratio"]),
                    max_top1pct=float(sw_defaults["sample_weight_opt_max_top1pct"]),
                    random_state=int(cfg_runtime.get("seed", 42)),
                    cfg_runtime=cfg_runtime,
                    bucket_codes=bucket_codes,
                )

                best_params[f"component_alphas_{stage_name}"] = res.component_alphas
                if stage_name == "base":
                    best_params["component_alphas"] = res.component_alphas

                report_rows.extend(
                    _build_stage_learnability_rows(
                        stage_name=stage_name,
                        X=np.asarray(X),
                        y_ret=sdf["y_ret"].astype(float).values,
                        label_intervals=label_intervals,
                        components=components,
                        optimized_weights=res.optimized_weights,
                        objective_value=float(res.objective_value),
                        cfg_runtime=cfg_runtime,
                        bucket_codes=bucket_codes,
                    )
                )
                _tprint_metrics(
                    "Stage optimisation complete",
                    stage=stage_name,
                    objective_value=float(res.objective_value),
                    mem_mb=_process_memory_mb(),
                )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(output_csv) if output_csv else (REPORTS_DIR / "sample_weight_optimization_report.csv")
    if not report_rows:
        report_rows = [{"stage": "base", "row_type": "summary", "component": "ALL", "objective_value": np.nan, "note": "defaults_only"}]
    if report_rows:
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
    else:
        pd.DataFrame(report_rows).to_csv(out, index=False)
    save_best_params_csv(SAMPLE_WEIGHT_BEST_PARAMS_CSV, best_params, metadata={"source": "sample_weight_optimization"})
    _tprint_metrics(
        "Offline sample-weight optimisation finished",
        report_rows=len(report_rows),
        report_csv=str(out),
        best_params_csv=str(SAMPLE_WEIGHT_BEST_PARAMS_CSV),
        mem_mb=_process_memory_mb(),
    )
    return {"report_csv": str(out), "best_params_csv": str(SAMPLE_WEIGHT_BEST_PARAMS_CSV), "best_params": best_params}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from .params_store import REPORTS_DIR

    parser = argparse.ArgumentParser(description="Offline sample-weight optimisation for extreme_price_movements")
    parser.add_argument("--component-csv", default="", help="Optional CSV with comp_* columns + y_ret,t_start,t_end")
    parser.add_argument("--output", default=str(REPORTS_DIR / "sample_weight_optimization_report.csv"), help="Output report CSV path")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials when component CSV is provided")
    args = parser.parse_args()

    result = run_offline_sample_weight_optimisation(
        component_csv=args.component_csv or None,
        output_csv=args.output,
        n_trials=int(args.trials),
    )
    tprint(f"Saved sample-weight optimisation report: {result['report_csv']}")
    tprint(f"Saved best params CSV: {result['best_params_csv']}")
