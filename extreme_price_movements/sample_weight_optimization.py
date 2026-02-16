"""Sample-weight optimization utilities for extreme_price_movements training."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None

from .purged_cv import IntervalPurgedKFold
from .utils import tprint


EPS = 1e-8


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

    for name, w in components.items():
        arr = np.asarray(w, dtype=float)
        p5 = float(np.nanpercentile(arr, 5))
        p95 = float(np.nanpercentile(arr, 95))
        span = p95 - p5
        ratio = p95 / (p5 + 1e-12)

        if not np.isfinite(span) or span < 1e-6 or ratio < 1.05:
            clipped[name] = np.ones_like(arr, dtype=float)
            local_weights[name] = 0.0
        else:
            base = np.maximum(arr, eps)
            clipped[name] = np.clip(base, p5, p95)

    log_w = np.zeros_like(next(iter(clipped.values())), dtype=float)
    for name, w in clipped.items():
        alpha = float(local_weights.get(name, 1.0))
        if alpha == 0.0:
            continue
        log_w += alpha * np.log(w + eps)

    w_final = np.exp(log_w)
    if not np.all(np.isfinite(w_final)) or np.sum(w_final) <= 0:
        w_final = np.ones_like(w_final, dtype=float)

    target_n_eff = min_n_eff_ratio * len(w_final)
    n_eff = compute_n_eff(w_final)

    if n_eff < target_n_eff:
        # temperature flattening: w^(1/T), T>=1.0
        lo, hi = 1.0, 8.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            w_t = np.power(np.maximum(w_final, eps), 1.0 / mid)
            w_t /= max(float(np.mean(w_t)), eps)
            if compute_n_eff(w_t) >= target_n_eff:
                hi = mid
            else:
                lo = mid
        w_final = np.power(np.maximum(w_final, eps), 1.0 / hi)

    w_final /= max(float(np.mean(w_final)), eps)
    return w_final


def compute_vol_weights(
    past_vol: np.ndarray,
    timestamps: np.ndarray,
    direction: str = "downweight_high",
    power: float = 0.5,
    min_group_size: int = 20,
) -> np.ndarray:
    vol_df = pd.DataFrame({"vol": np.asarray(past_vol, dtype=float), "ts": pd.to_datetime(timestamps)})

    def safe_median(x: pd.Series) -> float:
        if len(x) < min_group_size:
            return float(x.median()) if len(x) else 1.0
        return float(x.median())

    denom = vol_df.groupby("ts")["vol"].transform(lambda x: x / (safe_median(x) + EPS)).values
    denom = np.maximum(denom, EPS)
    if direction == "downweight_high":
        w = np.power(denom, -power)
    else:
        w = np.power(denom, power)
    return w / max(float(np.mean(w)), EPS)


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

    era_df = pd.DataFrame({"w": w, "era": np.asarray(era_indices)})
    era_neff = era_df.groupby("era")["w"].apply(lambda x: (x.sum() ** 2) / max((x ** 2).sum(), EPS))
    n_samples_per_era = era_df.groupby("era").size()

    min_era_neff = float(era_neff.min()) if len(era_neff) else len(w)
    min_expected = float((n_samples_per_era * min_era_neff_ratio).min()) if len(n_samples_per_era) else 0.0
    if min_era_neff < min_expected:
        w = np.power(w, 0.5)

    return w / max(float(np.mean(w)), EPS)


def check_component_redundancy(
    components: Dict[str, np.ndarray],
    threshold: float = 0.85,
) -> Dict[str, Any]:
    names = list(components.keys())
    arrays = [np.asarray(components[n], dtype=float) for n in names]
    n = len(arrays)
    corr = np.eye(n, dtype=float)

    redundant = []
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = spearmanr(arrays[i], arrays[j])
            r = float(0.0 if not np.isfinite(r) else r)
            corr[i, j] = corr[j, i] = r
            if abs(r) > threshold:
                redundant.append((names[i], names[j], r))

    corr_map = {name: corr[idx, :].tolist() for idx, name in enumerate(names)}
    return {"pairs": redundant, "corr_matrix": corr_map}


def log_weight_statistics(weights: np.ndarray, era_indices: np.ndarray, name: str) -> Dict[str, float]:
    w = np.asarray(weights, dtype=float)
    era_df = pd.DataFrame({"w": w, "era": np.asarray(era_indices)})
    era_neff = era_df.groupby("era")["w"].apply(lambda x: (x.sum() ** 2) / max((x ** 2).sum(), EPS))

    top1_k = max(1, int(len(w) * 0.01))
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
        "top1pct_share": float(np.sort(w)[-top1_k:].sum() / max(np.sum(w), EPS)),
        "era_neff_min": float(era_neff.min()) if len(era_neff) else 0.0,
        "era_neff_mean": float(era_neff.mean()) if len(era_neff) else 0.0,
    }
    tprint(f"{name} weights | " + " ".join([f"{k}={v:.4f}" for k, v in stats.items()]))
    return stats


def _make_model(model_family: str, random_state: int):
    fam = model_family.lower()
    if fam == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if fam == "randomforest":
        return RandomForestRegressor(
            n_estimators=80,
            max_depth=6,
            min_samples_leaf=50,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    return ExtraTreesRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=50,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
    )


def _safe_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ic, _ = spearmanr(y_true, y_pred)
    if not np.isfinite(ic):
        return 0.0
    return float(ic)


def _run_cv_ic(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    label_intervals: np.ndarray,
    model_family: str,
    n_splits: int,
    embargo_bars: int,
    seed: int,
) -> np.ndarray:
    splitter = IntervalPurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)
    fold_ics = []
    Xv = np.asarray(X, dtype=np.float32)
    yv = np.asarray(y, dtype=float)
    wv = np.asarray(sample_weight, dtype=float)

    Xs = Xv

    for tr, va in splitter.split(Xs, label_intervals=label_intervals):
        if len(tr) < 64 or len(va) < 32:
            continue
        model = _make_model(model_family, random_state=seed)
        model.fit(Xs[tr], yv[tr], sample_weight=wv[tr])
        pred = model.predict(Xs[va])
        fold_ics.append(_safe_ic(yv[va], pred))

    return np.asarray(fold_ics, dtype=float)


def constrained_objective(
    w: np.ndarray,
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    model_family: str,
    n_splits: int,
    embargo_bars: int,
    seeds: Iterable[int] = (42, 123),
    min_n_eff_ratio: float = 0.30,
    max_top1pct: float = 0.10,
) -> float:
    n_eff = compute_n_eff(w)
    min_n_eff = min_n_eff_ratio * len(w)
    k = max(1, int(np.ceil(len(w) * 0.01)))
    top1pct_share = float(np.sort(w)[-k:].sum() / max(float(np.sum(w)), EPS))

    if n_eff < min_n_eff or top1pct_share > max_top1pct:
        return -10.0

    fold_scores = []
    for seed in seeds:
        fold_ics = _run_cv_ic(
            X, y_ret, w, label_intervals, model_family=model_family,
            n_splits=n_splits, embargo_bars=embargo_bars, seed=int(seed),
        )
        if fold_ics.size:
            fold_scores.append(float(np.mean(fold_ics)))
    if not fold_scores:
        return -10.0
    ic_mean = float(np.mean(fold_scores))
    ic_std = float(np.std(fold_scores))
    return ic_mean - 0.5 * ic_std


@dataclass
class WeightOptimizationResult:
    optimized_weights: np.ndarray
    component_alphas: Dict[str, float]
    objective_value: float
    diagnostics: Dict[str, Any]


def optimize_component_weights(
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    components: Dict[str, np.ndarray],
    production_model: str = "ExtraTrees",
    n_trials: int = 20,
    n_splits: int = 5,
    embargo_bars: int = 10,
    min_n_eff_ratio: float = 0.30,
    max_top1pct: float = 0.10,
    random_state: int = 42,
) -> WeightOptimizationResult:
    if not components:
        ones = np.ones(len(y_ret), dtype=float)
        return WeightOptimizationResult(ones, {}, -10.0, {"reason": "no_components"})

    names = list(components.keys())
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
        )
        return WeightOptimizationResult(w, alphas, val, {"fallback": "optuna_unavailable_or_disabled"})

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def _obj(trial: "optuna.trial.Trial") -> float:
        alphas = {n: trial.suggest_float(f"alpha_{n}", 0.0, 1.5) for n in names}
        w = combine_weights_safely(components, alphas, min_n_eff_ratio=min_n_eff_ratio)
        score = constrained_objective(
            w, X, y_ret, label_intervals,
            model_family=production_model,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            min_n_eff_ratio=min_n_eff_ratio,
            max_top1pct=max_top1pct,
        )
        return float(score)

    study.optimize(_obj, n_trials=int(n_trials), show_progress_bar=False)

    best_params = study.best_params if len(study.trials) else {}
    alphas = {n: float(best_params.get(f"alpha_{n}", 1.0)) for n in names}
    optimized = combine_weights_safely(components, alphas, min_n_eff_ratio=min_n_eff_ratio)

    diagnostics = {
        "best_trial": int(study.best_trial.number) if study.best_trial is not None else -1,
        "n_trials": len(study.trials),
        "redundancy": check_component_redundancy(components),
    }
    return WeightOptimizationResult(optimized, alphas, float(study.best_value), diagnostics)


def run_ablation(
    X: np.ndarray,
    y_ret: np.ndarray,
    label_intervals: np.ndarray,
    components: Dict[str, np.ndarray],
    baseline_weights: np.ndarray,
    production_model: str = "ExtraTrees",
    n_splits: int = 5,
    embargo_bars: int = 10,
) -> list[tuple[str, float]]:
    results = []
    base_score = constrained_objective(
        baseline_weights, X, y_ret, label_intervals,
        model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
    )
    results.append(("baseline", float(base_score)))

    for name, w_comp in components.items():
        w = baseline_weights * np.asarray(w_comp, dtype=float)
        w /= max(float(np.mean(w)), EPS)
        score = constrained_objective(
            w, X, y_ret, label_intervals,
            model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
        )
        results.append((name, float(score)))

    for n1, n2 in combinations(components.keys(), 2):
        w = baseline_weights * np.asarray(components[n1], dtype=float) * np.asarray(components[n2], dtype=float)
        w /= max(float(np.mean(w)), EPS)
        score = constrained_objective(
            w, X, y_ret, label_intervals,
            model_family=production_model, n_splits=n_splits, embargo_bars=embargo_bars,
        )
        results.append((f"{n1}+{n2}", float(score)))

    return sorted(results, key=lambda x: x[1], reverse=True)
