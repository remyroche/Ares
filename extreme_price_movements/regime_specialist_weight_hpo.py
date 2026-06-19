from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from .regime_specialist_similarity import SpecialistWeightConfig


@dataclass(frozen=True)
class RegimeSpecialistWeightHPOSpace:
    analogue_gamma_low: float = 1.5
    analogue_gamma_high: float = 3.0
    replay_gamma_low: float = 1.5
    replay_gamma_high: float = 3.0
    tau_adaptive_low: float = 10_000.0
    tau_adaptive_high: float = 40_000.0
    tau_replay_low: float = 25_000.0
    tau_replay_high: float = 100_000.0
    min_current_plus_analogue_mass_low: float = 0.50
    min_current_plus_analogue_mass_high: float = 0.60
    less_interesting_max_mass_low: float = 0.30
    less_interesting_max_mass_high: float = 0.50
    current_gamma: float = 1.0
    recency_power: float = 0.5
    less_interesting_min_mass: float = 0.10


@dataclass(frozen=True)
class RegimeSpecialistWeightHPOConfig:
    n_trials: int = 150
    early_stop_patience: int = 30
    random_state: int = 42
    precision_2w_weight: float = 1.00
    precision_4w_weight: float = 0.50
    top30_return_4w_weight: float = 0.50
    auc_4w_weight: float = 0.25
    return_scale: float = 1.0
    max_weight_p99_p50: float = 20.0
    min_weighted_ess_frac: float = 0.03
    concentration_penalty_weight: float = 0.02
    low_ess_penalty_weight: float = 0.25
    adaptive_floor_penalty_weight: float = 2.0
    replay_cap_penalty_weight: float = 2.0
    min_total_n_eff_reliability: float = 0.50
    min_adaptive_n_eff_reliability: float = 0.25
    min_current_weight_mass: float = 0.08
    min_recent_4w_weight_mass: float = 0.10
    n_eff_reliability_penalty_weight: float = 1.0
    adaptive_n_eff_penalty_weight: float = 0.50
    current_focus_penalty_weight: float = 0.50
    recent_focus_penalty_weight: float = 0.50


def precision_at_fraction(y_true: Any, score: Any, fraction: float) -> float:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.nan_to_num(np.asarray(score, dtype=np.float32).reshape(-1), nan=-np.inf)
    if len(y) == 0 or len(pred) != len(y):
        return float("nan")
    k = int(np.ceil(float(np.clip(fraction, 1e-6, 1.0)) * len(y)))
    k = max(1, min(k, len(y)))
    top = np.argsort(pred, kind="mergesort")[-k:]
    return float(np.mean((y[top] >= 0.5).astype(np.float32)))


def precision_score_top_fracs(y_true: Any, score: Any) -> dict[str, float]:
    p10 = precision_at_fraction(y_true, score, 0.10)
    p20 = precision_at_fraction(y_true, score, 0.20)
    p30 = precision_at_fraction(y_true, score, 0.30)
    return {
        "p_at_10": float(p10),
        "p_at_20": float(p20),
        "p_at_30": float(p30),
        "precision_score": float(0.25 * p10 + 0.50 * p20 + 1.00 * p30),
        "rows": int(len(np.asarray(y_true).reshape(-1))),
    }


def _timestamp_series(timestamps: Any, n: int) -> pd.Series:
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)), utc=True, errors="coerce")
    if len(ts) != int(n) or not bool(ts.notna().any()):
        raise ValueError("regime specialist weight HPO requires valid aligned timestamps")
    return ts


def _auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if y.size < 2 or np.unique((y >= 0.5).astype(np.int8)).size < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score((y >= 0.5).astype(np.int8), s))
    except Exception:
        ranks = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
        pos = y >= 0.5
        n_pos = float(np.sum(pos))
        n_neg = float(np.sum(~pos))
        if n_pos <= 0.0 or n_neg <= 0.0:
            return float("nan")
        rank_sum_pos = float(np.sum(ranks[pos]))
        return float((rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def mean_return_at_fraction(returns: Any, score: Any, fraction: float) -> float:
    ret = np.asarray(returns, dtype=np.float64).reshape(-1)
    pred = np.nan_to_num(np.asarray(score, dtype=np.float64).reshape(-1), nan=-np.inf)
    mask = np.isfinite(ret) & np.isfinite(pred)
    ret = ret[mask]
    pred = pred[mask]
    if ret.size == 0:
        return float("nan")
    k = int(np.ceil(float(np.clip(fraction, 1e-6, 1.0)) * len(ret)))
    k = max(1, min(k, len(ret)))
    top = np.argsort(pred, kind="mergesort")[-k:]
    return float(np.mean(ret[top]))


def _weighted_ess_fraction(weights: Any) -> float:
    if weights is None:
        return 1.0
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size == 0:
        return 0.0
    ess = float(np.sum(w) ** 2 / max(np.sum(w * w), 1e-12))
    return float(ess / max(float(w.size), 1.0))


def _weight_p99_p50(weights: Any) -> float:
    if weights is None:
        return 1.0
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w) & (w > 0.0)]
    if w.size == 0:
        return float("inf")
    p50 = float(np.nanpercentile(w, 50.0))
    p99 = float(np.nanpercentile(w, 99.0))
    return float(p99 / max(p50, 1e-12))


def _recent_weight_mass(weights: Any | None, timestamps: Any | None, days: float) -> float:
    if weights is None or timestamps is None:
        return float("nan")
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    try:
        ts = _timestamp_series(timestamps, len(w))
    except Exception:
        return float("nan")
    valid = np.isfinite(w) & (w > 0.0) & ts.notna().to_numpy(dtype=bool)
    if not bool(valid.any()):
        return float("nan")
    latest = ts.loc[valid].max()
    recent = (ts >= latest - pd.Timedelta(days=float(days))).to_numpy(dtype=bool)
    total = float(np.sum(w[valid]))
    if total <= 1e-12:
        return float("nan")
    return float(np.sum(w[valid & recent]) / total)


def _combined_reliability(*values: float) -> float:
    miss = 1.0
    seen = False
    for value in values:
        try:
            rel = float(value)
        except Exception:
            continue
        if not np.isfinite(rel):
            continue
        miss *= 1.0 - float(np.clip(rel, 0.0, 1.0))
        seen = True
    return float(1.0 - miss) if seen else 0.0


def weight_hpo_penalty(
    weights: Any | None,
    diagnostics: Mapping[str, Any] | None,
    *,
    timestamps: Any | None = None,
    config: RegimeSpecialistWeightHPOConfig = RegimeSpecialistWeightHPOConfig(),
) -> dict[str, float]:
    diag = diagnostics if isinstance(diagnostics, Mapping) else {}
    p99_p50 = _weight_p99_p50(weights)
    ess_frac = _weighted_ess_fraction(weights)
    adaptive_mass = float(diag.get("actual_current_weight_mass", diag.get("current_mass", 0.0)) or 0.0) + float(
        diag.get("actual_analogue_weight_mass", diag.get("analogue_mass", 0.0)) or 0.0
    )
    adaptive_floor = float(diag.get("min_current_plus_analogue_mass", diag.get("configured_min_adaptive_mass", 0.0)) or 0.0)
    replay_mass = float(
        diag.get(
            "actual_less_interesting_weight_mass",
            diag.get("less_interesting_mass", 0.0),
        )
        or 0.0
    )
    replay_cap = float(diag.get("less_interesting_mass_cap", 1.0) or 1.0)
    adaptive_n_eff_rel = float(diag.get("adaptive_n_eff_reliability", 0.0) or 0.0)
    replay_n_eff_rel = float(diag.get("replay_n_eff_reliability", 0.0) or 0.0)
    total_n_eff_rel = _combined_reliability(adaptive_n_eff_rel, replay_n_eff_rel)
    current_mass = float(diag.get("actual_current_weight_mass", diag.get("current_mass", 0.0)) or 0.0)
    recent_4w_mass = _recent_weight_mass(weights, timestamps, 28.0)
    recent_focus_shortfall = (
        max(0.0, float(config.min_recent_4w_weight_mass) - recent_4w_mass)
        if np.isfinite(recent_4w_mass)
        else 0.0
    )
    return {
        "weight_concentration_penalty": float(
            max(0.0, p99_p50 / max(float(config.max_weight_p99_p50), 1e-12) - 1.0)
            * float(config.concentration_penalty_weight)
        ),
        "low_weighted_ess_penalty": float(
            max(0.0, float(config.min_weighted_ess_frac) - ess_frac)
            * float(config.low_ess_penalty_weight)
        ),
        "adaptive_floor_penalty": float(
            max(0.0, adaptive_floor - adaptive_mass)
            * float(config.adaptive_floor_penalty_weight)
        ),
        "replay_cap_penalty": float(
            max(0.0, replay_mass - replay_cap)
            * float(config.replay_cap_penalty_weight)
        ),
        "total_n_eff_reliability_penalty": float(
            max(0.0, float(config.min_total_n_eff_reliability) - total_n_eff_rel)
            * float(config.n_eff_reliability_penalty_weight)
        ),
        "adaptive_n_eff_reliability_penalty": float(
            max(0.0, float(config.min_adaptive_n_eff_reliability) - adaptive_n_eff_rel)
            * float(config.adaptive_n_eff_penalty_weight)
        ),
        "current_focus_penalty": float(
            max(0.0, float(config.min_current_weight_mass) - current_mass)
            * float(config.current_focus_penalty_weight)
        ),
        "recent_4w_focus_penalty": float(
            recent_focus_shortfall * float(config.recent_focus_penalty_weight)
        ),
    }


def score_regime_specialist_weight_trial(
    y_true: Any,
    score: Any,
    returns: Any,
    timestamps: Any,
    *,
    sample_weight: Any | None = None,
    weight_diagnostics: Mapping[str, Any] | None = None,
    extra_penalties: Mapping[str, float] | None = None,
    config: RegimeSpecialistWeightHPOConfig = RegimeSpecialistWeightHPOConfig(),
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float32).reshape(-1)
    pred = np.asarray(score, dtype=np.float32).reshape(-1)
    ret = np.asarray(returns, dtype=np.float32).reshape(-1)
    if len(y) != len(pred) or len(y) != len(ret):
        raise ValueError("aligned y_true, score, and returns are required")
    ts = _timestamp_series(timestamps, len(y))
    latest = ts.loc[ts.notna()].max()
    mask_2w = (ts >= latest - pd.Timedelta(days=14)).to_numpy(dtype=bool)
    mask_4w = (ts >= latest - pd.Timedelta(days=28)).to_numpy(dtype=bool)
    precision_2w = precision_score_top_fracs(y[mask_2w], pred[mask_2w])
    precision_4w = precision_score_top_fracs(y[mask_4w], pred[mask_4w])
    mean_return_top30_4w = mean_return_at_fraction(ret[mask_4w], pred[mask_4w], 0.30)
    auc_4w = _auc_score(y[mask_4w], pred[mask_4w])
    penalties = weight_hpo_penalty(
        sample_weight,
        weight_diagnostics,
        timestamps=ts,
        config=config,
    )
    if extra_penalties:
        for key, value in extra_penalties.items():
            try:
                penalties[str(key)] = float(value)
            except Exception:
                continue
    penalty_total = float(sum(v for v in penalties.values() if np.isfinite(v)))
    value = (
        float(config.precision_2w_weight) * float(precision_2w["precision_score"])
        + float(config.precision_4w_weight) * float(precision_4w["precision_score"])
        + float(config.top30_return_4w_weight)
        * float(mean_return_top30_4w)
        * float(config.return_scale)
        + float(config.auc_4w_weight) * float(0.0 if not np.isfinite(auc_4w) else auc_4w)
        - penalty_total
    )
    return {
        "objective_value": float(value),
        "precision_score_last_2w": precision_2w,
        "precision_score_last_4w": precision_4w,
        "mean_return_top30_last_4w": float(mean_return_top30_4w),
        "auc_last_4w": float(auc_4w),
        "penalties": penalties,
        "penalty_total": float(penalty_total),
        "weighted_ess_fraction": float(_weighted_ess_fraction(sample_weight)),
        "weight_p99_p50": float(_weight_p99_p50(sample_weight)),
        "recent_2w_weight_mass": float(_recent_weight_mass(sample_weight, ts, 14.0)),
        "recent_4w_weight_mass": float(_recent_weight_mass(sample_weight, ts, 28.0)),
        "latest_timestamp": latest.isoformat(),
        "rows_last_2w": int(np.sum(mask_2w)),
        "rows_last_4w": int(np.sum(mask_4w)),
    }


def suggest_weight_config(
    trial: Any,
    *,
    space: RegimeSpecialistWeightHPOSpace = RegimeSpecialistWeightHPOSpace(),
) -> SpecialistWeightConfig:
    analogue_gamma = float(
        trial.suggest_float(
            "analogue_gamma",
            float(space.analogue_gamma_low),
            float(space.analogue_gamma_high),
        ),
    )
    replay_gamma = float(
        trial.suggest_float(
            "replay_gamma",
            float(space.replay_gamma_low),
            float(space.replay_gamma_high),
        ),
    )
    tau_adaptive = float(
        trial.suggest_float(
            "tau_adaptive",
            float(space.tau_adaptive_low),
            float(space.tau_adaptive_high),
            log=True,
        ),
    )
    tau_replay = float(
        trial.suggest_float(
            "tau_replay",
            float(space.tau_replay_low),
            float(space.tau_replay_high),
            log=True,
        ),
    )
    min_adaptive = float(
        trial.suggest_float(
            "min_current_plus_analogue_mass",
            float(space.min_current_plus_analogue_mass_low),
            float(space.min_current_plus_analogue_mass_high),
        ),
    )
    less_max_raw = float(
        trial.suggest_float(
            "less_interesting_max_mass",
            float(space.less_interesting_max_mass_low),
            float(space.less_interesting_max_mass_high),
        ),
    )
    less_max = float(min(less_max_raw, max(0.0, 1.0 - min_adaptive)))
    return SpecialistWeightConfig(
        current_gamma=float(space.current_gamma),
        analogue_gamma=analogue_gamma,
        replay_gamma=replay_gamma,
        recency_power=float(space.recency_power),
        tau_current=tau_adaptive,
        tau_analogue=tau_adaptive,
        tau_normal=tau_replay,
        tau_irrelevant=tau_replay,
        min_current_plus_analogue_mass=min_adaptive,
        less_interesting_min_mass=float(space.less_interesting_min_mass),
        less_interesting_max_mass=less_max,
    )


def _float_from_mapping(cfg: Mapping[str, Any] | None, key: str, default: float) -> float:
    if not isinstance(cfg, Mapping):
        return float(default)
    raw = cfg.get(key, default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return float(value) if np.isfinite(value) else float(default)


def _int_from_mapping(cfg: Mapping[str, Any] | None, key: str, default: int) -> int:
    if not isinstance(cfg, Mapping):
        return int(default)
    raw = cfg.get(key, default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def hpo_config_from_mapping(
    cfg: Mapping[str, Any] | None,
) -> RegimeSpecialistWeightHPOConfig:
    defaults = RegimeSpecialistWeightHPOConfig()
    return RegimeSpecialistWeightHPOConfig(
        n_trials=_int_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_trials",
            defaults.n_trials,
        ),
        early_stop_patience=_int_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_early_stop_patience",
            defaults.early_stop_patience,
        ),
        random_state=_int_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_random_state",
            defaults.random_state,
        ),
        precision_2w_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_precision_2w_weight",
            defaults.precision_2w_weight,
        ),
        precision_4w_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_precision_4w_weight",
            defaults.precision_4w_weight,
        ),
        top30_return_4w_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_top30_return_4w_weight",
            defaults.top30_return_4w_weight,
        ),
        auc_4w_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_auc_4w_weight",
            defaults.auc_4w_weight,
        ),
        return_scale=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_return_scale",
            defaults.return_scale,
        ),
        max_weight_p99_p50=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_max_weight_p99_p50",
            defaults.max_weight_p99_p50,
        ),
        min_weighted_ess_frac=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_weighted_ess_frac",
            defaults.min_weighted_ess_frac,
        ),
        concentration_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_concentration_penalty_weight",
            defaults.concentration_penalty_weight,
        ),
        low_ess_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_low_ess_penalty_weight",
            defaults.low_ess_penalty_weight,
        ),
        adaptive_floor_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_adaptive_floor_penalty_weight",
            defaults.adaptive_floor_penalty_weight,
        ),
        replay_cap_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_replay_cap_penalty_weight",
            defaults.replay_cap_penalty_weight,
        ),
        min_total_n_eff_reliability=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_total_n_eff_reliability",
            defaults.min_total_n_eff_reliability,
        ),
        min_adaptive_n_eff_reliability=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_adaptive_n_eff_reliability",
            defaults.min_adaptive_n_eff_reliability,
        ),
        min_current_weight_mass=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_current_weight_mass",
            defaults.min_current_weight_mass,
        ),
        min_recent_4w_weight_mass=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_recent_4w_weight_mass",
            defaults.min_recent_4w_weight_mass,
        ),
        n_eff_reliability_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_n_eff_reliability_penalty_weight",
            defaults.n_eff_reliability_penalty_weight,
        ),
        adaptive_n_eff_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_adaptive_n_eff_penalty_weight",
            defaults.adaptive_n_eff_penalty_weight,
        ),
        current_focus_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_current_focus_penalty_weight",
            defaults.current_focus_penalty_weight,
        ),
        recent_focus_penalty_weight=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_recent_focus_penalty_weight",
            defaults.recent_focus_penalty_weight,
        ),
    )


def hpo_space_from_mapping(
    cfg: Mapping[str, Any] | None,
) -> RegimeSpecialistWeightHPOSpace:
    defaults = RegimeSpecialistWeightHPOSpace()
    return RegimeSpecialistWeightHPOSpace(
        analogue_gamma_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_analogue_gamma_low",
            defaults.analogue_gamma_low,
        ),
        analogue_gamma_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_analogue_gamma_high",
            defaults.analogue_gamma_high,
        ),
        replay_gamma_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_replay_gamma_low",
            defaults.replay_gamma_low,
        ),
        replay_gamma_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_replay_gamma_high",
            defaults.replay_gamma_high,
        ),
        tau_adaptive_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_tau_adaptive_low",
            defaults.tau_adaptive_low,
        ),
        tau_adaptive_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_tau_adaptive_high",
            defaults.tau_adaptive_high,
        ),
        tau_replay_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_tau_replay_low",
            defaults.tau_replay_low,
        ),
        tau_replay_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_tau_replay_high",
            defaults.tau_replay_high,
        ),
        min_current_plus_analogue_mass_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_current_plus_analogue_mass_low",
            defaults.min_current_plus_analogue_mass_low,
        ),
        min_current_plus_analogue_mass_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_min_current_plus_analogue_mass_high",
            defaults.min_current_plus_analogue_mass_high,
        ),
        less_interesting_max_mass_low=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_less_interesting_max_mass_low",
            defaults.less_interesting_max_mass_low,
        ),
        less_interesting_max_mass_high=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_less_interesting_max_mass_high",
            defaults.less_interesting_max_mass_high,
        ),
        current_gamma=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_current_gamma",
            defaults.current_gamma,
        ),
        recency_power=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_recency_power",
            defaults.recency_power,
        ),
        less_interesting_min_mass=_float_from_mapping(
            cfg,
            "lgbm_regime_specialist_weight_hpo_less_interesting_min_mass",
            defaults.less_interesting_min_mass,
        ),
    )


def optimize_regime_specialist_weight_hpo(
    evaluator: Callable[[SpecialistWeightConfig, Any], Mapping[str, Any]],
    *,
    hpo_config: RegimeSpecialistWeightHPOConfig = RegimeSpecialistWeightHPOConfig(),
    space: RegimeSpecialistWeightHPOSpace = RegimeSpecialistWeightHPOSpace(),
) -> dict[str, Any]:
    try:
        import optuna
        from optuna.trial import TrialState
    except Exception as exc:
        raise RuntimeError("regime specialist weight HPO requires optuna") from exc

    best_seen = {"value": -float("inf"), "trial": -1}

    def objective(trial: Any) -> float:
        weight_config = suggest_weight_config(trial, space=space)
        pack = evaluator(weight_config, trial)
        if "objective_value" in pack:
            score_pack = dict(pack)
        else:
            score_pack = score_regime_specialist_weight_trial(
                pack["y_true"],
                pack["score"],
                pack["returns"],
                pack["timestamps"],
                sample_weight=pack.get("sample_weight"),
                weight_diagnostics=pack.get("weight_diagnostics"),
                extra_penalties=pack.get("penalties"),
                config=hpo_config,
            )
        value = float(score_pack["objective_value"])
        trial.set_user_attr("weight_config", asdict(weight_config))
        for key, val in score_pack.items():
            if isinstance(val, (int, float, str, bool)) or val is None:
                trial.set_user_attr(str(key), val)
        return value

    def early_stop_callback(study: Any, trial: Any) -> None:
        if trial.state != TrialState.COMPLETE or trial.value is None:
            return
        if float(trial.value) > float(best_seen["value"]):
            best_seen["value"] = float(trial.value)
            best_seen["trial"] = int(trial.number)
        elif int(trial.number) - int(best_seen["trial"]) >= int(hpo_config.early_stop_patience):
            study.stop()

    sampler = optuna.samplers.TPESampler(
        seed=int(hpo_config.random_state),
        multivariate=True,
        group=True,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(0, int(hpo_config.n_trials)),
        callbacks=[early_stop_callback],
        n_jobs=1,
        show_progress_bar=False,
    )
    complete = [
        t
        for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not complete:
        raise RuntimeError("regime specialist weight HPO completed without valid trials")
    best = study.best_trial
    return {
        "source": "regime_specialist_weight_hpo",
        "sampler": "TPESampler",
        "n_trials_requested": int(hpo_config.n_trials),
        "early_stop_patience": int(hpo_config.early_stop_patience),
        "completed_trials": int(len(complete)),
        "best_trial": int(best.number),
        "best_value": float(best.value),
        "best_params": dict(best.params),
        "best_weight_config": dict(best.user_attrs.get("weight_config", {})),
        "best_user_attrs": dict(best.user_attrs),
        "hpo_config": asdict(hpo_config),
        "space": asdict(space),
    }


def save_regime_specialist_weight_hpo_result(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)
    tmp_path.replace(out_path)
    return out_path


__all__ = [
    "RegimeSpecialistWeightHPOConfig",
    "RegimeSpecialistWeightHPOSpace",
    "hpo_config_from_mapping",
    "hpo_space_from_mapping",
    "mean_return_at_fraction",
    "optimize_regime_specialist_weight_hpo",
    "precision_at_fraction",
    "precision_score_top_fracs",
    "save_regime_specialist_weight_hpo_result",
    "score_regime_specialist_weight_trial",
    "suggest_weight_config",
    "weight_hpo_penalty",
]
