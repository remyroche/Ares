#!/usr/bin/env python3
"""Optimize a causal per-head health overlay for reliability-blend policy replay.

The overlay adjusts only portfolio-manager inputs:

* effective rank score;
* per-row deployment threshold.

It does not retrain the meta model and does not use future outcomes at decision
time.  At timestamp t, health metrics are computed from candidate rows whose
exit timestamp is strictly before t.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import optuna
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    PortfolioPolicyParams,
    fit_monotone_ev_curve,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from scripts.run_reliability_blend_portfolio_policy_ablation import (
    _accepted_trades,
    _trade_metrics,
    _windowed_metrics,
)


DEFAULT_MANIFEST = Path(
    "data_perp/reports/reliability_blend_portfolio_policy_ablation_20260624"
    "/portfolio_policy_ablation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/head_health_portfolio_policy_ablation_20260624"
)
DEFAULT_THRESHOLDS = (0.70, 0.80, 0.90)
DEFAULT_HEAD_HEALTH_CONFIG: Dict[str, Any] = {
    "thresholds": list(DEFAULT_THRESHOLDS),
    "lookback_days": 28.0,
    "min_samples": 10,
    "shrink_samples": 49.673763999252344,
    "threshold_power": 3.9368491565880506,
    "hr_weight": 1.8920863716672436,
    "weighted_hr_weight": 0.6197699030820849,
    "weighted_hr_prediction_power": 1.2536999787776049,
    "ic_weight": 0.23283478820740225,
    "ev_weight": 1.1816637372570076,
    "adverse_weight": 0.05929740377331783,
    "health_volatility_weight": 1.1429490731537428,
    "health_volatility_min_timestamps": 20,
    "health_volatility_rank_floor": 0.70,
    "health_volatility_max_penalty": 0.592227136517074,
    "cross_head_crowding_weight": 1.3325616654123698,
    "cross_head_degrade_threshold": 0.056459322762647465,
    "cross_head_crowding_power": 0.909898393627609,
    "cross_head_crowding_max_penalty": 0.6689023208983774,
    "health_clip": 0.56627201976981,
    "rank_shift_scale": 0.01491044367175787,
    "max_rank_shift": 0.09324844561648114,
    "threshold_shift_scale": 0.008835009945747364,
    "max_threshold_shift": 0.07171829305783929,
    "threshold_floor": 0.70,
}
EPS = 1e-9


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return 0.0
    x = x[mask]
    y = y[mask]
    if float(np.nanstd(x)) <= EPS or float(np.nanstd(y)) <= EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _threshold_weights(thresholds: Iterable[float], power: float) -> Dict[float, float]:
    raw = {float(t): max(float(t), EPS) ** float(power) for t in thresholds}
    total = sum(raw.values()) or 1.0
    return {t: v / total for t, v in raw.items()}


def _weighted_mean_safe(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    default: float = 0.0,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if int(mask.sum()) == 0:
        return float(default)
    denom = float(weights[mask].sum())
    if denom <= EPS:
        return float(default)
    return float(np.dot(values[mask], weights[mask]) / denom)


def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0.0)]
    if len(weights) == 0:
        return 0.0
    sum_w = float(weights.sum())
    sum_w2 = float(np.dot(weights, weights))
    if sum_w2 <= EPS:
        return 0.0
    return float((sum_w * sum_w) / sum_w2)


def _prediction_weights(rank: np.ndarray, power: float) -> np.ndarray:
    rank = np.asarray(rank, dtype=float)
    clean_rank = np.clip(np.nan_to_num(rank, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if float(power) <= EPS:
        return np.ones_like(clean_rank, dtype=float)
    return np.power(np.maximum(clean_rank, EPS), float(power))


def _baseline_stats(
    reference: pd.DataFrame,
    *,
    thresholds: Iterable[float],
    weighted_hr_prediction_power: float = 1.0,
) -> Dict[tuple[str, float], Dict[str, float]]:
    stats: Dict[tuple[str, float], Dict[str, float]] = {}
    ref = reference.copy()
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True, errors="coerce")
    ref["rank"] = pd.to_numeric(ref["normalized_rank_score"], errors="coerce")
    ref["ret"] = pd.to_numeric(ref["net_return"], errors="coerce")
    ref["win"] = (ref["ret"] > 0.0).astype(float)
    reason = ref.get("simple_policy_exit_reason", pd.Series("", index=ref.index))
    ref["adverse"] = (
        reason.astype(str)
        .str.lower()
        .isin({"full_sl", "adverse_exit", "capital_protect"})
        .astype(float)
    )
    for strategy, strategy_rows in ref.groupby("strategy_id", sort=True):
        for threshold in thresholds:
            rows = strategy_rows.loc[strategy_rows["rank"] >= float(threshold)]
            if rows.empty:
                continue
            ret = rows["ret"].replace([np.inf, -np.inf], np.nan).dropna()
            rank = rows["rank"].to_numpy(dtype=float)
            win = rows["win"].to_numpy(dtype=float)
            prediction_weight = _prediction_weights(
                rank,
                float(weighted_hr_prediction_power),
            )
            timestamp_hr = rows.groupby("timestamp", sort=False)["win"].mean()
            stats[(str(strategy), float(threshold))] = {
                "n": float(len(rows)),
                "hr": float(rows["win"].mean()),
                "hr_ts_std": (
                    float(timestamp_hr.std(ddof=0)) if len(timestamp_hr) > 1 else 0.0
                ),
                "hr_ts_n": float(len(timestamp_hr)),
                "weighted_hr": _weighted_mean_safe(
                    win,
                    prediction_weight,
                    default=float(rows["win"].mean()),
                ),
                "weighted_hr_n_eff": _effective_sample_size(prediction_weight),
                "ret_mean": float(ret.mean()) if len(ret) else 0.0,
                "ret_std": float(max(ret.std(ddof=0), 0.005)) if len(ret) else 0.005,
                "adverse": float(rows["adverse"].mean()),
                "ic": _corr_safe(rows["rank"].to_numpy(), rows["ret"].to_numpy()),
            }
    return stats


@dataclass(frozen=True)
class HeadHealthState:
    """Training-period references used to score causal per-head health.

    The state is fitted once from the reference period and contains only
    deployable, outcome-matured replay statistics.  At replay timestamp t, the
    dynamic health score still uses only rows whose exit timestamp is before t.
    """

    thresholds: tuple[float, ...]
    baseline_stats: Dict[tuple[str, float], Dict[str, float]]

    @classmethod
    def fit(cls, reference: pd.DataFrame, config: Dict[str, Any]) -> "HeadHealthState":
        thresholds = tuple(float(v) for v in config.get("thresholds", DEFAULT_THRESHOLDS))
        return cls(
            thresholds=thresholds,
            baseline_stats=_baseline_stats(
                reference,
                thresholds=thresholds,
                weighted_hr_prediction_power=float(
                    config.get("weighted_hr_prediction_power", 1.0)
                ),
            ),
        )

    def health_by_timestamp(
        self,
        *,
        target: pd.DataFrame,
        history: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        state_config = dict(config)
        state_config["thresholds"] = list(self.thresholds)
        return _head_health_by_timestamp(
            target=target,
            history=history,
            reference=None,
            config=state_config,
            baseline_stats=self.baseline_stats,
        )


def _head_health_by_timestamp(
    *,
    target: pd.DataFrame,
    history: pd.DataFrame,
    reference: Optional[pd.DataFrame],
    config: Dict[str, Any],
    baseline_stats: Optional[Dict[tuple[str, float], Dict[str, float]]] = None,
) -> pd.DataFrame:
    thresholds = tuple(float(v) for v in config.get("thresholds", DEFAULT_THRESHOLDS))
    stats = baseline_stats
    if stats is None:
        if reference is None:
            raise ValueError("reference is required when baseline_stats is not provided")
        stats = _baseline_stats(
            reference,
            thresholds=thresholds,
            weighted_hr_prediction_power=float(
                config.get("weighted_hr_prediction_power", 1.0)
            ),
        )
    weights = _threshold_weights(thresholds, float(config["threshold_power"]))
    weighted_hr_prediction_power = float(config.get("weighted_hr_prediction_power", 1.0))
    lookback = pd.Timedelta(days=float(config["lookback_days"]))
    min_samples = int(config["min_samples"])
    shrink_samples = float(config["shrink_samples"])
    clip = float(config["health_clip"])
    volatility_weight = float(config.get("health_volatility_weight", 0.0))
    volatility_rank_floor = float(config.get("health_volatility_rank_floor", 0.70))
    volatility_min_timestamps = int(config.get("health_volatility_min_timestamps", 5))
    volatility_max_penalty = float(config.get("health_volatility_max_penalty", 1.0))

    hist = history.copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], utc=True, errors="coerce")
    hist["exit_timestamp"] = pd.to_datetime(
        hist["exit_timestamp"], utc=True, errors="coerce"
    )
    hist["rank"] = pd.to_numeric(hist["normalized_rank_score"], errors="coerce")
    hist["ret"] = pd.to_numeric(hist["net_return"], errors="coerce")
    hist["win"] = (hist["ret"] > 0.0).astype(float)
    reason = hist.get("simple_policy_exit_reason", pd.Series("", index=hist.index))
    hist["adverse"] = (
        reason.astype(str)
        .str.lower()
        .isin({"full_sl", "adverse_exit", "capital_protect"})
        .astype(float)
    )
    eval_pairs = (
        target[["timestamp", "strategy_id"]]
        .drop_duplicates()
        .sort_values(["strategy_id", "timestamp"])
    )
    rows: list[dict[str, Any]] = []
    for strategy, pairs in eval_pairs.groupby("strategy_id", sort=True):
        strategy_hist = hist.loc[hist["strategy_id"].astype(str).eq(str(strategy))].copy()
        for ts in pd.to_datetime(pairs["timestamp"], utc=True, errors="coerce"):
            score_sum = 0.0
            weight_sum = 0.0
            total_n = 0
            component_acc = {
                "hr_component": 0.0,
                "weighted_hr_component": 0.0,
                "ic_component": 0.0,
                "ev_component": 0.0,
                "adverse_component": 0.0,
            }
            health_volatility_component = 0.0
            health_volatility_penalty = 0.0
            health_volatility_ts_count = 0
            if volatility_weight > 0.0:
                volatility_base_threshold = min(
                    thresholds,
                    key=lambda value: abs(float(value) - volatility_rank_floor),
                )
                volatility_base = stats.get((str(strategy), float(volatility_base_threshold)))
                if volatility_base is not None:
                    volatility_recent = strategy_hist.loc[
                        (strategy_hist["rank"] >= volatility_rank_floor)
                        & (strategy_hist["exit_timestamp"] < ts)
                        & (strategy_hist["exit_timestamp"] >= ts - lookback)
                    ]
                    timestamp_hr = volatility_recent.groupby("timestamp", sort=False)[
                        "win"
                    ].mean()
                    health_volatility_ts_count = int(len(timestamp_hr))
                    if health_volatility_ts_count >= volatility_min_timestamps:
                        recent_std = float(timestamp_hr.std(ddof=0))
                        base_std = max(float(volatility_base.get("hr_ts_std", 0.0)), 0.05)
                        health_volatility_component = max(
                            0.0,
                            float(np.tanh((recent_std - base_std) / base_std / 3.0)),
                        )
                        health_volatility_penalty = min(
                            volatility_weight * health_volatility_component,
                            volatility_max_penalty,
                        )
            for threshold in thresholds:
                base = stats.get((str(strategy), float(threshold)))
                if base is None:
                    continue
                recent = strategy_hist.loc[
                    (strategy_hist["rank"] >= float(threshold))
                    & (strategy_hist["exit_timestamp"] < ts)
                    & (strategy_hist["exit_timestamp"] >= ts - lookback)
                ]
                n = int(len(recent))
                total_n += n
                if n < min_samples:
                    continue
                shrink = n / max(n + shrink_samples, EPS)
                thr_weight = weights[float(threshold)]
                win = recent["win"].to_numpy(dtype=float)
                ret = recent["ret"].to_numpy(dtype=float)
                adverse = recent["adverse"].to_numpy(dtype=float)
                rank = recent["rank"].to_numpy(dtype=float)
                hr = float(np.nanmean(win)) if len(win) else float(base["hr"])
                hr_se = max(
                    np.sqrt(max(base["hr"] * (1.0 - base["hr"]), EPS) / max(n, 1)),
                    0.025,
                )
                hr_z = np.tanh((hr - float(base["hr"])) / hr_se / 3.0)
                prediction_weight = _prediction_weights(
                    rank,
                    weighted_hr_prediction_power,
                )
                weighted_hr = _weighted_mean_safe(
                    win,
                    prediction_weight,
                    default=float(base.get("weighted_hr", base["hr"])),
                )
                weighted_n_eff = max(_effective_sample_size(prediction_weight), 1.0)
                base_weighted_hr = float(base.get("weighted_hr", base["hr"]))
                weighted_hr_se = max(
                    np.sqrt(
                        max(base_weighted_hr * (1.0 - base_weighted_hr), EPS)
                        / weighted_n_eff
                    ),
                    0.025,
                )
                weighted_hr_z = np.tanh(
                    (weighted_hr - base_weighted_hr) / weighted_hr_se / 3.0
                )
                ret_mean = float(np.nanmean(ret)) if len(ret) else float(base["ret_mean"])
                ret_se = max(float(base["ret_std"]) / np.sqrt(max(n, 1)), 0.003)
                ev_z = np.tanh((ret_mean - float(base["ret_mean"])) / ret_se / 3.0)
                adverse_mean = (
                    float(np.nanmean(adverse)) if len(adverse) else float(base["adverse"])
                )
                adverse_se = max(
                    np.sqrt(
                        max(float(base["adverse"]) * (1.0 - float(base["adverse"])), EPS)
                        / max(n, 1)
                    ),
                    0.025,
                )
                adverse_z = np.tanh(
                    (adverse_mean - float(base["adverse"])) / adverse_se / 3.0
                )
                ic = _corr_safe(recent["rank"].to_numpy(), ret)
                ic_z = np.tanh((ic - float(base["ic"])) * np.sqrt(max(n - 2, 1)) / 3.0)
                component = (
                    float(config["hr_weight"]) * hr_z
                    + float(config.get("weighted_hr_weight", 0.0)) * weighted_hr_z
                    + float(config["ic_weight"]) * ic_z
                    + float(config["ev_weight"]) * ev_z
                    - float(config["adverse_weight"]) * adverse_z
                )
                effective_weight = thr_weight * shrink
                score_sum += effective_weight * component
                weight_sum += effective_weight
                component_acc["hr_component"] += effective_weight * hr_z
                component_acc["weighted_hr_component"] += effective_weight * weighted_hr_z
                component_acc["ic_component"] += effective_weight * ic_z
                component_acc["ev_component"] += effective_weight * ev_z
                component_acc["adverse_component"] += effective_weight * adverse_z
            health = score_sum / max(weight_sum, EPS)
            health = float(np.clip(health, -clip, clip)) if weight_sum > 0 else 0.0
            health_before_volatility = health
            health = float(np.clip(health - health_volatility_penalty, -clip, clip))
            rec = {
                "timestamp": ts,
                "strategy_id": str(strategy),
                "head_health": health,
                "head_health_before_volatility_penalty": health_before_volatility,
                "head_health_n": int(total_n),
                "health_volatility_component": health_volatility_component,
                "health_volatility_penalty": health_volatility_penalty,
                "health_volatility_ts_count": health_volatility_ts_count,
            }
            if weight_sum > 0:
                rec.update({k: float(v / weight_sum) for k, v in component_acc.items()})
            else:
                rec.update({k: 0.0 for k in component_acc})
            rows.append(rec)
    health = pd.DataFrame(rows)
    if health.empty:
        return health
    return _apply_cross_head_crowding(health, config=config, clip=clip)


def _apply_cross_head_crowding(
    health: pd.DataFrame,
    *,
    config: Dict[str, Any],
    clip: float,
) -> pd.DataFrame:
    work = health.copy()
    weight = float(config.get("cross_head_crowding_weight", 0.0))
    degrade_threshold = float(config.get("cross_head_degrade_threshold", 0.0))
    crowding_power = float(config.get("cross_head_crowding_power", 1.0))
    max_penalty = float(config.get("cross_head_crowding_max_penalty", 1.0))
    pre_cross = pd.to_numeric(work["head_health"], errors="coerce").fillna(0.0)
    degradation = np.maximum(degrade_threshold - pre_cross.to_numpy(dtype=float), 0.0)
    work["head_health_before_cross_head_penalty"] = pre_cross
    work["_head_degradation"] = degradation
    grouped = work.groupby("timestamp", sort=False)["_head_degradation"]
    crowding = grouped.agg(
        cross_head_degraded_share=lambda values: float(np.mean(np.asarray(values) > 0.0)),
        cross_head_degradation_mean="mean",
    ).reset_index()
    crowding["cross_head_crowding_raw"] = (
        crowding["cross_head_degradation_mean"].astype(float)
        * np.power(
            np.clip(crowding["cross_head_degraded_share"].astype(float), 0.0, 1.0),
            crowding_power,
        )
    )
    crowding["cross_head_crowding_penalty"] = np.minimum(
        weight * crowding["cross_head_crowding_raw"].astype(float),
        max_penalty,
    )
    work = work.merge(crowding, on="timestamp", how="left")
    penalty = pd.to_numeric(
        work["cross_head_crowding_penalty"],
        errors="coerce",
    ).fillna(0.0)
    work["head_health"] = np.clip(pre_cross - penalty, -clip, clip)
    return work.drop(columns=["_head_degradation"])


def _apply_head_health(
    target: pd.DataFrame,
    *,
    history: pd.DataFrame,
    reference: pd.DataFrame,
    config: Dict[str, Any],
    state: Optional[HeadHealthState] = None,
) -> pd.DataFrame:
    work = target.copy()
    health_state = state if state is not None else HeadHealthState.fit(reference, config)
    health = health_state.health_by_timestamp(target=work, history=history, config=config)
    work = work.merge(health, on=["timestamp", "strategy_id"], how="left")
    work["head_health"] = pd.to_numeric(work["head_health"], errors="coerce").fillna(0.0)
    original_rank = pd.to_numeric(work["normalized_rank_score"], errors="coerce")
    original_score = pd.to_numeric(work["calibrated_score"], errors="coerce")
    original_threshold = pd.to_numeric(work["base_strategy_threshold"], errors="coerce")
    score_delta = np.clip(
        float(config["rank_shift_scale"]) * work["head_health"].to_numpy(dtype=float),
        -float(config["max_rank_shift"]),
        float(config["max_rank_shift"]),
    )
    threshold_delta = np.clip(
        float(config["threshold_shift_scale"]) * work["head_health"].to_numpy(dtype=float),
        -float(config["max_threshold_shift"]),
        float(config["max_threshold_shift"]),
    )
    work["head_health_score_delta"] = score_delta
    work["head_health_threshold_delta"] = threshold_delta
    work["head_health_original_rank_score"] = original_rank
    work["head_health_original_base_threshold"] = original_threshold
    work["normalized_rank_score"] = np.clip(original_rank + score_delta, 0.0, 1.0)
    work["auction_rank_score"] = work["normalized_rank_score"]
    work["calibrated_score"] = original_score + score_delta
    work["base_strategy_threshold"] = np.clip(
        original_threshold - threshold_delta,
        float(config["threshold_floor"]),
        0.999,
    )
    work["deployment_rank_threshold"] = work["base_strategy_threshold"]
    return work


def _evaluate_variant(
    *,
    sample: str,
    variant: str,
    candidates: pd.DataFrame,
    params: PortfolioPolicyParams,
    ev_curve: Dict[str, Any],
    market_mode: str,
    output_dir: Optional[Path] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions, equity, metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )
    accepted = _accepted_trades(candidates, decisions)
    if output_dir is not None:
        sample_dir = output_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        accepted.to_parquet(sample_dir / f"{variant}_accepted_trades.parquet", index=False)
        decisions.to_parquet(sample_dir / f"{variant}_decisions.parquet", index=False)
        equity.to_parquet(sample_dir / f"{variant}_equity_curve.parquet", index=False)
    max_ts = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce").max()
    summary = {
        "sample": sample,
        "variant": variant,
        "candidate_rows": int(len(candidates)),
        "timestamp_min": pd.to_datetime(candidates["timestamp"], utc=True).min().isoformat(),
        "timestamp_max": pd.to_datetime(candidates["timestamp"], utc=True).max().isoformat(),
        "objective": metrics.get("objective"),
        "net_pnl": metrics.get("net_pnl"),
        "gross_pnl": metrics.get("gross_pnl"),
        "compounded_return": metrics.get("compounded_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "trade_count": metrics.get("trade_count"),
        "full_sl_rate": metrics.get("full_sl_rate"),
        "timeout_rate": metrics.get("timeout_rate"),
        "strategy_concentration": metrics.get("strategy_concentration"),
        "side_concentration": metrics.get("side_concentration"),
        "missed_high_confidence_trades": metrics.get("missed_high_confidence_trades"),
    }
    windows = _windowed_metrics(
        accepted,
        sample=sample,
        variant=variant,
        max_timestamp=max_ts,
    )
    strategy_windows = _windowed_metrics(
        accepted,
        sample=sample,
        variant=variant,
        max_timestamp=max_ts,
        group_cols=("strategy_id",),
    )
    return summary, windows, strategy_windows


def _window_map(rows: list[dict[str, Any]]) -> Dict[str, dict[str, Any]]:
    return {str(row["window"]): row for row in rows}


def _objective_from_windows(
    candidate_windows: list[dict[str, Any]],
    baseline_windows: Dict[str, dict[str, Any]],
) -> float:
    cand = _window_map(candidate_windows)
    score = 0.0
    penalties = 0.0
    weights = {"all": 1.0, "last_4w": 0.55, "last_2w": 0.35, "last_1w": 0.25}
    for window, weight in weights.items():
        c = cand.get(window, {})
        b = baseline_windows.get(window, {})
        c_pnl = float(c.get("net_pnl", 0.0))
        b_pnl = float(b.get("net_pnl", 0.0))
        denom = max(abs(b_pnl), 1_000.0)
        delta = (c_pnl - b_pnl) / denom
        score += weight * delta
        if delta < 0:
            penalties += weight * abs(delta) * 2.0
    c_all = cand.get("all", {})
    b_all = baseline_windows.get("all", {})
    trade_ratio = float(c_all.get("trade_count", 0.0)) / max(
        float(b_all.get("trade_count", 1.0)),
        1.0,
    )
    if trade_ratio < 0.75:
        penalties += (0.75 - trade_ratio) * 2.0
    if trade_ratio > 1.35:
        penalties += (trade_ratio - 1.35) * 0.75
    return float(score - penalties)


def _read_base_config(path: Optional[Path]) -> Dict[str, Any]:
    config = dict(DEFAULT_HEAD_HEALTH_CONFIG)
    if path is None:
        return config
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "best_config" in data:
        data = data["best_config"]
    if not isinstance(data, dict):
        raise ValueError(f"Base config must be a JSON object: {path}")
    config.update(data)
    config["thresholds"] = list(config.get("thresholds", DEFAULT_THRESHOLDS))
    config["threshold_floor"] = 0.70
    config["health_volatility_rank_floor"] = 0.70
    return config


def _suggest_config(
    trial: optuna.Trial,
    *,
    search_layer: str,
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    config = dict(base_config)
    tune_signal = search_layer in {"joint", "signal"}
    tune_weights = search_layer in {"joint", "weights"}
    tune_action = search_layer in {"joint", "action"}
    if tune_signal:
        config["lookback_days"] = trial.suggest_categorical(
            "lookback_days",
            [3.0, 7.0, 14.0, 28.0],
        )
        config["min_samples"] = trial.suggest_categorical(
            "min_samples",
            [5, 10, 20, 30],
        )
        config["shrink_samples"] = trial.suggest_float("shrink_samples", 10.0, 120.0)
        config["threshold_power"] = trial.suggest_float("threshold_power", 0.0, 4.0)
        config["weighted_hr_prediction_power"] = trial.suggest_float(
            "weighted_hr_prediction_power",
            0.5,
            4.0,
        )
        config["health_volatility_min_timestamps"] = trial.suggest_categorical(
            "health_volatility_min_timestamps",
            [5, 10, 20],
        )
        config["cross_head_degrade_threshold"] = trial.suggest_float(
            "cross_head_degrade_threshold",
            -0.20,
            0.20,
        )
        config["cross_head_crowding_power"] = trial.suggest_float(
            "cross_head_crowding_power",
            0.5,
            3.0,
        )
    if tune_weights:
        config["hr_weight"] = trial.suggest_float("hr_weight", 0.0, 2.0)
        config["weighted_hr_weight"] = trial.suggest_float(
            "weighted_hr_weight",
            0.0,
            2.0,
        )
        config["ic_weight"] = trial.suggest_float("ic_weight", 0.0, 1.5)
        config["ev_weight"] = trial.suggest_float("ev_weight", 0.0, 2.0)
        config["adverse_weight"] = trial.suggest_float("adverse_weight", 0.0, 2.0)
        config["health_volatility_weight"] = trial.suggest_float(
            "health_volatility_weight",
            0.0,
            2.0,
        )
        config["cross_head_crowding_weight"] = trial.suggest_float(
            "cross_head_crowding_weight",
            0.0,
            2.0,
        )
    if tune_action:
        config["health_clip"] = trial.suggest_float("health_clip", 0.5, 3.0)
        config["rank_shift_scale"] = trial.suggest_float("rank_shift_scale", 0.0, 0.08)
        config["max_rank_shift"] = trial.suggest_float("max_rank_shift", 0.005, 0.10)
        config["threshold_shift_scale"] = trial.suggest_float(
            "threshold_shift_scale",
            0.0,
            0.08,
        )
        config["max_threshold_shift"] = trial.suggest_float(
            "max_threshold_shift",
            0.005,
            0.08,
        )
        config["health_volatility_max_penalty"] = trial.suggest_float(
            "health_volatility_max_penalty",
            0.05,
            1.0,
        )
        config["cross_head_crowding_max_penalty"] = trial.suggest_float(
            "cross_head_crowding_max_penalty",
            0.05,
            1.0,
        )
    config["threshold_floor"] = 0.70
    config["health_volatility_rank_floor"] = 0.70
    return config


def _component_configs(best: Dict[str, Any]) -> dict[str, Dict[str, Any]]:
    out = {
        "head_health_best": dict(best),
        "head_health_rank_only": {**best, "threshold_shift_scale": 0.0},
        "head_health_threshold_only": {**best, "rank_shift_scale": 0.0},
        "head_health_no_weighted_hr": {**best, "weighted_hr_weight": 0.0},
        "head_health_no_volatility": {**best, "health_volatility_weight": 0.0},
        "head_health_no_cross_head": {**best, "cross_head_crowding_weight": 0.0},
        "head_health_no_ic": {**best, "ic_weight": 0.0},
        "head_health_no_ev": {**best, "ev_weight": 0.0},
        "head_health_no_adverse": {**best, "adverse_weight": 0.0},
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-evaluations", type=int, default=160)
    parser.add_argument(
        "--skip-hpo",
        action="store_true",
        help=(
            "Replay the default or --base-config HeadHealth policy without "
            "running a new Optuna search."
        ),
    )
    parser.add_argument("--market-mode", type=str, default="perps")
    parser.add_argument("--seed", type=int, default=230623)
    parser.add_argument(
        "--search-layer",
        choices=("joint", "signal", "weights", "action"),
        default="joint",
        help="Tune all params jointly or one bounded layer around --base-config.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="JSON object or manifest containing best_config used as the layer base.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    train_path = Path(manifest["train_candidates"])
    oos_path = Path(manifest["oos_candidates"])
    oos_sample_name = str(manifest.get("oos_sample_name") or "oos_jun15_22")
    train = normalise_candidate_table(pd.read_parquet(train_path))
    oos = normalise_candidate_table(pd.read_parquet(oos_path))
    params = portfolio_policy_params_from_live_config(
        manifest["variant_params"]["refit_bar4_strategy_bar2"]
    )
    params = replace(params, global_threshold_floor=max(float(params.global_threshold_floor), 0.70))
    ev_curve = fit_monotone_ev_curve(train)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = _read_base_config(args.base_config)

    baseline_summary, baseline_windows, baseline_strategy = _evaluate_variant(
        sample="historical_refit",
        variant="static_refit_bar4_strategy_bar2",
        candidates=train,
        params=params,
        ev_curve=ev_curve,
        market_mode=args.market_mode,
        output_dir=output_dir,
    )
    baseline_window_map = _window_map(baseline_windows)

    best_config: Optional[Dict[str, Any]] = None
    best_objective = float("-inf")
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_config, best_objective
        config = _suggest_config(
            trial,
            search_layer=str(args.search_layer),
            base_config=base_config,
        )
        transformed = _apply_head_health(
            train,
            history=train,
            reference=train,
            config=config,
        )
        _, windows, _ = _evaluate_variant(
            sample="historical_refit",
            variant=f"trial_{trial.number}",
            candidates=transformed,
            params=params,
            ev_curve=ev_curve,
            market_mode=args.market_mode,
            output_dir=None,
        )
        value = _objective_from_windows(windows, baseline_window_map)
        trial.set_user_attr("config", config)
        trial.set_user_attr("windows", windows)
        trial_rows.append(
            {
                "trial": int(trial.number),
                "objective": float(value),
                **{f"config_{k}": _json_safe(v) for k, v in config.items() if k != "thresholds"},
            }
        )
        if value > best_objective:
            best_objective = float(value)
            best_config = dict(config)
        return float(value)

    if bool(args.skip_hpo):
        best_config = dict(base_config)
        transformed = _apply_head_health(
            train,
            history=train,
            reference=train,
            config=best_config,
        )
        _, windows, _ = _evaluate_variant(
            sample="historical_refit",
            variant="fixed_config",
            candidates=transformed,
            params=params,
            ev_curve=ev_curve,
            market_mode=args.market_mode,
            output_dir=None,
        )
        best_objective = _objective_from_windows(windows, baseline_window_map)
        trial_rows.append(
            {
                "trial": 0,
                "objective": float(best_objective),
                "fixed_config": True,
                **{
                    f"config_{k}": _json_safe(v)
                    for k, v in best_config.items()
                    if k != "thresholds"
                },
            }
        )
    else:
        if int(args.max_evaluations) <= 0:
            raise ValueError("--max-evaluations must be positive unless --skip-hpo is set")
        sampler = optuna.samplers.TPESampler(
            seed=int(args.seed),
            n_startup_trials=min(40, int(args.max_evaluations)),
            multivariate=True,
            group=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=int(args.max_evaluations), show_progress_bar=False)
    if best_config is None:
        best_config = dict(base_config)
    pd.DataFrame(trial_rows).to_csv(output_dir / "head_health_hpo_trials.csv", index=False)

    summaries = [baseline_summary]
    windows_all = list(baseline_windows)
    strategy_all = list(baseline_strategy)
    variant_configs = _component_configs(best_config)
    for variant, config in variant_configs.items():
        transformed = _apply_head_health(
            train,
            history=train,
            reference=train,
            config=config,
        )
        summary, windows, strategy = _evaluate_variant(
            sample="historical_refit",
            variant=variant,
            candidates=transformed,
            params=params,
            ev_curve=ev_curve,
            market_mode=args.market_mode,
            output_dir=output_dir,
        )
        summaries.append(summary)
        windows_all.extend(windows)
        strategy_all.extend(strategy)

    oos_baseline_summary, oos_baseline_windows, oos_baseline_strategy = _evaluate_variant(
        sample=oos_sample_name,
        variant="static_refit_bar4_strategy_bar2",
        candidates=oos,
        params=params,
        ev_curve=ev_curve,
        market_mode=args.market_mode,
        output_dir=output_dir,
    )
    summaries.append(oos_baseline_summary)
    windows_all.extend(oos_baseline_windows)
    strategy_all.extend(oos_baseline_strategy)
    history_for_oos = pd.concat([train, oos], ignore_index=True, sort=False)
    for variant, config in variant_configs.items():
        transformed_oos = _apply_head_health(
            oos,
            history=history_for_oos,
            reference=train,
            config=config,
        )
        summary, windows, strategy = _evaluate_variant(
            sample=oos_sample_name,
            variant=variant,
            candidates=transformed_oos,
            params=params,
            ev_curve=ev_curve,
            market_mode=args.market_mode,
            output_dir=output_dir,
        )
        summaries.append(summary)
        windows_all.extend(windows)
        strategy_all.extend(strategy)

    summary_df = pd.DataFrame(summaries)
    windows_df = pd.DataFrame(windows_all)
    strategy_df = pd.DataFrame(strategy_all)
    summary_df.to_csv(output_dir / "head_health_policy_summary.csv", index=False)
    windows_df.to_csv(output_dir / "head_health_policy_windows.csv", index=False)
    strategy_df.to_csv(output_dir / "head_health_policy_windows_by_strategy.csv", index=False)

    report = {
        "generated_by": "run_head_health_portfolio_policy_ablation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_variant": "refit_bar4_strategy_bar2",
        "train_candidates": str(train_path),
        "oos_candidates": str(oos_path),
        "market_mode": args.market_mode,
        "costs_included": True,
        "max_evaluations": int(args.max_evaluations),
        "skip_hpo": bool(args.skip_hpo),
        "search_layer": str(args.search_layer),
        "base_config_path": str(args.base_config) if args.base_config is not None else None,
        "best_hpo_objective": float(best_objective),
        "best_config": best_config,
        "baseline_policy_params": params.to_live_config(),
        "outputs": {
            "summary": str(output_dir / "head_health_policy_summary.csv"),
            "windows": str(output_dir / "head_health_policy_windows.csv"),
            "windows_by_strategy": str(output_dir / "head_health_policy_windows_by_strategy.csv"),
            "trials": str(output_dir / "head_health_hpo_trials.csv"),
        },
    }
    (output_dir / "head_health_policy_manifest.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2)[:8000])
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
