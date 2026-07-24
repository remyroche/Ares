#!/usr/bin/env python3
"""Run staged, constraint-aware 1m capital/trailing policy experiments."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.timestamp_contract import causal_signal_times  # noqa: E402
from extreme_price_movements.simple_policy_1m_ablation import (  # noqa: E402
    FAMILY_CURRENT,
    evaluate_results,
    objective_score_fast,
    params_to_vector,
    simulate_1m_paths,
)
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_CONSTANT,
    FAMILY_EXPONENTIAL,
    FAMILY_MULTILAYER,
    FAMILY_NAMES,
    FAMILY_RATIONAL,
    FAMILY_SIGMOID,
    FAMILY_SPLINE,
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
    constrained_params_to_vector,
    simulate_constrained_1m_paths,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
    _write_json,
)


CAPITAL_FAMILIES = (
    FAMILY_CONSTANT,
    FAMILY_MULTILAYER,
    FAMILY_SIGMOID,
    FAMILY_EXPONENTIAL,
    FAMILY_RATIONAL,
    FAMILY_SPLINE,
)

INNER_FOLDS = {
    "fold_1": {"search_end": "2026-05-08", "purge": "2026-05-08", "inner_start": "2026-05-09", "inner_end": "2026-05-14"},
    "fold_2": {"search_end": "2026-05-17", "purge": "2026-05-17", "inner_start": "2026-05-18", "inner_end": "2026-05-31"},
    "fold_3": {"search_end": "2026-05-31", "purge": "2026-05-31", "inner_start": "2026-06-01", "inner_end": "2026-06-14"},
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _indices_between(data: "ExperimentData", start: str, end: str) -> np.ndarray:
    ts = pd.to_datetime(data.rows["timestamp"], utc=True)
    mask = ts.ge(pd.Timestamp(start, tz="UTC")) & ts.lt(pd.Timestamp(end, tz="UTC")) & data.valid
    return np.flatnonzero(mask.to_numpy()).astype(np.int64)


def _causal_entry_atr(
    rows: pd.DataFrame,
    *,
    store_root: Path,
    deployed_by_side: Mapping[str, Mapping[str, Any]],
    parent_summary: Path,
    warmup_hours: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Compute entry-frozen live-style ATR from completed pre-entry 1m bars."""
    parent = pd.read_csv(parent_summary).set_index("side")
    side_contract: dict[str, dict[str, float]] = {}
    for side in ("long", "short"):
        row = parent.loc[side]
        side_contract[side] = {
            "power": float(row["param_atr_power"]),
            "multiplier": float(row["param_atr_multiplier"]),
            "median": float(row["param_policy_median_barrier_frac"]),
        }
    out = np.full(len(rows), np.nan, dtype=np.float64)
    audit_rows: list[dict[str, Any]] = []
    store = PartitionedOHLCVStore(str(store_root), timeframe="1m")
    for symbol, group in rows.groupby("symbol", sort=True):
        _, decision_ts = causal_signal_times(group, timeframe="1h")
        timestamps = pd.Series(decision_ts, index=group.index)
        frame = store.load(
            str(symbol),
            columns=["ts", "open", "high", "low", "close"],
            start_ts=timestamps.min() - pd.Timedelta(hours=warmup_hours + 2),
            end_ts=timestamps.max(),
        )
        if frame is None or frame.empty:
            continue
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        idx = frame.index.tz_localize("UTC") if frame.index.tz is None else frame.index.tz_convert("UTC")
        frame = frame.copy()
        frame.index = idx
        numeric = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        hourly = numeric.resample("1h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        )
        counts = numeric["close"].resample("1h", label="left", closed="left").count()
        hourly["count"] = counts
        for row_i, timestamp, side_value in zip(group.index, timestamps, group["side"]):
            expected = pd.date_range(
                timestamp - pd.Timedelta(hours=warmup_hours),
                timestamp - pd.Timedelta(hours=1),
                freq="1h",
                tz="UTC",
            )
            window = hourly.reindex(expected)
            complete = bool(len(window) == warmup_hours and (window["count"] == 60).all() and window[["high", "low", "close"]].notna().all().all())
            if not complete:
                audit_rows.append({"row": int(row_i), "symbol": str(symbol), "timestamp": timestamp, "status": "incomplete"})
                continue
            previous_close = window["close"].shift(1)
            true_range = pd.concat(
                [
                    window["high"] - window["low"],
                    (window["high"] - previous_close).abs(),
                    (window["low"] - previous_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            raw_atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False).mean().iloc[-1]
            raw_fraction = max(float(raw_atr / window["close"].iloc[-1]), 0.005)
            side = "long" if float(side_value) > 0.0 else "short"
            contract = side_contract[side]
            median = max(contract["median"], 1e-9)
            effective = contract["multiplier"] * median * (raw_fraction / median) ** contract["power"]
            out[int(row_i)] = effective
            audit_rows.append(
                {
                    "row": int(row_i), "symbol": str(symbol), "timestamp": timestamp, "side": side,
                    "raw_atr_fraction": raw_fraction, "effective_atr_fraction": effective,
                    "status": "ok", "last_completed_hour": expected[-1],
                }
            )
    audit = pd.DataFrame(audit_rows).sort_values("row").reset_index(drop=True)
    manifest = {
        "definition": "completed 1m -> hourly TR -> EWM(alpha=1/14, adjust=False), through t-1h; floor raw ATR/close at 0.005; deployed side power/multiplier/median; freeze at entry",
        "warmup_hours": int(warmup_hours),
        "valid_rows": int(np.isfinite(out).sum()),
        "invalid_rows": int((~np.isfinite(out)).sum()),
        "coverage": float(np.isfinite(out).mean()),
        "side_contract": side_contract,
        "forbidden_timing": "hourly candle stamped t is [t,t+1h) and is excluded because replay enters at t",
    }
    return out, audit, manifest


class ExperimentData:
    def __init__(
        self,
        rows: pd.DataFrame,
        open0: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        valid: np.ndarray,
        atr_frac: np.ndarray,
        spec: ConstrainedReplaySpec,
        deployed_by_side: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.rows = rows
        self.open0, self.high, self.low, self.close = open0, high, low, close
        self.valid = valid & np.isfinite(atr_frac)
        self.atr_frac = atr_frac
        self.spec = spec
        self.deployed_by_side = deployed_by_side
        self.side = pd.to_numeric(rows["side"], errors="coerce").to_numpy(dtype=np.float64)
        self.entry_spread = pd.to_numeric(rows["spread_cost_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.exit_spread = pd.to_numeric(rows["exit_spread_cost_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.timestamps = pd.to_datetime(rows["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
        self.symbol_codes = pd.Categorical(rows["symbol"].astype(str)).codes.astype(np.int32)
        rank_column = "ev_rank_pct" if "ev_rank_pct" in rows.columns else "rank_pct"
        self.rank = pd.to_numeric(rows[rank_column], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)

    def simulate(self, indices: np.ndarray, params_by_side: Mapping[str, Mapping[str, Any]], family: int) -> dict[str, np.ndarray]:
        ordered = np.asarray(indices, dtype=np.int64)
        keys = (
            "exit_bars", "exit_price", "gross_return", "net_return", "reason", "mfe", "mae",
            "capital_first_bar", "trailing_first_bar", "capital_binding_bars", "initial_capital_active", "order_valid",
            "trailing_layer_first_bar", "trailing_layer_binding_bars", "trailing_exit_layer",
        )
        outputs = {
            "exit_bars": np.full(len(ordered), -1, dtype=np.int32),
            "exit_price": np.full(len(ordered), np.nan),
            "gross_return": np.full(len(ordered), np.nan),
            "net_return": np.full(len(ordered), np.nan),
            "reason": np.zeros(len(ordered), dtype=np.int8),
            "mfe": np.full(len(ordered), np.nan),
            "mae": np.full(len(ordered), np.nan),
            "capital_first_bar": np.full(len(ordered), -1, dtype=np.int32),
            "trailing_first_bar": np.full(len(ordered), -1, dtype=np.int32),
            "capital_binding_bars": np.zeros(len(ordered), dtype=np.int32),
            "initial_capital_active": np.zeros(len(ordered), dtype=bool),
            "order_valid": np.ones(len(ordered), dtype=bool),
            "trailing_layer_first_bar": np.full((len(ordered), 3), -1, dtype=np.int32),
            "trailing_layer_binding_bars": np.zeros((len(ordered), 3), dtype=np.int32),
            "trailing_exit_layer": np.full(len(ordered), -1, dtype=np.int8),
        }
        for side_name, sign in (("long", 1.0), ("short", -1.0)):
            local = np.flatnonzero(self.side[ordered] * sign > 0.0)
            if not len(local):
                continue
            result = simulate_constrained_1m_paths(
                ordered[local], self.open0, self.high, self.low, self.close, self.side,
                self.atr_frac, self.entry_spread, self.exit_spread,
                constrained_params_to_vector(params_by_side[side_name]), int(family),
                self.spec.fee_per_side, self.spec.stop_base_gap_bps,
                self.spec.stop_through_fraction, self.spec.stop_max_gap_bps,
                self.spec.capital_trail_epsilon_atr,
            )
            for key, values in zip(keys, result):
                outputs[key][local] = values
        return outputs

    def simulate_deployed(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        ordered = np.asarray(indices, dtype=np.int64)
        keys = ("exit_bars", "exit_price", "gross_return", "net_return", "reason", "mfe", "mae")
        outputs = {k: np.full(len(ordered), -1 if k == "exit_bars" else np.nan) for k in keys}
        outputs["exit_bars"] = outputs["exit_bars"].astype(np.int32)
        outputs["reason"] = np.zeros(len(ordered), dtype=np.int8)
        for side_name, sign in (("long", 1.0), ("short", -1.0)):
            local = np.flatnonzero(self.side[ordered] * sign > 0.0)
            if not len(local):
                continue
            result = simulate_1m_paths(
                ordered[local], self.open0, self.high, self.low, self.close, self.side,
                self.atr_frac, self.entry_spread, self.exit_spread,
                params_to_vector(self.deployed_by_side[side_name]), FAMILY_CURRENT, 0,
                self.spec.fee_per_side, self.spec.stop_base_gap_bps,
                self.spec.stop_through_fraction, self.spec.stop_max_gap_bps,
            )
            for key, values in zip(keys, result):
                outputs[key][local] = values
        for key, fill, dtype in (
            ("capital_first_bar", -1, np.int32), ("trailing_first_bar", -1, np.int32),
            ("capital_binding_bars", 0, np.int32), ("initial_capital_active", False, bool),
            ("order_valid", True, bool),
        ):
            outputs[key] = np.full(len(ordered), fill, dtype=dtype)
        return outputs


def _objective(data: ExperimentData, indices: np.ndarray, outputs: Mapping[str, np.ndarray]) -> tuple[float, dict[str, Any]]:
    week_ts = pd.to_datetime(data.rows.iloc[indices]["timestamp"], utc=True).dt.tz_localize(None)
    weeks = pd.factorize(week_ts.dt.to_period("W").astype(str), sort=True)[0].astype(np.int32)
    score, total, worst, dd, n = objective_score_fast(
        data.timestamps[indices], data.symbol_codes[indices], data.rank[indices], weeks,
        outputs["exit_bars"], outputs["net_return"], data.spec.bar_minutes,
    )
    valid_order = float(np.mean(outputs["order_valid"])) if len(indices) else 0.0
    initial = float(np.mean(outputs["initial_capital_active"])) if len(indices) else 0.0
    return float(score), {
        "objective": float(score), "net_pnl_bankroll": float(total), "worst_week": float(worst),
        "max_drawdown": float(dd), "n_trades": int(n), "ordering_valid_rate": valid_order,
        "initial_capital_active_rate": initial,
    }


def _side_defaults(deployed: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "trailing_activation_decay_half_life_minutes", "trailing_activation_decay_start_minutes",
        "trailing_activation_min_mult", "adverse_exit_enabled", "adverse_exit_min_mae_atr",
        "adverse_exit_min_speed_per_15m", "adverse_exit_theta", "adverse_exit_fast_minutes",
        "adverse_exit_max_mfe_atr", "trailing_activation_cap_pct",
    )
    return {key: deployed.get(key) for key in keys}


def _bounded_product(value: float, delta: float, sign: float, low: float, high: float) -> float:
    return float(np.clip(value * math.exp(sign * delta), low, high))


def _suggest_params(
    trial: optuna.Trial,
    *,
    family: int,
    deployed_by_side: Mapping[str, Mapping[str, Any]],
    joint: bool,
) -> tuple[dict[str, dict[str, Any]], float, dict[str, Any]]:
    if joint:
        global_sl = trial.suggest_float("sl_mult", 1.5, 4.0)
        global_act = trial.suggest_float("trailing_activation_mult", 0.5, 3.0, log=True)
        global_power = trial.suggest_float("trailing_power", 1.1, 2.5)
        global_div = trial.suggest_float("trailing_squash_divisor", 1.0, 5.0, log=True)
        global_beta = trial.suggest_float("giveback_beta", 0.15, 0.95)
        delta_sl = trial.suggest_float("side_delta_sl", -0.20, 0.20)
        delta_act = trial.suggest_float("side_delta_activation", -0.20, 0.20)
        delta_beta = trial.suggest_float("side_delta_beta", -0.20, 0.20)
    else:
        global_sl = global_act = global_power = global_div = global_beta = np.nan
        delta_sl = delta_act = delta_beta = 0.0

    capital = family != FAMILY_TRAILING_ONLY
    if capital:
        entry_ratio = trial.suggest_float("entry_capital_ratio", 0.50, 0.95)
        terminal_ratio = trial.suggest_float("terminal_excess_ratio", 0.05, 0.80, log=True)
        center = trial.suggest_float("transition_center", 0.25, 6.0, log=True)
        shape = trial.suggest_float("transition_shape", 0.25, 4.0, log=True)
        delta_entry = trial.suggest_float("side_delta_entry", -0.20, 0.20)
        delta_shape = trial.suggest_float("side_delta_shape", -0.20, 0.20)
        clamp_mode = trial.suggest_categorical("clamp_mode", ("none", "min", "max", "both"))
        excess_min = trial.suggest_float("excess_min_ratio", 0.05, 0.80, log=True) if clamp_mode in ("min", "both") else 0.0
        excess_max = trial.suggest_float("excess_max_ratio", max(excess_min, 0.10), 2.0, log=True) if clamp_mode in ("max", "both") else 1e6
        use_current = trial.suggest_categorical("use_current_buffer", (False, True))
        current_ratio = (
            entry_ratio * trial.suggest_float("current_buffer_fraction_of_entry_gap", 0.10, 1.0)
            if use_current else 0.0
        )
        mix1 = trial.suggest_float("mixture_logit_1", -2.0, 2.0) if family == FAMILY_MULTILAYER else 0.0
        mix2 = trial.suggest_float("mixture_logit_2", -2.0, 2.0) if family == FAMILY_MULTILAYER else 0.0
        if family == FAMILY_SPLINE:
            retains = []
            previous = 1.0
            for knot in range(1, 6):
                q = trial.suggest_float(f"spline_stick_{knot}", 0.0, 1.0)
                previous = terminal_ratio + (previous - terminal_ratio) * q
                retains.append(previous)
        else:
            retains = [0.85, 0.70, 0.55, 0.40, terminal_ratio]
    else:
        entry_ratio = terminal_ratio = center = shape = 0.0
        delta_entry = delta_shape = 0.0
        clamp_mode, excess_min, excess_max, current_ratio = "none", 0.0, 1e6, 0.0
        mix1 = mix2 = 0.0
        retains = [1.0] * 5

    params_by_side: dict[str, dict[str, Any]] = {}
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        deployed = deployed_by_side[side_name]
        params = _side_defaults(deployed)
        if joint:
            params.update(
                {
                    "sl_mult": _bounded_product(global_sl, delta_sl, sign, 1.5, 4.0),
                    "trailing_activation_mult": _bounded_product(global_act, delta_act, sign, 0.5, 3.0),
                    "trailing_power": global_power,
                    "trailing_squash_divisor": global_div,
                    "giveback_beta": _bounded_product(global_beta, delta_beta, sign, 0.15, 0.95),
                }
            )
        else:
            params.update(
                {
                    "sl_mult": float(deployed["sl_mult"]),
                    "trailing_activation_mult": float(deployed["trailing_activation_mult"]),
                    "trailing_power": float(deployed["trailing_power"]),
                    "trailing_squash_divisor": float(deployed["trailing_squash_divisor"]),
                    "giveback_beta": float(deployed["giveback_beta"]),
                }
            )
        if capital:
            params.update(
                {
                    "entry_capital_ratio": _bounded_product(entry_ratio, delta_entry, sign, 0.50, 0.95),
                    "terminal_excess_ratio": terminal_ratio,
                    "transition_center": center,
                    "transition_shape": _bounded_product(shape, delta_shape, sign, 0.25, 4.0),
                    "mixture_logit_1": mix1,
                    "mixture_logit_2": mix2,
                    "excess_min_ratio": excess_min,
                    "excess_max_ratio": max(excess_max, excess_min),
                    "current_distance_sl_ratio": current_ratio,
                    "spline_retains": retains,
                }
            )
        params_by_side[side_name] = params
    deltas = np.asarray([delta_sl, delta_act, delta_beta, delta_entry, delta_shape], dtype=float)
    shrinkage_penalty = float(0.002 * np.sum((deltas / 0.20) ** 2))
    metadata = {"clamp_mode": clamp_mode, "use_current_buffer": current_ratio > 0.0, "side_deltas": deltas.tolist()}
    return params_by_side, shrinkage_penalty, metadata


def _optimise(
    data: ExperimentData,
    indices: np.ndarray,
    *,
    family: int,
    joint: bool,
    trials_per_seed: int,
    seeds: list[int],
    sampler_kind: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    best_value = -1e100
    best_params: dict[str, dict[str, Any]] | None = None
    best_meta: dict[str, Any] = {}
    seed_summaries = []
    for seed in seeds:
        if sampler_kind == "sobol":
            sampler = optuna.samplers.QMCSampler(qmc_type="sobol", scramble=True, seed=int(seed), warn_independent_sampling=False)
        else:
            sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True, group=True, n_startup_trials=min(32, max(8, trials_per_seed // 4)))
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: optuna.Trial) -> float:
            params_by_side, penalty, metadata = _suggest_params(
                trial, family=family, deployed_by_side=data.deployed_by_side, joint=joint
            )
            outputs = data.simulate(indices, params_by_side, family)
            score, diag = _objective(data, indices, outputs)
            for key, value in diag.items():
                trial.set_user_attr(key, value)
            trial.set_user_attr("metadata", metadata)
            trial.set_user_attr("runtime_params", params_by_side)
            trial.set_user_attr("shrinkage_penalty", penalty)
            if family != FAMILY_TRAILING_ONLY and (
                diag["ordering_valid_rate"] < 0.999999 or diag["initial_capital_active_rate"] < 0.999999
            ):
                return -1e6
            return float(score - penalty)

        study.optimize(objective, n_trials=int(trials_per_seed), show_progress_bar=False, gc_after_trial=False)
        chosen = study.best_trial.user_attrs["runtime_params"]
        penalty = float(study.best_trial.user_attrs["shrinkage_penalty"])
        metadata = study.best_trial.user_attrs["metadata"]
        seed_summaries.append(
            {"seed": int(seed), "best_value_penalized": float(study.best_value), "best_trial": int(study.best_trial.number), "best_user_attrs": dict(study.best_trial.user_attrs)}
        )
        if study.best_value > best_value:
            best_value = float(study.best_value)
            best_params = chosen
            best_meta = {"penalty": penalty, "metadata": metadata, "trial_params": dict(study.best_trial.params), "seed": int(seed)}
    if best_params is None:
        raise RuntimeError("No valid optimizer trial")
    return best_params, {"best_value_penalized": best_value, "best": best_meta, "seeds": seed_summaries, "total_trials": len(seeds) * trials_per_seed}


def _evaluate(data: ExperimentData, indices: np.ndarray, outputs: Mapping[str, np.ndarray], *, family: int) -> tuple[dict[str, Any], np.ndarray]:
    rows = data.rows.iloc[indices].reset_index(drop=True)
    metrics, selected = evaluate_results(
        rows, outputs["exit_bars"], outputs["gross_return"], outputs["net_return"],
        outputs["reason"], outputs["mfe"], outputs["mae"], bar_minutes=1, apply_capacity=True,
    )
    idx = np.flatnonzero(selected)
    if family == FAMILY_TRAILING_ONLY or not len(idx):
        metrics.update(
            {"initial_capital_active_rate": np.nan, "capital_before_trailing_rate": np.nan,
             "ordering_violation_rate": 0.0, "handover_rate": np.nan,
             "mean_pretrail_protected_minutes": np.nan, "mean_capital_binding_minutes": np.nan}
        )
    else:
        cap = outputs["capital_first_bar"][idx]
        trail = outputs["trailing_first_bar"][idx]
        before = (cap >= 0) & ((trail < 0) | (cap < trail))
        pretrail = np.where(trail >= 0, trail - cap, outputs["exit_bars"][idx] - cap + 1)
        metrics.update(
            {
                "initial_capital_active_rate": float(np.mean(outputs["initial_capital_active"][idx])),
                "capital_before_trailing_rate": float(np.mean(before)),
                "ordering_violation_rate": float(np.mean(~outputs["order_valid"][idx])),
                "handover_rate": float(np.mean(trail >= 0)),
                "mean_pretrail_protected_minutes": float(np.mean(np.maximum(pretrail, 0))),
                "mean_capital_binding_minutes": float(np.mean(outputs["capital_binding_bars"][idx])),
            }
        )
    return metrics, selected


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, family), group in frame.groupby(["stage", "family"], sort=False):
        scores = group["objective"].to_numpy(dtype=float)
        rows.append(
            {
                "stage": stage, "family": family, "folds": len(group),
                "stable_fold_objective": float(scores.mean() - 0.5 * scores.std() + 0.25 * scores.min()),
                "mean_objective": float(scores.mean()), "worst_objective": float(scores.min()),
                "mean_pnl": float(group["net_pnl_bankroll"].mean()),
                "worst_fold_pnl": float(group["net_pnl_bankroll"].min()),
                "worst_week": float(group["worst_week"].min()),
                "worst_drawdown": float(group["max_drawdown"].min()),
                "positive_fold_fraction": float((group["net_pnl_bankroll"] > 0).mean()),
                "total_trades": int(group["n_trades"].sum()),
                "capital_before_trailing_rate": float(group["capital_before_trailing_rate"].mean()) if group["capital_before_trailing_rate"].notna().any() else np.nan,
                "initial_capital_active_rate": float(group["initial_capital_active_rate"].mean()) if group["initial_capital_active_rate"].notna().any() else np.nan,
                "ordering_violation_rate": float(group["ordering_violation_rate"].mean()),
                "mean_pretrail_protected_minutes": float(group["mean_pretrail_protected_minutes"].mean()) if group["mean_pretrail_protected_minutes"].notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["stage", "stable_fold_objective"], ascending=[True, False])


def _local_robustness(
    data: ExperimentData,
    indices: np.ndarray,
    params_by_side: Mapping[str, Mapping[str, Any]],
    family: int,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    scores = []
    keys = ("sl_mult", "trailing_activation_mult", "trailing_power", "trailing_squash_divisor", "giveback_beta", "entry_capital_ratio", "terminal_excess_ratio", "transition_center", "transition_shape")
    for _ in range(n):
        perturbed = {side: dict(values) for side, values in params_by_side.items()}
        for values in perturbed.values():
            for key in keys:
                if key in values:
                    values[key] = float(values[key]) * float(np.exp(rng.normal(0.0, 0.08)))
            values["sl_mult"] = float(np.clip(values["sl_mult"], 1.5, 4.0))
            values["entry_capital_ratio"] = float(np.clip(values.get("entry_capital_ratio", 0.75), 0.50, 0.95))
            values["terminal_excess_ratio"] = float(np.clip(values.get("terminal_excess_ratio", 0.3), 0.01, 1.0))
            if values.get("current_distance_sl_ratio", 0.0) > 0.0:
                values["current_distance_sl_ratio"] = min(
                    float(values["current_distance_sl_ratio"]), float(values["entry_capital_ratio"])
                )
        outputs = data.simulate(indices, perturbed, family)
        score, _ = _objective(data, indices, outputs)
        scores.append(score)
    values = np.asarray(scores)
    return {"n": int(n), "median_objective": float(np.median(values)), "worst_objective": float(values.min()), "positive_fraction": float(np.mean(values > 0.0))}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        vals = []
        for col in columns:
            value = row.get(col, "")
            vals.append(f"{value:.4f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--store-root", default="data_perp/exchanges/krakenfutures/execution_1m")
    parser.add_argument("--path-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atr-warmup-hours", type=int, default=48)
    parser.add_argument("--stage1-trials-per-seed", type=int, default=48)
    parser.add_argument("--joint-trials-per-seed", type=int, default=96)
    parser.add_argument("--final-trials-per-seed", type=int, default=128)
    parser.add_argument("--search-seeds", type=int, default=3)
    parser.add_argument("--local-perturbations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    spec = ConstrainedReplaySpec()
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "side", "rank_pct"]).copy()
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    deployed_by_side, _proxy_atr = _load_deployed_side_params(Path(args.deployed_parent_summary))
    atr_frac, atr_audit, atr_manifest = _causal_entry_atr(
        rows, store_root=Path(args.store_root), deployed_by_side=deployed_by_side,
        parent_summary=Path(args.deployed_parent_summary), warmup_hours=int(args.atr_warmup_hours),
    )
    atr_audit.to_parquet(output / "causal_entry_atr_audit.parquet", index=False)
    _write_json(output / "causal_entry_atr_manifest.json", atr_manifest)
    if atr_manifest["coverage"] < 0.999:
        raise RuntimeError(f"Causal ATR coverage {atr_manifest['coverage']:.2%} below 99.9%; fail closed")

    open0, high, low, close, path_valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=Path(args.store_root), cache_dir=Path(args.path_cache_dir),
        spec=spec, rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, path_valid, atr_frac, spec, deployed_by_side)
    if data.valid.mean() < 0.999:
        raise RuntimeError(f"Combined replay/ATR coverage {data.valid.mean():.2%} below 99.9%")

    warm = np.flatnonzero(data.valid)[:4].astype(np.int64)
    warm_params = {side: {**_side_defaults(deployed_by_side[side]), **deployed_by_side[side], "entry_capital_ratio": 0.75, "terminal_excess_ratio": 0.3, "transition_center": 2.0, "transition_shape": 1.2} for side in ("long", "short")}
    data.simulate(warm, warm_params, FAMILY_RATIONAL)

    seeds = [int(args.seed + 10_000 * i) for i in range(int(args.search_seeds))]
    fold_rows: list[dict[str, Any]] = []
    all_params: dict[str, Any] = {}
    advancement: dict[str, list[str]] = {}

    for fold_no, fold in enumerate(FOLDS, start=1):
        inner = INNER_FOLDS[fold["fold"]]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        full_train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        print(f"[{fold['fold']}] search={len(search_idx)} inner={len(inner_idx)} train={len(full_train_idx)} outer={len(outer_idx)}", flush=True)
        inner_scores = []
        stage1_payload: dict[int, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
        for family in CAPITAL_FAMILIES:
            name = FAMILY_NAMES[family]
            print(f"  stage1 {name}", flush=True)
            params, diag = _optimise(
                data, search_idx, family=family, joint=False,
                trials_per_seed=int(args.stage1_trials_per_seed), seeds=[s + fold_no * 100 + family for s in seeds], sampler_kind="sobol",
            )
            stage1_payload[family] = (params, diag)
            inner_outputs = data.simulate(inner_idx, params, family)
            inner_metrics, _ = _evaluate(data, inner_idx, inner_outputs, family=family)
            outer_outputs = data.simulate(outer_idx, params, family)
            outer_metrics, _ = _evaluate(data, outer_idx, outer_outputs, family=family)
            fold_rows.append({"stage": "stage1_frozen_trailing", "family": name, "fold": fold["fold"], **outer_metrics})
            inner_scores.append((float(inner_metrics["objective"]), family))
            all_params[f"{fold['fold']}__stage1__{name}"] = {"params_by_side": params, "optimizer": diag, "inner_metrics": inner_metrics}
        advanced = [family for _, family in sorted(inner_scores, reverse=True)[:2]]
        advancement[fold["fold"]] = [FAMILY_NAMES[f] for f in advanced]
        print(f"  advanced={advancement[fold['fold']]}", flush=True)

        trailing_params, trailing_diag = _optimise(
            data, full_train_idx, family=FAMILY_TRAILING_ONLY, joint=True,
            trials_per_seed=int(args.joint_trials_per_seed), seeds=[s + fold_no * 1000 for s in seeds], sampler_kind="tpe",
        )
        trailing_outputs = data.simulate(outer_idx, trailing_params, FAMILY_TRAILING_ONLY)
        trailing_metrics, _ = _evaluate(data, outer_idx, trailing_outputs, family=FAMILY_TRAILING_ONLY)
        fold_rows.append({"stage": "joint", "family": "trailing_only", "fold": fold["fold"], **trailing_metrics})
        all_params[f"{fold['fold']}__joint__trailing_only"] = {"params_by_side": trailing_params, "optimizer": trailing_diag}

        for family in advanced:
            name = FAMILY_NAMES[family]
            print(f"  joint {name}", flush=True)
            params, diag = _optimise(
                data, full_train_idx, family=family, joint=True,
                trials_per_seed=int(args.joint_trials_per_seed), seeds=[s + fold_no * 1000 + family for s in seeds], sampler_kind="tpe",
            )
            outputs = data.simulate(outer_idx, params, family)
            metrics, _ = _evaluate(data, outer_idx, outputs, family=family)
            fold_rows.append({"stage": "joint", "family": name, "fold": fold["fold"], **metrics})
            all_params[f"{fold['fold']}__joint__{name}"] = {"params_by_side": params, "optimizer": diag}

        deployed_outputs = data.simulate_deployed(outer_idx)
        deployed_metrics, _ = _evaluate(data, outer_idx, deployed_outputs, family=FAMILY_TRAILING_ONLY)
        fold_rows.append({"stage": "baseline", "family": "deployed_policy", "fold": fold["fold"], **deployed_metrics})
        pd.DataFrame(fold_rows).to_csv(output / "nested_oos_fold_metrics.partial.csv", index=False)
        _write_json(output / "nested_params.partial.json", all_params)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "nested_oos_fold_metrics.csv", index=False)
    summary = _summary(folds)
    summary.to_csv(output / "nested_oos_summary.csv", index=False)

    stage1_summary = summary[(summary["stage"] == "stage1_frozen_trailing") & (summary["folds"] == 3)].sort_values("stable_fold_objective", ascending=False)
    locked_family_names = stage1_summary.head(2)["family"].tolist()
    locked_families = [next(k for k, v in FAMILY_NAMES.items() if v == name) for name in locked_family_names]
    _write_json(output / "locked_finalists_before_july.json", {"families": locked_family_names, "source": "three-fold stage1 policy-selection OOS"})

    final_train = _indices_between(data, "2026-05-01", "2026-06-30")
    july_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    final_rows = []
    final_params: dict[str, Any] = {}
    for family in [FAMILY_TRAILING_ONLY, *locked_families]:
        name = FAMILY_NAMES[family]
        params, diag = _optimise(
            data, final_train, family=family, joint=True,
            trials_per_seed=int(args.final_trials_per_seed), seeds=[s + 900_000 + family for s in seeds], sampler_kind="tpe",
        )
        robustness = _local_robustness(
            data, final_train, params, family, n=int(args.local_perturbations), seed=int(args.seed + 950_000 + family),
        )
        outputs = data.simulate(july_idx, params, family)
        metrics, selected = _evaluate(data, july_idx, outputs, family=family)
        final_rows.append({"family": name, **metrics, **{f"local_{k}": v for k, v in robustness.items()}})
        final_params[name] = {"params_by_side": params, "optimizer": diag, "local_robustness": robustness}
        ledger = data.rows.iloc[july_idx].reset_index(drop=True).copy()
        ledger["selected"] = selected
        for key, values in outputs.items():
            ledger[key] = values
        ledger.loc[ledger["selected"]].to_parquet(output / f"july_selected_{name}.parquet", index=False)

    deployed_outputs = data.simulate_deployed(july_idx)
    deployed_metrics, _ = _evaluate(data, july_idx, deployed_outputs, family=FAMILY_TRAILING_ONLY)
    final_rows.append({"family": "deployed_policy", **deployed_metrics})
    final = pd.DataFrame(final_rows)
    final.to_csv(output / "july_post_selection_diagnostic.csv", index=False)
    _write_json(output / "final_params.json", final_params)

    manifest = {
        "generated_by": "run_simple_policy_1m_constrained_search",
        "candidate_path": str(args.candidates), "candidate_rows": len(rows),
        "candidate_period": [str(rows["timestamp"].min()), str(rows["timestamp"].max())],
        "path_manifest": path_manifest, "atr_manifest": atr_manifest,
        "folds": FOLDS, "inner_folds": INNER_FOLDS, "advancement": advancement,
        "locked_finalists": locked_family_names,
        "search": {
            "families": [FAMILY_NAMES[f] for f in CAPITAL_FAMILIES], "search_seeds": len(seeds),
            "stage1_trials_per_seed": int(args.stage1_trials_per_seed),
            "joint_trials_per_seed": int(args.joint_trials_per_seed),
            "final_trials_per_seed": int(args.final_trials_per_seed),
            "local_perturbations": int(args.local_perturbations),
            "stage1_sampler": "scrambled Sobol", "joint_sampler": "multivariate grouped TPE",
            "side_pooling": "global parameters with symmetric multiplicative side deltas bounded to approximately +/-20%; fit-only L2 penalty",
            "conditional_clamps": "categorical none/min/max/both plus conditional prior-close promotion buffer",
        },
        "formula_contract": {
            "capital": "shadow trailing gap + strictly positive excess curve",
            "entry": "entry capital gap = entry_capital_ratio * full-stop gap, ratio in [0.50,0.95]",
            "handover": "capital active immediately; trailing takes control only when armed and tighter; effective stop never loosens",
            "spline_knots_u": [0, 0.5, 1, 2, 4, 8],
        },
        "cost_contract": {"round_trip_fee": 0.01, "spread": "entry/exit half spread embedded once", "stop_gap": "15bps + 5% through capped 75bps"},
        "july_status": "post-selection diagnostic only; July was previously inspected",
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output / "manifest.json", manifest)

    report_lines = [
        "# Constrained 1-minute capital/trailing search", "",
        "## Nested OOS summary", "",
        _markdown_table(summary, ["stage", "family", "folds", "stable_fold_objective", "mean_pnl", "worst_fold_pnl", "worst_week", "worst_drawdown", "capital_before_trailing_rate", "initial_capital_active_rate", "ordering_violation_rate", "total_trades"]),
        "", "## July post-selection diagnostic", "",
        _markdown_table(final, ["family", "net_pnl_bankroll", "worst_week", "max_drawdown", "n_trades", "hit_rate", "capital_before_trailing_rate", "initial_capital_active_rate", "ordering_violation_rate", "mean_pretrail_protected_minutes", "local_median_objective", "local_worst_objective", "local_positive_fraction"]),
        "", "July is not an untouched test. Finalists were locked from the three stage-1 policy-selection-OOS folds; a later forward window is required for promotion.", "",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps(_json_safe({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"], "locked_finalists": locked_family_names}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
