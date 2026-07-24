#!/usr/bin/env python3
"""Run the OOS-only 1m simple-policy capital-protection ablation."""

from __future__ import annotations

import argparse
import json
import os
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
from extreme_price_movements.timestamp_contract import (  # noqa: E402
    assert_first_path_timestamp,
    causal_execution_times,
)
from extreme_price_movements.simple_policy_1m_ablation import (  # noqa: E402
    FAMILY_A,
    FAMILY_B,
    FAMILY_CURRENT,
    FAMILY_EXPONENTIAL,
    FAMILY_SIGMOID,
    MOD_MAX_GIVEBACK,
    MOD_MIN_CURRENT_GAP,
    MOD_MIN_MFE_GAP,
    ReplaySpec,
    evaluate_results,
    objective_score_fast,
    params_to_vector,
    simulate_1m_paths,
)


FAMILY_NAMES = {
    FAMILY_CURRENT: "current",
    FAMILY_A: "a_constant_atr_from_mfe",
    FAMILY_B: "b_multilayer_envelope",
    FAMILY_SIGMOID: "c_sigmoid",
    FAMILY_EXPONENTIAL: "d_exponential",
}

FOLDS = [
    {
        "fold": "fold_1",
        "train_start": "2026-05-01",
        "train_end": "2026-05-14",
        "purge": "2026-05-14",
        "validation_start": "2026-05-15",
        "validation_end": "2026-06-01",
    },
    {
        "fold": "fold_2",
        "train_start": "2026-05-01",
        "train_end": "2026-05-31",
        "purge": "2026-05-31",
        "validation_start": "2026-06-01",
        "validation_end": "2026-06-15",
    },
    {
        "fold": "fold_3",
        "train_start": "2026-05-01",
        "train_end": "2026-06-14",
        "purge": "2026-06-14",
        "validation_start": "2026-06-15",
        "validation_end": "2026-07-01",
    },
]
CAUSAL_PATH_CONTRACT_VERSION = "signal_close_plus_delayed_entry_v1"


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
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")


def _arm_name(family: int, modifiers: int) -> str:
    if family == FAMILY_CURRENT:
        return "current_policy_reoptimised_1m"
    flags = []
    if modifiers & MOD_MAX_GIVEBACK:
        flags.append("maxgiveback")
    if modifiers & MOD_MIN_MFE_GAP:
        flags.append("minmfe")
    if modifiers & MOD_MIN_CURRENT_GAP:
        flags.append("mincurrent")
    return f"{FAMILY_NAMES[family]}__{'_'.join(flags) if flags else 'plain'}"


def _all_arms() -> list[tuple[int, int, str]]:
    arms = [(FAMILY_CURRENT, 0, _arm_name(FAMILY_CURRENT, 0))]
    for family in (FAMILY_A, FAMILY_B, FAMILY_SIGMOID, FAMILY_EXPONENTIAL):
        for modifiers in range(8):
            arms.append((family, modifiers, _arm_name(family, modifiers)))
    return arms


def _load_deployed_side_params(summary_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    frame = pd.read_csv(summary_path)
    params_by_side: dict[str, dict[str, Any]] = {}
    atr_frac_by_side: dict[str, float] = {}
    for _, row in frame.iterrows():
        side = str(row["side"])
        def val(name: str, default: float) -> float:
            value = row.get(name, default)
            try:
                value = float(value)
            except Exception:
                return float(default)
            return value if np.isfinite(value) else float(default)

        params_by_side[side] = {
            "sl_mult": val("param_sl_mult", 2.5),
            "trailing_activation_mult": val("param_trailing_activation_mult", 1.5),
            "trailing_activation_cap_pct": val("param_trailing_activation_cap_pct", 0.0),
            "trailing_power": val("param_trailing_power", 1.5),
            "trailing_squash_divisor": val("param_trailing_squash_divisor", 2.0),
            "giveback_beta": val("param_giveback_beta", 0.5),
            "p1": val("param_capital_protect_mfe_mult", 0.0),
            "capital_protect_regression_frac": val("param_capital_protect_regression_frac", 0.45),
            "capital_protect_lock_frac": val("param_capital_protect_lock_frac", np.nan),
            "capital_protect_min_lock_bps": val("param_capital_protect_min_lock_bps", 0.0),
            "capital_protect_spread_lock_mult": val("param_capital_protect_spread_lock_mult", 1.5),
            "trailing_activation_decay_half_life_minutes": 15.0 * val("param_trailing_activation_decay_half_life_bars", 0.0),
            "trailing_activation_decay_start_minutes": 15.0 * val("param_trailing_activation_decay_start_bars", 0.0),
            "trailing_activation_min_mult": val("param_trailing_activation_min_mult", 1.0),
            "adverse_exit_enabled": bool(row.get("param_adverse_exit_enabled", False)),
            "adverse_exit_min_mae_atr": val("param_adverse_exit_min_mae_atr", 1.0),
            "adverse_exit_min_speed_per_15m": val("param_adverse_exit_min_speed", 0.3),
            "adverse_exit_theta": val("param_adverse_exit_theta", 1e9),
            "adverse_exit_fast_minutes": 15.0 * val("param_adverse_exit_fast_bars", 0.0),
            "adverse_exit_max_mfe_atr": val("param_adverse_exit_max_mfe_atr", 0.25),
        }
        median = val("param_policy_median_barrier_frac", 0.01)
        power = np.clip(val("param_atr_power", 1.0), 0.5, 1.2)
        multiplier = np.clip(val("param_atr_multiplier", 1.0), 0.5, 2.0)
        raw = 0.03
        atr_frac_by_side[side] = float(multiplier * median * (raw / max(median, 1e-6)) ** power)
    return params_by_side, atr_frac_by_side


def _path_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "open": cache_dir / "open0.f32",
        "high": cache_dir / "high.f32",
        "low": cache_dir / "low.f32",
        "close": cache_dir / "close.f32",
        "valid": cache_dir / "valid.npy",
        "manifest": cache_dir / "manifest.json",
    }


def _load_or_build_path_cache(
    rows: pd.DataFrame,
    *,
    store_root: Path,
    cache_dir: Path,
    spec: ReplaySpec,
    rebuild: bool,
    signal_timeframe: str = "1h",
    entry_delay_minutes: int = 5,
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap, np.ndarray, dict[str, Any]]:
    paths = _path_cache_paths(cache_dir)
    shape = (len(rows), spec.path_len)
    if not rebuild and all(paths[key].exists() for key in ("open", "high", "low", "close", "valid", "manifest")):
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if (
            manifest.get("shape") == [len(rows), spec.path_len]
            and manifest.get("path_contract_version") == CAUSAL_PATH_CONTRACT_VERSION
            and manifest.get("signal_timeframe") == str(signal_timeframe)
            and int(manifest.get("entry_delay_minutes", -1)) == int(entry_delay_minutes)
        ):
            return (
                np.memmap(paths["open"], mode="r", dtype="float32", shape=(len(rows),)),
                np.memmap(paths["high"], mode="r", dtype="float32", shape=shape),
                np.memmap(paths["low"], mode="r", dtype="float32", shape=shape),
                np.memmap(paths["close"], mode="r", dtype="float32", shape=shape),
                np.load(paths["valid"]),
                manifest,
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    open0 = np.memmap(paths["open"], mode="w+", dtype="float32", shape=(len(rows),))
    high = np.memmap(paths["high"], mode="w+", dtype="float32", shape=shape)
    low = np.memmap(paths["low"], mode="w+", dtype="float32", shape=shape)
    close = np.memmap(paths["close"], mode="w+", dtype="float32", shape=shape)
    open0[:] = np.nan
    high[:] = np.nan
    low[:] = np.nan
    close[:] = np.nan
    store = PartitionedOHLCVStore(str(store_root), timeframe="1m")
    total_symbols = rows["symbol"].nunique()
    for number, (symbol, group) in enumerate(rows.groupby("symbol", sort=True), start=1):
        signal_ts, decision_ts, entry_ts = causal_execution_times(
            group,
            timeframe=signal_timeframe,
            delay_minutes=int(entry_delay_minutes),
        )
        timestamps = pd.Series(entry_ts, index=group.index)
        start = timestamps.min()
        end = timestamps.max() + pd.Timedelta(minutes=spec.horizon_minutes)
        bars = store.load(
            str(symbol),
            columns=["ts", "open", "high", "low", "close"],
            start_ts=start,
            end_ts=end,
        )
        if bars is None or bars.empty or not isinstance(bars.index, pd.DatetimeIndex):
            print(f"[path-cache {number}/{total_symbols}] {symbol} empty", flush=True)
            continue
        bars = bars[~bars.index.duplicated(keep="last")].sort_index()
        idx = bars.index.tz_localize("UTC") if bars.index.tz is None else bars.index.tz_convert("UTC")
        open_values = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float32)
        high_values = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=np.float32)
        low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=np.float32)
        close_values = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float32)
        filled = 0
        for row_i, timestamp, row_signal_ts in zip(
            group.index.to_numpy(dtype=np.int64),
            timestamps,
            signal_ts,
        ):
            position = int(idx.searchsorted(timestamp))
            if position + spec.path_len > len(idx):
                continue
            expected = pd.date_range(timestamp, periods=spec.path_len, freq="1min", tz="UTC")
            actual = idx[position : position + spec.path_len]
            if not actual.equals(expected):
                continue
            assert_first_path_timestamp(
                first_path_ts=pd.DatetimeIndex([actual[0]]),
                signal_ts=pd.DatetimeIndex([row_signal_ts]),
                timeframe=signal_timeframe,
            )
            open0[row_i] = open_values[position]
            high[row_i, :] = high_values[position : position + spec.path_len]
            low[row_i, :] = low_values[position : position + spec.path_len]
            close[row_i, :] = close_values[position : position + spec.path_len]
            filled += 1
        print(f"[path-cache {number}/{total_symbols}] {symbol} complete_rows={filled}/{len(group)}", flush=True)
    for array in (open0, high, low, close):
        array.flush()
    valid = np.isfinite(open0) & np.isfinite(high).all(axis=1) & np.isfinite(low).all(axis=1) & np.isfinite(close).all(axis=1)
    np.save(paths["valid"], valid)
    manifest = {
        "shape": [len(rows), spec.path_len],
        "path_contract_version": CAUSAL_PATH_CONTRACT_VERSION,
        "timeframe": spec.timeframe,
        "bar_minutes": spec.bar_minutes,
        "horizon_minutes": spec.horizon_minutes,
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "coverage": float(valid.mean()),
        "path_semantics": "signal-close plus delayed-entry minutes, then 1m OHLC; pessimistic same-minute stop collision",
        "signal_timeframe": str(signal_timeframe),
        "entry_delay_minutes": int(entry_delay_minutes),
        "store_root": str(store_root),
    }
    _write_json(paths["manifest"], manifest)
    return open0, high, low, close, valid, manifest


def _side_defaults(deployed: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "trailing_activation_decay_half_life_minutes",
        "trailing_activation_decay_start_minutes",
        "trailing_activation_min_mult",
        "adverse_exit_enabled",
        "adverse_exit_min_mae_atr",
        "adverse_exit_min_speed_per_15m",
        "adverse_exit_theta",
        "adverse_exit_fast_minutes",
        "adverse_exit_max_mfe_atr",
        "trailing_activation_cap_pct",
        "capital_protect_spread_lock_mult",
    )
    return {key: deployed.get(key) for key in keys}


def _suggest_params(
    trial: optuna.Trial,
    *,
    family: int,
    modifiers: int,
    deployed: Mapping[str, Any],
) -> dict[str, Any]:
    params: dict[str, Any] = _side_defaults(deployed)
    params.update(
        {
            "sl_mult": trial.suggest_float("sl_mult", 1.5, 4.0, step=0.1),
            "trailing_activation_mult": trial.suggest_float("trailing_activation_mult", 0.5, 3.0, step=0.1),
            "trailing_power": trial.suggest_float("trailing_power", 1.1, 2.5, step=0.1),
            "trailing_squash_divisor": trial.suggest_float("trailing_squash_divisor", 1.0, 5.0, step=0.25),
            "giveback_beta": trial.suggest_float("giveback_beta", 0.15, 0.95, step=0.05),
        }
    )
    if family == FAMILY_CURRENT:
        params.update(
            {
                "p1": trial.suggest_float("capital_protect_mfe_mult", 0.0, 3.0, step=0.1),
                "capital_protect_regression_frac": trial.suggest_float("capital_protect_regression_frac", 0.0, 1.0, step=0.05),
                "capital_protect_lock_frac": trial.suggest_float("capital_protect_lock_frac", 0.0, 0.7, step=0.05),
                "capital_protect_min_lock_bps": trial.suggest_float("capital_protect_min_lock_bps", 0.0, 75.0, step=5.0),
            }
        )
    elif family == FAMILY_A:
        params["p1"] = trial.suggest_float("x", 0.5, 6.0, step=0.1)
    elif family == FAMILY_B:
        params["p1"] = trial.suggest_float("x", 0.5, 6.0, step=0.1)
        params["p2"] = trial.suggest_float("y", 0.5, 6.0, step=0.1)
        params["p3"] = trial.suggest_float("z", 0.5, 6.0, step=0.1)
    elif family == FAMILY_SIGMOID:
        low = trial.suggest_float("low", 0.4, 1.5, step=0.05)
        params.update(
            {
                "p1": trial.suggest_float("high", max(1.5, low + 0.1), 4.0),
                "p2": low,
                "p3": trial.suggest_float("center", 1.0, 4.0, step=0.1),
                "p4": trial.suggest_float("k", 0.4, 2.5, step=0.1),
            }
        )
    else:
        params.update(
            {
                "p1": trial.suggest_float("floor", 0.4, 1.5, step=0.05),
                "p2": trial.suggest_float("amplitude", 0.8, 3.0, step=0.1),
                "p3": trial.suggest_float("decay", 0.1, 1.0, step=0.05),
            }
        )
    if modifiers & MOD_MIN_MFE_GAP:
        params["min_mfe_gap_atr"] = trial.suggest_float("min_mfe_gap_atr", 0.1, 1.5, step=0.1)
    if modifiers & MOD_MIN_CURRENT_GAP:
        params["min_current_gap_atr"] = trial.suggest_float("min_current_gap_atr", 0.1, 1.5, step=0.1)
    if modifiers & MOD_MAX_GIVEBACK:
        lower = max(1.0, float(params.get("min_mfe_gap_atr", 0.0)))
        params["max_giveback_atr"] = trial.suggest_float("max_giveback_atr", lower, 6.0)
    return params


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
        spec: ReplaySpec,
    ) -> None:
        self.rows = rows
        self.open0 = open0
        self.high = high
        self.low = low
        self.close = close
        self.valid = valid
        self.atr_frac = atr_frac.astype(np.float64)
        self.spec = spec
        self.side = pd.to_numeric(rows["side"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        self.entry_spread = pd.to_numeric(rows["spread_cost_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.exit_spread = pd.to_numeric(rows["exit_spread_cost_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        self.timestamps = pd.to_datetime(rows["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
        self.symbol_codes = pd.Categorical(rows["symbol"].astype(str)).codes.astype(np.int32)
        self.rank = pd.to_numeric(rows["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)

    def simulate(self, indices: np.ndarray, params: Mapping[str, Any], family: int, modifiers: int):
        return simulate_1m_paths(
            indices.astype(np.int64),
            self.open0,
            self.high,
            self.low,
            self.close,
            self.side,
            self.atr_frac,
            self.entry_spread,
            self.exit_spread,
            params_to_vector(params),
            int(family),
            int(modifiers),
            float(self.spec.fee_per_side),
            float(self.spec.stop_base_gap_bps),
            float(self.spec.stop_through_fraction),
            float(self.spec.stop_max_gap_bps),
        )


def _indices_between(data: ExperimentData, start: str, end: str, side: str | None = None) -> np.ndarray:
    timestamps = pd.to_datetime(data.rows["timestamp"], utc=True)
    mask = timestamps.ge(pd.Timestamp(start, tz="UTC")) & timestamps.lt(pd.Timestamp(end, tz="UTC")) & data.valid
    if side == "long":
        mask &= data.side > 0.0
    elif side == "short":
        mask &= data.side < 0.0
    return np.flatnonzero(mask.to_numpy() if hasattr(mask, "to_numpy") else mask).astype(np.int64)


def _fast_objective(data: ExperimentData, indices: np.ndarray, result: tuple[np.ndarray, ...]) -> tuple[float, dict[str, Any]]:
    exit_bars, _exit_px, _gross, net, _reason, _mfe, _mae = result
    week_ts = pd.to_datetime(data.rows.iloc[indices]["timestamp"], utc=True).dt.tz_localize(None)
    weeks = pd.factorize(week_ts.dt.to_period("W").astype(str), sort=True)[0].astype(np.int32)
    score, total, worst, max_dd, n = objective_score_fast(
        data.timestamps[indices],
        data.symbol_codes[indices],
        data.rank[indices],
        weeks,
        exit_bars,
        net,
        data.spec.bar_minutes,
    )
    return float(score), {
        "objective": float(score),
        "net_pnl_bankroll": float(total),
        "worst_week": float(worst),
        "max_drawdown": float(max_dd),
        "n_trades": int(n),
    }


def _optimise_side(
    data: ExperimentData,
    indices: np.ndarray,
    *,
    family: int,
    modifiers: int,
    deployed: Mapping[str, Any],
    n_trials: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(indices) == 0:
        raise ValueError("No rows available for side optimization")
    sampler = optuna.samplers.TPESampler(seed=int(seed), n_startup_trials=min(8, max(4, int(n_trials) // 3)))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, family=family, modifiers=modifiers, deployed=deployed)
        result = data.simulate(indices, params, family, modifiers)
        score, diag = _fast_objective(data, indices, result)
        for key, value in diag.items():
            trial.set_user_attr(key, value)
        return score

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False, gc_after_trial=False)
    best = _suggested_to_runtime_params(study.best_trial.params, family, modifiers, deployed)
    return best, {
        "best_value": float(study.best_value),
        "best_trial": int(study.best_trial.number),
        "best_user_attrs": dict(study.best_trial.user_attrs),
        "trials": int(len(study.trials)),
    }


def _suggested_to_runtime_params(
    chosen: Mapping[str, Any],
    family: int,
    modifiers: int,
    deployed: Mapping[str, Any],
) -> dict[str, Any]:
    params = _side_defaults(deployed)
    params.update(
        {
            "sl_mult": chosen["sl_mult"],
            "trailing_activation_mult": chosen["trailing_activation_mult"],
            "trailing_power": chosen["trailing_power"],
            "trailing_squash_divisor": chosen["trailing_squash_divisor"],
            "giveback_beta": chosen["giveback_beta"],
        }
    )
    if family == FAMILY_CURRENT:
        params.update(
            {
                "p1": chosen["capital_protect_mfe_mult"],
                "capital_protect_regression_frac": chosen["capital_protect_regression_frac"],
                "capital_protect_lock_frac": chosen["capital_protect_lock_frac"],
                "capital_protect_min_lock_bps": chosen["capital_protect_min_lock_bps"],
            }
        )
    elif family == FAMILY_A:
        params["p1"] = chosen["x"]
    elif family == FAMILY_B:
        params.update({"p1": chosen["x"], "p2": chosen["y"], "p3": chosen["z"]})
    elif family == FAMILY_SIGMOID:
        params.update({"p1": chosen["high"], "p2": chosen["low"], "p3": chosen["center"], "p4": chosen["k"]})
    else:
        params.update({"p1": chosen["floor"], "p2": chosen["amplitude"], "p3": chosen["decay"]})
    if modifiers & MOD_MIN_MFE_GAP:
        params["min_mfe_gap_atr"] = chosen["min_mfe_gap_atr"]
    if modifiers & MOD_MIN_CURRENT_GAP:
        params["min_current_gap_atr"] = chosen["min_current_gap_atr"]
    if modifiers & MOD_MAX_GIVEBACK:
        params["max_giveback_atr"] = chosen["max_giveback_atr"]
    return params


def _evaluate_combined(
    data: ExperimentData,
    indices: np.ndarray,
    *,
    params_by_side: Mapping[str, Mapping[str, Any]],
    family: int,
    modifiers: int,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    ordered = np.asarray(indices, dtype=np.int64)
    outputs = {
        "exit_bars": np.full(len(ordered), -1, dtype=np.int32),
        "exit_price": np.full(len(ordered), np.nan),
        "gross_return": np.full(len(ordered), np.nan),
        "net_return": np.full(len(ordered), np.nan),
        "reason": np.zeros(len(ordered), dtype=np.int8),
        "mfe": np.full(len(ordered), np.nan),
        "mae": np.full(len(ordered), np.nan),
    }
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        local_pos = np.flatnonzero(data.side[ordered] * sign > 0.0)
        if not len(local_pos):
            continue
        result = data.simulate(ordered[local_pos], params_by_side[side_name], family, modifiers)
        for key, values in zip(outputs, result):
            outputs[key][local_pos] = values
    metrics, selected = evaluate_results(
        data.rows.iloc[ordered].reset_index(drop=True),
        outputs["exit_bars"],
        outputs["gross_return"],
        outputs["net_return"],
        outputs["reason"],
        outputs["mfe"],
        outputs["mae"],
        bar_minutes=data.spec.bar_minutes,
        apply_capacity=True,
    )
    return metrics, selected, outputs


def _breakdowns(
    rows: pd.DataFrame,
    selected: np.ndarray,
    outputs: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    work = rows.copy().reset_index(drop=True)
    work["selected"] = selected
    work["net_return"] = outputs["net_return"]
    work["gross_return"] = outputs["gross_return"]
    work["exit_bars"] = outputs["exit_bars"]
    work["reason"] = outputs["reason"]
    work = work.loc[work["selected"]].copy()
    if work.empty:
        return pd.DataFrame()
    work["position_size"] = 0.075 + 0.075 * np.power(pd.to_numeric(work["rank_pct"], errors="coerce").clip(0, 1), 1.1)
    work["pnl"] = work["net_return"] * work["position_size"]
    ts = pd.to_datetime(work["timestamp"], utc=True)
    work["week"] = ts.dt.tz_localize(None).dt.to_period("W").astype(str)
    work["month"] = ts.dt.strftime("%Y-%m")
    rows_out: list[dict[str, Any]] = []
    groups = [
        ("overall", []),
        ("side", ["side_name"]),
        ("archetype", ["policy_archetype"]),
        ("side_x_archetype", ["side_name", "policy_archetype"]),
        ("week", ["week"]),
        ("month", ["month"]),
    ]
    for slice_name, columns in groups:
        iterator = [((), work)] if not columns else work.groupby(columns, dropna=False)
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"slice": slice_name, **{column: value for column, value in zip(columns, keys)}}
            row.update(
                {
                    "n_trades": int(len(group)),
                    "net_pnl_bankroll": float(group["pnl"].sum()),
                    "mean_net_return": float(group["net_return"].mean()),
                    "hit_rate": float((group["net_return"] > 0).mean()),
                    "mean_holding_hours": float((group["exit_bars"] + 1).mean() / 60.0),
                    "full_sl_rate": float((group["reason"] == 1).mean()),
                    "capital_rate": float((group["reason"] == 2).mean()),
                    "trailing_rate": float((group["reason"] == 3).mean()),
                    "timeout_rate": float((group["reason"] == 0).mean()),
                }
            )
            rows_out.append(row)
    return pd.DataFrame(rows_out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--deployed-parent-summary", required=True)
    parser.add_argument("--store-root", default="data_perp/exchanges/krakenfutures/execution_1m")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials-per-fold", type=int, default=24)
    parser.add_argument("--final-trials", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--rebuild-path-cache", action="store_true")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = ReplaySpec()
    rows = pd.read_parquet(args.candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    rows = rows.dropna(subset=["timestamp", "symbol", "side", "rank_pct"]).copy()
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    deployed_by_side, atr_by_side = _load_deployed_side_params(Path(args.deployed_parent_summary))
    atr_frac = np.where(pd.to_numeric(rows["side"], errors="coerce").to_numpy() > 0, atr_by_side["long"], atr_by_side["short"])

    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=Path(args.store_root),
        cache_dir=out_dir / "path_cache",
        spec=spec,
        rebuild=bool(args.rebuild_path_cache),
    )
    if float(valid.mean()) < 0.95:
        raise RuntimeError(f"1m path coverage {valid.mean():.2%} is below required 95%; refusing ablation")
    data = ExperimentData(rows, open0, high, low, close, valid, atr_frac, spec)
    # Warm both Numba kernels before timing the search.
    warm_idx = np.flatnonzero(valid)[: min(8, int(valid.sum()))].astype(np.int64)
    if len(warm_idx):
        warm = data.simulate(warm_idx, deployed_by_side["long"], FAMILY_CURRENT, 0)
        _fast_objective(data, warm_idx, warm)

    started = time.monotonic()
    nested_rows: list[dict[str, Any]] = []
    nested_params: dict[str, Any] = {}
    for arm_no, (family, modifiers, arm_name) in enumerate(_all_arms(), start=1):
        print(f"[nested {arm_no:02d}/33] {arm_name}", flush=True)
        nested_params[arm_name] = {}
        for fold_no, fold in enumerate(FOLDS, start=1):
            params_by_side: dict[str, dict[str, Any]] = {}
            fit_diag: dict[str, Any] = {}
            for side_no, side_name in enumerate(("long", "short")):
                train_idx = _indices_between(data, fold["train_start"], fold["train_end"], side_name)
                params, diag = _optimise_side(
                    data,
                    train_idx,
                    family=family,
                    modifiers=modifiers,
                    deployed=deployed_by_side[side_name],
                    n_trials=int(args.trials_per_fold),
                    seed=int(args.seed + arm_no * 1000 + fold_no * 10 + side_no),
                )
                params_by_side[side_name] = params
                fit_diag[side_name] = diag
            validation_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
            metrics, _selected, _outputs = _evaluate_combined(
                data,
                validation_idx,
                params_by_side=params_by_side,
                family=family,
                modifiers=modifiers,
            )
            nested_params[arm_name][fold["fold"]] = {
                "params_by_side": params_by_side,
                "fit_diagnostics": fit_diag,
                "validation_metrics": metrics,
            }
            nested_rows.append(
                {
                    "arm": arm_name,
                    "family": FAMILY_NAMES[family],
                    "modifiers": modifiers,
                    "fold": fold["fold"],
                    **metrics,
                }
            )
            print(
                f"  {fold['fold']} pnl={metrics['net_pnl_bankroll']:.5f} "
                f"worst_week={metrics['worst_week']:.5f} dd={metrics['max_drawdown']:.5f} "
                f"trades={metrics['n_trades']}",
                flush=True,
            )
        pd.DataFrame(nested_rows).to_csv(out_dir / "nested_oos_fold_metrics.partial.csv", index=False)
        _write_json(out_dir / "nested_params.partial.json", nested_params)

    # Fixed deployed 1m baseline, scored on exactly the same OOS validation rows.
    for fold in FOLDS:
        validation_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        metrics, _selected, _outputs = _evaluate_combined(
            data,
            validation_idx,
            params_by_side=deployed_by_side,
            family=FAMILY_CURRENT,
            modifiers=0,
        )
        nested_rows.append(
            {
                "arm": "deployed_policy_replayed_1m",
                "family": "deployed_baseline",
                "modifiers": 0,
                "fold": fold["fold"],
                **metrics,
            }
        )

    nested = pd.DataFrame(nested_rows)
    nested.to_csv(out_dir / "nested_oos_fold_metrics.csv", index=False)
    summary_rows = []
    for arm_name, group in nested.groupby("arm", sort=False):
        scores = group["objective"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "arm": arm_name,
                "family": group["family"].iloc[0],
                "modifiers": int(group["modifiers"].iloc[0]),
                "folds": int(len(group)),
                "mean_objective": float(np.mean(scores)),
                "std_objective": float(np.std(scores)),
                "worst_objective": float(np.min(scores)),
                "stable_fold_objective": float(np.mean(scores) - 0.5 * np.std(scores) + 0.25 * np.min(scores)),
                "positive_fold_fraction": float(np.mean(group["net_pnl_bankroll"] > 0.0)),
                "mean_pnl": float(group["net_pnl_bankroll"].mean()),
                "worst_fold_pnl": float(group["net_pnl_bankroll"].min()),
                "worst_week_across_folds": float(group["worst_week"].min()),
                "worst_drawdown": float(group["max_drawdown"].min()),
                "total_oos_trades": int(group["n_trades"].sum()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("stable_fold_objective", ascending=False)
    summary.to_csv(out_dir / "nested_oos_ablation_summary.csv", index=False)

    # Pre-register one winner per capital family from May/June nested OOS folds.
    family_winners: dict[str, str] = {"current": "current_policy_reoptimised_1m"}
    for family_name in (FAMILY_NAMES[FAMILY_A], FAMILY_NAMES[FAMILY_B], FAMILY_NAMES[FAMILY_SIGMOID], FAMILY_NAMES[FAMILY_EXPONENTIAL]):
        eligible = summary.loc[summary["family"] == family_name]
        family_winners[family_name] = str(eligible.iloc[0]["arm"])
    _write_json(out_dir / "family_winners_locked_before_july.json", family_winners)

    arm_lookup = {name: (family, modifiers) for family, modifiers, name in _all_arms()}
    final_train = _indices_between(data, "2026-05-01", "2026-06-30")
    july_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    final_metrics_rows: list[dict[str, Any]] = []
    final_params: dict[str, Any] = {}
    final_breakdowns: list[pd.DataFrame] = []
    final_trade_ledgers: list[pd.DataFrame] = []

    july_arms = [("deployed_policy_replayed_1m", FAMILY_CURRENT, 0, deployed_by_side)]
    for winner in family_winners.values():
        family, modifiers = arm_lookup[winner]
        params_by_side: dict[str, dict[str, Any]] = {}
        diag_by_side: dict[str, Any] = {}
        for side_no, side_name in enumerate(("long", "short")):
            side_train = final_train[(data.side[final_train] > 0) if side_name == "long" else (data.side[final_train] < 0)]
            params, diag = _optimise_side(
                data,
                side_train,
                family=family,
                modifiers=modifiers,
                deployed=deployed_by_side[side_name],
                n_trials=int(args.final_trials),
                seed=int(args.seed + 900_000 + len(final_params) * 10 + side_no),
            )
            params_by_side[side_name] = params
            diag_by_side[side_name] = diag
        final_params[winner] = {"family": family, "modifiers": modifiers, "params_by_side": params_by_side, "fit_diagnostics": diag_by_side}
        july_arms.append((winner, family, modifiers, params_by_side))

    # July is one-shot frozen confirmation for pre-locked family winners only.
    for arm_name, family, modifiers, params_by_side in july_arms:
        metrics, selected, outputs = _evaluate_combined(
            data,
            july_idx,
            params_by_side=params_by_side,
            family=family,
            modifiers=modifiers,
        )
        final_metrics_rows.append({"arm": arm_name, "family": FAMILY_NAMES.get(family, "current"), "modifiers": modifiers, **metrics})
        selected_rows = data.rows.iloc[july_idx].reset_index(drop=True)
        breakdown = _breakdowns(selected_rows, selected, outputs)
        if not breakdown.empty:
            breakdown.insert(0, "arm", arm_name)
            final_breakdowns.append(breakdown)
        ledger = selected_rows.copy()
        ledger["arm"] = arm_name
        ledger["selected"] = selected
        for key, values in outputs.items():
            ledger[key] = values
        final_trade_ledgers.append(ledger.loc[ledger["selected"]].copy())

    pd.DataFrame(final_metrics_rows).to_csv(out_dir / "july_frozen_family_winner_metrics.csv", index=False)
    if final_breakdowns:
        pd.concat(final_breakdowns, ignore_index=True).to_csv(out_dir / "july_frozen_detailed_breakdowns.csv", index=False)
    if final_trade_ledgers:
        pd.concat(final_trade_ledgers, ignore_index=True).to_parquet(out_dir / "july_frozen_selected_trade_ledger.parquet", index=False)
    _write_json(out_dir / "nested_params.json", nested_params)
    _write_json(out_dir / "final_refit_params.json", final_params)

    manifest = {
        "generated_by": "run_simple_policy_1m_capital_ablation",
        "candidate_path": str(args.candidates),
        "candidate_rows": int(len(rows)),
        "candidate_period": [str(rows["timestamp"].min()), str(rows["timestamp"].max())],
        "candidate_contract": "model/admission OOS; geometry parameters selected only on earlier policy-train rows",
        "replay_spec": spec.__dict__,
        "path_cache": path_manifest,
        "cost_contract": {
            "fee_per_side": spec.fee_per_side,
            "round_trip_fee": 2.0 * spec.fee_per_side,
            "spread": "entry and exit half-spread embedded once in executable prices",
            "stop_gap": "15bps base + 5% through, capped 75bps",
        },
        "atr_contract": {
            "definition": "entry-frozen deployable barrier/ATR proxy",
            "raw_barrier_pct": 0.03,
            "side_values": atr_by_side,
        },
        "formula_contract": {
            "u": "ratcheted favorable price excursion / entry-frozen ATR price distance",
            "a_gap": "x * ATR",
            "b_gap": "max(x*ATR, y*ATR*u**0.3, z*ATR*u**0.6)",
            "c_gap": "ATR * (low + (high-low)/(1+exp(k*(u-center))))",
            "d_gap": "ATR * (floor + amplitude*exp(-decay*u))",
            "capital_vs_trailing": "capital gap is floored at armed trailing gap, so capital preservation is looser",
            "minimum_current_distance": "uses prior completed 1m close and a monotone stop ratchet",
        },
        "folds": FOLDS,
        "fold_objective": "weekly_mean - 0.5*weekly_std + 0.25*worst_week - 0.10*abs(max_drawdown)",
        "family_selection_objective": "mean_fold - 0.5*std_fold + 0.25*worst_fold",
        "search_breadth": {
            "ablation_arms": 33,
            "sides": 2,
            "nested_folds": 3,
            "trials_per_fold": int(args.trials_per_fold),
            "locked_family_winners": len(family_winners),
            "final_trials_per_side": int(args.final_trials),
            "planned_nested_trials": 33 * 2 * 3 * int(args.trials_per_fold),
            "planned_final_trials": len(family_winners) * 2 * int(args.final_trials),
        },
        "july_status": "one-shot frozen replay for family winners locked from May/June nested OOS; not used to choose an arm",
        "family_winners": family_winners,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "nested_oos_fold_metrics": "nested_oos_fold_metrics.csv",
            "nested_oos_ablation_summary": "nested_oos_ablation_summary.csv",
            "july_frozen_family_winner_metrics": "july_frozen_family_winner_metrics.csv",
            "july_frozen_detailed_breakdowns": "july_frozen_detailed_breakdowns.csv",
            "july_frozen_selected_trade_ledger": "july_frozen_selected_trade_ledger.parquet",
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(_json_safe({"status": "complete", "elapsed_seconds": manifest["elapsed_seconds"], "family_winners": family_winners}), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
