#!/usr/bin/env python3
"""Re-optimize joint trailing geometry for 1.5x spread and zero fees.

Only strictly positive-volume entry candles are eligible. Missing forward
spreads are filled from a frozen May--June per-symbol mean (global May--June
mean only for unseen symbols). July is evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_1m_wallet_portfolio import (  # noqa: E402
    replay_marked_notional_wallet,
)
from scripts.report_simple_policy_1m_winner_daily_july import (  # noqa: E402
    OLD_ATR,
    OLD_CACHE,
    OLD_CANDIDATES,
    PARAMS,
    PARENT,
    POSTERIOR,
    RICH,
    STORE,
    _prediction_candidates,
)
from scripts.report_simple_policy_1m_winner_daily_nonzero_volume import (  # noqa: E402
    DAILY_DIR,
    FORWARD_DIR,
    VOLUME_CACHE,
    _entry_minute_volume,
    _period_metrics,
    _select_after_filter,
)
from scripts.report_simple_policy_1m_winner_forward_july import (  # noqa: E402
    CHAMPION,
    _forward_context,
)
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    ExperimentData,
    _evaluate,
    _indices_between,
    _local_robustness,
    _objective,
    _side_defaults,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bayesian_sizes,
    _load_atr,
    _load_context,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _combine_outputs(parts: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _normalise_monitor_symbol(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("/", "_", regex=False)


def _monitor_asset_spread_fill(
    rows: pd.DataFrame,
    *,
    monitor_root: Path,
    quantile: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fill missing spread from strictly pre-entry monitor observations."""
    out = rows.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    observed = pd.to_numeric(out["expected_spread_bps"], errors="coerce")
    missing = ~np.isfinite(observed)
    files = sorted(monitor_root.glob("kraken_futures_perp_spreads_*.parquet"))
    if not files:
        raise RuntimeError(f"No spread monitor snapshots found under {monitor_root}")
    parts: list[pd.DataFrame] = []
    for path in files:
        part = pd.read_parquet(path, columns=["observed_ts", "symbol", "spread_bps"])
        part["observed_ts"] = pd.to_datetime(part["observed_ts"], utc=True, errors="coerce")
        part["symbol"] = _normalise_monitor_symbol(part["symbol"])
        part["spread_bps"] = pd.to_numeric(part["spread_bps"], errors="coerce")
        part = part.loc[
            part["observed_ts"].notna()
            & np.isfinite(part["spread_bps"])
            & part["spread_bps"].ge(0.0)
        ]
        parts.append(part)
    monitor = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["observed_ts", "symbol"], keep="last"
    )
    fills: list[float] = []
    supports: list[int] = []
    sources: list[str] = []
    for row_i in np.flatnonzero(missing.to_numpy()):
        timestamp = ts.iloc[row_i]
        symbol = str(out.iloc[row_i]["symbol"]).replace("/", "_")
        prior = monitor.loc[monitor["observed_ts"].lt(timestamp)]
        asset_values = prior.loc[prior["symbol"].eq(symbol), "spread_bps"].to_numpy(float)
        if len(asset_values):
            values = asset_values
            source = "asset_monitor_quantile"
        else:
            values = prior["spread_bps"].to_numpy(float)
            source = "global_monitor_quantile"
        if not len(values):
            raise RuntimeError(f"No strictly pre-entry monitor spread for {symbol} at {timestamp}")
        fills.append(float(np.quantile(values, quantile)))
        supports.append(int(len(values)))
        sources.append(source)
    observed.loc[missing] = fills
    out["expected_spread_bps"] = observed
    out["policy_spread_bps"] = observed
    out["expected_half_spread_bps"] = 0.5 * observed
    out["spread_cost_bps"] = 0.5 * observed
    out["exit_quote_half_spread_bps"] = 0.5 * observed
    out["exit_spread_cost_bps"] = 0.5 * observed
    audit = {
        "monitor_root": str(monitor_root),
        "monitor_files": int(len(files)),
        "monitor_rows": int(len(monitor)),
        "monitor_symbols": int(monitor["symbol"].nunique()),
        "monitor_start_utc": str(monitor["observed_ts"].min()),
        "monitor_end_utc": str(monitor["observed_ts"].max()),
        "quantile": float(quantile),
        "timing": "strictly observed_ts < candidate entry timestamp",
        "missing_rows_before": int(missing.sum()),
        "filled_from_asset_monitor_quantile": int(np.sum(np.asarray(sources) == "asset_monitor_quantile")),
        "filled_from_global_monitor_quantile": int(np.sum(np.asarray(sources) == "global_monitor_quantile")),
        "support_min": int(np.min(supports)) if supports else 0,
        "support_median": float(np.median(supports)) if supports else 0.0,
        "support_p10": float(np.quantile(supports, 0.10)) if supports else 0.0,
        "imputed_spread_bps_mean": float(np.mean(fills)) if fills else np.nan,
        "imputed_spread_bps_median": float(np.median(fills)) if fills else np.nan,
        "missing_rows_after": int(out["expected_spread_bps"].isna().sum()),
    }
    return out, audit


def _cost_rows(rows: pd.DataFrame, multiplier: float) -> pd.DataFrame:
    out = rows.copy()
    out["spread_cost_bps"] = pd.to_numeric(out["spread_cost_bps"], errors="raise") * multiplier
    out["exit_spread_cost_bps"] = (
        pd.to_numeric(out["exit_spread_cost_bps"], errors="raise") * multiplier
    )
    return out


def _attach_ev_rank(rows: pd.DataFrame, rich: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach the causal corrected-EV rank used by live portfolio priority."""
    out = rows.copy()
    column = "threshold_basis_corrected_expected_ev_rank"
    if column not in out.columns:
        if rich is None:
            raise RuntimeError("Corrected EV rank is absent and no admitted ledger was supplied")
        keys = ["timestamp", "symbol", "side_name"]
        source = rich.loc[:, keys + [column]].copy()
        source["timestamp"] = pd.to_datetime(source["timestamp"], utc=True)
        out = out.merge(source, on=keys, how="left", validate="one_to_one")
    rank = pd.to_numeric(out[column], errors="coerce")
    if rank.isna().any():
        raise RuntimeError(f"Missing corrected EV rank for {int(rank.isna().sum())} rows")
    out["ev_rank_pct"] = rank.clip(0.0, 1.0)
    return out


def _fill_context(context: pd.DataFrame, fit_idx: np.ndarray) -> tuple[pd.DataFrame, dict[str, int]]:
    out = context.copy()
    counts: dict[str, int] = {}
    for column in out.columns:
        values = pd.to_numeric(out[column], errors="coerce")
        missing = ~np.isfinite(values.to_numpy(float))
        counts[column] = int(missing.sum())
        if missing.any():
            median = float(np.nanmedian(values.iloc[fit_idx].to_numpy(float)))
            values.loc[missing] = median
        out[column] = values
    if not np.isfinite(out.to_numpy(float)).all():
        raise RuntimeError("Context remains non-finite after frozen training-median fallback")
    return out, counts


def _sizing_profiles(
    data: ExperimentData,
    fit_idx: np.ndarray,
    geometry: Mapping[str, Mapping[str, Any]],
    context: pd.DataFrame,
    *,
    current_strength: float,
    current_ood_weight: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fit_outputs = data.simulate(fit_idx, geometry, FAMILY_TRAILING_ONLY)
    grid = {
        (1.5, 0.0), (1.5, 0.5), (1.5, 1.0),
        (3.0, 0.0), (3.0, 0.5), (3.0, 1.0),
        (4.5, 0.0), (4.5, 0.5), (4.5, 1.0),
        (float(current_strength), float(current_ood_weight)),
    }
    profiles: dict[str, np.ndarray] = {}
    states: dict[str, Any] = {}
    all_idx = np.arange(len(data.rows), dtype=np.int64)
    for strength, ood_weight in sorted(grid):
        name = f"strength_{strength:g}__ood_{ood_weight:g}"
        sizes, state = _bayesian_sizes(
            data,
            fit_idx,
            all_idx,
            fit_outputs,
            context,
            strength=strength,
            ood_weight=ood_weight,
        )
        profiles[name] = sizes
        states[name] = state
    return profiles, states


def _metrics_with_deltas(frame: pd.DataFrame, keys: list[str], baseline_arm: str) -> pd.DataFrame:
    baseline = frame.loc[
        frame["arm"].eq(baseline_arm), keys + ["net_pnl_bankroll", "net_ev_per_trade", "trades"]
    ].rename(
        columns={
            "net_pnl_bankroll": "baseline_net_pnl_bankroll",
            "net_ev_per_trade": "baseline_net_ev_per_trade",
            "trades": "baseline_trades",
        }
    )
    out = frame.merge(baseline, on=keys, how="left", validate="many_to_one")
    out["delta_net_pnl_vs_baseline"] = out["net_pnl_bankroll"] - out["baseline_net_pnl_bankroll"]
    out["delta_net_ev_vs_baseline"] = out["net_ev_per_trade"] - out["baseline_net_ev_per_trade"]
    return out


def _trial_from_geometry(params: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    long = params["long"]
    short = params["short"]
    result: dict[str, float] = {}
    for key, trial_key in (
        ("sl_mult", "sl_mult"),
        ("trailing_activation_mult", "trailing_activation_mult"),
        ("giveback_beta", "giveback_beta"),
    ):
        lv, sv = float(long[key]), float(short[key])
        result[trial_key] = float(np.sqrt(lv * sv))
        result[f"side_delta_{'activation' if key == 'trailing_activation_mult' else ('beta' if key == 'giveback_beta' else 'sl')}"] = float(
            0.5 * np.log(lv / sv)
        )
    result["trailing_power"] = float(long["trailing_power"])
    result["trailing_squash_divisor"] = float(long["trailing_squash_divisor"])
    return result


def _wallet_priority_order(
    data: ExperimentData, indices: np.ndarray
) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    return idx[np.lexsort((-data.rank[idx], data.timestamps[idx]))]


def _wallet_objective(
    data: ExperimentData,
    indices: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    size_multiplier: np.ndarray,
    *,
    holding_efficiency_weight: float,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Score an exact size-aware 80% marked-notional replay."""
    idx = np.asarray(indices, dtype=np.int64)
    base_fraction = 0.075 + 0.075 * np.power(np.clip(data.rank[idx], 0.0, 1.0), 1.1)
    replay = replay_marked_notional_wallet(
        timestamps_ns=data.timestamps[idx],
        symbol_codes=data.symbol_codes[idx],
        side=data.side[idx],
        raw_entry_prices=data.open0[idx],
        entry_half_spread_bps=data.entry_spread[idx],
        close_paths=data.close[idx],
        exit_bars=np.asarray(outputs["exit_bars"]),
        net_returns=np.asarray(outputs["net_return"]),
        requested_fractions=base_fraction * np.asarray(size_multiplier[idx]),
        bar_minutes=data.spec.bar_minutes,
        max_wallet_invested=0.80,
        max_new_per_bar=2,
        initial_wallet=1.0,
    )
    selected = np.asarray(replay["selected"], dtype=bool)
    admitted = np.asarray(replay["admitted_notional"], dtype=float)
    pnl = admitted * np.nan_to_num(np.asarray(outputs["net_return"]), nan=0.0)
    chosen = np.flatnonzero(selected)
    week = pd.to_datetime(data.timestamps[idx], utc=True).to_period("W").astype(str)
    weekly = pd.Series(pnl).groupby(week, sort=True).sum().to_numpy(float)
    if not len(weekly):
        weekly = np.zeros(1, dtype=float)
    equity = np.asarray(replay["equity_before"], dtype=float)
    finite_equity = equity[np.isfinite(equity)]
    if len(finite_equity):
        peak = np.maximum.accumulate(finite_equity)
        drawdown = float(np.min(finite_equity / np.maximum(peak, 1e-12) - 1.0))
    else:
        drawdown = 0.0
    total = float(replay["final_wallet"] - 1.0)
    hours = np.maximum((np.asarray(outputs["exit_bars"], dtype=float) + 1.0) / 60.0, 1.0 / 60.0)
    holding_denominator = float(np.sum(admitted * np.power(hours, 0.8)))
    holding_efficiency = total / max(holding_denominator, 1e-12)
    stable = float(weekly.mean() - 0.5 * weekly.std() + 0.25 * weekly.min() - 0.10 * abs(drawdown))
    score = stable + float(holding_efficiency_weight) * holding_efficiency
    requested = base_fraction * np.asarray(size_multiplier[idx])
    diagnostics = {
        "objective": score,
        "stable_portfolio_component": stable,
        "holding_efficiency": holding_efficiency,
        "holding_efficiency_weight": float(holding_efficiency_weight),
        "net_pnl_bankroll": total,
        "net_ev_per_trade": float(np.sum(pnl) / max(np.sum(admitted), 1e-12)),
        "worst_week": float(weekly.min()),
        "mean_week": float(weekly.mean()),
        "max_drawdown": drawdown,
        "n_trades": int(len(chosen)),
        "mean_holding_hours": float(np.mean(hours[chosen])) if len(chosen) else np.nan,
        "p80_holding_hours": float(np.quantile(hours[chosen], 0.8)) if len(chosen) else np.nan,
        "mean_requested_fraction": float(np.mean(requested[chosen])) if len(chosen) else np.nan,
        "mean_admitted_notional": float(np.mean(admitted[chosen])) if len(chosen) else np.nan,
        "max_wallet_utilization_before": float(np.nanmax(replay["wallet_cap_utilization_before"])),
        "wallet_cap_rejections": int(np.sum(np.asarray(replay["rejection_code"]) == 4)),
    }
    return score, diagnostics, replay


def _wallet_local_robustness(
    data: ExperimentData,
    indices: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    size_multiplier: np.ndarray,
    *,
    holding_efficiency_weight: float,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ordered = _wallet_priority_order(data, indices)
    scores: list[float] = []
    for _ in range(int(n)):
        trial_params = {side: dict(values) for side, values in params.items()}
        geometry_scale = float(np.exp(rng.normal(0.0, 0.035)))
        beta_scale = float(np.exp(rng.normal(0.0, 0.05)))
        for values in trial_params.values():
            values["sl_mult"] = float(values["sl_mult"] * geometry_scale)
            for key in (
                "trailing_activation_mult",
                "trailing_activation_mult_2",
                "trailing_activation_mult_3",
            ):
                values[key] = float(values[key] * geometry_scale)
            for key in ("giveback_beta", "giveback_beta_2", "giveback_beta_3"):
                values[key] = float(values[key] * beta_scale)
        outputs = data.simulate(ordered, trial_params, FAMILY_TRAILING_ONLY)
        score, _, _ = _wallet_objective(
            data,
            ordered,
            outputs,
            size_multiplier,
            holding_efficiency_weight=holding_efficiency_weight,
        )
        scores.append(float(score))
    values = np.asarray(scores)
    return {
        "n": int(len(values)),
        "median_objective": float(np.median(values)),
        "p10_objective": float(np.quantile(values, 0.10)),
        "worst_objective": float(np.min(values)),
        "positive_rate": float(np.mean(values > 0.0)),
    }


def _wallet_report(
    data: ExperimentData,
    indices: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    size_multiplier: np.ndarray,
    *,
    arm: str,
    holding_efficiency_weight: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ordered = _wallet_priority_order(data, indices)
    outputs = data.simulate(ordered, params, FAMILY_TRAILING_ONLY)
    _, metrics, replay = _wallet_objective(
        data,
        ordered,
        outputs,
        size_multiplier,
        holding_efficiency_weight=holding_efficiency_weight,
    )
    selected = np.asarray(replay["selected"], dtype=bool)
    admitted = np.asarray(replay["admitted_notional"], dtype=float)
    net_return = np.asarray(outputs["net_return"], dtype=float)
    pnl = admitted * np.nan_to_num(net_return, nan=0.0)
    report_rows = data.rows.iloc[ordered].copy().reset_index(drop=True)
    report_rows["week"] = pd.to_datetime(report_rows["timestamp"], utc=True).dt.to_period("W").astype(str)
    report_rows["selected"] = selected
    report_rows["admitted_notional"] = admitted
    report_rows["net_return"] = net_return
    report_rows["pnl"] = pnl
    report_rows["exit_bars"] = np.asarray(outputs["exit_bars"])
    report_rows["holding_hours"] = (report_rows["exit_bars"] + 1.0) / 60.0
    report_rows["size_multiplier"] = size_multiplier[ordered]
    report_rows["wallet_before"] = replay["wallet_before"]
    report_rows["equity_before"] = replay["equity_before"]
    report_rows["marked_notional_before"] = replay["marked_notional_before"]
    report_rows["wallet_cap_utilization_before"] = replay["wallet_cap_utilization_before"]
    report_rows["rejection_code"] = replay["rejection_code"]
    report_rows["trailing_exit_layer"] = np.asarray(outputs["trailing_exit_layer"])
    weekly_rows: list[dict[str, Any]] = []
    for week, part in report_rows.groupby("week", sort=True):
        chosen = part.loc[part["selected"]]
        weekly_rows.append(
            {
                "arm": arm,
                "week": str(week),
                "trades": int(len(chosen)),
                "net_pnl_bankroll": float(chosen["pnl"].sum()),
                "net_ev_per_trade": float(chosen["pnl"].sum() / max(chosen["admitted_notional"].sum(), 1e-12)),
                "mean_holding_hours": float(chosen["holding_hours"].mean()) if len(chosen) else np.nan,
                "mean_wallet_utilization": float(chosen["wallet_cap_utilization_before"].mean()) if len(chosen) else np.nan,
            }
        )
    metrics = {"arm": arm, **metrics}
    ledger = report_rows.loc[report_rows["selected"]].copy()
    ledger["arm"] = arm
    return metrics, pd.DataFrame(weekly_rows), ledger


def _optimise_expanded(
    data: ExperimentData,
    indices: np.ndarray,
    *,
    trials_per_seed: int,
    seeds: list[int],
    initial_geometry: Mapping[str, Mapping[str, Any]],
    sizing_profiles: Mapping[str, np.ndarray],
    holding_efficiency_weight: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Jointly search 1--3 causal layers, EV/Bayesian size, and wallet capacity."""
    best_value = -1e100
    best_params: dict[str, dict[str, Any]] | None = None
    best_meta: dict[str, Any] = {}
    seed_summaries: list[dict[str, Any]] = []
    default_sizing = next(iter(sizing_profiles))
    for seed in seeds:
        sampler = optuna.samplers.TPESampler(
            seed=int(seed),
            multivariate=True,
            group=True,
            n_startup_trials=min(48, max(12, trials_per_seed // 4)),
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        def objective(trial: optuna.Trial) -> float:
            global_sl = trial.suggest_float("sl_mult", 1.5, 8.0)
            layer_count = trial.suggest_int("trailing_layer_count", 1, 3)
            act_1 = trial.suggest_float("trailing_activation_mult", 0.10, 1.50)
            act_2 = act_1
            act_3 = act_1
            if layer_count >= 2:
                act_2 = act_1 + trial.suggest_float("activation_increment_2", 0.10, 3.00)
            if layer_count >= 3:
                act_3 = act_2 + trial.suggest_float("activation_increment_3", 0.10, 4.00)
            power = trial.suggest_float("trailing_power", 1.1, 4.0)
            divisor = trial.suggest_float("trailing_squash_divisor", 1.0, 12.0, log=True)
            beta_1 = trial.suggest_float("giveback_beta", 0.80, 3.00)
            beta_2 = beta_1
            beta_3 = beta_1
            if layer_count >= 2:
                beta_2 = beta_1 * trial.suggest_float("beta_retention_2", 0.20, 0.95)
            if layer_count >= 3:
                beta_3 = beta_2 * trial.suggest_float("beta_retention_3", 0.20, 0.95)
            delta_sl = trial.suggest_float("side_delta_sl", -0.40, 0.40)
            delta_act = trial.suggest_float("side_delta_activation", -0.40, 0.40)
            delta_beta = trial.suggest_float("side_delta_beta", -0.40, 0.40)
            sizing_profile = trial.suggest_categorical(
                "sizing_profile", list(sizing_profiles)
            )
            params: dict[str, dict[str, Any]] = {}
            for side, sign in (("long", 1.0), ("short", -1.0)):
                values = _side_defaults(data.deployed_by_side[side])
                activation_scale = float(np.exp(sign * delta_act))
                beta_scale = float(np.exp(sign * delta_beta))
                values.update(
                    {
                        "sl_mult": float(np.clip(global_sl * np.exp(sign * delta_sl), 1.5, 8.0)),
                        "trailing_layer_count": int(layer_count),
                        "trailing_activation_mult": float(np.clip(act_1 * activation_scale, 0.05, 8.0)),
                        "trailing_activation_mult_2": float(np.clip(act_2 * activation_scale, 0.05, 8.0)),
                        "trailing_activation_mult_3": float(np.clip(act_3 * activation_scale, 0.05, 8.0)),
                        "trailing_power": power,
                        "trailing_squash_divisor": divisor,
                        "giveback_beta": float(np.clip(beta_1 * beta_scale, 0.15, 4.0)),
                        "giveback_beta_2": float(np.clip(beta_2 * beta_scale, 0.10, 4.0)),
                        "giveback_beta_3": float(np.clip(beta_3 * beta_scale, 0.05, 4.0)),
                        "trailing_activation_decay_half_life_minutes": 0.0,
                    }
                )
                params[side] = values
            ordered = _wallet_priority_order(data, indices)
            outputs = data.simulate(ordered, params, FAMILY_TRAILING_ONLY)
            score, diagnostics, _ = _wallet_objective(
                data,
                ordered,
                outputs,
                sizing_profiles[str(sizing_profile)],
                holding_efficiency_weight=holding_efficiency_weight,
            )
            penalty = float(
                0.0005
                * np.sum((np.asarray([delta_sl, delta_act, delta_beta]) / 0.25) ** 2)
            )
            trial.set_user_attr("runtime_params", params)
            trial.set_user_attr("diagnostics", diagnostics)
            trial.set_user_attr("sizing_profile", str(sizing_profile))
            trial.set_user_attr("shrinkage_penalty", penalty)
            return float(score - penalty)

        study.optimize(objective, n_trials=trials_per_seed, show_progress_bar=False)
        params = study.best_trial.user_attrs["runtime_params"]
        summary = {
            "seed": int(seed),
            "best_value_penalized": float(study.best_value),
            "best_trial": int(study.best_trial.number),
            "trial_params": dict(study.best_trial.params),
            "diagnostics": study.best_trial.user_attrs["diagnostics"],
            "sizing_profile": study.best_trial.user_attrs["sizing_profile"],
        }
        seed_summaries.append(summary)
        if study.best_value > best_value:
            best_value = float(study.best_value)
            best_params = params
            best_meta = summary
    if best_params is None:
        raise RuntimeError("Expanded optimizer produced no valid trial")
    return best_params, {
        "bounds": {
            "sl_mult": [1.5, 8.0],
            "trailing_layer_count": [1, 3],
            "trailing_activation_mult_1": [0.10, 1.50],
            "activation_increment_2": [0.10, 3.00],
            "activation_increment_3": [0.10, 4.00],
            "trailing_power": [1.1, 4.0],
            "trailing_squash_divisor": [1.0, 12.0],
            "giveback_beta_1": [0.80, 3.00],
            "later_beta_retention": [0.20, 0.95],
            "side_deltas": [-0.40, 0.40],
        },
        "best_value_penalized": best_value,
        "best": best_meta,
        "seeds": seed_summaries,
        "total_trials": int(len(seeds) * trials_per_seed),
        "sizing_profiles": list(sizing_profiles),
        "default_sizing_profile": default_sizing,
        "holding_efficiency_weight": float(holding_efficiency_weight),
        "time_dependent_tightening": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "joint_layered_wallet80_holdeff_l2_20260718_v1",
    )
    parser.add_argument("--outer-trials-per-seed", type=int, default=160)
    parser.add_argument("--final-trials-per-seed", type=int, default=320)
    parser.add_argument("--search-seeds", type=int, default=3)
    parser.add_argument("--local-perturbations", type=int, default=128)
    parser.add_argument("--holding-efficiency-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--spread-monitor-root",
        type=Path,
        default=Path("data_perp/exchanges/krakenfutures/spread_snapshots"),
    )
    parser.add_argument("--monitor-quantile", type=float, choices=(0.75, 0.85), default=0.75)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.monotonic()

    deployed, _ = _load_deployed_side_params(PARENT)
    base_spec = ConstrainedReplaySpec()
    alt_spec = replace(base_spec, fee_per_side=0.0)
    saved_params = json.loads(PARAMS.read_text())
    fixed_geometry = saved_params["fold_3"]["full_train_parent"]
    sizing_params = saved_params["fold_3"]["sizing"]

    rich_rank = pd.read_parquet(
        RICH,
        columns=[
            "timestamp", "symbol", "side_name",
            "threshold_basis_corrected_expected_ev_rank",
        ],
    )
    old_rows = _attach_ev_rank(pd.read_parquet(OLD_CANDIDATES), rich_rank)
    old_rows["timestamp"] = pd.to_datetime(old_rows["timestamp"], utc=True)
    old_rows = old_rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    old_context, _, old_context_audit = _load_context(old_rows, RICH, POSTERIOR)
    old_atr = _load_atr(old_rows, OLD_ATR)
    oo, oh, ol, oc, ov, old_path_manifest = _load_or_build_path_cache(
        old_rows, store_root=STORE, cache_dir=OLD_CACHE, spec=base_spec, rebuild=False
    )
    old_volume_cache = np.memmap(
        VOLUME_CACHE, mode="r", dtype="float32", shape=(len(old_rows), base_spec.path_len)
    )
    old_liquid = np.isfinite(old_volume_cache[:, 0]) & (old_volume_cache[:, 0] > 0.0)

    forward = _attach_ev_rank(
        pd.read_parquet(FORWARD_DIR / "forward_candidates_jul11_16.parquet")
    )
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    forward_context, forward_context_audit = _forward_context(forward)
    forward_atr = _load_atr(forward, FORWARD_DIR / "causal_entry_atr_audit.parquet")
    fo, fh, fl, fc, fv, forward_path_manifest = _load_or_build_path_cache(
        forward,
        store_root=STORE,
        cache_dir=FORWARD_DIR / "path_cache",
        spec=base_spec,
        rebuild=False,
    )
    forward_volume = _entry_minute_volume(forward, STORE)
    forward_liquid = np.isfinite(forward_volume) & (forward_volume > 0.0)

    july17 = _attach_ev_rank(
        pd.read_parquet(DAILY_DIR / "july17_partial_candidates.parquet")
    )
    july17["timestamp"] = pd.to_datetime(july17["timestamp"], utc=True)
    july17 = july17.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    spread_reference = pd.read_parquet(
        "data_perp/reports/july_01_16_current_policy_metrics_20260717/"
        "current_policy_candidates_through_july16.parquet"
    )
    rebuilt17, july17_context = _prediction_candidates(
        DAILY_DIR / "jul17_prediction_ledger.parquet",
        pd.Timestamp("2026-07-17 08:00", tz="UTC"),
        spread_reference,
    )
    keys = ["timestamp", "symbol", "side", "rank_pct"]
    if not july17[keys].equals(rebuilt17[keys]):
        raise RuntimeError("July 17 candidates do not align to reconstructed context")
    july17_atr = _load_atr(july17, DAILY_DIR / "july17_causal_entry_atr_audit.parquet")
    jo, jh, jl, jc, jv, july17_path_manifest = _load_or_build_path_cache(
        july17,
        store_root=STORE,
        cache_dir=DAILY_DIR / "july17_path_cache",
        spec=base_spec,
        rebuild=False,
    )
    july17_volume = _entry_minute_volume(july17, STORE)
    july17_liquid = np.isfinite(july17_volume) & (july17_volume > 0.0)

    sizing_context_columns = [
        "expected_net_ev_after_1pct_mlp_direct",
        "meta_hit_probability_uncertainty_p1mp",
        "gmm_ood_score",
        "cluster_entropy_norm",
    ]
    full_rows = pd.concat([old_rows, forward, july17], ignore_index=True)
    full_context = pd.concat(
        [
            old_context.loc[:, sizing_context_columns],
            forward_context.loc[:, sizing_context_columns],
            july17_context.loc[:, sizing_context_columns],
        ],
        ignore_index=True,
    )
    old_baseline_data = ExperimentData(
        old_rows, oo, oh, ol, oc, ov, old_atr, base_spec, deployed
    )
    sizing_fit_idx = _indices_between(old_baseline_data, "2026-05-01", "2026-06-14")
    full_context, context_fallback_counts = _fill_context(full_context, sizing_fit_idx)
    sizing_fit_outputs = old_baseline_data.simulate(
        sizing_fit_idx, fixed_geometry, FAMILY_TRAILING_ONLY
    )
    sizing_data = SimpleNamespace(
        rows=full_rows,
        side=pd.to_numeric(full_rows["side"], errors="coerce").to_numpy(float),
        rank=pd.to_numeric(full_rows["rank_pct"], errors="coerce").to_numpy(float),
    )
    full_apply_idx = np.arange(len(full_rows), dtype=np.int64)
    full_sizes, sizing_state = _bayesian_sizes(
        sizing_data,
        sizing_fit_idx,
        full_apply_idx,
        sizing_fit_outputs,
        full_context,
        strength=float(sizing_params["strength"]),
        ood_weight=float(sizing_params["ood_weight"]),
    )
    frozen_sizes = np.concatenate(
        [
            full_sizes[: len(old_rows)][old_liquid],
            full_sizes[len(old_rows) : len(old_rows) + len(forward)][forward_liquid],
            full_sizes[len(old_rows) + len(forward) :][july17_liquid],
        ]
    )

    row_parts = [
        old_rows.loc[old_liquid],
        forward.loc[forward_liquid],
        july17.loc[july17_liquid],
    ]
    rows = pd.concat(row_parts, ignore_index=True)
    context = pd.concat(
        [
            old_context.loc[old_liquid, sizing_context_columns],
            forward_context.loc[forward_liquid, sizing_context_columns],
            july17_context.loc[july17_liquid, sizing_context_columns],
        ],
        ignore_index=True,
    )
    rows, spread_audit = _monitor_asset_spread_fill(
        rows,
        monitor_root=args.spread_monitor_root,
        quantile=float(args.monitor_quantile),
    )
    arrays = [
        np.concatenate([np.asarray(a[old_liquid]), np.asarray(b[forward_liquid]), np.asarray(c[july17_liquid])])
        for a, b, c in ((old_atr, forward_atr, july17_atr),)
    ]
    atr = arrays[0]
    open0 = np.concatenate([np.asarray(oo[old_liquid]), np.asarray(fo[forward_liquid]), np.asarray(jo[july17_liquid])])
    high = np.concatenate([np.asarray(oh[old_liquid]), np.asarray(fh[forward_liquid]), np.asarray(jh[july17_liquid])])
    low = np.concatenate([np.asarray(ol[old_liquid]), np.asarray(fl[forward_liquid]), np.asarray(jl[july17_liquid])])
    close = np.concatenate([np.asarray(oc[old_liquid]), np.asarray(fc[forward_liquid]), np.asarray(jc[july17_liquid])])
    valid = np.concatenate([np.asarray(ov[old_liquid]), np.asarray(fv[forward_liquid]), np.asarray(jv[july17_liquid])])
    if not rows["timestamp"].is_monotonic_increasing or not valid.all():
        raise RuntimeError("Filtered combined replay stream is incomplete or non-chronological")

    baseline_data = ExperimentData(rows, open0, high, low, close, valid, atr, base_spec, deployed)
    alt_rows = _cost_rows(rows, 1.5)
    alt_data = ExperimentData(alt_rows, open0, high, low, close, valid, atr, alt_spec, deployed)

    seeds = [args.seed + 10_000 * i for i in range(args.search_seeds)]
    fold_records: list[dict[str, Any]] = []
    fold_params: dict[str, Any] = {}
    for fold_no, fold in enumerate(FOLDS, start=1):
        train_idx = _indices_between(alt_data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(alt_data, fold["validation_start"], fold["validation_end"])
        baseline_geometry = saved_params[fold["fold"]]["full_train_parent"]
        fold_context, fold_context_fallbacks = _fill_context(context, train_idx)
        profiles, sizing_states = _sizing_profiles(
            alt_data,
            train_idx,
            baseline_geometry,
            fold_context,
            current_strength=float(sizing_params["strength"]),
            current_ood_weight=float(sizing_params["ood_weight"]),
        )
        params, diagnostics = _optimise_expanded(
            alt_data,
            train_idx,
            trials_per_seed=args.outer_trials_per_seed,
            seeds=[seed + fold_no * 1_000 for seed in seeds],
            initial_geometry=baseline_geometry,
            sizing_profiles=profiles,
            holding_efficiency_weight=args.holding_efficiency_weight,
        )
        best_profile = str(diagnostics["best"]["sizing_profile"])
        ordered_outer = _wallet_priority_order(alt_data, outer_idx)
        outputs = alt_data.simulate(ordered_outer, params, FAMILY_TRAILING_ONLY)
        _, metrics, _ = _wallet_objective(
            alt_data,
            ordered_outer,
            outputs,
            profiles[best_profile],
            holding_efficiency_weight=args.holding_efficiency_weight,
        )
        fixed_outputs = alt_data.simulate(ordered_outer, baseline_geometry, FAMILY_TRAILING_ONLY)
        current_profile = min(
            profiles,
            key=lambda name: abs(float(name.split("__")[0].split("_")[1]) - float(sizing_params["strength"]))
            + abs(float(name.rsplit("_", 1)[1]) - float(sizing_params["ood_weight"])),
        )
        _, fixed_metrics, _ = _wallet_objective(
            alt_data,
            ordered_outer,
            fixed_outputs,
            profiles[current_profile],
            holding_efficiency_weight=args.holding_efficiency_weight,
        )
        fold_records.extend(
            [
                {"fold": fold["fold"], "arm": "reoptimized_alt_cost", **metrics},
                {"fold": fold["fold"], "arm": "fixed_original_geometry_alt_cost", **fixed_metrics},
            ]
        )
        fold_params[fold["fold"]] = {
            "train": [fold["train_start"], fold["train_end"]],
            "validation": [fold["validation_start"], fold["validation_end"]],
            "params_by_side": params,
            "selected_sizing_profile": best_profile,
            "sizing_states": sizing_states,
            "context_fallback_counts": fold_context_fallbacks,
            "optimizer": diagnostics,
        }
        pd.DataFrame(fold_records).to_csv(args.output_dir / "nested_oos_fold_metrics.partial.csv", index=False)
        (args.output_dir / "nested_params.partial.json").write_text(
            json.dumps(_json_safe(fold_params), indent=2)
        )

    final_train_idx = _indices_between(alt_data, "2026-05-01", "2026-07-01")
    final_context, final_context_fallbacks = _fill_context(context, final_train_idx)
    final_profiles, final_sizing_states = _sizing_profiles(
        alt_data,
        final_train_idx,
        fixed_geometry,
        final_context,
        current_strength=float(sizing_params["strength"]),
        current_ood_weight=float(sizing_params["ood_weight"]),
    )
    final_params, final_optimizer = _optimise_expanded(
        alt_data,
        final_train_idx,
        trials_per_seed=args.final_trials_per_seed,
        seeds=[seed + 900_000 for seed in seeds],
        initial_geometry=fixed_geometry,
        sizing_profiles=final_profiles,
        holding_efficiency_weight=args.holding_efficiency_weight,
    )
    final_profile = str(final_optimizer["best"]["sizing_profile"])
    robustness = _wallet_local_robustness(
        alt_data,
        final_train_idx,
        final_params,
        final_profiles[final_profile],
        holding_efficiency_weight=args.holding_efficiency_weight,
        n=args.local_perturbations,
        seed=args.seed + 950_000,
    )

    report_idx = _indices_between(alt_data, "2026-06-29", "2026-07-18")
    report_rows = rows.iloc[report_idx].reset_index(drop=True)
    report_sizes = frozen_sizes[report_idx]
    july_eligible = np.ones(len(report_rows), dtype=bool)
    arms: list[tuple[str, ExperimentData, Mapping[str, Mapping[str, Any]], np.ndarray]] = [
        ("baseline_cost_fixed_geometry", baseline_data, fixed_geometry, report_sizes),
        ("spread_1p5_fixed_geometry", alt_data, fixed_geometry, report_sizes),
        ("spread_1p5_reoptimized_geometry", alt_data, final_params, report_sizes),
    ]
    global_records: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    selected_ledgers: list[pd.DataFrame] = []
    for arm, data, geometry, multipliers in arms:
        outputs = data.simulate(report_idx, geometry, FAMILY_TRAILING_ONLY)
        selected = _select_after_filter(report_rows, outputs, july_eligible)
        global_record, weekly = _period_metrics(
            report_rows,
            outputs,
            multipliers,
            selected,
            policy="joint_trailing_plus_bayesian_raw",
            cost_model=arm,
        )
        global_record["arm"] = arm
        weekly["arm"] = arm
        global_records.append(global_record)
        weekly_parts.append(weekly)
        chosen = np.flatnonzero(
            selected
            & pd.to_datetime(report_rows["timestamp"], utc=True).ge(
                pd.Timestamp("2026-07-01", tz="UTC")
            ).to_numpy()
        )
        ledger = report_rows.iloc[chosen][
            ["timestamp", "symbol", "side_name", "policy_archetype", "rank_pct", "expected_spread_bps"]
        ].copy()
        ledger["arm"] = arm
        ledger["exit_bars"] = np.asarray(outputs["exit_bars"])[chosen]
        ledger["net_return"] = np.asarray(outputs["net_return"])[chosen]
        ledger["size_multiplier"] = multipliers[chosen]
        selected_ledgers.append(ledger)

    global_metrics = _metrics_with_deltas(
        pd.DataFrame(global_records), ["policy"], "baseline_cost_fixed_geometry"
    )
    weekly_metrics = _metrics_with_deltas(
        pd.concat(weekly_parts, ignore_index=True),
        ["policy", "week"],
        "baseline_cost_fixed_geometry",
    )
    fold_metrics = pd.DataFrame(fold_records)
    fold_summary = (
        fold_metrics.groupby("arm", as_index=False)
        .agg(
            folds=("fold", "size"),
            mean_objective=("objective", "mean"),
            worst_objective=("objective", "min"),
            mean_pnl=("net_pnl_bankroll", "mean"),
            worst_fold_pnl=("net_pnl_bankroll", "min"),
            worst_week=("worst_week", "min"),
            worst_drawdown=("max_drawdown", "min"),
            positive_folds=("net_pnl_bankroll", lambda values: int(np.sum(np.asarray(values) > 0.0))),
        )
    )
    for row_i, row in fold_summary.iterrows():
        values = fold_metrics.loc[fold_metrics["arm"].eq(row["arm"]), "objective"].to_numpy(float)
        fold_summary.loc[row_i, "stable_fold_objective"] = (
            values.mean() - 0.5 * values.std() + 0.25 * values.min()
        )

    fold_metrics.to_csv(args.output_dir / "nested_oos_fold_metrics.csv", index=False)
    fold_summary.to_csv(args.output_dir / "nested_oos_summary.csv", index=False)
    global_metrics.to_csv(args.output_dir / "july_global_metrics.csv", index=False)
    weekly_metrics.to_csv(args.output_dir / "july_weekly_metrics.csv", index=False)
    pd.concat(selected_ledgers, ignore_index=True).to_parquet(
        args.output_dir / "july_selected_trade_ledger.parquet", index=False
    )
    params_artifact = {
        "fixed_original_geometry": fixed_geometry,
        "reoptimized_spread_1p5_no_fee_geometry": final_params,
        "optimizer": final_optimizer,
        "local_robustness": robustness,
        "fold_params": fold_params,
    }
    (args.output_dir / "selected_geometries.json").write_text(
        json.dumps(_json_safe(params_artifact), indent=2)
    )
    manifest = {
        "status": "complete_with_partial_july17",
        "evidence": {
            "nested_folds": "policy-selection OOS",
            "july": "evaluation-only diagnostic; July was previously inspected",
        },
        "cost": "zero fees; 1.5 times the full asset spread via 1.5 times each executable half-spread",
        "spread_imputation": spread_audit,
        "liquidity": {
            "rule": "strictly positive event-minute 1m volume before capacity",
            "old_rows": int(len(old_rows)),
            "old_liquid_rows": int(old_liquid.sum()),
            "forward_rows": int(len(forward)),
            "forward_liquid_rows": int(forward_liquid.sum()),
            "july17_rows": int(len(july17)),
            "july17_liquid_rows": int(july17_liquid.sum()),
        },
        "search": {
            "family": "joint trailing-only using total MFE",
            "outer_trials": int(3 * args.search_seeds * args.outer_trials_per_seed),
            "final_trials": int(args.search_seeds * args.final_trials_per_seed),
            "seeds": seeds,
            "local_perturbations": int(args.local_perturbations),
            "final_train": ["2026-05-01", "2026-07-01"],
        },
        "sizing": {
            "contract": "raw Bayesian multipliers frozen from original May1-Jun14 baseline-cost winner",
            "params": sizing_params,
            "state": sizing_state,
        },
        "context_fallback_counts": context_fallback_counts,
        "old_context_audit": old_context_audit,
        "forward_context_audit": forward_context_audit,
        "path_manifests": [old_path_manifest, forward_path_manifest, july17_path_manifest],
        "july17_entry_cutoff_exclusive_utc": "2026-07-17T08:00:00Z",
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2)
    )
    print("\nNESTED OOS SUMMARY\n", fold_summary.to_string(index=False), flush=True)
    print("\nJULY GLOBAL\n", global_metrics.to_string(index=False), flush=True)
    print("\nJULY WEEKLY\n", weekly_metrics.to_string(index=False), flush=True)
    print(f"elapsed_seconds={time.monotonic() - started:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
