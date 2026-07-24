#!/usr/bin/env python3
"""Jointly optimize layered exits and EV/Bayesian sizing under wallet capacity."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_constrained import (
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_1m_joint_objective import (
    evaluate_joint_wallet_objective,
)
from extreme_price_movements.simple_policy_candidate_context import (
    RAW_BAYESIAN_CONTEXT_COLUMNS,
    join_candidate_execution_context,
)
from scripts.reoptimise_simple_policy_1m_spread_only_geometry import (
    _cost_rows,
    _fill_context,
    _json_safe,
    _monitor_asset_spread_fill,
)
from scripts.report_simple_policy_1m_winner_daily_july import (
    OLD_ATR,
    OLD_CACHE,
    OLD_CANDIDATES,
    PARAMS,
    PARENT,
    STORE,
)
from scripts.report_simple_policy_1m_winner_daily_nonzero_volume import VOLUME_CACHE
from scripts.run_simple_policy_1m_capital_ablation import (
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (
    ExperimentData,
    _indices_between,
    _side_defaults,
)
from scripts.run_simple_policy_1m_contextual_ablation import _bayesian_sizes, _load_atr

RICH_LEDGER = (
    Path("data_perp/reports/meta_v9_recovery_20260717/")
    / "residual_state_mda95_hier_newaegmm_downstream_retrain_v1"
    / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
)
DEFAULT_OUTPUT = (
    Path("data_perp/reports/simple_policy_1m_joint_trailing_raw_bayesian_champion_20260718_v1")
    / "joint_layered_wallet80_holdeff_l2_20260718_v1"
)


def _params_from_trial(trial: optuna.Trial, data: ExperimentData) -> dict[str, dict[str, Any]]:
    # Layer one is the required early protective trail; layer two is the main
    # profit trail, and a third tightening stage is optional.
    layers = trial.suggest_int("trailing_layer_count", 2, 3)
    act1 = trial.suggest_float("activation_1", 0.10, 1.25, log=True)
    act2 = act1 + (trial.suggest_float("activation_increment_2", 0.10, 3.0, log=True) if layers >= 2 else 0.0)
    act3 = act2 + (trial.suggest_float("activation_increment_3", 0.10, 4.0, log=True) if layers >= 3 else 0.0)
    beta1 = trial.suggest_float("beta_1", 0.50, 2.50)
    beta2 = beta1 * (trial.suggest_float("beta_retention_2", 0.15, 1.0) if layers >= 2 else 1.0)
    beta3 = beta2 * (trial.suggest_float("beta_retention_3", 0.15, 1.0) if layers >= 3 else 1.0)
    sl = trial.suggest_float("sl_mult", 1.5, 8.0)
    power = trial.suggest_float("trailing_power", 1.1, 4.0)
    divisor = trial.suggest_float("trailing_squash_divisor", 1.0, 12.0, log=True)
    delta_sl = trial.suggest_float("side_delta_sl", -0.30, 0.30)
    delta_act = trial.suggest_float("side_delta_activation", -0.30, 0.30)
    delta_beta = trial.suggest_float("side_delta_beta", -0.30, 0.30)
    result: dict[str, dict[str, Any]] = {}
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        values = _side_defaults(data.deployed_by_side[side_name])
        act_scale = np.exp(sign * delta_act)
        beta_scale = np.exp(sign * delta_beta)
        values.update({
            "sl_mult": float(np.clip(sl * np.exp(sign * delta_sl), 1.5, 8.0)),
            "trailing_layer_count": int(layers),
            "trailing_activation_mult": float(np.clip(act1 * act_scale, 0.05, 8.0)),
            "trailing_activation_mult_2": float(np.clip(act2 * act_scale, 0.05, 8.0)),
            "trailing_activation_mult_3": float(np.clip(act3 * act_scale, 0.05, 8.0)),
            "giveback_beta": float(np.clip(beta1 * beta_scale, 0.05, 3.0)),
            "giveback_beta_2": float(np.clip(beta2 * beta_scale, 0.05, 3.0)),
            "giveback_beta_3": float(np.clip(beta3 * beta_scale, 0.05, 3.0)),
            "trailing_power": float(power),
            "trailing_squash_divisor": float(divisor),
            # Multi-layer mode structurally disables time-dependent tightening.
            "trailing_activation_decay_half_life_minutes": 0.0,
            "trailing_activation_decay_start_minutes": 0.0,
            "trailing_activation_min_mult": 1.0,
        })
        result[side_name] = values
    return result


def _subset_objective(
    data: ExperimentData,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    corrected_ev: np.ndarray,
    corrected_rank: np.ndarray,
    sizes: np.ndarray,
    *,
    holding_weight: float,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    return evaluate_joint_wallet_objective(
        rows=data.rows.iloc[idx].reset_index(drop=True),
        timestamps_ns=data.timestamps[idx], symbol_codes=data.symbol_codes[idx], side=data.side[idx],
        raw_entry_prices=data.open0[idx], entry_half_spread_bps=data.entry_spread[idx],
        close_paths=data.close[idx], exit_bars=outputs["exit_bars"], net_returns=outputs["net_return"],
        corrected_ev=corrected_ev[idx], corrected_ev_rank=corrected_rank[idx],
        bayesian_multiplier=sizes[idx], holding_power=0.8,
        holding_efficiency_weight=holding_weight, max_wallet_invested=0.80,
        max_new_per_bar=2, initial_wallet=1.0, bar_minutes=1,
    )


def _sizing_grid(
    data: ExperimentData,
    context: pd.DataFrame,
    fit_idx: np.ndarray,
    baseline_outputs: Mapping[str, np.ndarray],
) -> tuple[dict[tuple[float, float], np.ndarray], dict[str, Any]]:
    grid: dict[tuple[float, float], np.ndarray] = {}
    states: dict[str, Any] = {}
    apply = np.arange(len(data.rows), dtype=np.int64)
    for strength in (1.5, 3.0, 4.5):
        for ood_weight in (0.0, 0.5, 1.0):
            sizes, state = _bayesian_sizes(
                data, fit_idx, apply, baseline_outputs, context,
                strength=strength, ood_weight=ood_weight,
            )
            grid[(strength, ood_weight)] = sizes
            states[f"strength={strength},ood={ood_weight}"] = state
    return grid, states


def _optimize(
    data: ExperimentData,
    idx: np.ndarray,
    context: pd.DataFrame,
    corrected_ev: np.ndarray,
    corrected_rank: np.ndarray,
    baseline_geometry: Mapping[str, Mapping[str, Any]],
    *,
    trials: int,
    seeds: list[int],
    holding_weight: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], np.ndarray, dict[str, Any]]:
    baseline_outputs = data.simulate(idx, baseline_geometry, FAMILY_TRAILING_ONLY)
    size_grid, sizing_states = _sizing_grid(data, context, idx, baseline_outputs)
    best: tuple[float, dict[str, Any], dict[str, dict[str, Any]], np.ndarray, dict[str, Any]] | None = None
    seed_rows = []
    for seed in seeds:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(
                seed=seed, multivariate=True, group=True,
                n_startup_trials=min(48, max(16, trials // 4)),
            ),
        )
        def objective(trial: optuna.Trial) -> float:
            params = _params_from_trial(trial, data)
            strength = trial.suggest_categorical("bayesian_strength", [1.5, 3.0, 4.5])
            ood_weight = trial.suggest_categorical("bayesian_ood_weight", [0.0, 0.5, 1.0])
            sizes = size_grid[(float(strength), float(ood_weight))]
            outputs = data.simulate(idx, params, FAMILY_TRAILING_ONLY)
            score, metrics, _ = _subset_objective(
                data, idx, outputs, corrected_ev, corrected_rank, sizes,
                holding_weight=holding_weight,
            )
            # Mild hierarchical shrinkage; exits and sizing remain jointly selected.
            delta_penalty = 0.001 * sum(
                float(trial.params.get(name, 0.0)) ** 2
                for name in ("side_delta_sl", "side_delta_activation", "side_delta_beta")
            )
            trial.set_user_attr("params_by_side", params)
            trial.set_user_attr("metrics", metrics)
            return float(score - delta_penalty)
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        trial = study.best_trial
        params = trial.user_attrs["params_by_side"]
        strength = float(trial.params["bayesian_strength"])
        ood_weight = float(trial.params["bayesian_ood_weight"])
        sizes = size_grid[(strength, ood_weight)]
        sizing = {"strength": strength, "ood_weight": ood_weight}
        row = {"seed": seed, "value": float(study.best_value), "trial": int(trial.number), "metrics": trial.user_attrs["metrics"], "trial_params": dict(trial.params)}
        seed_rows.append(row)
        if best is None or study.best_value > best[0]:
            best = (float(study.best_value), sizing, params, sizes, row)
    assert best is not None
    return best[1], best[2], best[3], {
        "best_value": best[0], "best": best[4], "seeds": seed_rows,
        "trials": int(trials * len(seeds)), "sizing_states": sizing_states,
        "capacity": {"mode": "gross_marked_quote_notional", "wallet_fraction": 0.80, "max_new_per_bar": 2, "count_cap": None},
        "holding_efficiency": {"power": 0.8, "weight": holding_weight},
    }


def _evaluate_arm(data, idx, params, sizes, ev, rank, holding_weight):
    outputs = data.simulate(idx, params, FAMILY_TRAILING_ONLY)
    _, metrics, detail = _subset_objective(
        data, idx, outputs, ev, rank, sizes, holding_weight=holding_weight,
    )
    selected = np.asarray(detail["selected"], dtype=bool)
    denom = max(int(selected.sum()), 1)
    for layer in range(3):
        activated = np.asarray(outputs["trailing_layer_first_bar"])[:, layer] >= 0
        binding = np.asarray(outputs["trailing_layer_binding_bars"])[:, layer] > 0
        exited = np.asarray(outputs["trailing_exit_layer"]) == layer
        metrics[f"layer_{layer + 1}_activation_rate"] = float(np.sum(selected & activated) / denom)
        metrics[f"layer_{layer + 1}_binding_rate"] = float(np.sum(selected & binding) / denom)
        metrics[f"layer_{layer + 1}_exit_rate"] = float(np.sum(selected & exited) / denom)
    return outputs, metrics, detail


def _local_robustness(
    data, idx, params, sizes, ev, rank, *, n: int, seed: int, holding_weight: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    records = []
    for perturbation in range(int(n)):
        candidate = copy.deepcopy(params)
        scales = {
            "sl": float(np.exp(rng.normal(0.0, 0.04))),
            "activation": float(np.exp(rng.normal(0.0, 0.04))),
            "beta": float(np.exp(rng.normal(0.0, 0.04))),
            "power": float(np.exp(rng.normal(0.0, 0.025))),
            "divisor": float(np.exp(rng.normal(0.0, 0.04))),
        }
        for side in ("long", "short"):
            values = candidate[side]
            values["sl_mult"] = float(np.clip(values["sl_mult"] * scales["sl"], 1.5, 8.0))
            for key in ("trailing_activation_mult", "trailing_activation_mult_2", "trailing_activation_mult_3"):
                values[key] = float(np.clip(values[key] * scales["activation"], 0.05, 8.0))
            for key in ("giveback_beta", "giveback_beta_2", "giveback_beta_3"):
                values[key] = float(np.clip(values[key] * scales["beta"], 0.05, 3.0))
            values["trailing_power"] = float(np.clip(values["trailing_power"] * scales["power"], 1.1, 4.0))
            values["trailing_squash_divisor"] = float(np.clip(values["trailing_squash_divisor"] * scales["divisor"], 1.0, 12.0))
        _, metrics, _ = _evaluate_arm(data, idx, candidate, sizes, ev, rank, holding_weight)
        records.append({"perturbation": perturbation, **scales, **metrics})
    frame = pd.DataFrame(records)
    summary = {
        "count": int(len(frame)),
        "objective_median": float(frame["objective"].median()),
        "objective_p10": float(frame["objective"].quantile(0.10)),
        "objective_min": float(frame["objective"].min()),
        "pnl_median": float(frame["net_pnl_bankroll"].median()),
        "pnl_p10": float(frame["net_pnl_bankroll"].quantile(0.10)),
        "positive_pnl_rate": float(frame["net_pnl_bankroll"].gt(0.0).mean()),
    }
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outer-trials", type=int, default=96)
    parser.add_argument("--final-trials", type=int, default=192)
    parser.add_argument("--search-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--holding-efficiency-weight", type=float, default=0.10)
    parser.add_argument("--local-perturbations", type=int, default=128)
    parser.add_argument("--monitor-quantile", type=float, choices=(0.75, 0.85), default=0.75)
    parser.add_argument("--spread-monitor-root", type=Path, default=Path("data_perp/exchanges/krakenfutures/spread_snapshots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    started = time.monotonic()

    deployed, _ = _load_deployed_side_params(PARENT)
    base_spec = ConstrainedReplaySpec()
    spec = replace(base_spec, fee_per_side=0.0)
    saved = json.loads(PARAMS.read_text())
    baseline_geometry = saved["fold_3"]["full_train_parent"]

    rows = pd.read_parquet(OLD_CANDIDATES)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="stable").reset_index(drop=True)
    rows, ev_join = join_candidate_execution_context(rows, pd.read_parquet(RICH_LEDGER))
    ev_audit = ev_join.to_dict()
    context = rows.loc[:, list(RAW_BAYESIAN_CONTEXT_COLUMNS)].copy()
    context_audit = {"source": str(RICH_LEDGER), "columns": list(RAW_BAYESIAN_CONTEXT_COLUMNS)}
    atr = _load_atr(rows, OLD_ATR)
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=STORE, cache_dir=OLD_CACHE, spec=base_spec, rebuild=False,
    )
    volume = np.memmap(VOLUME_CACHE, mode="r", dtype="float32", shape=(len(rows), base_spec.path_len))
    liquid = np.isfinite(volume[:, 0]) & (volume[:, 0] > 0.0)
    rows = rows.loc[liquid].reset_index(drop=True)
    context = context.loc[liquid].reset_index(drop=True)
    atr = np.asarray(atr)[liquid]
    open0, high, low, close, valid = [np.asarray(value)[liquid] for value in (open0, high, low, close, valid)]
    rows, spread_audit = _monitor_asset_spread_fill(rows, monitor_root=args.spread_monitor_root, quantile=args.monitor_quantile)
    rows = _cost_rows(rows, 1.5)
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)
    corrected_ev = pd.to_numeric(rows["threshold_basis_corrected_expected_ev"], errors="raise").to_numpy(float)
    corrected_rank = pd.to_numeric(rows["threshold_basis_corrected_expected_ev_rank"], errors="raise").to_numpy(float)
    data.rank = corrected_rank.copy()
    context, fallback = _fill_context(context, _indices_between(data, "2026-05-01", "2026-06-14"))

    seeds = [args.seed + 10000 * i for i in range(args.search_seeds)]
    fold_records, fold_params = [], {}
    for fold_no, fold in enumerate(FOLDS, 1):
        train = _indices_between(data, fold["train_start"], fold["train_end"])
        valid_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        sizing, params, sizes, optimizer = _optimize(
            data, train, context, corrected_ev, corrected_rank,
            saved[fold["fold"]]["full_train_parent"], trials=args.outer_trials,
            seeds=[value + 1000 * fold_no for value in seeds],
            holding_weight=args.holding_efficiency_weight,
        )
        _, winner_metrics, _ = _evaluate_arm(data, valid_idx, params, sizes, corrected_ev, corrected_rank, args.holding_efficiency_weight)
        baseline_train = data.simulate(train, saved[fold["fold"]]["full_train_parent"], FAMILY_TRAILING_ONLY)
        baseline_sizing = saved[fold["fold"]]["sizing"]
        baseline_sizes, _ = _bayesian_sizes(
            data, train, np.arange(len(rows)), baseline_train, context,
            strength=float(baseline_sizing["strength"]),
            ood_weight=float(baseline_sizing["ood_weight"]),
        )
        _, baseline_metrics, _ = _evaluate_arm(data, valid_idx, saved[fold["fold"]]["full_train_parent"], baseline_sizes, corrected_ev, corrected_rank, args.holding_efficiency_weight)
        fold_records += [
            {"fold": fold["fold"], "arm": "joint_layered_wallet80", **winner_metrics},
            {"fold": fold["fold"], "arm": "joint_trailing_raw_bayesian_wallet80_baseline", **baseline_metrics},
        ]
        fold_params[fold["fold"]] = {"sizing": sizing, "params_by_side": params, "optimizer": optimizer, "train": [fold["train_start"], fold["train_end"]], "validation": [fold["validation_start"], fold["validation_end"]]}
        pd.DataFrame(fold_records).to_csv(args.output_dir / "nested_oos_fold_metrics.partial.csv", index=False)
        (args.output_dir / "nested_params.partial.json").write_text(json.dumps(_json_safe(fold_params), indent=2))

    final_train = _indices_between(data, "2026-05-01", "2026-07-01")
    final_sizing, final_params, final_sizes, final_optimizer = _optimize(
        data, final_train, context, corrected_ev, corrected_rank, baseline_geometry,
        trials=args.final_trials, seeds=[value + 900000 for value in seeds],
        holding_weight=args.holding_efficiency_weight,
    )
    robustness_frame, robustness = _local_robustness(
        data, final_train, final_params, final_sizes, corrected_ev, corrected_rank,
        n=args.local_perturbations, seed=args.seed + 950000,
        holding_weight=args.holding_efficiency_weight,
    )
    report_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    winner_outputs, winner_metrics, winner_detail = _evaluate_arm(data, report_idx, final_params, final_sizes, corrected_ev, corrected_rank, args.holding_efficiency_weight)
    base_train_outputs = data.simulate(final_train, baseline_geometry, FAMILY_TRAILING_ONLY)
    deployed_sizing = saved["fold_3"]["sizing"]
    base_sizes, base_state = _bayesian_sizes(
        data, final_train, np.arange(len(rows)), base_train_outputs, context,
        strength=float(deployed_sizing["strength"]),
        ood_weight=float(deployed_sizing["ood_weight"]),
    )
    _, baseline_metrics, _ = _evaluate_arm(data, report_idx, baseline_geometry, base_sizes, corrected_ev, corrected_rank, args.holding_efficiency_weight)

    fold_frame = pd.DataFrame(fold_records)
    summary = fold_frame.groupby("arm", as_index=False).agg(
        folds=("fold", "size"), mean_objective=("objective", "mean"), worst_objective=("objective", "min"),
        mean_pnl=("net_pnl_bankroll", "mean"), worst_fold_pnl=("net_pnl_bankroll", "min"),
        worst_week=("worst_week", "min"), worst_drawdown=("max_drawdown", "min"),
        positive_folds=("net_pnl_bankroll", lambda x: int(np.sum(np.asarray(x) > 0))),
    )
    global_metrics = pd.DataFrame([
        {"arm": "joint_layered_wallet80", **winner_metrics},
        {"arm": "joint_trailing_raw_bayesian_wallet80_baseline", **baseline_metrics},
    ])
    for col in ("net_pnl_bankroll", "net_ev_per_trade", "worst_week", "max_drawdown"):
        base = float(global_metrics.loc[global_metrics["arm"].str.contains("baseline"), col].iloc[0])
        global_metrics[f"delta_{col}_vs_baseline"] = global_metrics[col] - base

    selected = np.flatnonzero(winner_detail["selected"])
    ledger = rows.iloc[report_idx].iloc[selected][[
        "timestamp", "symbol", "side_name", "policy_archetype",
        "threshold_basis_corrected_expected_ev", "threshold_basis_corrected_expected_ev_rank",
    ]].copy()
    ledger["exit_bars"] = winner_outputs["exit_bars"][selected]
    ledger["net_return"] = winner_outputs["net_return"][selected]
    ledger["admitted_notional"] = winner_detail["admitted_notional"][selected]
    ledger["pnl"] = winner_detail["pnl"][selected]
    for layer in range(3):
        ledger[f"layer_{layer + 1}_first_bar"] = winner_outputs["trailing_layer_first_bar"][selected, layer]
        ledger[f"layer_{layer + 1}_binding_bars"] = winner_outputs["trailing_layer_binding_bars"][selected, layer]
    ledger["trailing_exit_layer"] = winner_outputs["trailing_exit_layer"][selected] + 1

    fold_frame.to_csv(args.output_dir / "nested_oos_fold_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "nested_oos_summary.csv", index=False)
    global_metrics.to_csv(args.output_dir / "july_01_10_frozen_metrics.csv", index=False)
    ledger.to_parquet(args.output_dir / "july_01_10_selected_ledger.parquet", index=False)
    robustness_frame.to_csv(args.output_dir / "local_perturbation_metrics.csv", index=False)
    (args.output_dir / "winner_params.json").write_text(json.dumps(_json_safe({"sizing": final_sizing, "params_by_side": final_params, "optimizer": final_optimizer, "local_robustness": robustness, "baseline_sizing_state": base_state}), indent=2))
    manifest = {
        "status": "complete", "runtime_seconds": time.monotonic() - started,
        "candidate_source": str(OLD_CANDIDATES), "rows_before_liquidity": int(len(liquid)),
        "rows_after_positive_entry_volume": int(liquid.sum()), "ev_join": ev_audit,
        "context_audit": context_audit, "context_fallback": fallback,
        "spread_audit": spread_audit, "path_manifest": path_manifest,
        "cost": "zero fees plus 1.5x point-in-time spread", "l2": "supplementary post-freeze only",
        "optimization": "nested walk-forward, exact 1m prior-MFE, 1-3 ordered layers, actual EV/Bayesian sizes before 80% marked-notional admission",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2))
    print(global_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
