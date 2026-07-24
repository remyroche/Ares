#!/usr/bin/env python3
"""Jointly tune exposure-neutral DCA and winner trailing activation geometry."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import pandas as pd

from extreme_price_movements.simple_policy_1m_constrained import FAMILY_TRAILING_ONLY, ConstrainedReplaySpec
from extreme_price_movements.simple_policy_optimiser import _with_policy_spread_cost_columns
from scripts.report_simple_policy_1m_winner_forward_july import BASE, CHAMPION, FORWARD_SOURCE, _forward_context
from scripts.report_simple_policy_1m_winner_weekly import _json_safe
from scripts.run_simple_policy_1m_capital_ablation import _load_deployed_side_params, _load_or_build_path_cache
from scripts.run_simple_policy_1m_constrained_search import FOLDS, INNER_FOLDS, ExperimentData, _indices_between
from scripts.run_simple_policy_1m_contextual_ablation import _bayesian_sizes, _load_atr, _load_context
from scripts.run_simple_policy_1m_dca_ablation import (
    _apply_dca,
    _combine_outputs,
    _metric,
    _weekly_ledger,
)


PARAM_BOUNDS = {
    "trailing_activation_mult": (0.45, 4.5),
    "trailing_power": (0.8, 3.2),
    "trailing_squash_divisor": (1.0, 7.0),
    "giveback_beta": (0.15, 0.95),
}


def _adjust_geometry(
    base: Mapping[str, Mapping[str, Any]], factors: Mapping[str, float]
) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(base)
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        values = result[side_name]
        act = (
            float(values["trailing_activation_mult"])
            * float(factors["activation_scale"])
            * math.exp(sign * float(factors["activation_side_tilt"]))
        )
        values["trailing_activation_mult"] = float(np.clip(act, *PARAM_BOUNDS["trailing_activation_mult"]))
        for key, factor_key in (
            ("trailing_power", "power_scale"),
            ("trailing_squash_divisor", "squash_scale"),
            ("giveback_beta", "giveback_scale"),
        ):
            values[key] = float(
                np.clip(float(values[key]) * float(factors[factor_key]), *PARAM_BOUNDS[key])
            )
    return result


def _suggest(trial: optuna.Trial) -> tuple[int, float, dict[str, float]]:
    x = trial.suggest_categorical("x_total_tranches", [1, 2, 3, 4, 5, 6])
    y = 0.0 if x == 1 else trial.suggest_float("dca_spacing_fraction", 0.0005, 0.03, log=True)
    factors = {
        "activation_scale": trial.suggest_float("activation_scale", 0.65, 1.55, log=True),
        "activation_side_tilt": trial.suggest_float("activation_side_tilt", -0.18, 0.18),
        "power_scale": trial.suggest_float("power_scale", 0.75, 1.35, log=True),
        "squash_scale": trial.suggest_float("squash_scale", 0.70, 1.40, log=True),
        "giveback_scale": trial.suggest_float("giveback_scale", 0.72, 1.35, log=True),
    }
    return int(x), float(y), factors


def _run_search(
    data: ExperimentData,
    idx: np.ndarray,
    sizes: np.ndarray,
    base_params: Mapping[str, Mapping[str, Any]],
    *,
    seeds: list[int],
    trials_per_seed: int,
    fold: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for seed in seeds:
        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True, group=True,
            n_startup_trials=min(24, max(12, trials_per_seed // 3)),
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.enqueue_trial(
            {
                "x_total_tranches": 1,
                "activation_scale": 1.0,
                "activation_side_tilt": 0.0,
                "power_scale": 1.0,
                "squash_scale": 1.0,
                "giveback_scale": 1.0,
            }
        )

        def objective(trial: optuna.Trial) -> float:
            x, y, factors = _suggest(trial)
            params = _adjust_geometry(base_params, factors)
            exits = data.simulate(idx, params, FAMILY_TRAILING_ONLY)
            outputs, diag = _apply_dca(
                data, idx, exits, x=x, y=y, literal=False, dca_first=False
            )
            metrics = _metric(data, idx, outputs, sizes, diag, x=x, y=y, literal=False)
            payload = {
                "fold": fold,
                "seed": seed,
                "trial": trial.number,
                "x": x,
                "y_fraction": y,
                **factors,
                **metrics,
            }
            records.append(payload)
            trial.set_user_attr("payload", payload)
            trial.set_user_attr("params_by_side", params)
            return float(metrics["objective"])

        study.optimize(objective, n_trials=trials_per_seed, show_progress_bar=False)
        candidate = {
            "objective": float(study.best_value),
            "payload": study.best_trial.user_attrs["payload"],
            "params_by_side": study.best_trial.user_attrs["params_by_side"],
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    if best is None:
        raise RuntimeError("Joint DCA/exit search produced no trial")
    return best, pd.DataFrame(records)


def _evaluate_arm(
    data: ExperimentData,
    idx: np.ndarray,
    sizes: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    *,
    x: int,
    y: float,
    dca_first: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    exits = data.simulate(idx, params, FAMILY_TRAILING_ONLY)
    outputs, diag = _apply_dca(
        data, idx, exits, x=x, y=y, literal=False, dca_first=dca_first
    )
    metrics = _metric(data, idx, outputs, sizes, diag, x=x, y=y, literal=False)
    return metrics, outputs, diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials-per-seed", type=int, default=64)
    parser.add_argument("--search-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-dir", type=Path, default=CHAMPION / "joint_dca_exit_activation_v1")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    params = json.loads((CHAMPION / "evidence/nested_params.json").read_text())
    rows = pd.read_parquet(candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, context_audit = _load_context(rows, rich, posterior)
    atr = _load_atr(rows, CHAMPION / "replay/causal_entry_atr_audit.parquet")
    deployed, _ = _load_deployed_side_params(parent_summary)
    spec = ConstrainedReplaySpec()
    store_root = Path("data_perp/exchanges/krakenfutures/execution_1m")
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows, store_root=store_root, cache_dir=CHAMPION / "replay/path_cache", spec=spec, rebuild=False
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)

    fold_records: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    choices: dict[str, Any] = {}
    seeds = [int(args.seed + 10000 * i) for i in range(args.search_seeds)]
    for fold_no, fold in enumerate(FOLDS, start=1):
        name = fold["fold"]
        inner = INNER_FOLDS[name]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        search_base = params[name]["search_parent"]
        full_base = params[name]["full_train_parent"]
        sizing = params[name]["sizing"]
        frozen_search = data.simulate(search_idx, search_base, FAMILY_TRAILING_ONLY)
        frozen_train = data.simulate(train_idx, full_base, FAMILY_TRAILING_ONLY)
        sizes_inner, _ = _bayesian_sizes(
            data, search_idx, inner_idx, frozen_search, context,
            strength=float(sizing["strength"]), ood_weight=float(sizing["ood_weight"]),
        )
        sizes_outer, sizing_state = _bayesian_sizes(
            data, train_idx, outer_idx, frozen_train, context,
            strength=float(sizing["strength"]), ood_weight=float(sizing["ood_weight"]),
        )
        best, trials = _run_search(
            data, inner_idx, sizes_inner, search_base,
            seeds=[s + fold_no * 101 for s in seeds], trials_per_seed=args.trials_per_seed, fold=name,
        )
        trial_frames.append(trials)
        chosen = best["payload"]
        factors = {key: float(chosen[key]) for key in (
            "activation_scale", "activation_side_tilt", "power_scale", "squash_scale", "giveback_scale"
        )}
        x, y = int(chosen["x"]), float(chosen["y_fraction"])
        outer_params = _adjust_geometry(full_base, factors)
        base_metrics, _, _ = _evaluate_arm(data, outer_idx, sizes_outer, full_base, x=1, y=0.0)
        joint_metrics, _, _ = _evaluate_arm(data, outer_idx, sizes_outer, outer_params, x=x, y=y)
        bound_metrics, _, _ = _evaluate_arm(
            data, outer_idx, sizes_outer, outer_params, x=x, y=y, dca_first=True
        )
        fold_records.extend([
            {"fold": name, "policy": "winner_baseline", **base_metrics},
            {"fold": name, "policy": "joint_dca_exit_activation_exit_first", **joint_metrics},
            {"fold": name, "policy": "joint_dca_exit_activation_dca_first_bound", **bound_metrics},
        ])
        choices[name] = {
            "x": x, "y_fraction": y, "factors": factors,
            "inner_objective": float(best["objective"]),
            "inner_trial": {"seed": int(chosen["seed"]), "trial": int(chosen["trial"])},
            "outer_params_by_side": outer_params,
            "sizing_state": sizing_state,
        }
        print(f"{name}: x={x} y={100*y:.4f}% inner={best['objective']:.6f} outer={joint_metrics['objective']:.6f}", flush=True)

    fold_metrics = pd.DataFrame(fold_records)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(trial_frames, ignore_index=True).to_parquet(args.output_dir / "search_trials.parquet", index=False)
    summary = fold_metrics.groupby("policy", sort=False).agg(
        folds=("fold", "count"), total_net_pnl=("net_pnl_bankroll", "sum"),
        mean_net_pnl=("net_pnl_bankroll", "mean"), worst_fold_pnl=("net_pnl_bankroll", "min"),
        worst_week=("worst_week", "min"), worst_drawdown=("max_drawdown", "min"),
        mean_hit_rate=("hit_rate", "mean"), mean_holding_hours=("mean_holding_hours", "mean"),
        mean_exposure=("actual_to_target_exposure", "mean"), mean_dca_trigger=("dca_trigger_rate", "mean"),
    ).reset_index()
    summary.to_csv(args.output_dir / "fold_summary.csv", index=False)

    manifest = {
        "status": "complete",
        "experiment": "joint exposure-neutral DCA and exit activation optimization",
        "winner_basis": "joint_trailing_total_mfe_raw_bayesian_v1",
        "evidence": "nested walk-forward policy-validation OOS",
        "search": {
            "trials_per_seed": args.trials_per_seed, "seeds": seeds,
            "total_trials_per_fold": args.trials_per_seed * args.search_seeds,
            "x_total_tranches": [1, 6], "y_fraction": [0.0005, 0.03],
            "parameter_bounds": PARAM_BOUNDS,
            "factors": {
                "activation_scale": [0.65, 1.55], "activation_side_tilt": [-0.18, 0.18],
                "power_scale": [0.75, 1.35], "squash_scale": [0.70, 1.40],
                "giveback_scale": [0.72, 1.35],
            },
        },
        "dca_contract": "x total equal tranches; initial target/x and x-1 adverse adds; maximum target exposure 1.0",
        "exit_contract": "total-MFE trailing family; catastrophic and frozen adverse guard retained; activation/power/squash/giveback jointly adjusted around winner",
        "collision_contract": "exit-first primary; same configuration replayed DCA-first as optimistic OHLC bound",
        "fit_contract": "adjustment factors selected on inner window around search-parent geometry, then transferred to full-train parent for one outer evaluation",
        "cost_contract": "1% round-trip fee plus side-correct entry/exit spread once; every filled tranche charged",
        "capacity_contract": "same size-independent 8-open/2-new admission; full target reserved at initial entry",
        "folds": FOLDS, "inner_folds": INNER_FOLDS, "choices": choices,
        "context_audit": context_audit, "path_manifest": path_manifest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
