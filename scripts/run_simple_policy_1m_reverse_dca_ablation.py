#!/usr/bin/env python3
"""Nested walk-forward favorable-path reverse-DCA ablation on the winner."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_ablation import evaluate_results
from extreme_price_movements.simple_policy_1m_constrained import (
    ConstrainedReplaySpec,
    constrained_params_to_vector,
)
from extreme_price_movements.simple_policy_1m_contextual import stable_fold_objective
from extreme_price_movements.simple_policy_1m_reverse_dca import (
    EXIT_ANCHOR_INITIAL,
    EXIT_ANCHOR_WEIGHTED,
    SPACING_ABSOLUTE_FRACTION,
    SPACING_ATR_MULTIPLE,
    simulate_reverse_dca_1m_paths,
)
from scripts.report_simple_policy_1m_winner_forward_july import BASE, CHAMPION
from scripts.report_simple_policy_1m_winner_weekly import _json_safe
from scripts.run_simple_policy_1m_capital_ablation import (
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (
    INNER_FOLDS,
    ExperimentData,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (
    _bayesian_sizes,
    _load_atr,
    _load_context,
    _weighted_evaluate,
)


ABSOLUTE_GRID = np.asarray(
    [
        0.0005,
        0.000025,
        0.00005,
        0.000075,
        0.0001,
        0.0002,
        0.00035,
        0.00075,
        0.0010,
        0.0015,
        0.0020,
        0.0025,
        0.0035,
        0.0050,
        0.0070,
        0.0100,
        0.0140,
        0.0200,
        0.0280,
        0.0400,
        0.0550,
        0.0750,
    ],
    dtype=np.float64,
)
ATR_GRID = np.asarray(
    [
        0.025,
        0.001,
        0.002,
        0.003,
        0.005,
        0.010,
        0.015,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.30,
        0.45,
        0.60,
        0.80,
        1.00,
        1.30,
        1.70,
        2.20,
        3.00,
        4.00,
    ],
    dtype=np.float64,
)

MODES = (
    ("absolute_initial_anchor", SPACING_ABSOLUTE_FRACTION, EXIT_ANCHOR_INITIAL),
    ("absolute_weighted_anchor", SPACING_ABSOLUTE_FRACTION, EXIT_ANCHOR_WEIGHTED),
    ("atr_initial_anchor", SPACING_ATR_MULTIPLE, EXIT_ANCHOR_INITIAL),
    ("atr_weighted_anchor", SPACING_ATR_MULTIPLE, EXIT_ANCHOR_WEIGHTED),
)


def _empty(n: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    outputs = {
        "exit_bars": np.full(n, -1, dtype=np.int32),
        "exit_price": np.full(n, np.nan),
        "gross_return": np.full(n, np.nan),
        "net_return": np.full(n, np.nan),
        "reason": np.zeros(n, dtype=np.int8),
        "mfe": np.full(n, np.nan),
        "mae": np.full(n, np.nan),
    }
    diag = {
        "filled_fraction": np.full(n, np.nan),
        "additions": np.zeros(n, dtype=np.int16),
        "average_entry": np.full(n, np.nan),
        "last_level_distance": np.zeros(n),
        "first_add_bar": np.full(n, -1, dtype=np.int32),
        "full_target_bar": np.full(n, -1, dtype=np.int32),
        "geometry_mfe_atr": np.full(n, np.nan),
        "order_valid": np.zeros(n, dtype=bool),
    }
    return outputs, diag


def _simulate(
    data: ExperimentData,
    idx: np.ndarray,
    params_by_side: Mapping[str, Mapping[str, Any]],
    *,
    x: int,
    y: float,
    spacing_mode: int,
    anchor_mode: int,
    add_first: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    idx = np.asarray(idx, dtype=np.int64)
    outputs, diag = _empty(len(idx))
    keys = tuple(outputs) + tuple(diag)
    for side_name, sign in (("long", 1.0), ("short", -1.0)):
        local = np.flatnonzero(data.side[idx] * sign > 0.0)
        if not len(local):
            continue
        values = simulate_reverse_dca_1m_paths(
            idx[local],
            data.open0,
            data.high,
            data.low,
            data.close,
            data.side,
            data.atr_frac,
            data.entry_spread,
            data.exit_spread,
            constrained_params_to_vector(params_by_side[side_name]),
            data.spec.fee_per_side,
            data.spec.stop_base_gap_bps,
            data.spec.stop_through_fraction,
            data.spec.stop_max_gap_bps,
            int(x),
            float(y),
            int(spacing_mode),
            int(anchor_mode),
            bool(add_first),
        )
        for key, array in zip(keys, values):
            target = outputs[key] if key in outputs else diag[key]
            target[local] = array
    return outputs, diag


def _selection(rows: pd.DataFrame, outputs: Mapping[str, np.ndarray]) -> np.ndarray:
    _, selected = evaluate_results(
        rows.reset_index(drop=True),
        outputs["exit_bars"],
        outputs["gross_return"],
        outputs["net_return"],
        outputs["reason"],
        outputs["mfe"],
        outputs["mae"],
        bar_minutes=1,
        apply_capacity=True,
    )
    return selected


def _metrics(
    data: ExperimentData,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    diag: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    *,
    x: int,
    y: float,
    spacing_mode: int,
    anchor_mode: int,
) -> dict[str, Any]:
    result = _weighted_evaluate(data, idx, outputs, sizes)
    rows = data.rows.iloc[idx].reset_index(drop=True)
    selected = _selection(rows, outputs)
    chosen = np.flatnonzero(selected)
    filled = np.asarray(diag["filled_fraction"], dtype=np.float64)[chosen]
    additions = np.asarray(diag["additions"], dtype=np.int16)[chosen]
    rank = pd.to_numeric(rows.iloc[chosen]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(float)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    target = base * np.asarray(sizes, dtype=np.float64)[idx][chosen]
    actual = target * filled
    target_sum = float(target.sum())
    actual_sum = float(actual.sum())
    first_add = np.asarray(diag["first_add_bar"], dtype=np.int32)[chosen]
    result.update(
        {
            "x_total_tranches": int(x),
            "y": float(y),
            "spacing_mode": "absolute_fraction" if spacing_mode == SPACING_ABSOLUTE_FRACTION else "atr_multiple",
            "exit_anchor": "initial_entry" if anchor_mode == EXIT_ANCHOR_INITIAL else "weighted_entry",
            "initial_target_fraction": float(1.0 / max(x, 1)),
            "maximum_target_fraction": 1.0,
            "target_exposure": target_sum,
            "actual_exposure": actual_sum,
            "actual_to_target_exposure": float(actual_sum / max(target_sum, 1e-12)),
            "mean_filled_fraction": float(np.average(filled, weights=target)) if len(chosen) else 0.0,
            "add_trigger_rate": float(np.mean(additions > 0)) if len(chosen) else 0.0,
            "mean_additions": float(np.mean(additions)) if len(chosen) else 0.0,
            "p90_additions": float(np.quantile(additions, 0.9)) if len(chosen) else 0.0,
            "max_additions": int(additions.max()) if len(chosen) else 0,
            "full_target_fill_rate": float(np.mean(filled >= 1.0 - 1e-12)) if len(chosen) else 0.0,
            "mean_first_add_minutes": float(np.mean(first_add[first_add >= 0] + 1)) if np.any(first_add >= 0) else np.nan,
            "exposure_normalized_pnl_diagnostic": float(result["net_pnl_bankroll"] / max(actual_sum / max(target_sum, 1e-12), 1e-12)),
        }
    )
    return result


def _grid(spacing_mode: int) -> list[tuple[int, float]]:
    values = ABSOLUTE_GRID if spacing_mode == SPACING_ABSOLUTE_FRACTION else ATR_GRID
    x_values = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24)
    return [(1, 0.0)] + [(x, float(y)) for x in x_values for y in values]


def _search_mode(
    data: ExperimentData,
    idx: np.ndarray,
    sizes: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
    *,
    fold: str,
    mode_name: str,
    spacing_mode: int,
    anchor_mode: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for x, y in _grid(spacing_mode):
        outputs, diag = _simulate(
            data,
            idx,
            params,
            x=x,
            y=y,
            spacing_mode=spacing_mode,
            anchor_mode=anchor_mode,
        )
        metrics = _metrics(
            data,
            idx,
            outputs,
            diag,
            sizes,
            x=x,
            y=y,
            spacing_mode=spacing_mode,
            anchor_mode=anchor_mode,
        )
        records.append({"fold": fold, "mode": mode_name, "stage": "inner_search", **metrics})
    frame = pd.DataFrame(records)
    ordered = frame.sort_values(
        ["objective", "worst_week", "max_drawdown", "actual_to_target_exposure"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return ordered.iloc[0].to_dict(), frame


def _execution_aware_choice(frame: pd.DataFrame, spacing_mode: int) -> dict[str, Any]:
    floor = 0.0015 if spacing_mode == SPACING_ABSOLUTE_FRACTION else 0.05
    eligible = frame.loc[
        frame["x_total_tranches"].le(8)
        & (
            frame["x_total_tranches"].eq(1)
            | frame["y"].ge(floor)
        )
    ]
    ordered = eligible.sort_values(
        ["objective", "worst_week", "max_drawdown", "actual_to_target_exposure"],
        ascending=[False, False, False, False],
        kind="mergesort",
    )
    return ordered.iloc[0].to_dict()


def _identity_check(
    data: ExperimentData,
    idx: np.ndarray,
    params: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    baseline = data.simulate(idx, params, 0)
    reverse, _ = _simulate(
        data,
        idx,
        params,
        x=1,
        y=0.0,
        spacing_mode=SPACING_ABSOLUTE_FRACTION,
        anchor_mode=EXIT_ANCHOR_INITIAL,
        add_first=True,
    )
    checks = {}
    for key in ("exit_bars", "exit_price", "gross_return", "net_return", "reason"):
        a = np.asarray(baseline[key])
        b = np.asarray(reverse[key])
        checks[key] = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
    if checks["exit_bars"] != 0.0 or checks["reason"] != 0.0 or checks["net_return"] > 1e-10:
        raise RuntimeError(f"x=1 reverse-DCA identity failed: {checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "reverse_dca_favorable_ablation_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    params = json.loads((CHAMPION / "evidence/nested_params.json").read_text())
    rows = pd.read_parquet(candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(
        ["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort"
    ).reset_index(drop=True)
    context, _, context_audit = _load_context(rows, rich, posterior)
    atr = _load_atr(rows, CHAMPION / "replay/causal_entry_atr_audit.parquet")
    deployed, _ = _load_deployed_side_params(parent_summary)
    spec = ConstrainedReplaySpec()
    open0, high, low, close, valid, path_manifest = _load_or_build_path_cache(
        rows,
        store_root=Path("data_perp/exchanges/krakenfutures/execution_1m"),
        cache_dir=CHAMPION / "replay/path_cache",
        spec=spec,
        rebuild=False,
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)

    fold_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    search_frames: list[pd.DataFrame] = []
    choices: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    for fold in FOLDS:
        name = fold["fold"]
        inner = INNER_FOLDS[name]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        search_params = params[name]["search_parent"]
        outer_params = params[name]["full_train_parent"]
        sizing = params[name]["sizing"]
        fit_search = data.simulate(search_idx, search_params, 0)
        fit_outer = data.simulate(train_idx, outer_params, 0)
        sizes_inner, _ = _bayesian_sizes(
            data,
            search_idx,
            inner_idx,
            fit_search,
            context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )
        sizes_outer, sizing_state = _bayesian_sizes(
            data,
            train_idx,
            outer_idx,
            fit_outer,
            context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )
        identity[name] = _identity_check(data, outer_idx[: min(512, len(outer_idx))], outer_params)
        choices[name] = {}
        for mode_name, spacing_mode, anchor_mode in MODES:
            best, trials = _search_mode(
                data,
                inner_idx,
                sizes_inner,
                search_params,
                fold=name,
                mode_name=mode_name,
                spacing_mode=spacing_mode,
                anchor_mode=anchor_mode,
            )
            search_frames.append(trials)
            profile_choices = {
                "unrestricted_diagnostic": best,
                "execution_aware": _execution_aware_choice(trials, spacing_mode),
            }
            for profile, selected_choice in profile_choices.items():
                policy_name = f"{mode_name}__{profile}"
                x = int(selected_choice["x_total_tranches"])
                y = float(selected_choice["y"])
                outputs, diag = _simulate(
                    data,
                    outer_idx,
                    outer_params,
                    x=x,
                    y=y,
                    spacing_mode=spacing_mode,
                    anchor_mode=anchor_mode,
                )
                metrics = _metrics(
                    data,
                    outer_idx,
                    outputs,
                    diag,
                    sizes_outer,
                    x=x,
                    y=y,
                    spacing_mode=spacing_mode,
                    anchor_mode=anchor_mode,
                )
                fold_rows.append({"fold": name, "policy": policy_name, "search_profile": profile, **metrics})
                bound_outputs, bound_diag = _simulate(
                    data,
                    outer_idx,
                    outer_params,
                    x=x,
                    y=y,
                    spacing_mode=spacing_mode,
                    anchor_mode=anchor_mode,
                    add_first=False,
                )
                bound_metrics = _metrics(
                    data,
                    outer_idx,
                    bound_outputs,
                    bound_diag,
                    sizes_outer,
                    x=x,
                    y=y,
                    spacing_mode=spacing_mode,
                    anchor_mode=anchor_mode,
                )
                sensitivity_rows.extend(
                    [
                        {"fold": name, "policy": policy_name, "collision_order": "add_first_primary", **metrics},
                        {"fold": name, "policy": policy_name, "collision_order": "exit_first_bound", **bound_metrics},
                    ]
                )
                choices[name][policy_name] = {
                    "x_total_tranches": x,
                    "y": y,
                    "inner_objective": float(selected_choice["objective"]),
                    "inner_net_pnl": float(selected_choice["net_pnl_bankroll"]),
                    "sizing_state": sizing_state,
                }
                print(
                    f"{name} {policy_name}: x={x} y={y:.6g} "
                    f"inner={selected_choice['objective']:.6f} outer={metrics['objective']:.6f}",
                    flush=True,
                )

        base_outputs, base_diag = _simulate(
            data,
            outer_idx,
            outer_params,
            x=1,
            y=0.0,
            spacing_mode=SPACING_ABSOLUTE_FRACTION,
            anchor_mode=EXIT_ANCHOR_INITIAL,
        )
        base_metrics = _metrics(
            data,
            outer_idx,
            base_outputs,
            base_diag,
            sizes_outer,
            x=1,
            y=0.0,
            spacing_mode=SPACING_ABSOLUTE_FRACTION,
            anchor_mode=EXIT_ANCHOR_INITIAL,
        )
        fold_rows.append({"fold": name, "policy": "winner_baseline", **base_metrics})

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(
        args.output_dir / "collision_sensitivity.csv", index=False
    )
    search_trials = pd.concat(search_frames, ignore_index=True)
    search_trials.to_parquet(args.output_dir / "inner_search_grid.parquet", index=False)
    summary = (
        fold_metrics.groupby("policy", sort=False)
        .agg(
            folds=("fold", "count"),
            total_trades=("n_trades", "sum"),
            mean_trades_per_day=("trades_per_day", "mean"),
            total_gross_pnl=("gross_pnl_bankroll", "sum"),
            total_fee_pnl=("fee_pnl_bankroll", "sum"),
            total_net_pnl=("net_pnl_bankroll", "sum"),
            mean_net_pnl=("net_pnl_bankroll", "mean"),
            worst_fold_pnl=("net_pnl_bankroll", "min"),
            worst_week=("worst_week", "min"),
            worst_drawdown=("max_drawdown", "min"),
            mean_net_return=("mean_net_return", "mean"),
            mean_hit_rate=("hit_rate", "mean"),
            mean_full_sl_rate=("full_sl_rate", "mean"),
            mean_adverse_exit_rate=("adverse_exit_rate", "mean"),
            mean_trailing_rate=("trailing_rate", "mean"),
            mean_timeout_rate=("timeout_rate", "mean"),
            mean_holding_hours=("mean_holding_hours", "mean"),
            mean_exposure_ratio=("actual_to_target_exposure", "mean"),
            mean_add_trigger_rate=("add_trigger_rate", "mean"),
            mean_additions=("mean_additions", "mean"),
            mean_full_target_fill_rate=("full_target_fill_rate", "mean"),
        )
        .reset_index()
    )
    baseline = summary.loc[summary["policy"].eq("winner_baseline")].iloc[0]
    summary["delta_total_net_pnl_vs_winner"] = summary["total_net_pnl"] - float(
        baseline["total_net_pnl"]
    )
    summary["delta_worst_fold_vs_winner"] = summary["worst_fold_pnl"] - float(
        baseline["worst_fold_pnl"]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)

    manifest = {
        "status": "complete",
        "experiment": "exposure-neutral favorable-path reverse-DCA",
        "winner_basis": "joint_trailing_total_mfe_raw_bayesian_v1",
        "evidence": "nested walk-forward policy-validation OOS",
        "search_contract": {
            "x_total_tranches": [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24],
            "absolute_fraction_grid": ABSOLUTE_GRID.tolist(),
            "atr_multiple_grid": ATR_GRID.tolist(),
            "modes": [name for name, _, _ in MODES],
            "selection": "highest inner stability objective; worst-week/drawdown/exposure tie-breaks",
            "execution_aware_profile": {
                "maximum_total_tranches": 8,
                "minimum_absolute_spacing_fraction": 0.0015,
                "minimum_atr_spacing_multiple": 0.05,
                "rationale": "approximately one spread or more; bounded order count",
            },
        },
        "entry_contract": "x equal tranches; initial target/x; x-1 favorable adds; maximum original target exposure 1.0",
        "trigger_contract": "successive levels from initial raw entry, either k*y percent or k*y*entry-frozen causal ATR",
        "exit_anchor_contract": {
            "initial_entry": "original winner exit path and total-MFE geometry remain anchored to initial executable entry",
            "weighted_entry": "catastrophic and total-MFE trailing geometry re-anchor to current weighted executable entry immediately in the add-first primary and from the next bar in the exit-first sensitivity; stops may never loosen and the fast adverse guard remains initial-entry frozen",
        },
        "collision_contract": "add-first primary assumes every touched favorable tranche fills before a same-minute exit; exit-first is reported as a sensitivity bound. For weighted anchors add-first can also tighten the stop, so it is an exposure-pessimistic ordering rather than a guaranteed PnL lower bound",
        "cost_contract": "each filled tranche pays entry half-spread and 0.5% fee; filled quantity pays executable exit spread and 0.5% fee; 1% round trip applied once",
        "capacity_contract": "same size-independent 8-open/2-new admission; full raw-Bayesian target reserved at initial entry",
        "folds": FOLDS,
        "inner_folds": INNER_FOLDS,
        "choices": choices,
        "identity_check": identity,
        "elapsed_seconds": time.monotonic() - started,
        "candidate_rows": int(len(rows)),
        "valid_path_rows": int(valid.sum()),
        "context_audit": context_audit,
        "path_manifest": path_manifest,
        "outputs": ["summary.csv", "fold_metrics.csv", "inner_search_grid.parquet", "collision_sensitivity.csv"],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
