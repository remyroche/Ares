#!/usr/bin/env python3
"""Walk-forward DCA ablation on the frozen joint-trailing/raw-Bayesian winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_1m_ablation import evaluate_results  # noqa: E402
from extreme_price_movements.simple_policy_1m_constrained import (  # noqa: E402
    FAMILY_TRAILING_ONLY,
    ConstrainedReplaySpec,
)
from extreme_price_movements.simple_policy_1m_contextual import (  # noqa: E402
    stable_fold_objective,
)
from extreme_price_movements.simple_policy_1m_dca import (  # noqa: E402
    apply_dca_to_frozen_exits,
)
from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _with_policy_spread_cost_columns,
)
from scripts.report_simple_policy_1m_winner_forward_july import (  # noqa: E402
    BASE,
    CHAMPION,
    FORWARD_SOURCE,
    _forward_context,
)
from scripts.report_simple_policy_1m_winner_weekly import _json_safe  # noqa: E402
from scripts.run_simple_policy_1m_capital_ablation import (  # noqa: E402
    FOLDS,
    _load_deployed_side_params,
    _load_or_build_path_cache,
)
from scripts.run_simple_policy_1m_constrained_search import (  # noqa: E402
    INNER_FOLDS,
    ExperimentData,
    _indices_between,
)
from scripts.run_simple_policy_1m_contextual_ablation import (  # noqa: E402
    _bayesian_sizes,
    _load_atr,
    _load_context,
    _weighted_evaluate,
)


Y_GRID = np.asarray(
    [
        0.0005,
        0.0010,
        0.0015,
        0.0020,
        0.0025,
        0.0035,
        0.0050,
        0.0070,
        0.0100,
        0.0125,
        0.0150,
        0.0200,
        0.0250,
        0.0300,
        0.0400,
        0.0500,
    ],
    dtype=np.float64,
)
PRIMARY_GRID = [(1, 0.0)] + [
    (x, float(y)) for x in range(2, 9) for y in Y_GRID
]
LITERAL_GRID = [(x, float(y)) for x in range(2, 9) for y in Y_GRID]


def _copy_outputs(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in outputs.items()}


def _combine_outputs(parts: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([np.asarray(part[key]) for part in parts]) for key in parts[0]}


def _apply_dca(
    data: ExperimentData,
    idx: np.ndarray,
    frozen: Mapping[str, np.ndarray],
    *,
    x: int,
    y: float,
    literal: bool,
    dca_first: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = apply_dca_to_frozen_exits(
        np.asarray(idx, dtype=np.int64),
        data.open0,
        data.high,
        data.low,
        data.side,
        data.entry_spread,
        np.asarray(frozen["exit_bars"], dtype=np.int32),
        np.asarray(frozen["exit_price"], dtype=np.float64),
        np.asarray(frozen["reason"], dtype=np.int8),
        data.spec.fee_per_side,
        int(x),
        float(y),
        bool(literal),
        bool(dca_first),
    )
    out = _copy_outputs(frozen)
    out["gross_return"] = values[0]
    out["net_return"] = values[1]
    diag = {
        "filled_fraction": values[2],
        "additions": values[3],
        "average_entry": values[4],
        "last_level_fraction": values[5],
        "raw_adverse_before_exit": values[6],
        "raw_adverse_including_exit": values[7],
    }
    return out, diag


def _selection(
    rows: pd.DataFrame, outputs: Mapping[str, np.ndarray]
) -> np.ndarray:
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


def _metric(
    data: Any,
    idx: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    diag: Mapping[str, np.ndarray],
    *,
    x: int,
    y: float,
    literal: bool,
) -> dict[str, Any]:
    metrics = _weighted_evaluate(data, idx, outputs, sizes)
    rows = data.rows.iloc[idx].reset_index(drop=True)
    selected = _selection(rows, outputs)
    chosen = np.flatnonzero(selected)
    filled = np.asarray(diag["filled_fraction"], dtype=np.float64)[chosen]
    additions = np.asarray(diag["additions"], dtype=np.float64)[chosen]
    rank = pd.to_numeric(rows.iloc[chosen]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    mult = np.asarray(sizes, dtype=np.float64)[idx][chosen]
    target = base * mult
    actual = target * filled
    target_sum = float(target.sum())
    actual_sum = float(actual.sum())
    raw_pre = np.asarray(diag["raw_adverse_before_exit"], dtype=np.float64)[chosen]
    raw_including = np.asarray(
        diag["raw_adverse_including_exit"], dtype=np.float64
    )[chosen]
    max_fraction = (x + 1.0) / max(x, 1) if literal else 1.0
    metrics.update(
        {
            "x": int(x),
            "y_fraction": float(y),
            "y_pct": float(100.0 * y),
            "literal_additional_dcas": bool(literal),
            "initial_target_fraction": float(1.0 / max(x, 1)),
            "maximum_target_fraction": float(max_fraction),
            "mean_filled_fraction": float(np.average(filled, weights=target)) if len(chosen) else 0.0,
            "actual_entry_exposure": actual_sum,
            "target_entry_exposure": target_sum,
            "actual_to_target_exposure": float(actual_sum / max(target_sum, 1e-12)),
            "dca_trigger_rate": float(np.mean(additions > 0.0)) if len(chosen) else 0.0,
            "mean_additions": float(np.mean(additions)) if len(chosen) else 0.0,
            "p90_additions": float(np.quantile(additions, 0.9)) if len(chosen) else 0.0,
            "max_additions_observed": int(np.max(additions)) if len(chosen) else 0,
            "full_target_fill_rate": float(np.mean(filled >= 1.0 - 1e-12)) if len(chosen) else 0.0,
            "dca_exposure_normalized_pnl": float(metrics["net_pnl_bankroll"] / max(actual_sum / max(target_sum, 1e-12), 1e-12)),
            "raw_mae_before_exit_mean": float(np.mean(raw_pre)) if len(chosen) else 0.0,
            "raw_mae_before_exit_median": float(np.median(raw_pre)) if len(chosen) else 0.0,
            "raw_mae_including_exit_mean": float(np.mean(raw_including)) if len(chosen) else 0.0,
            "raw_mae_including_exit_median": float(np.median(raw_including)) if len(chosen) else 0.0,
            "raw_adverse_025_before_exit_rate": float(np.mean(raw_pre >= 0.0025)) if len(chosen) else 0.0,
            "raw_adverse_025_including_exit_rate": float(np.mean(raw_including >= 0.0025)) if len(chosen) else 0.0,
        }
    )
    return metrics


def _baseline_diag(n: int) -> dict[str, np.ndarray]:
    return {
        "filled_fraction": np.ones(n, dtype=np.float64),
        "additions": np.zeros(n, dtype=np.int16),
        "average_entry": np.full(n, np.nan),
        "last_level_fraction": np.zeros(n, dtype=np.float64),
        "raw_adverse_before_exit": np.zeros(n, dtype=np.float64),
        "raw_adverse_including_exit": np.zeros(n, dtype=np.float64),
    }


def _loss_reconciliation(
    rows: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    diag: Mapping[str, np.ndarray],
    *,
    period: str,
) -> list[dict[str, Any]]:
    selected = _selection(rows.reset_index(drop=True), outputs)
    chosen = np.flatnonzero(selected)
    net = np.asarray(outputs["net_return"], dtype=np.float64)[chosen]
    reasons = np.asarray(outputs["reason"], dtype=np.int8)[chosen]
    pre = np.asarray(diag["raw_adverse_before_exit"], dtype=np.float64)[chosen]
    including = np.asarray(
        diag["raw_adverse_including_exit"], dtype=np.float64
    )[chosen]
    reason_names = {0: "timeout", 1: "full_sl", 2: "capital", 3: "trailing", 4: "adverse"}
    records: list[dict[str, Any]] = []
    for result_name, result_mask in (
        ("all", np.ones(len(chosen), dtype=bool)),
        ("net_win", net > 0.0),
        ("net_loss", net <= 0.0),
    ):
        for reason_code in (-1, 0, 1, 2, 3, 4):
            mask = result_mask if reason_code < 0 else result_mask & (reasons == reason_code)
            count = int(mask.sum())
            records.append(
                {
                    "period": period,
                    "net_result": result_name,
                    "exit_reason": "all" if reason_code < 0 else reason_names[reason_code],
                    "trades": count,
                    "share_of_period_trades": float(count / max(len(chosen), 1)),
                    "raw_adverse_025_before_exit_rate": float(np.mean(pre[mask] >= 0.0025)) if count else 0.0,
                    "raw_adverse_025_including_exit_rate": float(np.mean(including[mask] >= 0.0025)) if count else 0.0,
                    "raw_adverse_any_before_exit_rate": float(np.mean(pre[mask] > 0.0)) if count else 0.0,
                    "raw_adverse_any_including_exit_rate": float(np.mean(including[mask] > 0.0)) if count else 0.0,
                }
            )
    return records


def _search_grid(
    data: ExperimentData,
    idx: np.ndarray,
    frozen: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    grid: list[tuple[int, float]],
    *,
    fold: str,
    literal: bool,
    dca_first: bool = False,
) -> tuple[tuple[int, float], pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for x, y in grid:
        outputs, diag = _apply_dca(
            data, idx, frozen, x=x, y=y, literal=literal, dca_first=dca_first
        )
        metrics = _metric(data, idx, outputs, sizes, diag, x=x, y=y, literal=literal)
        records.append({"fold": fold, "stage": "inner_search", **metrics})
    frame = pd.DataFrame(records)
    best = frame.sort_values(
        ["objective", "worst_week", "net_pnl_bankroll", "x", "y_fraction"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).iloc[0]
    return (int(best["x"]), float(best["y_fraction"])), frame


def _local_grid(x: int, y: float, *, literal: bool) -> list[tuple[int, float]]:
    xs = sorted(set(max(2 if literal else 1, min(8, x + delta)) for delta in (-1, 0, 1)))
    if x == 1 and not literal:
        return [(1, 0.0)] + [(2, float(v)) for v in Y_GRID[:6]]
    ys = sorted(
        set(float(np.clip(y * factor, 0.00025, 0.0600)) for factor in (0.8, 0.9, 1.0, 1.1, 1.2))
    )
    return [(x0, y0) for x0 in xs for y0 in ys]


def _weekly_ledger(
    rows: pd.DataFrame,
    outputs: Mapping[str, np.ndarray],
    sizes: np.ndarray,
    diag: Mapping[str, np.ndarray],
    *,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = _selection(rows, outputs)
    chosen = np.flatnonzero(selected)
    local = rows.iloc[chosen].reset_index(drop=True).copy()
    rank = pd.to_numeric(local["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    base = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    mult = np.asarray(sizes, dtype=np.float64)[chosen]
    filled = np.asarray(diag["filled_fraction"], dtype=np.float64)[chosen]
    local["policy"] = policy
    local["exit_bars"] = np.asarray(outputs["exit_bars"])[chosen]
    local["exit_reason_code"] = np.asarray(outputs["reason"])[chosen]
    local["gross_return_on_target"] = np.asarray(outputs["gross_return"])[chosen]
    local["net_return_on_target"] = np.asarray(outputs["net_return"])[chosen]
    local["size_multiplier"] = mult
    local["filled_fraction"] = filled
    local["additions"] = np.asarray(diag["additions"])[chosen]
    local["target_size"] = base * mult
    local["actual_filled_size"] = base * mult * filled
    local["gross_pnl_bankroll"] = local["gross_return_on_target"] * local["target_size"]
    local["net_pnl_bankroll"] = local["net_return_on_target"] * local["target_size"]
    local["week"] = pd.to_datetime(local["timestamp"], utc=True).dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
    records: list[dict[str, Any]] = []
    for week, group in local.groupby("week", sort=True):
        pnl = group["net_pnl_bankroll"].to_numpy(dtype=np.float64)
        equity = np.cumsum(pnl)
        dd = equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
        records.append(
            {
                "policy": policy,
                "week": week,
                "trades": int(len(group)),
                "net_pnl_bankroll": float(pnl.sum()),
                "gross_pnl_bankroll": float(group["gross_pnl_bankroll"].sum()),
                "fee_and_spread_pnl_bankroll": float((group["gross_pnl_bankroll"] - group["net_pnl_bankroll"]).sum()),
                "hit_rate": float(np.mean(group["net_return_on_target"] > 0.0)),
                "mean_net_return_on_target": float(group["net_return_on_target"].mean()),
                "max_drawdown": float(dd.min()),
                "target_entry_exposure": float(group["target_size"].sum()),
                "actual_entry_exposure": float(group["actual_filled_size"].sum()),
                "actual_to_target_exposure": float(group["actual_filled_size"].sum() / max(group["target_size"].sum(), 1e-12)),
                "dca_trigger_rate": float(np.mean(group["additions"] > 0)),
                "mean_additions": float(group["additions"].mean()),
            }
        )
    return pd.DataFrame(records), local


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHAMPION / "dca_ablation_v1",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    old_candidates = BASE / "execution_candidates_may_july_v1/simple_policy_candidates_with_archetypes.parquet"
    rich = BASE / "admission_may_july_oos_v1/admitted_oos_rows_execution_ledger.parquet"
    posterior = BASE / "complete_parent_state_july_v1/complete_oos_residual_event_states.parquet"
    parent_summary = BASE / "simple_policy_mayjune_fit_july_holdout_v1/side_parent_policy_summary.csv"
    old_atr_path = CHAMPION / "replay/causal_entry_atr_audit.parquet"
    old_cache = CHAMPION / "replay/path_cache"
    params = json.loads((CHAMPION / "evidence/nested_params.json").read_text())
    store_root = Path("data_perp/exchanges/krakenfutures/execution_1m")
    spec = ConstrainedReplaySpec()

    rows = pd.read_parquet(old_candidates)
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True)
    rows = rows.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    context, _, context_audit = _load_context(rows, rich, posterior)
    atr = _load_atr(rows, old_atr_path)
    deployed, _ = _load_deployed_side_params(parent_summary)
    open0, high, low, close, valid, old_path_manifest = _load_or_build_path_cache(
        rows, store_root=store_root, cache_dir=old_cache, spec=spec, rebuild=False
    )
    data = ExperimentData(rows, open0, high, low, close, valid, atr, spec, deployed)

    fold_records: list[dict[str, Any]] = []
    grid_frames: list[pd.DataFrame] = []
    local_records: list[dict[str, Any]] = []
    reconciliation_records: list[dict[str, Any]] = []
    choices: dict[str, Any] = {}
    fold_runtime: dict[str, Any] = {}

    for fold in FOLDS:
        name = fold["fold"]
        inner = INNER_FOLDS[name]
        search_idx = _indices_between(data, fold["train_start"], inner["search_end"])
        inner_idx = _indices_between(data, inner["inner_start"], inner["inner_end"])
        train_idx = _indices_between(data, fold["train_start"], fold["train_end"])
        outer_idx = _indices_between(data, fold["validation_start"], fold["validation_end"])
        search_params = params[name]["search_parent"]
        full_params = params[name]["full_train_parent"]
        sizing = params[name]["sizing"]

        frozen_search = data.simulate(search_idx, search_params, FAMILY_TRAILING_ONLY)
        frozen_inner = data.simulate(inner_idx, search_params, FAMILY_TRAILING_ONLY)
        frozen_train = data.simulate(train_idx, full_params, FAMILY_TRAILING_ONLY)
        frozen_outer = data.simulate(outer_idx, full_params, FAMILY_TRAILING_ONLY)
        sizes_search, _ = _bayesian_sizes(
            data,
            search_idx,
            inner_idx,
            frozen_search,
            context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )
        sizes_full, sizing_state = _bayesian_sizes(
            data,
            train_idx,
            outer_idx,
            frozen_train,
            context,
            strength=float(sizing["strength"]),
            ood_weight=float(sizing["ood_weight"]),
        )

        primary_choice, primary_grid = _search_grid(
            data, inner_idx, frozen_inner, sizes_search, PRIMARY_GRID, fold=name, literal=False
        )
        dca_first_choice, dca_first_grid = _search_grid(
            data, inner_idx, frozen_inner, sizes_search, PRIMARY_GRID,
            fold=name, literal=False, dca_first=True,
        )
        literal_choice, literal_grid = _search_grid(
            data, inner_idx, frozen_inner, sizes_search, LITERAL_GRID, fold=name, literal=True
        )
        primary_grid["formulation"] = "exposure_neutral_exit_first"
        dca_first_grid["formulation"] = "exposure_neutral_dca_first_bound"
        literal_grid["formulation"] = "literal_additional"
        grid_frames.extend([primary_grid, dca_first_grid, literal_grid])
        choices[name] = {
            "exposure_neutral_exit_first": {"x": primary_choice[0], "y_fraction": primary_choice[1]},
            "exposure_neutral_dca_first_bound": {
                "x": dca_first_choice[0], "y_fraction": dca_first_choice[1]
            },
            "literal_additional": {"x": literal_choice[0], "y_fraction": literal_choice[1]},
        }

        _, base_diag = _apply_dca(
            data, outer_idx, frozen_outer, x=1, y=0.0, literal=False
        )
        base_metrics = _metric(
            data,
            outer_idx,
            frozen_outer,
            sizes_full,
            base_diag,
            x=1,
            y=0.0,
            literal=False,
        )
        fold_records.append({"fold": name, "policy": "winner_no_dca", **base_metrics})
        reconciliation_records.extend(
            _loss_reconciliation(
                data.rows.iloc[outer_idx], frozen_outer, base_diag, period=name
            )
        )
        for policy, choice, literal, dca_first in (
            ("winner_dca_exposure_neutral_exit_first", primary_choice, False, False),
            ("winner_dca_exposure_neutral_dca_first_bound", dca_first_choice, False, True),
            ("winner_dca_literal_additional", literal_choice, True, False),
        ):
            output, diag = _apply_dca(
                data, outer_idx, frozen_outer, x=choice[0], y=choice[1],
                literal=literal, dca_first=dca_first,
            )
            metrics = _metric(
                data,
                outer_idx,
                output,
                sizes_full,
                diag,
                x=choice[0],
                y=choice[1],
                literal=literal,
            )
            fold_records.append({"fold": name, "policy": policy, **metrics})

            for local_x, local_y in _local_grid(choice[0], choice[1], literal=literal):
                inner_out, inner_diag = _apply_dca(
                    data, inner_idx, frozen_inner, x=local_x, y=local_y,
                    literal=literal, dca_first=dca_first,
                )
                outer_out, outer_diag = _apply_dca(
                    data, outer_idx, frozen_outer, x=local_x, y=local_y,
                    literal=literal, dca_first=dca_first,
                )
                inner_metrics = _metric(
                    data, inner_idx, inner_out, sizes_search, inner_diag,
                    x=local_x, y=local_y, literal=literal,
                )
                outer_metrics = _metric(
                    data, outer_idx, outer_out, sizes_full, outer_diag,
                    x=local_x, y=local_y, literal=literal,
                )
                local_records.append(
                    {
                        "fold": name,
                        "formulation": (
                            "literal_additional" if literal else
                            "exposure_neutral_dca_first_bound" if dca_first else
                            "exposure_neutral_exit_first"
                        ),
                        "selected_x": choice[0],
                        "selected_y_fraction": choice[1],
                        "x": local_x,
                        "y_fraction": local_y,
                        "inner_objective": inner_metrics["objective"],
                        "inner_net_pnl": inner_metrics["net_pnl_bankroll"],
                        "outer_objective_diagnostic": outer_metrics["objective"],
                        "outer_net_pnl_diagnostic": outer_metrics["net_pnl_bankroll"],
                        "outer_max_drawdown_diagnostic": outer_metrics["max_drawdown"],
                    }
                )
        fold_runtime[name] = {
            "search_rows": int(len(search_idx)),
            "inner_rows": int(len(inner_idx)),
            "train_rows": int(len(train_idx)),
            "outer_rows": int(len(outer_idx)),
            "sizing_state": sizing_state,
        }

    fold_metrics = pd.DataFrame(fold_records)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(args.output_dir / "inner_search_grid.csv", index=False)
    pd.DataFrame(local_records).to_csv(args.output_dir / "local_perturbations.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for policy, group in fold_metrics.groupby("policy", sort=False):
        summary_rows.append(
            {
                "policy": policy,
                "folds": int(len(group)),
                "positive_pnl_folds": int(np.sum(group["net_pnl_bankroll"] > 0.0)),
                "total_oos_net_pnl_bankroll": float(group["net_pnl_bankroll"].sum()),
                "mean_fold_net_pnl_bankroll": float(group["net_pnl_bankroll"].mean()),
                "worst_fold_net_pnl_bankroll": float(group["net_pnl_bankroll"].min()),
                "stable_fold_objective": stable_fold_objective(group["objective"].to_numpy(dtype=float)),
                "worst_week": float(group["worst_week"].min()),
                "worst_max_drawdown": float(group["max_drawdown"].min()),
                "total_trades": int(group["n_trades"].sum()),
                "mean_hit_rate": float(group["hit_rate"].mean()),
                "mean_actual_to_target_exposure": float(group["actual_to_target_exposure"].mean()),
                "mean_dca_trigger_rate": float(group["dca_trigger_rate"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "fold_summary.csv", index=False)

    # Frozen July application: fold-3 choice and parameters, extended through
    # July 16 with the canonical historical production frontier.
    fold3 = next(item for item in FOLDS if item["fold"] == "fold_3")
    fold3_params = params["fold_3"]["full_train_parent"]
    fold3_sizing = params["fold_3"]["sizing"]
    train3_idx = _indices_between(data, fold3["train_start"], fold3["train_end"])
    july_old_idx = _indices_between(data, "2026-07-01", "2026-07-11")
    frozen_train3 = data.simulate(train3_idx, fold3_params, FAMILY_TRAILING_ONLY)
    frozen_july_old = data.simulate(july_old_idx, fold3_params, FAMILY_TRAILING_ONLY)
    old_sizes3, _ = _bayesian_sizes(
        data,
        train3_idx,
        july_old_idx,
        frozen_train3,
        context,
        strength=float(fold3_sizing["strength"]),
        ood_weight=float(fold3_sizing["ood_weight"]),
    )

    forward = pd.read_parquet(FORWARD_SOURCE)
    forward["timestamp"] = pd.to_datetime(forward["timestamp"], utc=True)
    forward = forward.loc[
        forward["timestamp"].ge(pd.Timestamp("2026-07-11", tz="UTC"))
        & forward["timestamp"].lt(pd.Timestamp("2026-07-17", tz="UTC"))
    ].copy()
    forward = _with_policy_spread_cost_columns(forward, market_mode="perps")
    forward = forward.sort_values(["timestamp", "rank_pct"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    forward_context, forward_context_audit = _forward_context(forward)
    fallback_counts: dict[str, int] = {}
    for column in forward_context.columns:
        values = pd.to_numeric(forward_context[column], errors="coerce")
        missing = ~np.isfinite(values.to_numpy(dtype=np.float64))
        fallback_counts[column] = int(missing.sum())
        if missing.any():
            values.loc[missing] = float(
                np.nanmedian(pd.to_numeric(context.iloc[train3_idx][column], errors="coerce"))
            )
        forward_context[column] = values
    forward_context_audit["frozen_train_median_fallback_counts"] = fallback_counts
    forward_atr = _load_atr(
        forward,
        CHAMPION / "forward_replay_jul11_17_v1/causal_entry_atr_audit.parquet",
    )
    f_open, f_high, f_low, f_close, f_valid, forward_path_manifest = _load_or_build_path_cache(
        forward,
        store_root=store_root,
        cache_dir=CHAMPION / "forward_replay_jul11_17_v1/path_cache",
        spec=spec,
        rebuild=False,
    )
    forward_data = ExperimentData(
        forward, f_open, f_high, f_low, f_close, f_valid, forward_atr, spec, deployed
    )
    forward_idx = np.arange(len(forward), dtype=np.int64)
    frozen_forward = forward_data.simulate(forward_idx, fold3_params, FAMILY_TRAILING_ONLY)

    combined_rows_for_size = pd.concat([rows, forward], ignore_index=True, copy=False)
    combined_context = pd.concat([context, forward_context], ignore_index=True, copy=False)
    sizing_data = SimpleNamespace(
        rows=combined_rows_for_size,
        side=pd.to_numeric(combined_rows_for_size["side"], errors="coerce").to_numpy(dtype=np.float64),
        rank=pd.to_numeric(combined_rows_for_size["rank_pct"], errors="coerce").to_numpy(dtype=np.float64),
    )
    forward_combined_idx = np.arange(len(rows), len(combined_rows_for_size), dtype=np.int64)
    combined_sizes, _ = _bayesian_sizes(
        sizing_data,
        train3_idx,
        forward_combined_idx,
        frozen_train3,
        combined_context,
        strength=float(fold3_sizing["strength"]),
        ood_weight=float(fold3_sizing["ood_weight"]),
    )
    forward_sizes = combined_sizes[forward_combined_idx]

    july_rows = pd.concat([rows.iloc[july_old_idx], forward], ignore_index=True, copy=False)
    july_data = SimpleNamespace(rows=july_rows)
    july_idx = np.arange(len(july_rows), dtype=np.int64)
    july_sizes = np.concatenate([old_sizes3[july_old_idx], forward_sizes])
    frozen_july = _combine_outputs([frozen_july_old, frozen_forward])
    _, old_baseline_diag = _apply_dca(
        data, july_old_idx, frozen_july_old, x=1, y=0.0, literal=False
    )
    _, forward_baseline_diag = _apply_dca(
        forward_data, forward_idx, frozen_forward, x=1, y=0.0, literal=False
    )
    baseline_diag = {
        key: np.concatenate([old_baseline_diag[key], forward_baseline_diag[key]])
        for key in old_baseline_diag
    }
    july_metrics_records: list[dict[str, Any]] = []
    july_weekly_parts: list[pd.DataFrame] = []
    july_ledger_parts: list[pd.DataFrame] = []

    baseline_metrics = _metric(
        july_data, july_idx, frozen_july, july_sizes, baseline_diag,
        x=1, y=0.0, literal=False,
    )
    july_metrics_records.append({"policy": "winner_no_dca", **baseline_metrics})
    reconciliation_records.extend(
        _loss_reconciliation(
            july_rows, frozen_july, baseline_diag, period="july_frozen_through_16"
        )
    )
    weekly, ledger = _weekly_ledger(
        july_rows, frozen_july, july_sizes, baseline_diag, policy="winner_no_dca"
    )
    july_weekly_parts.append(weekly); july_ledger_parts.append(ledger)

    for policy, choice_key, literal, dca_first in (
        (
            "winner_dca_exposure_neutral_exit_first",
            "exposure_neutral_exit_first", False, False,
        ),
        (
            "winner_dca_exposure_neutral_dca_first_bound",
            "exposure_neutral_dca_first_bound", False, True,
        ),
        ("winner_dca_literal_additional", "literal_additional", True, False),
    ):
        choice = choices["fold_3"][choice_key]
        old_out, old_diag = _apply_dca(
            data, july_old_idx, frozen_july_old,
            x=int(choice["x"]), y=float(choice["y_fraction"]), literal=literal,
            dca_first=dca_first,
        )
        f_out, f_diag = _apply_dca(
            forward_data, forward_idx, frozen_forward,
            x=int(choice["x"]), y=float(choice["y_fraction"]), literal=literal,
            dca_first=dca_first,
        )
        output = _combine_outputs([old_out, f_out])
        diag = {
            key: np.concatenate([old_diag[key], f_diag[key]]) for key in old_diag
        }
        metrics = _metric(
            july_data, july_idx, output, july_sizes, diag,
            x=int(choice["x"]), y=float(choice["y_fraction"]), literal=literal,
        )
        july_metrics_records.append({"policy": policy, **metrics})
        weekly, ledger = _weekly_ledger(july_rows, output, july_sizes, diag, policy=policy)
        july_weekly_parts.append(weekly); july_ledger_parts.append(ledger)

    july_metrics = pd.DataFrame(july_metrics_records)
    baseline_pnl = float(july_metrics.loc[july_metrics["policy"].eq("winner_no_dca"), "net_pnl_bankroll"].iloc[0])
    baseline_dd = float(july_metrics.loc[july_metrics["policy"].eq("winner_no_dca"), "max_drawdown"].iloc[0])
    july_metrics["delta_net_pnl_vs_winner"] = july_metrics["net_pnl_bankroll"] - baseline_pnl
    july_metrics["delta_max_drawdown_vs_winner"] = july_metrics["max_drawdown"] - baseline_dd
    july_metrics.to_csv(args.output_dir / "july_frozen_metrics.csv", index=False)
    pd.concat(july_weekly_parts, ignore_index=True).to_csv(
        args.output_dir / "july_weekly_metrics.csv", index=False
    )
    pd.concat(july_ledger_parts, ignore_index=True).to_parquet(
        args.output_dir / "july_selected_trade_ledger.parquet", index=False
    )
    pd.DataFrame(reconciliation_records).to_csv(
        args.output_dir / "loss_reconciliation.csv", index=False
    )

    # x=1 must reproduce the frozen winner exactly before the search is trusted.
    x1_old, _ = _apply_dca(data, july_old_idx, frozen_july_old, x=1, y=0.0, literal=False)
    identity_error = {
        key: float(np.nanmax(np.abs(x1_old[key] - frozen_july_old[key])))
        for key in ("gross_return", "net_return")
    }
    if max(identity_error.values()) > 1e-12:
        raise RuntimeError(f"DCA x=1 identity check failed: {identity_error}")

    manifest = {
        "status": "complete",
        "experiment": "winner position staging / DCA ablation",
        "evidence": {
            "folds": "nested walk-forward policy-validation OOS",
            "july": "frozen application of fold-3 choice through 2026-07-16 23:00 UTC",
            "july17": "not scored because a complete 24h outcome path was not observable",
        },
        "winner": "joint_trailing_total_mfe_raw_bayesian_v1",
        "primary_definition": "x total equal tranches; initial target/x; up to x-1 additions at original raw entry * (1-side*k*y); max target exposure=1.0",
        "literal_definition": "initial target/x plus up to x additional equal tranches; max target exposure=1+1/x; diagnostic only",
        "collision_contract": "frozen catastrophic/adverse/trailing exit wins same-minute DCA collision; timeout final bar may fill before close",
        "collision_sensitivity": "exposure-neutral search is also run with DCA-first on the exit candle as an optimistic OHLC ordering bound",
        "geometry_contract": "winner exit geometry, ATR, MFE state, and exit time remain frozen from original entry; DCA changes weighted entry PnL only",
        "cost_contract": "every filled tranche pays entry half-spread and 0.5% entry fee; filled quantity pays frozen executable exit spread and 0.5% exit fee; costs applied once",
        "capacity_contract": "8-open/2-new admission remains size-independent and identical across arms; full winner target capacity is reserved at entry",
        "search_space": {
            "x_exposure_neutral": [1, 8],
            "x_literal": [2, 8],
            "y_pct": (100.0 * Y_GRID).tolist(),
            "primary_arms_per_fold": len(PRIMARY_GRID),
            "literal_arms_per_fold": len(LITERAL_GRID),
        },
        "folds": FOLDS,
        "inner_folds": INNER_FOLDS,
        "choices": choices,
        "fold_runtime": fold_runtime,
        "context_audit": context_audit,
        "forward_context_audit": forward_context_audit,
        "old_path_manifest": old_path_manifest,
        "forward_path_manifest": forward_path_manifest,
        "identity_x1_max_abs_error": identity_error,
        "outputs": [
            "fold_metrics.csv",
            "fold_summary.csv",
            "inner_search_grid.csv",
            "local_perturbations.csv",
            "july_frozen_metrics.csv",
            "july_weekly_metrics.csv",
            "july_selected_trade_ledger.parquet",
            "loss_reconciliation.csv",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8"
    )
    print("\nFOLD METRICS\n", fold_metrics.to_string(index=False))
    print("\nFOLD SUMMARY\n", summary.to_string(index=False))
    print("\nJULY FROZEN\n", july_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
