#!/usr/bin/env python3
"""Frozen parent-rank and sizing ablation for breakout-path EBM probabilities.

This is not a meta-model retrain. It measures whether the two already OOS
path-quality probabilities add economic information as a small continuous
overlay to a frozen parent rank. Penalty/sizing coefficients are selected only
before ``--eval-start`` and evaluated once afterwards.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ("__ts__", "__symbol__", "side_name", "archetype_policy_key")
RISK_COLUMNS = (
    "breakout_rapid_reversal_probability_ebm",
    "breakout_rapid_reversal_probability_reliability",
    "breakout_severe_retention_probability_ebm",
    "breakout_severe_retention_probability_reliability",
)


def _metrics(frame: pd.DataFrame, weight: np.ndarray | None = None) -> dict[str, float]:
    valid_mask = pd.to_numeric(frame["ev_after_1pct"], errors="coerce").notna().to_numpy()
    valid = frame.loc[valid_mask].copy()
    if weight is None:
        weight = np.ones(len(valid), dtype=np.float64)
    else:
        weight = np.asarray(weight, dtype=np.float64)[valid_mask]
    ev = pd.to_numeric(valid["ev_after_1pct"], errors="coerce").to_numpy(np.float64)
    clean = pd.to_numeric(valid["clean_exec"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    bad_mae = pd.to_numeric(valid["first_touch_bad_mae_1r"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    timeout = pd.to_numeric(valid["timeout"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    total_weight = max(float(weight.sum()), 1e-12)
    return {
        "selected_rows": float(len(valid)),
        "effective_rows": total_weight,
        "mean_ev_after_1pct": float(np.dot(weight, ev) / total_weight),
        "sum_ev_after_1pct": float(np.dot(weight, ev)),
        "clean_exec_rate": float(np.dot(weight, clean) / total_weight),
        "bad_mae_rate": float(np.dot(weight, bad_mae) / total_weight),
        "timeout_rate": float(np.dot(weight, timeout) / total_weight),
    }


def _risk(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rapid = pd.to_numeric(frame[RISK_COLUMNS[0]], errors="coerce").fillna(0.0).to_numpy(np.float64)
    rapid_rel = pd.to_numeric(frame[RISK_COLUMNS[1]], errors="coerce").fillna(0.0).to_numpy(np.float64)
    retention = pd.to_numeric(frame[RISK_COLUMNS[2]], errors="coerce").fillna(0.0).to_numpy(np.float64)
    retention_rel = pd.to_numeric(frame[RISK_COLUMNS[3]], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return rapid * rapid_rel, retention * retention_rel


def _with_context(parent: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    local_context = context.rename(columns={"__archetype_policy_key__": "archetype_policy_key"})
    if local_context.duplicated(KEYS).any():
        raise ValueError("Path-quality context is not unique on timestamp/symbol/side/archetype")
    result = parent.merge(
        local_context.loc[:, [*KEYS, *RISK_COLUMNS]], on=list(KEYS), how="left", indicator=True
    )
    result["breakout_path_context_available"] = result.pop("_merge").eq("both").astype(np.int8)
    for column in RISK_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0).astype(np.float32)
    return result


def _select(
    frame: pd.DataFrame,
    alpha_rapid: float,
    alpha_retention: float,
    threshold: float,
    rank_column: str,
    selection_mode: str,
) -> pd.DataFrame:
    rapid, retention = _risk(frame)
    parent_rank = pd.to_numeric(frame[rank_column], errors="coerce").fillna(0.0).to_numpy(np.float64)
    adjusted_rank = parent_rank - alpha_rapid * rapid - alpha_retention * retention
    output = frame.copy()
    output["breakout_path_adjusted_rank"] = adjusted_rank.astype(np.float32)
    if selection_mode == "historical_threshold":
        return output.loc[adjusted_rank >= threshold].copy()
    if selection_mode == "cross_sectional_top10":
        # Preserve a fixed per-timestamp budget and let the path penalty replace
        # risky short-breakout rows with the next-best parent-score candidates.
        output["__tie_key__"] = (
            output["__symbol__"].astype(str)
            + "|" + output["side_name"].astype(str)
            + "|" + output["archetype_policy_key"].astype(str)
        )
        ordered = output.sort_values(
            ["__ts__", "breakout_path_adjusted_rank", "__tie_key__"],
            ascending=[True, False, True], kind="stable",
        )
        ordered["__rank_position__"] = ordered.groupby("__ts__", observed=True).cumcount()
        ordered["__rank_budget__"] = ordered.groupby("__ts__", observed=True)["__tie_key__"].transform("size")
        selected = ordered.loc[
            ordered["__rank_position__"] < np.ceil(ordered["__rank_budget__"] * (1.0 - threshold))
        ].copy()
        return selected.drop(columns=["__tie_key__", "__rank_position__", "__rank_budget__"])
    raise ValueError(f"Unknown selection mode: {selection_mode}")


def _fit_penalty(
    train: pd.DataFrame,
    threshold: float,
    minimum_activity: float,
    rank_column: str,
    selection_mode: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    baseline = _select(train, 0.0, 0.0, threshold, rank_column, selection_mode)
    baseline_metrics = _metrics(baseline)
    rows: list[dict[str, float]] = []
    for alpha_rapid, alpha_retention in itertools.product((0.0, 0.01, 0.02, 0.05, 0.10), repeat=2):
        selected = _select(train, alpha_rapid, alpha_retention, threshold, rank_column, selection_mode)
        metrics = _metrics(selected)
        activity_ratio = metrics["selected_rows"] / max(baseline_metrics["selected_rows"], 1.0)
        feasible = (
            activity_ratio >= minimum_activity
            and metrics["clean_exec_rate"] >= baseline_metrics["clean_exec_rate"]
            and metrics["sum_ev_after_1pct"] >= baseline_metrics["sum_ev_after_1pct"]
        )
        rows.append({
            "alpha_rapid": alpha_rapid,
            "alpha_retention": alpha_retention,
            "activity_ratio": activity_ratio,
            "feasible": feasible,
            **metrics,
        })
    search = pd.DataFrame(rows)
    pool = search.loc[search["feasible"]].copy()
    if pool.empty:
        chosen = {"alpha_rapid": 0.0, "alpha_retention": 0.0}
    else:
        chosen_row = pool.sort_values(
            ["mean_ev_after_1pct", "clean_exec_rate", "sum_ev_after_1pct"],
            ascending=False,
            kind="stable",
        ).iloc[0]
        chosen = {"alpha_rapid": float(chosen_row["alpha_rapid"]), "alpha_retention": float(chosen_row["alpha_retention"])}
    return chosen, search


def _fit_sizing(
    train: pd.DataFrame,
    threshold: float,
    minimum_exposure: float,
    rank_column: str,
    selection_mode: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    selected = _select(train, 0.0, 0.0, threshold, rank_column, selection_mode)
    baseline = _metrics(selected)
    rapid, retention = _risk(selected)
    rows: list[dict[str, float]] = []
    for lambda_rapid, lambda_retention in itertools.product((0.0, 0.10, 0.20, 0.30, 0.40), repeat=2):
        weight = np.clip(1.0 - lambda_rapid * rapid - lambda_retention * retention, 0.50, 1.00)
        metrics = _metrics(selected, weight)
        exposure_ratio = metrics["effective_rows"] / max(baseline["effective_rows"], 1.0)
        feasible = exposure_ratio >= minimum_exposure and metrics["sum_ev_after_1pct"] >= baseline["sum_ev_after_1pct"]
        rows.append({"lambda_rapid": lambda_rapid, "lambda_retention": lambda_retention, "exposure_ratio": exposure_ratio, "feasible": feasible, **metrics})
    search = pd.DataFrame(rows)
    pool = search.loc[search["feasible"]].copy()
    if pool.empty:
        chosen = {"lambda_rapid": 0.0, "lambda_retention": 0.0}
    else:
        chosen_row = pool.sort_values(["mean_ev_after_1pct", "sum_ev_after_1pct"], ascending=False, kind="stable").iloc[0]
        chosen = {"lambda_rapid": float(chosen_row["lambda_rapid"]), "lambda_retention": float(chosen_row["lambda_retention"])}
    return chosen, search


def run(args: argparse.Namespace) -> dict[str, object]:
    parent_columns = [*KEYS, args.rank_column, "ev_after_1pct", "clean_exec", "first_touch_bad_mae_1r", "timeout"]
    parent = pd.read_parquet(args.parent_ledger, columns=parent_columns)
    parent["__ts__"] = pd.to_datetime(parent["__ts__"], utc=True, errors="coerce")
    parent = parent.dropna(subset=["__ts__"])
    context = pd.read_parquet(args.path_context)
    context["__ts__"] = pd.to_datetime(context["__ts__"], utc=True, errors="coerce")
    rows = _with_context(parent, context)
    start = pd.Timestamp(args.train_start, tz="UTC")
    eval_start = pd.Timestamp(args.eval_start, tz="UTC")
    eval_end = pd.Timestamp(args.eval_end, tz="UTC")
    train = rows.loc[rows["__ts__"].ge(start) & rows["__ts__"].lt(eval_start)].copy()
    evaluation = rows.loc[rows["__ts__"].ge(eval_start) & rows["__ts__"].lt(eval_end)].copy()
    if parent[args.rank_column].notna().sum() == 0:
        raise ValueError(f"Parent rank column {args.rank_column!r} has no populated rows")
    chosen_penalty, penalty_search = _fit_penalty(
        train, args.rank_threshold, args.minimum_activity, args.rank_column, args.selection_mode
    )
    chosen_sizing, sizing_search = _fit_sizing(
        train, args.rank_threshold, args.minimum_exposure, args.rank_column, args.selection_mode
    )
    baseline_eval = _select(
        evaluation, 0.0, 0.0, args.rank_threshold, args.rank_column, args.selection_mode
    )
    penalty_eval = _select(
        evaluation, **chosen_penalty, threshold=args.rank_threshold,
        rank_column=args.rank_column, selection_mode=args.selection_mode,
    )
    sizing_rapid, sizing_retention = _risk(baseline_eval)
    sizing_weight = np.clip(
        1.0 - chosen_sizing["lambda_rapid"] * sizing_rapid - chosen_sizing["lambda_retention"] * sizing_retention,
        0.50, 1.00,
    )
    baseline_name = f"baseline_parent_{args.selection_mode}"
    results = pd.DataFrame([
        {"variant": baseline_name, **_metrics(baseline_eval)},
        {"variant": "ebm_soft_penalty", **chosen_penalty, **_metrics(penalty_eval)},
        {"variant": "ebm_sizing_modifier_proxy", **chosen_sizing, **_metrics(baseline_eval, sizing_weight)},
    ])
    baseline = results.iloc[0]
    for column in ("mean_ev_after_1pct", "sum_ev_after_1pct", "clean_exec_rate", "bad_mae_rate", "timeout_rate"):
        results[f"delta_{column}_vs_baseline"] = results[column] - baseline[column]
    results["activity_or_exposure_ratio_vs_baseline"] = results["effective_rows"] / max(float(baseline["effective_rows"]), 1.0)
    args.output.mkdir(parents=True, exist_ok=True)
    penalty_search.to_csv(args.output / "penalty_grid_train_only.csv", index=False)
    sizing_search.to_csv(args.output / "sizing_grid_train_only.csv", index=False)
    results.to_csv(args.output / "oos_results.csv", index=False)
    monthly: list[pd.DataFrame] = []
    for name, selected, weight in (
        ("baseline_parent_rank", baseline_eval, None),
        ("ebm_soft_penalty", penalty_eval, None),
        ("ebm_sizing_modifier_proxy", baseline_eval, sizing_weight),
    ):
        local = selected.copy()
        local["month"] = local["__ts__"].dt.to_period("M").astype(str)
        rows_by_month = []
        for month, group in local.groupby("month", observed=True):
            group_weight = None if weight is None else weight[local.index.get_indexer(group.index)]
            rows_by_month.append({"variant": name, "month": month, **_metrics(group, group_weight)})
        monthly.append(pd.DataFrame(rows_by_month))
    pd.concat(monthly, ignore_index=True, copy=False).to_csv(args.output / "oos_monthly_results.csv", index=False)
    manifest = {
        "schema": "breakout_path_quality_parent_ablation_v1",
        "status": "research_only_no_meta_retrain_no_policy_change",
        "parent_ledger": str(args.parent_ledger),
        "path_context": str(args.path_context),
        "train_period": [str(start), str(eval_start)],
        "evaluation_period": [str(eval_start), str(eval_end)],
        "selected_penalty": chosen_penalty,
        "selected_sizing_proxy": chosen_sizing,
        "rank_threshold": args.rank_threshold,
        "rank_column": args.rank_column,
        "selection_mode": args.selection_mode,
        "context_coverage": {
            "all_parent_rows": int(len(rows)),
            "all_parent_context_rows": int(rows["breakout_path_context_available"].sum()),
            "short_breakout_parent_rows": int(((rows["side_name"] == "short") & (rows["archetype_policy_key"] == "short_breakout_precision")).sum()),
            "short_breakout_context_rows": int(rows.loc[(rows["side_name"] == "short") & (rows["archetype_policy_key"] == "short_breakout_precision"), "breakout_path_context_available"].sum()),
        },
        "leakage_contract": (
            "Penalty and sizing coefficients are selected only on the pre-April path-model OOS / "
            "parent-OOF overlap. April-June is held out once. Path probabilities and reliability "
            "are pre-entry OOS fields; realized outcomes only score the report."
        ),
        "limitation": (
            "The soft-penalty test keeps the parent threshold fixed and the sizing result is an "
            "exposure-normalized proxy. A true meta-feature test requires retraining the meta model "
            "with these fields in its feature contract."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "train_rows": int(len(train)), "evaluation_rows": int(len(evaluation))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-ledger", type=Path, required=True)
    parser.add_argument("--path-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-start", default="2025-07-01")
    parser.add_argument("--eval-start", default="2026-04-01")
    parser.add_argument("--eval-end", default="2026-07-01")
    parser.add_argument("--rank-threshold", type=float, default=0.90)
    parser.add_argument("--rank-column", default="historical_rank")
    parser.add_argument("--selection-mode", choices=("historical_threshold", "cross_sectional_top10"), default="historical_threshold")
    parser.add_argument("--minimum-activity", type=float, default=0.98)
    parser.add_argument("--minimum-exposure", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
