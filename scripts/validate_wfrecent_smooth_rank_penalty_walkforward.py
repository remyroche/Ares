#!/usr/bin/env python3
"""Chronological validation for fixed wf_recent smooth rank penalties."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.ablate_wfrecent_smooth_rank_penalty import (  # noqa: E402
    SmoothRule,
    _fit_threshold,
    _penalty_values,
)
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)


def _month_splits(first_holdout: str, last_holdout_end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(first_holdout, tz="UTC")
    end = pd.Timestamp(last_holdout_end, tz="UTC")
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = pd.Timestamp(cur + pd.offsets.MonthBegin(1))
        out.append((cur, min(nxt, end)))
        cur = nxt
    return out


def _fixed_rules(rule_set: str) -> list[SmoothRule]:
    if rule_set == "leaders":
        return [
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 1.0),
            SmoothRule("uncertainty_risk", "long_dist", 0.90, 0.70, 0.01, 2.0),
        ]
    if rule_set == "lead_robustness":
        return [
            # Winning rule from the fixed replay.
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 1.0),
            # Penalty magnitude sensitivity.
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.025, 1.0),
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.075, 1.0),
            # Risk threshold sensitivity.
            SmoothRule("composite_risk", "long_dist", 0.85, 0.70, 0.05, 1.0),
            SmoothRule("composite_risk", "long_dist", 0.95, 0.70, 0.05, 1.0),
            # Candidate rank threshold sensitivity.
            SmoothRule("composite_risk", "long_dist", 0.90, 0.80, 0.05, 1.0),
            # Shape sensitivity.
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 0.5),
            SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 2.0),
        ]
    raise ValueError(f"Unknown rule_set: {rule_set}")


def _leader_rules() -> list[SmoothRule]:
    return [
        SmoothRule("composite_risk", "long_dist", 0.90, 0.70, 0.05, 1.0),
        SmoothRule("uncertainty_risk", "long_dist", 0.90, 0.70, 0.01, 2.0),
    ]


def _apply_rule_to_holdout(train_scored: pd.DataFrame, holdout_scored: pd.DataFrame, rule: SmoothRule) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = holdout_scored.copy()
    base_adj = (
        pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "portfolio_rank_adjustment" in out.columns
        else np.zeros(len(out), dtype=np.float32)
    )
    threshold = _fit_threshold(train_scored, rule)
    penalty = _penalty_values(out, rule, threshold)
    out["portfolio_rank_adjustment"] = np.clip(base_adj + penalty, -1.0, 1.0).astype("float32")
    return out, {
        "threshold": threshold,
        "penalized_rows": int(np.sum(penalty < 0.0)),
        "mean_penalty": float(np.mean(penalty[penalty < 0.0])) if np.any(penalty < 0.0) else 0.0,
    }


def _delta_row(base: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "objective_week",
        "q20_week_net_pnl",
        "q35_week_net_pnl",
        "worst_week_net_pnl",
        "positive_weeks",
    ):
        row[f"baseline_{key}"] = base[key]
        row[f"challenger_{key}"] = challenger[key]
        row[f"delta_{key}"] = float(challenger[key] - base[key])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_walkforward_20260701"))
    parser.add_argument("--first-holdout", default="2026-02-01")
    parser.add_argument("--last-holdout-end", default="2026-06-27")
    parser.add_argument("--rule-set", choices=["leaders", "lead_robustness"], default="leaders")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(args.input_dir / "combo_candidates.parquet")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in candidates.columns:
        candidates["portfolio_rank_adjustment"] = np.float32(0.0)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    rules = _fixed_rules(args.rule_set)
    split_rows: list[dict[str, Any]] = []
    weekly_rows: list[pd.DataFrame] = []

    for split_id, (holdout_start, holdout_end) in enumerate(_month_splits(args.first_holdout, args.last_holdout_end)):
        raw_train = candidates[candidates["timestamp"].lt(holdout_start)].copy().reset_index(drop=True)
        raw_holdout = candidates[candidates["timestamp"].ge(holdout_start) & candidates["timestamp"].lt(holdout_end)].copy().reset_index(drop=True)
        if raw_train.empty or raw_holdout.empty:
            continue
        refs = _fit_percentile_reference(raw_train)
        train_scored = _apply_risk_scores(raw_train, refs).reset_index(drop=True)
        holdout_scored = _apply_risk_scores(raw_holdout, refs).reset_index(drop=True)
        ev_curve = fit_hierarchical_ev_curves(train_scored)

        base_decisions, _base_equity, base_metrics = replay_candidates(
            holdout_scored,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        _base_daily, base_weekly = _period_tables(base_decisions)
        base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, args.q35_weight, args.q20_weight)

        for rule in rules:
            adjusted, rule_info = _apply_rule_to_holdout(train_scored, holdout_scored, rule)
            decisions, _equity, metrics = replay_candidates(
                adjusted,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode="perps",
            )
            _daily, weekly = _period_tables(decisions)
            cur_summary = _summary(rule.label, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
            row = {
                "split_id": int(split_id),
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "label": rule.label,
                **rule.__dict__,
                **rule_info,
                **_delta_row(base_summary, cur_summary),
            }
            split_rows.append(row)
            weekly = weekly.copy()
            weekly["split_id"] = int(split_id)
            weekly["label"] = rule.label
            weekly["holdout_start"] = holdout_start.isoformat()
            weekly_rows.append(weekly)

    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        raise RuntimeError("No walk-forward split rows were produced")
    summary_rows = []
    for label, group in split_df.groupby("label", sort=False):
        summary_rows.append(
            {
                "label": label,
                "splits": int(len(group)),
                "sum_delta_net_pnl": float(group["delta_net_pnl"].sum()),
                "median_delta_net_pnl": float(group["delta_net_pnl"].median()),
                "positive_delta_net_pnl_share": float((group["delta_net_pnl"] > 0.0).mean()),
                "sum_delta_objective_week": float(group["delta_objective_week"].sum()),
                "median_delta_objective_week": float(group["delta_objective_week"].median()),
                "positive_delta_objective_share": float((group["delta_objective_week"] > 0.0).mean()),
                "sum_delta_worst_week_net_pnl": float(group["delta_worst_week_net_pnl"].sum()),
                "mean_delta_hit_rate": float(group["delta_hit_rate"].mean()),
                "mean_delta_full_sl_rate": float(group["delta_full_sl_rate"].mean()),
                "mean_delta_timeout_rate": float(group["delta_timeout_rate"].mean()),
                "sum_penalized_rows": int(group["penalized_rows"].sum()),
                "mean_penalized_rows": float(group["penalized_rows"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["sum_delta_objective_week", "sum_delta_net_pnl"], ascending=[False, False])
    weekly_out = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()

    split_df.to_csv(args.output_dir / "smooth_rank_penalty_walkforward_splits.csv", index=False)
    summary.to_csv(args.output_dir / "smooth_rank_penalty_walkforward_summary.csv", index=False)
    weekly_out.to_csv(args.output_dir / "smooth_rank_penalty_walkforward_weekly.csv", index=False)
    manifest = {
        "generated_by": "validate_wfrecent_smooth_rank_penalty_walkforward",
        "input_dir": str(args.input_dir),
        "first_holdout": args.first_holdout,
        "last_holdout_end": args.last_holdout_end,
        "rules": [rule.__dict__ for rule in rules],
        "rule_set": args.rule_set,
        "candidate_rows": int(len(candidates)),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    best = summary.iloc[0]
    lines = [
        "# wf_recent Smooth Rank-Penalty Walk-Forward Validation",
        "",
        "Fixed smooth rank-penalty challengers are evaluated on monthly chronological holdouts. Diagnostic references and EV curves are fit from prior rows only.",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "label",
                "splits",
                "sum_delta_net_pnl",
                "median_delta_net_pnl",
                "positive_delta_net_pnl_share",
                "sum_delta_objective_week",
                "median_delta_objective_week",
                "positive_delta_objective_share",
                "sum_delta_worst_week_net_pnl",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "mean_delta_timeout_rate",
                "sum_penalized_rows",
            ],
        ),
        "",
        "## Split Detail For Best Rule",
        "",
        _fmt_table(
            split_df[split_df["label"].eq(str(best["label"]))],
            [
                "holdout_start",
                "holdout_end",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_worst_week_net_pnl",
                "penalized_rows",
            ],
        ),
    ]
    (args.output_dir / "smooth_rank_penalty_walkforward_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
