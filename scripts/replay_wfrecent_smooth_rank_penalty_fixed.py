#!/usr/bin/env python3
"""Continuous replay for a fixed wf_recent smooth rank-penalty challenger."""

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
    _apply_smooth_rule_expanding,
)
from scripts.replay_wfrecent_row_guard_expanding import (  # noqa: E402
    _delta_summary,
    _monthly_table,
)
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)


def _parse_rule(args: argparse.Namespace) -> SmoothRule:
    return SmoothRule(
        score_name=str(args.score_name),
        scope=str(args.scope),
        risk_quantile=float(args.risk_quantile),
        min_rank_pct=float(args.min_rank_pct),
        max_penalty=float(args.max_penalty),
        power=float(args.power),
    )


def _per_head_table(base_decisions: pd.DataFrame, challenger_decisions: pd.DataFrame) -> pd.DataFrame:
    def prep(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        accepted = frame[frame["accepted"].astype(bool)].copy()
        if accepted.empty:
            return pd.DataFrame(columns=["head"])
        accepted["head"] = accepted["strategy_id"].map(_head_name)
        size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
        gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
        accepted["net_pnl"] = size * net
        accepted["gross_pnl"] = size * gross
        accepted["is_win"] = net > 0.0
        reason = accepted["position_exit_reason"].astype(str) if "position_exit_reason" in accepted.columns else pd.Series("", index=accepted.index)
        accepted["is_full_sl"] = reason.str.contains("sl", case=False, na=False)
        accepted["is_timeout"] = reason.str.contains("timeout", case=False, na=False)
        out = (
            accepted.groupby("head", as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                gross_pnl=("gross_pnl", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
        )
        return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "head"})

    out = prep(base_decisions, "baseline").merge(prep(challenger_decisions, "challenger"), on="head", how="outer")
    for key in ("net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate"):
        out[f"delta_{key}"] = out[f"challenger_{key}"] - out[f"baseline_{key}"]
    return out.sort_values("head").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_fixed_replay_20260701"))
    parser.add_argument("--first-guard-month", default="2026-02-01")
    parser.add_argument("--end", default="2026-06-27")
    parser.add_argument("--score-name", default="composite_risk")
    parser.add_argument("--scope", default="long_dist")
    parser.add_argument("--risk-quantile", type=float, default=0.90)
    parser.add_argument("--min-rank-pct", type=float, default=0.70)
    parser.add_argument("--max-penalty", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=1.0)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rule = _parse_rule(args)
    first_guard = pd.Timestamp(args.first_guard_month, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    candidates = pd.read_parquet(args.input_dir / "combo_candidates.parquet")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in candidates.columns:
        candidates["portfolio_rank_adjustment"] = np.float32(0.0)

    challenger, schedule = _apply_smooth_rule_expanding(candidates, rule, first_guard, end)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_train = candidates[candidates["timestamp"].lt(first_guard)].copy().reset_index(drop=True)
    ev_curve = fit_hierarchical_ev_curves(ev_train)

    base_decisions, _base_equity, base_metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    challenger_decisions, _challenger_equity, challenger_metrics = replay_candidates(
        challenger,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _base_daily, base_weekly = _period_tables(base_decisions)
    _challenger_daily, challenger_weekly = _period_tables(challenger_decisions)
    base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, args.q35_weight, args.q20_weight)
    challenger_summary = _summary(rule.label, challenger_decisions, challenger_weekly, challenger_metrics, args.q35_weight, args.q20_weight)
    delta = _delta_summary(base_summary, challenger_summary)
    monthly = _monthly_table(base_weekly, challenger_weekly)
    per_head = _per_head_table(base_decisions, challenger_decisions)

    pd.DataFrame([delta]).to_csv(args.output_dir / "fixed_smooth_rank_penalty_summary.csv", index=False)
    schedule.to_csv(args.output_dir / "fixed_smooth_rank_penalty_schedule.csv", index=False)
    base_weekly.to_csv(args.output_dir / "fixed_smooth_rank_penalty_baseline_weekly.csv", index=False)
    challenger_weekly.to_csv(args.output_dir / "fixed_smooth_rank_penalty_challenger_weekly.csv", index=False)
    monthly.to_csv(args.output_dir / "fixed_smooth_rank_penalty_monthly.csv", index=False)
    per_head.to_csv(args.output_dir / "fixed_smooth_rank_penalty_per_head.csv", index=False)
    challenger_decisions.to_parquet(args.output_dir / "fixed_smooth_rank_penalty_decisions.parquet", index=False)

    manifest = {
        "generated_by": "replay_wfrecent_smooth_rank_penalty_fixed",
        "input_dir": str(args.input_dir),
        "first_guard_month": args.first_guard_month,
        "end": args.end,
        "rule": rule.__dict__,
        "candidate_rows": int(len(candidates)),
        "penalized_rows_total": int(schedule["penalized_rows"].sum()) if not schedule.empty else 0,
        "ev_curve_fit": "pre_guard_history_only",
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Fixed Smooth Rank-Penalty Continuous Replay",
        "",
        "Standalone continuous replay of one fixed smooth rank-penalty challenger. Diagnostic references are expanding prior-month only; the EV curve is fit once on pre-guard history and shared by both arms.",
        "",
        "## Summary",
        "",
        _fmt_table(
            pd.DataFrame([delta]),
            [
                "baseline_net_pnl",
                "guard_net_pnl",
                "delta_net_pnl",
                "baseline_trade_count",
                "guard_trade_count",
                "delta_trade_count",
                "baseline_hit_rate",
                "guard_hit_rate",
                "delta_hit_rate",
                "baseline_full_sl_rate",
                "guard_full_sl_rate",
                "delta_full_sl_rate",
                "baseline_timeout_rate",
                "guard_timeout_rate",
                "delta_timeout_rate",
                "baseline_objective_week",
                "guard_objective_week",
                "delta_objective_week",
                "baseline_worst_week_net_pnl",
                "guard_worst_week_net_pnl",
                "delta_worst_week_net_pnl",
            ],
        ),
        "",
        "## Schedule",
        "",
        _fmt_table(schedule, ["month_start", "month_end", "train_rows", "month_rows", "threshold", "penalized_rows", "mean_penalty"]),
        "",
        "## Monthly Deltas",
        "",
        _fmt_table(
            monthly,
            [
                "month",
                "baseline_net_pnl",
                "guard_net_pnl",
                "delta_net_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_worst_week_net_pnl",
            ],
        ),
        "",
        "## Per-Head Deltas",
        "",
        _fmt_table(
            per_head,
            [
                "head",
                "baseline_net_pnl",
                "challenger_net_pnl",
                "delta_net_pnl",
                "baseline_trades",
                "challenger_trades",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
            ],
        ),
    ]
    (args.output_dir / "fixed_smooth_rank_penalty_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
