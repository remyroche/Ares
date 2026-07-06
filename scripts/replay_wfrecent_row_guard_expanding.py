#!/usr/bin/env python3
"""Continuous replay for a fixed wf_recent row guard with expanding thresholds."""

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
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    VetoRule,
    _apply_risk_scores,
    _apply_veto,
    _fit_percentile_reference,
    _fit_rule_thresholds,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = pd.Timestamp(cur + pd.offsets.MonthBegin(1))
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def _delta_summary(base: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    row = {f"baseline_{k}": v for k, v in base.items() if k != "label"}
    row.update({f"guard_{k}": v for k, v in guard.items() if k != "label"})
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
        row[f"delta_{key}"] = float(row[f"guard_{key}"] - row[f"baseline_{key}"])
    return row


def _monthly_table(base_weekly: pd.DataFrame, guard_weekly: pd.DataFrame) -> pd.DataFrame:
    def prep(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cur = frame[frame["period_type"].eq("week")].copy()
        cur["week_start"] = pd.PeriodIndex(cur["week"], freq="W").start_time
        cur["month"] = cur["week_start"].dt.to_period("M").astype(str)
        out = (
            cur.groupby("month", as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                trades=("trades", "sum"),
                hit_rate=("hit_rate", "mean"),
                full_sl_rate=("full_sl_rate", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                worst_week_net_pnl=("net_pnl", "min"),
            )
        )
        return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "month"})

    out = prep(base_weekly, "baseline").merge(prep(guard_weekly, "guard"), on="month", how="outer")
    for key in ("net_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate", "worst_week_net_pnl"):
        out[f"delta_{key}"] = out[f"guard_{key}"] - out[f"baseline_{key}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_row_guard_expanding_replay_20260701"))
    parser.add_argument("--first-guard-month", default="2026-02-01")
    parser.add_argument("--end", default="2026-06-27")
    parser.add_argument("--score-name", default="recent_perf_risk")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--risk-quantile", type=float, default=0.98)
    parser.add_argument("--min-rank-pct", type=float, default=0.90)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(args.input_dir / "combo_candidates.parquet")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    guarded = candidates.copy()
    if "portfolio_rank_adjustment" not in guarded.columns:
        guarded["portfolio_rank_adjustment"] = 0.0
    else:
        guarded["portfolio_rank_adjustment"] = pd.to_numeric(guarded["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)

    first_guard = pd.Timestamp(args.first_guard_month, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    rule = VetoRule(args.score_name, args.scope, float(args.risk_quantile), float(args.min_rank_pct))
    guard_rows: list[dict[str, Any]] = []
    for start, stop in _month_ranges(first_guard, end):
        train_raw = candidates[candidates["timestamp"].lt(start)].copy().reset_index(drop=True)
        month_raw = candidates[candidates["timestamp"].ge(start) & candidates["timestamp"].lt(stop)].copy().reset_index(drop=True)
        if train_raw.empty or month_raw.empty:
            continue
        refs = _fit_percentile_reference(train_raw)
        train_scored = _apply_risk_scores(train_raw, refs)
        month_scored = _apply_risk_scores(month_raw, refs)
        threshold = _fit_rule_thresholds(train_scored, [rule]).get(rule, float("nan"))
        month_guarded, veto_count = _apply_veto(month_scored, rule, threshold)
        month_idx = candidates.index[candidates["timestamp"].ge(start) & candidates["timestamp"].lt(stop)]
        guarded.loc[month_idx, "portfolio_rank_adjustment"] = pd.to_numeric(
            month_guarded["portfolio_rank_adjustment"], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        guard_rows.append(
            {
                "month_start": start.isoformat(),
                "month_end": stop.isoformat(),
                "train_rows": int(len(train_raw)),
                "month_rows": int(len(month_raw)),
                "threshold": threshold,
                "veto_count": int(veto_count),
            }
        )

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    # Fit the EV reference on pre-guard history only to avoid using future
    # holdout outcomes in the auction priority curve. Both arms share it.
    ev_train = candidates[candidates["timestamp"].lt(first_guard)].copy().reset_index(drop=True)
    ev_curve = fit_hierarchical_ev_curves(ev_train)
    base_decisions, _base_equity, base_metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    guard_decisions, _guard_equity, guard_metrics = replay_candidates(
        guarded,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _base_daily, base_weekly = _period_tables(base_decisions)
    _guard_daily, guard_weekly = _period_tables(guard_decisions)
    base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, args.q35_weight, args.q20_weight)
    guard_summary = _summary("expanding_guard", guard_decisions, guard_weekly, guard_metrics, args.q35_weight, args.q20_weight)
    delta = _delta_summary(base_summary, guard_summary)
    guard_schedule = pd.DataFrame(guard_rows)
    monthly = _monthly_table(base_weekly, guard_weekly)

    pd.DataFrame([delta]).to_csv(args.output_dir / "expanding_row_guard_summary.csv", index=False)
    guard_schedule.to_csv(args.output_dir / "expanding_row_guard_schedule.csv", index=False)
    base_weekly.to_csv(args.output_dir / "expanding_row_guard_baseline_weekly.csv", index=False)
    guard_weekly.to_csv(args.output_dir / "expanding_row_guard_guard_weekly.csv", index=False)
    monthly.to_csv(args.output_dir / "expanding_row_guard_monthly.csv", index=False)
    guard_decisions.to_parquet(args.output_dir / "expanding_row_guard_decisions.parquet", index=False)
    manifest = {
        "generated_by": "replay_wfrecent_row_guard_expanding",
        "input_dir": str(args.input_dir),
        "rule": {
            "score_name": args.score_name,
            "scope": args.scope,
            "risk_quantile": float(args.risk_quantile),
            "min_rank_pct": float(args.min_rank_pct),
        },
        "first_guard_month": args.first_guard_month,
        "end": args.end,
        "ev_curve_fit": "pre_guard_history_only",
        "candidate_rows": int(len(candidates)),
        "veto_count_total": int(guard_schedule["veto_count"].sum()) if not guard_schedule.empty else 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Expanding Row Guard Continuous Replay",
        "",
        "Continuous full-period replay using expanding prior-month diagnostic thresholds. The EV curve is fit once on pre-guard history and shared by baseline and guard.",
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
                "baseline_max_drawdown",
                "guard_max_drawdown",
                "delta_max_drawdown",
                "baseline_objective_week",
                "guard_objective_week",
                "delta_objective_week",
                "baseline_worst_week_net_pnl",
                "guard_worst_week_net_pnl",
                "delta_worst_week_net_pnl",
            ],
        ),
        "",
        "## Guard Schedule",
        "",
        _fmt_table(guard_schedule, ["month_start", "month_end", "train_rows", "month_rows", "threshold", "veto_count"]),
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
    ]
    (args.output_dir / "expanding_row_guard_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
