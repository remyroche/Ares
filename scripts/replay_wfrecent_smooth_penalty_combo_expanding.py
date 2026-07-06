#!/usr/bin/env python3
"""Continuous replay for bounded wf_recent smooth-penalty combos with expanding references."""

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
from scripts.ablate_wfrecent_smooth_rank_penalty import _fit_threshold, _penalty_values  # noqa: E402
from scripts.replay_wfrecent_row_guard_expanding import _delta_summary, _month_ranges, _monthly_table  # noqa: E402
from scripts.replay_wfrecent_smooth_rank_penalty_fixed import _per_head_table  # noqa: E402
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)
from scripts.validate_wfrecent_smooth_penalty_combo_holdout import Combo, RULE_LIBRARY, _parse_combos  # noqa: E402


def _load_candidates(input_dir: Path) -> pd.DataFrame:
    candidate_path = input_dir if input_dir.is_file() else input_dir / "combo_candidates.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(
        ["timestamp", "strategy_id", "symbol"]
    ).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in candidates.columns:
        candidates["portfolio_rank_adjustment"] = np.float32(0.0)
    else:
        candidates["portfolio_rank_adjustment"] = (
            pd.to_numeric(candidates["portfolio_rank_adjustment"], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )
    return candidates


def _apply_combo_one_month(
    month_scored: pd.DataFrame,
    train_scored: pd.DataFrame,
    combo: Combo,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    total = np.zeros(len(month_scored), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for leg in combo.legs:
        if leg.rule_name not in RULE_LIBRARY:
            raise ValueError(f"Unknown combo rule: {leg.rule_name}")
        rule = RULE_LIBRARY[leg.rule_name]
        threshold = _fit_threshold(train_scored, rule)
        penalty = _penalty_values(month_scored, rule, threshold).astype(np.float32) * float(leg.weight)
        total += penalty
        mask = penalty < 0.0
        rows.append(
            {
                "combo": combo.label,
                "rule_name": leg.rule_name,
                "weight": float(leg.weight),
                "threshold": float(threshold),
                "raw_penalized_rows": int(np.sum(mask)),
                "raw_penalized_share": float(np.mean(mask)) if len(mask) else 0.0,
                "raw_mean_penalty": float(np.mean(penalty[mask])) if np.any(mask) else 0.0,
                "raw_min_penalty": float(np.min(penalty[mask])) if np.any(mask) else 0.0,
            }
        )
    capped = np.clip(total, -float(combo.total_cap), 0.0).astype(np.float32)
    mask = capped < 0.0
    rows.append(
        {
            "combo": combo.label,
            "rule_name": "__combined__",
            "weight": 1.0,
            "threshold": np.nan,
            "raw_penalized_rows": int(np.sum(mask)),
            "raw_penalized_share": float(np.mean(mask)) if len(mask) else 0.0,
            "raw_mean_penalty": float(np.mean(capped[mask])) if np.any(mask) else 0.0,
            "raw_min_penalty": float(np.min(capped[mask])) if np.any(mask) else 0.0,
            "total_cap": float(combo.total_cap),
        }
    )
    return capped, rows


def _apply_combo_expanding(
    candidates: pd.DataFrame,
    combo: Combo,
    first_guard: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = candidates.copy()
    base_adj = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    adjustment = base_adj.copy()
    schedule_rows: list[dict[str, Any]] = []

    for start, stop in _month_ranges(first_guard, end):
        train_raw = candidates[candidates["timestamp"].lt(start)].copy().reset_index(drop=True)
        month_mask = candidates["timestamp"].ge(start) & candidates["timestamp"].lt(stop)
        month_raw = candidates[month_mask].copy().reset_index(drop=True)
        if train_raw.empty or month_raw.empty:
            continue
        refs = _fit_percentile_reference(train_raw)
        train_scored = _apply_risk_scores(train_raw, refs)
        month_scored = _apply_risk_scores(month_raw, refs)
        penalty, rows = _apply_combo_one_month(month_scored, train_scored, combo)
        idx = candidates.index[month_mask].to_numpy(dtype=np.int64)
        adjustment[idx] = np.clip(adjustment[idx] + penalty, -1.0, 1.0)
        for row in rows:
            schedule_rows.append(
                {
                    "month_start": start.isoformat(),
                    "month_end": stop.isoformat(),
                    "train_rows": int(len(train_raw)),
                    "month_rows": int(len(month_raw)),
                    **row,
                }
            )

    out["portfolio_rank_adjustment"] = adjustment.astype("float32")
    out["smooth_penalty_variant"] = combo.label
    out["smooth_penalty_components"] = json.dumps([leg.__dict__ for leg in combo.legs], sort_keys=True)
    return out, pd.DataFrame(schedule_rows)


def _summary_delta(base_summary: dict[str, Any], challenger_summary: dict[str, Any]) -> dict[str, Any]:
    delta = _delta_summary(base_summary, challenger_summary)
    return {
        key.replace("guard_", "challenger_"): value
        for key, value in delta.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_expanding_20260701"),
    )
    parser.add_argument("--first-guard-month", default="2026-02-01")
    parser.add_argument("--end", default="2026-06-27")
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument("--combos", default="")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_guard = pd.Timestamp(args.first_guard_month, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    candidates = _load_candidates(args.input_dir)
    combos = _parse_combos(args.combos)

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
    _base_daily, base_weekly = _period_tables(base_decisions)
    base_summary = _summary("baseline", base_decisions, base_weekly, base_metrics, args.q35_weight, args.q20_weight)

    summary_rows: list[dict[str, Any]] = []
    schedule_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    per_head_rows: list[pd.DataFrame] = []
    for combo in combos:
        challenger, schedule = _apply_combo_expanding(candidates, combo, first_guard, end)
        decisions, _equity, metrics = replay_candidates(
            challenger,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        _daily, weekly = _period_tables(decisions)
        challenger_summary = _summary(combo.label, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
        row = _summary_delta(base_summary, challenger_summary)
        row["variant"] = combo.label
        row["total_cap"] = float(combo.total_cap)
        row["legs"] = ",".join(f"{leg.weight:g}*{leg.rule_name}" for leg in combo.legs)
        combined = schedule[schedule["rule_name"].eq("__combined__")]
        if not combined.empty:
            row["penalized_rows"] = int(combined["raw_penalized_rows"].sum())
            row["penalized_share_month_mean"] = float(combined["raw_penalized_share"].mean())
            penalized = combined["raw_penalized_rows"].sum()
            row["mean_penalty_weighted"] = (
                float((combined["raw_mean_penalty"] * combined["raw_penalized_rows"]).sum() / penalized)
                if penalized
                else 0.0
            )
        summary_rows.append(row)
        schedule_rows.append(schedule)
        cur_weekly = weekly.copy()
        cur_weekly["variant"] = combo.label
        weekly_rows.append(cur_weekly)
        monthly = _monthly_table(base_weekly, weekly)
        monthly["variant"] = combo.label
        monthly_rows.append(monthly)
        per_head = _per_head_table(base_decisions, decisions)
        per_head["variant"] = combo.label
        per_head_rows.append(per_head)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["delta_objective_week", "delta_net_pnl"],
        ascending=[False, False],
    )
    schedule_out = pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()
    weekly_out = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    monthly_out = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    per_head_out = pd.concat(per_head_rows, ignore_index=True) if per_head_rows else pd.DataFrame()

    summary.to_csv(args.output_dir / "combo_expanding_summary.csv", index=False)
    schedule_out.to_csv(args.output_dir / "combo_expanding_schedule.csv", index=False)
    base_weekly.to_csv(args.output_dir / "combo_expanding_baseline_weekly.csv", index=False)
    weekly_out.to_csv(args.output_dir / "combo_expanding_weekly.csv", index=False)
    monthly_out.to_csv(args.output_dir / "combo_expanding_monthly.csv", index=False)
    per_head_out.to_csv(args.output_dir / "combo_expanding_per_head.csv", index=False)

    manifest = {
        "generated_by": "replay_wfrecent_smooth_penalty_combo_expanding",
        "input_dir": str(args.input_dir),
        "first_guard_month": args.first_guard_month,
        "end": args.end,
        "candidate_rows": int(len(candidates)),
        "ev_curve_fit": "pre_guard_history_only",
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "combos": [
            {"label": combo.label, "total_cap": combo.total_cap, "legs": [leg.__dict__ for leg in combo.legs]}
            for combo in combos
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Smooth-Penalty Combo Expanding Replay",
        "",
        "Continuous replay of bounded smooth-penalty combos. Diagnostic references and thresholds are refit using only prior-month history for each replay month. The EV curve is fit on pre-guard history and shared across all arms.",
        "",
        f"First guard month: `{first_guard.isoformat()}`",
        f"End: `{end.isoformat()}`",
        "",
        "## Summary",
        "",
        _fmt_table(
            summary,
            [
                "variant",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_q35_week_net_pnl",
                "delta_q20_week_net_pnl",
                "delta_worst_week_net_pnl",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_trade_count",
                "penalized_rows",
                "penalized_share_month_mean",
                "mean_penalty_weighted",
            ],
        ),
        "",
        "## Monthly Deltas",
        "",
        _fmt_table(
            monthly_out,
            [
                "variant",
                "month",
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
            per_head_out,
            [
                "variant",
                "head",
                "delta_net_pnl",
                "delta_gross_pnl",
                "delta_trades",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
            ],
        ),
    ]
    (args.output_dir / "combo_expanding_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
