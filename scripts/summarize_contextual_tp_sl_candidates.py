#!/usr/bin/env python3
"""Create a consolidated report for contextual TP/SL carry-forward candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


CANDIDATES = ["longbars_weekgate_only", "longbars_uncertainty_only", "longbars_drift_only"]
ROLE_BY_LABEL = {
    "longbars_weekgate_only": "raw-pnl / recent-HR-surprise challenger",
    "longbars_uncertainty_only": "daily-weekly PnL-tail challenger",
    "longbars_drift_only": "production-pre-OOS / defensive challenger",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values: List[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_gate = _read_csv(root / "promotion_gate" / "promotion_gate_summary.csv")
    daily_gate = _read_csv(root / "daily_weekly_gate" / "daily_weekly_objective_summary.csv")
    weekly_head = _read_csv(root / "ab_report_top3" / "weekly_head_ab_summary.csv")
    weekly_global = _read_csv(root / "ab_report_top3" / "weekly_global_ab_summary.csv")
    monthly_ab = _read_csv(root / "ab_report_top3" / "monthly_ab_summary.csv")

    monthly_gate = monthly_gate[monthly_gate["label"].isin(CANDIDATES)].copy()
    daily_gate = daily_gate[daily_gate["label"].isin(CANDIDATES)].copy()
    weekly_head = weekly_head[weekly_head["label"].isin(CANDIDATES)].copy()
    weekly_global = weekly_global[weekly_global["label"].isin(CANDIDATES)].copy()
    monthly_ab = monthly_ab[monthly_ab["label"].isin(CANDIDATES)].copy()

    comparison = monthly_gate[
        [
            "label",
            "diagnostic_family",
            "monthly_objective",
            "sum_delta_net_pnl",
            "positive_month_count",
            "positive_week_count",
            "mean_month_full_sl_delta",
            "mean_month_drawdown_delta",
            "june_net_delta",
            "june_full_sl_delta",
        ]
    ].merge(
        daily_gate[
            [
                "label",
                "daily_weekly_objective",
                "avg_week_delta_net_pnl",
                "q35_day_delta_net_pnl",
                "q20_day_delta_net_pnl",
                "positive_day_count",
                "mean_day_full_sl_delta",
            ]
        ],
        on="label",
        how="left",
    )
    comparison["role"] = comparison["label"].map(ROLE_BY_LABEL)
    comparison = comparison.sort_values("daily_weekly_objective", ascending=False)

    head_pivot = weekly_head[
        [
            "label",
            "head",
            "sum_delta_net_pnl",
            "positive_period_count",
            "mean_delta_hit_rate",
            "mean_delta_full_sl_rate",
            "sum_delta_trades",
        ]
    ].sort_values(["label", "sum_delta_net_pnl"], ascending=[True, False])

    comparison.to_csv(out_dir / "final_candidate_comparison.csv", index=False)
    head_pivot.to_csv(out_dir / "final_candidate_weekly_head_contributions.csv", index=False)
    weekly_global.to_csv(out_dir / "final_candidate_weekly_global_summary.csv", index=False)
    monthly_ab.to_csv(out_dir / "final_candidate_monthly_summary.csv", index=False)

    md = [
        "# Contextual TP/SL Final Candidate Comparison",
        "",
        "Scope: January-June 2026 development walk-forward replay. This consolidates existing replay summaries and does not rerun portfolio simulation. It is not untouched OOS.",
        "",
        "Baseline: `wf_recent`.",
        "",
        "## Candidate Roles",
        "",
        "- `longbars_weekgate_only`: highest raw PnL and recent-HR-surprise challenger, but weaker recurrence.",
        "- `longbars_uncertainty_only`: best gated candidate under the daily-weekly PnL/tail objective.",
        "- `longbars_drift_only`: most conservative production-pre-OOS and tail-stability candidate.",
        "",
        "## Consolidated Metrics",
        "",
        _markdown_table(
            comparison,
            [
                "label",
                "role",
                "daily_weekly_objective",
                "monthly_objective",
                "sum_delta_net_pnl",
                "positive_month_count",
                "positive_week_count",
                "q20_day_delta_net_pnl",
                "mean_day_full_sl_delta",
                "mean_month_drawdown_delta",
                "june_net_delta",
            ],
        ),
        "",
        "## Weekly Global Summary",
        "",
        _markdown_table(
            weekly_global,
            [
                "label",
                "periods",
                "objective",
                "sum_delta_net_pnl",
                "positive_period_count",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "sum_delta_trades",
            ],
        ),
        "",
        "## Weekly Per-Head Contributions",
        "",
        _markdown_table(
            head_pivot,
            [
                "label",
                "head",
                "sum_delta_net_pnl",
                "positive_period_count",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "sum_delta_trades",
            ],
        ),
        "",
        "## Decision",
        "",
        "Carry forward all three candidates into the next frozen/live-equivalent replay, but with distinct purposes:",
        "",
        "1. Use `longbars_drift_only` as the risk-controlled default challenger.",
        "2. Use `longbars_uncertainty_only` as the daily-weekly PnL-tail challenger.",
        "3. Use `longbars_weekgate_only` as the high-upside research challenger.",
        "",
        "No candidate is promoted from this development replay alone. The next decisive evidence should use a later frozen interval with identical candidates, costs, rank contracts, and portfolio logic.",
        "",
    ]
    (out_dir / "final_candidate_comparison_report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "final_candidate_comparison_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "root_dir": str(root),
                    "candidates": CANDIDATES,
                    "outputs": [
                        "final_candidate_comparison.csv",
                        "final_candidate_weekly_head_contributions.csv",
                        "final_candidate_weekly_global_summary.csv",
                        "final_candidate_monthly_summary.csv",
                        "final_candidate_comparison_report.md",
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_dir / "final_candidate_comparison_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
