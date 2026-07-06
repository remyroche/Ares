#!/usr/bin/env python3
"""Summarise shortlisted contextual TP/SL combo candidates.

This report is intentionally downstream of the expensive sweep.  It compares
static, full-window winners, and walk-forward-selected candidates using the
already materialised daily/weekly replay metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


STATIC_COMBO_ID = "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
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


def _combo(summary: pd.DataFrame, combo_id: str) -> pd.Series:
    rows = summary.loc[summary["combo_id"].eq(combo_id)]
    if rows.empty:
        raise KeyError(f"Missing combo_id: {combo_id}")
    return rows.iloc[0]


def _collect_shortlist(
    summary: pd.DataFrame,
    monthly_selected: pd.DataFrame,
) -> pd.DataFrame:
    combo_ids: Dict[str, str] = {
        "static": STATIC_COMBO_ID,
        "full_window_best_balanced": str(
            summary.sort_values("balanced_score", ascending=False).iloc[0]["combo_id"]
        ),
        "full_window_best_net": str(summary.sort_values("net_pnl", ascending=False).iloc[0]["combo_id"]),
        "full_window_best_objective": str(
            summary.sort_values("objective", ascending=False).iloc[0]["combo_id"]
        ),
    }
    for _, row in monthly_selected.iterrows():
        combo_ids[f"wf_objective_{row['split']}"] = str(row["selected_combo_id"])
    rows: List[Dict[str, Any]] = []
    seen = set()
    for label, combo_id in combo_ids.items():
        if combo_id in seen:
            continue
        seen.add(combo_id)
        s = _combo(summary, combo_id)
        rec: Dict[str, Any] = {
            "candidate_label": label,
            "combo_id": combo_id,
        }
        for col in (
            "long_bars_arm",
            "long_dist_arm",
            "short_asset_arm",
            "short_bollinger_arm",
            "net_pnl",
            "gross_pnl",
            "trade_count",
            "mean_net_return",
            "full_sl_rate",
            "timeout_rate",
            "max_drawdown",
            "worst_week_return",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "daily_q35_pnl",
            "weekly_q10_pnl",
            "weekly_min_pnl",
            "objective",
            "balanced_score",
        ):
            rec[col] = s.get(col, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def _monthly_delta_for_shortlist(
    monthly_all: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.DataFrame:
    holdout = monthly_all.loc[monthly_all["period_role"].eq("holdout")].copy()
    static = holdout.loc[holdout["combo_id"].eq(STATIC_COMBO_ID)].copy()
    static = static[
        [
            "split",
            "net_pnl",
            "gross_pnl",
            "trade_count",
            "hit_rate",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "weekly_q10_pnl",
            "max_drawdown_pnl",
            "objective",
            "balanced_score",
        ]
    ].rename(columns={c: f"{c}_static" for c in [
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "daily_q10_pnl",
        "daily_q20_pnl",
        "weekly_q10_pnl",
        "max_drawdown_pnl",
        "objective",
        "balanced_score",
    ]})
    pieces: List[pd.DataFrame] = []
    label_map = shortlist[["candidate_label", "combo_id"]].drop_duplicates()
    for _, cand in label_map.iterrows():
        combo_id = str(cand["combo_id"])
        cur = holdout.loc[holdout["combo_id"].eq(combo_id)].copy()
        if cur.empty:
            continue
        cur.insert(0, "candidate_label", str(cand["candidate_label"]))
        merged = cur.merge(static, on="split", how="left")
        for col in (
            "net_pnl",
            "gross_pnl",
            "trade_count",
            "hit_rate",
            "daily_q10_pnl",
            "daily_q20_pnl",
            "weekly_q10_pnl",
            "max_drawdown_pnl",
            "objective",
            "balanced_score",
        ):
            merged[f"delta_{col}"] = merged[col] - merged[f"{col}_static"]
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _robustness(monthly_delta: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, group in monthly_delta.groupby("candidate_label", sort=False):
        net = pd.to_numeric(group["delta_net_pnl"], errors="coerce")
        rows.append(
            {
                "candidate_label": label,
                "months": int(len(group)),
                "positive_net_months": int((net > 0.0).sum()),
                "sum_delta_net_pnl": float(net.sum()),
                "median_delta_net_pnl": float(net.median()),
                "min_delta_net_pnl": float(net.min()),
                "sum_delta_gross_pnl": float(pd.to_numeric(group["delta_gross_pnl"], errors="coerce").sum()),
                "median_delta_daily_q10_pnl": float(
                    pd.to_numeric(group["delta_daily_q10_pnl"], errors="coerce").median()
                ),
                "median_delta_daily_q20_pnl": float(
                    pd.to_numeric(group["delta_daily_q20_pnl"], errors="coerce").median()
                ),
                "median_delta_weekly_q10_pnl": float(
                    pd.to_numeric(group["delta_weekly_q10_pnl"], errors="coerce").median()
                ),
                "median_delta_max_drawdown_pnl": float(
                    pd.to_numeric(group["delta_max_drawdown_pnl"], errors="coerce").median()
                ),
                "median_delta_objective": float(
                    pd.to_numeric(group["delta_objective"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["positive_net_months", "sum_delta_net_pnl", "median_delta_daily_q10_pnl"],
        ascending=[False, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--monthly-holdout-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.sweep_dir / "head_arm_combination_summary.csv")
    selected = pd.read_csv(args.monthly_holdout_dir / "temporal_holdout_selected_combos.csv")
    monthly_all = pd.read_csv(args.monthly_holdout_dir / "temporal_holdout_all_combo_metrics.csv")
    shortlist = _collect_shortlist(summary, selected)
    monthly_delta = _monthly_delta_for_shortlist(monthly_all, shortlist)
    robust = _robustness(monthly_delta)

    shortlist.to_csv(args.out_dir / "contextual_tp_sl_shortlist_full_period.csv", index=False)
    monthly_delta.to_csv(args.out_dir / "contextual_tp_sl_shortlist_monthly_deltas.csv", index=False)
    robust.to_csv(args.out_dir / "contextual_tp_sl_shortlist_robustness.csv", index=False)

    payload = {
        "sweep_dir": str(args.sweep_dir),
        "monthly_holdout_dir": str(args.monthly_holdout_dir),
        "static_combo_id": STATIC_COMBO_ID,
        "shortlist": shortlist.to_dict(orient="records"),
        "robustness": robust.to_dict(orient="records"),
    }
    (args.out_dir / "contextual_tp_sl_shortlist_report.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Contextual TP/SL Candidate Shortlist",
        "",
        f"Sweep: `{args.sweep_dir}`",
        f"Monthly holdout: `{args.monthly_holdout_dir}`",
        "",
        "## Full-Period Metrics",
        "",
        shortlist.to_markdown(index=False),
        "",
        "## Walk-Forward Monthly Robustness",
        "",
        robust.to_markdown(index=False),
        "",
        "## Monthly Deltas vs Static",
        "",
        monthly_delta[
            [
                "candidate_label",
                "split",
                "combo_id",
                "delta_net_pnl",
                "delta_gross_pnl",
                "delta_trade_count",
                "delta_hit_rate",
                "delta_daily_q10_pnl",
                "delta_daily_q20_pnl",
                "delta_weekly_q10_pnl",
                "delta_max_drawdown_pnl",
                "delta_objective",
            ]
        ].to_markdown(index=False),
    ]
    (args.out_dir / "contextual_tp_sl_shortlist_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "candidates": int(len(shortlist)),
                    "best_robustness": robust.head(1).to_dict(orient="records"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
