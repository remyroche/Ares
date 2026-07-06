#!/usr/bin/env python3
"""Summarize contextual TP/SL candidate readiness across existing artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _weekly_summary(path: Path, baseline_label: str = "static") -> pd.DataFrame:
    weekly = _read(path)
    if weekly.empty:
        return weekly
    rows: list[dict[str, object]] = []
    for label, group in weekly.groupby("label", dropna=False):
        values = pd.to_numeric(group["net_pnl"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        deltas = pd.to_numeric(group.get("delta_net_pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "label": label,
                "weeks": int(values.size),
                "sum_net_pnl": float(np.sum(values)) if values.size else np.nan,
                "avg_week_net_pnl": float(np.mean(values)) if values.size else np.nan,
                "q10_week_net_pnl": float(np.quantile(values, 0.10)) if values.size else np.nan,
                "q20_week_net_pnl": float(np.quantile(values, 0.20)) if values.size else np.nan,
                "q35_week_net_pnl": float(np.quantile(values, 0.35)) if values.size else np.nan,
                "worst_week_net_pnl": float(np.min(values)) if values.size else np.nan,
                "positive_weeks": int(np.sum(values > 0.0)) if values.size else 0,
                "delta_sum_net_pnl": float(np.sum(deltas)) if deltas.size else 0.0,
                "delta_q20_week_net_pnl": float(np.quantile(deltas, 0.20)) if deltas.size else 0.0,
                "delta_worst_week_net_pnl": float(np.min(deltas)) if deltas.size else 0.0,
                "positive_delta_weeks": int(np.sum(deltas > 0.0)) if deltas.size else 0,
            }
        )
    out = pd.DataFrame(rows)
    if "delta_sum_net_pnl" in out.columns:
        out = out.sort_values("delta_sum_net_pnl", ascending=False)
    return out.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_candidate_readiness_20260701"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    materialized_root = Path("data_perp/reports/contextual_tp_sl_materialized_comparison_q35w07_q20w03_6mo_20260701")
    frozen_root = Path("data_perp/reports/contextual_tp_sl_frozen_validation_may03_jun28_q35w06_q20w025_20260701")
    holdout_root = Path("data_perp/reports/contextual_tp_sl_temporal_holdout_monthly_tailgate_with_perf_q35w07_q20w03_20260701")

    dev_global = _read(materialized_root / "materialized_replay_global_comparison.csv")
    dev_week = _weekly_summary(materialized_root / "materialized_replay_week_comparison.csv")
    frozen_global = _read(frozen_root / "comparison/materialized_replay_global_comparison.csv")
    frozen_week = _weekly_summary(frozen_root / "comparison/materialized_replay_week_comparison.csv")
    holdout = _read(holdout_root / "temporal_holdout_comparison.csv")
    selected = _read(holdout_root / "temporal_holdout_selected_combos.csv")

    holdout_delta = holdout[holdout["variant"].eq("delta_selected_minus_static")].copy() if not holdout.empty else pd.DataFrame()
    if not holdout_delta.empty:
        holdout_delta["delta_net_pnl"] = pd.to_numeric(holdout_delta["net_pnl"], errors="coerce")
        holdout_delta["delta_objective"] = pd.to_numeric(holdout_delta["objective"], errors="coerce")
        holdout_delta["delta_weekly_min_pnl"] = pd.to_numeric(holdout_delta["weekly_min_pnl"], errors="coerce")
        holdout_summary = pd.DataFrame(
            [
                {
                    "splits": int(len(holdout_delta)),
                    "sum_delta_net_pnl": float(holdout_delta["delta_net_pnl"].sum()),
                    "median_delta_net_pnl": float(holdout_delta["delta_net_pnl"].median()),
                    "positive_split_share": float((holdout_delta["delta_net_pnl"] > 0).mean()),
                    "worst_split_delta_net_pnl": float(holdout_delta["delta_net_pnl"].min()),
                    "sum_delta_objective": float(holdout_delta["delta_objective"].sum()),
                    "median_delta_objective": float(holdout_delta["delta_objective"].median()),
                    "positive_objective_share": float((holdout_delta["delta_objective"] > 0).mean()),
                }
            ]
        )
    else:
        holdout_summary = pd.DataFrame()

    for name, frame in {
        "development_global.csv": dev_global,
        "development_week_summary.csv": dev_week,
        "frozen_may03_jun28_global.csv": frozen_global,
        "frozen_may03_jun28_week_summary.csv": frozen_week,
        "temporal_holdout_selected_combos.csv": selected,
        "temporal_holdout_delta_selected_minus_static.csv": holdout_delta,
        "temporal_holdout_summary.csv": holdout_summary,
    }.items():
        if not frame.empty:
            frame.to_csv(args.output_dir / name, index=False)

    lines = [
        "# Contextual TP/SL Candidate Readiness",
        "",
        "This report consolidates existing artifacts only. It does not rerun portfolio replay and should be read as a current-state audit, not untouched OOS proof.",
        "",
        "## Development Materialized Replay",
        "",
        "Source: `contextual_tp_sl_materialized_comparison_q35w07_q20w03_6mo_20260701`.",
        "",
        _fmt_table(
            dev_global,
            [
                "label",
                "combo_id",
                "net_pnl",
                "delta_net_pnl",
                "trade_count",
                "delta_trade_count",
                "full_sl_rate",
                "delta_full_sl_rate",
                "max_drawdown",
                "delta_max_drawdown",
            ],
        ),
        "",
        "Development weekly summary:",
        "",
        _fmt_table(
            dev_week,
            [
                "label",
                "weeks",
                "sum_net_pnl",
                "q20_week_net_pnl",
                "q35_week_net_pnl",
                "worst_week_net_pnl",
                "delta_sum_net_pnl",
                "delta_q20_week_net_pnl",
                "delta_worst_week_net_pnl",
                "positive_delta_weeks",
            ],
        ),
        "",
        "## Frozen May 3-June 28 Window",
        "",
        "Source: `contextual_tp_sl_frozen_validation_may03_jun28_q35w06_q20w025_20260701`. Costs are included. This is development/frozen validation, not untouched live OOS.",
        "",
        _fmt_table(
            frozen_global,
            [
                "label",
                "combo_id",
                "net_pnl",
                "delta_net_pnl",
                "trade_count",
                "delta_trade_count",
                "full_sl_rate",
                "delta_full_sl_rate",
                "timeout_rate",
                "delta_timeout_rate",
                "max_drawdown",
                "delta_max_drawdown",
            ],
        ),
        "",
        "Frozen weekly summary:",
        "",
        _fmt_table(
            frozen_week,
            [
                "label",
                "weeks",
                "sum_net_pnl",
                "q20_week_net_pnl",
                "q35_week_net_pnl",
                "worst_week_net_pnl",
                "delta_sum_net_pnl",
                "delta_q20_week_net_pnl",
                "delta_worst_week_net_pnl",
                "positive_delta_weeks",
            ],
        ),
        "",
        "## Monthly Temporal Holdout",
        "",
        "Source: `contextual_tp_sl_temporal_holdout_monthly_tailgate_with_perf_q35w07_q20w03_20260701`. Each selected combo is chosen from prior months and evaluated on the next month.",
        "",
        "Selected combo per holdout:",
        "",
        _fmt_table(
            selected,
            [
                "split",
                "train_end",
                "holdout_start",
                "holdout_end",
                "selected_combo_id",
                "selected_train_delta_net_pnl",
                "selected_train_delta_week_q20_pnl",
                "selected_train_positive_week_delta_share",
            ],
        ),
        "",
        "Selected-minus-static holdout deltas:",
        "",
        _fmt_table(
            holdout_delta,
            [
                "split",
                "holdout_start",
                "holdout_end",
                "combo_id",
                "delta_net_pnl",
                "hit_rate",
                "daily_q20_pnl",
                "daily_q35_pnl",
                "weekly_min_pnl",
                "max_drawdown_pnl",
                "delta_objective",
            ],
        ),
        "",
        "Holdout summary:",
        "",
        _fmt_table(
            holdout_summary,
            [
                "splits",
                "sum_delta_net_pnl",
                "median_delta_net_pnl",
                "positive_split_share",
                "worst_split_delta_net_pnl",
                "sum_delta_objective",
                "median_delta_objective",
                "positive_objective_share",
            ],
        ),
        "",
        "## Readout",
        "",
        "- Development full-window materialized candidates are strong, especially `wf_recent` and `best_balanced`.",
        "- The May 3-June 28 frozen window favors `best_balanced` on net PnL and max drawdown, but it worsens full-SL and timeout rates versus static.",
        "- Monthly temporal holdout is mixed: February, March, May, and June selected-minus-static improve net PnL; April deteriorates materially.",
        "- The candidate family is promising enough for continued ablation, but not cleanly promotable until the April-style degradation is controlled.",
        "- Next work should target a conservative fallback/default-to-static rule around the temporal selector, using only prior-month evidence and explicit tail-loss vetoes.",
    ]
    report = args.output_dir / "candidate_readiness_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(report)


if __name__ == "__main__":
    main()
