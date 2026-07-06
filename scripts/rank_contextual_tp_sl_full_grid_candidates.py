#!/usr/bin/env python3
"""Rank contextual TP/SL head-arm combinations with tail-aware gates.

This consumes the full combination sweep outputs from
`sweep_contextual_tp_sl_arm_combinations.py` and adds promotion-style deltas
against the static baseline.  It is intentionally lightweight: it does not
replay trades, it ranks already-computed replay summaries and weekly PnL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


HEAD_ARM_COLS = ["long_bars_arm", "long_dist_arm", "short_asset_arm", "short_bollinger_arm"]
STATIC_COMBO = "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S"


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


def _read_inputs(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = source_dir / "head_arm_combination_summary.csv"
    weekly_path = source_dir / "head_arm_combination_weekly.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly metrics: {weekly_path}")
    return pd.read_csv(summary_path), pd.read_csv(weekly_path)


def _weekly_delta_metrics(weekly: pd.DataFrame, baseline_combo: str) -> pd.DataFrame:
    base = weekly.loc[weekly["combo_id"].eq(baseline_combo), ["week", "net_pnl"]].rename(
        columns={"net_pnl": "baseline_week_net_pnl"}
    )
    if base.empty:
        raise ValueError(f"Baseline combo {baseline_combo!r} not found in weekly metrics")
    merged = weekly.merge(base, on="week", how="left")
    merged["delta_week_net_pnl"] = merged["net_pnl"] - merged["baseline_week_net_pnl"]
    rows: List[Dict[str, Any]] = []
    for combo_id, g in merged.groupby("combo_id", sort=False):
        d = pd.to_numeric(g["delta_week_net_pnl"], errors="coerce").dropna()
        rows.append(
            {
                "combo_id": combo_id,
                "delta_week_count": int(len(d)),
                "delta_week_sum_pnl": float(d.sum()) if len(d) else np.nan,
                "delta_week_mean_pnl": float(d.mean()) if len(d) else np.nan,
                "delta_week_q05_pnl": float(d.quantile(0.05)) if len(d) else np.nan,
                "delta_week_q10_pnl": float(d.quantile(0.10)) if len(d) else np.nan,
                "delta_week_q20_pnl": float(d.quantile(0.20)) if len(d) else np.nan,
                "delta_week_q50_pnl": float(d.quantile(0.50)) if len(d) else np.nan,
                "delta_worst_week_pnl": float(d.min()) if len(d) else np.nan,
                "positive_week_delta_share": float((d > 0.0).mean()) if len(d) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _add_baseline_deltas(summary: pd.DataFrame, baseline_combo: str) -> pd.DataFrame:
    base = summary.loc[summary["combo_id"].eq(baseline_combo)]
    if base.empty:
        raise ValueError(f"Baseline combo {baseline_combo!r} not found in summary")
    base_row = base.iloc[0]
    out = summary.copy()
    for col in ("net_pnl", "gross_pnl", "trade_count", "full_sl_rate", "timeout_rate", "max_drawdown"):
        if col in out.columns:
            out[f"delta_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(base_row[col])
    return out


def _add_gates(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = frame.copy()
    out["uses_performance_arm"] = out[HEAD_ARM_COLS].eq("performance_only").any(axis=1)
    out["uses_joint_or_independent_arm"] = out[HEAD_ARM_COLS].isin(["joint_all", "independent_all"]).any(axis=1)
    out["coverage_pass"] = (
        (pd.to_numeric(out["candidate_rows"], errors="coerce") >= int(args.min_candidate_rows))
        & (pd.to_numeric(out["trade_count"], errors="coerce") >= int(args.min_trade_count))
        & (pd.to_numeric(out["weekly_count"], errors="coerce") >= int(args.min_weeks))
    )
    out["economics_tail_pass"] = (
        (pd.to_numeric(out["delta_net_pnl"], errors="coerce") >= float(args.min_delta_net_pnl))
        & (pd.to_numeric(out["delta_full_sl_rate"], errors="coerce") <= float(args.max_delta_full_sl_rate))
        & (pd.to_numeric(out["delta_max_drawdown"], errors="coerce") >= float(args.min_delta_max_drawdown))
        & (pd.to_numeric(out["delta_week_q10_pnl"], errors="coerce") >= float(args.min_delta_week_q10_pnl))
        & (pd.to_numeric(out["delta_week_q20_pnl"], errors="coerce") >= float(args.min_delta_week_q20_pnl))
        & (
            pd.to_numeric(out["positive_week_delta_share"], errors="coerce")
            >= float(args.min_positive_week_delta_share)
        )
    )
    out["gate_pass"] = out["coverage_pass"] & out["economics_tail_pass"]
    out["tail_adjusted_score"] = (
        pd.to_numeric(out["delta_net_pnl"], errors="coerce").fillna(-1.0e18)
        + 0.7 * pd.to_numeric(out["delta_week_q20_pnl"], errors="coerce").fillna(-1.0e6)
        + 0.3 * pd.to_numeric(out["delta_week_q10_pnl"], errors="coerce").fillna(-1.0e6)
    )
    return out


def _write_report(out_dir: Path, ranked: pd.DataFrame, args: argparse.Namespace) -> None:
    columns = [
        "combo_id",
        *HEAD_ARM_COLS,
        "net_pnl",
        "delta_net_pnl",
        "trade_count",
        "full_sl_rate",
        "delta_full_sl_rate",
        "max_drawdown",
        "delta_max_drawdown",
        "delta_week_q10_pnl",
        "delta_week_q20_pnl",
        "delta_worst_week_pnl",
        "positive_week_delta_share",
        "uses_performance_arm",
        "gate_pass",
        "tail_adjusted_score",
    ]
    existing = [c for c in columns if c in ranked.columns]
    gate_pass = ranked.loc[ranked["gate_pass"]].sort_values(
        ["tail_adjusted_score", "delta_net_pnl"], ascending=[False, False]
    )
    top_all = ranked.sort_values(["tail_adjusted_score", "delta_net_pnl"], ascending=[False, False]).head(25)
    top_perf = ranked.loc[ranked["uses_performance_arm"]].sort_values(
        ["tail_adjusted_score", "delta_net_pnl"], ascending=[False, False]
    ).head(25)
    lines = [
        "# Contextual TP/SL Full-Grid Candidate Ranking",
        "",
        f"Source directory: `{args.source_dir}`",
        f"Baseline combo: `{args.baseline_combo}`",
        f"Rows evaluated: `{len(ranked)}`",
        f"Gate-pass combos: `{int(ranked['gate_pass'].sum())}`",
        f"Performance-arm combos: `{int(ranked['uses_performance_arm'].sum())}`",
        f"Performance-arm gate-pass combos: `{int((ranked['uses_performance_arm'] & ranked['gate_pass']).sum())}`",
        "",
        "## Gate Thresholds",
        "",
        pd.DataFrame(
            [
                {
                    "min_candidate_rows": args.min_candidate_rows,
                    "min_trade_count": args.min_trade_count,
                    "min_weeks": args.min_weeks,
                    "min_delta_net_pnl": args.min_delta_net_pnl,
                    "max_delta_full_sl_rate": args.max_delta_full_sl_rate,
                    "min_delta_max_drawdown": args.min_delta_max_drawdown,
                    "min_delta_week_q10_pnl": args.min_delta_week_q10_pnl,
                    "min_delta_week_q20_pnl": args.min_delta_week_q20_pnl,
                    "min_positive_week_delta_share": args.min_positive_week_delta_share,
                }
            ]
        ).to_markdown(index=False),
        "",
        "## Top Gate-Pass Combos",
        "",
        gate_pass[existing].head(25).to_markdown(index=False) if not gate_pass.empty else "_None._",
        "",
        "## Top Tail-Adjusted Combos",
        "",
        top_all[existing].to_markdown(index=False),
        "",
        "## Top Combos Using Performance Arm",
        "",
        top_perf[existing].to_markdown(index=False) if not top_perf.empty else "_None._",
    ]
    (out_dir / "full_grid_candidate_ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-combo", default=STATIC_COMBO)
    parser.add_argument("--min-candidate-rows", type=int, default=90000)
    parser.add_argument("--min-trade-count", type=int, default=5000)
    parser.add_argument("--min-weeks", type=int, default=20)
    parser.add_argument("--min-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--max-delta-full-sl-rate", type=float, default=0.0)
    parser.add_argument("--min-delta-max-drawdown", type=float, default=0.0)
    parser.add_argument("--min-delta-week-q10-pnl", type=float, default=-1000.0)
    parser.add_argument("--min-delta-week-q20-pnl", type=float, default=0.0)
    parser.add_argument("--min-positive-week-delta-share", type=float, default=0.60)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, weekly = _read_inputs(args.source_dir)
    weekly_deltas = _weekly_delta_metrics(weekly, str(args.baseline_combo))
    ranked = _add_baseline_deltas(summary, str(args.baseline_combo)).merge(weekly_deltas, on="combo_id", how="left")
    ranked = _add_gates(ranked, args)
    ranked = ranked.sort_values(["gate_pass", "tail_adjusted_score", "delta_net_pnl"], ascending=[False, False, False])
    ranked.to_csv(args.out_dir / "full_grid_candidate_ranking.csv", index=False)
    payload: Dict[str, Any] = {
        "generated_by": "rank_contextual_tp_sl_full_grid_candidates",
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "baseline_combo": str(args.baseline_combo),
        "row_count": int(len(ranked)),
        "gate_pass_count": int(ranked["gate_pass"].sum()),
        "performance_arm_count": int(ranked["uses_performance_arm"].sum()),
        "performance_arm_gate_pass_count": int((ranked["uses_performance_arm"] & ranked["gate_pass"]).sum()),
        "top_gate_pass_combo": ranked.loc[ranked["gate_pass"], "combo_id"].iloc[0]
        if bool(ranked["gate_pass"].any())
        else None,
    }
    (args.out_dir / "full_grid_candidate_ranking_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    _write_report(args.out_dir, ranked, args)
    print(json.dumps(_json_safe(payload), indent=2))


if __name__ == "__main__":
    main()
