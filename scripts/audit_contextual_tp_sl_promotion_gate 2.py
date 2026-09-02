#!/usr/bin/env python3
"""Audit whether contextual TP/SL replay candidates satisfy promotion gates.

This script reads a directory produced by
`compare_materialized_contextual_tp_sl_replays.py` and applies explicit
coverage and economics/tail gates.  It intentionally separates "coverage is
large enough" from "this is a forward/OOS validation", because a long in-sample
or development replay can be useful evidence without being promotion proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def _load_inputs(comparison_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_path = comparison_dir / "materialized_replay_global_comparison.csv"
    head_path = comparison_dir / "materialized_replay_head_comparison.csv"
    week_path = comparison_dir / "materialized_replay_week_comparison.csv"
    missing = [str(p) for p in (global_path, head_path, week_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing comparison files: {missing}")
    return pd.read_csv(global_path), pd.read_csv(head_path), pd.read_csv(week_path)


def _candidate_gate_rows(
    global_df: pd.DataFrame,
    head_df: pd.DataFrame,
    week_df: pd.DataFrame,
    *,
    baseline_label: str,
    validation_role: str,
    min_candidate_rows: int,
    min_trade_count: int,
    min_weeks: int,
    min_active_heads: int,
    min_positive_week_share: float,
    min_positive_week_delta_share: float,
    min_delta_net_pnl: float,
    max_delta_full_sl_rate: float,
    min_delta_max_drawdown: float,
    min_delta_week_q10_pnl: float,
    min_delta_week_q20_pnl: float,
    min_delta_worst_week_pnl: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, rec in global_df.iterrows():
        label = str(rec.get("label"))
        if label == baseline_label:
            continue
        candidate_rows = int(_num(rec.get("candidate_rows"), 0.0))
        trade_count = int(_num(rec.get("trade_count"), 0.0))
        cur_weeks = week_df.loc[week_df["label"].astype(str).eq(label)].copy()
        cur_heads = head_df.loc[head_df["label"].astype(str).eq(label)].copy()
        week_count = int(cur_weeks["week"].nunique()) if "week" in cur_weeks.columns else 0
        active_heads = int((pd.to_numeric(cur_heads.get("trades", 0), errors="coerce").fillna(0) > 0).sum())
        positive_week_share = 0.0
        positive_week_delta_share = 0.0
        delta_week_q10 = 0.0
        delta_week_q20 = 0.0
        delta_worst_week = 0.0
        if not cur_weeks.empty and "net_pnl" in cur_weeks.columns:
            positive_week_share = float((pd.to_numeric(cur_weeks["net_pnl"], errors="coerce") > 0.0).mean())
        if not cur_weeks.empty and "delta_net_pnl" in cur_weeks.columns:
            week_delta = pd.to_numeric(cur_weeks["delta_net_pnl"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if not week_delta.empty:
                arr = week_delta.to_numpy(dtype=float)
                positive_week_delta_share = float(np.mean(arr > 0.0))
                delta_week_q10 = float(np.nanpercentile(arr, 10))
                delta_week_q20 = float(np.nanpercentile(arr, 20))
                delta_worst_week = float(np.nanmin(arr))
        delta_net = _num(rec.get("delta_net_pnl"))
        delta_full_sl = _num(rec.get("delta_full_sl_rate"))
        delta_dd = _num(rec.get("delta_max_drawdown"))

        coverage_pass = (
            candidate_rows >= min_candidate_rows
            and trade_count >= min_trade_count
            and week_count >= min_weeks
            and active_heads >= min_active_heads
        )
        economics_pass = (
            delta_net >= min_delta_net_pnl
            and delta_full_sl <= max_delta_full_sl_rate
            and delta_dd >= min_delta_max_drawdown
            and positive_week_share >= min_positive_week_share
            and positive_week_delta_share >= min_positive_week_delta_share
            and delta_week_q10 >= min_delta_week_q10_pnl
            and delta_week_q20 >= min_delta_week_q20_pnl
            and delta_worst_week >= min_delta_worst_week_pnl
        )
        forward_role_pass = validation_role in {"forward", "oos", "shadow", "prospective"}
        coverage_fail_reasons = [
            reason
            for reason, failed in (
                ("candidate_rows", candidate_rows < min_candidate_rows),
                ("trade_count", trade_count < min_trade_count),
                ("week_count", week_count < min_weeks),
                ("active_heads", active_heads < min_active_heads),
            )
            if failed
        ]
        economics_fail_reasons = [
            reason
            for reason, failed in (
                ("delta_net_pnl", delta_net < min_delta_net_pnl),
                ("delta_full_sl_rate", delta_full_sl > max_delta_full_sl_rate),
                ("delta_max_drawdown", delta_dd < min_delta_max_drawdown),
                ("positive_week_share", positive_week_share < min_positive_week_share),
                ("positive_week_delta_share", positive_week_delta_share < min_positive_week_delta_share),
                ("delta_week_q10_pnl", delta_week_q10 < min_delta_week_q10_pnl),
                ("delta_week_q20_pnl", delta_week_q20 < min_delta_week_q20_pnl),
                ("delta_worst_week_pnl", delta_worst_week < min_delta_worst_week_pnl),
            )
            if failed
        ]
        rows.append(
            {
                "label": label,
                "combo_id": rec.get("combo_id"),
                "validation_role": validation_role,
                "candidate_rows": candidate_rows,
                "trade_count": trade_count,
                "week_count": week_count,
                "active_heads": active_heads,
                "positive_week_share": positive_week_share,
                "positive_week_delta_share": positive_week_delta_share,
                "delta_net_pnl": delta_net,
                "delta_full_sl_rate": delta_full_sl,
                "delta_max_drawdown": delta_dd,
                "delta_week_q10_pnl": delta_week_q10,
                "delta_week_q20_pnl": delta_week_q20,
                "delta_worst_week_pnl": delta_worst_week,
                "coverage_pass": coverage_pass,
                "economics_tail_pass": economics_pass,
                "forward_role_pass": forward_role_pass,
                "promotion_ready": bool(coverage_pass and economics_pass and forward_role_pass),
                "coverage_fail_reasons": ",".join(coverage_fail_reasons) if coverage_fail_reasons else "none",
                "economics_fail_reasons": ",".join(economics_fail_reasons) if economics_fail_reasons else "none",
            }
        )
    return pd.DataFrame(rows)


def _write_report(out_dir: Path, gate: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Contextual TP/SL Promotion Gate Audit",
        "",
        f"Comparison directory: `{args.comparison_dir}`",
        f"Validation role: `{args.validation_role}`",
        f"Baseline label: `{args.baseline_label}`",
        "",
        "## Gates",
        "",
        "| gate | value |",
        "|---|---:|",
        f"| min_candidate_rows | {args.min_candidate_rows} |",
        f"| min_trade_count | {args.min_trade_count} |",
        f"| min_weeks | {args.min_weeks} |",
        f"| min_active_heads | {args.min_active_heads} |",
        f"| min_positive_week_share | {args.min_positive_week_share} |",
        f"| min_positive_week_delta_share | {args.min_positive_week_delta_share} |",
        f"| min_delta_net_pnl | {args.min_delta_net_pnl} |",
        f"| max_delta_full_sl_rate | {args.max_delta_full_sl_rate} |",
        f"| min_delta_max_drawdown | {args.min_delta_max_drawdown} |",
        f"| min_delta_week_q10_pnl | {args.min_delta_week_q10_pnl} |",
        f"| min_delta_week_q20_pnl | {args.min_delta_week_q20_pnl} |",
        f"| min_delta_worst_week_pnl | {args.min_delta_worst_week_pnl} |",
        "",
        "## Candidate Results",
        "",
        gate.to_markdown(index=False),
    ]
    (out_dir / "contextual_tp_sl_promotion_gate_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-label", default="static")
    parser.add_argument(
        "--validation-role",
        default="development",
        help="development, forward, oos, shadow, or prospective. Only forward-like roles can pass promotion_ready.",
    )
    parser.add_argument("--min-candidate-rows", type=int, default=1000)
    parser.add_argument("--min-trade-count", type=int, default=500)
    parser.add_argument("--min-weeks", type=int, default=4)
    parser.add_argument("--min-active-heads", type=int, default=3)
    parser.add_argument("--min-positive-week-share", type=float, default=0.60)
    parser.add_argument("--min-positive-week-delta-share", type=float, default=0.60)
    parser.add_argument("--min-delta-net-pnl", type=float, default=0.0)
    parser.add_argument("--max-delta-full-sl-rate", type=float, default=0.0)
    parser.add_argument("--min-delta-max-drawdown", type=float, default=0.0)
    parser.add_argument("--min-delta-week-q10-pnl", type=float, default=-1000.0)
    parser.add_argument("--min-delta-week-q20-pnl", type=float, default=0.0)
    parser.add_argument("--min-delta-worst-week-pnl", type=float, default=-1.0e18)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    global_df, head_df, week_df = _load_inputs(args.comparison_dir)
    gate = _candidate_gate_rows(
        global_df,
        head_df,
        week_df,
        baseline_label=str(args.baseline_label),
        validation_role=str(args.validation_role),
        min_candidate_rows=int(args.min_candidate_rows),
        min_trade_count=int(args.min_trade_count),
        min_weeks=int(args.min_weeks),
        min_active_heads=int(args.min_active_heads),
        min_positive_week_share=float(args.min_positive_week_share),
        min_positive_week_delta_share=float(args.min_positive_week_delta_share),
        min_delta_net_pnl=float(args.min_delta_net_pnl),
        max_delta_full_sl_rate=float(args.max_delta_full_sl_rate),
        min_delta_max_drawdown=float(args.min_delta_max_drawdown),
        min_delta_week_q10_pnl=float(args.min_delta_week_q10_pnl),
        min_delta_week_q20_pnl=float(args.min_delta_week_q20_pnl),
        min_delta_worst_week_pnl=float(args.min_delta_worst_week_pnl),
    )
    gate.to_csv(args.out_dir / "contextual_tp_sl_promotion_gate.csv", index=False)
    payload: Dict[str, Any] = {
        "comparison_dir": str(args.comparison_dir),
        "validation_role": str(args.validation_role),
        "baseline_label": str(args.baseline_label),
        "gates": {
            "min_candidate_rows": int(args.min_candidate_rows),
            "min_trade_count": int(args.min_trade_count),
            "min_weeks": int(args.min_weeks),
            "min_active_heads": int(args.min_active_heads),
            "min_positive_week_share": float(args.min_positive_week_share),
            "min_positive_week_delta_share": float(args.min_positive_week_delta_share),
            "min_delta_net_pnl": float(args.min_delta_net_pnl),
            "max_delta_full_sl_rate": float(args.max_delta_full_sl_rate),
            "min_delta_max_drawdown": float(args.min_delta_max_drawdown),
            "min_delta_week_q10_pnl": float(args.min_delta_week_q10_pnl),
            "min_delta_week_q20_pnl": float(args.min_delta_week_q20_pnl),
            "min_delta_worst_week_pnl": float(args.min_delta_worst_week_pnl),
        },
        "results": gate.to_dict(orient="records"),
    }
    (args.out_dir / "contextual_tp_sl_promotion_gate.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )
    _write_report(args.out_dir, gate, args)
    print(
        json.dumps(
            _json_safe(
                {
                    "out_dir": str(args.out_dir),
                    "candidates": int(len(gate)),
                    "promotion_ready_count": int(gate["promotion_ready"].sum()) if not gate.empty else 0,
                    "coverage_pass_count": int(gate["coverage_pass"].sum()) if not gate.empty else 0,
                    "economics_tail_pass_count": int(gate["economics_tail_pass"].sum()) if not gate.empty else 0,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
