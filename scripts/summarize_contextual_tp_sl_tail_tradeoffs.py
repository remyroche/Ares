#!/usr/bin/env python3
"""Summarize contextual TP/SL replay variants with tail-aware objectives."""

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


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _variant_dirs(root: Path, exclude_parts: set[str]) -> List[Path]:
    dirs: List[Path] = []
    for path in root.glob("*/materialized/*"):
        if any(part in exclude_parts for part in path.parts):
            continue
        if (path / "combo_replay_manifest.json").exists():
            dirs.append(path)
    return sorted(dirs)


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        return json.loads((path / "combo_replay_manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _period_slice(frame: pd.DataFrame, period_type: str, head: str | None = None) -> pd.DataFrame:
    if frame.empty or "period_type" not in frame.columns:
        return pd.DataFrame()
    out = frame.loc[frame["period_type"].astype(str).eq(period_type)].copy()
    if head is None:
        if "head" in out.columns:
            out = out.loc[out["head"].isna()].copy()
    else:
        out = out.loc[out.get("head", pd.Series(index=out.index, dtype=object)).astype(str).eq(head)].copy()
    return out


def _metrics_from_periods(daily: pd.DataFrame, weekly: pd.DataFrame, *, head: str | None = None) -> Dict[str, float]:
    day = _period_slice(daily, "day" if head is None else "day_head", head=head)
    week = _period_slice(weekly, "week" if head is None else "week_head", head=head)
    day_pnl = pd.to_numeric(day.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    week_pnl = pd.to_numeric(week.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    day_full_sl = pd.to_numeric(day.get("full_sl_rate", pd.Series(dtype=float)), errors="coerce").dropna()
    week_full_sl = pd.to_numeric(week.get("full_sl_rate", pd.Series(dtype=float)), errors="coerce").dropna()
    day_hit = pd.to_numeric(day.get("hit_rate", pd.Series(dtype=float)), errors="coerce").dropna()
    week_hit = pd.to_numeric(week.get("hit_rate", pd.Series(dtype=float)), errors="coerce").dropna()
    return {
        "day_count": float(len(day_pnl)),
        "week_count": float(len(week_pnl)),
        "sum_net_pnl": float(day_pnl.sum()) if len(day_pnl) else np.nan,
        "avg_day_net_pnl": float(day_pnl.mean()) if len(day_pnl) else np.nan,
        "q20_day_net_pnl": float(day_pnl.quantile(0.20)) if len(day_pnl) else np.nan,
        "q35_day_net_pnl": float(day_pnl.quantile(0.35)) if len(day_pnl) else np.nan,
        "q50_day_net_pnl": float(day_pnl.quantile(0.50)) if len(day_pnl) else np.nan,
        "avg_week_net_pnl": float(week_pnl.mean()) if len(week_pnl) else np.nan,
        "q15_week_net_pnl": float(week_pnl.quantile(0.15)) if len(week_pnl) else np.nan,
        "q25_week_net_pnl": float(week_pnl.quantile(0.25)) if len(week_pnl) else np.nan,
        "q50_week_net_pnl": float(week_pnl.quantile(0.50)) if len(week_pnl) else np.nan,
        "positive_day_share": float((day_pnl > 0.0).mean()) if len(day_pnl) else np.nan,
        "positive_week_share": float((week_pnl > 0.0).mean()) if len(week_pnl) else np.nan,
        "avg_day_full_sl_rate": float(day_full_sl.mean()) if len(day_full_sl) else np.nan,
        "avg_week_full_sl_rate": float(week_full_sl.mean()) if len(week_full_sl) else np.nan,
        "avg_day_hit_rate": float(day_hit.mean()) if len(day_hit) else np.nan,
        "avg_week_hit_rate": float(week_hit.mean()) if len(week_hit) else np.nan,
    }


def _tail_objective(row: pd.Series, q35_weight: float, q20_weight: float) -> float:
    return float(row["avg_week_net_pnl"] + q35_weight * row["q35_day_net_pnl"] + q20_weight * row["q20_day_net_pnl"])


def _family_from_risk_column(value: Any) -> str:
    text = str(value)
    for family in ("recent_hr_surprise", "uncertainty", "drift", "ood", "composite"):
        if family in text:
            return family
    return "unknown"


def _markdown_table(frame: pd.DataFrame, columns: List[str], limit: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    cur = frame[[col for col in columns if col in frame.columns]].head(limit).copy()
    for col in cur.columns:
        if pd.api.types.is_float_dtype(cur[col]):
            cur[col] = cur[col].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    return cur.to_markdown(index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True, action="append", help="Replay root directory. May be passed multiple times.")
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    parser.add_argument(
        "--exclude-root-names",
        default="noop_baseline,summary,tail_tradeoff_summary",
        help="Comma-separated root child directory names to exclude from candidate variants.",
    )
    args = parser.parse_args()

    roots = [Path(path) for path in args.root_dir]
    baseline_dir = Path(args.baseline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_daily = _read_csv(baseline_dir / "combo_replay_daily_metrics.csv")
    baseline_weekly = _read_csv(baseline_dir / "combo_replay_weekly_metrics.csv")
    baseline_global = _metrics_from_periods(baseline_daily, baseline_weekly)
    baseline_global["tail_objective"] = _tail_objective(pd.Series(baseline_global), args.q35_weight, args.q20_weight)

    baseline_by_head: Dict[str, Dict[str, float]] = {}
    heads = sorted(
        set(
            baseline_daily.get("head", pd.Series(dtype=object)).dropna().astype(str).tolist()
            + baseline_weekly.get("head", pd.Series(dtype=object)).dropna().astype(str).tolist()
        )
    )
    for head in heads:
        metrics = _metrics_from_periods(baseline_daily, baseline_weekly, head=head)
        metrics["tail_objective"] = _tail_objective(pd.Series(metrics), args.q35_weight, args.q20_weight)
        baseline_by_head[head] = metrics

    global_rows: List[Dict[str, Any]] = []
    head_rows: List[Dict[str, Any]] = []
    exclude_parts = {part.strip() for part in str(args.exclude_root_names).split(",") if part.strip()}
    for root in roots:
        for variant_dir in _variant_dirs(root, exclude_parts):
            manifest = _load_manifest(variant_dir)
            daily = _read_csv(variant_dir / "combo_replay_daily_metrics.csv")
            weekly = _read_csv(variant_dir / "combo_replay_weekly_metrics.csv")
            row: Dict[str, Any] = {
                "source_root": str(root),
                "head": manifest.get("gate_head", variant_dir.parent.parent.name),
                "label": variant_dir.name,
                "risk_column": manifest.get("risk_column"),
                "diagnostic_family": _family_from_risk_column(manifest.get("risk_column")),
                "threshold": manifest.get("threshold"),
                "action": manifest.get("action"),
                "size_multiplier": manifest.get("size_multiplier"),
                "gate_rows": manifest.get("gate_rows"),
                "gate_row_share_within_head": manifest.get("gate_row_share_within_head"),
                "trade_count": manifest.get("metrics", {}).get("trade_count"),
                "full_sl_rate": manifest.get("metrics", {}).get("full_sl_rate"),
                "timeout_rate": manifest.get("metrics", {}).get("timeout_rate"),
                "max_drawdown": manifest.get("metrics", {}).get("max_drawdown"),
            }
            row.update(_metrics_from_periods(daily, weekly))
            row["tail_objective"] = _tail_objective(pd.Series(row), args.q35_weight, args.q20_weight)
            for key, value in baseline_global.items():
                row[f"baseline_{key}"] = value
                row[f"delta_{key}"] = row.get(key, np.nan) - value
            row["delta_max_drawdown"] = row["max_drawdown"] - float(
                _load_manifest(baseline_dir).get("metrics", {}).get("max_drawdown", np.nan)
            )
            global_rows.append(row)

            for head in heads:
                hrow = {
                    "source_root": str(root),
                    "gate_head": row["head"],
                    "metric_head": head,
                    "label": row["label"],
                    "risk_column": row["risk_column"],
                    "diagnostic_family": row["diagnostic_family"],
                    "threshold": row["threshold"],
                    "action": row["action"],
                    "size_multiplier": row["size_multiplier"],
                    "gate_rows": row["gate_rows"],
                }
                hrow.update(_metrics_from_periods(daily, weekly, head=head))
                hrow["tail_objective"] = _tail_objective(pd.Series(hrow), args.q35_weight, args.q20_weight)
                base_head = baseline_by_head.get(head, {})
                for key, value in base_head.items():
                    hrow[f"baseline_{key}"] = value
                    hrow[f"delta_{key}"] = hrow.get(key, np.nan) - value
                head_rows.append(hrow)

    # Keep exact duplicate runs only once when multiple roots overlap.
    if global_rows:
        global_rows_df = pd.DataFrame(global_rows).drop_duplicates(
            subset=["head", "label", "risk_column", "threshold", "action", "size_multiplier", "gate_rows"],
            keep="first",
        )
        global_rows = global_rows_df.to_dict("records")
    if head_rows:
        head_rows_df = pd.DataFrame(head_rows).drop_duplicates(
            subset=[
                "gate_head",
                "metric_head",
                "label",
                "risk_column",
                "threshold",
                "action",
                "size_multiplier",
                "gate_rows",
            ],
            keep="first",
        )
        head_rows = head_rows_df.to_dict("records")

    global_df = pd.DataFrame(global_rows)
    head_df = pd.DataFrame(head_rows)
    if not global_df.empty:
        global_df = global_df.sort_values(["delta_tail_objective", "delta_sum_net_pnl"], ascending=False)
    if not head_df.empty:
        head_df = head_df.sort_values(["gate_head", "metric_head", "delta_tail_objective"], ascending=[True, True, False])

    best_by_gate_head = (
        global_df.sort_values(["head", "delta_tail_objective", "delta_sum_net_pnl"], ascending=[True, False, False])
        .groupby("head", as_index=False)
        .head(5)
        if not global_df.empty
        else pd.DataFrame()
    )
    best_by_family = (
        global_df.sort_values(
            ["head", "diagnostic_family", "delta_tail_objective", "delta_sum_net_pnl"],
            ascending=[True, True, False, False],
        )
        .groupby(["head", "diagnostic_family"], as_index=False)
        .head(1)
        if not global_df.empty
        else pd.DataFrame()
    )
    family_summary = (
        best_by_family.groupby("diagnostic_family", as_index=False)
        .agg(
            heads_tested=("head", "nunique"),
            heads_positive_tail_objective=("delta_tail_objective", lambda s: int((s > 0.0).sum())),
            heads_positive_net_pnl=("delta_sum_net_pnl", lambda s: int((s > 0.0).sum())),
            sum_best_delta_tail_objective=("delta_tail_objective", "sum"),
            sum_best_delta_net_pnl=("delta_sum_net_pnl", "sum"),
            mean_delta_q20_day_net_pnl=("delta_q20_day_net_pnl", "mean"),
            mean_delta_q35_day_net_pnl=("delta_q35_day_net_pnl", "mean"),
            mean_delta_q15_week_net_pnl=("delta_q15_week_net_pnl", "mean"),
        )
        .sort_values(["sum_best_delta_tail_objective", "sum_best_delta_net_pnl"], ascending=False)
        if not best_by_family.empty
        else pd.DataFrame()
    )

    global_df.to_csv(out_dir / "tail_tradeoff_all_variants.csv", index=False)
    head_df.to_csv(out_dir / "tail_tradeoff_per_metric_head.csv", index=False)
    best_by_gate_head.to_csv(out_dir / "tail_tradeoff_best_by_gate_head.csv", index=False)
    best_by_family.to_csv(out_dir / "tail_tradeoff_best_by_head_and_family.csv", index=False)
    family_summary.to_csv(out_dir / "tail_tradeoff_family_summary.csv", index=False)

    cols = [
        "head",
        "diagnostic_family",
        "label",
        "threshold",
        "size_multiplier",
        "gate_rows",
        "delta_tail_objective",
        "delta_sum_net_pnl",
        "delta_avg_week_net_pnl",
        "delta_q35_day_net_pnl",
        "delta_q20_day_net_pnl",
        "delta_q15_week_net_pnl",
        "delta_avg_week_full_sl_rate",
    ]
    lines = [
        "# Contextual TP/SL Tail Tradeoff Summary",
        "",
        "Roots: " + ", ".join(f"`{root}`" for root in roots),
        f"Baseline: `{baseline_dir}`",
        f"Objective: avg_week_net_pnl + {args.q35_weight:g} * q35_day_net_pnl + {args.q20_weight:g} * q20_day_net_pnl",
        f"Baseline objective: `{baseline_global['tail_objective']:.6f}`",
        "",
        "## Family Summary",
        "",
        _markdown_table(
            family_summary,
            [
                "diagnostic_family",
                "heads_tested",
                "heads_positive_tail_objective",
                "heads_positive_net_pnl",
                "sum_best_delta_tail_objective",
                "sum_best_delta_net_pnl",
                "mean_delta_q20_day_net_pnl",
                "mean_delta_q35_day_net_pnl",
                "mean_delta_q15_week_net_pnl",
            ],
        ),
        "",
        "## Best Variants By Gate Head",
        "",
        _markdown_table(best_by_gate_head, cols, 25),
        "",
        "## Best Variant Per Head And Diagnostic Family",
        "",
        _markdown_table(best_by_family, cols, 40),
    ]
    (out_dir / "tail_tradeoff_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "generated_by": "summarize_contextual_tp_sl_tail_tradeoffs",
        "root_dirs": [str(root) for root in roots],
        "baseline_dir": str(baseline_dir),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
        "exclude_root_names": sorted(exclude_parts),
        "outputs": [
            "tail_tradeoff_report.md",
            "tail_tradeoff_all_variants.csv",
            "tail_tradeoff_per_metric_head.csv",
            "tail_tradeoff_best_by_gate_head.csv",
            "tail_tradeoff_best_by_head_and_family.csv",
            "tail_tradeoff_family_summary.csv",
        ],
    }
    (out_dir / "tail_tradeoff_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"out_dir": str(out_dir), "variant_count": int(len(global_df))}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
