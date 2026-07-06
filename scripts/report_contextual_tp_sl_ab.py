#!/usr/bin/env python3
"""Aggregate contextual TP/SL monthly replay artifacts into A/B tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


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


def _weighted_rate(group: pd.DataFrame, column: str) -> float:
    trades = group["trades"].astype(float).to_numpy()
    values = group[column].astype(float).to_numpy()
    mask = np.isfinite(trades) & np.isfinite(values) & (trades > 0)
    denom = trades[mask].sum()
    if denom <= 0:
        return np.nan
    return float(np.sum(values[mask] * trades[mask]) / denom)


def _agg_period(group: pd.DataFrame, period_col: str) -> pd.Series:
    return pd.Series(
        {
            period_col: group[period_col].iloc[0],
            "head": group["head"].iloc[0],
            "net_pnl": float(group["net_pnl"].sum()),
            "gross_pnl": float(group["gross_pnl"].sum()),
            "trades": int(group["trades"].sum()),
            "hit_rate": _weighted_rate(group, "hit_rate"),
            "full_sl_rate": _weighted_rate(group, "full_sl_rate"),
            "timeout_rate": _weighted_rate(group, "timeout_rate"),
        }
    )


def _read_weekly(path: Path, label: str) -> pd.DataFrame:
    weekly_path = path / "combo_replay_weekly_metrics.csv"
    if not weekly_path.exists():
        raise FileNotFoundError(f"Missing weekly metrics: {weekly_path}")
    df = pd.read_csv(weekly_path)
    df["label"] = label
    df["head"] = df["head"].fillna("__global__")
    return df


def _load_weekly(global_csv: Path, labels: Iterable[str]) -> pd.DataFrame:
    source = pd.read_csv(global_csv)
    rows: List[pd.DataFrame] = []
    wanted = set(labels)
    missing = sorted(wanted - set(source["label"].unique()))
    if missing:
        raise ValueError(f"Labels not present in global csv: {missing}")
    for _, record in source[source["label"].isin(wanted)].iterrows():
        rows.append(_read_weekly(Path(str(record["path"])), str(record["label"])))
    raw = pd.concat(rows, ignore_index=True)
    grouped_rows: List[Dict[str, Any]] = []
    for (label, period_type, week, head), group in raw.groupby(
        ["label", "period_type", "week", "head"], dropna=False, sort=True
    ):
        row = _agg_period(group, "week").to_dict()
        row.update({"label": label, "period_type": period_type, "week": week, "head": head})
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows)


def _load_monthly(global_csv: Path, labels: Iterable[str]) -> pd.DataFrame:
    df = pd.read_csv(global_csv)
    df = df[df["label"].isin(set(labels))].copy()
    cols = [
        "month",
        "label",
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "strategy_concentration",
        "side_concentration",
    ]
    return df[cols].rename(columns={"trade_count": "trades"})


def _with_deltas(df: pd.DataFrame, period_col: str, baseline_label: str) -> pd.DataFrame:
    key_cols = [period_col]
    if "head" in df.columns:
        key_cols.append("head")
    baseline = df[df["label"] == baseline_label].copy()
    baseline = baseline.rename(
        columns={
            "net_pnl": "net_pnl_baseline",
            "gross_pnl": "gross_pnl_baseline",
            "trades": "trades_baseline",
            "hit_rate": "hit_rate_baseline",
            "full_sl_rate": "full_sl_rate_baseline",
            "timeout_rate": "timeout_rate_baseline",
            "max_drawdown": "max_drawdown_baseline",
            "strategy_concentration": "strategy_concentration_baseline",
            "side_concentration": "side_concentration_baseline",
        }
    )
    baseline_cols = [c for c in baseline.columns if c in key_cols or c.endswith("_baseline")]
    merged = df.merge(baseline[baseline_cols], on=key_cols, how="left")
    for column in ["net_pnl", "gross_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate", "max_drawdown"]:
        base_col = f"{column}_baseline"
        if column in merged.columns and base_col in merged.columns:
            merged[f"delta_{column}"] = merged[column] - merged[base_col]
    return merged


def _summary(df: pd.DataFrame, period_col: str, baseline_label: str) -> pd.DataFrame:
    challengers = df[df["label"] != baseline_label].copy()
    rows: List[Dict[str, Any]] = []
    group_cols = ["label"]
    if "head" in challengers.columns:
        group_cols.append("head")
    for keys, group in challengers.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        delta_net = group["delta_net_pnl"].astype(float)
        row.update(
            {
                "periods": int(group[period_col].nunique()),
                "sum_delta_net_pnl": float(delta_net.sum()),
                "mean_delta_net_pnl": float(delta_net.mean()),
                "median_delta_net_pnl": float(delta_net.median()),
                "q35_delta_net_pnl": float(delta_net.quantile(0.35)),
                "q20_delta_net_pnl": float(delta_net.quantile(0.20)),
                "positive_period_share": float((delta_net > 0).mean()),
                "positive_period_count": int((delta_net > 0).sum()),
                "sum_delta_trades": int(group["delta_trades"].sum()) if "delta_trades" in group else 0,
                "mean_delta_hit_rate": float(group["delta_hit_rate"].mean()) if "delta_hit_rate" in group else np.nan,
                "mean_delta_full_sl_rate": float(group["delta_full_sl_rate"].mean()),
                "mean_delta_timeout_rate": float(group["delta_timeout_rate"].mean()),
                "mean_delta_max_drawdown": float(group["delta_max_drawdown"].mean())
                if "delta_max_drawdown" in group
                else np.nan,
                "objective": float(delta_net.mean() + 0.7 * delta_net.quantile(0.35) + 0.3 * delta_net.quantile(0.20)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["objective", "sum_delta_net_pnl"], ascending=False)


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--labels", default="wf_recent,longbars_weekgate_only,longbars_drift_only")
    parser.add_argument("--baseline-label", default="wf_recent")
    args = parser.parse_args()

    global_csv = Path(args.global_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [part.strip() for part in str(args.labels).split(",") if part.strip()]
    if args.baseline_label not in labels:
        labels.insert(0, args.baseline_label)

    weekly = _with_deltas(_load_weekly(global_csv, labels), "week", args.baseline_label)
    monthly = _with_deltas(_load_monthly(global_csv, labels), "month", args.baseline_label)
    global_weekly = weekly[(weekly["period_type"] == "week") & (weekly["head"] == "__global__")].copy()
    head_weekly = weekly[weekly["period_type"] == "week_head"].copy()
    monthly_summary = _summary(monthly, "month", args.baseline_label)
    weekly_summary = _summary(global_weekly, "week", args.baseline_label)
    head_weekly_summary = _summary(head_weekly, "week", args.baseline_label)

    monthly.to_csv(out_dir / "monthly_ab_metrics.csv", index=False)
    global_weekly.to_csv(out_dir / "weekly_global_ab_metrics.csv", index=False)
    head_weekly.to_csv(out_dir / "weekly_head_ab_metrics.csv", index=False)
    monthly_summary.to_csv(out_dir / "monthly_ab_summary.csv", index=False)
    weekly_summary.to_csv(out_dir / "weekly_global_ab_summary.csv", index=False)
    head_weekly_summary.to_csv(out_dir / "weekly_head_ab_summary.csv", index=False)

    md: List[str] = [
        "# Contextual TP/SL A/B Report",
        "",
        "This aggregates existing monthly replay artifacts. It does not rerun portfolio simulation and is not untouched OOS.",
        "",
        f"Baseline: `{args.baseline_label}`.",
        "",
        "Objective shown in summaries: `mean_delta_net_pnl + 0.7*q35_delta_net_pnl + 0.3*q20_delta_net_pnl`.",
        "",
        "## Monthly Global Summary",
        "",
        _markdown_table(
            monthly_summary,
            [
                "label",
                "periods",
                "objective",
                "sum_delta_net_pnl",
                "positive_period_count",
                "mean_delta_full_sl_rate",
                "mean_delta_max_drawdown",
                "sum_delta_trades",
            ],
        ),
        "",
        "## Weekly Global Summary",
        "",
        _markdown_table(
            weekly_summary,
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
        "## Weekly Per-Head Summary",
        "",
        _markdown_table(
            head_weekly_summary,
            [
                "label",
                "head",
                "periods",
                "sum_delta_net_pnl",
                "positive_period_count",
                "mean_delta_hit_rate",
                "mean_delta_full_sl_rate",
                "sum_delta_trades",
            ],
        ),
        "",
    ]
    (out_dir / "contextual_tp_sl_ab_report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "contextual_tp_sl_ab_manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "global_csv": str(global_csv),
                    "labels": labels,
                    "baseline_label": args.baseline_label,
                    "outputs": [
                        "monthly_ab_metrics.csv",
                        "weekly_global_ab_metrics.csv",
                        "weekly_head_ab_metrics.csv",
                        "monthly_ab_summary.csv",
                        "weekly_global_ab_summary.csv",
                        "weekly_head_ab_summary.csv",
                        "contextual_tp_sl_ab_report.md",
                    ],
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(out_dir / "contextual_tp_sl_ab_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
