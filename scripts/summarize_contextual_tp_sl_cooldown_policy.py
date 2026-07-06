#!/usr/bin/env python3
"""Summarize a sparse cooldown overlay from materialized replay metrics.

The sparse selector decides which already materialized replay variant to use per
week. This script reconstructs the chosen weekly/daily metric panel without
rerunning the replay, then compares it with the static default and no-op
baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RATE_COLUMNS = ("hit_rate", "full_sl_rate", "timeout_rate")
VALUE_COLUMNS = ("net_pnl", "gross_pnl", "trades")


def _read_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for col in VALUE_COLUMNS + RATE_COLUMNS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "head" in frame.columns:
        frame["head"] = frame["head"].fillna("__global__")
    return frame


def _week_start(period: str) -> pd.Timestamp:
    return pd.Timestamp(str(period).split("/")[0])


def _week_end(period: str) -> pd.Timestamp:
    return pd.Timestamp(str(period).split("/")[-1])


def _selected_week_source(selections: pd.DataFrame) -> dict[str, str]:
    """Return week -> source label.

    For the current sparse-cooldown design, a triggered ``__noop__`` action uses
    the no-op replay for that week. All other rows are evaluated as the static
    default replay.
    """
    out: dict[str, str] = {}
    for row in selections.itertuples(index=False):
        action = str(getattr(row, "action_label"))
        triggered = bool(getattr(row, "triggered"))
        out[str(getattr(row, "eval_week"))] = "noop" if triggered and action == "__noop__" else "default"
    return out


def _stitch_weekly(default_weekly: pd.DataFrame, noop_weekly: pd.DataFrame, week_source: dict[str, str]) -> pd.DataFrame:
    frames = []
    noop_by_week = {week: part.copy() for week, part in noop_weekly.groupby("week", sort=False)}
    default_by_week = {week: part.copy() for week, part in default_weekly.groupby("week", sort=False)}
    all_weeks = list(default_by_week)
    for week in all_weeks:
        source = week_source.get(week, "default")
        part = (noop_by_week if source == "noop" else default_by_week)[week].copy()
        part["source_replay"] = source
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def _stitch_daily(default_daily: pd.DataFrame, noop_daily: pd.DataFrame, week_source: dict[str, str]) -> pd.DataFrame:
    frames = []
    default_daily = default_daily.copy()
    noop_daily = noop_daily.copy()
    default_daily["day_ts"] = pd.to_datetime(default_daily["day"])
    noop_daily["day_ts"] = pd.to_datetime(noop_daily["day"])
    noop_weeks = [
        (_week_start(week), _week_end(week))
        for week, source in week_source.items()
        if source == "noop"
    ]
    for _, row in default_daily[["day", "day_ts"]].drop_duplicates().iterrows():
        day = row["day_ts"]
        source = "noop" if any(start <= day <= end for start, end in noop_weeks) else "default"
        base = noop_daily if source == "noop" else default_daily
        part = base[base["day_ts"] == day].copy()
        part["source_replay"] = source
        frames.append(part)
    out = pd.concat(frames, ignore_index=True)
    return out.drop(columns=["day_ts"])


def _aggregate_period(frame: pd.DataFrame, period_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["weighted_hit"] = frame["hit_rate"] * frame["trades"]
    frame["weighted_full_sl"] = frame["full_sl_rate"] * frame["trades"]
    frame["weighted_timeout"] = frame["timeout_rate"] * frame["trades"]
    grouped = frame.groupby([period_col, "head"], dropna=False, sort=True)
    sums = grouped[["net_pnl", "gross_pnl", "trades", "weighted_hit", "weighted_full_sl", "weighted_timeout"]].sum(min_count=1).reset_index()
    denom = sums["trades"].replace(0, np.nan)
    sums["hit_rate"] = sums["weighted_hit"] / denom
    sums["full_sl_rate"] = sums["weighted_full_sl"] / denom
    sums["timeout_rate"] = sums["weighted_timeout"] / denom
    return sums.drop(columns=["weighted_hit", "weighted_full_sl", "weighted_timeout"])


def _monthly_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = pd.to_datetime(daily["day"]).dt.to_period("M").astype(str)
    return _aggregate_period(daily, "month")


def _compare(left: pd.DataFrame, right: pd.DataFrame, key_cols: Iterable[str], right_name: str) -> pd.DataFrame:
    key_cols = list(key_cols)
    merged = left.merge(right, on=key_cols, how="left", suffixes=("", f"_{right_name}"))
    for col in VALUE_COLUMNS + RATE_COLUMNS:
        rhs = f"{col}_{right_name}"
        if rhs in merged.columns and col in merged.columns:
            merged[f"delta_{col}_vs_{right_name}"] = merged[col] - merged[rhs]
    return merged


def _write_markdown(
    out_path: Path,
    weekly_global: pd.DataFrame,
    monthly_global: pd.DataFrame,
    summary_head: pd.DataFrame,
    triggered: pd.DataFrame,
) -> None:
    def fmt_table(frame: pd.DataFrame, cols: list[str]) -> str:
        view = frame[cols].copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
        return view.to_markdown(index=False)

    lines = [
        "# Cooldown4 Materialized Metrics",
        "",
        "This reconstructs the sparse `cooldown4` policy from existing replay artifacts: static no-drift default everywhere except triggered no-op weeks.",
        "",
        "## Triggered Weeks",
        "",
        triggered.to_markdown(index=False) if not triggered.empty else "No triggered weeks.",
        "",
        "## Global Weekly Metrics",
        "",
        fmt_table(
            weekly_global,
            [
                "week",
                "source_replay",
                "net_pnl",
                "delta_net_pnl_vs_default",
                "delta_net_pnl_vs_noop",
                "trades",
                "hit_rate",
                "full_sl_rate",
                "timeout_rate",
            ],
        ),
        "",
        "## Global Monthly Metrics",
        "",
        fmt_table(
            monthly_global,
            [
                "month",
                "net_pnl",
                "delta_net_pnl_vs_default",
                "delta_net_pnl_vs_noop",
                "trades",
                "hit_rate",
                "full_sl_rate",
                "timeout_rate",
            ],
        ),
        "",
        "## Summary By Head",
        "",
        fmt_table(
            summary_head,
            [
                "head",
                "net_pnl",
                "delta_net_pnl_vs_default",
                "delta_net_pnl_vs_noop",
                "trades",
                "hit_rate",
                "full_sl_rate",
                "timeout_rate",
            ],
        ),
        "",
    ]
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--default-dir", type=Path, required=True)
    parser.add_argument("--noop-dir", type=Path, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selections = pd.read_csv(args.selections)
    week_source = _selected_week_source(selections)
    triggered = selections[selections["triggered"]].copy()

    default_weekly = _read_metrics(args.default_dir / "combo_replay_weekly_metrics.csv")
    noop_weekly = _read_metrics(args.noop_dir / "combo_replay_weekly_metrics.csv")
    default_daily = _read_metrics(args.default_dir / "combo_replay_daily_metrics.csv")
    noop_daily = _read_metrics(args.noop_dir / "combo_replay_daily_metrics.csv")

    cooldown_weekly = _stitch_weekly(default_weekly, noop_weekly, week_source)
    cooldown_daily = _stitch_daily(default_daily, noop_daily, week_source)
    cooldown_monthly = _monthly_from_daily(cooldown_daily)
    cooldown_summary = _aggregate_period(cooldown_daily.assign(period="all"), "period")

    default_monthly = _monthly_from_daily(default_daily)
    noop_monthly = _monthly_from_daily(noop_daily)
    default_summary = _aggregate_period(default_daily.assign(period="all"), "period")
    noop_summary = _aggregate_period(noop_daily.assign(period="all"), "period")

    weekly_cmp = _compare(cooldown_weekly, default_weekly, ["week", "head"], "default")
    weekly_cmp = _compare(weekly_cmp, noop_weekly, ["week", "head"], "noop")
    monthly_cmp = _compare(cooldown_monthly, default_monthly, ["month", "head"], "default")
    monthly_cmp = _compare(monthly_cmp, noop_monthly, ["month", "head"], "noop")
    summary_cmp = _compare(cooldown_summary, default_summary, ["period", "head"], "default")
    summary_cmp = _compare(summary_cmp, noop_summary, ["period", "head"], "noop")

    cooldown_weekly.to_csv(args.output_dir / "cooldown4_weekly_metrics.csv", index=False)
    weekly_cmp.to_csv(args.output_dir / "cooldown4_weekly_metrics_with_deltas.csv", index=False)
    cooldown_daily.to_csv(args.output_dir / "cooldown4_daily_metrics.csv", index=False)
    monthly_cmp.to_csv(args.output_dir / "cooldown4_monthly_metrics_with_deltas.csv", index=False)
    summary_cmp.to_csv(args.output_dir / "cooldown4_summary_by_head_with_deltas.csv", index=False)
    triggered.to_csv(args.output_dir / "cooldown4_triggered_weeks.csv", index=False)

    weekly_global = weekly_cmp[weekly_cmp["head"] == "__global__"].copy()
    monthly_global = monthly_cmp[monthly_cmp["head"] == "__global__"].copy()
    summary_head = summary_cmp[summary_cmp["head"] != "__global__"].copy()
    _write_markdown(
        args.output_dir / "cooldown4_materialized_metrics_report.md",
        weekly_global,
        monthly_global,
        summary_head,
        triggered[["eval_week", "action_label", "signal", "eval_delta_net_pnl", "default_eval_delta_net_pnl", "incremental_delta_vs_default"]],
    )
    print(args.output_dir / "cooldown4_materialized_metrics_report.md")


if __name__ == "__main__":
    main()
