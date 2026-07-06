#!/usr/bin/env python3
"""Evaluate rolling prior-week selection of contextual TP/SL overlay rules.

This does not replay the portfolio again. It combines already materialized
weekly/daily replay outputs and asks whether a rule chosen from prior weeks
would have improved the next week versus fixed references.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"


def _objective(daily: pd.Series, weekly: pd.Series) -> Dict[str, float]:
    daily = pd.to_numeric(daily, errors="coerce").dropna()
    weekly = pd.to_numeric(weekly, errors="coerce").dropna()
    avg_week = float(weekly.mean()) if not weekly.empty else 0.0
    daily_q20 = float(daily.quantile(0.20)) if not daily.empty else 0.0
    daily_q35 = float(daily.quantile(0.35)) if not daily.empty else 0.0
    weekly_q05 = float(weekly.quantile(0.05)) if not weekly.empty else 0.0
    weekly_q10 = float(weekly.quantile(0.10)) if not weekly.empty else 0.0
    weekly_q20 = float(weekly.quantile(0.20)) if not weekly.empty else 0.0
    weekly_q35 = float(weekly.quantile(0.35)) if not weekly.empty else 0.0
    return {
        OBJECTIVE_COL: avg_week + 0.7 * daily_q35 + 0.3 * daily_q20,
        "avg_week_pnl": avg_week,
        "net_pnl": float(daily.sum()) if not daily.empty else float(weekly.sum()),
        "daily_q20_pnl": daily_q20,
        "daily_q35_pnl": daily_q35,
        "weekly_q05_pnl": weekly_q05,
        "weekly_q10_pnl": weekly_q10,
        "weekly_q20_pnl": weekly_q20,
        "weekly_q35_pnl": weekly_q35,
        "weekly_count": int(len(weekly)),
        "daily_count": int(len(daily)),
        "positive_week_rate": float((weekly > 0.0).mean()) if not weekly.empty else np.nan,
    }


def _load(replay_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(replay_dir / "conditional_filter_daily.csv")
    weekly = pd.read_csv(replay_dir / "conditional_filter_weekly.csv")
    daily["day_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["week_start"] = pd.to_datetime(
        weekly["week"].astype(str).str.split("/", n=1).str[0],
        utc=True,
        errors="coerce",
    )
    daily = daily.dropna(subset=["day_start"])
    weekly = weekly.dropna(subset=["week_start"])
    return daily, weekly


def _subset_rule(frame: pd.DataFrame, combo_id: str, rule_id: str) -> pd.DataFrame:
    return frame.loc[frame["combo_id"].eq(combo_id) & frame["rule_id"].eq(rule_id)]


def _score_rules(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    combo_id: str,
    rule_ids: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    d_win = daily.loc[daily["day_start"].ge(start) & daily["day_start"].lt(end)]
    w_win = weekly.loc[weekly["week_start"].ge(start) & weekly["week_start"].lt(end)]
    for rule_id in rule_ids:
        d = _subset_rule(d_win, combo_id, rule_id)
        w = _subset_rule(w_win, combo_id, rule_id)
        rec: Dict[str, Any] = {"combo_id": combo_id, "rule_id": rule_id}
        rec.update(_objective(d["net_pnl"], w["net_pnl"]))
        rec["trades"] = int(pd.to_numeric(d.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def _build_rolling(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    combo_id: str,
    rule_ids: List[str],
    lookback_weeks: int,
    min_history_weeks: int,
    fallback_rule: str,
    selection_mode: str,
    min_objective_delta: float,
) -> pd.DataFrame:
    weeks = sorted(_subset_rule(weekly, combo_id, fallback_rule)["week_start"].dropna().unique())
    rows: List[Dict[str, Any]] = []
    for week_start_raw in weeks:
        week_start = pd.Timestamp(week_start_raw)
        history_start = week_start - pd.Timedelta(weeks=int(lookback_weeks))
        history = weekly.loc[
            weekly["combo_id"].eq(combo_id)
            & weekly["rule_id"].eq(fallback_rule)
            & weekly["week_start"].ge(history_start)
            & weekly["week_start"].lt(week_start)
        ]
        if len(history) < int(min_history_weeks):
            selected_rule = fallback_rule
            selected_score = np.nan
            selected_reason = "fallback_insufficient_history"
        else:
            scores = _score_rules(daily, weekly, combo_id, rule_ids, history_start, week_start)
            fallback_score = scores.loc[scores["rule_id"].eq(fallback_rule)]
            if fallback_score.empty:
                fallback_score = scores.sort_values(OBJECTIVE_COL, ascending=False).head(1)
            fallback_score = fallback_score.iloc[0]
            ordered = scores.sort_values(OBJECTIVE_COL, ascending=False)
            best = ordered.iloc[0]
            if selection_mode == "best":
                selected_rule = str(best["rule_id"])
                selected_score = float(best[OBJECTIVE_COL])
                selected_reason = "rolling_selected"
            elif selection_mode == "conservative_tail":
                eligible = ordered.loc[
                    ordered[OBJECTIVE_COL].ge(float(fallback_score[OBJECTIVE_COL]) + float(min_objective_delta))
                    & ordered["weekly_q05_pnl"].ge(float(fallback_score["weekly_q05_pnl"]))
                    & ordered["weekly_q20_pnl"].ge(float(fallback_score["weekly_q20_pnl"]))
                    & ordered["daily_q20_pnl"].ge(float(fallback_score["daily_q20_pnl"]))
                ]
                if eligible.empty:
                    selected_rule = fallback_rule
                    selected_score = float(fallback_score[OBJECTIVE_COL])
                    selected_reason = "fallback_tail_guard"
                else:
                    best = eligible.iloc[0]
                    selected_rule = str(best["rule_id"])
                    selected_score = float(best[OBJECTIVE_COL])
                    selected_reason = "rolling_selected_tail_guard"
            else:
                raise ValueError(f"Unknown selection mode `{selection_mode}`")
        selected_week = weekly.loc[
            weekly["combo_id"].eq(combo_id)
            & weekly["rule_id"].eq(selected_rule)
            & weekly["week_start"].eq(week_start)
        ]
        fallback_week = weekly.loc[
            weekly["combo_id"].eq(combo_id)
            & weekly["rule_id"].eq(fallback_rule)
            & weekly["week_start"].eq(week_start)
        ]
        if selected_week.empty or fallback_week.empty:
            continue
        selected_week = selected_week.iloc[0]
        fallback_week = fallback_week.iloc[0]
        rows.append(
            {
                "combo_id": combo_id,
                "lookback_weeks": int(lookback_weeks),
                "min_history_weeks": int(min_history_weeks),
                "selection_mode": selection_mode,
                "min_objective_delta": float(min_objective_delta),
                "week": selected_week["week"],
                "week_start": str(week_start),
                "selected_rule": selected_rule,
                "selected_reason": selected_reason,
                "selected_history_objective": selected_score,
                "selected_net_pnl": float(selected_week["net_pnl"]),
                "fallback_rule": fallback_rule,
                "fallback_net_pnl": float(fallback_week["net_pnl"]),
                "delta_vs_fallback": float(selected_week["net_pnl"] - fallback_week["net_pnl"]),
                "selected_trades": int(selected_week.get("trades", 0)),
                "fallback_trades": int(fallback_week.get("trades", 0)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_series(frame: pd.DataFrame, value_col: str, prefix: str) -> Dict[str, float]:
    values = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_net_pnl": 0.0,
            f"{prefix}_avg_week_pnl": 0.0,
            f"{prefix}_weekly_q05_pnl": 0.0,
            f"{prefix}_weekly_q10_pnl": 0.0,
            f"{prefix}_weekly_q20_pnl": 0.0,
            f"{prefix}_weekly_q35_pnl": 0.0,
        }
    return {
        f"{prefix}_net_pnl": float(values.sum()),
        f"{prefix}_avg_week_pnl": float(values.mean()),
        f"{prefix}_weekly_q05_pnl": float(values.quantile(0.05)),
        f"{prefix}_weekly_q10_pnl": float(values.quantile(0.10)),
        f"{prefix}_weekly_q20_pnl": float(values.quantile(0.20)),
        f"{prefix}_weekly_q35_pnl": float(values.quantile(0.35)),
    }


def _summarize_rolling(rolling: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (combo_id, mode, lookback), group in rolling.groupby(["combo_id", "selection_mode", "lookback_weeks"], sort=False):
        rec: Dict[str, Any] = {
            "combo_id": combo_id,
            "selection_mode": mode,
            "lookback_weeks": int(lookback),
            "weeks": int(len(group)),
            "selection_rate": float(group["selected_reason"].astype(str).str.startswith("rolling_selected").mean()),
            "unique_selected_rules": int(group["selected_rule"].nunique()),
            "most_common_rule": str(group["selected_rule"].mode().iloc[0]) if not group.empty else "",
        }
        rec.update(_summarize_series(group, "selected_net_pnl", "selected"))
        rec.update(_summarize_series(group, "fallback_net_pnl", "fallback"))
        rec.update(_summarize_series(group, "delta_vs_fallback", "delta"))
        rec["positive_delta_week_rate"] = float(pd.to_numeric(group["delta_vs_fallback"], errors="coerce").gt(0).mean())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("selected_avg_week_pnl", ascending=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--combo-id", default="long_bars:S_long_dist:R_short_asset:R_short_bollinger:R")
    parser.add_argument("--fallback-rule", default="none")
    parser.add_argument("--lookback-weeks", default="4,8,12")
    parser.add_argument("--min-history-weeks", type=int, default=4)
    parser.add_argument("--selection-modes", default="best,conservative_tail")
    parser.add_argument("--min-objective-delta", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily, weekly = _load(args.replay_dir)
    rules = sorted(_subset_rule(weekly, args.combo_id, args.fallback_rule)["rule_id"].unique())
    # The line above only yields fallback; use daily/weekly combo rows to collect all rules.
    rules = sorted(
        set(weekly.loc[weekly["combo_id"].eq(args.combo_id), "rule_id"].dropna().astype(str).unique())
    )
    lookbacks = [int(v.strip()) for v in str(args.lookback_weeks).split(",") if v.strip()]
    selection_modes = [str(v.strip()) for v in str(args.selection_modes).split(",") if v.strip()]
    rolling_frames = [
        _build_rolling(
            daily,
            weekly,
            args.combo_id,
            rules,
            lookback,
            args.min_history_weeks,
            args.fallback_rule,
            mode,
            args.min_objective_delta,
        )
        for lookback in lookbacks
        for mode in selection_modes
    ]
    rolling = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    summary = _summarize_rolling(rolling) if not rolling.empty else pd.DataFrame()
    rolling.to_csv(args.out_dir / "rolling_rule_selection_weekly.csv", index=False)
    summary.to_csv(args.out_dir / "rolling_rule_selection_summary.csv", index=False)

    top_cols = [
        "combo_id",
        "selection_mode",
        "lookback_weeks",
        "weeks",
        "selection_rate",
        "unique_selected_rules",
        "most_common_rule",
        "selected_net_pnl",
        "fallback_net_pnl",
        "delta_net_pnl",
        "selected_avg_week_pnl",
        "fallback_avg_week_pnl",
        "delta_avg_week_pnl",
        "selected_weekly_q05_pnl",
        "fallback_weekly_q05_pnl",
        "delta_weekly_q05_pnl",
        "selected_weekly_q20_pnl",
        "fallback_weekly_q20_pnl",
        "delta_weekly_q20_pnl",
        "positive_delta_week_rate",
    ]
    lines = [
        "# Rolling Reliability Rule Selection",
        "",
        f"Replay dir: `{args.replay_dir}`",
        f"Combo: `{args.combo_id}`",
        f"Fallback rule: `{args.fallback_rule}`",
        f"Lookbacks: `{lookbacks}`",
        f"Selection modes: `{selection_modes}`",
        f"Minimum objective delta: `{args.min_objective_delta}`",
        "Selection uses only prior weeks; the selected rule is applied to the next week from existing replay outputs.",
        "Costs are included. This is a replay-period walk-forward audit, not untouched live OOS.",
        "",
        "## Summary",
        "",
        summary[[c for c in top_cols if c in summary.columns]].round(6).to_markdown(index=False)
        if not summary.empty
        else "_No rows._",
        "",
        "## Last 12 Weekly Decisions",
        "",
        rolling.tail(12).round(6).to_markdown(index=False) if not rolling.empty else "_No rows._",
    ]
    payload = {
        "replay_dir": str(args.replay_dir),
        "out_dir": str(args.out_dir),
        "combo_id": args.combo_id,
        "fallback_rule": args.fallback_rule,
        "lookback_weeks": lookbacks,
        "selection_modes": selection_modes,
        "min_objective_delta": float(args.min_objective_delta),
        "rows": int(len(rolling)),
    }
    (args.out_dir / "rolling_rule_selection_report.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "rolling_rule_selection_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"out_dir": str(args.out_dir), "rows": int(len(rolling))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
