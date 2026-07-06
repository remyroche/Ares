#!/usr/bin/env python3
"""Pairwise rolling switch audit for contextual TP/SL combo challengers.

The script recombines already replayed daily/weekly combo outcomes. It chooses
between a fixed champion combo and one challenger using only prior weeks, then
scores the selected stream against the champion over full and validation windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"
CHAMPION_COMBO = "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R"


def _read_first_existing(root: Path, names: Iterable[str]) -> pd.DataFrame:
    for name in names:
        path = root / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No expected files in {root}: {list(names)}")


def _load(replay_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    for root in replay_dirs:
        daily = _read_first_existing(root, ["head_arm_combination_daily.csv"])
        weekly = _read_first_existing(root, ["head_arm_combination_weekly.csv"])
        daily["_source_replay_dir"] = str(root)
        weekly["_source_replay_dir"] = str(root)
        daily_frames.append(daily)
        weekly_frames.append(weekly)
    daily = pd.concat(daily_frames, ignore_index=True).drop_duplicates(["combo_id", "day"])
    weekly = pd.concat(weekly_frames, ignore_index=True).drop_duplicates(["combo_id", "week"])
    daily["day_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["week_start"] = pd.to_datetime(
        weekly["week"].astype(str).str.split("/", n=1).str[0],
        utc=True,
        errors="coerce",
    )
    return daily.dropna(subset=["day_start"]), weekly.dropna(subset=["week_start"])


def _objective(daily_pnl: pd.Series, weekly_pnl: pd.Series) -> Dict[str, float]:
    daily = pd.to_numeric(daily_pnl, errors="coerce").dropna()
    weekly = pd.to_numeric(weekly_pnl, errors="coerce").dropna()
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
    }


def _score_combo_window(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    combo_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, float]:
    d = daily.loc[
        daily["combo_id"].eq(combo_id)
        & daily["day_start"].ge(start)
        & daily["day_start"].lt(end)
    ]
    w = weekly.loc[
        weekly["combo_id"].eq(combo_id)
        & weekly["week_start"].ge(start)
        & weekly["week_start"].lt(end)
    ]
    return _objective(d["net_pnl"], w["net_pnl"])


def _select_for_week(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    champion_combo: str,
    challenger_combo: str,
    week_start: pd.Timestamp,
    lookback_weeks: int,
    min_history_weeks: int,
    mode: str,
    min_objective_delta: float,
) -> Tuple[str, str, Dict[str, float]]:
    history_start = week_start - pd.Timedelta(weeks=int(lookback_weeks))
    hist_weeks = weekly.loc[
        weekly["combo_id"].eq(champion_combo)
        & weekly["week_start"].ge(history_start)
        & weekly["week_start"].lt(week_start)
    ]
    if len(hist_weeks) < int(min_history_weeks):
        return champion_combo, "fallback_insufficient_history", {}

    champ = _score_combo_window(daily, weekly, champion_combo, history_start, week_start)
    chall = _score_combo_window(daily, weekly, challenger_combo, history_start, week_start)
    delta_obj = chall[OBJECTIVE_COL] - champ[OBJECTIVE_COL]
    delta_net = chall["net_pnl"] - champ["net_pnl"]
    delta_wq20 = chall["weekly_q20_pnl"] - champ["weekly_q20_pnl"]
    delta_dq20 = chall["daily_q20_pnl"] - champ["daily_q20_pnl"]
    stats = {
        "history_delta_objective": delta_obj,
        "history_delta_net_pnl": delta_net,
        "history_delta_weekly_q20": delta_wq20,
        "history_delta_daily_q20": delta_dq20,
    }
    if mode == "objective":
        use_challenger = delta_obj > float(min_objective_delta)
    elif mode == "net":
        use_challenger = delta_net > float(min_objective_delta)
    elif mode == "objective_tail_guard":
        use_challenger = delta_obj > float(min_objective_delta) and delta_wq20 >= 0.0 and delta_dq20 >= 0.0
    elif mode == "net_tail_guard":
        use_challenger = delta_net > float(min_objective_delta) and delta_wq20 >= 0.0 and delta_dq20 >= 0.0
    else:
        raise ValueError(f"Unknown mode `{mode}`")
    if use_challenger:
        return challenger_combo, f"selected_{mode}", stats
    return champion_combo, f"fallback_{mode}", stats


def _build_switch_stream(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    champion_combo: str,
    challenger_combo: str,
    lookback_weeks: int,
    min_history_weeks: int,
    mode: str,
    min_objective_delta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(weekly.loc[weekly["combo_id"].eq(champion_combo), "week_start"].dropna().unique())
    decision_rows: List[Dict[str, Any]] = []
    selected_weekly: List[pd.DataFrame] = []
    selected_daily: List[pd.DataFrame] = []
    for week_start_raw in weeks:
        week_start = pd.Timestamp(week_start_raw)
        selected_combo, reason, stats = _select_for_week(
            daily,
            weekly,
            champion_combo=champion_combo,
            challenger_combo=challenger_combo,
            week_start=week_start,
            lookback_weeks=lookback_weeks,
            min_history_weeks=min_history_weeks,
            mode=mode,
            min_objective_delta=min_objective_delta,
        )
        w = weekly.loc[weekly["combo_id"].eq(selected_combo) & weekly["week_start"].eq(week_start)].copy()
        d = daily.loc[
            daily["combo_id"].eq(selected_combo)
            & daily["day_start"].ge(week_start)
            & daily["day_start"].lt(week_start + pd.Timedelta(days=7))
        ].copy()
        if w.empty:
            continue
        w["selected_combo_id"] = selected_combo
        d["selected_combo_id"] = selected_combo
        selected_weekly.append(w)
        selected_daily.append(d)
        champion_w = weekly.loc[weekly["combo_id"].eq(champion_combo) & weekly["week_start"].eq(week_start)]
        champion_pnl = float(champion_w["net_pnl"].iloc[0]) if not champion_w.empty else np.nan
        selected_pnl = float(w["net_pnl"].iloc[0])
        decision_rows.append(
            {
                "champion_combo": champion_combo,
                "challenger_combo": challenger_combo,
                "lookback_weeks": int(lookback_weeks),
                "min_history_weeks": int(min_history_weeks),
                "selection_mode": mode,
                "min_objective_delta": float(min_objective_delta),
                "week": str(w["week"].iloc[0]),
                "week_start": str(week_start),
                "selected_combo": selected_combo,
                "selected_reason": reason,
                "selected_net_pnl": selected_pnl,
                "champion_net_pnl": champion_pnl,
                "delta_vs_champion": selected_pnl - champion_pnl,
                **stats,
            }
        )
    return (
        pd.DataFrame(decision_rows),
        pd.concat(selected_daily, ignore_index=True) if selected_daily else pd.DataFrame(),
        pd.concat(selected_weekly, ignore_index=True) if selected_weekly else pd.DataFrame(),
    )


def _score_stream_windows(
    decisions: pd.DataFrame,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    champion_daily: pd.DataFrame,
    champion_weekly: pd.DataFrame,
    windows: List[Tuple[str, pd.Timestamp | None, pd.Timestamp | None]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label, start, end in windows:
        d = daily
        w = weekly
        cd = champion_daily
        cw = champion_weekly
        if start is not None:
            d = d.loc[d["day_start"].ge(start)]
            w = w.loc[w["week_start"].ge(start)]
            cd = cd.loc[cd["day_start"].ge(start)]
            cw = cw.loc[cw["week_start"].ge(start)]
        if end is not None:
            d = d.loc[d["day_start"].le(end)]
            w = w.loc[w["week_start"].le(end)]
            cd = cd.loc[cd["day_start"].le(end)]
            cw = cw.loc[cw["week_start"].le(end)]
        selected = _objective(d["net_pnl"], w["net_pnl"])
        champ = _objective(cd["net_pnl"], cw["net_pnl"])
        rec = {
            "window": label,
            "selected_weeks": int(len(w)),
            "challenger_week_share": float(
                decisions.loc[
                    (pd.to_datetime(decisions["week_start"], utc=True, errors="coerce").isin(w["week_start"]))
                    & decisions["selected_combo"].eq(decisions["challenger_combo"])
                ].shape[0]
                / len(w)
            )
            if len(w)
            else 0.0,
        }
        for key, value in selected.items():
            rec[f"selected_{key}"] = value
        for key, value in champ.items():
            rec[f"champion_{key}"] = value
            rec[f"delta_{key}"] = selected[key] - value
        rec["pass_pnl_tail_gate"] = bool(
            rec[f"delta_{OBJECTIVE_COL}"] >= 0.0
            and rec["delta_net_pnl"] >= 0.0
            and rec["delta_weekly_q20_pnl"] >= 0.0
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--champion-combo", default=CHAMPION_COMBO)
    parser.add_argument("--challenger-combo", action="append", default=None)
    parser.add_argument("--lookback-weeks", default="2,4,8")
    parser.add_argument("--min-history-weeks", type=int, default=2)
    parser.add_argument("--selection-modes", default="objective,objective_tail_guard,net,net_tail_guard")
    parser.add_argument("--min-objective-delta", type=float, default=0.0)
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily, weekly = _load(args.replay_dir)
    challengers = args.challenger_combo or [
        combo
        for combo in sorted(weekly["combo_id"].dropna().astype(str).unique())
        if combo != args.champion_combo
    ]
    lookbacks = [int(v.strip()) for v in str(args.lookback_weeks).split(",") if v.strip()]
    modes = [str(v.strip()) for v in str(args.selection_modes).split(",") if v.strip()]
    max_day = daily["day_start"].max()
    windows = [
        ("full", None, max_day),
        ("validation_may_june", pd.Timestamp(args.validation_start, tz="UTC"), max_day),
        ("june_only", pd.Timestamp(args.june_start, tz="UTC"), max_day),
    ]
    champion_daily = daily.loc[daily["combo_id"].eq(args.champion_combo)].copy()
    champion_weekly = weekly.loc[weekly["combo_id"].eq(args.champion_combo)].copy()
    all_decisions: List[pd.DataFrame] = []
    all_scores: List[pd.DataFrame] = []
    for challenger in challengers:
        for lookback in lookbacks:
            for mode in modes:
                decisions, selected_daily, selected_weekly = _build_switch_stream(
                    daily,
                    weekly,
                    champion_combo=args.champion_combo,
                    challenger_combo=challenger,
                    lookback_weeks=lookback,
                    min_history_weeks=args.min_history_weeks,
                    mode=mode,
                    min_objective_delta=args.min_objective_delta,
                )
                if decisions.empty:
                    continue
                scores = _score_stream_windows(decisions, selected_daily, selected_weekly, champion_daily, champion_weekly, windows)
                for frame in (decisions, scores):
                    frame["challenger_combo"] = challenger
                    frame["lookback_weeks"] = lookback
                    frame["selection_mode"] = mode
                all_decisions.append(decisions)
                all_scores.append(scores)
    decisions_all = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    scores_all = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    decisions_all.to_csv(args.out_dir / "pairwise_combo_switch_decisions.csv", index=False)
    scores_all.to_csv(args.out_dir / "pairwise_combo_switch_scores.csv", index=False)

    validation = scores_all.loc[scores_all["window"].eq("validation_may_june")].copy()
    top_validation = validation.sort_values(f"delta_{OBJECTIVE_COL}", ascending=False).head(20)
    passing = validation.loc[validation["pass_pnl_tail_gate"].astype(bool)].sort_values(f"delta_{OBJECTIVE_COL}", ascending=False)
    show_cols = [
        "challenger_combo",
        "lookback_weeks",
        "selection_mode",
        "window",
        "challenger_week_share",
        f"delta_{OBJECTIVE_COL}",
        "delta_net_pnl",
        "delta_weekly_q20_pnl",
        "delta_daily_q20_pnl",
        "selected_net_pnl",
        "champion_net_pnl",
        "pass_pnl_tail_gate",
    ]
    lines = [
        "# Pairwise Combo Switching Audit",
        "",
        f"Champion: `{args.champion_combo}`",
        "Selection uses only prior weeks and recombines existing replay outputs; no portfolio replay is run here.",
        "Costs are included. This is a replay-period walk-forward audit, not untouched live OOS.",
        "",
        "## May-June Passing Switches",
        "",
        passing[[c for c in show_cols if c in passing.columns]].head(30).round(6).to_markdown(index=False)
        if not passing.empty
        else "_No May-June passing switches._",
        "",
        "## Top May-June Switches",
        "",
        top_validation[[c for c in show_cols if c in top_validation.columns]].round(6).to_markdown(index=False)
        if not top_validation.empty
        else "_No rows._",
        "",
        "## Interpretation",
        "",
        "- A switch passes only if it improves May-June objective, net PnL, and weekly Q20 versus the fixed champion.",
        "- Passing rows are candidates for a true replay implementation; failing rows remain diagnostics.",
    ]
    (args.out_dir / "pairwise_combo_switch_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "replay_dirs": [str(p) for p in args.replay_dir],
        "out_dir": str(args.out_dir),
        "champion_combo": args.champion_combo,
        "challenger_count": len(challengers),
        "rows": int(len(scores_all)),
    }
    (args.out_dir / "pairwise_combo_switch_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
