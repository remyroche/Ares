#!/usr/bin/env python3
"""Report chronological robustness for contextual TP/SL replay grids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def _period_objective(daily_pnl: pd.Series, weekly_pnl: pd.Series) -> Dict[str, float]:
    daily = pd.to_numeric(daily_pnl, errors="coerce").dropna()
    weekly = pd.to_numeric(weekly_pnl, errors="coerce").dropna()
    net_pnl = float(daily.sum()) if not daily.empty else float(weekly.sum())
    avg_week = float(weekly.mean()) if not weekly.empty else 0.0
    daily_q20 = float(daily.quantile(0.20)) if not daily.empty else 0.0
    daily_q35 = float(daily.quantile(0.35)) if not daily.empty else 0.0
    weekly_q05 = float(weekly.quantile(0.05)) if not weekly.empty else 0.0
    weekly_q10 = float(weekly.quantile(0.10)) if not weekly.empty else 0.0
    weekly_q20 = float(weekly.quantile(0.20)) if not weekly.empty else 0.0
    weekly_q35 = float(weekly.quantile(0.35)) if not weekly.empty else 0.0
    objective = avg_week + 0.7 * daily_q35 + 0.3 * daily_q20
    return {
        "objective_avgweek_0p7dayq35_0p3dayq20": float(objective),
        "avg_week_pnl": avg_week,
        "net_pnl": net_pnl,
        "daily_q20_pnl": daily_q20,
        "daily_q35_pnl": daily_q35,
        "weekly_q05_pnl": weekly_q05,
        "weekly_q10_pnl": weekly_q10,
        "weekly_q20_pnl": weekly_q20,
        "weekly_q35_pnl": weekly_q35,
        "daily_count": int(len(daily)),
        "weekly_count": int(len(weekly)),
        "positive_week_rate": float((weekly > 0.0).mean()) if not weekly.empty else np.nan,
    }


def _load_periods(replay_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(replay_dir / "conditional_filter_daily.csv")
    weekly = pd.read_csv(replay_dir / "conditional_filter_weekly.csv")
    daily["period_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly_start = weekly["week"].astype(str).str.split("/", n=1).str[0]
    weekly["period_start"] = pd.to_datetime(weekly_start, utc=True, errors="coerce")
    return daily.dropna(subset=["period_start"]), weekly.dropna(subset=["period_start"])


def _window_mask(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame["period_start"].ge(start)
    if end is not None:
        mask &= frame["period_start"].le(end)
    return mask


def _score_window(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    label: str,
) -> pd.DataFrame:
    d = daily.loc[_window_mask(daily, start, end)]
    w = weekly.loc[_window_mask(weekly, start, end)]
    keys = sorted(set(map(tuple, d[["combo_id", "rule_id"]].drop_duplicates().to_numpy())).union(
        set(map(tuple, w[["combo_id", "rule_id"]].drop_duplicates().to_numpy()))
    ))
    rows: List[Dict[str, Any]] = []
    for combo_id, rule_id in keys:
        d_sub = d.loc[d["combo_id"].eq(combo_id) & d["rule_id"].eq(rule_id)]
        w_sub = w.loc[w["combo_id"].eq(combo_id) & w["rule_id"].eq(rule_id)]
        rec: Dict[str, Any] = {
            "window": label,
            "combo_id": combo_id,
            "rule_id": rule_id,
            "start": str(start) if start is not None else "",
            "end": str(end) if end is not None else "",
            "trades": int(pd.to_numeric(d_sub.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        }
        rec.update(_period_objective(d_sub.get("net_pnl", pd.Series(dtype=float)), w_sub.get("net_pnl", pd.Series(dtype=float))))
        rows.append(rec)
    return pd.DataFrame(rows)


def _select_best(score: pd.DataFrame, combo_id: str | None = None) -> pd.Series:
    work = score
    if combo_id is not None:
        work = work.loc[work["combo_id"].eq(combo_id)]
    if work.empty:
        raise ValueError("No rows available for selection")
    return work.sort_values("objective_avgweek_0p7dayq35_0p3dayq20", ascending=False).iloc[0]


def _append_ref_deltas(score: pd.DataFrame, refs: Iterable[Tuple[str, str, str]]) -> pd.DataFrame:
    out = score.copy()
    for ref_name, combo_id, rule_id in refs:
        ref = out.loc[out["combo_id"].eq(combo_id) & out["rule_id"].eq(rule_id)]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for col in (
            "objective_avgweek_0p7dayq35_0p3dayq20",
            "avg_week_pnl",
            "net_pnl",
            "daily_q20_pnl",
            "daily_q35_pnl",
            "weekly_q05_pnl",
            "weekly_q10_pnl",
            "weekly_q20_pnl",
            "weekly_q35_pnl",
            "trades",
        ):
            out[f"delta_vs_{ref_name}_{col}"] = pd.to_numeric(out[col], errors="coerce") - float(ref[col])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-combo", default="long_bars:S_long_dist:S_short_asset:S_short_bollinger:S")
    parser.add_argument("--champion-combo", default="long_bars:S_long_dist:R_short_asset:R_short_bollinger:R")
    parser.add_argument("--fit-end", default="2026-04-30")
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily, weekly = _load_periods(args.replay_dir)
    fit_end = pd.Timestamp(args.fit_end, tz="UTC")
    validation_start = pd.Timestamp(args.validation_start, tz="UTC")
    june_start = pd.Timestamp(args.june_start, tz="UTC")
    max_day = daily["period_start"].max()
    windows = [
        ("full", None, max_day),
        ("fit_through_april", None, fit_end),
        ("validation_may_june", validation_start, max_day),
        ("june_only", june_start, max_day),
    ]
    frames = [_score_window(daily, weekly, start=start, end=end, label=label) for label, start, end in windows]
    scored = pd.concat(frames, ignore_index=True)
    refs = [
        ("all_static_none", args.baseline_combo, "none"),
        ("champ_none", args.champion_combo, "none"),
    ]
    scored = pd.concat(
        [_append_ref_deltas(group.copy(), refs) for _, group in scored.groupby("window", sort=False)],
        ignore_index=True,
    )
    scored.to_csv(args.out_dir / "split_robustness_scores.csv", index=False)

    fit_scores = scored.loc[scored["window"].eq("fit_through_april")]
    fit_best = _select_best(fit_scores, args.champion_combo)
    full_best = _select_best(scored.loc[scored["window"].eq("full")], args.champion_combo)
    selected = []
    for name, row in (("fit_selected_champion_combo", fit_best), ("full_selected_champion_combo", full_best)):
        for window in ("full", "fit_through_april", "validation_may_june", "june_only"):
            match = scored.loc[
                scored["window"].eq(window)
                & scored["combo_id"].eq(row["combo_id"])
                & scored["rule_id"].eq(row["rule_id"])
            ]
            if match.empty:
                continue
            rec = match.iloc[0].to_dict()
            rec["selection"] = name
            selected.append(rec)
    selected_df = pd.DataFrame(selected)
    selected_df.to_csv(args.out_dir / "split_robustness_selected_rules.csv", index=False)

    top_full = scored.loc[scored["window"].eq("full")].sort_values(
        "objective_avgweek_0p7dayq35_0p3dayq20", ascending=False
    ).head(15)
    top_validation = scored.loc[scored["window"].eq("validation_may_june")].sort_values(
        "objective_avgweek_0p7dayq35_0p3dayq20", ascending=False
    ).head(15)
    top_june = scored.loc[scored["window"].eq("june_only")].sort_values(
        "objective_avgweek_0p7dayq35_0p3dayq20", ascending=False
    ).head(15)

    show_cols = [
        "window",
        "combo_id",
        "rule_id",
        "objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_all_static_none_objective_avgweek_0p7dayq35_0p3dayq20",
        "delta_vs_champ_none_objective_avgweek_0p7dayq35_0p3dayq20",
        "net_pnl",
        "delta_vs_all_static_none_net_pnl",
        "delta_vs_champ_none_net_pnl",
        "weekly_q20_pnl",
        "daily_q20_pnl",
        "trades",
    ]
    lines = [
        "# Contextual TP/SL Split Robustness",
        "",
        f"Replay dir: `{args.replay_dir}`",
        f"Fit selection window: through `{args.fit_end}`",
        f"Validation window: `{args.validation_start}` through `{max_day.date()}`",
        "Costs are included. This is a replay-period split audit, not untouched live OOS.",
        "",
        "## Fit-Selected And Full-Selected Rules",
        "",
        selected_df[[c for c in show_cols + ["selection"] if c in selected_df.columns]]
        .round(6)
        .to_markdown(index=False),
        "",
        "## Top Full-Period Rules",
        "",
        top_full[[c for c in show_cols if c in top_full.columns]].round(6).to_markdown(index=False),
        "",
        "## Top May-June Validation Rules",
        "",
        top_validation[[c for c in show_cols if c in top_validation.columns]].round(6).to_markdown(index=False),
        "",
        "## Top June-Only Rules",
        "",
        top_june[[c for c in show_cols if c in top_june.columns]].round(6).to_markdown(index=False),
    ]
    payload = {
        "replay_dir": str(args.replay_dir),
        "out_dir": str(args.out_dir),
        "fit_best": fit_best.to_dict(),
        "full_best": full_best.to_dict(),
        "windows": [{"label": label, "start": str(start), "end": str(end)} for label, start, end in windows],
    }
    (args.out_dir / "split_robustness_report.md").write_text("\n".join(lines) + "\n")
    (args.out_dir / "split_robustness_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"out_dir": str(args.out_dir), "rows": int(len(scored))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
