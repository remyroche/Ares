#!/usr/bin/env python3
"""Chronological robustness report for contextual TP/SL head-repair combos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sweep_contextual_tp_sl_arm_combinations import _head_name  # noqa: E402


OBJECTIVE_COL = "objective_avgweek_0p7dayq35_0p3dayq20"
STATIC_COMBO = "long_bars:S_long_dist:S_short_asset:S_short_bollinger:S"
CHAMPION_COMBO = "long_bars:S_long_dist:R_short_asset:R_short_bollinger:R"


def _read_first_existing(root: Path, names: Iterable[str]) -> pd.DataFrame:
    for name in names:
        path = root / name
        if path.exists():
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
    raise FileNotFoundError(f"No expected files in {root}: {list(names)}")


def _load_dirs(replay_dirs: List[Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    summary_frames: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []
    for root in replay_dirs:
        daily = _read_first_existing(root, ["head_arm_combination_daily.csv"])
        weekly = _read_first_existing(root, ["head_arm_combination_weekly.csv"])
        summary = _read_first_existing(root, ["head_arm_combination_summary.csv"])
        accepted = _read_first_existing(root, ["head_arm_combination_accepted_decisions.parquet"])
        for frame in (daily, weekly, summary, accepted):
            frame["_source_replay_dir"] = str(root)
        daily_frames.append(daily)
        weekly_frames.append(weekly)
        summary_frames.append(summary)
        accepted_frames.append(accepted)
    daily_all = pd.concat(daily_frames, ignore_index=True).drop_duplicates(["combo_id", "day"])
    weekly_all = pd.concat(weekly_frames, ignore_index=True).drop_duplicates(["combo_id", "week"])
    summary_all = pd.concat(summary_frames, ignore_index=True).drop_duplicates(["combo_id"])
    accepted_all = pd.concat(accepted_frames, ignore_index=True)
    # Some frontier reports intentionally reuse the same combo replays. Avoid
    # double-counting shared accepted decisions when computing per-head deltas.
    dedupe_cols = [
        col
        for col in (
            "combo_id",
            "timestamp",
            "entry_time",
            "exit_time",
            "strategy_id",
            "symbol",
            "position_size",
            "position_net_return",
            "position_exit_reason",
        )
        if col in accepted_all.columns
    ]
    if dedupe_cols:
        accepted_all = accepted_all.drop_duplicates(dedupe_cols)
    else:
        accepted_all = accepted_all.drop_duplicates()
    return daily_all, weekly_all, summary_all, accepted_all


def _period_objective(daily_pnl: pd.Series, weekly_pnl: pd.Series) -> Dict[str, float]:
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


def _window_mask(frame: pd.DataFrame, period_col: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame[period_col].ge(start)
    if end is not None:
        mask &= frame[period_col].le(end)
    return mask


def _score_global_windows(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    summary: pd.DataFrame,
    windows: List[Tuple[str, pd.Timestamp | None, pd.Timestamp | None]],
) -> pd.DataFrame:
    daily = daily.copy()
    weekly = weekly.copy()
    daily["period_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["period_start"] = pd.to_datetime(weekly["week"].astype(str).str.split("/", n=1).str[0], utc=True, errors="coerce")
    rows: List[Dict[str, Any]] = []
    combos = sorted(summary["combo_id"].dropna().astype(str).unique())
    arm_cols = [c for c in ["long_bars_arm", "long_dist_arm", "short_asset_arm", "short_bollinger_arm"] if c in summary.columns]
    arms = summary.set_index("combo_id")[arm_cols].to_dict(orient="index")
    for label, start, end in windows:
        d_win = daily.loc[_window_mask(daily, "period_start", start, end)]
        w_win = weekly.loc[_window_mask(weekly, "period_start", start, end)]
        for combo_id in combos:
            d = d_win.loc[d_win["combo_id"].eq(combo_id)]
            w = w_win.loc[w_win["combo_id"].eq(combo_id)]
            rec: Dict[str, Any] = {"window": label, "combo_id": combo_id}
            rec.update(arms.get(combo_id, {}))
            rec.update(_period_objective(d["net_pnl"], w["net_pnl"]))
            rec["trades"] = int(pd.to_numeric(d.get("trades", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
            rows.append(rec)
    return pd.DataFrame(rows)


def _prepare_accepted(accepted: pd.DataFrame) -> pd.DataFrame:
    work = accepted.copy()
    if "timestamp" in work.columns:
        work["period_start"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    elif "entry_time" in work.columns:
        work["period_start"] = pd.to_datetime(work["entry_time"], utc=True, errors="coerce")
    else:
        raise ValueError("accepted decisions need timestamp or entry_time")
    work["head"] = work["strategy_id"].astype(str).map(_head_name)
    size = pd.to_numeric(work.get("position_size", 0.0), errors="coerce").fillna(0.0)
    net = pd.to_numeric(work.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    gross = pd.to_numeric(work.get("position_gross_return", 0.0), errors="coerce").fillna(0.0)
    work["net_pnl_calc"] = size * net
    work["gross_pnl_calc"] = size * gross
    work["hit"] = net.gt(0.0)
    work["full_sl"] = work.get("position_exit_reason", "").astype(str).eq("full_sl")
    return work.dropna(subset=["period_start"])


def _score_head_windows(
    accepted: pd.DataFrame,
    windows: List[Tuple[str, pd.Timestamp | None, pd.Timestamp | None]],
) -> pd.DataFrame:
    work = _prepare_accepted(accepted)
    rows: List[Dict[str, Any]] = []
    for label, start, end in windows:
        win = work.loc[_window_mask(work, "period_start", start, end)]
        grouped = (
            win.groupby(["combo_id", "head"], dropna=False)
            .agg(
                trades=("head", "size"),
                net_pnl=("net_pnl_calc", "sum"),
                gross_pnl=("gross_pnl_calc", "sum"),
                hit_rate=("hit", "mean"),
                full_sl_rate=("full_sl", "mean"),
            )
            .reset_index()
        )
        grouped.insert(0, "window", label)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _append_ref_deltas(global_scores: pd.DataFrame, refs: List[Tuple[str, str]]) -> pd.DataFrame:
    out_frames: List[pd.DataFrame] = []
    for window, group in global_scores.groupby("window", sort=False):
        group = group.copy()
        for ref_name, combo_id in refs:
            ref = group.loc[group["combo_id"].eq(combo_id)]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            for col in (OBJECTIVE_COL, "avg_week_pnl", "net_pnl", "daily_q20_pnl", "daily_q35_pnl", "weekly_q05_pnl", "weekly_q10_pnl", "weekly_q20_pnl", "weekly_q35_pnl", "trades"):
                group[f"delta_vs_{ref_name}_{col}"] = pd.to_numeric(group[col], errors="coerce") - float(ref[col])
        out_frames.append(group)
    return pd.concat(out_frames, ignore_index=True)


def _append_head_ref_deltas(head_scores: pd.DataFrame, refs: List[Tuple[str, str]]) -> pd.DataFrame:
    out_frames: List[pd.DataFrame] = []
    for (window, head), group in head_scores.groupby(["window", "head"], sort=False):
        group = group.copy()
        for ref_name, combo_id in refs:
            ref = group.loc[group["combo_id"].eq(combo_id)]
            if ref.empty:
                continue
            ref = ref.iloc[0]
            for col in ("trades", "net_pnl", "gross_pnl", "hit_rate", "full_sl_rate"):
                group[f"delta_vs_{ref_name}_{col}"] = pd.to_numeric(group[col], errors="coerce") - float(ref[col])
        out_frames.append(group)
    return pd.concat(out_frames, ignore_index=True)


def _candidate_gate(global_scores: pd.DataFrame, head_scores: pd.DataFrame, window: str, combo_id: str) -> Dict[str, Any]:
    g = global_scores.loc[global_scores["window"].eq(window) & global_scores["combo_id"].eq(combo_id)]
    if g.empty:
        return {}
    g = g.iloc[0]
    heads = head_scores.loc[head_scores["window"].eq(window) & head_scores["combo_id"].eq(combo_id)]
    head_map = heads.set_index("head").to_dict(orient="index")
    profitable_ok = all(
        head_map.get(h, {}).get("delta_vs_champ_net_pnl", -np.inf) >= -2500.0
        for h in ("long_dist", "short_bollinger")
    )
    weak_improves = (
        head_map.get("short_asset", {}).get("delta_vs_champ_net_pnl", -np.inf) > 0
        or head_map.get("long_bars", {}).get("delta_vs_champ_net_pnl", -np.inf) > 0
    )
    global_ok = (
        float(g.get(f"delta_vs_champ_{OBJECTIVE_COL}", -np.inf)) >= 0.0
        and float(g.get("delta_vs_champ_net_pnl", -np.inf)) >= 0.0
        and float(g.get("delta_vs_champ_weekly_q20_pnl", -np.inf)) >= 0.0
    )
    return {
        "combo_id": combo_id,
        "window": window,
        "global_ok": bool(global_ok),
        "profitable_heads_ok": bool(profitable_ok),
        "weak_head_improves_vs_champ": bool(weak_improves),
        "pass_head_repair_gate": bool(global_ok and profitable_ok and weak_improves),
        "delta_objective_vs_champ": float(g.get(f"delta_vs_champ_{OBJECTIVE_COL}", np.nan)),
        "delta_net_pnl_vs_champ": float(g.get("delta_vs_champ_net_pnl", np.nan)),
        "delta_weekly_q20_vs_champ": float(g.get("delta_vs_champ_weekly_q20_pnl", np.nan)),
        "delta_long_dist_vs_champ": float(head_map.get("long_dist", {}).get("delta_vs_champ_net_pnl", np.nan)),
        "delta_short_bollinger_vs_champ": float(head_map.get("short_bollinger", {}).get("delta_vs_champ_net_pnl", np.nan)),
        "delta_short_asset_vs_champ": float(head_map.get("short_asset", {}).get("delta_vs_champ_net_pnl", np.nan)),
        "delta_long_bars_vs_champ": float(head_map.get("long_bars", {}).get("delta_vs_champ_net_pnl", np.nan)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fit-end", default="2026-04-30")
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily, weekly, summary, accepted = _load_dirs(args.replay_dir)
    max_day = pd.to_datetime(daily["day"], utc=True, errors="coerce").max()
    windows = [
        ("full", None, max_day),
        ("fit_through_april", None, pd.Timestamp(args.fit_end, tz="UTC")),
        ("validation_may_june", pd.Timestamp(args.validation_start, tz="UTC"), max_day),
        ("june_only", pd.Timestamp(args.june_start, tz="UTC"), max_day),
    ]
    refs = [("static", STATIC_COMBO), ("champ", CHAMPION_COMBO)]
    global_scores = _append_ref_deltas(_score_global_windows(daily, weekly, summary, windows), refs)
    head_scores = _append_head_ref_deltas(_score_head_windows(accepted, windows), refs)

    global_scores.to_csv(args.out_dir / "head_repair_global_window_scores.csv", index=False)
    head_scores.to_csv(args.out_dir / "head_repair_per_head_window_scores.csv", index=False)

    fit = global_scores.loc[global_scores["window"].eq("fit_through_april")]
    fit_champ_locked = fit.loc[
        fit["long_dist_arm"].eq("rank_only")
        & fit["short_bollinger_arm"].eq("rank_only")
    ]
    fit_best_locked = fit_champ_locked.sort_values(OBJECTIVE_COL, ascending=False).head(1)
    full_top = global_scores.loc[global_scores["window"].eq("full")].sort_values(OBJECTIVE_COL, ascending=False).head(15)
    validation_top = global_scores.loc[global_scores["window"].eq("validation_may_june")].sort_values(OBJECTIVE_COL, ascending=False).head(15)

    gate_rows: List[Dict[str, Any]] = []
    candidate_ids = set(full_top["combo_id"]).union(validation_top["combo_id"]).union(fit_best_locked["combo_id"])
    candidate_ids.add(CHAMPION_COMBO)
    for combo_id in sorted(candidate_ids):
        for window in ("full", "fit_through_april", "validation_may_june", "june_only"):
            rec = _candidate_gate(global_scores, head_scores, window, combo_id)
            if rec:
                gate_rows.append(rec)
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(args.out_dir / "head_repair_gate_summary.csv", index=False)

    show_cols = [
        "window",
        "combo_id",
        "long_bars_arm",
        "long_dist_arm",
        "short_asset_arm",
        "short_bollinger_arm",
        OBJECTIVE_COL,
        f"delta_vs_champ_{OBJECTIVE_COL}",
        "net_pnl",
        "delta_vs_champ_net_pnl",
        "weekly_q20_pnl",
        "delta_vs_champ_weekly_q20_pnl",
        "daily_q20_pnl",
        "trades",
    ]
    gate_cols = [
        "window",
        "combo_id",
        "pass_head_repair_gate",
        "global_ok",
        "profitable_heads_ok",
        "weak_head_improves_vs_champ",
        "delta_objective_vs_champ",
        "delta_net_pnl_vs_champ",
        "delta_weekly_q20_vs_champ",
        "delta_long_dist_vs_champ",
        "delta_short_bollinger_vs_champ",
        "delta_short_asset_vs_champ",
        "delta_long_bars_vs_champ",
    ]
    selected_sections: List[str] = []
    if not fit_best_locked.empty:
        combo_id = str(fit_best_locked.iloc[0]["combo_id"])
        selected_sections.append(f"Fit-through-April locked-head selection: `{combo_id}`")
        selected_sections.append("")
        selected_sections.append(
            global_scores.loc[global_scores["combo_id"].eq(combo_id), [c for c in show_cols if c in global_scores.columns]]
            .round(6)
            .to_markdown(index=False)
        )
    selected_lines = selected_sections if selected_sections else ["_No locked-head fit selection available._"]
    lines = [
        "# Head-Specific Repair Robustness",
        "",
        "Replay dirs:",
        *[f"- `{p}`" for p in args.replay_dir],
        "",
        "References: all-static and S/R/R/R champion. Costs are included.",
        "",
        "## Fit-Selected Locked Profitable Heads",
        "",
        *selected_lines,
        "",
        "## Top Full-Period Combos",
        "",
        full_top[[c for c in show_cols if c in full_top.columns]].round(6).to_markdown(index=False),
        "",
        "## Top May-June Combos",
        "",
        validation_top[[c for c in show_cols if c in validation_top.columns]].round(6).to_markdown(index=False),
        "",
        "## Gate Summary For Candidate Set",
        "",
        gate[[c for c in gate_cols if c in gate.columns]].round(6).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- No candidate should pass unless it improves global objective/net/tail versus S/R/R/R, does not materially damage long_dist or short_bollinger, and improves at least one weak head versus S/R/R/R.",
        "- This report uses accepted decisions for per-head PnL, so head damage is measured after portfolio admission, not from standalone arm metrics.",
    ]
    (args.out_dir / "head_repair_robustness_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "replay_dirs": [str(p) for p in args.replay_dir],
        "out_dir": str(args.out_dir),
        "combo_count": int(summary["combo_id"].nunique()),
        "accepted_rows": int(len(accepted)),
    }
    (args.out_dir / "head_repair_robustness_summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
