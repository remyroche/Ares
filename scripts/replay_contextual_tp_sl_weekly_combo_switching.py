#!/usr/bin/env python3
"""Replay weekly contextual TP/SL combo switching through the full portfolio auction.

Unlike ``report_contextual_tp_sl_pairwise_combo_switching.py``, this script does
not recombine already accepted trades. It uses prior-week combo metrics only to
choose which head-arm combo should be active for each future week, then builds a
single candidate ledger and runs the normal portfolio replay once so open
positions, capacity, symbol conflicts, and costs remain live-equivalent within
the replay.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    replay_candidates,
)
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    _accepted_period_tables,
    _head_name,
    _json_safe,
    _period_metrics,
)
from scripts.report_contextual_tp_sl_pairwise_combo_switching import (  # noqa: E402
    CHAMPION_COMBO,
    OBJECTIVE_COL,
    _load,
    _objective,
    _select_for_week,
)


LABEL_TO_ARM = {
    "S": "static",
    "R": "rank_only",
    "P": "performance_only",
    "J": "joint_all",
    "I": "independent_all",
}
ARM_TO_LABEL = {v: k for k, v in LABEL_TO_ARM.items()}
HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")


def _parse_combo_id(combo_id: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    pattern = "|".join(re.escape(head) for head in HEAD_ORDER)
    for head, label in re.findall(rf"({pattern}):([A-Z])", str(combo_id)):
        if label not in LABEL_TO_ARM:
            raise ValueError(f"Unknown arm label `{label}` in combo `{combo_id}`")
        mapping[head] = LABEL_TO_ARM[label]
    missing = [head for head in HEAD_ORDER if head not in mapping]
    if missing:
        raise ValueError(f"Combo `{combo_id}` is missing heads: {missing}")
    return mapping


def _combo_id(mapping: Mapping[str, str]) -> str:
    return "_".join(f"{head}:{ARM_TO_LABEL[mapping[head]]}" for head in HEAD_ORDER)


def _read_arm_table(source_dir: Path, arm: str) -> pd.DataFrame:
    path = source_dir / "portfolio_replay" / f"{arm}_contextual_tp_sl_candidates.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate table: {path}")
    frame = pd.read_parquet(path)
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["head"] = frame["strategy_id"].map(_head_name)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.dropna(subset=["timestamp"])


def _load_arm_tables(source_dir: Path, combos: Sequence[str]) -> Dict[str, pd.DataFrame]:
    needed = sorted({arm for combo in combos for arm in _parse_combo_id(combo).values()})
    return {arm: _read_arm_table(source_dir, arm) for arm in needed}


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    naive = ts.dt.tz_convert(None)
    starts = naive.dt.to_period("W").dt.start_time
    return pd.to_datetime(starts, utc=True, errors="coerce")


def _build_combo_candidates(
    tables: Mapping[str, pd.DataFrame],
    combo_mapping: Mapping[str, str],
    *,
    week_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for head, arm in combo_mapping.items():
        source = tables[arm]
        frame = source.loc[source["head"].eq(head)].copy()
        if week_start is not None:
            if "_week_start" not in frame.columns:
                frame["_week_start"] = _week_start(frame["timestamp"])
            frame = frame.loc[frame["_week_start"].eq(week_start)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["head", "_week_start"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )


def _decision_weeks(weekly: pd.DataFrame, champion_combo: str) -> List[pd.Timestamp]:
    values = weekly.loc[weekly["combo_id"].eq(champion_combo), "week_start"].dropna().unique()
    return [pd.Timestamp(v) for v in sorted(values)]


def _build_switch_candidate_stream(
    tables: Mapping[str, pd.DataFrame],
    daily_metrics: pd.DataFrame,
    weekly_metrics: pd.DataFrame,
    *,
    champion_combo: str,
    challenger_combo: str,
    lookback_weeks: int,
    min_history_weeks: int,
    selection_mode: str,
    min_objective_delta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    candidate_frames: List[pd.DataFrame] = []
    champion_mapping = _parse_combo_id(champion_combo)
    challenger_mapping = _parse_combo_id(challenger_combo)
    for week_start in _decision_weeks(weekly_metrics, champion_combo):
        selected_combo, reason, stats = _select_for_week(
            daily_metrics,
            weekly_metrics,
            champion_combo=champion_combo,
            challenger_combo=challenger_combo,
            week_start=week_start,
            lookback_weeks=lookback_weeks,
            min_history_weeks=min_history_weeks,
            mode=selection_mode,
            min_objective_delta=min_objective_delta,
        )
        mapping = challenger_mapping if selected_combo == challenger_combo else champion_mapping
        week_candidates = _build_combo_candidates(tables, mapping, week_start=week_start)
        if week_candidates.empty:
            continue
        week_candidates["selected_combo_id"] = selected_combo
        week_candidates["selected_challenger"] = selected_combo == challenger_combo
        candidate_frames.append(week_candidates)
        rows.append(
            {
                "week_start": week_start.isoformat(),
                "champion_combo": champion_combo,
                "challenger_combo": challenger_combo,
                "selected_combo": selected_combo,
                "selected_reason": reason,
                "lookback_weeks": int(lookback_weeks),
                "min_history_weeks": int(min_history_weeks),
                "selection_mode": selection_mode,
                "min_objective_delta": float(min_objective_delta),
                "candidate_rows": int(len(week_candidates)),
                **stats,
            }
        )
    if not candidate_frames:
        return pd.DataFrame(), pd.DataFrame(rows)
    candidates = (
        pd.concat(candidate_frames, ignore_index=True)
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    return candidates, pd.DataFrame(rows)


def _score_windows(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    baseline_weekly: pd.DataFrame,
    *,
    validation_start: str,
    june_start: str,
) -> pd.DataFrame:
    d = daily.copy()
    w = weekly.copy()
    bd = baseline_daily.copy()
    bw = baseline_weekly.copy()
    for frame, col in ((d, "day"), (bd, "day")):
        if not frame.empty:
            frame["day_start"] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    for frame in (w, bw):
        if not frame.empty:
            frame["week_start"] = pd.to_datetime(
                frame["week"].astype(str).str.split("/", n=1).str[0],
                utc=True,
                errors="coerce",
            )
    max_day = d["day_start"].max() if not d.empty else bd["day_start"].max()
    windows = [
        ("full", None, max_day),
        ("validation_may_june", pd.Timestamp(validation_start, tz="UTC"), max_day),
        ("june_only", pd.Timestamp(june_start, tz="UTC"), max_day),
    ]
    rows: List[Dict[str, Any]] = []
    for label, start, end in windows:
        cur_d, cur_w, base_d, base_w = d, w, bd, bw
        if start is not None:
            cur_d = cur_d.loc[cur_d["day_start"].ge(start)]
            cur_w = cur_w.loc[cur_w["week_start"].ge(start)]
            base_d = base_d.loc[base_d["day_start"].ge(start)]
            base_w = base_w.loc[base_w["week_start"].ge(start)]
        if end is not None:
            cur_d = cur_d.loc[cur_d["day_start"].le(end)]
            cur_w = cur_w.loc[cur_w["week_start"].le(end)]
            base_d = base_d.loc[base_d["day_start"].le(end)]
            base_w = base_w.loc[base_w["week_start"].le(end)]
        selected = _objective(cur_d.get("net_pnl", pd.Series(dtype=float)), cur_w.get("net_pnl", pd.Series(dtype=float)))
        baseline = _objective(base_d.get("net_pnl", pd.Series(dtype=float)), base_w.get("net_pnl", pd.Series(dtype=float)))
        rec: Dict[str, Any] = {"window": label}
        for key, value in selected.items():
            rec[f"selected_{key}"] = value
        for key, value in baseline.items():
            rec[f"baseline_{key}"] = value
            rec[f"delta_{key}"] = selected[key] - value
        rec["pass_pnl_tail_gate"] = bool(
            rec[f"delta_{OBJECTIVE_COL}"] >= 0.0
            and rec["delta_net_pnl"] >= 0.0
            and rec["delta_weekly_q20_pnl"] >= 0.0
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _summarize_decisions(decisions: pd.DataFrame) -> Dict[str, Any]:
    if decisions.empty:
        return {"weeks": 0, "challenger_week_share": 0.0}
    selected_challenger = decisions["selected_combo"].eq(decisions["challenger_combo"])
    return {
        "weeks": int(len(decisions)),
        "challenger_weeks": int(selected_challenger.sum()),
        "challenger_week_share": float(selected_challenger.mean()),
    }


def _run_replay(candidates: pd.DataFrame, market_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ev_curve = fit_hierarchical_ev_curves(candidates)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    return replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--champion-combo", default=CHAMPION_COMBO)
    parser.add_argument("--challenger-combo", action="append", required=True)
    parser.add_argument("--lookback-weeks", default="8")
    parser.add_argument("--min-history-weeks", type=int, default=2)
    parser.add_argument("--selection-modes", default="objective,net")
    parser.add_argument("--min-objective-delta", type=float, default=0.0)
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    parser.add_argument("--save-decisions", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily_metrics, weekly_metrics = _load(args.replay_dir)
    challengers = [str(v) for v in args.challenger_combo]
    lookbacks = [int(v.strip()) for v in str(args.lookback_weeks).split(",") if v.strip()]
    modes = [str(v.strip()) for v in str(args.selection_modes).split(",") if v.strip()]
    combos = [args.champion_combo, *challengers]
    tables = _load_arm_tables(args.source_dir, combos)

    baseline_candidates = _build_combo_candidates(tables, _parse_combo_id(args.champion_combo))
    baseline_decisions, _baseline_equity, baseline_metrics = _run_replay(baseline_candidates, args.market_mode)
    baseline_daily, baseline_weekly = _accepted_period_tables(baseline_decisions)

    score_rows: List[pd.DataFrame] = []
    decision_rows: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    for challenger in challengers:
        for lookback in lookbacks:
            for mode in modes:
                candidates, switch_decisions = _build_switch_candidate_stream(
                    tables,
                    daily_metrics,
                    weekly_metrics,
                    champion_combo=args.champion_combo,
                    challenger_combo=challenger,
                    lookback_weeks=lookback,
                    min_history_weeks=args.min_history_weeks,
                    selection_mode=mode,
                    min_objective_delta=args.min_objective_delta,
                )
                if candidates.empty:
                    continue
                decisions, _equity, metrics = _run_replay(candidates, args.market_mode)
                daily, weekly = _accepted_period_tables(decisions)
                scores = _score_windows(
                    daily,
                    weekly,
                    baseline_daily,
                    baseline_weekly,
                    validation_start=args.validation_start,
                    june_start=args.june_start,
                )
                meta = {
                    "champion_combo": args.champion_combo,
                    "challenger_combo": challenger,
                    "lookback_weeks": int(lookback),
                    "selection_mode": mode,
                    "min_history_weeks": int(args.min_history_weeks),
                    "min_objective_delta": float(args.min_objective_delta),
                    "total_candidate_rows": int(len(candidates)),
                    "trade_count": int(metrics.get("trade_count", 0) or 0),
                    "net_pnl": float(metrics.get("net_pnl", 0.0)),
                    "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                    **_summarize_decisions(switch_decisions),
                }
                for key, value in meta.items():
                    scores[key] = value
                    switch_decisions[key] = value
                    if not daily.empty:
                        daily[key] = value
                    if not weekly.empty:
                        weekly[key] = value
                score_rows.append(scores)
                decision_rows.append(switch_decisions)
                daily_frames.append(daily)
                weekly_frames.append(weekly)
                if args.save_decisions and "accepted" in decisions.columns:
                    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
                    if not accepted.empty:
                        for key, value in meta.items():
                            accepted[key] = value
                        accepted_frames.append(accepted)

    scores_all = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    decisions_all = pd.concat(decision_rows, ignore_index=True) if decision_rows else pd.DataFrame()
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    weekly_all = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    scores_all.to_csv(args.out_dir / "weekly_combo_switch_full_replay_scores.csv", index=False)
    decisions_all.to_csv(args.out_dir / "weekly_combo_switch_decisions.csv", index=False)
    daily_all.to_csv(args.out_dir / "weekly_combo_switch_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "weekly_combo_switch_weekly.csv", index=False)
    baseline_daily.to_csv(args.out_dir / "baseline_daily.csv", index=False)
    baseline_weekly.to_csv(args.out_dir / "baseline_weekly.csv", index=False)
    if args.save_decisions:
        accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
        accepted_all.to_parquet(args.out_dir / "weekly_combo_switch_accepted_decisions.parquet", index=False)

    validation = scores_all.loc[scores_all["window"].eq("validation_may_june")].copy()
    top_validation = validation.sort_values(f"delta_{OBJECTIVE_COL}", ascending=False).head(20)
    passing = validation.loc[validation["pass_pnl_tail_gate"].astype(bool)].sort_values(
        f"delta_{OBJECTIVE_COL}",
        ascending=False,
    )
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
        "baseline_net_pnl",
        "pass_pnl_tail_gate",
    ]
    lines = [
        "# Weekly Combo Switching Full Replay",
        "",
        f"Source: `{args.source_dir}`",
        f"Champion: `{args.champion_combo}`",
        "Selection uses prior standalone combo metrics, then the selected weekly candidate stream is replayed once through the full portfolio auction.",
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
        "## Baseline Metrics",
        "",
        json.dumps(_json_safe(baseline_metrics), indent=2),
    ]
    (args.out_dir / "weekly_combo_switch_full_replay_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "replay_dirs": [str(p) for p in args.replay_dir],
        "out_dir": str(args.out_dir),
        "champion_combo": args.champion_combo,
        "challenger_count": len(challengers),
        "rows": int(len(scores_all)),
        "baseline_metrics": _json_safe(baseline_metrics),
    }
    (args.out_dir / "weekly_combo_switch_full_replay_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2)
    )
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(scores_all)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
