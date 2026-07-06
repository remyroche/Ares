#!/usr/bin/env python3
"""One-week full-replay intervention oracle for contextual TP/SL rules.

Each trial replaces the baseline combo for exactly one calendar week with a
candidate ``combo + conditional rule`` and replays the full portfolio. This is
more expensive than accepted-trade recombination, but it preserves portfolio
state, capacity, symbol conflicts, cooldowns, and downstream path effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.replay_contextual_tp_sl_weekly_combo_switching import (  # noqa: E402
    CHAMPION_COMBO,
    OBJECTIVE_COL,
    _build_combo_candidates,
    _load_arm_tables,
    _parse_combo_id,
)
from scripts.replay_contextual_tp_sl_weekly_rule_selector import (  # noqa: E402
    _build_candidate_cache,
    _run_replay,
    _score_windows,
)
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    _accepted_period_tables,
    _json_safe,
)


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    naive = ts.dt.tz_convert(None)
    starts = naive.dt.to_period("W").dt.start_time
    return pd.to_datetime(starts, utc=True, errors="coerce")


def _load_baseline_tables(baseline_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(baseline_dir / "baseline_daily.csv"), pd.read_csv(baseline_dir / "baseline_weekly.csv")


def _candidate_id(candidate_combo: str, rule_id: str) -> str:
    return f"{candidate_combo}|{rule_id}"


def _available_weeks(baseline_candidates: pd.DataFrame, start_week: str | None, end_week: str | None) -> List[pd.Timestamp]:
    work = baseline_candidates.copy()
    work["_week_start"] = _week_start(work["timestamp"])
    weeks = sorted(pd.Timestamp(v) for v in work["_week_start"].dropna().unique())
    if start_week:
        start = pd.Timestamp(start_week, tz="UTC")
        weeks = [w for w in weeks if w >= start]
    if end_week:
        end = pd.Timestamp(end_week, tz="UTC")
        weeks = [w for w in weeks if w <= end]
    return weeks


def _stream_for_one_week(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    intervention_week: pd.Timestamp,
) -> pd.DataFrame:
    if "_week_start" not in baseline.columns:
        baseline = baseline.copy()
        baseline["_week_start"] = _week_start(baseline["timestamp"])
    if "_week_start" not in candidate.columns:
        candidate = candidate.copy()
        candidate["_week_start"] = _week_start(candidate["timestamp"])
    frames = [
        baseline.loc[~baseline["_week_start"].eq(intervention_week)].copy(),
        candidate.loc[candidate["_week_start"].eq(intervention_week)].copy(),
    ]
    stream = (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["_week_start"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    return stream


def _intervention_week_score(
    weekly: pd.DataFrame,
    baseline_weekly: pd.DataFrame,
    intervention_week: pd.Timestamp,
) -> Dict[str, float]:
    work = weekly.copy()
    base = baseline_weekly.copy()
    for frame in (work, base):
        frame["week_start"] = pd.to_datetime(
            frame["week"].astype(str).str.split("/", n=1).str[0],
            utc=True,
            errors="coerce",
        )
    cur = work.loc[work["week_start"].eq(intervention_week)]
    ref = base.loc[base["week_start"].eq(intervention_week)]
    cur_pnl = float(pd.to_numeric(cur.get("net_pnl", pd.Series(dtype=float)), errors="coerce").sum()) if not cur.empty else 0.0
    ref_pnl = float(pd.to_numeric(ref.get("net_pnl", pd.Series(dtype=float)), errors="coerce").sum()) if not ref.empty else 0.0
    return {
        "intervention_week_net_pnl": cur_pnl,
        "baseline_intervention_week_net_pnl": ref_pnl,
        "delta_intervention_week_net_pnl": cur_pnl - ref_pnl,
    }


def _score_post_intervention(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    baseline_weekly: pd.DataFrame,
    intervention_week: pd.Timestamp,
) -> Dict[str, float]:
    scores = _score_windows(
        daily,
        weekly,
        baseline_daily,
        baseline_weekly,
        validation_start=intervention_week.isoformat(),
        june_start=intervention_week.isoformat(),
    )
    row = scores.loc[scores["window"].eq("validation_may_june")].iloc[0]
    return {
        "delta_post_intervention_objective": float(row[f"delta_{OBJECTIVE_COL}"]),
        "delta_post_intervention_net_pnl": float(row["delta_net_pnl"]),
        "delta_post_intervention_weekly_q20_pnl": float(row["delta_weekly_q20_pnl"]),
        "delta_post_intervention_daily_q20_pnl": float(row["delta_daily_q20_pnl"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--baseline-combo", default=CHAMPION_COMBO)
    parser.add_argument("--candidate-combo", default="long_bars:I_long_dist:R_short_asset:S_short_bollinger:R")
    parser.add_argument("--candidate-rule", action="append", required=True)
    parser.add_argument("--start-week", default=None)
    parser.add_argument("--end-week", default=None)
    parser.add_argument("--threshold-mode", default="expanding", choices=["full_sample", "expanding"])
    parser.add_argument("--min-threshold-history", type=int, default=500)
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    parser.add_argument("--save-decisions", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = [str(v) for v in args.candidate_rule]
    tables = _load_arm_tables(args.source_dir, [args.baseline_combo, args.candidate_combo])
    cache = _build_candidate_cache(
        tables,
        baseline_combo=args.baseline_combo,
        candidate_combo=args.candidate_combo,
        candidate_rules=rules,
        threshold_mode=args.threshold_mode,
        min_threshold_history=args.min_threshold_history,
    )
    baseline = cache["baseline"]
    baseline_daily, baseline_weekly = _load_baseline_tables(args.baseline_dir)
    weeks = _available_weeks(baseline, args.start_week, args.end_week)

    rows: List[Dict[str, Any]] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []
    for week in weeks:
        for rule in rules:
            cid = _candidate_id(args.candidate_combo, rule)
            stream = _stream_for_one_week(baseline, cache[cid], week)
            decisions, _equity, metrics = _run_replay(stream, args.market_mode)
            daily, weekly = _accepted_period_tables(decisions)
            scores = _score_windows(
                daily,
                weekly,
                baseline_daily,
                baseline_weekly,
                validation_start=args.validation_start,
                june_start=args.june_start,
            )
            week_score = _intervention_week_score(weekly, baseline_weekly, week)
            post_score = _score_post_intervention(daily, weekly, baseline_daily, baseline_weekly, week)
            row: Dict[str, Any] = {
                "intervention_week": week.isoformat(),
                "candidate_combo": args.candidate_combo,
                "candidate_rule": rule,
                "candidate_rows": int(len(stream)),
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "net_pnl": float(metrics.get("net_pnl", 0.0)),
                "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                **week_score,
                **post_score,
            }
            for _, score_row in scores.iterrows():
                prefix = str(score_row["window"])
                row[f"delta_{prefix}_objective"] = float(score_row[f"delta_{OBJECTIVE_COL}"])
                row[f"delta_{prefix}_net_pnl"] = float(score_row["delta_net_pnl"])
                row[f"delta_{prefix}_weekly_q20_pnl"] = float(score_row["delta_weekly_q20_pnl"])
                row[f"delta_{prefix}_daily_q20_pnl"] = float(score_row["delta_daily_q20_pnl"])
                row[f"pass_{prefix}_pnl_tail_gate"] = bool(score_row["pass_pnl_tail_gate"])
            rows.append(row)
            for frame in (daily, weekly):
                if not frame.empty:
                    frame["intervention_week"] = week.isoformat()
                    frame["candidate_rule"] = rule
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            if args.save_decisions and "accepted" in decisions.columns:
                accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
                if not accepted.empty:
                    accepted["intervention_week"] = week.isoformat()
                    accepted["candidate_rule"] = rule
                    accepted_frames.append(accepted)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values("delta_validation_may_june_objective", ascending=False).reset_index(drop=True)
    summary.to_csv(args.out_dir / "weekly_intervention_oracle_summary.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(args.out_dir / "weekly_intervention_oracle_daily.csv", index=False)
    pd.concat(weekly_frames, ignore_index=True).to_csv(args.out_dir / "weekly_intervention_oracle_weekly.csv", index=False)
    if args.save_decisions:
        accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
        accepted_all.to_parquet(args.out_dir / "weekly_intervention_oracle_accepted_decisions.parquet", index=False)

    show_cols = [
        "intervention_week",
        "candidate_rule",
        "delta_validation_may_june_objective",
        "delta_validation_may_june_net_pnl",
        "delta_validation_may_june_weekly_q20_pnl",
        "delta_june_only_net_pnl",
        "delta_full_net_pnl",
        "delta_intervention_week_net_pnl",
        "delta_post_intervention_net_pnl",
    ]
    lines = [
        "# Weekly Intervention Oracle",
        "",
        f"Baseline combo: `{args.baseline_combo}`",
        f"Candidate combo: `{args.candidate_combo}`",
        f"Candidate rules: `{', '.join(rules)}`",
        "Each row replays one week of candidate intervention through the full portfolio auction. Costs are included.",
        "",
        "## Top May-June Objective Interventions",
        "",
        summary[[c for c in show_cols if c in summary.columns]].head(30).round(6).to_markdown(index=False)
        if not summary.empty
        else "_No rows._",
        "",
        "## Positive Full-Path And Tail Interventions",
        "",
    ]
    if not summary.empty:
        positive = summary.loc[
            (summary["delta_validation_may_june_net_pnl"] > 0.0)
            & (summary["delta_validation_may_june_weekly_q20_pnl"] >= 0.0)
        ]
        lines.append(
            positive[[c for c in show_cols if c in positive.columns]].head(30).round(6).to_markdown(index=False)
            if not positive.empty
            else "_No positive May-June PnL+tail interventions._"
        )
    else:
        lines.append("_No rows._")
    (args.out_dir / "weekly_intervention_oracle_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "baseline_dir": str(args.baseline_dir),
        "out_dir": str(args.out_dir),
        "baseline_combo": args.baseline_combo,
        "candidate_combo": args.candidate_combo,
        "candidate_rules": rules,
        "weeks": [w.isoformat() for w in weeks],
        "rows": int(len(summary)),
    }
    (args.out_dir / "weekly_intervention_oracle_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(summary)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
