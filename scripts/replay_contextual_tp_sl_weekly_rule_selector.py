#!/usr/bin/env python3
"""Replay weekly rule selection through the full portfolio auction.

This is a stricter successor to recombination audits. It chooses among a fixed
baseline and candidate ``combo + conditional rule`` alternatives using only
prior-week metrics, constructs one weekly-switched candidate ledger, and runs a
single full portfolio replay.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
from scripts.ablate_contextual_tp_sl_conditional_head_filters import (  # noqa: E402
    DEFAULT_RULES,
    _add_condition_flags,
    _apply_rule,
)
from scripts.replay_contextual_tp_sl_weekly_combo_switching import (  # noqa: E402
    CHAMPION_COMBO,
    OBJECTIVE_COL,
    _build_combo_candidates,
    _load_arm_tables,
    _parse_combo_id,
)
from scripts.report_contextual_tp_sl_pairwise_combo_switching import _objective  # noqa: E402
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    _accepted_period_tables,
    _json_safe,
)


BASELINE_ID = "baseline"


def _week_start_from_week(values: pd.Series) -> pd.Series:
    return pd.to_datetime(
        values.astype(str).str.split("/", n=1).str[0],
        utc=True,
        errors="coerce",
    )


def _load_metric_tables(
    *,
    baseline_daily_path: Path,
    baseline_weekly_path: Path,
    conditional_daily_path: Path,
    conditional_weekly_path: Path,
    candidate_rules: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_daily = pd.read_csv(baseline_daily_path)
    base_weekly = pd.read_csv(baseline_weekly_path)
    base_daily["candidate_id"] = BASELINE_ID
    base_daily["rule_id"] = "none"
    base_daily["combo_id"] = CHAMPION_COMBO
    base_weekly["candidate_id"] = BASELINE_ID
    base_weekly["rule_id"] = "none"
    base_weekly["combo_id"] = CHAMPION_COMBO

    cond_daily = pd.read_csv(conditional_daily_path)
    cond_weekly = pd.read_csv(conditional_weekly_path)
    rules = {str(r) for r in candidate_rules}
    cond_daily = cond_daily.loc[cond_daily["rule_id"].astype(str).isin(rules)].copy()
    cond_weekly = cond_weekly.loc[cond_weekly["rule_id"].astype(str).isin(rules)].copy()
    cond_daily["candidate_id"] = cond_daily["combo_id"].astype(str) + "|" + cond_daily["rule_id"].astype(str)
    cond_weekly["candidate_id"] = cond_weekly["combo_id"].astype(str) + "|" + cond_weekly["rule_id"].astype(str)

    daily = pd.concat([base_daily, cond_daily], ignore_index=True)
    weekly = pd.concat([base_weekly, cond_weekly], ignore_index=True)
    daily["day_start"] = pd.to_datetime(daily["day"], utc=True, errors="coerce")
    weekly["week_start"] = _week_start_from_week(weekly["week"])
    return daily.dropna(subset=["day_start"]), weekly.dropna(subset=["week_start"])


def _score_candidate_window(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    candidate_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, float]:
    d = daily.loc[
        daily["candidate_id"].eq(candidate_id)
        & daily["day_start"].ge(start)
        & daily["day_start"].lt(end)
    ]
    w = weekly.loc[
        weekly["candidate_id"].eq(candidate_id)
        & weekly["week_start"].ge(start)
        & weekly["week_start"].lt(end)
    ]
    return _objective(d["net_pnl"], w["net_pnl"])


def _select_for_week(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    week_start: pd.Timestamp,
    lookback_weeks: int,
    min_history_weeks: int,
    mode: str,
    min_objective_delta: float,
    min_net_delta: float,
) -> Tuple[str, str, Dict[str, Any]]:
    history_start = week_start - pd.Timedelta(weeks=int(lookback_weeks))
    baseline_hist = weekly.loc[
        weekly["candidate_id"].eq(BASELINE_ID)
        & weekly["week_start"].ge(history_start)
        & weekly["week_start"].lt(week_start)
    ]
    if len(baseline_hist) < int(min_history_weeks):
        return BASELINE_ID, "fallback_insufficient_history", {}
    baseline = _score_candidate_window(daily, weekly, BASELINE_ID, history_start, week_start)
    best_id = BASELINE_ID
    best_reason = f"fallback_{mode}"
    best_stats: Dict[str, Any] = {}
    best_score = -np.inf
    for candidate_id in candidate_ids:
        if candidate_id == BASELINE_ID:
            continue
        score = _score_candidate_window(daily, weekly, candidate_id, history_start, week_start)
        delta_obj = score[OBJECTIVE_COL] - baseline[OBJECTIVE_COL]
        delta_net = score["net_pnl"] - baseline["net_pnl"]
        delta_wq20 = score["weekly_q20_pnl"] - baseline["weekly_q20_pnl"]
        delta_dq20 = score["daily_q20_pnl"] - baseline["daily_q20_pnl"]
        if mode == "objective_tail_guard":
            eligible = delta_obj > min_objective_delta and delta_wq20 >= 0.0 and delta_dq20 >= 0.0
            rank_score = delta_obj
        elif mode == "net_tail_guard":
            eligible = delta_net > min_net_delta and delta_wq20 >= 0.0 and delta_dq20 >= 0.0
            rank_score = delta_net
        elif mode == "objective_soft_tail":
            eligible = delta_obj > min_objective_delta and delta_wq20 >= -abs(min_objective_delta)
            rank_score = delta_obj + 0.5 * min(delta_wq20, 0.0)
        else:
            raise ValueError(f"Unknown mode `{mode}`")
        stats = {
            "history_candidate_id": candidate_id,
            "history_delta_objective": float(delta_obj),
            "history_delta_net_pnl": float(delta_net),
            "history_delta_weekly_q20": float(delta_wq20),
            "history_delta_daily_q20": float(delta_dq20),
        }
        if eligible and rank_score > best_score:
            best_id = candidate_id
            best_reason = f"selected_{mode}"
            best_score = float(rank_score)
            best_stats = stats
    return best_id, best_reason, best_stats


def _candidate_combo_rule(candidate_id: str, candidate_combo: str) -> Tuple[str, str]:
    if candidate_id == BASELINE_ID:
        return CHAMPION_COMBO, "none"
    if "|" not in candidate_id:
        raise ValueError(f"Candidate id `{candidate_id}` must be `combo|rule`")
    combo, rule = candidate_id.rsplit("|", 1)
    if not combo:
        combo = candidate_combo
    return combo, rule


def _build_candidate_cache(
    tables: Mapping[str, pd.DataFrame],
    *,
    baseline_combo: str,
    candidate_combo: str,
    candidate_rules: Sequence[str],
    threshold_mode: str,
    min_threshold_history: int,
) -> Dict[str, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    baseline = _build_combo_candidates(tables, _parse_combo_id(baseline_combo))
    cache[BASELINE_ID] = baseline
    candidate_base = _build_combo_candidates(tables, _parse_combo_id(candidate_combo))
    candidate_base = _add_condition_flags(
        candidate_base,
        threshold_mode=threshold_mode,
        min_history=min_threshold_history,
    )
    for rule_id in candidate_rules:
        if rule_id == "none":
            frame = candidate_base
        else:
            rule = DEFAULT_RULES.get(rule_id)
            if rule is None:
                raise ValueError(f"Unknown default rule `{rule_id}`")
            frame = _apply_rule(
                candidate_base,
                rule,
                threshold_mode=threshold_mode,
                min_history=min_threshold_history,
            )
        cache[f"{candidate_combo}|{rule_id}"] = frame
    return cache


def _week_start(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values, utc=True, errors="coerce")
    naive = ts.dt.tz_convert(None)
    starts = naive.dt.to_period("W").dt.start_time
    return pd.to_datetime(starts, utc=True, errors="coerce")


def _build_weekly_stream(
    cache: Mapping[str, pd.DataFrame],
    metrics_daily: pd.DataFrame,
    metrics_weekly: pd.DataFrame,
    *,
    candidate_ids: Sequence[str],
    candidate_combo: str,
    lookback_weeks: int,
    min_history_weeks: int,
    mode: str,
    min_objective_delta: float,
    min_net_delta: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weeks = sorted(metrics_weekly.loc[metrics_weekly["candidate_id"].eq(BASELINE_ID), "week_start"].dropna().unique())
    decision_rows: List[Dict[str, Any]] = []
    frames: List[pd.DataFrame] = []
    for week_raw in weeks:
        week_start = pd.Timestamp(week_raw)
        selected_id, reason, stats = _select_for_week(
            metrics_daily,
            metrics_weekly,
            candidate_ids=candidate_ids,
            week_start=week_start,
            lookback_weeks=lookback_weeks,
            min_history_weeks=min_history_weeks,
            mode=mode,
            min_objective_delta=min_objective_delta,
            min_net_delta=min_net_delta,
        )
        combo, rule = _candidate_combo_rule(selected_id, candidate_combo)
        source = cache[selected_id]
        if "_week_start" not in source.columns:
            source = source.copy()
            source["_week_start"] = _week_start(source["timestamp"])
            if isinstance(cache, dict):
                cache[selected_id] = source
        week = source.loc[source["_week_start"].eq(week_start)].copy()
        if week.empty:
            continue
        week["selected_candidate_id"] = selected_id
        week["selected_combo_id"] = combo
        week["selected_rule_id"] = rule
        frames.append(week)
        decision_rows.append(
            {
                "week_start": week_start.isoformat(),
                "selected_candidate_id": selected_id,
                "selected_combo_id": combo,
                "selected_rule_id": rule,
                "selected_reason": reason,
                "lookback_weeks": int(lookback_weeks),
                "selection_mode": mode,
                "candidate_rows": int(len(week)),
                **stats,
            }
        )
    if not frames:
        return pd.DataFrame(), pd.DataFrame(decision_rows)
    candidates = (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["_week_start"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
    )
    return candidates, pd.DataFrame(decision_rows)


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
        frame["day_start"] = pd.to_datetime(frame[col], utc=True, errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns, UTC]")
    for frame in (w, bw):
        if not frame.empty:
            frame["week_start"] = _week_start_from_week(frame["week"])
    max_day = d["day_start"].max() if not d.empty else bd["day_start"].max()
    rows: List[Dict[str, Any]] = []
    for label, start, end in (
        ("full", None, max_day),
        ("validation_may_june", pd.Timestamp(validation_start, tz="UTC"), max_day),
        ("june_only", pd.Timestamp(june_start, tz="UTC"), max_day),
    ):
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


def _run_replay(candidates: pd.DataFrame, market_mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    ev_curve = fit_hierarchical_ev_curves(candidates)
    return replay_candidates(
        candidates,
        PortfolioPolicyParams(global_threshold_floor=0.0),
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=market_mode,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--conditional-metrics-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--baseline-combo", default=CHAMPION_COMBO)
    parser.add_argument("--candidate-combo", default="long_bars:I_long_dist:R_short_asset:S_short_bollinger:R")
    parser.add_argument("--candidate-rule", action="append", required=True)
    parser.add_argument("--lookback-weeks", default="4,8")
    parser.add_argument("--min-history-weeks", type=int, default=2)
    parser.add_argument("--selection-modes", default="objective_tail_guard,net_tail_guard,objective_soft_tail")
    parser.add_argument("--min-objective-delta", type=float, default=0.0)
    parser.add_argument("--min-net-delta", type=float, default=0.0)
    parser.add_argument("--threshold-mode", default="expanding", choices=["full_sample", "expanding"])
    parser.add_argument("--min-threshold-history", type=int, default=500)
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    parser.add_argument("--save-decisions", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = [str(rule) for rule in args.candidate_rule]
    metrics_daily, metrics_weekly = _load_metric_tables(
        baseline_daily_path=args.baseline_dir / "baseline_daily.csv",
        baseline_weekly_path=args.baseline_dir / "baseline_weekly.csv",
        conditional_daily_path=args.conditional_metrics_dir / "conditional_filter_daily.csv",
        conditional_weekly_path=args.conditional_metrics_dir / "conditional_filter_weekly.csv",
        candidate_rules=rules,
    )
    candidate_ids = [BASELINE_ID, *[f"{args.candidate_combo}|{rule}" for rule in rules]]
    tables = _load_arm_tables(args.source_dir, [args.baseline_combo, args.candidate_combo])
    cache = _build_candidate_cache(
        tables,
        baseline_combo=args.baseline_combo,
        candidate_combo=args.candidate_combo,
        candidate_rules=rules,
        threshold_mode=args.threshold_mode,
        min_threshold_history=args.min_threshold_history,
    )
    baseline_daily = pd.read_csv(args.baseline_dir / "baseline_daily.csv")
    baseline_weekly = pd.read_csv(args.baseline_dir / "baseline_weekly.csv")

    score_frames: List[pd.DataFrame] = []
    decision_frames: List[pd.DataFrame] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    accepted_frames: List[pd.DataFrame] = []
    for lookback in [int(v.strip()) for v in str(args.lookback_weeks).split(",") if v.strip()]:
        for mode in [str(v.strip()) for v in str(args.selection_modes).split(",") if v.strip()]:
            candidates, decisions = _build_weekly_stream(
                cache,
                metrics_daily,
                metrics_weekly,
                candidate_ids=candidate_ids,
                candidate_combo=args.candidate_combo,
                lookback_weeks=lookback,
                min_history_weeks=args.min_history_weeks,
                mode=mode,
                min_objective_delta=float(args.min_objective_delta),
                min_net_delta=float(args.min_net_delta),
            )
            if candidates.empty:
                continue
            replay_decisions, _equity, metrics = _run_replay(candidates, args.market_mode)
            daily, weekly = _accepted_period_tables(replay_decisions)
            scores = _score_windows(
                daily,
                weekly,
                baseline_daily,
                baseline_weekly,
                validation_start=args.validation_start,
                june_start=args.june_start,
            )
            selected_share = float(decisions["selected_candidate_id"].ne(BASELINE_ID).mean()) if not decisions.empty else 0.0
            meta = {
                "lookback_weeks": int(lookback),
                "selection_mode": mode,
                "candidate_rules": ",".join(rules),
                "selected_candidate_week_share": selected_share,
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "net_pnl": float(metrics.get("net_pnl", 0.0)),
                "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                "total_candidate_rows": int(len(candidates)),
            }
            for frame in (scores, decisions, daily, weekly):
                for key, value in meta.items():
                    frame[key] = value
            score_frames.append(scores)
            decision_frames.append(decisions)
            daily_frames.append(daily)
            weekly_frames.append(weekly)
            if args.save_decisions and "accepted" in replay_decisions.columns:
                accepted = replay_decisions.loc[replay_decisions["accepted"].astype(bool)].copy()
                if not accepted.empty:
                    for key, value in meta.items():
                        accepted[key] = value
                    accepted_frames.append(accepted)

    scores_all = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    decisions_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    weekly_all = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    scores_all.to_csv(args.out_dir / "weekly_rule_selector_scores.csv", index=False)
    decisions_all.to_csv(args.out_dir / "weekly_rule_selector_decisions.csv", index=False)
    daily_all.to_csv(args.out_dir / "weekly_rule_selector_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "weekly_rule_selector_weekly.csv", index=False)
    if args.save_decisions:
        accepted_all = pd.concat(accepted_frames, ignore_index=True) if accepted_frames else pd.DataFrame()
        accepted_all.to_parquet(args.out_dir / "weekly_rule_selector_accepted_decisions.parquet", index=False)

    validation = scores_all.loc[scores_all["window"].eq("validation_may_june")].copy()
    top_validation = validation.sort_values(f"delta_{OBJECTIVE_COL}", ascending=False).head(20)
    passing = validation.loc[validation["pass_pnl_tail_gate"].astype(bool)].sort_values(
        f"delta_{OBJECTIVE_COL}",
        ascending=False,
    )
    show_cols = [
        "lookback_weeks",
        "selection_mode",
        "window",
        "selected_candidate_week_share",
        f"delta_{OBJECTIVE_COL}",
        "delta_net_pnl",
        "delta_weekly_q20_pnl",
        "delta_daily_q20_pnl",
        "selected_net_pnl",
        "baseline_net_pnl",
        "pass_pnl_tail_gate",
    ]
    lines = [
        "# Weekly Rule Selector Full Replay",
        "",
        f"Baseline combo: `{args.baseline_combo}`",
        f"Candidate combo: `{args.candidate_combo}`",
        f"Candidate rules: `{', '.join(rules)}`",
        "Selection uses prior standalone rule metrics, then the selected weekly candidate stream is replayed once through the full portfolio auction.",
        "Costs are included. This is a replay-period walk-forward audit, not untouched live OOS.",
        "",
        "## May-June Passing Selectors",
        "",
        passing[[c for c in show_cols if c in passing.columns]].head(30).round(6).to_markdown(index=False)
        if not passing.empty
        else "_No May-June passing selectors._",
        "",
        "## Top May-June Selectors",
        "",
        top_validation[[c for c in show_cols if c in top_validation.columns]].round(6).to_markdown(index=False)
        if not top_validation.empty
        else "_No rows._",
    ]
    (args.out_dir / "weekly_rule_selector_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "conditional_metrics_dir": str(args.conditional_metrics_dir),
        "baseline_dir": str(args.baseline_dir),
        "out_dir": str(args.out_dir),
        "baseline_combo": args.baseline_combo,
        "candidate_combo": args.candidate_combo,
        "candidate_rules": rules,
        "rows": int(len(scores_all)),
    }
    (args.out_dir / "weekly_rule_selector_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(scores_all)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
