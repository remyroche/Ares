#!/usr/bin/env python3
"""Replay a baseline-default selector trained on prior intervention oracle labels.

The selector never uses future oracle outcomes for the current week. For week t
it looks only at prior one-week intervention labels, chooses no action by
default, and intervenes only when a rule has prior full-replay evidence passing
strict net/tail gates. The selected weekly stream is then replayed once through
the full portfolio auction.
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

from scripts.replay_contextual_tp_sl_weekly_combo_switching import (  # noqa: E402
    CHAMPION_COMBO,
    OBJECTIVE_COL,
    _load_arm_tables,
)
from scripts.replay_contextual_tp_sl_weekly_intervention_oracle import (  # noqa: E402
    _candidate_id,
    _load_baseline_tables,
    _stream_for_one_week,
)
from scripts.replay_contextual_tp_sl_weekly_rule_selector import (  # noqa: E402
    BASELINE_ID,
    _build_candidate_cache,
    _run_replay,
    _score_windows,
)
from scripts.sweep_contextual_tp_sl_arm_combinations import (  # noqa: E402
    _accepted_period_tables,
    _json_safe,
)


def _week_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _load_oracle(path: Path, rules: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["week_start"] = pd.to_datetime(frame["intervention_week"], utc=True, errors="coerce")
    frame = frame.loc[frame["candidate_rule"].astype(str).isin({str(r) for r in rules})].copy()
    for col in (
        "delta_full_objective",
        "delta_full_net_pnl",
        "delta_full_weekly_q20_pnl",
        "delta_intervention_week_net_pnl",
        "delta_post_intervention_net_pnl",
    ):
        frame[col] = pd.to_numeric(frame.get(col), errors="coerce")
    frame["full_net_tail_positive"] = (
        frame["delta_full_net_pnl"].gt(0.0) & frame["delta_full_weekly_q20_pnl"].ge(0.0)
    )
    return frame.dropna(subset=["week_start"])


def _history_stats(
    oracle: pd.DataFrame,
    *,
    rule: str,
    week_start: pd.Timestamp,
    lookback_weeks: int,
) -> Dict[str, float]:
    start = week_start - pd.Timedelta(weeks=int(lookback_weeks))
    hist = oracle.loc[
        oracle["candidate_rule"].astype(str).eq(str(rule))
        & oracle["week_start"].ge(start)
        & oracle["week_start"].lt(week_start)
    ].copy()
    if hist.empty:
        return {"history_n": 0}
    return {
        "history_n": int(len(hist)),
        "history_full_gate_count": int(hist["pass_full_pnl_tail_gate"].astype(bool).sum()),
        "history_net_tail_count": int(hist["full_net_tail_positive"].sum()),
        "history_median_full_objective": float(hist["delta_full_objective"].median()),
        "history_median_full_net": float(hist["delta_full_net_pnl"].median()),
        "history_median_full_weekly_q20": float(hist["delta_full_weekly_q20_pnl"].median()),
        "history_median_intervention_week_net": float(hist["delta_intervention_week_net_pnl"].median()),
        "history_best_full_objective": float(hist["delta_full_objective"].max()),
        "history_best_full_net": float(hist["delta_full_net_pnl"].max()),
        "history_best_full_weekly_q20": float(hist["delta_full_weekly_q20_pnl"].max()),
        "history_best_intervention_week_net": float(hist["delta_intervention_week_net_pnl"].max()),
    }


def _select_rule(
    oracle: pd.DataFrame,
    *,
    rules: Sequence[str],
    week_start: pd.Timestamp,
    lookback_weeks: int,
    mode: str,
    min_history: int,
    min_successes: int,
    min_median_net: float,
) -> Tuple[str, str, Dict[str, Any]]:
    best_rule = BASELINE_ID
    best_score = -np.inf
    best_stats: Dict[str, Any] = {}
    for rule in rules:
        stats = _history_stats(oracle, rule=rule, week_start=week_start, lookback_weeks=lookback_weeks)
        if int(stats.get("history_n", 0)) < int(min_history):
            continue
        if mode == "prior_full_gate":
            eligible = (
                stats.get("history_full_gate_count", 0) >= min_successes
                and stats.get("history_median_full_net", -np.inf) >= min_median_net
                and stats.get("history_median_full_weekly_q20", -np.inf) >= 0.0
            )
            score = stats.get("history_median_full_objective", -np.inf)
        elif mode == "prior_net_tail":
            eligible = (
                stats.get("history_net_tail_count", 0) >= min_successes
                and stats.get("history_median_full_net", -np.inf) >= min_median_net
                and stats.get("history_median_full_weekly_q20", -np.inf) >= 0.0
            )
            score = stats.get("history_median_full_net", -np.inf)
        elif mode == "prior_intervention_week_net_tail":
            eligible = (
                stats.get("history_net_tail_count", 0) >= min_successes
                and stats.get("history_median_intervention_week_net", -np.inf) >= min_median_net
                and stats.get("history_median_full_weekly_q20", -np.inf) >= 0.0
            )
            score = stats.get("history_median_intervention_week_net", -np.inf)
        elif mode == "prior_best_full_gate":
            eligible = (
                stats.get("history_full_gate_count", 0) >= min_successes
                and stats.get("history_best_full_net", -np.inf) >= min_median_net
                and stats.get("history_best_full_weekly_q20", -np.inf) >= 0.0
            )
            score = stats.get("history_best_full_objective", -np.inf)
        elif mode == "prior_best_net_tail":
            eligible = (
                stats.get("history_net_tail_count", 0) >= min_successes
                and stats.get("history_best_full_net", -np.inf) >= min_median_net
                and stats.get("history_best_full_weekly_q20", -np.inf) >= 0.0
            )
            score = stats.get("history_best_full_net", -np.inf)
        else:
            raise ValueError(f"Unknown mode `{mode}`")
        if eligible and score > best_score:
            best_rule = str(rule)
            best_score = float(score)
            best_stats = stats
    if best_rule == BASELINE_ID:
        return BASELINE_ID, f"fallback_{mode}", {}
    return best_rule, f"selected_{mode}", best_stats


def _build_selected_stream(
    cache: Mapping[str, pd.DataFrame],
    oracle: pd.DataFrame,
    *,
    candidate_combo: str,
    rules: Sequence[str],
    lookback_weeks: int,
    mode: str,
    min_history: int,
    min_successes: int,
    min_median_net: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = cache[BASELINE_ID]
    if "_week_start" not in baseline.columns:
        baseline = baseline.copy()
        baseline["_week_start"] = pd.to_datetime(baseline["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("W").dt.start_time
        baseline["_week_start"] = pd.to_datetime(baseline["_week_start"], utc=True, errors="coerce")
    weeks = sorted(pd.Timestamp(v) for v in baseline["_week_start"].dropna().unique())
    frames: List[pd.DataFrame] = []
    decisions: List[Dict[str, Any]] = []
    for week in weeks:
        selected, reason, stats = _select_rule(
            oracle,
            rules=rules,
            week_start=week,
            lookback_weeks=lookback_weeks,
            mode=mode,
            min_history=min_history,
            min_successes=min_successes,
            min_median_net=min_median_net,
        )
        if selected == BASELINE_ID:
            source = baseline
            selected_id = BASELINE_ID
        else:
            selected_id = _candidate_id(candidate_combo, selected)
            source = cache[selected_id]
        if "_week_start" not in source.columns:
            source = source.copy()
            source["_week_start"] = pd.to_datetime(source["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.to_period("W").dt.start_time
            source["_week_start"] = pd.to_datetime(source["_week_start"], utc=True, errors="coerce")
            if isinstance(cache, dict):
                cache[selected_id] = source
        week_frame = source.loc[source["_week_start"].eq(week)].copy()
        if week_frame.empty:
            continue
        week_frame["selected_candidate_id"] = selected_id
        frames.append(week_frame)
        decisions.append(
            {
                "week_start": week.isoformat(),
                "selected_rule": selected,
                "selected_candidate_id": selected_id,
                "selected_reason": reason,
                "lookback_weeks": int(lookback_weeks),
                "selection_mode": mode,
                "candidate_rows": int(len(week_frame)),
                **stats,
            }
        )
    stream = (
        pd.concat(frames, ignore_index=True)
        .drop(columns=["_week_start"], errors="ignore")
        .sort_values(["timestamp", "strategy_id", "symbol"])
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    return stream, pd.DataFrame(decisions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--oracle-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--market-mode", default="perps", choices=["spot", "perps"])
    parser.add_argument("--baseline-combo", default=CHAMPION_COMBO)
    parser.add_argument("--candidate-combo", default="long_bars:I_long_dist:R_short_asset:S_short_bollinger:R")
    parser.add_argument("--candidate-rule", action="append", required=True)
    parser.add_argument("--lookback-weeks", default="4,8,12")
    parser.add_argument("--selection-modes", default="prior_full_gate,prior_net_tail,prior_intervention_week_net_tail")
    parser.add_argument("--min-history", type=int, default=2)
    parser.add_argument("--min-successes", type=int, default=1)
    parser.add_argument("--min-median-net", type=float, default=0.0)
    parser.add_argument("--threshold-mode", default="expanding", choices=["full_sample", "expanding"])
    parser.add_argument("--min-threshold-history", type=int, default=500)
    parser.add_argument("--validation-start", default="2026-05-01")
    parser.add_argument("--june-start", default="2026-06-01")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = [str(r) for r in args.candidate_rule]
    oracle = _load_oracle(args.oracle_summary, rules)
    tables = _load_arm_tables(args.source_dir, [args.baseline_combo, args.candidate_combo])
    cache = _build_candidate_cache(
        tables,
        baseline_combo=args.baseline_combo,
        candidate_combo=args.candidate_combo,
        candidate_rules=rules,
        threshold_mode=args.threshold_mode,
        min_threshold_history=args.min_threshold_history,
    )
    baseline_daily, baseline_weekly = _load_baseline_tables(args.baseline_dir)
    scores: List[pd.DataFrame] = []
    decision_frames: List[pd.DataFrame] = []
    daily_frames: List[pd.DataFrame] = []
    weekly_frames: List[pd.DataFrame] = []
    lookbacks = [int(v.strip()) for v in str(args.lookback_weeks).split(",") if v.strip()]
    modes = [str(v.strip()) for v in str(args.selection_modes).split(",") if v.strip()]
    for lookback in lookbacks:
        for mode in modes:
            stream, decisions = _build_selected_stream(
                cache,
                oracle,
                candidate_combo=args.candidate_combo,
                rules=rules,
                lookback_weeks=lookback,
                mode=mode,
                min_history=args.min_history,
                min_successes=args.min_successes,
                min_median_net=args.min_median_net,
            )
            if stream.empty:
                continue
            replay_decisions, _equity, metrics = _run_replay(stream, args.market_mode)
            daily, weekly = _accepted_period_tables(replay_decisions)
            result = _score_windows(
                daily,
                weekly,
                baseline_daily,
                baseline_weekly,
                validation_start=args.validation_start,
                june_start=args.june_start,
            )
            selected_share = float(decisions["selected_rule"].ne(BASELINE_ID).mean()) if not decisions.empty else 0.0
            meta = {
                "lookback_weeks": int(lookback),
                "selection_mode": mode,
                "selected_candidate_week_share": selected_share,
                "trade_count": int(metrics.get("trade_count", 0) or 0),
                "net_pnl": float(metrics.get("net_pnl", 0.0)),
                "full_sl_rate": float(metrics.get("full_sl_rate", 0.0)),
                "candidate_rows": int(len(stream)),
            }
            for frame in (result, decisions, daily, weekly):
                for key, value in meta.items():
                    frame[key] = value
            scores.append(result)
            decision_frames.append(decisions)
            daily_frames.append(daily)
            weekly_frames.append(weekly)

    scores_all = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    decisions_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()
    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    weekly_all = pd.concat(weekly_frames, ignore_index=True) if weekly_frames else pd.DataFrame()
    scores_all.to_csv(args.out_dir / "oracle_label_selector_scores.csv", index=False)
    decisions_all.to_csv(args.out_dir / "oracle_label_selector_decisions.csv", index=False)
    daily_all.to_csv(args.out_dir / "oracle_label_selector_daily.csv", index=False)
    weekly_all.to_csv(args.out_dir / "oracle_label_selector_weekly.csv", index=False)

    validation = scores_all.loc[scores_all["window"].eq("validation_may_june")].copy() if not scores_all.empty else pd.DataFrame()
    top_validation = validation.sort_values(f"delta_{OBJECTIVE_COL}", ascending=False).head(20) if not validation.empty else validation
    show_cols = [
        "lookback_weeks",
        "selection_mode",
        "window",
        "selected_candidate_week_share",
        f"delta_{OBJECTIVE_COL}",
        "delta_net_pnl",
        "delta_weekly_q20_pnl",
        "delta_daily_q20_pnl",
        "pass_pnl_tail_gate",
    ]
    lines = [
        "# Oracle-Label Selector Replay",
        "",
        f"Oracle labels: `{args.oracle_summary}`",
        f"Candidate rules: `{', '.join(rules)}`",
        "Selector uses only prior full-replay intervention labels. Costs are included.",
        "",
        "## Top May-June Selectors",
        "",
        top_validation[[c for c in show_cols if c in top_validation.columns]].round(6).to_markdown(index=False)
        if not top_validation.empty
        else "_No rows._",
    ]
    (args.out_dir / "oracle_label_selector_report.md").write_text("\n".join(lines) + "\n")
    payload = {
        "source_dir": str(args.source_dir),
        "baseline_dir": str(args.baseline_dir),
        "oracle_summary": str(args.oracle_summary),
        "out_dir": str(args.out_dir),
        "candidate_rules": rules,
        "rows": int(len(scores_all)),
    }
    (args.out_dir / "oracle_label_selector_summary.json").write_text(json.dumps(_json_safe(payload), indent=2))
    print(json.dumps(_json_safe({"out_dir": str(args.out_dir), "rows": len(scores_all)}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
