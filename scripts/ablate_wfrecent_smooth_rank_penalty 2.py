#!/usr/bin/env python3
"""Smooth diagnostic rank-penalty ablation around the fixed wf_recent combo.

The hard row veto failed under continuous portfolio-state replay. This script
tests a softer alternative: fit diagnostic risk percentiles from prior rows,
then subtract a bounded rank penalty from high-risk candidates. A cheap proxy
screen keeps the number of full portfolio replays small.

This is a development ablation, not a production promotion gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from scripts.validate_wfrecent_row_guard_walkforward import (  # noqa: E402
    RISK_SCORE_NAMES,
    _apply_risk_scores,
    _fit_percentile_reference,
    _fmt_table,
    _head_name,
    _json_safe,
    _period_tables,
    _summary,
)


HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")


@dataclass(frozen=True)
class SmoothRule:
    score_name: str
    scope: str
    risk_quantile: float
    min_rank_pct: float
    max_penalty: float
    power: float

    @property
    def label(self) -> str:
        q = int(round(self.risk_quantile * 100))
        r = int(round(self.min_rank_pct * 100))
        p = str(self.max_penalty).replace(".", "p")
        pow_txt = str(self.power).replace(".", "p")
        return f"{self.scope}__{self.score_name}__q{q}__rank{r}__pen{p}__pow{pow_txt}"


@dataclass
class MonthCache:
    start: pd.Timestamp
    stop: pd.Timestamp
    indices: np.ndarray
    train_rows: int
    month_rows: int
    month_scored: pd.DataFrame
    thresholds: dict[tuple[str, str, float], float]


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = pd.Timestamp(cur + pd.offsets.MonthBegin(1))
        ranges.append((cur, min(nxt, end)))
        cur = nxt
    return ranges


def _candidate_rules() -> list[SmoothRule]:
    # Keep the grid deliberately bounded. The goal is to test whether smooth
    # penalties are directionally better than hard vetoes, not to HPO a policy.
    rules: list[SmoothRule] = []
    for score_name in RISK_SCORE_NAMES:
        for scope in ("all", *HEAD_ORDER):
            for risk_quantile in (0.85, 0.90, 0.95, 0.98):
                for min_rank_pct in (0.70, 0.80, 0.90):
                    for max_penalty in (0.01, 0.025, 0.05, 0.10):
                        for power in (0.5, 1.0, 2.0):
                            rules.append(SmoothRule(score_name, scope, risk_quantile, min_rank_pct, max_penalty, power))
    return rules


def _rank_series(frame: pd.DataFrame) -> pd.Series:
    if "rank_pct" in frame.columns:
        return pd.to_numeric(frame["rank_pct"], errors="coerce").fillna(0.0)
    if "policy_rank_pct" in frame.columns:
        return pd.to_numeric(frame["policy_rank_pct"], errors="coerce").fillna(0.0)
    return pd.to_numeric(frame.get("normalized_rank_score"), errors="coerce").fillna(0.0)


def _fit_threshold(frame: pd.DataFrame, rule: SmoothRule) -> float:
    scope_mask = pd.Series(True, index=frame.index) if rule.scope == "all" else frame["head"].eq(rule.scope)
    vals = (
        pd.to_numeric(frame.loc[scope_mask, rule.score_name], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return float(vals.quantile(rule.risk_quantile)) if len(vals) else float("nan")


def _fit_threshold_grid(train_scored: pd.DataFrame) -> dict[tuple[str, str, float], float]:
    thresholds: dict[tuple[str, str, float], float] = {}
    for score_name in RISK_SCORE_NAMES:
        if score_name not in train_scored.columns:
            continue
        for scope in ("all", *HEAD_ORDER):
            scope_mask = pd.Series(True, index=train_scored.index) if scope == "all" else train_scored["head"].eq(scope)
            vals = (
                pd.to_numeric(train_scored.loc[scope_mask, score_name], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            for risk_quantile in (0.85, 0.90, 0.95, 0.98):
                thresholds[(score_name, scope, float(risk_quantile))] = (
                    float(vals.quantile(risk_quantile)) if len(vals) else float("nan")
                )
    return thresholds


def _build_month_cache(candidates: pd.DataFrame, first_guard: pd.Timestamp, end: pd.Timestamp) -> list[MonthCache]:
    caches: list[MonthCache] = []
    for start, stop in _month_ranges(first_guard, end):
        train_raw = candidates[candidates["timestamp"].lt(start)].copy().reset_index(drop=True)
        month_mask = candidates["timestamp"].ge(start) & candidates["timestamp"].lt(stop)
        month_raw = candidates[month_mask].copy().reset_index(drop=True)
        if train_raw.empty or month_raw.empty:
            continue
        refs = _fit_percentile_reference(train_raw)
        train_scored = _apply_risk_scores(train_raw, refs)
        month_scored = _apply_risk_scores(month_raw, refs)
        caches.append(
            MonthCache(
                start=start,
                stop=stop,
                indices=candidates.index[month_mask].to_numpy(dtype=np.int64),
                train_rows=int(len(train_raw)),
                month_rows=int(len(month_raw)),
                month_scored=month_scored,
                thresholds=_fit_threshold_grid(train_scored),
            )
        )
    return caches


def _penalty_values(frame: pd.DataFrame, rule: SmoothRule, threshold: float) -> np.ndarray:
    if not np.isfinite(float(threshold)):
        return np.zeros(len(frame), dtype=np.float32)
    scope_mask = pd.Series(True, index=frame.index) if rule.scope == "all" else frame["head"].eq(rule.scope)
    rank_mask = _rank_series(frame).ge(rule.min_rank_pct)
    score = pd.to_numeric(frame[rule.score_name], errors="coerce").fillna(-np.inf).to_numpy(dtype=np.float64)
    denom = max(1.0 - float(threshold), 1e-6)
    intensity = np.clip((score - float(threshold)) / denom, 0.0, 1.0)
    if abs(rule.power - 1.0) > 1e-12:
        intensity = np.power(intensity, float(rule.power))
    mask = (scope_mask & rank_mask).to_numpy(dtype=bool)
    penalty = np.zeros(len(frame), dtype=np.float32)
    penalty[mask] = (-float(rule.max_penalty) * intensity[mask]).astype(np.float32)
    return penalty


def _apply_smooth_rule_expanding(
    candidates: pd.DataFrame,
    rule: SmoothRule,
    first_guard: pd.Timestamp,
    end: pd.Timestamp,
    month_cache: list[MonthCache] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = candidates.copy()
    base_adj = (
        pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "portfolio_rank_adjustment" in out.columns
        else np.zeros(len(out), dtype=np.float32)
    )
    adjustment = base_adj.copy()
    rows: list[dict[str, Any]] = []
    caches = month_cache if month_cache is not None else _build_month_cache(candidates, first_guard, end)
    for cache in caches:
        threshold = cache.thresholds.get((rule.score_name, rule.scope, float(rule.risk_quantile)), float("nan"))
        penalty = _penalty_values(cache.month_scored, rule, threshold)
        idx = cache.indices
        adjustment[idx] = np.clip(adjustment[idx] + penalty, -1.0, 1.0)
        rows.append(
            {
                "label": rule.label,
                "month_start": cache.start.isoformat(),
                "month_end": cache.stop.isoformat(),
                "train_rows": cache.train_rows,
                "month_rows": cache.month_rows,
                "threshold": threshold,
                "penalized_rows": int(np.sum(penalty < 0.0)),
                "mean_penalty": float(np.mean(penalty[penalty < 0.0])) if np.any(penalty < 0.0) else 0.0,
            }
        )
    out["portfolio_rank_adjustment"] = adjustment.astype("float32")
    return out, pd.DataFrame(rows)


def _delta_summary(base: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    row = {f"baseline_{k}": v for k, v in base.items() if k != "label"}
    row.update({f"smooth_{k}": v for k, v in guard.items() if k != "label"})
    for key in (
        "net_pnl",
        "gross_pnl",
        "trade_count",
        "hit_rate",
        "full_sl_rate",
        "timeout_rate",
        "max_drawdown",
        "objective_week",
        "q20_week_net_pnl",
        "q35_week_net_pnl",
        "worst_week_net_pnl",
        "positive_weeks",
    ):
        row[f"delta_{key}"] = float(row[f"smooth_{key}"] - row[f"baseline_{key}"])
    return row


def _accepted_proxy(
    candidates: pd.DataFrame,
    baseline_decisions: pd.DataFrame,
    rule: SmoothRule,
    first_guard: pd.Timestamp,
    end: pd.Timestamp,
    month_cache: list[MonthCache] | None = None,
) -> dict[str, Any]:
    adjusted, schedule = _apply_smooth_rule_expanding(candidates, rule, first_guard, end, month_cache)
    accepted = baseline_decisions[baseline_decisions["accepted"].astype(bool)].copy()
    if accepted.empty or "candidate_index" not in accepted.columns:
        return {"label": rule.label, **asdict(rule), "proxy_score": -np.inf, "penalized_accepted_count": 0}
    accepted["candidate_index"] = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("int64")
    accepted = accepted[accepted["candidate_index"].between(0, len(candidates) - 1)].copy()
    idx = accepted["candidate_index"].to_numpy(dtype=np.int64)
    penalty = pd.to_numeric(adjusted["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    base_penalty = (
        pd.to_numeric(candidates["portfolio_rank_adjustment"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "portfolio_rank_adjustment" in candidates.columns
        else np.zeros(len(candidates), dtype=np.float64)
    )
    delta_penalty = penalty[idx] - base_penalty[idx]
    affected = delta_penalty < -1e-12
    removed = accepted[affected].copy()
    n = int(len(removed))
    if n == 0:
        proxy_score = -np.inf
        net = 0.0
        full_sl = 0.0
        hit = 0.0
    else:
        size = pd.to_numeric(removed["position_size"], errors="coerce").fillna(0.0)
        net_ret = pd.to_numeric(removed["position_net_return"], errors="coerce").fillna(0.0)
        net = float((size * net_ret).sum())
        reason = removed["position_exit_reason"].astype(str)
        full_sl = float(reason.str.contains("sl", case=False, na=False).mean())
        hit = float((net_ret > 0.0).mean())
        # The proxy rewards penalizing losing/full-SL accepted trades but also
        # penalizes broad intervention because smooth penalties can reshuffle
        # winners through the auction.
        mean_abs_penalty = float(np.mean(np.abs(delta_penalty[affected]))) if np.any(affected) else 0.0
        proxy_score = -net + 500.0 * full_sl - 250.0 * hit - 2000.0 * mean_abs_penalty
    return {
        "label": rule.label,
        **asdict(rule),
        "proxy_score": float(proxy_score),
        "penalized_accepted_count": n,
        "penalized_accepted_net_pnl": net,
        "penalized_accepted_full_sl_rate": full_sl,
        "penalized_accepted_hit_rate": hit,
        "total_penalized_rows": int(schedule["penalized_rows"].sum()) if not schedule.empty else 0,
    }


def _monthly_table(base_weekly: pd.DataFrame, smooth_weekly: pd.DataFrame) -> pd.DataFrame:
    def prep(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        cur = frame[frame["period_type"].eq("week")].copy()
        cur["week_start"] = pd.PeriodIndex(cur["week"], freq="W").start_time
        cur["month"] = cur["week_start"].dt.to_period("M").astype(str)
        out = (
            cur.groupby("month", as_index=False)
            .agg(
                net_pnl=("net_pnl", "sum"),
                trades=("trades", "sum"),
                hit_rate=("hit_rate", "mean"),
                full_sl_rate=("full_sl_rate", "mean"),
                timeout_rate=("timeout_rate", "mean"),
                worst_week_net_pnl=("net_pnl", "min"),
            )
        )
        return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "month"})

    out = prep(base_weekly, "baseline").merge(prep(smooth_weekly, "smooth"), on="month", how="outer")
    for key in ("net_pnl", "trades", "hit_rate", "full_sl_rate", "timeout_rate", "worst_week_net_pnl"):
        out[f"delta_{key}"] = out[f"smooth_{key}"] - out[f"baseline_{key}"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_rank_penalty_20260701"))
    parser.add_argument("--first-guard-month", default="2026-02-01")
    parser.add_argument("--end", default="2026-06-27")
    parser.add_argument("--top-rules", type=int, default=16)
    parser.add_argument("--min-penalized-accepted", type=int, default=10)
    parser.add_argument("--include-family-representatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(args.input_dir / "combo_candidates.parquet")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    if "portfolio_rank_adjustment" not in candidates.columns:
        candidates["portfolio_rank_adjustment"] = np.float32(0.0)
    first_guard = pd.Timestamp(args.first_guard_month, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    month_cache = _build_month_cache(candidates, first_guard, end)

    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    ev_train = candidates[candidates["timestamp"].lt(first_guard)].copy().reset_index(drop=True)
    ev_curve = fit_hierarchical_ev_curves(ev_train)
    baseline_decisions, _base_equity, baseline_metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    _base_daily, baseline_weekly = _period_tables(baseline_decisions)
    baseline_summary = _summary("baseline", baseline_decisions, baseline_weekly, baseline_metrics, args.q35_weight, args.q20_weight)

    proxy_rows = []
    for rule in _candidate_rules():
        row = _accepted_proxy(candidates, baseline_decisions, rule, first_guard, end, month_cache)
        if int(row["penalized_accepted_count"]) >= int(args.min_penalized_accepted):
            proxy_rows.append(row)
    proxy = pd.DataFrame(proxy_rows)
    if proxy.empty:
        raise RuntimeError("No smooth rank-penalty rules passed proxy screening")
    proxy = proxy.sort_values(["proxy_score", "penalized_accepted_full_sl_rate"], ascending=[False, False]).reset_index(drop=True)
    shortlist_parts = [proxy.head(int(args.top_rules)).copy()]
    if bool(args.include_family_representatives):
        family_best = (
            proxy.sort_values("proxy_score", ascending=False)
            .groupby("score_name", as_index=False)
            .head(1)
            .copy()
        )
        shortlist_parts.append(family_best)
    shortlist = (
        pd.concat(shortlist_parts, ignore_index=True)
        .drop_duplicates(subset=["label"])
        .sort_values("proxy_score", ascending=False)
        .reset_index(drop=True)
    )

    summary_rows = [{**baseline_summary, "label": "baseline", "run_id": -1}]
    schedule_rows: list[pd.DataFrame] = []
    weekly_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    for run_id, row in shortlist.iterrows():
        rule = SmoothRule(
            str(row["score_name"]),
            str(row["scope"]),
            float(row["risk_quantile"]),
            float(row["min_rank_pct"]),
            float(row["max_penalty"]),
            float(row["power"]),
        )
        adjusted, schedule = _apply_smooth_rule_expanding(candidates, rule, first_guard, end, month_cache)
        decisions, _equity, metrics = replay_candidates(
            adjusted,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            market_mode="perps",
        )
        _daily, weekly = _period_tables(decisions)
        summary = _summary(rule.label, decisions, weekly, metrics, args.q35_weight, args.q20_weight)
        delta = _delta_summary(baseline_summary, summary)
        summary.update({**asdict(rule), "run_id": int(run_id), **{k: v for k, v in delta.items() if k.startswith("delta_")}})
        summary["proxy_score"] = float(row["proxy_score"])
        summary["total_penalized_rows"] = int(schedule["penalized_rows"].sum()) if not schedule.empty else 0
        summary["penalized_accepted_count"] = int(row["penalized_accepted_count"])
        summary_rows.append(summary)
        if not schedule.empty:
            schedule = schedule.copy()
            schedule["run_id"] = int(run_id)
            schedule_rows.append(schedule)
        weekly = weekly.copy()
        weekly["run_id"] = int(run_id)
        weekly["label"] = rule.label
        weekly_rows.append(weekly)
        monthly = _monthly_table(baseline_weekly, weekly)
        monthly["run_id"] = int(run_id)
        monthly["label"] = rule.label
        monthly_rows.append(monthly)

    summary_df = pd.DataFrame(summary_rows)
    guard_df = summary_df[summary_df["run_id"].ge(0)].copy()
    guard_df = guard_df.sort_values(
        ["delta_objective_week", "delta_net_pnl", "delta_full_sl_rate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    summary_out = pd.concat([summary_df[summary_df["run_id"].lt(0)], guard_df], ignore_index=True)
    schedule_out = pd.concat(schedule_rows, ignore_index=True) if schedule_rows else pd.DataFrame()
    weekly_out = pd.concat(weekly_rows, ignore_index=True) if weekly_rows else pd.DataFrame()
    monthly_out = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()

    proxy.to_csv(args.output_dir / "smooth_rank_penalty_proxy_screen.csv", index=False)
    shortlist.to_csv(args.output_dir / "smooth_rank_penalty_shortlist.csv", index=False)
    summary_out.to_csv(args.output_dir / "smooth_rank_penalty_replay_summary.csv", index=False)
    schedule_out.to_csv(args.output_dir / "smooth_rank_penalty_schedule.csv", index=False)
    baseline_weekly.to_csv(args.output_dir / "smooth_rank_penalty_baseline_weekly.csv", index=False)
    weekly_out.to_csv(args.output_dir / "smooth_rank_penalty_replay_weekly.csv", index=False)
    monthly_out.to_csv(args.output_dir / "smooth_rank_penalty_monthly.csv", index=False)

    best = guard_df.iloc[0]
    best_tail = guard_df.sort_values(["delta_full_sl_rate", "delta_net_pnl"], ascending=[True, False]).iloc[0]
    manifest = {
        "generated_by": "ablate_wfrecent_smooth_rank_penalty",
        "input_dir": str(args.input_dir),
        "candidate_rows": int(len(candidates)),
        "rules_screened": int(len(proxy)),
        "rules_replayed": int(len(shortlist)),
        "include_family_representatives": bool(args.include_family_representatives),
        "first_guard_month": args.first_guard_month,
        "end": args.end,
        "ev_curve_fit": "pre_guard_history_only",
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Smooth Diagnostic Rank-Penalty Ablation",
        "",
        "Smooth rank penalties are applied using expanding prior-month diagnostic references. The EV curve is fit once on pre-guard history and shared across arms.",
        "",
        "## Baseline And Best Rules",
        "",
        _fmt_table(
            summary_out.head(8),
            [
                "label",
                "net_pnl",
                "delta_net_pnl",
                "objective_week",
                "delta_objective_week",
                "hit_rate",
                "delta_hit_rate",
                "full_sl_rate",
                "delta_full_sl_rate",
                "timeout_rate",
                "delta_timeout_rate",
                "worst_week_net_pnl",
                "delta_worst_week_net_pnl",
                "trade_count",
                "delta_trade_count",
                "total_penalized_rows",
            ],
        ),
        "",
        "## Best Objective Rule",
        "",
        _fmt_table(pd.DataFrame([best]), ["label", "score_name", "scope", "risk_quantile", "min_rank_pct", "max_penalty", "power", "delta_net_pnl", "delta_objective_week", "delta_full_sl_rate", "delta_worst_week_net_pnl"]),
        "",
        "## Best Full-SL Rule",
        "",
        _fmt_table(pd.DataFrame([best_tail]), ["label", "score_name", "scope", "risk_quantile", "min_rank_pct", "max_penalty", "power", "delta_net_pnl", "delta_objective_week", "delta_full_sl_rate", "delta_worst_week_net_pnl"]),
        "",
        "## Monthly Deltas For Best Objective Rule",
        "",
        _fmt_table(monthly_out[monthly_out["run_id"].eq(int(best["run_id"]))], ["month", "delta_net_pnl", "delta_trades", "delta_hit_rate", "delta_full_sl_rate", "delta_timeout_rate", "delta_worst_week_net_pnl"]),
    ]
    (args.output_dir / "smooth_rank_penalty_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
