#!/usr/bin/env python3
"""Chronological validation for row-level wf_recent execution guards.

This validates the promising row-level veto family without using full-sample
diagnostic percentiles or thresholds. For each monthly holdout:

1. Fit diagnostic risk percentiles on prior candidate rows only.
2. Replay the prior training period to identify accepted-trade failure modes.
3. Screen veto rules on prior accepted trades.
4. Optionally replay the top screened rules on the prior training period.
5. Apply the selected rule to the next month and replay against a baseline
   using train-fitted EV curves.

This is a chronological development validation, not fresh untouched OOS.
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


HEAD_ORDER = ("long_bars", "long_dist", "short_asset", "short_bollinger")
RAW_GROUPS: dict[str, tuple[tuple[str, bool], ...]] = {
    "uncertainty_risk": (
        ("generated_score_uncertainty_p1mp", False),
        ("generated_score_entropy", False),
        ("generated_score_abs_distance_from_half", True),
    ),
    "drift_risk": (
        ("generated_score_abs_diff_1", False),
        ("generated_score_abs_diff_4", False),
        ("generated_score_abs_diff_24", False),
        ("generated_score_abs_minus_prev24_mean", False),
        ("generated_score_prev24_std", False),
        ("generated_strategy_score_shift_abs_z", False),
    ),
    "ood_risk": (
        ("generated_strategy_score_ood_abs_z", False),
        ("generated_strategy_barrier_ood_abs_z", False),
        ("generated_strategy_friction_ood_abs_z", False),
    ),
    "recent_perf_risk": (
        ("generated_hr_surprise_24", True),
        ("generated_hr_surprise_96", True),
        ("generated_weighted_hr_surprise_24", True),
        ("generated_weighted_hr_surprise_96", True),
        ("generated_loss_rate_24", False),
        ("generated_loss_rate_96", False),
    ),
    "friction_risk": (
        ("expected_friction_bps", False),
        ("price_gap_bps", False),
        ("entry_gap_bps", False),
        ("entry_slippage_proxy_bps", False),
        ("orderbook_slippage_bps", False),
        ("delay_max_adverse_bps", False),
        ("liquidity_capacity_weight", True),
    ),
}
RISK_SCORE_NAMES = tuple(RAW_GROUPS) + ("composite_risk",)


@dataclass(frozen=True)
class VetoRule:
    score_name: str
    scope: str
    risk_quantile: float
    min_rank_pct: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[cols].head(max_rows).copy() if max_rows else frame[cols].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.3f}")
    return view.to_markdown(index=False)


def _head_name(strategy_id: Any) -> str:
    text = str(strategy_id)
    if text.startswith("short_bollinger"):
        return "short_bollinger"
    parts = text.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else text


def _period_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["day"] = ts.dt.date.astype(str)
    accepted["week"] = ts.dt.to_period("W").astype(str)
    accepted["head"] = accepted["strategy_id"].map(_head_name)
    size = pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
    net = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(accepted["position_gross_return"], errors="coerce").fillna(0.0)
    accepted["net_pnl_amount"] = size * net
    accepted["gross_pnl_amount"] = size * gross
    accepted["is_win"] = net > 0.0
    reason = accepted["position_exit_reason"].astype(str) if "position_exit_reason" in accepted.columns else pd.Series("", index=accepted.index)
    accepted["is_full_sl"] = reason.str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = reason.str.contains("timeout", case=False, na=False)
    frames_daily: list[pd.DataFrame] = []
    frames_weekly: list[pd.DataFrame] = []
    for cols, target in ((["day"], frames_daily), (["day", "head"], frames_daily), (["week"], frames_weekly), (["week", "head"], frames_weekly)):
        cur = (
            accepted.groupby(cols, as_index=False)
            .agg(
                net_pnl=("net_pnl_amount", "sum"),
                gross_pnl=("gross_pnl_amount", "sum"),
                trades=("accepted", "size"),
                hit_rate=("is_win", "mean"),
                full_sl_rate=("is_full_sl", "mean"),
                timeout_rate=("is_timeout", "mean"),
            )
            .sort_values(cols)
        )
        cur.insert(0, "period_type", "_".join(cols))
        target.append(cur)
    return pd.concat(frames_daily, ignore_index=True), pd.concat(frames_weekly, ignore_index=True)


def _objective_from_weekly(weekly: pd.DataFrame, q35_weight: float, q20_weight: float) -> float:
    values = pd.to_numeric(
        weekly.loc[weekly["period_type"].eq("week"), "net_pnl"], errors="coerce"
    ).dropna().to_numpy(dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values) + q35_weight * np.quantile(values, 0.35) + q20_weight * np.quantile(values, 0.20))


def _candidate_rules(recent_perf_only: bool = False) -> list[VetoRule]:
    scopes = ("all",) + HEAD_ORDER
    score_names = ("recent_perf_risk", "friction_risk", "composite_risk") if recent_perf_only else RISK_SCORE_NAMES
    rules: list[VetoRule] = []
    for score_name in score_names:
        for scope in scopes:
            for risk_quantile in (0.90, 0.95, 0.98):
                for min_rank_pct in (0.70, 0.80, 0.90):
                    rules.append(VetoRule(score_name, scope, risk_quantile, min_rank_pct))
    return rules


def _fixed_challenger_rules() -> list[VetoRule]:
    return [
        VetoRule("recent_perf_risk", "all", 0.90, 0.70),
        VetoRule("recent_perf_risk", "all", 0.95, 0.70),
        VetoRule("recent_perf_risk", "all", 0.95, 0.80),
        VetoRule("recent_perf_risk", "all", 0.98, 0.90),
        VetoRule("recent_perf_risk", "long_bars", 0.90, 0.70),
        VetoRule("recent_perf_risk", "short_asset", 0.90, 0.70),
        VetoRule("recent_perf_risk", "short_bollinger", 0.98, 0.70),
        VetoRule("friction_risk", "long_dist", 0.90, 0.70),
    ]


def _fit_percentile_reference(train: pd.DataFrame) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    refs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for head, group in train.groupby("head", sort=False):
        refs[str(head)] = {}
        for cols in RAW_GROUPS.values():
            for col, _invert in cols:
                if col in refs[str(head)]:
                    continue
                vals = pd.to_numeric(group.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=np.float64)
                refs[str(head)][col] = {"sorted": np.sort(vals) if vals.size else np.asarray([], dtype=np.float64)}
    return refs


def _percentile_from_ref(values: pd.Series, heads: pd.Series, refs: dict[str, dict[str, dict[str, np.ndarray]]], col: str) -> np.ndarray:
    out = np.full(len(values), 0.5, dtype=np.float32)
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    head_arr = heads.astype(str).to_numpy()
    for head in np.unique(head_arr):
        mask = head_arr == head
        sorted_vals = refs.get(str(head), {}).get(col, {}).get("sorted", np.asarray([], dtype=np.float64))
        if sorted_vals.size == 0:
            continue
        cur = vals[mask]
        pct = np.full(cur.size, 0.5, dtype=np.float64)
        finite = np.isfinite(cur)
        pct[finite] = np.searchsorted(sorted_vals, cur[finite], side="right") / float(sorted_vals.size)
        out[mask] = np.clip(pct, 0.0, 1.0).astype(np.float32)
    return out


def _apply_risk_scores(frame: pd.DataFrame, refs: dict[str, dict[str, dict[str, np.ndarray]]]) -> pd.DataFrame:
    out = frame.copy()
    for score_name, cols in RAW_GROUPS.items():
        parts = []
        for col, invert in cols:
            if col not in out.columns:
                continue
            pct = _percentile_from_ref(out[col], out["head"], refs, col)
            if invert:
                pct = 1.0 - pct
            parts.append(pd.Series(pct, index=out.index))
        if parts:
            out[score_name] = pd.concat(parts, axis=1).mean(axis=1, skipna=True).fillna(0.5).astype("float32")
        else:
            out[score_name] = np.float32(0.5)
    out["composite_risk"] = out[list(RAW_GROUPS)].mean(axis=1, skipna=True).astype("float32")
    return out


def _fit_rule_thresholds(train: pd.DataFrame, rules: list[VetoRule]) -> dict[VetoRule, float]:
    thresholds: dict[VetoRule, float] = {}
    for rule in rules:
        scope_mask = pd.Series(True, index=train.index) if rule.scope == "all" else train["head"].eq(rule.scope)
        vals = pd.to_numeric(train.loc[scope_mask, rule.score_name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        thresholds[rule] = float(vals.quantile(rule.risk_quantile)) if len(vals) else float("nan")
    return thresholds


def _rule_mask(frame: pd.DataFrame, rule: VetoRule, threshold: float) -> pd.Series:
    if not np.isfinite(float(threshold)):
        return pd.Series(False, index=frame.index)
    scope_mask = pd.Series(True, index=frame.index) if rule.scope == "all" else frame["head"].eq(rule.scope)
    ranks = pd.to_numeric(frame.get("rank_pct", frame.get("policy_rank_pct")), errors="coerce").fillna(0.0)
    score = pd.to_numeric(frame[rule.score_name], errors="coerce").fillna(-np.inf)
    return scope_mask & ranks.ge(rule.min_rank_pct) & score.ge(float(threshold))


def _apply_veto(frame: pd.DataFrame, rule: VetoRule, threshold: float) -> tuple[pd.DataFrame, int]:
    out = frame.copy()
    if "portfolio_rank_adjustment" not in out.columns:
        out["portfolio_rank_adjustment"] = 0.0
    else:
        out["portfolio_rank_adjustment"] = pd.to_numeric(out["portfolio_rank_adjustment"], errors="coerce").fillna(0.0)
    mask = _rule_mask(out, rule, threshold)
    out.loc[mask, "portfolio_rank_adjustment"] = -1.0
    return out, int(mask.sum())


def _summary(label: str, decisions: pd.DataFrame, weekly: pd.DataFrame, metrics: dict[str, Any], q35_weight: float, q20_weight: float) -> dict[str, Any]:
    accepted = decisions[decisions["accepted"].astype(bool)].copy()
    values = pd.to_numeric(
        weekly.loc[weekly["period_type"].eq("week"), "net_pnl"], errors="coerce"
    ).dropna().to_numpy(dtype=np.float64)
    return {
        "label": label,
        "net_pnl": float(metrics.get("net_pnl", np.nan)),
        "gross_pnl": float(metrics.get("gross_pnl", np.nan)),
        "trade_count": int(metrics.get("trade_count", len(accepted))),
        "hit_rate": float((pd.to_numeric(accepted["position_net_return"], errors="coerce") > 0.0).mean()) if len(accepted) else np.nan,
        "full_sl_rate": float(metrics.get("full_sl_rate", np.nan)),
        "timeout_rate": float(metrics.get("timeout_rate", np.nan)),
        "max_drawdown": float(metrics.get("max_drawdown", np.nan)),
        "objective_week": _objective_from_weekly(weekly, q35_weight, q20_weight),
        "q20_week_net_pnl": float(np.quantile(values, 0.20)) if values.size else np.nan,
        "q35_week_net_pnl": float(np.quantile(values, 0.35)) if values.size else np.nan,
        "worst_week_net_pnl": float(np.min(values)) if values.size else np.nan,
        "positive_weeks": int(np.sum(values > 0.0)) if values.size else 0,
    }


def _screen_rules(
    train: pd.DataFrame,
    train_decisions: pd.DataFrame,
    rules: list[VetoRule],
    thresholds: dict[VetoRule, float],
    min_removed: int,
) -> pd.DataFrame:
    accepted = train_decisions[train_decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["candidate_index"] = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("int64")
    accepted = accepted.merge(
        train[["head", *RISK_SCORE_NAMES]].reset_index(names="candidate_index"),
        on="candidate_index",
        how="left",
    )
    accepted["net_pnl_amount"] = (
        pd.to_numeric(accepted["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0)
    )
    accepted["is_win"] = pd.to_numeric(accepted["position_net_return"], errors="coerce").fillna(0.0) > 0.0
    accepted["is_full_sl"] = accepted["position_exit_reason"].astype(str).str.contains("sl", case=False, na=False)
    accepted["is_timeout"] = accepted["position_exit_reason"].astype(str).str.contains("timeout", case=False, na=False)
    accepted_idx = accepted["candidate_index"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for rule in rules:
        threshold = thresholds.get(rule, float("nan"))
        mask = _rule_mask(train, rule, threshold)
        removed = accepted[mask.iloc[accepted_idx].to_numpy(dtype=bool)].copy()
        n = int(len(removed))
        if n < int(min_removed):
            continue
        removed_net = float(removed["net_pnl_amount"].sum())
        removed_full_sl = int(removed["is_full_sl"].sum())
        removed_timeout = int(removed["is_timeout"].sum())
        removed_win = int(removed["is_win"].sum())
        proxy_delta = -removed_net
        proxy_score = proxy_delta + 25.0 * removed_full_sl + 10.0 * removed_timeout - 25.0 * removed_win
        rows.append(
            {
                **asdict(rule),
                "threshold": threshold,
                "candidate_veto_count": int(mask.sum()),
                "removed_accepted_count": n,
                "removed_net_pnl": removed_net,
                "proxy_delta_net_no_backfill": proxy_delta,
                "proxy_score": float(proxy_score),
                "removed_full_sl_rate": float(removed["is_full_sl"].mean()) if n else 0.0,
                "removed_timeout_rate": float(removed["is_timeout"].mean()) if n else 0.0,
                "removed_hit_rate": float(removed["is_win"].mean()) if n else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["proxy_score", "proxy_delta_net_no_backfill"], ascending=[False, False]).reset_index(drop=True)


def _replay(frame: pd.DataFrame, ev_train_frame: pd.DataFrame, params: PortfolioPolicyParams) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ev_curve = fit_hierarchical_ev_curves(ev_train_frame)
    decisions, _equity, metrics = replay_candidates(frame, params, mode="global_auction", ev_curve=ev_curve, market_mode="perps")
    _daily, weekly = _period_tables(decisions)
    return decisions, weekly, metrics


def _month_splits(candidates: pd.DataFrame, first_holdout: str, last_holdout: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(first_holdout, tz="UTC")
    end = pd.Timestamp(last_holdout, tz="UTC")
    splits: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = cur + pd.offsets.MonthBegin(1)
        splits.append((cur, min(pd.Timestamp(nxt), end)))
        cur = pd.Timestamp(nxt)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_materialized_wf_recent_q35w07_q20w03_6mo_20260701"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_perp/reports/contextual_tp_sl_wfrecent_row_guard_walkforward_20260701"))
    parser.add_argument("--first-holdout", default="2026-02-01")
    parser.add_argument("--last-holdout-end", default="2026-06-27")
    parser.add_argument("--top-train-replays", type=int, default=4)
    parser.add_argument("--min-removed", type=int, default=20)
    parser.add_argument("--recent-perf-focused", action="store_true", default=True)
    parser.add_argument("--q35-weight", type=float, default=0.70)
    parser.add_argument("--q20-weight", type=float, default=0.30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_parquet(args.input_dir / "combo_candidates.parquet")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True, errors="coerce")
    candidates = candidates[candidates["timestamp"].notna()].sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_head_name)
    params = PortfolioPolicyParams(global_threshold_floor=0.0)
    rules = _candidate_rules(recent_perf_only=bool(args.recent_perf_focused))
    splits = _month_splits(candidates, args.first_holdout, args.last_holdout_end)

    split_rows: list[dict[str, Any]] = []
    train_rule_rows: list[pd.DataFrame] = []
    holdout_weekly_rows: list[pd.DataFrame] = []
    fixed_rule_rows: list[dict[str, Any]] = []
    fixed_weekly_rows: list[pd.DataFrame] = []
    fixed_rules = _fixed_challenger_rules()

    for split_id, (holdout_start, holdout_end) in enumerate(splits):
        raw_train = candidates[candidates["timestamp"].lt(holdout_start)].copy().reset_index(drop=True)
        raw_holdout = candidates[candidates["timestamp"].ge(holdout_start) & candidates["timestamp"].lt(holdout_end)].copy().reset_index(drop=True)
        if raw_train.empty or raw_holdout.empty:
            continue
        refs = _fit_percentile_reference(raw_train)
        train = _apply_risk_scores(raw_train, refs).reset_index(drop=True)
        holdout = _apply_risk_scores(raw_holdout, refs).reset_index(drop=True)
        thresholds = _fit_rule_thresholds(train, rules)

        train_base_decisions, train_base_weekly, train_base_metrics = _replay(train, train, params)
        screen = _screen_rules(train, train_base_decisions, rules, thresholds, args.min_removed)
        selected_rule: VetoRule | None = None
        selected_threshold = float("nan")
        selected_train_summary: dict[str, Any] = {}
        if not screen.empty:
            candidates_to_replay = screen.head(int(args.top_train_replays)).copy()
            train_eval_rows = []
            train_base_summary = _summary("train_baseline", train_base_decisions, train_base_weekly, train_base_metrics, args.q35_weight, args.q20_weight)
            for train_run_id, row in candidates_to_replay.iterrows():
                rule = VetoRule(str(row["score_name"]), str(row["scope"]), float(row["risk_quantile"]), float(row["min_rank_pct"]))
                threshold = float(row["threshold"])
                train_guarded, train_veto_count = _apply_veto(train, rule, threshold)
                train_decisions, train_weekly, train_metrics = _replay(train_guarded, train, params)
                cur = _summary("train_guard", train_decisions, train_weekly, train_metrics, args.q35_weight, args.q20_weight)
                cur.update({**asdict(rule), "threshold": threshold, "train_veto_count": train_veto_count, "train_run_id": int(train_run_id)})
                for key in ("net_pnl", "objective_week", "q20_week_net_pnl", "worst_week_net_pnl", "full_sl_rate", "timeout_rate", "max_drawdown", "hit_rate"):
                    cur[f"delta_{key}_vs_train_baseline"] = float(cur[key] - train_base_summary[key])
                train_eval_rows.append(cur)
            train_eval = pd.DataFrame(train_eval_rows)
            if not train_eval.empty:
                train_eval["passes_gate"] = (
                    train_eval["delta_net_pnl_vs_train_baseline"].ge(0.0)
                    & train_eval["delta_objective_week_vs_train_baseline"].ge(0.0)
                    & train_eval["delta_full_sl_rate_vs_train_baseline"].le(0.0)
                    & train_eval["delta_worst_week_net_pnl_vs_train_baseline"].ge(-500.0)
                )
                train_eval["split_id"] = split_id
                train_eval["holdout_start"] = holdout_start.isoformat()
                train_eval["holdout_end"] = holdout_end.isoformat()
                train_rule_rows.append(train_eval)
                eligible = train_eval[train_eval["passes_gate"]].copy()
                if not eligible.empty:
                    best_train = eligible.sort_values(
                        ["delta_objective_week_vs_train_baseline", "delta_net_pnl_vs_train_baseline", "delta_full_sl_rate_vs_train_baseline"],
                        ascending=[False, False, True],
                    ).iloc[0]
                    selected_rule = VetoRule(str(best_train["score_name"]), str(best_train["scope"]), float(best_train["risk_quantile"]), float(best_train["min_rank_pct"]))
                    selected_threshold = float(best_train["threshold"])
                    selected_train_summary = best_train.to_dict()

        holdout_base_decisions, holdout_base_weekly, holdout_base_metrics = _replay(holdout, train, params)
        holdout_base_summary = _summary("holdout_baseline", holdout_base_decisions, holdout_base_weekly, holdout_base_metrics, args.q35_weight, args.q20_weight)

        for fixed_rule in fixed_rules:
            fixed_threshold = thresholds.get(fixed_rule, float("nan"))
            fixed_holdout, fixed_veto_count = _apply_veto(holdout, fixed_rule, fixed_threshold)
            fixed_decisions, fixed_weekly, fixed_metrics = _replay(fixed_holdout, train, params)
            fixed_summary = _summary("fixed_guard", fixed_decisions, fixed_weekly, fixed_metrics, args.q35_weight, args.q20_weight)
            fixed_label = f"{fixed_rule.scope}__{fixed_rule.score_name}__q{int(fixed_rule.risk_quantile*100)}__rank{int(fixed_rule.min_rank_pct*100)}"
            fixed_row = {
                "split_id": split_id,
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "fixed_label": fixed_label,
                "fixed_threshold": fixed_threshold,
                "holdout_veto_count": int(fixed_veto_count),
                **{f"baseline_{k}": v for k, v in holdout_base_summary.items() if k != "label"},
                **{f"guard_{k}": v for k, v in fixed_summary.items() if k != "label"},
            }
            for key in ("net_pnl", "gross_pnl", "trade_count", "hit_rate", "full_sl_rate", "timeout_rate", "max_drawdown", "objective_week", "q20_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "positive_weeks"):
                fixed_row[f"delta_{key}"] = float(fixed_row[f"guard_{key}"] - fixed_row[f"baseline_{key}"])
            fixed_rule_rows.append(fixed_row)
            fixed_weekly = fixed_weekly.copy()
            fixed_weekly["split_id"] = split_id
            fixed_weekly["variant"] = fixed_label
            fixed_weekly_rows.append(fixed_weekly)

        if selected_rule is None:
            holdout_guard_summary = dict(holdout_base_summary)
            holdout_guard_summary["label"] = "holdout_no_guard"
            holdout_guard_weekly = holdout_base_weekly.copy()
            holdout_veto_count = 0
            selected_label = "no_guard"
        else:
            holdout_guarded, holdout_veto_count = _apply_veto(holdout, selected_rule, selected_threshold)
            holdout_guard_decisions, holdout_guard_weekly, holdout_guard_metrics = _replay(holdout_guarded, train, params)
            holdout_guard_summary = _summary("holdout_guard", holdout_guard_decisions, holdout_guard_weekly, holdout_guard_metrics, args.q35_weight, args.q20_weight)
            holdout_guard_weekly = holdout_guard_weekly.copy()
            selected_label = f"{selected_rule.scope}__{selected_rule.score_name}__q{int(selected_rule.risk_quantile*100)}__rank{int(selected_rule.min_rank_pct*100)}"

        row = {
            "split_id": split_id,
            "holdout_start": holdout_start.isoformat(),
            "holdout_end": holdout_end.isoformat(),
            "train_rows": int(len(train)),
            "holdout_rows": int(len(holdout)),
            "selected_label": selected_label,
            "selected_threshold": selected_threshold,
            "holdout_veto_count": int(holdout_veto_count),
            **{f"baseline_{k}": v for k, v in holdout_base_summary.items() if k != "label"},
            **{f"guard_{k}": v for k, v in holdout_guard_summary.items() if k != "label"},
        }
        for key in ("net_pnl", "gross_pnl", "trade_count", "hit_rate", "full_sl_rate", "timeout_rate", "max_drawdown", "objective_week", "q20_week_net_pnl", "q35_week_net_pnl", "worst_week_net_pnl", "positive_weeks"):
            row[f"delta_{key}"] = float(row[f"guard_{key}"] - row[f"baseline_{key}"])
        for key, val in selected_train_summary.items():
            if key in {"score_name", "scope", "risk_quantile", "min_rank_pct", "threshold"}:
                row[f"selected_{key}"] = val
            elif str(key).startswith("delta_"):
                row[f"selected_train_{key}"] = val
        split_rows.append(row)

        holdout_base_weekly = holdout_base_weekly.copy()
        holdout_base_weekly["split_id"] = split_id
        holdout_base_weekly["variant"] = "baseline"
        holdout_guard_weekly["split_id"] = split_id
        holdout_guard_weekly["variant"] = "guard"
        holdout_weekly_rows.extend([holdout_base_weekly, holdout_guard_weekly])

    split_df = pd.DataFrame(split_rows)
    train_rules_df = pd.concat(train_rule_rows, ignore_index=True) if train_rule_rows else pd.DataFrame()
    weekly_df = pd.concat(holdout_weekly_rows, ignore_index=True) if holdout_weekly_rows else pd.DataFrame()
    fixed_df = pd.DataFrame(fixed_rule_rows)
    fixed_weekly_df = pd.concat(fixed_weekly_rows, ignore_index=True) if fixed_weekly_rows else pd.DataFrame()
    split_df.to_csv(args.output_dir / "row_guard_walkforward_splits.csv", index=False)
    train_rules_df.to_csv(args.output_dir / "row_guard_walkforward_train_rule_replays.csv", index=False)
    weekly_df.to_csv(args.output_dir / "row_guard_walkforward_weekly.csv", index=False)
    fixed_df.to_csv(args.output_dir / "row_guard_walkforward_fixed_rules.csv", index=False)
    fixed_weekly_df.to_csv(args.output_dir / "row_guard_walkforward_fixed_weekly.csv", index=False)

    summary = {}
    if not split_df.empty:
        summary = {
            "splits": int(len(split_df)),
            "sum_delta_net_pnl": float(split_df["delta_net_pnl"].sum()),
            "median_delta_net_pnl": float(split_df["delta_net_pnl"].median()),
            "positive_delta_net_pnl_share": float((split_df["delta_net_pnl"] > 0).mean()),
            "sum_delta_objective_week": float(split_df["delta_objective_week"].sum()),
            "median_delta_objective_week": float(split_df["delta_objective_week"].median()),
            "positive_delta_objective_share": float((split_df["delta_objective_week"] > 0).mean()),
            "sum_delta_worst_week_net_pnl": float(split_df["delta_worst_week_net_pnl"].sum()),
            "median_delta_worst_week_net_pnl": float(split_df["delta_worst_week_net_pnl"].median()),
            "mean_delta_full_sl_rate": float(split_df["delta_full_sl_rate"].mean()),
            "mean_delta_timeout_rate": float(split_df["delta_timeout_rate"].mean()),
            "mean_delta_hit_rate": float(split_df["delta_hit_rate"].mean()),
            "guard_selected_splits": int((split_df["selected_label"] != "no_guard").sum()),
        }
    pd.DataFrame([summary]).to_csv(args.output_dir / "row_guard_walkforward_summary.csv", index=False)
    fixed_summary = pd.DataFrame()
    if not fixed_df.empty:
        fixed_summary = (
            fixed_df.groupby("fixed_label", as_index=False)
            .agg(
                splits=("split_id", "size"),
                sum_delta_net_pnl=("delta_net_pnl", "sum"),
                median_delta_net_pnl=("delta_net_pnl", "median"),
                positive_delta_net_pnl_share=("delta_net_pnl", lambda x: float((x > 0).mean())),
                sum_delta_objective_week=("delta_objective_week", "sum"),
                median_delta_objective_week=("delta_objective_week", "median"),
                positive_delta_objective_share=("delta_objective_week", lambda x: float((x > 0).mean())),
                sum_delta_worst_week_net_pnl=("delta_worst_week_net_pnl", "sum"),
                median_delta_worst_week_net_pnl=("delta_worst_week_net_pnl", "median"),
                mean_delta_full_sl_rate=("delta_full_sl_rate", "mean"),
                mean_delta_timeout_rate=("delta_timeout_rate", "mean"),
                mean_delta_hit_rate=("delta_hit_rate", "mean"),
                mean_holdout_veto_count=("holdout_veto_count", "mean"),
            )
            .sort_values(["sum_delta_objective_week", "sum_delta_net_pnl"], ascending=[False, False])
            .reset_index(drop=True)
        )
    fixed_summary.to_csv(args.output_dir / "row_guard_walkforward_fixed_rule_summary.csv", index=False)
    manifest = {
        "generated_by": "validate_wfrecent_row_guard_walkforward",
        "input_dir": str(args.input_dir),
        "first_holdout": args.first_holdout,
        "last_holdout_end": args.last_holdout_end,
        "top_train_replays": int(args.top_train_replays),
        "min_removed": int(args.min_removed),
        "recent_perf_focused": bool(args.recent_perf_focused),
        "q35_weight": float(args.q35_weight),
        "q20_weight": float(args.q20_weight),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n")

    lines = [
        "# wf_recent Row Guard Walk-Forward Validation",
        "",
        "Monthly chronological validation. Diagnostic risk percentiles, veto thresholds, EV curves, and rule selection use prior data only. This is not fresh untouched OOS, but it is stricter than the full-sample development ablation.",
        "",
        "## Summary",
        "",
        _fmt_table(pd.DataFrame([summary]) if summary else pd.DataFrame(), list(summary.keys()) if summary else []),
        "",
        "## Holdout Splits",
        "",
        _fmt_table(
            split_df,
            [
                "holdout_start",
                "holdout_end",
                "selected_label",
                "holdout_veto_count",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_worst_week_net_pnl",
                "baseline_net_pnl",
                "guard_net_pnl",
            ],
        ),
        "",
        "## Fixed Challenger Rule Summary",
        "",
        _fmt_table(
            fixed_summary,
            [
                "fixed_label",
                "splits",
                "sum_delta_net_pnl",
                "median_delta_net_pnl",
                "positive_delta_net_pnl_share",
                "sum_delta_objective_week",
                "median_delta_objective_week",
                "positive_delta_objective_share",
                "sum_delta_worst_week_net_pnl",
                "mean_delta_full_sl_rate",
                "mean_delta_timeout_rate",
                "mean_delta_hit_rate",
            ],
        ),
        "",
        "## Fixed Challenger Split Detail",
        "",
        _fmt_table(
            fixed_df.sort_values(["fixed_label", "holdout_start"]) if not fixed_df.empty else fixed_df,
            [
                "fixed_label",
                "holdout_start",
                "holdout_veto_count",
                "delta_net_pnl",
                "delta_objective_week",
                "delta_hit_rate",
                "delta_full_sl_rate",
                "delta_timeout_rate",
                "delta_worst_week_net_pnl",
            ],
            max_rows=80,
        ),
        "",
        "## Selected Train Rule Evidence",
        "",
        _fmt_table(
            split_df,
            [
                "holdout_start",
                "selected_label",
                "selected_train_delta_net_pnl_vs_train_baseline",
                "selected_train_delta_objective_week_vs_train_baseline",
                "selected_train_delta_full_sl_rate_vs_train_baseline",
                "selected_train_delta_worst_week_net_pnl_vs_train_baseline",
            ],
        ),
    ]
    (args.output_dir / "row_guard_walkforward_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
