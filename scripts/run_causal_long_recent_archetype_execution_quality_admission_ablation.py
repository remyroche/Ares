#!/usr/bin/env python3
"""Leakage-safe long-only recent-archetype admission ablation.

This runner consumes an already executable 1-minute per-trade replay.  It never
resimulates exits or costs: it builds the canonical portfolio candidate table
from the stored precomputed net/gross returns and varies only admission rank
and threshold nudges.  At every decision timestamp, recent evidence is
restricted to rows whose ``exit_timestamp`` is strictly earlier than that
decision timestamp.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    PortfolioPolicyParams,
    fit_hierarchical_ev_curves,
    load_portfolio_policy_params,
    replay_candidates,
)
from scripts.report_long_policy_replay_daily import to_portfolio_candidates  # noqa: E402


INPUT_DEFAULT = Path(
    "data_perp/reports/s59_h5_signalclose_causal_base_sharedstore_mda_hpo150_wf30_20260722_"
    "residual_only_hpo150_wf30_v1/simple_policy_long_only_oos_1m_v2_unconstrained_geometry/"
    "apr01_jul10_portfolio_holdout/per_trade_replay.parquet"
)
POLICY_DEFAULT = INPUT_DEFAULT.parent / "post_portfolio_replay/policy_fit/optimized_portfolio_policy_config.json"
OUTPUT_DEFAULT = INPUT_DEFAULT.parent / "causal_long_recent_archetype_execution_quality_admission_ablation_20260722"


@dataclass(frozen=True)
class Arm:
    label: str
    window_days: int | None = None
    mode: str = "none"
    magnitude: float = 0.0
    prior_support: float = 20.0


def _utc(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _arms() -> list[Arm]:
    arms = [Arm(label="no_adjustment")]
    for window_days in (7, 14, 21, 28):
        for mode in ("rank", "threshold", "rank_threshold"):
            for magnitude in (0.01, 0.02):
                arms.append(
                    Arm(
                        label=f"{mode}_{window_days}d_{int(magnitude * 100):02d}p",
                        window_days=window_days,
                        mode=mode,
                        magnitude=magnitude,
                    )
                )
    return arms


def _load_replay(path: Path) -> pd.DataFrame:
    rows = pd.read_parquet(path).copy()
    required = {
        "timestamp",
        "decision_timestamp",
        "exit_timestamp",
        "side_name",
        "policy_archetype",
        "net_return_notional",
        "rank_pct",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"replay is missing required fields: {missing}")
    for column in ("timestamp", "decision_timestamp", "exit_timestamp"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    rows["net_return_notional"] = pd.to_numeric(
        rows["net_return_notional"], errors="coerce"
    )
    rows["rank_pct"] = pd.to_numeric(rows["rank_pct"], errors="coerce")
    rows["side_name"] = rows["side_name"].astype(str).str.lower()
    rows["policy_archetype"] = rows["policy_archetype"].fillna("missing").astype(str)
    rows = rows.dropna(
        subset=["decision_timestamp", "exit_timestamp", "net_return_notional", "rank_pct"]
    ).copy()
    if not rows["side_name"].eq("long").all():
        raise ValueError("this ablation is intentionally limited to long replay rows")
    return rows.sort_values(
        ["decision_timestamp", "symbol", "policy_archetype"]
    ).reset_index(drop=True)


def add_causal_recent_quality(
    rows: pd.DataFrame,
    *,
    window_days: int,
    prior_support: float,
) -> pd.DataFrame:
    """Attach side x archetype estimates from strictly resolved prior exits.

    The calculation is deliberately performed one decision timestamp at a
    time.  All candidates at the same decision use exactly the same resolved
    history; outcomes with an equal exit timestamp remain unavailable.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if prior_support < 0:
        raise ValueError("prior_support must be non-negative")
    work = rows.copy()
    work["decision_timestamp"] = pd.to_datetime(work["decision_timestamp"], utc=True)
    work["exit_timestamp"] = pd.to_datetime(work["exit_timestamp"], utc=True)
    work["side_name"] = work["side_name"].astype(str).str.lower()
    work["policy_archetype"] = work["policy_archetype"].fillna("missing").astype(str)
    work["net_return_notional"] = pd.to_numeric(work["net_return_notional"], errors="coerce")
    if work[["decision_timestamp", "exit_timestamp", "net_return_notional"]].isna().any().any():
        raise ValueError("causal quality input contains invalid timestamps or net returns")

    result = pd.DataFrame(index=work.index)
    result["recent_window_days"] = int(window_days)
    result["recent_parent_support"] = 0
    result["recent_local_support"] = 0
    result["recent_parent_ev"] = 0.0
    result["recent_parent_hit_rate"] = 0.5
    result["recent_shrunk_ev"] = 0.0
    result["recent_shrunk_hit_rate"] = 0.5
    result["recent_ev_delta"] = 0.0
    result["recent_hit_rate_delta"] = 0.0
    result["recent_quality_score"] = 0.0
    result["recent_resolved_cutoff"] = pd.Series(
        pd.NaT, index=work.index, dtype="datetime64[ns, UTC]"
    )

    resolved = work.sort_values("exit_timestamp").copy()
    decision_groups = work.groupby("decision_timestamp", sort=True, observed=True).groups
    exit_times = resolved["exit_timestamp"].to_numpy()
    resolved_count = 0
    window = pd.Timedelta(days=int(window_days))
    for decision_ts, decision_index in decision_groups.items():
        # Strict inequality is the causal boundary.  The rows at the current
        # decision cannot affect each other even if one would exit immediately.
        while resolved_count < len(resolved) and exit_times[resolved_count] < decision_ts:
            resolved_count += 1
        history = resolved.iloc[:resolved_count]
        history = history.loc[history["exit_timestamp"].ge(decision_ts - window)]
        current = work.loc[decision_index]
        result.loc[decision_index, "recent_resolved_cutoff"] = decision_ts
        if history.empty:
            continue
        history = history.copy()
        history["hit"] = history["net_return_notional"].gt(0.0).astype(float)
        parents = history.groupby("side_name", observed=True).agg(
            support=("net_return_notional", "size"),
            ev=("net_return_notional", "mean"),
            hit=("hit", "mean"),
        )
        locals_ = history.groupby(["side_name", "policy_archetype"], observed=True).agg(
            support=("net_return_notional", "size"),
            ev=("net_return_notional", "mean"),
            hit=("hit", "mean"),
        )
        for side, idx in current.groupby("side_name", observed=True).groups.items():
            parent = parents.loc[side] if side in parents.index else None
            parent_support = int(parent["support"]) if parent is not None else 0
            parent_ev = float(parent["ev"]) if parent is not None else 0.0
            parent_hit = float(parent["hit"]) if parent is not None else 0.5
            archetypes = work.loc[idx, "policy_archetype"]
            for archetype, archetype_idx in archetypes.groupby(archetypes, observed=True).groups.items():
                key = (side, archetype)
                local = locals_.loc[key] if key in locals_.index else None
                local_support = int(local["support"]) if local is not None else 0
                local_ev = float(local["ev"]) if local is not None else parent_ev
                local_hit = float(local["hit"]) if local is not None else parent_hit
                denom = float(local_support) + float(prior_support)
                weight = float(local_support) / denom if denom > 0.0 else 0.0
                shrunk_ev = weight * local_ev + (1.0 - weight) * parent_ev
                shrunk_hit = weight * local_hit + (1.0 - weight) * parent_hit
                ev_delta = shrunk_ev - parent_ev
                hit_delta = shrunk_hit - parent_hit
                # 1% net return and 10 percentage-point hit-rate moves each
                # contribute one unit before clipping the admission quality.
                quality = float(np.clip(0.5 * (ev_delta / 0.01 + hit_delta / 0.10), -1.0, 1.0))
                result.loc[archetype_idx, "recent_parent_support"] = parent_support
                result.loc[archetype_idx, "recent_local_support"] = local_support
                result.loc[archetype_idx, "recent_parent_ev"] = parent_ev
                result.loc[archetype_idx, "recent_parent_hit_rate"] = parent_hit
                result.loc[archetype_idx, "recent_shrunk_ev"] = shrunk_ev
                result.loc[archetype_idx, "recent_shrunk_hit_rate"] = shrunk_hit
                result.loc[archetype_idx, "recent_ev_delta"] = ev_delta
                result.loc[archetype_idx, "recent_hit_rate_delta"] = hit_delta
                result.loc[archetype_idx, "recent_quality_score"] = quality
    return pd.concat([work, result], axis=1)


def apply_arm(candidates: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    """Apply a bounded admission-only nudge to canonical candidates."""
    out = candidates.copy()
    out["admission_arm"] = arm.label
    out["portfolio_rank_adjustment"] = 0.0
    out["admission_threshold_nudge"] = 0.0
    if arm.mode == "none":
        return out
    quality = pd.to_numeric(out["recent_quality_score"], errors="coerce").fillna(0.0)
    nudge = quality * float(arm.magnitude)
    if arm.mode in {"rank", "rank_threshold"}:
        out["portfolio_rank_adjustment"] = nudge.astype("float32")
    if arm.mode in {"threshold", "rank_threshold"}:
        # Positive quality loosens threshold, negative quality tightens it.
        out["admission_threshold_nudge"] = (-nudge).astype("float32")
        out["base_strategy_threshold"] = np.clip(
            pd.to_numeric(out["base_strategy_threshold"], errors="coerce").fillna(1.0)
            - nudge,
            0.0,
            1.0,
        ).astype("float32")
    return out


def to_candidates_with_recent_quality(
    rows: pd.DataFrame,
    *,
    base_strategy_threshold: float,
) -> pd.DataFrame:
    """Retain causal admission context across the canonical replay adapter."""
    candidates = to_portfolio_candidates(
        rows, base_strategy_threshold=base_strategy_threshold
    )
    metadata_columns = [
        "decision_timestamp",
        "symbol",
        "strategy_id",
        "recent_quality_score",
        "recent_local_support",
    ]
    available = [column for column in metadata_columns if column in rows.columns]
    if len(available) <= 3:
        candidates["recent_quality_score"] = 0.0
        candidates["recent_local_support"] = 0
        return candidates
    metadata = rows.loc[:, available].copy()
    metadata = metadata.rename(columns={"decision_timestamp": "timestamp"})
    metadata["timestamp"] = pd.to_datetime(metadata["timestamp"], utc=True)
    candidates = candidates.merge(
        metadata,
        on=["timestamp", "symbol", "strategy_id"],
        how="left",
        validate="one_to_one",
    )
    candidates["recent_quality_score"] = pd.to_numeric(
        candidates["recent_quality_score"], errors="coerce"
    ).fillna(0.0)
    candidates["recent_local_support"] = pd.to_numeric(
        candidates["recent_local_support"], errors="coerce"
    ).fillna(0).astype(int)
    return candidates


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"].astype(bool)].copy()
    if accepted.empty:
        return accepted
    accepted["candidate_index"] = pd.to_numeric(
        accepted["candidate_index"], errors="raise"
    ).astype(int)
    source = candidates.reset_index(drop=True).copy()
    source["candidate_index"] = source.index
    return accepted.merge(
        source[
            ["candidate_index", "policy_archetype", "recent_quality_score", "recent_local_support"]
        ],
        on="candidate_index",
        how="left",
        validate="one_to_one",
    )


def _summary(
    accepted: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    days = max(int((end.normalize() - start.normalize()).days), 1)
    if accepted.empty:
        empty = pd.DataFrame()
        return {
            "trade_count": 0,
            "trades_per_day": 0.0,
            "notional_net_ev_per_trade": np.nan,
            "bankroll_pnl": 0.0,
            "worst_week_bankroll_pnl": 0.0,
            "positive_weeks": 0,
            "full_stop_rate": np.nan,
            "timeout_rate": np.nan,
            "stability_score": -np.inf,
        }, empty, empty
    frame = accepted.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["position_size"] = pd.to_numeric(frame["position_size"], errors="coerce").fillna(0.0)
    frame["position_net_return"] = pd.to_numeric(frame["position_net_return"], errors="coerce").fillna(0.0)
    frame["bankroll_pnl"] = frame["position_size"] * frame["position_net_return"]
    frame["week_start"] = frame["timestamp"].dt.floor("D") - pd.to_timedelta(frame["timestamp"].dt.weekday, unit="D")
    weekly = frame.groupby("week_start", observed=True).agg(
        trades=("candidate_index", "size"),
        bankroll_pnl=("bankroll_pnl", "sum"),
        notional=("position_size", "sum"),
    ).reset_index()
    weekly["notional_net_ev_per_trade"] = weekly["bankroll_pnl"] / weekly["notional"].replace(0.0, np.nan)
    pnl = float(frame["bankroll_pnl"].sum())
    notional = float(frame["position_size"].sum())
    worst_week = float(weekly["bankroll_pnl"].min())
    week_std = float(weekly["bankroll_pnl"].std(ddof=0))
    stability = float(weekly["bankroll_pnl"].mean() - 0.5 * week_std + 0.25 * worst_week)
    per_arch = frame.groupby("policy_archetype", observed=True).agg(
        trades=("candidate_index", "size"),
        notional=("position_size", "sum"),
        bankroll_pnl=("bankroll_pnl", "sum"),
        positive_rate=("position_net_return", lambda value: value.gt(0.0).mean()),
        full_stop_rate=("position_exit_reason", lambda value: value.astype(str).eq("full_sl").mean()),
        timeout_rate=("position_exit_reason", lambda value: value.astype(str).eq("timeout").mean()),
        mean_recent_quality=("recent_quality_score", "mean"),
        mean_recent_local_support=("recent_local_support", "mean"),
    ).reset_index()
    per_arch["notional_net_ev_per_trade"] = per_arch["bankroll_pnl"] / per_arch["notional"].replace(0.0, np.nan)
    return {
        "trade_count": int(len(frame)),
        "trades_per_day": float(len(frame) / days),
        "notional_net_ev_per_trade": float(pnl / notional) if notional else np.nan,
        "bankroll_pnl": pnl,
        "worst_week_bankroll_pnl": worst_week,
        "positive_weeks": int(weekly["bankroll_pnl"].gt(0.0).sum()),
        "full_stop_rate": float(frame["position_exit_reason"].astype(str).eq("full_sl").mean()),
        "timeout_rate": float(frame["position_exit_reason"].astype(str).eq("timeout").mean()),
        "stability_score": stability,
    }, weekly, per_arch


def _replay_period(
    *,
    candidates: pd.DataFrame,
    ev_curve_candidates: pd.DataFrame,
    params: PortfolioPolicyParams,
    arm: Arm,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    period = candidates.loc[
        candidates["timestamp"].ge(start) & candidates["timestamp"].lt(end)
    ].copy()
    if period.empty:
        raise ValueError(f"no candidates in evaluation period {start} to {end}")
    adjusted = apply_arm(period, arm)
    ev_curve = fit_hierarchical_ev_curves(ev_curve_candidates)
    decisions, _equity, _metrics = replay_candidates(
        adjusted,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode="perps",
    )
    accepted = _accepted_trades(adjusted, decisions)
    summary, weekly, per_arch = _summary(accepted, start=start, end=end)
    return summary, decisions, weekly, per_arch


def _delta_row(arm: Arm, period: str, baseline: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    row = {"arm": arm.label, "period": period, **asdict(arm)}
    for key, value in challenger.items():
        row[f"challenger_{key}"] = value
        row[f"baseline_{key}"] = baseline.get(key)
        if isinstance(value, (float, int, np.floating, np.integer)):
            base = baseline.get(key)
            row[f"delta_{key}"] = float(value - base) if base is not None and np.isfinite(base) else np.nan
    return row


def _fmt(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    return frame.loc[:, available].to_markdown(index=False) if available else "No rows."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--policy-config", type=Path, default=POLICY_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--fit-start", default="2026-04-01T00:00:00+00:00")
    parser.add_argument("--validation-start", default="2026-05-01T00:00:00+00:00")
    parser.add_argument("--july-start", default="2026-07-01T00:00:00+00:00")
    parser.add_argument("--end", default="2026-07-11T00:00:00+00:00")
    args = parser.parse_args()

    fit_start = _utc(args.fit_start)
    validation_start = _utc(args.validation_start)
    july_start = _utc(args.july_start)
    end = _utc(args.end)
    if not (fit_start < validation_start < july_start < end):
        raise ValueError("require fit_start < validation_start < july_start < end")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_replay(args.input)
    for label, start, period_end in (
        ("fit", fit_start, validation_start),
        ("validation", validation_start, july_start),
        ("july_holdout", july_start, end),
    ):
        if not rows["decision_timestamp"].between(start, period_end, inclusive="left").any():
            raise ValueError(f"input replay has no executable decisions in {label}")
    params = load_portfolio_policy_params(args.policy_config)
    base_candidates = to_candidates_with_recent_quality(
        rows,
        base_strategy_threshold=float(params.global_threshold_floor),
    )
    base_candidates = base_candidates.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)

    # Metrics for May and June use an EV curve fitted before their respective
    # start.  July uses only the completed April-June fit window.  The selector
    # consumes May-June validation results after all of June has resolved.
    period_specs = [
        # Disclosed in-sample fit diagnostic, not validation or promotion
        # evidence. It completes the requested Apr-Jun report coverage.
        ("april_fit_in_sample", fit_start, validation_start, fit_start, validation_start),
        ("may_validation", validation_start, _utc("2026-06-01T00:00:00+00:00"), fit_start, validation_start),
        ("june_validation", _utc("2026-06-01T00:00:00+00:00"), july_start, fit_start, _utc("2026-06-01T00:00:00+00:00")),
        ("july_holdout", july_start, end, fit_start, july_start),
    ]
    all_rows: list[dict[str, Any]] = []
    all_weekly: list[pd.DataFrame] = []
    all_arch: list[pd.DataFrame] = []
    all_decisions: list[pd.DataFrame] = []
    baseline_by_period: dict[str, dict[str, Any]] = {}
    quality_candidates_by_window: dict[int, pd.DataFrame] = {}
    for arm in _arms():
        if arm.mode == "none":
            quality_candidates = base_candidates.copy()
        else:
            window_days = int(arm.window_days or 7)
            if window_days not in quality_candidates_by_window:
                quality_rows = add_causal_recent_quality(
                    rows,
                    window_days=window_days,
                    prior_support=float(arm.prior_support),
                )
                quality_candidates_by_window[window_days] = (
                    to_candidates_with_recent_quality(
                        quality_rows,
                        base_strategy_threshold=float(params.global_threshold_floor),
                    )
                    .sort_values(["timestamp", "strategy_id", "symbol"])
                    .reset_index(drop=True)
                )
            quality_candidates = quality_candidates_by_window[window_days].copy()
        # Baseline has neutral causal fields so all reporting schemas match.
        for column, default in {
            "recent_quality_score": 0.0,
            "recent_local_support": 0,
        }.items():
            if column not in quality_candidates:
                quality_candidates[column] = default
        for period, start, period_end, curve_start, curve_end in period_specs:
            curve_rows = quality_candidates.loc[
                quality_candidates["timestamp"].ge(curve_start)
                & quality_candidates["timestamp"].lt(curve_end)
            ].copy()
            summary, decisions, weekly, per_arch = _replay_period(
                candidates=quality_candidates,
                ev_curve_candidates=curve_rows,
                params=params,
                arm=arm,
                start=start,
                end=period_end,
            )
            if arm.mode == "none":
                baseline_by_period[period] = summary
            row = _delta_row(arm, period, baseline_by_period.get(period, summary), summary)
            all_rows.append(row)
            weekly["arm"] = arm.label
            weekly["period"] = period
            all_weekly.append(weekly)
            per_arch["arm"] = arm.label
            per_arch["period"] = period
            all_arch.append(per_arch)
            decisions["arm"] = arm.label
            decisions["period"] = period
            all_decisions.append(decisions)

    summary = pd.DataFrame(all_rows)
    validation = summary.loc[summary["period"].isin(["may_validation", "june_validation"])].copy()
    stable = validation.groupby("arm", observed=True).agg(
        pre_july_delta_bankroll_pnl=("delta_bankroll_pnl", "sum"),
        pre_july_delta_stability_score=("delta_stability_score", "sum"),
        pre_july_worst_validation_week=("delta_worst_week_bankroll_pnl", "min"),
        validation_periods=("period", "size"),
    ).reset_index()
    july = summary.loc[summary["period"].eq("july_holdout")].copy()
    selection = stable.merge(
        july[["arm", "delta_bankroll_pnl", "delta_stability_score", "delta_worst_week_bankroll_pnl"]],
        on="arm",
        how="left",
        suffixes=("", "_july"),
        validate="one_to_one",
    )
    selection["pre_july_stable"] = (
        selection["pre_july_delta_stability_score"].gt(0.0)
        & selection["pre_july_delta_bankroll_pnl"].gt(0.0)
        & selection["pre_july_worst_validation_week"].ge(0.0)
    )
    selection["july_improved"] = (
        selection["delta_stability_score"].gt(0.0)
        & selection["delta_bankroll_pnl"].gt(0.0)
    )
    selection["promotion_pass"] = (
        selection["arm"].ne("no_adjustment")
        & selection["pre_july_stable"]
        & selection["july_improved"]
    )
    selection = selection.sort_values(
        ["promotion_pass", "pre_july_delta_stability_score", "delta_stability_score", "delta_bankroll_pnl"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    weekly = pd.concat(all_weekly, ignore_index=True) if all_weekly else pd.DataFrame()
    per_arch = pd.concat(all_arch, ignore_index=True) if all_arch else pd.DataFrame()
    decisions = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    summary.to_csv(args.output_dir / "arm_period_summary.csv", index=False)
    selection.to_csv(args.output_dir / "selection_and_july_holdout.csv", index=False)
    weekly.to_csv(args.output_dir / "weekly_metrics.csv", index=False)
    per_arch.to_csv(args.output_dir / "per_archetype_metrics.csv", index=False)
    decisions.to_parquet(args.output_dir / "per_candidate_decisions.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal_long_recent_archetype_execution_quality_admission_ablation_v1",
        "input": str(args.input),
        "policy_config": str(args.policy_config),
        "fit_window": [fit_start.isoformat(), validation_start.isoformat()],
        "fit_metrics_are_in_sample": True,
        "validation_windows": [
            [validation_start.isoformat(), july_start.isoformat()],
        ],
        "july_holdout": [july_start.isoformat(), end.isoformat()],
        "arms": [asdict(arm) for arm in _arms()],
        "portfolio_cost_contract": "stored net_return_notional/gross_return_notional are mapped unchanged into canonical replay; no new fee, spread, or slippage deduction is applied",
        "causal_contract": "at decision t, side x policy_archetype recent EV/hit estimates use only replay rows with exit_timestamp strictly earlier than t; local estimates shrink to long parent with support prior 20",
        "selection_contract": "April metrics are in-sample fit diagnostics only; May and June validation choose the arm after June resolution; July 1-10 is not used for selection",
        "promotion_pass_count": int(selection["promotion_pass"].sum()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = [
        "# Causal Long Recent-Archetype Execution-Quality Admission Ablation",
        "",
        "The input is the corrected executable 1-minute replay. This ablation changes only canonical portfolio admission rank/threshold inputs; stored net returns, fee reconciliation, spreads, exits, and portfolio constraints are replayed unchanged.",
        "",
        "## Temporal Contract",
        "",
        f"- April fit/warm-up: `{fit_start.isoformat()}` to `{validation_start.isoformat()}`.",
        f"- April is in-sample fit-only. Pre-July validation and selection: `{validation_start.isoformat()}` to `{july_start.isoformat()}`.",
        f"- Untouched July holdout: `{july_start.isoformat()}` to `{end.isoformat()}`.",
        "- At each decision, each side x policy-archetype EV/hit-rate estimate only includes exits strictly earlier than the decision timestamp. It is shrunk to the long parent with a 20-trade empirical-Bayes prior.",
        "",
        "## Arm Results",
        "",
        _fmt(summary.sort_values(["period", "delta_stability_score"], ascending=[True, False]), [
            "period", "arm", "challenger_trade_count", "challenger_trades_per_day", "challenger_notional_net_ev_per_trade", "challenger_bankroll_pnl", "challenger_worst_week_bankroll_pnl", "challenger_positive_weeks", "challenger_full_stop_rate", "challenger_timeout_rate", "delta_bankroll_pnl", "delta_stability_score",
        ]),
        "",
        "## Selection And July Gate",
        "",
        _fmt(selection, [
            "arm", "pre_july_delta_bankroll_pnl", "pre_july_delta_stability_score", "pre_july_worst_validation_week", "delta_bankroll_pnl", "delta_stability_score", "delta_worst_week_bankroll_pnl", "pre_july_stable", "july_improved", "promotion_pass",
        ]),
        "",
        "## Per Archetype",
        "",
        _fmt(per_arch.loc[per_arch["period"].eq("july_holdout")].sort_values(["arm", "policy_archetype"]), [
            "arm", "policy_archetype", "trades", "notional_net_ev_per_trade", "bankroll_pnl", "positive_rate", "full_stop_rate", "timeout_rate", "mean_recent_quality", "mean_recent_local_support",
        ]),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(selection.head(12).to_string(index=False))
    if not bool(selection["promotion_pass"].any()):
        print("FAIL: no non-baseline arm improved both pre-July stability and untouched July holdout.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
