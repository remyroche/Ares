#!/usr/bin/env python3
"""Matched-admission BCF/current-MC1 sizing ablation (offline only).

This is the deliberately narrow follow-up to the 2025-optimised position-size
study.  It holds the canonical dual-MC1 admission population fixed:

    BCF MC1 >= 30 bps AND current-v5 MC1 >= 30 bps

and holds the auction priority fixed to BCF MC1 EV.  Thus no arm can create or
remove an admission based on the sizing score.  It may only change how much
capital an already admitted candidate requests; normal live-equivalent
concurrency, symbol, two-new-per-hour and 80%-gross-capacity constraints still
apply, so actual fills can differ only as a capacity consequence.

All experiments are offline research.  They consume the frozen source-aligned
parent-policy outcome ledger and cannot write exchange state or alter the live
stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from run_strict_r3_dual_mc1_position_sizing_ablations import (  # frozen shared substrate
    DEFAULT_BCF,
    DEFAULT_CURRENT,
    POLICY_COLUMNS,
    _read_panel,
    _sha,
    _utc,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_matched_admission_sizing_2025dev_2026val_20260822_v3"

ADMISSION_FLOOR_BPS = 30.0
FIXED_WEIGHT = 0.10
MIN_CAPACITY_CLIPPED_WEIGHT = 0.05  # Existing canonical auction behaviour.
MIN_GROSS_WEIGHT = 0.80
MAX_NEW_PER_HOUR = 2
MAX_CONCURRENT = 8
INITIAL_NAV = 1_000.0
MIN_P90_ROWS = 1_000
DISAGREEMENT_PENALTY = 0.10  # Frozen prior-development A5 coefficient; no 2026 tuning.

ScoreSource = Literal["bcf", "current", "blend70", "blend50_disagreement"]
SizingMode = Literal["fixed", "range", "zero_sum"]


@dataclass(frozen=True)
class ArmSpec:
    name: str
    stage: str
    source: ScoreSource
    mode: SizingMode
    minimum_weight: float
    maximum_weight: float
    description: str


def _score(panel: pd.DataFrame, source: ScoreSource) -> np.ndarray:
    bcf = panel["ev1_bcf_bps"].to_numpy(dtype=float)
    current = panel["ev2_current_bps"].to_numpy(dtype=float)
    if source == "bcf":
        return bcf
    if source == "current":
        return current
    if source == "blend70":
        return 0.70 * bcf + 0.30 * current
    if source == "blend50_disagreement":
        return 0.50 * bcf + 0.50 * current - DISAGREEMENT_PENALTY * np.abs(bcf - current)
    raise ValueError(f"unsupported score source {source}")


def _causal_month_p90(panel: pd.DataFrame, values: np.ndarray, *, freeze_after: pd.Timestamp) -> dict[str, float]:
    """Prequential p90 score scale, frozen after the development window.

    P90 is only a sizing normalizer.  It is fit from previous prediction rows,
    never outcomes, and no score distribution from 2026 can alter a 2026 size.
    """
    work = pd.DataFrame({"timestamp": panel["timestamp"].to_numpy(), "value": values})
    work["month_start"] = work["timestamp"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    result: dict[str, float] = {}
    for month in sorted(work["month_start"].unique()):
        cutoff = min(pd.Timestamp(month), freeze_after)
        prior = work.loc[work["timestamp"].lt(cutoff), "value"].to_numpy(dtype=float)
        prior = prior[np.isfinite(prior)]
        key = pd.Timestamp(month).strftime("%Y-%m")
        result[key] = float(np.quantile(prior, 0.90)) if len(prior) >= MIN_P90_ROWS else float("nan")
    return result


def _range_weight(value: float, p90: float, *, low: float, high: float) -> float:
    """Linear bounded sizing score; no score can reject an admitted candidate."""
    if not np.isfinite(value) or not np.isfinite(p90):
        return low
    x = float(np.clip((value - ADMISSION_FLOOR_BPS) / max(p90 - ADMISSION_FLOOR_BPS, 1e-9), 0.0, 1.0))
    return low + (high - low) * x


def _zero_sum_weights(values: np.ndarray, *, delta: float) -> np.ndarray:
    """Return timestamp-local weights centred exactly on the fixed 10% size."""
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([FIXED_WEIGHT], dtype=float)
    # Stable order makes tied scores deterministic while retaining exact
    # rank-centering.  Score values are contemporaneous prediction outputs.
    order = np.argsort(values, kind="stable")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    z = 2.0 * rank / float(n - 1) - 1.0
    weights = FIXED_WEIGHT + delta * z
    if not np.isclose(float(weights.mean()), FIXED_WEIGHT, atol=1e-12):
        raise AssertionError("zero-sum sizing failed to preserve mean 10% weight")
    return weights


def _arm_specs() -> tuple[ArmSpec, ...]:
    sources: tuple[tuple[ScoreSource, str], ...] = (
        ("bcf", "BCF MC1 EV"),
        ("current", "current-v5 MC1 EV"),
        ("blend70", "70/30 BCF/current EV"),
        ("blend50_disagreement", "50/50 EV less 0.10× disagreement"),
    )
    arms: list[ArmSpec] = [
        ArmSpec("B0_fixed10", "B0", "bcf", "fixed", .10, .10,
                "Canonical dual admission; BCF auction; every entry requests 10%"),
    ]
    for source, label in sources:
        arms.append(ArmSpec(f"B1_range5_15_{source}", "B1", source, "range", .05, .15,
                            f"Matched admission; linear 5–15% sizing from {label}"))
    for source, label in sources:
        arms.append(ArmSpec(f"B2_zero_sum5_15_{source}", "B2", source, "zero_sum", .05, .15,
                            f"Matched admission; timestamp-centred 5–15% zero-sum tilt from {label}"))
    for source, label in sources:
        arms.append(ArmSpec(f"B3_zero_sum7p5_12p5_{source}", "B3", source, "zero_sum", .075, .125,
                            f"Matched admission; timestamp-centred 7.5–12.5% zero-sum tilt from {label}"))
    for source, label in sources:
        arms.append(ArmSpec(f"B4_derisk5_10_{source}", "B4", source, "range", .05, .10,
                            f"Matched admission; EV may only de-risk to 5% using {label}"))
    for source, label in sources:
        arms.append(ArmSpec(f"B5_upsize10_15_{source}", "B5", source, "range", .10, .15,
                            f"Matched admission; EV may only up-size to 15% using {label}"))
    return tuple(arms)


def _close_positions(
    positions: list[dict[str, object]], *, timestamp: pd.Timestamp, cash: float,
    exits: list[dict[str, object]],
) -> tuple[list[dict[str, object]], float]:
    remaining: list[dict[str, object]] = []
    for position in positions:
        if pd.Timestamp(position["exit_ts"]) <= timestamp:
            amount = float(position["amount"])
            pnl = amount * float(position["policy_net_bps"]) / 10_000.0
            cash += amount + pnl
            exits.append({
                "candidate_id": position["candidate_id"],
                "exit_ts": position["exit_ts"],
                "realized_pnl": pnl,
                "policy_net_bps": position["policy_net_bps"],
            })
        else:
            remaining.append(position)
    return remaining, cash


def _mark_positions(positions: list[dict[str, object]], marks: dict[str, float], *, stats: dict[str, int]) -> None:
    for position in positions:
        stats["attempts"] += 1
        mark = marks.get(str(position["symbol"]))
        if mark is None or not np.isfinite(mark) or mark <= 0.0:
            stats["fallbacks"] += 1
            continue
        position["market_value"] = float(position["quantity"]) * float(mark)
        position["last_mark_price"] = float(mark)
        stats["updates"] += 1


def _metrics(
    trade: pd.DataFrame, state: pd.DataFrame, *, arm: str, start: pd.Timestamp,
    end: pd.Timestamp, initial_nav: float, final_nav: float,
) -> dict[str, object]:
    days = pd.date_range(start.normalize(), (end - pd.Timedelta(seconds=1)).normalize(), freq="1D", tz="UTC")
    marked = state.sort_values("timestamp").groupby("timestamp", sort=True)["nav"].last() if not state.empty else pd.Series(dtype=float)
    daily_nav = marked.groupby(marked.index.normalize()).last().reindex(days).ffill().fillna(initial_nav) if not marked.empty else pd.Series(initial_nav, index=days)
    daily_pnl = daily_nav.diff().fillna(daily_nav.iloc[0] - initial_nav)
    daily_return = daily_pnl / daily_nav.shift(1, fill_value=initial_nav).replace(0.0, np.nan)
    downside = daily_return.loc[daily_return.lt(0.0)]
    sortino = float(daily_return.mean() / downside.std(ddof=0) * math.sqrt(365.0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else float("nan")
    if marked.empty:
        max_dd = 0.0
    else:
        max_dd = float((marked / marked.cummax() - 1.0).min())
    entry_daily = trade.groupby(trade["timestamp"].dt.normalize()).size().reindex(days, fill_value=0) if not trade.empty else pd.Series(0, index=days)
    total_amount = float(trade["amount"].sum()) if not trade.empty else 0.0
    # This is intentionally distinct from portfolio NAV change.  A reporting
    # boundary can inherit positions opened earlier; their exits are part of
    # live portfolio PnL but must not be attributed to the new period's entry
    # selections when judging a sizing signal.
    selected_trade_net_pnl = float((trade["amount"] * trade["policy_net_bps"] / 10_000.0).sum()) if not trade.empty else 0.0
    net_pnl = float(final_nav - initial_nav)
    util = state["utilization"] if not state.empty else pd.Series(dtype=float)
    gross = state["gross_weight"] if not state.empty else pd.Series(dtype=float)
    month_pnl = daily_pnl.groupby(daily_pnl.index.to_period("M")).sum()
    week_pnl = daily_pnl.groupby(daily_pnl.index.to_period("W-MON")).sum()
    return {
        "arm": arm, "start": start, "end_exclusive": end,
        "trades": int(len(trade)), "trades_per_day": float(len(trade) / max(len(days), 1)),
        "initial_nav": float(initial_nav), "final_nav": float(final_nav), "net_pnl": net_pnl,
        "net_portfolio_return": float(final_nav / initial_nav - 1.0),
        "capital_deployed": total_amount, "selected_trade_net_pnl": selected_trade_net_pnl,
        "portfolio_nav_change_bps_per_selected_capital": float(1e4 * net_pnl / total_amount) if total_amount else float("nan"),
        "net_ev_bps_per_trade": float(trade["policy_net_bps"].mean()) if not trade.empty else float("nan"),
        "weighted_net_ev_bps": float(1e4 * selected_trade_net_pnl / total_amount) if total_amount else float("nan"),
        "avg_target_size": float(trade["target_weight"].mean()) if not trade.empty else float("nan"),
        "avg_actual_size": float(trade["actual_weight"].mean()) if not trade.empty else float("nan"),
        "p05_actual_size": float(trade["actual_weight"].quantile(.05)) if not trade.empty else float("nan"),
        "p50_actual_size": float(trade["actual_weight"].quantile(.50)) if not trade.empty else float("nan"),
        "p95_actual_size": float(trade["actual_weight"].quantile(.95)) if not trade.empty else float("nan"),
        "avg_gross_utilization": float(util.mean()) if len(util) else 0.0,
        "median_gross_utilization": float(util.median()) if len(util) else 0.0,
        "p90_gross_utilization": float(util.quantile(.90)) if len(util) else 0.0,
        "avg_gross_weight": float(gross.mean()) if len(gross) else 0.0,
        "max_drawdown": max_dd,
        "worst_day_pnl": float(daily_pnl.min()) if len(daily_pnl) else 0.0,
        "worst_week_pnl": float(week_pnl.min()) if len(week_pnl) else 0.0,
        "worst_month_pnl": float(month_pnl.min()) if len(month_pnl) else 0.0,
        "positive_months": int(month_pnl.gt(0.0).sum()), "months": int(len(month_pnl)),
        "sortino_daily": sortino,
        "calmar_like": float((final_nav / initial_nav - 1.0) / max(abs(max_dd), .01)),
        "days_lt_1_trade": int(entry_daily.lt(1).sum()), "days_lt_5_trades": int(entry_daily.lt(5).sum()),
    }


def _simulate(
    panel: pd.DataFrame, arm: ArmSpec, *, start: pd.Timestamp, end: pd.Timestamp,
    freeze_after: pd.Timestamp, capture: bool,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay one arm.  Candidate admission and BCF priority are invariant."""
    score = _score(panel, arm.source)
    work = panel.copy()
    work["sizing_score_bps"] = score
    work["month"] = work["timestamp"].dt.to_period("M").astype(str)
    work["p90_bps"] = work["month"].map(_causal_month_p90(work, score, freeze_after=freeze_after))
    # This is the one, frozen admitted population for every arm.  Score only
    # affects a requested size.  It never appears in an admission condition.
    eligible = work.loc[(work["ev1_bcf_bps"] >= ADMISSION_FLOOR_BPS) & (work["ev2_current_bps"] >= ADMISSION_FLOOR_BPS)].copy()
    all_current = work.loc[work["timestamp"].lt(end)].copy()
    eligible = eligible.loc[eligible["timestamp"].lt(end)].copy()
    if eligible.empty:
        raise ValueError("no dual-MC1 admitted rows")
    candidate_by_time = {
        timestamp: group.sort_values(["ev1_bcf_bps", "candidate_id"], ascending=[False, True], kind="stable")
        for timestamp, group in eligible.groupby("timestamp", sort=True)
    }
    mark_by_time = {
        timestamp: dict(zip(group["symbol"].astype(str), group["policy_entry_price"].astype(float)))
        for timestamp, group in all_current.groupby("timestamp", sort=True)
    }
    timestamps = pd.DatetimeIndex(all_current["timestamp"].drop_duplicates().sort_values())
    cash = INITIAL_NAV
    positions: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    exits: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    rejects: dict[str, int] = {"max_new_per_hour": 0, "max_concurrent": 0, "symbol_already_open": 0, "max_gross_capacity": 0}
    mark_stats = {"attempts": 0, "updates": 0, "fallbacks": 0}
    starting_nav: float | None = None
    for timestamp in timestamps:
        positions, cash = _close_positions(positions, timestamp=timestamp, cash=cash, exits=exits)
        _mark_positions(positions, mark_by_time.get(timestamp, {}), stats=mark_stats)
        nav = cash + float(sum(float(item["market_value"]) for item in positions))
        in_window = timestamp >= start
        if in_window and starting_nav is None:
            starting_nav = nav
        gross = float(sum(float(item["market_value"]) for item in positions))
        if in_window:
            states.append({"timestamp": timestamp, "nav": nav, "gross_amount": gross,
                           "gross_weight": gross / max(nav, 1e-12),
                           "utilization": gross / max(MIN_GROSS_WEIGHT * nav, 1e-12),
                           "open_positions": len(positions)})
        group = candidate_by_time.get(timestamp)
        if group is None:
            continue
        # February 2025 is the shared score-scale warm-up.  Skipping it for
        # every arm keeps the prequential normalisation contract identical;
        # it is not an EV-based admission decision.
        if not np.isfinite(group["p90_bps"].to_numpy(dtype=float)).all():
            continue
        scores = group["sizing_score_bps"].to_numpy(dtype=float)
        if arm.mode == "fixed":
            requested = np.full(len(group), FIXED_WEIGHT, dtype=float)
        elif arm.mode == "range":
            requested = np.asarray([
                _range_weight(float(value), float(p90), low=arm.minimum_weight, high=arm.maximum_weight)
                for value, p90 in zip(scores, group["p90_bps"].to_numpy(dtype=float), strict=True)
            ])
        elif arm.mode == "zero_sum":
            requested = _zero_sum_weights(scores, delta=(arm.maximum_weight - arm.minimum_weight) / 2.0)
        else:
            raise ValueError(f"unsupported mode {arm.mode}")
        entries = 0
        open_symbols = {str(item["symbol"]) for item in positions}
        for row, target_weight in zip(group.itertuples(index=False), requested, strict=True):
            # BCF sorting is already fixed in group.  All rejections below are
            # common portfolio constraints, not EV-score gates.
            if entries >= MAX_NEW_PER_HOUR:
                rejects["max_new_per_hour"] += 1
                continue
            if len(positions) >= MAX_CONCURRENT:
                rejects["max_concurrent"] += 1
                continue
            if str(row.symbol) in open_symbols:
                rejects["symbol_already_open"] += 1
                continue
            gross = float(sum(float(item["market_value"]) for item in positions))
            capacity = max(MIN_GROSS_WEIGHT * nav - gross, 0.0)
            # The canonical auction allows a final entry clipped by residual
            # capacity down to 5% of NAV.  This floor is deliberately common
            # to every arm; using each arm's requested lower bound instead
            # would silently change the B0 control's portfolio path.
            minimum_amount = MIN_CAPACITY_CLIPPED_WEIGHT * nav
            if capacity + 1e-12 < minimum_amount:
                rejects["max_gross_capacity"] += 1
                continue
            amount = min(float(target_weight) * nav, capacity)
            actual_weight = amount / max(nav, 1e-12)
            position = {
                "candidate_id": str(row.candidate_id), "symbol": str(row.symbol), "entry_ts": timestamp,
                "exit_ts": row.exit_ts, "amount": amount,
                "quantity": amount / float(row.policy_entry_price), "market_value": amount,
                "last_mark_price": float(row.policy_entry_price), "policy_net_bps": float(row.policy_net_bps),
            }
            positions.append(position)
            cash -= amount
            entries += 1
            open_symbols.add(str(row.symbol))
            if in_window:
                decisions.append({
                    "arm": arm.name, "stage": arm.stage, "candidate_id": str(row.candidate_id), "timestamp": timestamp,
                    "symbol": str(row.symbol), "exit_ts": row.exit_ts,
                    "ev1_bcf_bps": float(row.ev1_bcf_bps), "ev2_current_bps": float(row.ev2_current_bps),
                    "sizing_score_bps": float(row.sizing_score_bps),
                    "disagreement_bps": abs(float(row.ev1_bcf_bps) - float(row.ev2_current_bps)),
                    "target_weight": float(target_weight), "actual_weight": actual_weight, "amount": amount,
                    "policy_net_bps": float(row.policy_net_bps), "policy_gross_bps": float(row.policy_gross_bps),
                    "holding_hours": (pd.Timestamp(row.exit_ts) - timestamp).total_seconds() / 3600.0,
                })
    positions, cash = _close_positions(positions, timestamp=end, cash=cash, exits=exits)
    final_nav = cash + float(sum(float(item["market_value"]) for item in positions))
    if starting_nav is None:
        raise ValueError("reporting window has no decision timestamps")
    states.append({"timestamp": end, "nav": final_nav,
                   "gross_amount": float(sum(float(item["market_value"]) for item in positions)),
                   "gross_weight": float(sum(float(item["market_value"]) for item in positions)) / max(final_nav, 1e-12),
                   "utilization": float(sum(float(item["market_value"]) for item in positions)) / max(MIN_GROSS_WEIGHT * final_nav, 1e-12),
                   "open_positions": len(positions)})
    trade = pd.DataFrame(decisions)
    state = pd.DataFrame(states)
    exit_frame = pd.DataFrame(exits)
    metric = _metrics(trade, state, arm=arm.name, start=start, end=end, initial_nav=starting_nav, final_nav=final_nav)
    metric.update({"stage": arm.stage, "source": arm.source, "mode": arm.mode,
                   "minimum_weight": arm.minimum_weight, "maximum_weight": arm.maximum_weight,
                   "admission_population": "BCF>=30 AND current-v5>=30; BCF priority fixed",
                   "unresolved_open_positions_at_end": len(positions),
                   "mark_updates": mark_stats["updates"], "mark_fallbacks": mark_stats["fallbacks"],
                   "mark_coverage": mark_stats["updates"] / mark_stats["attempts"] if mark_stats["attempts"] else 1.0,
                   **{f"portfolio_rejected_{reason}": count for reason, count in rejects.items()}})
    return metric, trade if capture else pd.DataFrame(), state if capture else pd.DataFrame(), exit_frame if capture else pd.DataFrame()


def _monthly(trade: pd.DataFrame, *, arm: str, scope: str) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame(columns=["arm", "scope", "month", "trades", "net_ev_bps", "weighted_net_ev_bps"])
    work = trade.copy()
    work["month"] = work["timestamp"].dt.to_period("M").astype(str)
    work["pnl"] = work["amount"] * work["policy_net_bps"] / 10_000.0
    out = work.groupby("month", sort=True).agg(trades=("candidate_id", "size"), capital=("amount", "sum"),
                                                  net_pnl=("pnl", "sum"), net_ev_bps=("policy_net_bps", "mean"),
                                                  avg_size=("actual_weight", "mean"))
    out["weighted_net_ev_bps"] = 1e4 * out["net_pnl"] / out["capital"]
    out = out.reset_index()
    out.insert(0, "scope", scope); out.insert(0, "arm", arm)
    return out


def _scope_metrics_from_full(
    *, arm: ArmSpec, full_metric: dict[str, object], full_trade: pd.DataFrame,
    full_state: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Derive a reporting scope from one continuous causal portfolio path.

    Replaying an arm once through July is essential: January 2026 must inherit
    the real cash and positions left by 2025.  Replaying each report scope from
    a fresh book would change the capacity question this study is designed to
    isolate.
    """
    trade = full_trade.loc[full_trade["timestamp"].ge(start) & full_trade["timestamp"].lt(end)].copy()
    state = full_state.loc[full_state["timestamp"].ge(start) & full_state["timestamp"].le(end)].copy()
    if state.empty:
        raise ValueError(f"no state coverage for {start}..{end}")
    initial_nav = float(state.sort_values("timestamp").iloc[0]["nav"])
    final_nav = float(state.sort_values("timestamp").iloc[-1]["nav"])
    metric = _metrics(trade, state, arm=arm.name, start=start, end=end, initial_nav=initial_nav, final_nav=final_nav)
    metric.update({
        "stage": arm.stage, "source": arm.source, "mode": arm.mode,
        "minimum_weight": arm.minimum_weight, "maximum_weight": arm.maximum_weight,
        "admission_population": "BCF>=30 AND current-v5>=30; BCF priority fixed",
        # The path-wide counters are intentionally labelled as such rather
        # than misleadingly presenting them as independently restarted scope
        # counts.  Candidate admission itself remains identical per hour.
        "mark_coverage_path": full_metric["mark_coverage"],
    })
    return metric, trade, state


def _decile_rows(frame: pd.DataFrame, *, source: ScoreSource, population: str, scope: str, weights: pd.Series | None = None) -> pd.DataFrame:
    """Outcome-only EV-signal diagnostic; deciles never drive an allocation."""
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    score = _score(work, source)
    rank = pd.Series(score, index=work.index).rank(method="first", pct=True)
    work["decile"] = np.minimum(np.ceil(rank * 10.0).astype(int), 10)
    work["realized_net_bps"] = pd.to_numeric(work["policy_net_bps"], errors="coerce")
    if weights is None:
        work["diagnostic_weight"] = 1.0
    else:
        work["diagnostic_weight"] = pd.to_numeric(weights.reindex(work.index), errors="coerce").fillna(0.0)
    rows: list[dict[str, object]] = []
    for decile, group in work.groupby("decile", sort=True):
        y = group["realized_net_bps"].to_numpy(dtype=float)
        w = group["diagnostic_weight"].to_numpy(dtype=float)
        valid_w = w > 0.0
        weighted = float(np.average(y[valid_w], weights=w[valid_w])) if valid_w.any() else float("nan")
        rows.append({"population": population, "scope": scope, "source": source, "decile": int(decile),
                     "rows": int(len(group)), "realized_net_mean_bps": float(np.mean(y)),
                     "realized_net_median_bps": float(np.median(y)), "weighted_realized_net_bps": weighted,
                     "hit_rate": float(np.mean(y > 0.0)), "p10_bps": float(np.quantile(y, .10)), "p05_bps": float(np.quantile(y, .05))})
    values = work["realized_net_bps"]
    rho = float(pd.Series(score, index=work.index).corr(values, method="spearman"))
    means = [row["realized_net_mean_bps"] for row in rows]
    violations = int(sum(right < left for left, right in zip(means, means[1:])))
    spread = means[-1] - means[0] if len(means) > 1 else float("nan")
    for row in rows:
        row.update({"spearman_rank_ic": rho, "monotonicity_violations": violations, "top_minus_bottom_bps": spread})
    return pd.DataFrame(rows)


def _monthly_signal_diagnostics(frame: pd.DataFrame, *, source: ScoreSource, population: str, scope: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    work = frame.copy()
    work["month"] = work["timestamp"].dt.to_period("M").astype(str)
    score = pd.Series(_score(work, source), index=work.index)
    rows: list[dict[str, object]] = []
    for month, group in work.groupby("month", sort=True):
        s = score.loc[group.index]
        y = group["policy_net_bps"]
        rank = s.rank(method="first", pct=True)
        decile = np.minimum(np.ceil(rank * 10.0).astype(int), 10)
        means = group.assign(_decile=decile).groupby("_decile")["policy_net_bps"].mean().tolist()
        rows.append({"population": population, "scope": scope, "source": source, "month": month,
                     "rows": len(group), "spearman_rank_ic": float(s.corr(y, method="spearman")),
                     "top_minus_bottom_bps": float(means[-1] - means[0]) if len(means) > 1 else float("nan"),
                     "monotonicity_violations": int(sum(right < left for left, right in zip(means, means[1:])))})
    return pd.DataFrame(rows)


def _execution_input_audit(panel: pd.DataFrame) -> dict[str, object]:
    required = ("live_full_spread_bps", "entry_impact_bps", "adverse_delay_gap_bps", "execution_adjusted_ev_bps")
    present = sorted(set(required).intersection(panel.columns))
    return {"B6_fixed10_execution_ev": "not_run", "B7_execution_ev_sizing": "not_run",
            "reason": "frozen matched score/outcome ledgers contain no causal per-decision execution-cost fields; values were not invented or joined post hoc",
            "required_fields": list(required), "present_fields": present,
            "outcome": "B6/B7 remain explicitly blocked pending a source-aligned point-in-time execution ledger"}


def _assert_fixed_control_parity(actual: pd.DataFrame, reference: Path) -> None:
    """Guard the new experiment's B0 path against a changed portfolio contract."""
    if not reference.exists():
        raise FileNotFoundError(f"required fixed-control reference is absent: {reference}")
    expected = pd.read_parquet(reference, columns=["candidate_id", "actual_weight"])
    left = actual.loc[:, ["candidate_id", "actual_weight"]].sort_values("candidate_id", kind="stable").reset_index(drop=True)
    right = expected.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not left["candidate_id"].equals(right["candidate_id"]):
        raise AssertionError("B0 candidate identities differ from the established fixed-10% control")
    if not np.allclose(left["actual_weight"].to_numpy(float), right["actual_weight"].to_numpy(float), atol=1e-12, rtol=0.0):
        raise AssertionError("B0 actual weights differ from the established fixed-10% control")


def _plain(frame: pd.DataFrame) -> str:
    return "```text\n" + frame.to_string(index=False) + "\n```"


def _report(out: Path, *, metrics: pd.DataFrame, diagnostics: pd.DataFrame, monthly_signal: pd.DataFrame,
            execution_audit: dict[str, object], dev_start: pd.Timestamp, dev_end: pd.Timestamp,
            val_start: pd.Timestamp, val_end: pd.Timestamp) -> None:
    validation = metrics.loc[metrics["scope"].eq("2026_validation")].copy()
    cols = ["arm", "stage", "source", "mode", "trades", "net_ev_bps_per_trade", "weighted_net_ev_bps", "portfolio_nav_change_bps_per_selected_capital", "net_pnl", "max_drawdown", "sortino_daily", "avg_actual_size", "avg_gross_utilization", "days_lt_1_trade"]
    diag = diagnostics.loc[(diagnostics["scope"].eq("2026_validation")) & (diagnostics["population"].eq("dual_admitted_pre_auction"))]
    report = [
        "# Matched-admission sizing ablation\n",
        f"- **Development/reference window:** {dev_start.date()} through {(dev_end - pd.Timedelta(days=1)).date()}. No sizing parameter was selected on 2026.\n",
        f"- **Untouched validation:** {val_start.date()} through {(val_end - pd.Timedelta(days=1)).date()}.\n",
        "- **Admission:** invariant `BCF MC1 >= 30 bps AND current-v5 MC1 >= 30 bps`.\n",
        "- **Auction:** invariant BCF-MC1 EV descending, then candidate ID.\n",
        "- **Only changed decision:** requested capital size; portfolio capacity can consequently alter fills.\n",
        "- **Outcome:** source-aligned optimized parent-policy net bps; invalid paths were already excluded in the frozen parent ledger.\n",
        "- **Live stack:** untouched. This is an offline research-only replay.\n\n",
        "## 2026 untouched validation\n\n", _plain(validation[cols].sort_values(["stage", "arm"])),
        "\n\n## 2026 EV-signal diagnostic on pre-auction canonical admissions\n\n", _plain(diag),
        "\n\n## 2026 monthly portability of EV signals\n\n", _plain(monthly_signal.loc[(monthly_signal["scope"].eq("2026_validation")) & (monthly_signal["population"].eq("dual_admitted_pre_auction"))]),
        "\n\n## Execution-EV scope\n\n```json\n", json.dumps(execution_audit, indent=2, sort_keys=True), "\n```\n",
    ]
    (out / "REPORT.md").write_text("".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-predictions", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--current-predictions", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-start", default="2025-03-01")
    parser.add_argument("--development-end", default="2026-01-01")
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--validation-end", default="2026-08-01")
    parser.add_argument(
        "--fixed-control-reference", type=Path,
        default=ROOT / "data_perp/artifacts/strict_r3_dual_mc1_position_sizing_2025opt_2026val_20260821_v1/decisions/A0_fixed_dual30_2026_validation_trades.parquet",
        help="Established fixed-10%% control required for exact validation-path parity.",
    )
    args = parser.parse_args()
    dev_start, dev_end = _utc(args.development_start), _utc(args.development_end)
    val_start, val_end = _utc(args.validation_start), _utc(args.validation_end)
    if not (dev_start < dev_end <= val_start < val_end):
        raise ValueError("require chronological development then validation windows")
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    decisions_root = args.out_dir / "decisions"
    decisions_root.mkdir()
    panel = _read_panel(args.bcf_predictions, args.current_predictions)
    # Full candidate score contract is never modified.  This assertion makes
    # the fixed-admission claim mechanically checkable.
    panel["canonical_dual_admitted"] = (panel["ev1_bcf_bps"] >= ADMISSION_FLOOR_BPS) & (panel["ev2_current_bps"] >= ADMISSION_FLOOR_BPS)
    arms = _arm_specs()
    metrics_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []
    baseline_trades: dict[str, pd.DataFrame] = {}
    periods = {"2025_development": (dev_start, dev_end), "2026_validation": (val_start, val_end),
               "full_2025_2026": (dev_start, val_end)}
    for arm in arms:
        # One continuous run preserves the live-equivalent portfolio state;
        # the three output periods below are views of this exact same path.
        full_metric, full_trade, full_state, _ = _simulate(
            panel, arm, start=dev_start, end=val_end, freeze_after=dev_end, capture=True,
        )
        for scope, (start, end) in periods.items():
            metric, trade, state = _scope_metrics_from_full(
                arm=arm, full_metric=full_metric, full_trade=full_trade, full_state=full_state, start=start, end=end,
            )
            metric["scope"] = scope
            metrics_rows.append(metric)
            trade.to_parquet(decisions_root / f"{arm.name}_{scope}_trades.parquet", index=False, compression="zstd")
            state.to_parquet(decisions_root / f"{arm.name}_{scope}_state.parquet", index=False, compression="zstd")
            monthly_rows.append(_monthly(trade, arm=arm.name, scope=scope))
            if arm.name == "B0_fixed10":
                baseline_trades[scope] = trade
                if scope == "2026_validation":
                    _assert_fixed_control_parity(trade, args.fixed_control_reference)
        print(json.dumps({"event": "arm_complete", "arm": arm.name}), flush=True)
    metrics = pd.DataFrame(metrics_rows)
    base = metrics.loc[metrics["arm"].eq("B0_fixed10")].set_index("scope")
    for field in ("trades", "net_pnl", "net_ev_bps_per_trade", "weighted_net_ev_bps", "max_drawdown", "sortino_daily", "days_lt_1_trade", "days_lt_5_trades"):
        metrics[f"delta_vs_b0_{field}"] = metrics.apply(lambda row: row[field] - base.loc[row["scope"], field], axis=1)
    metrics.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False, compression="zstd")
    metrics.to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    pd.concat(monthly_rows, ignore_index=True).to_parquet(args.out_dir / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly_rows, ignore_index=True).to_csv(args.out_dir / "monthly_metrics.csv", index=False)
    # The score premise is diagnosed before auction and again on the B0 trades.
    diagnostic_rows: list[pd.DataFrame] = []
    monthly_signal_rows: list[pd.DataFrame] = []
    for scope, (start, end) in periods.items():
        admitted = panel.loc[panel["canonical_dual_admitted"] & panel["timestamp"].ge(start) & panel["timestamp"].lt(end)].copy()
        accepted = baseline_trades[scope].copy()
        for source in ("bcf", "current", "blend70", "blend50_disagreement"):
            diagnostic_rows.append(_decile_rows(admitted, source=source, population="dual_admitted_pre_auction", scope=scope))
            monthly_signal_rows.append(_monthly_signal_diagnostics(admitted, source=source, population="dual_admitted_pre_auction", scope=scope))
            if not accepted.empty:
                accepted_for_diag = accepted.rename(columns={"sizing_score_bps": "_discard"}).copy()
                diagnostic_rows.append(_decile_rows(accepted_for_diag, source=source, population="B0_portfolio_accepted", scope=scope, weights=accepted_for_diag["amount"]))
                monthly_signal_rows.append(_monthly_signal_diagnostics(accepted_for_diag, source=source, population="B0_portfolio_accepted", scope=scope))
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    monthly_signal = pd.concat(monthly_signal_rows, ignore_index=True)
    diagnostics.to_parquet(args.out_dir / "ev_signal_diagnostics.parquet", index=False, compression="zstd")
    monthly_signal.to_parquet(args.out_dir / "ev_signal_monthly_portability.parquet", index=False, compression="zstd")
    execution_audit = _execution_input_audit(panel)
    (args.out_dir / "execution_ev_scope_audit.json").write_text(json.dumps(execution_audit, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "arm_definitions.json").write_text(json.dumps([asdict(a) for a in arms], indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": "strict_r3_matched_admission_sizing_ablation_v1", "status": "complete",
        "purpose": "offline matched-admission sizing research; no live-stack mutation",
        "inputs": {"bcf": {"path": str(args.bcf_predictions), "sha256": _sha(args.bcf_predictions)},
                   "current": {"path": str(args.current_predictions), "sha256": _sha(args.current_predictions)}},
        "admission": "invariant BCF MC1 >= 30 bps AND current-v5 MC1 >= 30 bps",
        "priority": "invariant BCF MC1 EV descending, candidate ID tie-break",
        "constraints": {"max_new_per_hour": MAX_NEW_PER_HOUR, "max_concurrent": MAX_CONCURRENT, "max_gross_nav": MIN_GROSS_WEIGHT},
        "development_reference": {"start": dev_start.isoformat(), "end_exclusive": dev_end.isoformat()},
        "untouched_validation": {"start": val_start.isoformat(), "end_exclusive": val_end.isoformat()},
        "execution_ev": execution_audit,
        "arms": [asdict(a) for a in arms],
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    _report(args.out_dir, metrics=metrics, diagnostics=diagnostics, monthly_signal=monthly_signal, execution_audit=execution_audit,
            dev_start=dev_start, dev_end=dev_end, val_start=val_start, val_end=val_end)
    print(args.out_dir)


if __name__ == "__main__":
    main()
