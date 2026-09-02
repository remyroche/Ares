#!/usr/bin/env python3
"""Causal BCF/current-v5 MC1 position-sizing ablation.

This is an offline, research-only replay.  It deliberately consumes the
already prequential BCF and current-v5 MC1 ledgers and the source-aligned
parent-policy outcome.  It never changes admission, scoring, execution, or
live state.  The only decision under test is capital sizing after a candidate
has a causal mapped-EV estimate.

The 2025 development period is used to select the small, sequential funnel of
monotone sizing parameters.  The selected configuration is then frozen and
replayed without retuning over 2026.  Execution-cost variants are intentionally
outside this runner: ``ev_exec`` is the robust combination of the two mapped
net-EV estimates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BCF = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_bcf/predictions_bcf_mc1_d2.parquet"
)
DEFAULT_CURRENT = ROOT / (
    "data_perp/artifacts/strict_r3_score_family_matched_mc1_canonical_policy_"
    "20260817_v7_current/predictions_current_v5_mc1_d2.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_dual_mc1_position_sizing_2025opt_2026val_20260821_v1"

E_MIN = 30.0
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.15
MAX_GROSS_WEIGHT = 0.80
MAX_NEW_PER_HOUR = 2
MAX_CONCURRENT = 8
MIN_P90_ROWS = 1_000
INITIAL_NAV = 1_000.0

POLICY_COLUMNS = (
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class SizingParams:
    """All parameters are dimensionless except ``emin_bps``.

    ``kf`` and ``kh`` are multiples of ``Efull0 - Emin``.  ``eta`` scales the
    open-book expected-EV proxy in the same units.  The proxy is explicitly an
    entry-EV proxy because no causal remaining-EV label is available in this
    frozen historical ledger.
    """

    w_bcf: float = 0.5
    disagreement_lambda: float = 0.0
    emin_bps: float = E_MIN
    gamma0: float = 1.0
    gamma1: float = 0.0
    scarcity_mode: str = "none"  # none | linear | convex
    u0: float = 0.0
    scarcity_power: float = 1.0
    kf: float = 0.0
    kh: float = 0.0
    midpoint_utilization: bool = False
    shadow_mode: str = "none"  # none | p20_entry_ev | marginal_entry_ev
    eta: float = 0.0
    admission_mode: str = "robust"  # robust | dual_control
    fixed_weight: float | None = None


def _read_panel(bcf_path: Path, current_path: Path) -> pd.DataFrame:
    bcf_fields = [
        "candidate_id", "__decision_ts__", "__symbol__", "mc1_expected_bps", *POLICY_COLUMNS,
    ]
    current_fields = ["candidate_id", "__decision_ts__", "mc1_expected_bps", *POLICY_COLUMNS]
    bcf = pd.read_parquet(bcf_path, columns=bcf_fields).rename(columns={
        "mc1_expected_bps": "ev1_bcf_bps",
    })
    current = pd.read_parquet(current_path, columns=current_fields).rename(columns={
        "mc1_expected_bps": "ev2_current_bps",
    })
    for frame in (bcf, current):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        frame["policy_label_available_ts"] = pd.to_datetime(
            frame["policy_label_available_ts"], utc=True, errors="coerce"
        )
        if frame["candidate_id"].duplicated().any():
            raise ValueError("MC1 source has duplicate candidate identities")
    merged = bcf.merge(
        current,
        on="candidate_id",
        suffixes=("_bcf", "_current"),
        validate="one_to_one",
    )
    for field in ("__decision_ts__", *POLICY_COLUMNS):
        left, right = merged[f"{field}_bcf"], merged[f"{field}_current"]
        if pd.api.types.is_numeric_dtype(left):
            equal = np.isclose(
                pd.to_numeric(left, errors="coerce").to_numpy(float),
                pd.to_numeric(right, errors="coerce").to_numpy(float),
                equal_nan=True,
            ).all()
        else:
            equal = left.fillna("__null__").astype(str).equals(right.fillna("__null__").astype(str))
        if not equal:
            raise AssertionError(f"policy field differs between BCF/current ledgers: {field}")
    out = pd.DataFrame({
        "candidate_id": merged["candidate_id"].astype(str),
        "timestamp": pd.to_datetime(merged["__decision_ts___bcf"], utc=True),
        "symbol": merged["__symbol__"].astype(str),
        "ev1_bcf_bps": pd.to_numeric(merged["ev1_bcf_bps"], errors="coerce"),
        "ev2_current_bps": pd.to_numeric(merged["ev2_current_bps"], errors="coerce"),
    })
    for field in POLICY_COLUMNS:
        out[field] = merged[f"{field}_bcf"]
    out["policy_path_valid"] = out["policy_path_valid"].fillna(False).astype(bool)
    required_numeric = (
        "ev1_bcf_bps", "ev2_current_bps", "policy_net_bps", "policy_gross_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    )
    finite = out.loc[:, list(required_numeric)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    # This sizing study is deliberately limited to the common BCF/current
    # score contract.  Invalid policy paths are excluded from entry and from
    # the mark universe; when an already-open symbol is absent at a later
    # common-score decision, its last causal mark is retained explicitly.
    out = out.loc[out["policy_path_valid"] & finite].copy()
    out["policy_net_bps"] = pd.to_numeric(out["policy_net_bps"], errors="raise")
    out["policy_gross_bps"] = pd.to_numeric(out["policy_gross_bps"], errors="raise")
    out["policy_entry_price"] = pd.to_numeric(out["policy_entry_price"], errors="raise")
    out["exit_bar"] = pd.to_numeric(out["policy_exit_bar_15m"], errors="coerce").fillna(-1).astype(int)
    out["exit_ts"] = out["timestamp"] + pd.to_timedelta(np.maximum(out["exit_bar"] + 1, 0) * 15, unit="min")
    out["month"] = out["timestamp"].dt.to_period("M").astype(str)
    if out.empty:
        raise ValueError("no common valid policy rows")
    out = out.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)
    return out


def _robust_ev(panel: pd.DataFrame, params: SizingParams) -> np.ndarray:
    e1 = panel["ev1_bcf_bps"].to_numpy(float)
    e2 = panel["ev2_current_bps"].to_numpy(float)
    return params.w_bcf * e1 + (1.0 - params.w_bcf) * e2 - params.disagreement_lambda * np.abs(e1 - e2)


def _month_p90(
    panel: pd.DataFrame, robust: np.ndarray, *, freeze_after: pd.Timestamp,
) -> dict[str, float]:
    """Causal P90 scale, frozen at the end of the 2025 selection window.

    Months before ``freeze_after`` use only earlier score rows.  Every later
    month reuses the one P90 fitted on rows strictly before the cutoff, so no
    2026 validation score distribution can tune the sizing curve.
    """
    work = pd.DataFrame({"timestamp": panel["timestamp"].to_numpy(), "robust": robust})
    work["month_start"] = work["timestamp"].dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
    values: dict[str, float] = {}
    for month in sorted(work["month_start"].unique()):
        cutoff = min(pd.Timestamp(month), freeze_after)
        prior = work.loc[work["timestamp"].lt(cutoff), "robust"].to_numpy(float)
        prior = prior[np.isfinite(prior)]
        key = pd.Timestamp(month).strftime("%Y-%m")
        values[key] = float(np.quantile(prior, .90)) if len(prior) >= MIN_P90_ROWS else float("nan")
    return values


def _g(utilization: float, params: SizingParams) -> float:
    if params.scarcity_mode == "none":
        return 0.0
    if params.scarcity_mode == "linear":
        return float(np.clip(utilization, 0.0, 1.0))
    x = np.clip((utilization - params.u0) / max(1.0 - params.u0, 1e-9), 0.0, 1.0)
    return float(x ** params.scarcity_power)


def _weighted_bottom_ev(positions: list[dict[str, float]], amount_to_displace: float, emin: float) -> float:
    if amount_to_displace <= 0.0 or not positions:
        return emin
    remaining = amount_to_displace
    total, weighted = 0.0, 0.0
    for position in sorted(positions, key=lambda item: item["entry_ev_bps"]):
        take = min(remaining, position["amount"])
        weighted += take * position["entry_ev_bps"]
        total += take
        remaining -= take
        if remaining <= 1e-12:
            break
    return weighted / total if total > 1e-12 else emin


def _shadow_ev(
    positions: list[dict[str, float]], *, mode: str, desired_amount: float,
    capacity_remaining: float, emin: float,
) -> float:
    if mode == "none" or not positions:
        return emin
    if mode == "p20_entry_ev":
        return float(np.quantile(np.asarray([x["entry_ev_bps"] for x in positions], dtype=float), .20))
    if mode == "marginal_entry_ev":
        return _weighted_bottom_ev(positions, max(desired_amount - capacity_remaining, 0.0), emin)
    raise ValueError(f"unsupported shadow mode: {mode}")


def _desired_weight(
    *, ev_bps: float, efull0: float, nav: float, gross_amount: float,
    positions: list[dict[str, float]], params: SizingParams,
) -> tuple[float, float, float, float, float]:
    """Return target weight, utilization, shadow EV, hurdle and full threshold."""
    if params.fixed_weight is not None:
        utilization = float(np.clip(gross_amount / max(MAX_GROSS_WEIGHT * nav, 1e-12), 0.0, 1.0))
        return params.fixed_weight, utilization, params.emin_bps, params.emin_bps, efull0
    if not np.isfinite(efull0) or ev_bps < params.emin_bps:
        return 0.0, 0.0, params.emin_bps, float("nan"), float("nan")
    span = max(efull0 - params.emin_bps, 1e-6)
    size = MIN_WEIGHT
    # The midpoint iteration is intentionally capped and deterministic.
    iterations = 5 if params.midpoint_utilization else 1
    last = (size, 0.0, params.emin_bps, params.emin_bps, efull0)
    for _ in range(iterations):
        amount = size * nav
        if params.midpoint_utilization:
            util = (gross_amount + 0.5 * amount) / max(MAX_GROSS_WEIGHT * nav, 1e-12)
        else:
            util = gross_amount / max(MAX_GROSS_WEIGHT * nav, 1e-12)
        util = float(np.clip(util, 0.0, 1.0))
        g = _g(util, params)
        # Use a preliminary non-shadow size to identify the capital that the
        # candidate would displace.  This is a proxy only; it cannot close an
        # incumbent or use any future outcome.
        preliminary_h = params.emin_bps + params.kh * span * g
        preliminary_full = efull0 + (preliminary_h - params.emin_bps) + params.kf * span * g
        preliminary_x = float(np.clip((ev_bps - preliminary_h) / max(preliminary_full - preliminary_h, 1e-6), 0.0, 1.0))
        preliminary_size = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * preliminary_x ** max(params.gamma0, 1e-6)
        shadow = _shadow_ev(
            positions, mode=params.shadow_mode, desired_amount=preliminary_size * nav,
            capacity_remaining=max(MAX_GROSS_WEIGHT * nav - gross_amount, 0.0), emin=params.emin_bps,
        )
        hurdle = params.emin_bps + params.kh * span * g + params.eta * g * max(shadow - params.emin_bps, 0.0)
        full = efull0 + (hurdle - params.emin_bps) + params.kf * span * g
        x = float(np.clip((ev_bps - hurdle) / max(full - hurdle, 1e-6), 0.0, 1.0))
        gamma = max(params.gamma0 + params.gamma1 * g, 1e-6)
        size = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * x ** gamma
        last = (size, util, shadow, hurdle, full)
    return last


def _base_admission(row: pd.Series, ev_bps: float, params: SizingParams) -> bool:
    if params.admission_mode == "dual_control":
        return bool(row["ev1_bcf_bps"] >= params.emin_bps and row["ev2_current_bps"] >= params.emin_bps)
    if params.admission_mode == "robust":
        return bool(ev_bps >= params.emin_bps)
    raise ValueError(f"unsupported admission mode {params.admission_mode}")


def _close_positions(
    positions: list[dict[str, float]], *, timestamp: pd.Timestamp, cash: float,
    exits: list[dict[str, object]],
) -> tuple[list[dict[str, float]], float]:
    remaining: list[dict[str, float]] = []
    for position in positions:
        if position["exit_ts"] <= timestamp:
            pnl = position["amount"] * position["policy_net_bps"] / 10_000.0
            # The policy net return is the canonical realised all-in outcome;
            # return the original invested capital plus its labelled net PnL
            # to cash exactly once at the policy exit.
            cash += position["amount"] + pnl
            exits.append({
                "candidate_id": position["candidate_id"], "exit_ts": position["exit_ts"],
                "realized_pnl": pnl, "policy_net_bps": position["policy_net_bps"],
            })
        else:
            remaining.append(position)
    return remaining, cash


def _mark_positions(
    positions: list[dict[str, float]], marks: dict[str, float], *, mark_stats: dict[str, int],
) -> None:
    """Mark current invested capital from the decision-time entry/open price.

    The shared BCF/current score ledger exposes the executable decision-time
    price for its common score rows.  It is used only to value positions which
    were already open before this decision.  A missing common-score mark
    retains the preceding causal mark and is explicitly counted; no expanded
    universe or future outcome is consulted.
    """
    for position in positions:
        mark_stats["attempts"] += 1
        mark = marks.get(str(position["symbol"]))
        if mark is None or not np.isfinite(mark) or mark <= 0.0:
            mark_stats["fallbacks"] += 1
            continue
        position["market_value"] = position["quantity"] * float(mark)
        position["last_mark_price"] = float(mark)
        mark_stats["updates"] += 1


def _simulate(
    panel: pd.DataFrame, params: SizingParams, *, start: pd.Timestamp, end: pd.Timestamp,
    arm: str, capture: bool, p90_freeze_after: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sequentially replay one sizing policy with no post-decision state."""
    robust = _robust_ev(panel, params)
    work = panel.copy()
    work["ev_exec_bps"] = robust  # costs intentionally excluded in this study.
    p90_map = _month_p90(work, robust, freeze_after=p90_freeze_after)
    # Replay from the beginning of the common-score ledger, not from an empty
    # book at the reporting boundary.  This carries exposure, cash, and open
    # positions into holdout/validation periods exactly as a live portfolio
    # would.  Only decisions and metrics in [start, end) are reported.
    all_current = work.loc[work["timestamp"].lt(end)].copy()
    evaluation = all_current.copy()
    if evaluation.empty:
        raise ValueError("empty evaluation interval")
    evaluation["efull0_bps"] = evaluation["month"].map(p90_map)
    times = pd.DatetimeIndex(all_current["timestamp"].drop_duplicates().sort_values())
    candidate_by_time = {
        timestamp: group.sort_values(["ev_exec_bps", "candidate_id"], ascending=[False, True], kind="stable")
        for timestamp, group in evaluation.groupby("timestamp", sort=True)
    }
    marks_by_time = {
        timestamp: dict(zip(group["symbol"].astype(str), group["policy_entry_price"].astype(float)))
        for timestamp, group in all_current.groupby("timestamp", sort=True)
    }
    cash = INITIAL_NAV
    positions: list[dict[str, float]] = []
    decisions: list[dict[str, object]] = []
    exits: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    mark_stats = {"attempts": 0, "updates": 0, "fallbacks": 0}
    starting_nav: float | None = None
    for timestamp in times:
        positions, cash = _close_positions(positions, timestamp=timestamp, cash=cash, exits=exits)
        _mark_positions(positions, marks_by_time.get(timestamp, {}), mark_stats=mark_stats)
        nav = cash + float(sum(item["market_value"] for item in positions))
        in_report_window = timestamp >= start
        if in_report_window and starting_nav is None:
            starting_nav = nav
        group = candidate_by_time.get(timestamp)
        if group is None:
            if in_report_window:
                states.append({
                    "timestamp": timestamp, "nav": nav, "gross_amount": float(sum(item["market_value"] for item in positions)),
                    "gross_weight": float(sum(item["market_value"] for item in positions)) / max(nav, 1e-12),
                    "utilization": float(sum(item["market_value"] for item in positions)) / max(MAX_GROSS_WEIGHT * nav, 1e-12),
                    "open_positions": len(positions),
                })
            continue
        gross_amount = float(sum(item["market_value"] for item in positions))
        if in_report_window:
            states.append({
                "timestamp": timestamp, "nav": nav, "gross_amount": gross_amount,
                "gross_weight": gross_amount / max(nav, 1e-12),
                "utilization": gross_amount / max(MAX_GROSS_WEIGHT * nav, 1e-12),
                "open_positions": len(positions),
            })
        entries = 0
        open_symbols = {str(item["symbol"]) for item in positions}
        for _, row in group.iterrows():
            ev = float(row["ev_exec_bps"])
            if not _base_admission(row, ev, params):
                continue
            if not np.isfinite(float(row["efull0_bps"])):
                continue  # February 2025 is the p90 warm-up only.
            if entries >= MAX_NEW_PER_HOUR or len(positions) >= MAX_CONCURRENT or str(row["symbol"]) in open_symbols:
                continue
            gross_amount = float(sum(item["market_value"] for item in positions))
            target_weight, util, shadow, hurdle, full = _desired_weight(
                ev_bps=ev, efull0=float(row["efull0_bps"]), nav=nav,
                gross_amount=gross_amount, positions=positions, params=params,
            )
            capacity = max(MAX_GROSS_WEIGHT * nav - gross_amount, 0.0)
            if capacity < MIN_WEIGHT * nav:
                continue
            actual_amount = min(target_weight * nav, capacity)
            if actual_amount < MIN_WEIGHT * nav - 1e-9:
                continue
            actual_weight = actual_amount / max(nav, 1e-12)
            position = {
                "candidate_id": str(row["candidate_id"]), "symbol": str(row["symbol"]),
                "entry_ts": timestamp, "exit_ts": row["exit_ts"], "amount": actual_amount,
                "quantity": actual_amount / float(row["policy_entry_price"]), "market_value": actual_amount,
                "last_mark_price": float(row["policy_entry_price"]),
                "entry_ev_bps": ev, "policy_net_bps": float(row["policy_net_bps"]),
            }
            positions.append(position)
            cash -= actual_amount
            open_symbols.add(str(row["symbol"]))
            entries += 1
            if in_report_window:
                decisions.append({
                    "arm": arm, "candidate_id": str(row["candidate_id"]), "timestamp": timestamp,
                    "symbol": str(row["symbol"]), "exit_ts": row["exit_ts"],
                    "ev1_bcf_bps": float(row["ev1_bcf_bps"]), "ev2_current_bps": float(row["ev2_current_bps"]),
                    "ev_exec_bps": ev, "disagreement_bps": abs(float(row["ev1_bcf_bps"]) - float(row["ev2_current_bps"])),
                    "efull0_bps": float(row["efull0_bps"]), "hurdle_bps": hurdle, "efull_bps": full,
                    "shadow_ev_bps": shadow, "pre_trade_utilization": util,
                    "target_weight": target_weight, "actual_weight": actual_weight,
                    "amount": actual_amount, "policy_net_bps": float(row["policy_net_bps"]),
                    "policy_gross_bps": float(row["policy_gross_bps"]),
                    "holding_hours": (row["exit_ts"] - timestamp).total_seconds() / 3600.0,
                })
    # Close every outcome resolved inside the declared period.  Positions may
    # still be open at an artificial interval end only if their real policy
    # exit is after it; report them explicitly rather than inventing a price.
    positions, cash = _close_positions(positions, timestamp=end, cash=cash, exits=exits)
    final_nav = cash + float(sum(item["market_value"] for item in positions))
    if starting_nav is None:
        raise ValueError("reporting interval has no common-score decision timestamps")
    states.append({
        "timestamp": end, "nav": final_nav, "gross_amount": float(sum(item["market_value"] for item in positions)),
        "gross_weight": float(sum(item["market_value"] for item in positions)) / max(final_nav, 1e-12),
        "utilization": float(sum(item["market_value"] for item in positions)) / max(MAX_GROSS_WEIGHT * final_nav, 1e-12),
        "open_positions": len(positions),
    })
    trade = pd.DataFrame(decisions)
    exit_frame = pd.DataFrame(exits)
    if not exit_frame.empty:
        exit_frame = exit_frame.loc[pd.to_datetime(exit_frame["exit_ts"], utc=True).ge(start)].copy()
    state = pd.DataFrame(states)
    metrics = _metrics(
        trade, exit_frame, state, start=start, end=end, initial_nav=starting_nav,
        final_nav=final_nav, arm=arm,
    )
    metrics["unresolved_open_positions_at_end"] = int(len(positions))
    metrics["mark_updates"] = int(mark_stats["updates"])
    metrics["mark_fallbacks"] = int(mark_stats["fallbacks"])
    metrics["mark_coverage"] = float(mark_stats["updates"] / mark_stats["attempts"]) if mark_stats["attempts"] else 1.0
    return metrics, trade if capture else pd.DataFrame(), state if capture else pd.DataFrame(), exit_frame if capture else pd.DataFrame()


def _metrics(
    trade: pd.DataFrame, exits: pd.DataFrame, state: pd.DataFrame, *, start: pd.Timestamp,
    end: pd.Timestamp, initial_nav: float, final_nav: float, arm: str,
) -> dict[str, object]:
    days = pd.date_range(start.normalize(), (end - pd.Timedelta(seconds=1)).normalize(), freq="1D", tz="UTC")
    if state.empty:
        daily_nav = pd.Series(initial_nav, index=days)
        marked_nav = pd.Series(dtype=float)
    else:
        marked_nav = state.sort_values("timestamp").groupby("timestamp", sort=True)["nav"].last()
        daily_nav = marked_nav.groupby(marked_nav.index.normalize()).last().reindex(days).ffill().fillna(initial_nav)
    daily_pnl = daily_nav.diff().fillna(daily_nav.iloc[0] - initial_nav)
    prior_nav = daily_nav.shift(1, fill_value=initial_nav)
    daily_return = daily_pnl / prior_nav.replace(0.0, np.nan)
    downside = daily_return.loc[daily_return.lt(0.0)]
    sortino = float(daily_return.mean() / downside.std(ddof=0) * math.sqrt(365.0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else float("nan")
    # Drawdown must include the causal hourly marks of still-open positions,
    # rather than only realised policy exits.  This is deliberately restricted
    # to the common BCF/current mark contract used by the sizing study.
    if not marked_nav.empty:
        marked_peak = marked_nav.cummax()
        max_dd = float((marked_nav / marked_peak - 1.0).min()) if len(marked_nav) else 0.0
    else:
        peak = daily_nav.cummax()
        max_dd = float((daily_nav / peak - 1.0).min()) if len(daily_nav) else 0.0
    exit_month = daily_pnl.groupby(daily_pnl.index.to_period("M")).sum()
    exit_week = daily_pnl.groupby(daily_pnl.index.to_period("W-MON")).sum()
    deployed = float(trade["amount"].sum()) if not trade.empty else 0.0
    capital_days = float((trade["actual_weight"] * trade["holding_hours"] / 24.0).sum()) if not trade.empty else 0.0
    util = state["utilization"] if not state.empty else pd.Series(dtype=float)
    gross = state["gross_weight"] if not state.empty else pd.Series(dtype=float)
    net_pnl = float(final_nav - INITIAL_NAV)
    result: dict[str, object] = {
        "arm": arm, "start": start, "end_exclusive": end,
        "trades": int(len(trade)), "trades_per_day": float(len(trade) / max(len(days), 1)),
        "initial_nav": float(initial_nav), "net_pnl": net_pnl, "final_nav": float(final_nav),
        "net_portfolio_return": float(final_nav / initial_nav - 1.0),
        "return_on_capital_deployed": net_pnl / deployed if deployed > 0 else float("nan"),
        "capital_deployed": deployed, "capital_days_deployed": capital_days,
        "turnover_nav": deployed / max(float(state["nav"].mean()) if not state.empty else INITIAL_NAV, 1e-12),
        "net_ev_bps_per_trade": float(trade["policy_net_bps"].mean()) if not trade.empty else float("nan"),
        "weighted_net_ev_bps": float(1e4 * net_pnl / deployed) if deployed > 0 else float("nan"),
        "avg_target_size": float(trade["target_weight"].mean()) if not trade.empty else float("nan"),
        "avg_actual_size": float(trade["actual_weight"].mean()) if not trade.empty else float("nan"),
        "p05_actual_size": float(trade["actual_weight"].quantile(.05)) if not trade.empty else float("nan"),
        "p50_actual_size": float(trade["actual_weight"].quantile(.50)) if not trade.empty else float("nan"),
        "p95_actual_size": float(trade["actual_weight"].quantile(.95)) if not trade.empty else float("nan"),
        "avg_gross_utilization": float(util.mean()) if len(util) else 0.0,
        "median_gross_utilization": float(util.median()) if len(util) else 0.0,
        "p90_gross_utilization": float(util.quantile(.90)) if len(util) else 0.0,
        "p95_gross_utilization": float(util.quantile(.95)) if len(util) else 0.0,
        "avg_gross_weight": float(gross.mean()) if len(gross) else 0.0,
        "max_drawdown": max_dd,
        "worst_day_pnl": float(daily_pnl.min()) if len(daily_pnl) else 0.0,
        "worst_week_pnl": float(exit_week.min()) if len(exit_week) else 0.0,
        "worst_month_pnl": float(exit_month.min()) if len(exit_month) else 0.0,
        "positive_months": int(exit_month.gt(0.0).sum()),
        "months": int(len(exit_month)),
        "sortino_daily": sortino,
        "calmar_like": float((final_nav / initial_nav - 1.0) / max(abs(max_dd), .01)),
        "days_lt_1_trade": int((trade.groupby("timestamp").size().reindex(pd.date_range(start, end - pd.Timedelta(hours=1), freq="1h", tz="UTC"), fill_value=0).groupby(lambda x: x.normalize()).sum() < 1).sum()) if False else 0,
    }
    # Day-level entry drought metrics must count entries, not realised exits.
    if not trade.empty:
        entry_daily = trade.groupby(trade["timestamp"].dt.normalize()).size().reindex(days, fill_value=0)
    else:
        entry_daily = pd.Series(0, index=days)
    result["days_lt_1_trade"] = int(entry_daily.lt(1).sum())
    result["days_lt_5_trades"] = int(entry_daily.lt(5).sum())
    return result


def _selection_score(metrics: dict[str, object]) -> tuple[float, float, float]:
    """Predeclared development objective: return/drawdown, then PnL, then EV."""
    calmar = float(metrics["calmar_like"])
    pnl = float(metrics["net_pnl"])
    ev = float(metrics["weighted_net_ev_bps"])
    return (calmar if np.isfinite(calmar) else -np.inf, pnl, ev if np.isfinite(ev) else -np.inf)


def _run_grid(
    panel: pd.DataFrame, *, stage: str, base: SizingParams, candidates: Iterable[SizingParams],
    dev_start: pd.Timestamp, dev_end: pd.Timestamp, p90_freeze_after: pd.Timestamp,
) -> tuple[SizingParams, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    best: SizingParams | None = None
    best_key: tuple[float, float, float] | None = None
    for index, params in enumerate(candidates):
        print(json.dumps({"event": "hpo_trial_start", "stage": stage, "trial": index}), flush=True)
        metric, _, _, _ = _simulate(
            panel, params, start=dev_start, end=dev_end, arm=stage, capture=False,
            p90_freeze_after=p90_freeze_after,
        )
        key = _selection_score(metric)
        rows.append({"stage": stage, "trial": index, **asdict(params), **metric, "selection_score": key[0]})
        if best_key is None or key > best_key:
            best, best_key = params, key
        gc.collect()
    if best is None:
        raise RuntimeError(f"no HPO candidate completed for {stage}")
    return best, pd.DataFrame(rows)


def _funnel(
    panel: pd.DataFrame, *, dev_start: pd.Timestamp, dev_end: pd.Timestamp,
) -> tuple[dict[str, SizingParams], pd.DataFrame]:
    chosen: dict[str, SizingParams] = {}
    hpo: list[pd.DataFrame] = []
    a0 = SizingParams(w_bcf=1.0, admission_mode="dual_control", fixed_weight=.10)
    chosen["A0_fixed_dual30"] = a0
    chosen["A1_ev1_only"] = SizingParams(w_bcf=1.0)
    chosen["A2_ev2_only"] = SizingParams(w_bcf=0.0)
    chosen["A3_mean50"] = SizingParams(w_bcf=.50)

    a4, out = _run_grid(panel, stage="A4_blend_lambda0", base=SizingParams(), candidates=(
        SizingParams(w_bcf=w) for w in np.arange(0.0, 1.01, .10)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A4_optimized_blend"] = a4

    coarse = [SizingParams(w_bcf=w, disagreement_lambda=l) for w in np.arange(0.0, 1.01, .10) for l in (0.0, .10, .25, .50, .75, 1.0)]
    coarse_best, out = _run_grid(panel, stage="A5_blend_disagreement_coarse", base=a4, candidates=coarse, dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out)
    refine_w = sorted({float(np.clip(coarse_best.w_bcf + delta, 0.0, 1.0)) for delta in (-.05, 0.0, .05)})
    refine_l = sorted({max(0.0, coarse_best.disagreement_lambda + delta) for delta in (-.10, 0.0, .10)})
    a5, out = _run_grid(panel, stage="A5_blend_disagreement_refine", base=coarse_best, candidates=(
        replace(coarse_best, w_bcf=w, disagreement_lambda=l) for w in refine_w for l in refine_l
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A5_optimized_blend_disagreement"] = a5

    a6, out = _run_grid(panel, stage="A6_gamma", base=a5, candidates=(
        replace(a5, gamma0=gamma) for gamma in (.50, .75, 1.0, 1.25, 1.50, 2.0, 2.5, 3.0)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A6_gamma"] = a6

    a7, out = _run_grid(panel, stage="A7_linear_scarcity", base=a6, candidates=(
        replace(a6, scarcity_mode="linear", u0=0.0, scarcity_power=1.0, kf=kf, kh=0.0) for kf in (0.0, .25, .50, 1.0, 1.5)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A7_linear_scarcity"] = a7

    a8, out = _run_grid(panel, stage="A8_convex_scarcity", base=a6, candidates=(
        replace(a6, scarcity_mode="convex", u0=u0, scarcity_power=power, kf=kf, kh=0.0)
        for u0 in (0.0, .25, .40, .50) for power in (1.0, 1.5, 2.0, 3.0) for kf in (0.0, .25, .50, 1.0, 1.5)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A8_convex_scarcity"] = a8

    a9, out = _run_grid(panel, stage="A9_hurdle", base=a8, candidates=(
        replace(a8, kh=kh) for kh in (0.0, .25, .50, 1.0, 1.5)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A9_hurdle"] = a9

    a10, out = _run_grid(panel, stage="A10_gamma_utilization", base=a9, candidates=(
        replace(a9, gamma1=gamma1) for gamma1 in (0.0, .50, 1.0, 1.5, 2.0)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A10_gamma_utilization"] = a10

    chosen["A11_midpoint_utilization"] = replace(a9, midpoint_utilization=True)
    a12, out = _run_grid(panel, stage="A12_shadow_ev", base=a9, candidates=(
        replace(a9, shadow_mode="p20_entry_ev" if eta > 0.0 else "none", eta=eta) for eta in (0.0, .25, .50, 1.0)
    ), dev_start=dev_start, dev_end=dev_end, p90_freeze_after=dev_end)
    hpo.append(out); chosen["A12_shadow_ev"] = a12
    chosen["A13_marginal_displaced_ev"] = replace(a12, shadow_mode="marginal_entry_ev")
    return chosen, pd.concat(hpo, ignore_index=True)


def _monthly(trade: pd.DataFrame, arm: str) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame(columns=["arm", "month", "trades", "net_pnl", "weighted_net_ev_bps"])
    work = trade.copy()
    work["month"] = work["timestamp"].dt.to_period("M").astype(str)
    work["pnl"] = work["amount"] * work["policy_net_bps"] / 10_000.0
    out = work.groupby("month", sort=True).agg(trades=("candidate_id", "size"), net_pnl=("pnl", "sum"), capital=("amount", "sum"), net_ev_bps=("policy_net_bps", "mean"))
    out["weighted_net_ev_bps"] = 1e4 * out["net_pnl"] / out["capital"]
    out = out.reset_index()
    out.insert(0, "arm", arm)
    return out


def _diagnostics(trade: pd.DataFrame, arm: str) -> pd.DataFrame:
    if trade.empty:
        return pd.DataFrame()
    work = trade.copy()
    # Quantiles are formed only within the completed evaluation output and are
    # diagnostics, never inputs to the sizing or admission decision.
    for source, target in (("ev_exec_bps", "ev_quintile"), ("pre_trade_utilization", "util_quintile"), ("disagreement_bps", "disagreement_quintile")):
        ranks = work[source].rank(method="first", pct=True)
        work[target] = np.minimum((ranks * 5).astype(int), 5).clip(lower=1)
    work["size_bucket"] = pd.cut(work["actual_weight"], bins=[0.0, .075, .10, .125, .151], labels=["5-7.5%", "7.5-10%", "10-12.5%", "12.5-15%"], include_lowest=True).astype(str)
    work["pnl"] = work["amount"] * work["policy_net_bps"] / 10_000.0
    groupings = {
        "ev_exec_quintile": ["ev_quintile"],
        "utilization_quintile": ["util_quintile"],
        "ev_x_utilization": ["ev_quintile", "util_quintile"],
        "size_bucket": ["size_bucket"],
        "disagreement_quintile": ["disagreement_quintile"],
    }
    blocks: list[pd.DataFrame] = []
    for name, cols in groupings.items():
        result = work.groupby(cols, dropna=False).agg(
            trades=("candidate_id", "size"), net_pnl=("pnl", "sum"), capital=("amount", "sum"),
            mean_net_bps=("policy_net_bps", "mean"), mean_ev_exec_bps=("ev_exec_bps", "mean"),
            mean_size=("actual_weight", "mean"),
        ).reset_index()
        result["weighted_net_ev_bps"] = 1e4 * result["net_pnl"] / result["capital"]
        result.insert(0, "diagnostic", name)
        result.insert(0, "arm", arm)
        blocks.append(result)
    return pd.concat(blocks, ignore_index=True, sort=False)


def _plain_table(frame: pd.DataFrame) -> str:
    """Dependency-free report rendering for the sealed research artifact."""
    return "```text\n" + frame.to_string(index=False) + "\n```"


def _write_terminal_artifacts(
    args: argparse.Namespace, *, hpo: pd.DataFrame, metrics: pd.DataFrame,
    dev_start: pd.Timestamp, dev_end: pd.Timestamp, val_start: pd.Timestamp, val_end: pd.Timestamp,
) -> None:
    report = [
        "# Dual-MC1 causal position-sizing ablation\n",
        "- **Development selection:** 2025-03-01 through 2025-09-30.\n",
        "- **Untouched validation:** 2026-01-01 through 2026-07-31.\n",
        "- **Outcome:** source-aligned optimized parent-policy net bps; invalid paths excluded.\n",
        "- **Execution-cost variants:** intentionally excluded; `EVexec = EVrobust`.\n",
        "- **A14 active replacement:** not evaluated: the frozen outcome ledger does not contain a causal mark-to-market/close price at an arbitrary replacement timestamp. Treating the original terminal policy outcome as an immediate exit would be leakage.\n\n",
        "## Final arm metrics\n\n",
        _plain_table(metrics.loc[metrics["scope"].isin(["2025_full", "2026_validation", "full_2025_2026"])]),
        "\n\n## 2025 development HPO winners\n\n",
        _plain_table(hpo.sort_values(["stage", "selection_score", "net_pnl"], ascending=[True, False, False]).groupby("stage", as_index=False).head(1)),
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    manifest = {
        "schema": "strict_r3_dual_mc1_causal_position_sizing_ablation_v1",
        "status": "complete",
        "purpose": "offline sequential sizing-only research; live stack untouched",
        "inputs": {
            "bcf_predictions": {"path": str(args.bcf_predictions), "sha256": _sha(args.bcf_predictions)},
            "current_predictions": {"path": str(args.current_predictions), "sha256": _sha(args.current_predictions)},
        },
        "population": "common BCF/current score identities with valid source-aligned optimized policy outcomes; invalid paths excluded before portfolio allocation; gross exposure marked only from the current common-score intersection, with last-causal-mark carry-forward when an open symbol is absent",
        "selection": {"development_start": dev_start.isoformat(), "development_end_exclusive": dev_end.isoformat(), "objective": "maximise sequential Calmar-like return/drawdown, then net PnL, then weighted net EV"},
        "validation": {"start": val_start.isoformat(), "end_exclusive": val_end.isoformat(), "frozen_after_2025": True},
        "sizing": {"weights": [MIN_WEIGHT, MAX_WEIGHT], "max_gross_nav": MAX_GROSS_WEIGHT, "max_new_per_hour": MAX_NEW_PER_HOUR, "max_concurrent": MAX_CONCURRENT, "efull0": "monthly p90 of prior target-free robust EV through the 2025 development cutoff, then frozen; minimum 1,000 earlier rows; February 2025 warm-up"},
        "execution_cost_ablation": "not run; robust mapped net EV is used directly as EVexec",
        "shadow_ev": "open-position entry mapped EV proxy only; remaining EV is not available in this ledger",
        "not_run": {"A14_active_incumbent_replacement": "requires causal arbitrary-timestamp mark-to-market and realised close paths, absent from frozen parent-policy outcomes"},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-predictions", type=Path, default=DEFAULT_BCF)
    parser.add_argument("--current-predictions", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--development-start", default="2025-03-01")
    parser.add_argument("--development-end", default="2025-10-01", help="Exclusive; this is the 2025 tuning interval.")
    parser.add_argument("--validation-start", default="2026-01-01")
    parser.add_argument("--validation-end", default="2026-08-01")
    parser.add_argument("--finalize-existing", action="store_true", help="Write only the terminal report/manifest from completed numeric artifacts.")
    args = parser.parse_args()
    dev_start, dev_end = _utc(args.development_start), _utc(args.development_end)
    val_start, val_end = _utc(args.validation_start), _utc(args.validation_end)
    if not (dev_start < dev_end <= val_start < val_end):
        raise ValueError("require development < validation chronology")
    if args.out_dir.exists():
        if not args.finalize_existing:
            raise FileExistsError(f"immutable output exists: {args.out_dir}")
        _write_terminal_artifacts(
            args,
            hpo=pd.read_parquet(args.out_dir / "hpo_trials_2025_development.parquet"),
            metrics=pd.read_parquet(args.out_dir / "portfolio_metrics.parquet"),
            dev_start=dev_start, dev_end=dev_end, val_start=val_start, val_end=val_end,
        )
        print(args.out_dir)
        return
    args.out_dir.mkdir(parents=True)
    panel = _read_panel(args.bcf_predictions, args.current_predictions)
    chosen, hpo = _funnel(panel, dev_start=dev_start, dev_end=dev_end)
    hpo.to_parquet(args.out_dir / "hpo_trials_2025_development.parquet", index=False, compression="zstd")
    hpo.to_csv(args.out_dir / "hpo_trials_2025_development.csv", index=False)
    (args.out_dir / "selected_params.json").write_text(json.dumps({k: asdict(v) for k, v in chosen.items()}, indent=2, sort_keys=True) + "\n")

    all_metrics: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    decisions_root = args.out_dir / "decisions"
    decisions_root.mkdir()
    # 2025 starts in March because February is the causal Efull0 p90 warm-up.
    periods = {
        "2025_development": (dev_start, dev_end),
        "2025_holdout": (dev_end, _utc("2026-01-01")),
        "2025_full": (dev_start, _utc("2026-01-01")),
        "2026_validation": (val_start, val_end),
        "full_2025_2026": (dev_start, val_end),
    }
    for arm, params in chosen.items():
        for scope, (start, end) in periods.items():
            metric, trade, state, exits = _simulate(
                panel, params, start=start, end=end, arm=arm, capture=True,
                p90_freeze_after=dev_end,
            )
            metric["scope"] = scope
            all_metrics.append(metric)
            trade.to_parquet(decisions_root / f"{arm}_{scope}_trades.parquet", index=False, compression="zstd")
            state.to_parquet(decisions_root / f"{arm}_{scope}_state.parquet", index=False, compression="zstd")
            exits.to_parquet(decisions_root / f"{arm}_{scope}_exits.parquet", index=False, compression="zstd")
            month = _monthly(trade, arm)
            month["scope"] = scope
            monthly.append(month)
            diag = _diagnostics(trade, arm)
            if not diag.empty:
                diag["scope"] = scope
                diagnostics.append(diag)
        print(json.dumps({"event": "arm_complete", "arm": arm}), flush=True)
    metrics = pd.DataFrame(all_metrics)
    baseline = metrics.loc[metrics["arm"].eq("A0_fixed_dual30")].set_index("scope")
    for field in ("net_pnl", "net_portfolio_return", "weighted_net_ev_bps", "max_drawdown", "sortino_daily", "trades", "days_lt_1_trade", "days_lt_5_trades"):
        metrics[f"delta_vs_a0_{field}"] = metrics.apply(
            lambda row: row[field] - baseline.loc[row["scope"], field] if row["scope"] in baseline.index else float("nan"), axis=1
        )
    metrics.to_parquet(args.out_dir / "portfolio_metrics.parquet", index=False, compression="zstd")
    metrics.to_csv(args.out_dir / "portfolio_metrics.csv", index=False)
    pd.concat(monthly, ignore_index=True).to_parquet(args.out_dir / "monthly_metrics.parquet", index=False, compression="zstd")
    pd.concat(monthly, ignore_index=True).to_csv(args.out_dir / "monthly_metrics.csv", index=False)
    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_parquet(args.out_dir / "sizing_diagnostics.parquet", index=False, compression="zstd")
    _write_terminal_artifacts(
        args, hpo=hpo, metrics=metrics, dev_start=dev_start, dev_end=dev_end,
        val_start=val_start, val_end=val_end,
    )
    print(args.out_dir)


if __name__ == "__main__":
    main()
