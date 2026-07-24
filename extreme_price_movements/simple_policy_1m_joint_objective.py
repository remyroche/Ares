"""Portfolio-aware objective for exact one-minute exit-geometry searches."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.simple_policy_1m_wallet_portfolio import (
    replay_marked_notional_wallet,
)


def ev_bayesian_requested_fractions(
    corrected_ev_rank: np.ndarray,
    bayesian_multiplier: np.ndarray,
) -> np.ndarray:
    """Live-aligned base allocation followed by the raw Bayesian overlay."""
    rank = np.clip(np.asarray(corrected_ev_rank, dtype=np.float64), 0.0, 1.0)
    multiplier = np.asarray(bayesian_multiplier, dtype=np.float64)
    if rank.shape != multiplier.shape:
        raise ValueError("corrected_ev_rank and bayesian_multiplier must align")
    return (0.075 + 0.075 * np.power(rank, 1.1)) * np.clip(multiplier, 0.0, np.inf)


def priority_order(
    timestamps_ns: np.ndarray,
    corrected_ev: np.ndarray,
    corrected_ev_rank: np.ndarray,
) -> np.ndarray:
    """Chronological order, with corrected EV deciding simultaneous capacity."""
    ts = np.asarray(timestamps_ns, dtype=np.int64)
    ev = np.nan_to_num(np.asarray(corrected_ev, dtype=np.float64), nan=-np.inf)
    rank = np.nan_to_num(np.asarray(corrected_ev_rank, dtype=np.float64), nan=-np.inf)
    # lexsort uses the last key as primary.
    return np.lexsort((-rank, -ev, ts)).astype(np.int64)


def _max_drawdown_from_exit_events(
    timestamps_ns: np.ndarray,
    exit_bars: np.ndarray,
    admitted_notional: np.ndarray,
    net_returns: np.ndarray,
    *,
    bar_minutes: int,
    initial_wallet: float,
) -> float:
    selected = admitted_notional > 0.0
    if not np.any(selected):
        return 0.0
    minute_ns = int(bar_minutes) * 60 * 1_000_000_000
    exit_ts = timestamps_ns[selected] + (exit_bars[selected].astype(np.int64) + 1) * minute_ns
    pnl = admitted_notional[selected] * net_returns[selected]
    order = np.argsort(exit_ts, kind="stable")
    equity = float(initial_wallet) + np.cumsum(pnl[order])
    curve = np.concatenate(([float(initial_wallet)], equity))
    peak = np.maximum.accumulate(curve)
    return float(np.min((curve - peak) / np.maximum(peak, 1e-12)))


def evaluate_joint_wallet_objective(
    *,
    rows: pd.DataFrame,
    timestamps_ns: np.ndarray,
    symbol_codes: np.ndarray,
    side: np.ndarray,
    raw_entry_prices: np.ndarray,
    entry_half_spread_bps: np.ndarray,
    close_paths: np.ndarray,
    exit_bars: np.ndarray,
    net_returns: np.ndarray,
    corrected_ev: np.ndarray,
    corrected_ev_rank: np.ndarray,
    bayesian_multiplier: np.ndarray,
    holding_power: float = 0.8,
    holding_efficiency_weight: float = 0.10,
    max_wallet_invested: float = 0.80,
    max_new_per_bar: int = 2,
    initial_wallet: float = 1.0,
    bar_minutes: int = 1,
) -> tuple[float, dict[str, Any], dict[str, np.ndarray]]:
    """Replay actual sizes/capacity and score stability plus capital velocity.

    Weekly PnL is attributed to entry week, matching the existing policy search.
    Drawdown is computed on chronologically realized exit events.
    """
    order = priority_order(timestamps_ns, corrected_ev, corrected_ev_rank)
    requested = ev_bayesian_requested_fractions(corrected_ev_rank, bayesian_multiplier)
    replay = replay_marked_notional_wallet(
        timestamps_ns=np.asarray(timestamps_ns)[order],
        symbol_codes=np.asarray(symbol_codes)[order],
        side=np.asarray(side)[order],
        raw_entry_prices=np.asarray(raw_entry_prices)[order],
        entry_half_spread_bps=np.asarray(entry_half_spread_bps)[order],
        close_paths=np.asarray(close_paths)[order],
        exit_bars=np.asarray(exit_bars)[order],
        net_returns=np.asarray(net_returns)[order],
        requested_fractions=requested[order],
        bar_minutes=bar_minutes,
        max_wallet_invested=max_wallet_invested,
        max_new_per_bar=max_new_per_bar,
        initial_wallet=initial_wallet,
    )
    admitted_ordered = np.asarray(replay["admitted_notional"], dtype=np.float64)
    nonfinite_admitted = np.flatnonzero(~np.isfinite(admitted_ordered))
    if len(nonfinite_admitted):
        i = int(nonfinite_admitted[0])
        raise RuntimeError(
            "wallet replay produced non-finite admitted notional at ordered row "
            f"{i}: requested_fraction={requested[order][i]!r}, "
            f"entry={np.asarray(raw_entry_prices)[order][i]!r}, "
            f"entry_spread={np.asarray(entry_half_spread_bps)[order][i]!r}, "
            f"wallet_before={np.asarray(replay['wallet_before'])[i]!r}, "
            f"equity_before={np.asarray(replay['equity_before'])[i]!r}, "
            f"marked_before={np.asarray(replay['marked_notional_before'])[i]!r}"
        )
    admitted = np.zeros(len(order), dtype=np.float64)
    selected = np.zeros(len(order), dtype=bool)
    admitted[order] = admitted_ordered
    selected[order] = np.asarray(replay["selected"], dtype=bool)
    pnl = admitted * np.nan_to_num(np.asarray(net_returns, dtype=np.float64), nan=0.0)

    ts = pd.to_datetime(rows["timestamp"], utc=True)
    week = ts.dt.tz_localize(None).dt.to_period("W").astype(str).to_numpy()
    weekly = np.asarray([pnl[week == value].sum() for value in np.unique(week)])
    weekly_mean = float(weekly.mean()) if len(weekly) else 0.0
    weekly_std = float(weekly.std()) if len(weekly) else 0.0
    worst_week = float(weekly.min()) if len(weekly) else 0.0
    drawdown = _max_drawdown_from_exit_events(
        np.asarray(timestamps_ns), np.asarray(exit_bars), admitted,
        np.asarray(net_returns), bar_minutes=bar_minutes,
        initial_wallet=initial_wallet,
    )
    hours = np.maximum((np.asarray(exit_bars, dtype=np.float64) + 1.0) * bar_minutes / 60.0, 1.0 / 60.0)
    efficiency_pnl = float(np.sum(pnl / np.power(hours, float(holding_power))))
    stability = weekly_mean - 0.5 * weekly_std + 0.25 * worst_week - 0.10 * abs(drawdown)
    objective = float(stability + float(holding_efficiency_weight) * efficiency_pnl)
    chosen = np.flatnonzero(selected)
    metrics: dict[str, Any] = {
        "objective": objective,
        "stability_component": float(stability),
        "holding_efficiency_pnl": efficiency_pnl,
        "holding_power": float(holding_power),
        "holding_efficiency_weight": float(holding_efficiency_weight),
        "net_pnl_bankroll": float(pnl.sum()),
        "final_wallet": float(replay["final_wallet"]),
        "worst_week": worst_week,
        "weekly_mean": weekly_mean,
        "weekly_std": weekly_std,
        "max_drawdown": drawdown,
        "n_trades": int(len(chosen)),
        "net_ev_per_trade": float(np.mean(np.asarray(net_returns)[chosen])) if len(chosen) else np.nan,
        "size_weighted_net_ev": float(pnl.sum() / max(admitted.sum(), 1e-12)),
        "average_holding_minutes": float(np.mean((np.asarray(exit_bars)[chosen] + 1) * bar_minutes)) if len(chosen) else np.nan,
        "peak_wallet_utilization_before": float(np.nanmax(replay["wallet_cap_utilization_before"])) if len(order) else 0.0,
        "mean_admitted_notional": float(np.mean(admitted[chosen])) if len(chosen) else 0.0,
        "requested_notional_fraction_mean": float(np.mean(requested[chosen])) if len(chosen) else 0.0,
        "wallet_cap_rejections": int(np.sum(np.asarray(replay["rejection_code"]) == 4)),
        "symbol_rejections": int(np.sum(np.asarray(replay["rejection_code"]) == 2)),
        "entry_throttle_rejections": int(np.sum(np.asarray(replay["rejection_code"]) == 3)),
    }
    detail = {
        "selected": selected,
        "admitted_notional": admitted,
        "requested_fraction": requested,
        "pnl": pnl,
        "priority_order": order,
        "rejection_code_ordered": np.asarray(replay["rejection_code"]),
    }
    return objective, metrics, detail
