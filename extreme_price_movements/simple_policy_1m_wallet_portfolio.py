"""Exact one-minute wallet-capacity replay for simple-policy research.

The replay is deliberately size-aware: requested EV/Bayesian size is computed
before admission, open positions are marked from their causal one-minute close
paths, and gross marked quote notional may not exceed 80% of current equity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit


@njit(cache=True)
def _wallet_replay_kernel(
    timestamps_ns: np.ndarray,
    symbol_codes: np.ndarray,
    side: np.ndarray,
    entry_prices: np.ndarray,
    close_paths: np.ndarray,
    exit_bars: np.ndarray,
    net_returns: np.ndarray,
    requested_fractions: np.ndarray,
    bar_minutes: int,
    max_wallet_invested: float,
    max_new_per_bar: int,
    initial_wallet: float,
) -> tuple[np.ndarray, ...]:
    n = len(timestamps_ns)
    selected = np.zeros(n, dtype=np.bool_)
    admitted_notional = np.zeros(n, dtype=np.float64)
    wallet_before = np.full(n, np.nan, dtype=np.float64)
    equity_before = np.full(n, np.nan, dtype=np.float64)
    marked_before = np.full(n, np.nan, dtype=np.float64)
    utilization_before = np.full(n, np.nan, dtype=np.float64)
    rejection_code = np.zeros(n, dtype=np.int8)

    # In the candidate universe a practical upper bound of n slots avoids any
    # count-based admission assumption while preserving simple Numba storage.
    active_row = np.full(n, -1, dtype=np.int64)
    active_exit_ns = np.full(n, -1, dtype=np.int64)
    active_symbol = np.full(n, -1, dtype=np.int32)
    active_entry_notional = np.zeros(n, dtype=np.float64)
    active_quantity = np.zeros(n, dtype=np.float64)
    active_entry_price = np.zeros(n, dtype=np.float64)
    active_side = np.zeros(n, dtype=np.float64)
    active_net_return = np.zeros(n, dtype=np.float64)
    active_count = 0

    minute_ns = int(bar_minutes) * 60 * 1_000_000_000
    wallet = float(initial_wallet)
    current_ts = np.int64(-9223372036854775807)
    new_at_ts = 0

    for i in range(n):
        ts = timestamps_ns[i]
        if ts != current_ts:
            current_ts = ts
            new_at_ts = 0

        # Close due positions before marking/admitting this timestamp.
        write = 0
        for slot in range(active_count):
            if active_exit_ns[slot] <= ts:
                wallet += active_entry_notional[slot] * active_net_return[slot]
                continue
            if write != slot:
                active_row[write] = active_row[slot]
                active_exit_ns[write] = active_exit_ns[slot]
                active_symbol[write] = active_symbol[slot]
                active_entry_notional[write] = active_entry_notional[slot]
                active_quantity[write] = active_quantity[slot]
                active_entry_price[write] = active_entry_price[slot]
                active_side[write] = active_side[slot]
                active_net_return[write] = active_net_return[slot]
            write += 1
        active_count = write

        marked = 0.0
        unrealized = 0.0
        duplicate = False
        for slot in range(active_count):
            source = active_row[slot]
            elapsed = int((ts - timestamps_ns[source]) // minute_ns)
            path_idx = elapsed - 1
            mark = active_entry_price[slot]
            if path_idx >= 0:
                if path_idx >= close_paths.shape[1]:
                    path_idx = close_paths.shape[1] - 1
                candidate_mark = close_paths[source, path_idx]
                if np.isfinite(candidate_mark) and candidate_mark > 0.0:
                    mark = candidate_mark
            position_marked = active_quantity[slot] * mark
            marked += position_marked
            signed_return = active_side[slot] * (
                mark / max(active_entry_price[slot], 1e-12) - 1.0
            )
            unrealized += active_entry_notional[slot] * signed_return
            if active_symbol[slot] == symbol_codes[i]:
                duplicate = True

        equity = wallet + unrealized
        limit = max(float(max_wallet_invested), 0.0) * max(equity, 0.0)
        remaining = max(limit - marked, 0.0)
        wallet_before[i] = wallet
        equity_before[i] = equity
        marked_before[i] = marked
        utilization_before[i] = marked / max(limit, 1e-12) if limit > 0.0 else np.inf

        if exit_bars[i] < 0 or not np.isfinite(net_returns[i]):
            rejection_code[i] = 1  # invalid path
            continue
        if duplicate:
            rejection_code[i] = 2  # symbol already open
            continue
        if new_at_ts >= max_new_per_bar:
            rejection_code[i] = 3  # entry throttle
            continue
        if equity <= 0.0 or remaining <= 0.0:
            rejection_code[i] = 4  # wallet cap
            continue
        entry = entry_prices[i]
        if not np.isfinite(entry) or entry <= 0.0:
            rejection_code[i] = 1
            continue
        requested = max(requested_fractions[i], 0.0) * max(equity, 0.0)
        notional = min(requested, remaining)
        if notional <= 0.0:
            rejection_code[i] = 4
            continue

        selected[i] = True
        admitted_notional[i] = notional
        new_at_ts += 1
        slot = active_count
        active_row[slot] = i
        active_exit_ns[slot] = ts + (int(exit_bars[i]) + 1) * minute_ns
        active_symbol[slot] = symbol_codes[i]
        active_entry_notional[slot] = notional
        active_quantity[slot] = notional / entry
        active_entry_price[slot] = entry
        active_side[slot] = 1.0 if side[i] >= 0.0 else -1.0
        active_net_return[slot] = net_returns[i]
        active_count += 1

    # Realize remaining positions for the final bankroll diagnostic.
    for slot in range(active_count):
        wallet += active_entry_notional[slot] * active_net_return[slot]
    return (
        selected,
        admitted_notional,
        wallet_before,
        equity_before,
        marked_before,
        utilization_before,
        rejection_code,
        np.asarray([wallet], dtype=np.float64),
    )


def replay_marked_notional_wallet(
    *,
    timestamps_ns: np.ndarray,
    symbol_codes: np.ndarray,
    side: np.ndarray,
    raw_entry_prices: np.ndarray,
    entry_half_spread_bps: np.ndarray,
    close_paths: np.ndarray,
    exit_bars: np.ndarray,
    net_returns: np.ndarray,
    requested_fractions: np.ndarray,
    bar_minutes: int = 1,
    max_wallet_invested: float = 0.80,
    max_new_per_bar: int = 2,
    initial_wallet: float = 1.0,
) -> dict[str, Any]:
    """Replay size-aware admissions under gross marked-notional capacity."""
    side_arr = np.where(np.asarray(side, dtype=np.float64) >= 0.0, 1.0, -1.0)
    entry = np.asarray(raw_entry_prices, dtype=np.float64) * (
        1.0
        + side_arr
        * np.maximum(np.asarray(entry_half_spread_bps, dtype=np.float64), 0.0)
        / 10_000.0
    )
    result = _wallet_replay_kernel(
        np.asarray(timestamps_ns, dtype=np.int64),
        np.asarray(symbol_codes, dtype=np.int32),
        side_arr,
        entry,
        np.asarray(close_paths, dtype=np.float64),
        np.asarray(exit_bars, dtype=np.int32),
        np.asarray(net_returns, dtype=np.float64),
        np.asarray(requested_fractions, dtype=np.float64),
        int(bar_minutes),
        float(max_wallet_invested),
        int(max_new_per_bar),
        float(initial_wallet),
    )
    keys = (
        "selected",
        "admitted_notional",
        "wallet_before",
        "equity_before",
        "marked_notional_before",
        "wallet_cap_utilization_before",
        "rejection_code",
        "final_wallet_array",
    )
    out = {key: value for key, value in zip(keys, result)}
    out["final_wallet"] = float(out.pop("final_wallet_array")[0])
    return out


__all__ = ["replay_marked_notional_wallet"]
