"""Executable favorable pyramiding on top of frozen one-minute exits.

This module deliberately separates entry scheduling from exit optimization.  A
call receives the winner's already-frozen executable exit bar and price, then
simulates favorable adds using only completed one-minute closes.  It is suited
to repeated optimizer evaluations because the path loop is a parallel Numba
kernel and all outputs are dense NumPy arrays.

Execution contract
------------------

* The original position fills at ``opens0`` plus the entry half-spread and is
  represented by fill bar ``-1`` (the open immediately before path bar zero).
* An add is considered only on a completed close strictly before the frozen
  exit bar, requires strictly positive volume, and fills at that close plus the
  side-correct half-spread.
* Consecutive fills are at least ``minimum_bars_between_fills`` apart.  With the
  initial fill at bar ``-1``, the default of five permits the first add at bar
  four, five elapsed minutes after entry.
* At most one tranche fills per bar.  After a fill, the next trigger is rebuilt
  from that fill's actual raw close; a gap cannot backfill skipped levels.
* The favorable gap is the maximum of ``atr_multiplier * entry_ATR`` and the
  last raw fill times ``max(full_spread_bps, minimum_gap_bps)``.  Entry ATR is
  frozen and causal.
* Exits never move: every filled tranche closes at ``frozen_exit_price``.

Returns are expressed per tranche and as contributions relative to the
original planned target.  Tranche weights are normalized to one and then
multiplied by ``exposure_multiple``; values above one explicitly permit total
exposure beyond the original target.  Each filled tranche pays the entry and
exit fee once.  The supplied frozen exit price is assumed to include its
side-correct exit spread already.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit, prange

MAX_TRANCHES = 8
DEFAULT_MINIMUM_BARS_BETWEEN_FILLS = 5
DEFAULT_MINIMUM_GAP_BPS = 50.0

# Fill-bar sentinels.  Path bars start at zero; the initial open precedes them.
INITIAL_FILL_BAR = -1
UNFILLED_BAR = -2


@njit(cache=True, parallel=True)
def simulate_executable_pyramiding_kernel(
    row_index: np.ndarray,
    opens0: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    side_all: np.ndarray,
    atr_frac_all: np.ndarray,
    entry_half_spread_bps_all: np.ndarray,
    full_spread_bps_all: np.ndarray,
    frozen_exit_bar_all: np.ndarray,
    frozen_exit_price_all: np.ndarray,
    target_weights: np.ndarray,
    atr_multiplier: float,
    fee_per_side: float,
    minimum_bars_between_fills: int = DEFAULT_MINIMUM_BARS_BETWEEN_FILLS,
    minimum_gap_bps: float = DEFAULT_MINIMUM_GAP_BPS,
) -> tuple[np.ndarray, ...]:
    """Numba kernel for completed-close, frozen-exit favorable pyramiding.

    Inputs ending in ``_all`` use global candidate indexing; ``row_index``
    chooses the candidates to replay.  ``closes`` and ``volumes`` have shape
    ``(n_candidates, n_path_bars)``.  ``target_weights`` is already normalized
    and exposure-scaled by :func:`simulate_executable_pyramiding`.

    The tuple contains, in order: ``fill_bars``, ``fill_raw_prices``,
    ``fill_exec_prices``, ``trigger_raw_prices``, ``tranche_gross_returns``,
    ``tranche_net_returns``, aggregate gross and net return contributions,
    filled tranche count, filled exposure, weighted average executable entry,
    completed bars inspected, price-trigger bars rejected for zero volume,
    price-trigger bars rejected by the time gate, exit-bar trigger collisions,
    aggregate favorable overshoot in bps, effective initial gap in bps, and an
    order-valid flag.
    """

    n = len(row_index)
    tranche_count = len(target_weights)
    horizon = closes.shape[1]

    fill_bars = np.full((n, MAX_TRANCHES), UNFILLED_BAR, dtype=np.int32)
    fill_raw_prices = np.full((n, MAX_TRANCHES), np.nan, dtype=np.float64)
    fill_exec_prices = np.full((n, MAX_TRANCHES), np.nan, dtype=np.float64)
    trigger_raw_prices = np.full((n, MAX_TRANCHES), np.nan, dtype=np.float64)
    tranche_gross = np.full((n, MAX_TRANCHES), np.nan, dtype=np.float64)
    tranche_net = np.full((n, MAX_TRANCHES), np.nan, dtype=np.float64)
    aggregate_gross = np.full(n, np.nan, dtype=np.float64)
    aggregate_net = np.full(n, np.nan, dtype=np.float64)
    filled_count = np.zeros(n, dtype=np.int8)
    filled_exposure = np.zeros(n, dtype=np.float64)
    average_exec_entry = np.full(n, np.nan, dtype=np.float64)
    bars_inspected = np.zeros(n, dtype=np.int32)
    zero_volume_rejections = np.zeros(n, dtype=np.int32)
    time_gate_rejections = np.zeros(n, dtype=np.int32)
    exit_bar_collisions = np.zeros(n, dtype=np.int8)
    favorable_overshoot_bps = np.zeros(n, dtype=np.float64)
    effective_initial_gap_bps = np.full(n, np.nan, dtype=np.float64)
    order_valid = np.ones(n, dtype=np.bool_)

    spacing_atr = max(float(atr_multiplier), 0.0)
    min_bars = max(int(minimum_bars_between_fills), 1)
    hard_floor_bps = max(float(minimum_gap_bps), 0.0)

    for out_i in prange(n):
        i = int(row_index[out_i])
        raw_entry = float(opens0[i])
        side = 1.0 if float(side_all[i]) >= 0.0 else -1.0
        exit_bar = int(frozen_exit_bar_all[i])
        exit_price = float(frozen_exit_price_all[i])
        if (
            not np.isfinite(raw_entry)
            or raw_entry <= 0.0
            or not np.isfinite(exit_price)
            or exit_price <= 0.0
            or exit_bar < 0
            or exit_bar >= horizon
            or tranche_count < 1
            or tranche_count > MAX_TRANCHES
        ):
            order_valid[out_i] = False
            continue

        half_spread = max(float(entry_half_spread_bps_all[i]), 0.0) / 10_000.0
        spread_floor_bps = max(float(full_spread_bps_all[i]), hard_floor_bps)
        atr_abs = raw_entry * max(float(atr_frac_all[i]), 0.0)

        initial_exec = raw_entry * (1.0 + side * half_spread)
        fill_bars[out_i, 0] = INITIAL_FILL_BAR
        fill_raw_prices[out_i, 0] = raw_entry
        fill_exec_prices[out_i, 0] = initial_exec
        filled_count[out_i] = 1

        last_raw_fill = raw_entry
        last_fill_bar = INITIAL_FILL_BAR
        next_tranche = 1
        initial_gap = max(
            spacing_atr * atr_abs,
            raw_entry * spread_floor_bps / 10_000.0,
        )
        effective_initial_gap_bps[out_i] = initial_gap / raw_entry * 10_000.0
        if tranche_count > 1:
            trigger_raw_prices[out_i, 1] = raw_entry + side * initial_gap

        # Adds are forbidden on the frozen exit bar, irrespective of whether
        # the exit happened at its open, intrabar, or close.
        for j in range(exit_bar):
            close = float(closes[i, j])
            volume = float(volumes[i, j])
            if not np.isfinite(close) or close <= 0.0:
                continue
            bars_inspected[out_i] += 1
            if next_tranche >= tranche_count:
                continue

            gap_abs = max(
                spacing_atr * atr_abs,
                last_raw_fill * spread_floor_bps / 10_000.0,
            )
            trigger = last_raw_fill + side * gap_abs
            trigger_raw_prices[out_i, next_tranche] = trigger
            favorable = close >= trigger if side > 0.0 else close <= trigger
            if not favorable:
                continue
            if j - last_fill_bar < min_bars:
                time_gate_rejections[out_i] += 1
                continue
            if not np.isfinite(volume) or volume <= 0.0:
                zero_volume_rejections[out_i] += 1
                continue

            # Fill only this tranche at the observable close.  Resetting the
            # anchor to this raw close enforces both one-add-per-bar and no
            # backfill after a jump through multiple theoretical levels.
            raw_fill = close
            exec_fill = raw_fill * (1.0 + side * half_spread)
            fill_bars[out_i, next_tranche] = j
            fill_raw_prices[out_i, next_tranche] = raw_fill
            fill_exec_prices[out_i, next_tranche] = exec_fill
            favorable_move = side * (raw_fill - last_raw_fill)
            favorable_overshoot_bps[out_i] += max(
                favorable_move - gap_abs, 0.0
            ) / last_raw_fill * 10_000.0
            last_raw_fill = raw_fill
            last_fill_bar = j
            next_tranche += 1
            filled_count[out_i] += 1
            if next_tranche < tranche_count:
                next_gap = max(
                    spacing_atr * atr_abs,
                    last_raw_fill * spread_floor_bps / 10_000.0,
                )
                trigger_raw_prices[out_i, next_tranche] = (
                    last_raw_fill + side * next_gap
                )

        # Record a trigger that would otherwise have collided with the frozen
        # exit.  It remains diagnostic only and can never fill.
        if next_tranche < tranche_count:
            exit_close = float(closes[i, exit_bar])
            exit_volume = float(volumes[i, exit_bar])
            if np.isfinite(exit_close) and exit_close > 0.0:
                gap_abs = max(
                    spacing_atr * atr_abs,
                    last_raw_fill * spread_floor_bps / 10_000.0,
                )
                trigger = last_raw_fill + side * gap_abs
                favorable = exit_close >= trigger if side > 0.0 else exit_close <= trigger
                if (
                    favorable
                    and exit_bar - last_fill_bar >= min_bars
                    and np.isfinite(exit_volume)
                    and exit_volume > 0.0
                ):
                    exit_bar_collisions[out_i] = 1

        gross_total = 0.0
        net_total = 0.0
        exposure_total = 0.0
        weighted_entry_total = 0.0
        for tranche in range(int(filled_count[out_i])):
            fill = fill_exec_prices[out_i, tranche]
            weight = float(target_weights[tranche])
            gross = side * (exit_price / fill - 1.0)
            # Entry fee is paid on entry notional; exit fee is paid on exit
            # notional.  Exit spread must already be in frozen_exit_price.
            net = gross - fee_per_side - fee_per_side * (1.0 + gross)
            tranche_gross[out_i, tranche] = gross
            tranche_net[out_i, tranche] = net
            gross_total += weight * gross
            net_total += weight * net
            exposure_total += weight
            weighted_entry_total += weight * fill

        aggregate_gross[out_i] = gross_total
        aggregate_net[out_i] = net_total
        filled_exposure[out_i] = exposure_total
        average_exec_entry[out_i] = weighted_entry_total / max(exposure_total, 1e-12)

    return (
        fill_bars,
        fill_raw_prices,
        fill_exec_prices,
        trigger_raw_prices,
        tranche_gross,
        tranche_net,
        aggregate_gross,
        aggregate_net,
        filled_count,
        filled_exposure,
        average_exec_entry,
        bars_inspected,
        zero_volume_rejections,
        time_gate_rejections,
        exit_bar_collisions,
        favorable_overshoot_bps,
        effective_initial_gap_bps,
        order_valid,
    )


def simulate_executable_pyramiding(
    row_index: np.ndarray,
    opens0: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    side_all: np.ndarray,
    atr_frac_all: np.ndarray,
    entry_half_spread_bps_all: np.ndarray,
    full_spread_bps_all: np.ndarray,
    frozen_exit_bar_all: np.ndarray,
    frozen_exit_price_all: np.ndarray,
    tranche_weights: np.ndarray,
    atr_multiplier: float,
    fee_per_side: float,
    *,
    exposure_multiple: float = 1.0,
    minimum_bars_between_fills: int = DEFAULT_MINIMUM_BARS_BETWEEN_FILLS,
    minimum_gap_bps: float = DEFAULT_MINIMUM_GAP_BPS,
) -> dict[str, Any]:
    """Validate inputs and replay an executable favorable-pyramiding schedule.

    Parameters
    ----------
    tranche_weights:
        One positive-or-zero relative weight per tranche, including the initial
        tranche.  Between one and eight values are accepted; the first must be
        positive and at least one weight must be positive.  Weights are
        normalized internally.
    exposure_multiple:
        Total target exposure relative to the winner's original planned size.
        ``1.0`` is exposure-neutral; values above one intentionally permit
        pyramiding beyond that target.  Only filled tranche weights contribute
        to actual exposure.
    atr_multiplier:
        Favorable distance in multiples of the causal entry ATR.  The effective
        distance is never below the full-spread/``minimum_gap_bps`` floor.

    Returns
    -------
    dict
        Dense row-by-tranche fill and return arrays, aggregate return
        contributions, execution diagnostics, and the normalized/scaled weight
        vectors used by the kernel.  Unfilled bars are ``UNFILLED_BAR`` and
        unfilled prices/returns are NaN; the initial fill bar is
        ``INITIAL_FILL_BAR``.
    """

    idx = np.asarray(row_index, dtype=np.int64)
    weights = np.asarray(tranche_weights, dtype=np.float64)
    if weights.ndim != 1 or not 1 <= len(weights) <= MAX_TRANCHES:
        raise ValueError("tranche_weights must contain between 1 and 8 values")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("tranche_weights must be finite and non-negative")
    if weights[0] <= 0.0 or float(weights.sum()) <= 0.0:
        raise ValueError("the initial tranche and total tranche weight must be positive")
    if not np.isfinite(exposure_multiple) or exposure_multiple <= 0.0:
        raise ValueError("exposure_multiple must be finite and positive")
    if not np.isfinite(atr_multiplier) or atr_multiplier < 0.0:
        raise ValueError("atr_multiplier must be finite and non-negative")
    if not np.isfinite(fee_per_side) or fee_per_side < 0.0:
        raise ValueError("fee_per_side must be finite and non-negative")
    if int(minimum_bars_between_fills) < 1:
        raise ValueError("minimum_bars_between_fills must be at least one")
    if not np.isfinite(minimum_gap_bps) or minimum_gap_bps < 0.0:
        raise ValueError("minimum_gap_bps must be finite and non-negative")

    opens = np.asarray(opens0)
    close_path = np.asarray(closes)
    volume_path = np.asarray(volumes)
    if close_path.ndim != 2 or volume_path.shape != close_path.shape:
        raise ValueError("closes and volumes must be matching two-dimensional arrays")
    n_all = len(opens)
    one_dimensional = {
        "side_all": side_all,
        "atr_frac_all": atr_frac_all,
        "entry_half_spread_bps_all": entry_half_spread_bps_all,
        "full_spread_bps_all": full_spread_bps_all,
        "frozen_exit_bar_all": frozen_exit_bar_all,
        "frozen_exit_price_all": frozen_exit_price_all,
    }
    if close_path.shape[0] != n_all:
        raise ValueError("path row count must match opens0")
    for name, values in one_dimensional.items():
        if np.asarray(values).ndim != 1 or len(values) != n_all:
            raise ValueError(f"{name} must be one-dimensional with len(opens0)")
    if len(idx) and (int(idx.min()) < 0 or int(idx.max()) >= n_all):
        raise IndexError("row_index contains an out-of-range candidate index")

    normalized_weights = weights / weights.sum()
    target_weights = normalized_weights * float(exposure_multiple)
    names = (
        "fill_bars",
        "fill_raw_prices",
        "fill_exec_prices",
        "trigger_raw_prices",
        "tranche_gross_return",
        "tranche_net_return",
        "gross_return_contribution",
        "net_return_contribution",
        "filled_tranche_count",
        "filled_exposure_multiple",
        "average_exec_entry",
        "bars_inspected",
        "zero_volume_rejections",
        "time_gate_rejections",
        "exit_bar_collisions",
        "favorable_overshoot_bps",
        "effective_initial_gap_bps",
        "order_valid",
    )
    values = simulate_executable_pyramiding_kernel(
        idx,
        opens,
        close_path,
        volume_path,
        np.asarray(side_all),
        np.asarray(atr_frac_all),
        np.asarray(entry_half_spread_bps_all),
        np.asarray(full_spread_bps_all),
        np.asarray(frozen_exit_bar_all),
        np.asarray(frozen_exit_price_all),
        target_weights,
        float(atr_multiplier),
        float(fee_per_side),
        int(minimum_bars_between_fills),
        float(minimum_gap_bps),
    )
    result = dict(zip(names, values))
    result["normalized_tranche_weights"] = normalized_weights
    result["target_tranche_exposure_multiples"] = target_weights
    result["target_exposure_multiple"] = float(exposure_multiple)
    return result


__all__ = [
    "DEFAULT_MINIMUM_BARS_BETWEEN_FILLS",
    "DEFAULT_MINIMUM_GAP_BPS",
    "INITIAL_FILL_BAR",
    "MAX_TRANCHES",
    "UNFILLED_BAR",
    "simulate_executable_pyramiding",
    "simulate_executable_pyramiding_kernel",
]
