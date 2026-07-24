"""Numba kernel for staged entry/DCA on frozen one-minute exit paths."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from extreme_price_movements.simple_policy_1m_constrained import REASON_TIMEOUT


@njit(cache=True, parallel=True)
def apply_dca_to_frozen_exits(
    row_index: np.ndarray,
    opens0: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    side_all: np.ndarray,
    entry_half_spread_bps_all: np.ndarray,
    exit_bar: np.ndarray,
    exit_price: np.ndarray,
    exit_reason: np.ndarray,
    fee_per_side: float,
    tranche_count: int,
    adverse_spacing_fraction: float,
    literal_additional_dcas: bool = False,
    dca_before_exit_on_collision: bool = False,
) -> tuple[np.ndarray, ...]:
    """Apply equal-sized adverse-price tranches while keeping exits frozen.

    Exposure-neutral mode uses ``tranche_count`` total slices: one at entry and
    at most ``tranche_count - 1`` additions.  Literal mode uses the same initial
    slice but permits ``tranche_count`` additions, so maximum exposure is
    ``1 + 1 / tranche_count`` times the original target.

    A stop/adverse/trailing exit wins a same-minute collision.  A timeout exits
    at the last close, so a DCA level touched during that final bar may fill.
    """

    n = len(row_index)
    gross_return_on_target = np.full(n, np.nan, dtype=np.float64)
    net_return_on_target = np.full(n, np.nan, dtype=np.float64)
    filled_fraction = np.full(n, np.nan, dtype=np.float64)
    additions = np.zeros(n, dtype=np.int16)
    average_entry = np.full(n, np.nan, dtype=np.float64)
    last_level_fraction = np.zeros(n, dtype=np.float64)
    raw_adverse_before_exit = np.zeros(n, dtype=np.float64)
    raw_adverse_including_exit = np.zeros(n, dtype=np.float64)

    slices = max(int(tranche_count), 1)
    tranche_weight = 1.0 / slices
    max_additions = slices if literal_additional_dcas else slices - 1
    spacing = max(float(adverse_spacing_fraction), 0.0)

    for out_i in prange(n):
        i = int(row_index[out_i])
        raw_entry = float(opens0[i])
        side = 1.0 if side_all[i] >= 0.0 else -1.0
        last_bar = int(exit_bar[out_i])
        final_price = float(exit_price[out_i])
        if (
            not np.isfinite(raw_entry)
            or raw_entry <= 0.0
            or last_bar < 0
            or not np.isfinite(final_price)
            or final_price <= 0.0
        ):
            continue

        spread = max(float(entry_half_spread_bps_all[i]), 0.0) / 10_000.0
        first_entry = raw_entry * (1.0 + side * spread)
        weighted_entry = tranche_weight * first_entry
        weight_filled = tranche_weight
        gross = tranche_weight * side * (final_price / first_entry - 1.0)
        first_gross = side * (final_price / first_entry - 1.0)
        net = tranche_weight * (
            first_gross - fee_per_side - fee_per_side * (1.0 + first_gross)
        )

        next_level = 1
        inclusive_last = (
            int(exit_reason[out_i]) == REASON_TIMEOUT
            or dca_before_exit_on_collision
        )
        bar_limit = last_bar + 1 if inclusive_last else last_bar
        for j in range(max(bar_limit, 0)):
            if next_level > max_additions:
                break
            high = float(highs[i, j])
            low = float(lows[i, j])
            if not (np.isfinite(high) and np.isfinite(low)):
                break
            while next_level <= max_additions:
                level_fraction = spacing * next_level
                raw_trigger = raw_entry * (1.0 - side * level_fraction)
                touched = low <= raw_trigger if side > 0.0 else high >= raw_trigger
                if not touched:
                    break
                fill = raw_trigger * (1.0 + side * spread)
                tranche_gross = side * (final_price / fill - 1.0)
                gross += tranche_weight * tranche_gross
                net += tranche_weight * (
                    tranche_gross
                    - fee_per_side
                    - fee_per_side * (1.0 + tranche_gross)
                )
                weighted_entry += tranche_weight * fill
                weight_filled += tranche_weight
                additions[out_i] += 1
                last_level_fraction[out_i] = level_fraction
                next_level += 1

        for j in range(last_bar + 1):
            high = float(highs[i, j])
            low = float(lows[i, j])
            if not (np.isfinite(high) and np.isfinite(low)):
                break
            adverse = (
                max(raw_entry - low, 0.0) / raw_entry
                if side > 0.0
                else max(high - raw_entry, 0.0) / raw_entry
            )
            if j < last_bar:
                raw_adverse_before_exit[out_i] = max(
                    raw_adverse_before_exit[out_i], adverse
                )
            raw_adverse_including_exit[out_i] = max(
                raw_adverse_including_exit[out_i], adverse
            )

        gross_return_on_target[out_i] = gross
        net_return_on_target[out_i] = net
        filled_fraction[out_i] = weight_filled
        average_entry[out_i] = weighted_entry / max(weight_filled, 1e-12)

    return (
        gross_return_on_target,
        net_return_on_target,
        filled_fraction,
        additions,
        average_entry,
        last_level_fraction,
        raw_adverse_before_exit,
        raw_adverse_including_exit,
    )
