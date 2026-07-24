"""Exposure-neutral favorable-path pyramiding for the one-minute winner.

The position target is split into ``x`` equal tranches.  One tranche is
entered immediately and the remaining ``x - 1`` tranches are filled at
successive favorable-price levels.  Maximum exposure therefore remains the
original raw-Bayesian target.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from extreme_price_movements.simple_policy_1m_constrained import (
    P_ADV_ENABLED,
    P_ADV_FAST_MIN,
    P_ADV_MAX_MFE,
    P_ADV_MIN_MAE,
    P_ADV_MIN_SPEED,
    P_ADV_THETA,
    P_DECAY_HALF_MIN,
    P_DECAY_MIN_MULT,
    P_DECAY_START_MIN,
    P_SL,
    P_TRAIL_ACT,
    P_TRAIL_ACT_CAP_FRAC,
    P_TRAIL_BETA,
    P_TRAIL_DIV,
    P_TRAIL_POWER,
    REASON_ADVERSE,
    REASON_FULL_SL,
    REASON_TIMEOUT,
    REASON_TRAILING,
    _stop_fill,
)


SPACING_ABSOLUTE_FRACTION = 0
SPACING_ATR_MULTIPLE = 1

EXIT_ANCHOR_INITIAL = 0
EXIT_ANCHOR_WEIGHTED = 1


@njit(cache=True, parallel=True)
def simulate_reverse_dca_1m_paths(
    row_index: np.ndarray,
    opens0: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    side_all: np.ndarray,
    atr_frac_all: np.ndarray,
    entry_half_spread_bps_all: np.ndarray,
    exit_half_spread_bps_all: np.ndarray,
    params: np.ndarray,
    fee_per_side: float,
    stop_base_gap_bps: float,
    stop_through_fraction: float,
    stop_max_gap_bps: float,
    tranche_count: int,
    spacing_value: float,
    spacing_mode: int,
    exit_anchor_mode: int,
    add_before_exit_on_collision: bool,
) -> tuple[np.ndarray, ...]:
    """Replay favorable adds and the total-MFE trailing policy jointly.

    ``add_before_exit_on_collision`` selects the pessimistic exposure bound:
    all touched favorable levels fill before a same-minute stop.  Otherwise
    exit checks precede fills and a fill changes geometry from the next bar.

    The fast adverse guard remains anchored to the initial executable entry;
    in weighted mode the catastrophic and trailing geometry use the current
    weighted executable entry and their effective stop may only tighten.
    """

    n = len(row_index)
    exit_bar = np.full(n, -1, dtype=np.int32)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    gross_return = np.full(n, np.nan, dtype=np.float64)
    net_return = np.full(n, np.nan, dtype=np.float64)
    reason = np.full(n, REASON_TIMEOUT, dtype=np.int8)
    mfe_out = np.full(n, np.nan, dtype=np.float64)
    mae_out = np.full(n, np.nan, dtype=np.float64)
    filled_fraction = np.full(n, np.nan, dtype=np.float64)
    additions = np.zeros(n, dtype=np.int16)
    average_entry = np.full(n, np.nan, dtype=np.float64)
    last_level_distance = np.zeros(n, dtype=np.float64)
    first_add_bar = np.full(n, -1, dtype=np.int32)
    full_target_bar = np.full(n, -1, dtype=np.int32)
    geometry_mfe_atr = np.full(n, np.nan, dtype=np.float64)
    order_valid = np.ones(n, dtype=np.bool_)
    horizon = highs.shape[1]
    slices = max(int(tranche_count), 1)
    tranche_weight = 1.0 / slices
    max_additions = slices - 1
    spacing = max(float(spacing_value), 0.0)

    for out_i in prange(n):
        i = int(row_index[out_i])
        raw_entry = float(opens0[i])
        side = 1.0 if side_all[i] >= 0.0 else -1.0
        if not np.isfinite(raw_entry) or raw_entry <= 0.0:
            order_valid[out_i] = False
            continue

        atr = raw_entry * max(float(atr_frac_all[i]), 1e-6)
        entry_spread = max(float(entry_half_spread_bps_all[i]), 0.0) / 10_000.0
        initial_entry = raw_entry * (1.0 + side * entry_spread)
        weighted_entry_sum = tranche_weight * initial_entry
        weight_filled = tranche_weight
        geometry_entry = initial_entry
        next_level = 1

        sl_gap_atr = max(params[P_SL], 0.1)
        full_sl = geometry_entry - side * sl_gap_atr * atr
        effective_stop = full_sl
        trailing_stop = full_sl
        best_fav_price = initial_entry
        max_fav_initial = 0.0
        max_adv_initial = 0.0
        completed = False

        for j in range(horizon):
            high = float(highs[i, j])
            low = float(lows[i, j])
            close = float(closes[i, j])
            if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
                break

            if add_before_exit_on_collision:
                while next_level <= max_additions:
                    level_distance = (
                        spacing * raw_entry
                        if spacing_mode == SPACING_ABSOLUTE_FRACTION
                        else spacing * atr
                    ) * next_level
                    raw_trigger = raw_entry + side * level_distance
                    touched = high >= raw_trigger if side > 0.0 else low <= raw_trigger
                    if not touched:
                        break
                    fill = raw_trigger * (1.0 + side * entry_spread)
                    weighted_entry_sum += tranche_weight * fill
                    weight_filled += tranche_weight
                    additions[out_i] += 1
                    last_level_distance[out_i] = level_distance / raw_entry
                    if first_add_bar[out_i] < 0:
                        first_add_bar[out_i] = j
                    if next_level == max_additions:
                        full_target_bar[out_i] = j
                    next_level += 1

            if exit_anchor_mode == EXIT_ANCHOR_WEIGHTED:
                geometry_entry = weighted_entry_sum / max(weight_filled, 1e-12)
            else:
                geometry_entry = initial_entry

            # Re-anchoring after a favorable add can tighten the catastrophic
            # stop but can never loosen the already executable stop.
            full_sl = geometry_entry - side * sl_gap_atr * atr
            if side > 0.0:
                effective_stop = max(effective_stop, full_sl)
            else:
                effective_stop = min(effective_stop, full_sl)

            base_trail_act = max(params[P_TRAIL_ACT], 0.05) * atr
            if params[P_TRAIL_ACT_CAP_FRAC] > 0.0:
                base_trail_act = min(
                    base_trail_act, geometry_entry * params[P_TRAIL_ACT_CAP_FRAC]
                )
            trail_act = base_trail_act
            decay_half = max(params[P_DECAY_HALF_MIN], 0.0)
            decay_start = max(params[P_DECAY_START_MIN], 0.0)
            decay_min = min(max(params[P_DECAY_MIN_MULT], 0.01), 1.0)
            if decay_half > 0.0 and decay_min < 1.0 and j > decay_start:
                decay = 0.5 ** ((j - decay_start) / decay_half)
                trail_act *= decay_min + (1.0 - decay_min) * decay

            max_fav_geometry = (
                max(best_fav_price - geometry_entry, 0.0)
                if side > 0.0
                else max(geometry_entry - best_fav_price, 0.0)
            )
            trailing_armed = max_fav_geometry >= trail_act
            dynamic = (
                max_fav_geometry
                / max(atr * max(params[P_TRAIL_DIV], 0.05), 1e-12)
            ) ** max(params[P_TRAIL_POWER], 0.05)
            dynamic = min(max(dynamic, 0.0), 1.0)
            trail_gap = max(
                max_fav_geometry * max(params[P_TRAIL_BETA], 0.0) * (1.0 - dynamic),
                geometry_entry * 0.003,
            )
            trail_candidate = geometry_entry + side * max(
                max_fav_geometry - trail_gap, 0.0
            )
            if trailing_armed:
                if side > 0.0:
                    trailing_stop = max(trailing_stop, trail_candidate)
                    effective_stop = max(effective_stop, trailing_stop)
                else:
                    trailing_stop = min(trailing_stop, trail_candidate)
                    effective_stop = min(effective_stop, trailing_stop)

            exit_spread_bps = max(float(exit_half_spread_bps_all[i]), 0.0)
            trigger = (
                effective_stop / max(1.0 - exit_spread_bps / 10_000.0, 1e-12)
                if side > 0.0
                else effective_stop / (1.0 + exit_spread_bps / 10_000.0)
            )
            stop_hit = low <= trigger if side > 0.0 else high >= trigger
            if stop_hit:
                exit_bar[out_i] = j
                exit_price[out_i] = _stop_fill(
                    side,
                    effective_stop,
                    high,
                    low,
                    exit_spread_bps,
                    stop_base_gap_bps,
                    stop_through_fraction,
                    stop_max_gap_bps,
                )
                reason[out_i] = REASON_TRAILING if trailing_armed else REASON_FULL_SL
                completed = True
                break

            # Preserve the winner's frozen early adverse guard on the initial
            # signal entry; this avoids turning a favorable add into a new
            # synthetic adverse-state event.
            if params[P_ADV_ENABLED] > 0.5 and j <= params[P_ADV_FAST_MIN]:
                cur_fav_adv = (
                    max(high - initial_entry, 0.0)
                    if side > 0.0
                    else max(initial_entry - low, 0.0)
                )
                cur_mae_adv = (
                    max(initial_entry - low, 0.0)
                    if side > 0.0
                    else max(high - initial_entry, 0.0)
                )
                adv_mfe_atr = max(max_fav_initial, cur_fav_adv) / max(atr, 1e-12)
                adv_mae_atr = max(max_adv_initial, cur_mae_adv) / max(atr, 1e-12)
                elapsed_15m = max((j + 1) / 15.0, 1.0 / 15.0)
                speed = adv_mae_atr / elapsed_15m
                score = (
                    np.log1p(0.75)
                    + np.log1p(max(adv_mae_atr, 0.0))
                    + np.log1p(max(speed, 0.0))
                )
                eligible = (
                    adv_mae_atr >= params[P_ADV_MIN_MAE]
                    and speed >= params[P_ADV_MIN_SPEED]
                    and adv_mfe_atr <= params[P_ADV_MAX_MFE]
                )
                if eligible and score > params[P_ADV_THETA]:
                    exit_bar[out_i] = j
                    exit_price[out_i] = close * (
                        1.0 - side * exit_spread_bps / 10_000.0
                    )
                    reason[out_i] = REASON_ADVERSE
                    completed = True
                    break

            if not add_before_exit_on_collision:
                # Exit-first sensitivity bound: fills become active only after
                # every exit check on the candle and affect the next minute.
                while next_level <= max_additions:
                    level_distance = (
                        spacing * raw_entry
                        if spacing_mode == SPACING_ABSOLUTE_FRACTION
                        else spacing * atr
                    ) * next_level
                    raw_trigger = raw_entry + side * level_distance
                    touched = high >= raw_trigger if side > 0.0 else low <= raw_trigger
                    if not touched:
                        break
                    fill = raw_trigger * (1.0 + side * entry_spread)
                    weighted_entry_sum += tranche_weight * fill
                    weight_filled += tranche_weight
                    additions[out_i] += 1
                    last_level_distance[out_i] = level_distance / raw_entry
                    if first_add_bar[out_i] < 0:
                        first_add_bar[out_i] = j
                    if next_level == max_additions:
                        full_target_bar[out_i] = j
                    next_level += 1

            cur_fav = (
                max(high - initial_entry, 0.0)
                if side > 0.0
                else max(initial_entry - low, 0.0)
            )
            cur_adv = (
                max(initial_entry - low, 0.0)
                if side > 0.0
                else max(high - initial_entry, 0.0)
            )
            max_fav_initial = max(max_fav_initial, cur_fav)
            max_adv_initial = max(max_adv_initial, cur_adv)
            if side > 0.0:
                best_fav_price = max(best_fav_price, high)
            else:
                best_fav_price = min(best_fav_price, low)

        if not completed:
            last = -1
            for k in range(horizon - 1, -1, -1):
                if np.isfinite(closes[i, k]):
                    last = k
                    break
            if last < 0:
                order_valid[out_i] = False
                continue
            exit_spread = max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0
            exit_bar[out_i] = last
            exit_price[out_i] = float(closes[i, last]) * (1.0 - side * exit_spread)
            reason[out_i] = REASON_TIMEOUT

        gross = 0.0
        net = 0.0
        # Reconstruct deterministic tranche fills from the observed count.
        for level in range(int(additions[out_i]) + 1):
            if level == 0:
                fill = initial_entry
            else:
                level_distance = (
                    spacing * raw_entry
                    if spacing_mode == SPACING_ABSOLUTE_FRACTION
                    else spacing * atr
                ) * level
                fill = (raw_entry + side * level_distance) * (
                    1.0 + side * entry_spread
                )
            tranche_gross = side * (exit_price[out_i] / fill - 1.0)
            gross += tranche_weight * tranche_gross
            net += tranche_weight * (
                tranche_gross
                - fee_per_side
                - fee_per_side * (1.0 + tranche_gross)
            )

        gross_return[out_i] = gross
        net_return[out_i] = net
        filled_fraction[out_i] = weight_filled
        average_entry[out_i] = weighted_entry_sum / max(weight_filled, 1e-12)
        mfe_out[out_i] = max_fav_initial / initial_entry
        mae_out[out_i] = max_adv_initial / initial_entry
        geometry_mfe_atr[out_i] = (
            max(best_fav_price - geometry_entry, 0.0)
            if side > 0.0
            else max(geometry_entry - best_fav_price, 0.0)
        ) / max(atr, 1e-12)

    return (
        exit_bar,
        exit_price,
        gross_return,
        net_return,
        reason,
        mfe_out,
        mae_out,
        filled_fraction,
        additions,
        average_entry,
        last_level_distance,
        first_add_bar,
        full_target_bar,
        geometry_mfe_atr,
        order_valid,
    )
