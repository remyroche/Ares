"""Constrained one-minute capital/trailing research kernel.

The parameterisation makes the intended ordering structural:

* capital protection is inside the catastrophic full stop at entry;
* the capital gap is the shadow trailing gap plus a positive excess;
* trailing takes control only when armed and at least as protective;
* the effective stop never loosens at handover.

This module is research-only and deliberately separate from live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numba import njit, prange

FAMILY_TRAILING_ONLY = 0
FAMILY_CONSTANT = 1
FAMILY_MULTILAYER = 2
FAMILY_SIGMOID = 3
FAMILY_EXPONENTIAL = 4
FAMILY_RATIONAL = 5
FAMILY_SPLINE = 6

FAMILY_NAMES = {
    FAMILY_TRAILING_ONLY: "trailing_only",
    FAMILY_CONSTANT: "constant_relative",
    FAMILY_MULTILAYER: "multilayer_mixture",
    FAMILY_SIGMOID: "sigmoid_relative",
    FAMILY_EXPONENTIAL: "exponential_relative",
    FAMILY_RATIONAL: "rational_relative",
    FAMILY_SPLINE: "monotone_spline",
}

ACTIVATION_CURVE_TOTAL_MFE = 0
ACTIVATION_CURVE_POST_ACTIVATION = 1
ACTIVATION_CURVE_BLENDED = 2

REASON_TIMEOUT = 0
REASON_FULL_SL = 1
REASON_CAPITAL = 2
REASON_TRAILING = 3
REASON_ADVERSE = 4


@dataclass(frozen=True)
class ConstrainedReplaySpec:
    timeframe: str = "1m"
    bar_minutes: int = 1
    horizon_minutes: int = 1_440
    fee_per_side: float = 0.005
    stop_base_gap_bps: float = 15.0
    stop_through_fraction: float = 0.05
    stop_max_gap_bps: float = 75.0
    capital_trail_epsilon_atr: float = 0.02

    @property
    def path_len(self) -> int:
        return self.horizon_minutes // self.bar_minutes


# Compact parameter vector for Numba.
P_SL = 0
P_TRAIL_ACT = 1
P_TRAIL_POWER = 2
P_TRAIL_DIV = 3
P_TRAIL_BETA = 4
P_ENTRY_RATIO = 5
P_TERMINAL_RATIO = 6
P_CENTER = 7
P_SHAPE = 8
P_EXTRA_1 = 9
P_EXTRA_2 = 10
P_EXCESS_MIN_RATIO = 11
P_EXCESS_MAX_RATIO = 12
P_CURRENT_SL_RATIO = 13
P_SPLINE_1 = 14
P_SPLINE_2 = 15
P_SPLINE_3 = 16
P_SPLINE_4 = 17
P_SPLINE_5 = 18
P_DECAY_HALF_MIN = 19
P_DECAY_START_MIN = 20
P_DECAY_MIN_MULT = 21
P_ADV_ENABLED = 22
P_ADV_MIN_MAE = 23
P_ADV_MIN_SPEED = 24
P_ADV_THETA = 25
P_ADV_FAST_MIN = 26
P_ADV_MAX_MFE = 27
P_TRAIL_ACT_CAP_FRAC = 28
P_TRAIL_LAYER_COUNT = 29
P_TRAIL_ACT_2 = 30
P_TRAIL_ACT_3 = 31
P_TRAIL_BETA_2 = 32
P_TRAIL_BETA_3 = 33
N_PARAMS = 34


def constrained_params_to_vector(params: Mapping[str, Any]) -> np.ndarray:
    out = np.zeros(N_PARAMS, dtype=np.float64)
    out[P_SL] = float(params.get("sl_mult", 2.5))
    out[P_TRAIL_ACT] = float(params.get("trailing_activation_mult", 1.5))
    out[P_TRAIL_POWER] = float(params.get("trailing_power", 1.5))
    out[P_TRAIL_DIV] = float(params.get("trailing_squash_divisor", 2.0))
    out[P_TRAIL_BETA] = float(params.get("giveback_beta", 0.5))
    out[P_ENTRY_RATIO] = float(params.get("entry_capital_ratio", 0.75))
    out[P_TERMINAL_RATIO] = float(params.get("terminal_excess_ratio", 0.3))
    out[P_CENTER] = float(params.get("transition_center", 2.0))
    out[P_SHAPE] = float(params.get("transition_shape", 1.2))
    out[P_EXTRA_1] = float(params.get("mixture_logit_1", 0.0))
    out[P_EXTRA_2] = float(params.get("mixture_logit_2", 0.0))
    out[P_EXCESS_MIN_RATIO] = float(params.get("excess_min_ratio", 0.0))
    out[P_EXCESS_MAX_RATIO] = float(params.get("excess_max_ratio", 1e6))
    out[P_CURRENT_SL_RATIO] = float(params.get("current_distance_sl_ratio", 0.0))
    retains = params.get("spline_retains", (0.85, 0.70, 0.55, 0.40, 0.25))
    for offset, value in enumerate(retains):
        out[P_SPLINE_1 + offset] = float(value)
    out[P_DECAY_HALF_MIN] = float(params.get("trailing_activation_decay_half_life_minutes", 0.0))
    out[P_DECAY_START_MIN] = float(params.get("trailing_activation_decay_start_minutes", 0.0))
    out[P_DECAY_MIN_MULT] = float(params.get("trailing_activation_min_mult", 1.0))
    out[P_ADV_ENABLED] = float(bool(params.get("adverse_exit_enabled", False)))
    out[P_ADV_MIN_MAE] = float(params.get("adverse_exit_min_mae_atr", 1.0))
    out[P_ADV_MIN_SPEED] = float(params.get("adverse_exit_min_speed_per_15m", 0.3))
    out[P_ADV_THETA] = float(params.get("adverse_exit_theta", 1e9))
    out[P_ADV_FAST_MIN] = float(params.get("adverse_exit_fast_minutes", 0.0))
    out[P_ADV_MAX_MFE] = float(params.get("adverse_exit_max_mfe_atr", 0.25))
    out[P_TRAIL_ACT_CAP_FRAC] = float(params.get("trailing_activation_cap_pct", 0.0))
    out[P_TRAIL_LAYER_COUNT] = float(params.get("trailing_layer_count", 0))
    out[P_TRAIL_ACT_2] = float(
        params.get("trailing_activation_mult_2", out[P_TRAIL_ACT])
    )
    out[P_TRAIL_ACT_3] = float(
        params.get("trailing_activation_mult_3", out[P_TRAIL_ACT_2])
    )
    out[P_TRAIL_BETA_2] = float(params.get("giveback_beta_2", out[P_TRAIL_BETA]))
    out[P_TRAIL_BETA_3] = float(params.get("giveback_beta_3", out[P_TRAIL_BETA_2]))
    return out


@njit(cache=True, inline="always")
def _trail_gap_atr(
    u: float, entry: float, atr: float, params: np.ndarray, beta_override: float = -1.0
) -> float:
    power = max(params[P_TRAIL_POWER], 0.05)
    divisor = max(params[P_TRAIL_DIV], 0.05)
    beta = max(
        params[P_TRAIL_BETA] if beta_override < 0.0 else beta_override,
        0.0,
    )
    dynamic = min(max((max(u, 0.0) / divisor) ** power, 0.0), 1.0)
    return max(max(u, 0.0) * beta * (1.0 - dynamic), entry * 0.003 / max(atr, 1e-12))


@njit(cache=True, inline="always")
def _activation_curve_u(u: float, activation_u: float, mode: int, blend: float) -> float:
    if mode == ACTIVATION_CURVE_POST_ACTIVATION:
        return max(u - activation_u, 0.0)
    if mode == ACTIVATION_CURVE_BLENDED:
        return max(u - min(max(blend, 0.0), 1.0) * activation_u, 0.0)
    return max(u, 0.0)


@njit(cache=True, inline="always")
def _spline_ratio(u: float, params: np.ndarray) -> float:
    knots = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    if u <= 0.0:
        return 1.0
    previous = 1.0
    for k in range(1, 6):
        current = min(max(params[P_SPLINE_1 + k - 1], 0.0), previous)
        if u <= knots[k]:
            weight = (u - knots[k - 1]) / (knots[k] - knots[k - 1])
            return previous + weight * (current - previous)
        previous = current
    return previous


@njit(cache=True, inline="always")
def _excess_ratio(family: int, u: float, params: np.ndarray) -> float:
    terminal = min(max(params[P_TERMINAL_RATIO], 0.01), 1.0)
    center = max(params[P_CENTER], 0.05)
    shape = max(params[P_SHAPE], 0.05)
    u = max(u, 0.0)
    if family == FAMILY_CONSTANT:
        ratio = 1.0
    elif family == FAMILY_MULTILAYER:
        # Smooth convex mixture of three normalized decay layers.  The logits
        # make weights non-negative and sum to one; every component equals one
        # at entry, so the entry constraint remains exact.
        e1 = np.exp(min(max(params[P_EXTRA_1], -20.0), 20.0))
        e2 = np.exp(min(max(params[P_EXTRA_2], -20.0), 20.0))
        denom = 1.0 + e1 + e2
        w0, w1, w2 = 1.0 / denom, e1 / denom, e2 / denom
        q = (
            w0 * np.exp(-(u / center) ** 0.3)
            + w1 * np.exp(-(u / center) ** 0.6)
            + w2 * np.exp(-(u / center) ** shape)
        )
        ratio = terminal + (1.0 - terminal) * q
    elif family == FAMILY_SIGMOID:
        exponent = min(max(shape * (u - center), -60.0), 60.0)
        raw = 1.0 / (1.0 + np.exp(exponent))
        raw0 = 1.0 / (1.0 + np.exp(min(max(-shape * center, -60.0), 60.0)))
        ratio = terminal + (1.0 - terminal) * raw / max(raw0, 1e-12)
    elif family == FAMILY_EXPONENTIAL:
        ratio = terminal + (1.0 - terminal) * np.exp(-shape * u)
    elif family == FAMILY_RATIONAL:
        ratio = terminal + (1.0 - terminal) / (1.0 + (u / center) ** shape)
    else:
        ratio = _spline_ratio(u, params)
    lower = max(params[P_EXCESS_MIN_RATIO], 0.0)
    upper = max(params[P_EXCESS_MAX_RATIO], lower)
    return min(max(ratio, lower), upper)


@njit(cache=True, inline="always")
def _stop_fill(
    side: float,
    stop: float,
    high: float,
    low: float,
    exit_half_spread_bps: float,
    base_gap_bps: float,
    through_fraction: float,
    max_gap_bps: float,
) -> float:
    spread = max(exit_half_spread_bps, 0.0) / 10_000.0
    quote_px = stop * (1.0 - side * spread)
    through = max(stop - low, 0.0) if side > 0.0 else max(high - stop, 0.0)
    gap = min(stop * base_gap_bps / 10_000.0 + through_fraction * through, stop * max_gap_bps / 10_000.0)
    return quote_px - side * gap


@njit(cache=True, parallel=True)
def simulate_constrained_1m_paths(
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
    family: int,
    fee_per_side: float,
    stop_base_gap_bps: float,
    stop_through_fraction: float,
    stop_max_gap_bps: float,
    capital_trail_epsilon_atr: float,
    activation_curve_mode: int = ACTIVATION_CURVE_TOTAL_MFE,
    activation_curve_blend: float = 0.0,
) -> tuple[np.ndarray, ...]:
    n = len(row_index)
    exit_bar = np.full(n, -1, dtype=np.int32)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    gross_return = np.full(n, np.nan, dtype=np.float64)
    net_return = np.full(n, np.nan, dtype=np.float64)
    reason = np.full(n, REASON_TIMEOUT, dtype=np.int8)
    mfe_out = np.full(n, np.nan, dtype=np.float64)
    mae_out = np.full(n, np.nan, dtype=np.float64)
    capital_first_bar = np.full(n, -1, dtype=np.int32)
    trailing_first_bar = np.full(n, -1, dtype=np.int32)
    trailing_layer_first_bar = np.full((n, 3), -1, dtype=np.int32)
    trailing_layer_binding_bars = np.zeros((n, 3), dtype=np.int32)
    trailing_exit_layer = np.full(n, -1, dtype=np.int8)
    capital_binding_bars = np.zeros(n, dtype=np.int32)
    initial_capital_active = np.zeros(n, dtype=np.bool_)
    order_valid = np.ones(n, dtype=np.bool_)
    horizon = highs.shape[1]

    for out_i in prange(n):
        i = int(row_index[out_i])
        raw_entry = float(opens0[i])
        side = 1.0 if side_all[i] >= 0.0 else -1.0
        if not np.isfinite(raw_entry) or raw_entry <= 0.0:
            order_valid[out_i] = False
            continue
        atr = raw_entry * max(float(atr_frac_all[i]), 1e-6)
        entry = raw_entry * (1.0 + side * max(float(entry_half_spread_bps_all[i]), 0.0) / 10_000.0)
        sl_gap_atr = max(params[P_SL], 0.1)
        full_sl = entry - side * sl_gap_atr * atr
        base_trail_act = max(params[P_TRAIL_ACT], 0.05) * atr
        if params[P_TRAIL_ACT_CAP_FRAC] > 0.0:
            base_trail_act = min(base_trail_act, entry * params[P_TRAIL_ACT_CAP_FRAC])

        max_fav = 0.0
        max_adv = 0.0
        # The entry-time promotion buffer is measured from the executable
        # entry, avoiding spread-induced deactivation of an otherwise valid
        # capital stop.  Subsequent updates use completed raw closes.
        prior_close = entry
        capital_stop = full_sl
        trailing_stop = full_sl
        trailing_stop_layer = -1
        effective_stop = full_sl
        handed_over = family == FAMILY_TRAILING_ONLY
        layer_count = min(max(int(params[P_TRAIL_LAYER_COUNT]), 0), 3)
        completed = False

        for j in range(horizon):
            high = float(highs[i, j])
            low = float(lows[i, j])
            close = float(closes[i, j])
            if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
                break

            trail_act = base_trail_act
            decay_half = max(params[P_DECAY_HALF_MIN], 0.0)
            decay_start = max(params[P_DECAY_START_MIN], 0.0)
            decay_min = min(max(params[P_DECAY_MIN_MULT], 0.01), 1.0)
            if (
                layer_count == 0
                and decay_half > 0.0
                and decay_min < 1.0
                and j > decay_start
            ):
                decay = 0.5 ** ((j - decay_start) / decay_half)
                trail_act *= decay_min + (1.0 - decay_min) * decay
            u = max_fav / max(atr, 1e-12)
            activation_u = trail_act / max(atr, 1e-12)
            u_curve = _activation_curve_u(u, activation_u, activation_curve_mode, activation_curve_blend)
            trail_gap_atr = _trail_gap_atr(u_curve, entry, atr, params)
            trail_candidate = entry + side * max(max_fav - trail_gap_atr * atr, 0.0)
            trail_eligible = max_fav >= trail_act
            binding_layer = -1

            if layer_count > 0:
                # Ordered multi-activation total-MFE layers.  MFE is frozen at
                # the start of the bar, so a threshold crossed by this candle
                # can only arm on the next candle.
                best_candidate = full_sl
                for layer in range(layer_count):
                    activation_mult = params[P_TRAIL_ACT]
                    beta = params[P_TRAIL_BETA]
                    if layer == 1:
                        activation_mult = params[P_TRAIL_ACT_2]
                        beta = params[P_TRAIL_BETA_2]
                    elif layer == 2:
                        activation_mult = params[P_TRAIL_ACT_3]
                        beta = params[P_TRAIL_BETA_3]
                    activation = max(activation_mult, 0.05) * atr
                    if params[P_TRAIL_ACT_CAP_FRAC] > 0.0:
                        activation = min(
                            activation, entry * params[P_TRAIL_ACT_CAP_FRAC]
                        )
                    if max_fav < activation:
                        continue
                    if trailing_layer_first_bar[out_i, layer] < 0:
                        trailing_layer_first_bar[out_i, layer] = j
                    layer_gap = _trail_gap_atr(u, entry, atr, params, beta)
                    # Unlike the legacy single layer, the early layer may lock
                    # a controlled loss before the main profit trail activates.
                    candidate = entry + side * (max_fav - layer_gap * atr)
                    tighter = (
                        candidate > best_candidate
                        if side > 0.0
                        else candidate < best_candidate
                    )
                    if tighter:
                        best_candidate = candidate
                        binding_layer = layer
                trail_eligible = binding_layer >= 0
                trail_candidate = best_candidate
                if trail_eligible:
                    trailing_layer_binding_bars[out_i, binding_layer] += 1
                    if trailing_first_bar[out_i] < 0:
                        trailing_first_bar[out_i] = j

            if family != FAMILY_TRAILING_ONLY:
                entry_target_gap = min(max(params[P_ENTRY_RATIO], 0.05), 0.98) * sl_gap_atr
                trail_gap0 = entry * 0.003 / max(atr, 1e-12)
                delta0 = max(entry_target_gap - trail_gap0, capital_trail_epsilon_atr)
                delta_ratio = _excess_ratio(family, u, params)
                cap_gap_atr = trail_gap_atr + max(delta0 * delta_ratio, capital_trail_epsilon_atr)
                cap_candidate = entry + side * (max_fav - cap_gap_atr * atr)
                current_ratio = min(max(params[P_CURRENT_SL_RATIO], 0.0), 0.98)
                if current_ratio > 0.0:
                    current_bound = prior_close - side * current_ratio * sl_gap_atr * atr
                    cap_candidate = min(cap_candidate, current_bound) if side > 0.0 else max(cap_candidate, current_bound)
                capital_stop = max(capital_stop, cap_candidate) if side > 0.0 else min(capital_stop, cap_candidate)
                cap_effective = capital_stop > full_sl + 1e-12 if side > 0.0 else capital_stop < full_sl - 1e-12
                if cap_effective and capital_first_bar[out_i] < 0:
                    capital_first_bar[out_i] = j
                if j == 0:
                    initial_capital_active[out_i] = cap_effective
                if not handed_over and cap_effective:
                    capital_binding_bars[out_i] += 1

                trail_tighter = trail_candidate >= capital_stop - 1e-12 if side > 0.0 else trail_candidate <= capital_stop + 1e-12
                if not handed_over and trail_eligible and trail_tighter:
                    handed_over = True
                    trailing_first_bar[out_i] = j
                    if capital_first_bar[out_i] < 0 or capital_first_bar[out_i] >= j:
                        order_valid[out_i] = False

            if handed_over and trail_eligible:
                trail_tightened = (
                    trail_candidate > trailing_stop
                    if side > 0.0
                    else trail_candidate < trailing_stop
                )
                if trail_tightened:
                    trailing_stop = trail_candidate
                    if layer_count > 0:
                        trailing_stop_layer = binding_layer
                next_stop = trailing_stop
                next_reason = REASON_TRAILING
            elif family != FAMILY_TRAILING_ONLY:
                next_stop = capital_stop
                next_reason = REASON_CAPITAL if capital_first_bar[out_i] >= 0 else REASON_FULL_SL
            else:
                next_stop = full_sl
                next_reason = REASON_FULL_SL

            # A state transition may tighten but never loosen the executable stop.
            if side > 0.0:
                if next_stop > effective_stop:
                    effective_stop = next_stop
            else:
                if next_stop < effective_stop:
                    effective_stop = next_stop

            trigger = effective_stop / max(1.0 - max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0, 1e-12) if side > 0.0 else effective_stop / (1.0 + max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0)
            stop_hit = low <= trigger if side > 0.0 else high >= trigger
            if stop_hit:
                exit_bar[out_i] = j
                exit_price[out_i] = _stop_fill(
                    side, effective_stop, high, low, float(exit_half_spread_bps_all[i]),
                    stop_base_gap_bps, stop_through_fraction, stop_max_gap_bps,
                )
                reason[out_i] = next_reason
                if next_reason == REASON_TRAILING:
                    trailing_exit_layer[out_i] = trailing_stop_layer
                completed = True
                break

            # Keep the deployed fast-adverse guard fixed across search families.
            if params[P_ADV_ENABLED] > 0.5 and j <= params[P_ADV_FAST_MIN]:
                cur_fav_adv = max(high - entry, 0.0) if side > 0.0 else max(entry - low, 0.0)
                cur_mae_adv = max(entry - low, 0.0) if side > 0.0 else max(high - entry, 0.0)
                adv_mfe_atr = max(max_fav, cur_fav_adv) / max(atr, 1e-12)
                adv_mae_atr = max(max_adv, cur_mae_adv) / max(atr, 1e-12)
                elapsed_15m = max((j + 1) / 15.0, 1.0 / 15.0)
                speed = adv_mae_atr / elapsed_15m
                score = np.log1p(0.75) + np.log1p(max(adv_mae_atr, 0.0)) + np.log1p(max(speed, 0.0))
                eligible = adv_mae_atr >= params[P_ADV_MIN_MAE] and speed >= params[P_ADV_MIN_SPEED] and adv_mfe_atr <= params[P_ADV_MAX_MFE]
                if eligible and score > params[P_ADV_THETA]:
                    spread = max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0
                    exit_bar[out_i] = j
                    exit_price[out_i] = close * (1.0 - side * spread)
                    reason[out_i] = REASON_ADVERSE
                    completed = True
                    break

            cur_fav = max(high - entry, 0.0) if side > 0.0 else max(entry - low, 0.0)
            cur_adv = max(entry - low, 0.0) if side > 0.0 else max(high - entry, 0.0)
            max_fav = max(max_fav, cur_fav)
            max_adv = max(max_adv, cur_adv)
            prior_close = close

        if not completed:
            last = -1
            for k in range(horizon - 1, -1, -1):
                if np.isfinite(closes[i, k]):
                    last = k
                    break
            if last < 0:
                order_valid[out_i] = False
                continue
            spread = max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0
            exit_bar[out_i] = last
            exit_price[out_i] = float(closes[i, last]) * (1.0 - side * spread)
            reason[out_i] = REASON_TIMEOUT

        if family != FAMILY_TRAILING_ONLY and capital_first_bar[out_i] < 0:
            order_valid[out_i] = False
        gross = side * (exit_price[out_i] / entry - 1.0)
        fees = fee_per_side + fee_per_side * (1.0 + gross)
        gross_return[out_i] = gross
        net_return[out_i] = gross - fees
        mfe_out[out_i] = max_fav / entry
        mae_out[out_i] = max_adv / entry

    return (
        exit_bar, exit_price, gross_return, net_return, reason, mfe_out, mae_out,
        capital_first_bar, trailing_first_bar, capital_binding_bars,
        initial_capital_active, order_valid,
        trailing_layer_first_bar, trailing_layer_binding_bars,
        trailing_exit_layer,
    )
