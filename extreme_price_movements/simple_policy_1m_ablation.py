"""Fast, resolution-explicit kernels for the 1m capital-protection ablation.

This module is intentionally isolated from the deployed policy implementation.
It is research-only until a winning geometry is separately implemented and
parity-tested in live inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numba import njit, prange


FAMILY_CURRENT = 0
FAMILY_A = 1
FAMILY_B = 2
FAMILY_SIGMOID = 3
FAMILY_EXPONENTIAL = 4

MOD_MAX_GIVEBACK = 1
MOD_MIN_MFE_GAP = 2
MOD_MIN_CURRENT_GAP = 4

REASON_TIMEOUT = 0
REASON_FULL_SL = 1
REASON_CAPITAL = 2
REASON_TRAILING = 3
REASON_ADVERSE = 4


@dataclass(frozen=True)
class ReplaySpec:
    timeframe: str = "1m"
    bar_minutes: int = 1
    horizon_minutes: int = 1_440
    fee_per_side: float = 0.005
    stop_base_gap_bps: float = 15.0
    stop_through_fraction: float = 0.05
    stop_max_gap_bps: float = 75.0

    @property
    def path_len(self) -> int:
        return int(self.horizon_minutes // self.bar_minutes)


# Parameter vector indices used by the JIT kernel.
P_SL = 0
P_TRAIL_ACT = 1
P_TRAIL_POWER = 2
P_TRAIL_DIV = 3
P_TRAIL_BETA = 4
P_1 = 5
P_2 = 6
P_3 = 7
P_4 = 8
P_MIN_MFE = 9
P_MIN_CURRENT = 10
P_MAX_GIVEBACK = 11
P_LEGACY_REGRESSION = 12
P_LEGACY_LOCK_FRAC = 13
P_LEGACY_MIN_LOCK_BPS = 14
P_DECAY_HALF_MIN = 15
P_DECAY_START_MIN = 16
P_DECAY_MIN_MULT = 17
P_ADV_ENABLED = 18
P_ADV_MIN_MAE = 19
P_ADV_MIN_SPEED = 20
P_ADV_THETA = 21
P_ADV_FAST_MIN = 22
P_ADV_MAX_MFE = 23
P_TRAIL_ACT_CAP_FRAC = 24
P_LEGACY_SPREAD_LOCK_MULT = 25
N_PARAMS = 26


def params_to_vector(params: Mapping[str, Any]) -> np.ndarray:
    out = np.zeros(N_PARAMS, dtype=np.float64)
    out[P_SL] = float(params.get("sl_mult", 2.5))
    out[P_TRAIL_ACT] = float(params.get("trailing_activation_mult", 1.5))
    out[P_TRAIL_POWER] = float(params.get("trailing_power", 1.5))
    out[P_TRAIL_DIV] = float(params.get("trailing_squash_divisor", 2.0))
    out[P_TRAIL_BETA] = float(params.get("giveback_beta", 0.5))
    out[P_1] = float(params.get("p1", params.get("capital_protect_mfe_mult", 1.0)))
    out[P_2] = float(params.get("p2", 1.7))
    out[P_3] = float(params.get("p3", 0.45))
    out[P_4] = float(params.get("p4", 1.2))
    out[P_MIN_MFE] = float(params.get("min_mfe_gap_atr", 0.0))
    out[P_MIN_CURRENT] = float(params.get("min_current_gap_atr", 0.0))
    out[P_MAX_GIVEBACK] = float(params.get("max_giveback_atr", 1e9))
    out[P_LEGACY_REGRESSION] = float(params.get("capital_protect_regression_frac", 0.45))
    lock = params.get("capital_protect_lock_frac", np.nan)
    out[P_LEGACY_LOCK_FRAC] = float(lock) if lock is not None else np.nan
    out[P_LEGACY_MIN_LOCK_BPS] = float(params.get("capital_protect_min_lock_bps", 0.0))
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
    out[P_LEGACY_SPREAD_LOCK_MULT] = float(params.get("capital_protect_spread_lock_mult", 1.5))
    return out


@njit(cache=True, inline="always")
def _capital_gap(family: int, u: float, atr: float, params: np.ndarray) -> float:
    if family == FAMILY_A:
        return max(params[P_1] * atr, 0.0)
    if family == FAMILY_B:
        # A protection envelope uses the loosest of the simultaneously active
        # layers.  This avoids a zero-distance stop at u=0 and honors the
        # requirement that capital protection remain looser than profit trail.
        g1 = params[P_1] * atr
        g2 = params[P_2] * atr * max(u, 0.0) ** 0.3
        g3 = params[P_3] * atr * max(u, 0.0) ** 0.6
        return max(g1, g2, g3, 0.0)
    if family == FAMILY_SIGMOID:
        high = params[P_1]
        low = params[P_2]
        center = params[P_3]
        k = params[P_4]
        exponent = min(max(k * (u - center), -60.0), 60.0)
        mult = low + (high - low) / (1.0 + np.exp(exponent))
        return max(mult * atr, 0.0)
    # Gentle exponential.
    floor = params[P_1]
    amplitude = params[P_2]
    decay = params[P_3]
    return max((floor + amplitude * np.exp(-decay * max(u, 0.0))) * atr, 0.0)


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
def simulate_1m_paths(
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
    modifiers: int,
    fee_per_side: float,
    stop_base_gap_bps: float,
    stop_through_fraction: float,
    stop_max_gap_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(row_index)
    exit_bar = np.full(n, -1, dtype=np.int32)
    exit_price = np.full(n, np.nan, dtype=np.float64)
    gross_return = np.full(n, np.nan, dtype=np.float64)
    net_return = np.full(n, np.nan, dtype=np.float64)
    reason = np.full(n, REASON_TIMEOUT, dtype=np.int8)
    mfe_out = np.full(n, np.nan, dtype=np.float64)
    mae_out = np.full(n, np.nan, dtype=np.float64)
    horizon = highs.shape[1]

    for out_i in prange(n):
        i = int(row_index[out_i])
        raw_entry = float(opens0[i])
        side = 1.0 if side_all[i] >= 0.0 else -1.0
        if not np.isfinite(raw_entry) or raw_entry <= 0.0:
            continue
        atr = raw_entry * max(float(atr_frac_all[i]), 1e-6)
        entry = raw_entry * (1.0 + side * max(float(entry_half_spread_bps_all[i]), 0.0) / 10_000.0)
        sl_dist = max(params[P_SL], 0.05) * atr
        full_sl = entry - side * sl_dist
        base_trail_act = max(params[P_TRAIL_ACT], 0.0) * atr
        if params[P_TRAIL_ACT_CAP_FRAC] > 0.0:
            base_trail_act = min(base_trail_act, entry * params[P_TRAIL_ACT_CAP_FRAC])
        trail_power = max(params[P_TRAIL_POWER], 0.05)
        trail_div = max(params[P_TRAIL_DIV], 0.05)
        trail_beta = max(params[P_TRAIL_BETA], 0.0)
        max_fav = 0.0
        max_adv = 0.0
        prior_close = raw_entry
        capital_stop = full_sl
        legacy_armed = False
        completed = False

        for j in range(horizon):
            high = float(highs[i, j])
            low = float(lows[i, j])
            close = float(closes[i, j])
            if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
                break

            # Pessimistic collision contract: the original full stop is checked
            # before favorable excursion from the same one-minute candle.
            full_trigger = full_sl / max(1.0 - max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0, 1e-12) if side > 0.0 else full_sl / (1.0 + max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0)
            full_hit = low <= full_trigger if side > 0.0 else high >= full_trigger
            if full_hit:
                fill = _stop_fill(side, full_sl, high, low, float(exit_half_spread_bps_all[i]), stop_base_gap_bps, stop_through_fraction, stop_max_gap_bps)
                exit_bar[out_i] = j
                exit_price[out_i] = fill
                reason[out_i] = REASON_FULL_SL
                completed = True
                break

            trail_act = base_trail_act
            decay_half = max(params[P_DECAY_HALF_MIN], 0.0)
            decay_start = max(params[P_DECAY_START_MIN], 0.0)
            decay_min = min(max(params[P_DECAY_MIN_MULT], 0.01), 1.0)
            if decay_half > 0.0 and decay_min < 1.0 and j > decay_start:
                decay = 0.5 ** ((j - decay_start) / decay_half)
                trail_act *= decay_min + (1.0 - decay_min) * decay
            trailing_armed = max_fav >= trail_act
            dynamic = (max_fav / max(atr * trail_div, 1e-12)) ** trail_power
            dynamic = min(max(dynamic, 0.0), 1.0)
            trail_gap = max(max_fav * trail_beta * (1.0 - dynamic), entry * 0.003)
            trail_stop = entry + side * max(max_fav - trail_gap, 0.0)

            # Keep the deployed fast-adverse guard fixed across capital arms.
            # Speed is normalized to 15-minute equivalents so changing replay
            # resolution does not multiply its economic aggressiveness by 15.
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
                    fill = close * (1.0 - side * spread)
                    exit_bar[out_i] = j
                    exit_price[out_i] = fill
                    reason[out_i] = REASON_ADVERSE
                    completed = True
                    break

            cause = REASON_FULL_SL
            effective_stop = full_sl
            if family == FAMILY_CURRENT:
                activation = max(params[P_1], 0.0) * atr
                if activation > 0.0 and not legacy_armed and max_fav >= activation:
                    legacy_armed = True
                if legacy_armed:
                    lock_frac = params[P_LEGACY_LOCK_FRAC]
                    if np.isfinite(lock_frac):
                        lock = activation * min(max(lock_frac, -1.0), 1.0)
                    else:
                        lock = activation - params[P_LEGACY_REGRESSION] * (activation + sl_dist)
                    lock = max(lock, entry * max(params[P_LEGACY_MIN_LOCK_BPS], 0.0) / 10_000.0)
                    full_spread_bps = 2.0 * max(float(entry_half_spread_bps_all[i]), 0.0)
                    lock = max(lock, entry * full_spread_bps * max(params[P_LEGACY_SPREAD_LOCK_MULT], 0.0) / 10_000.0)
                    candidate = entry + side * lock
                    if (side > 0.0 and candidate > effective_stop) or (side < 0.0 and candidate < effective_stop):
                        effective_stop = candidate
                        cause = REASON_CAPITAL
            else:
                u = max_fav / max(atr, 1e-12)
                gap = _capital_gap(family, u, atr, params)
                if modifiers & MOD_MIN_MFE_GAP:
                    gap = max(gap, params[P_MIN_MFE] * atr)
                if modifiers & MOD_MAX_GIVEBACK:
                    gap = min(gap, params[P_MAX_GIVEBACK] * atr)
                # Capital preservation must be looser than an armed profit trail.
                if trailing_armed:
                    gap = max(gap, trail_gap)
                candidate = entry + side * (max_fav - gap)
                if modifiers & MOD_MIN_CURRENT_GAP:
                    current_bound = prior_close - side * params[P_MIN_CURRENT] * atr
                    candidate = min(candidate, current_bound) if side > 0.0 else max(candidate, current_bound)
                # Always-on and monotone: protection may tighten but never loosen
                # the original stop or its own previously promoted level.
                if side > 0.0:
                    capital_stop = max(capital_stop, candidate)
                    if capital_stop > effective_stop:
                        effective_stop = capital_stop
                        cause = REASON_CAPITAL
                else:
                    capital_stop = min(capital_stop, candidate)
                    if capital_stop < effective_stop:
                        effective_stop = capital_stop
                        cause = REASON_CAPITAL

            if trailing_armed and ((side > 0.0 and trail_stop > effective_stop) or (side < 0.0 and trail_stop < effective_stop)):
                effective_stop = trail_stop
                cause = REASON_TRAILING

            trigger = effective_stop / max(1.0 - max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0, 1e-12) if side > 0.0 else effective_stop / (1.0 + max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0)
            stop_hit = low <= trigger if side > 0.0 else high >= trigger
            if stop_hit:
                fill = _stop_fill(side, effective_stop, high, low, float(exit_half_spread_bps_all[i]), stop_base_gap_bps, stop_through_fraction, stop_max_gap_bps)
                exit_bar[out_i] = j
                exit_price[out_i] = fill
                reason[out_i] = cause
                completed = True
                break

            cur_fav = max(high - entry, 0.0) if side > 0.0 else max(entry - low, 0.0)
            cur_adv = max(entry - low, 0.0) if side > 0.0 else max(high - entry, 0.0)
            max_fav = max(max_fav, cur_fav)
            max_adv = max(max_adv, cur_adv)
            prior_close = close

        if not completed:
            j = max(exit_bar[out_i], 0)
            last = -1
            for k in range(horizon - 1, -1, -1):
                if np.isfinite(closes[i, k]):
                    last = k
                    break
            if last < 0:
                continue
            spread = max(float(exit_half_spread_bps_all[i]), 0.0) / 10_000.0
            fill = float(closes[i, last]) * (1.0 - side * spread)
            exit_bar[out_i] = last
            exit_price[out_i] = fill
            reason[out_i] = REASON_TIMEOUT

        gross = side * (exit_price[out_i] / entry - 1.0)
        fees = fee_per_side + fee_per_side * (1.0 + gross)
        gross_return[out_i] = gross
        net_return[out_i] = gross - fees
        mfe_out[out_i] = max_fav / entry
        mae_out[out_i] = max_adv / entry

    return exit_bar, exit_price, gross_return, net_return, reason, mfe_out, mae_out


@njit(cache=True)
def capacity_select(
    timestamps_ns: np.ndarray,
    symbol_codes: np.ndarray,
    exit_bars: np.ndarray,
    bar_minutes: int,
    max_open: int = 8,
    max_new_per_bar: int = 2,
) -> np.ndarray:
    """Select already timestamp/rank-sorted rows under portfolio capacity."""
    n = len(timestamps_ns)
    selected = np.zeros(n, dtype=np.bool_)
    open_exit = np.full(max_open, -1, dtype=np.int64)
    open_symbol = np.full(max_open, -1, dtype=np.int32)
    minute_ns = int(bar_minutes) * 60 * 1_000_000_000
    current_ts = np.int64(-9223372036854775807)
    new_at_ts = 0
    for i in range(n):
        ts = timestamps_ns[i]
        if ts != current_ts:
            current_ts = ts
            new_at_ts = 0
        for slot in range(max_open):
            if open_exit[slot] <= ts:
                open_exit[slot] = -1
                open_symbol[slot] = -1
        if new_at_ts >= max_new_per_bar or exit_bars[i] < 0:
            continue
        symbol = symbol_codes[i]
        duplicate = False
        free_slot = -1
        for slot in range(max_open):
            if open_exit[slot] < 0 and free_slot < 0:
                free_slot = slot
            if open_exit[slot] >= 0 and open_symbol[slot] == symbol:
                duplicate = True
        if duplicate or free_slot < 0:
            continue
        selected[i] = True
        new_at_ts += 1
        open_exit[free_slot] = ts + (int(exit_bars[i]) + 1) * minute_ns
        open_symbol[free_slot] = symbol
    return selected


@njit(cache=True)
def objective_score_fast(
    timestamps_ns: np.ndarray,
    symbol_codes: np.ndarray,
    rank_pct: np.ndarray,
    week_codes: np.ndarray,
    exit_bars: np.ndarray,
    net_return: np.ndarray,
    bar_minutes: int = 1,
) -> tuple[float, float, float, float, int]:
    selected = capacity_select(
        timestamps_ns,
        symbol_codes,
        exit_bars,
        int(bar_minutes),
    )
    max_week = -1
    for i in range(len(week_codes)):
        if week_codes[i] > max_week:
            max_week = int(week_codes[i])
    weekly = np.zeros(max_week + 1 if max_week >= 0 else 1, dtype=np.float64)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    total = 0.0
    n = 0
    for i in range(len(net_return)):
        if not selected[i] or not np.isfinite(net_return[i]):
            continue
        size = 0.075 + 0.075 * min(max(rank_pct[i], 0.0), 1.0) ** 1.1
        pnl = net_return[i] * size
        total += pnl
        equity += pnl
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        code = int(week_codes[i])
        if code >= 0:
            weekly[code] += pnl
        n += 1
    if n == 0:
        return -1e9, 0.0, 0.0, 0.0, 0
    mean = 0.0
    count = 0
    worst = 1e100
    for value in weekly:
        mean += value
        worst = min(worst, value)
        count += 1
    mean /= max(count, 1)
    variance = 0.0
    for value in weekly:
        variance += (value - mean) ** 2
    std = np.sqrt(variance / max(count, 1))
    objective = mean - 0.5 * std + 0.25 * worst - 0.10 * abs(max_dd)
    return objective, total, worst, max_dd, n


def evaluate_results(
    rows: pd.DataFrame,
    exit_bars: np.ndarray,
    gross_return: np.ndarray,
    net_return: np.ndarray,
    reason: np.ndarray,
    mfe: np.ndarray,
    mae: np.ndarray,
    *,
    bar_minutes: int = 1,
    apply_capacity: bool = True,
) -> tuple[dict[str, Any], np.ndarray]:
    n = len(rows)
    finite = np.isfinite(net_return) & (exit_bars >= 0)
    selected = finite.copy()
    if apply_capacity and n:
        ts = pd.to_datetime(rows["timestamp"], utc=True).astype("int64").to_numpy(dtype=np.int64)
        symbols = pd.Categorical(rows["symbol"].astype(str)).codes.astype(np.int32)
        cap = capacity_select(ts, symbols, exit_bars.astype(np.int32), int(bar_minutes))
        selected &= cap
    idx = np.flatnonzero(selected)
    if len(idx) == 0:
        return {
            "candidate_count": int(n), "valid_path_count": int(finite.sum()), "n_trades": 0,
            "net_pnl_bankroll": 0.0, "mean_net_return": np.nan, "worst_week": 0.0,
            "worst_month": 0.0, "max_drawdown": 0.0, "objective": -1e9,
        }, selected
    rank = pd.to_numeric(rows.iloc[idx]["rank_pct"], errors="coerce").fillna(0.9).to_numpy(dtype=np.float64)
    sizes = 0.075 + 0.075 * np.power(np.clip(rank, 0.0, 1.0), 1.1)
    pnl = net_return[idx] * sizes
    gross_pnl = gross_return[idx] * sizes
    ts = pd.to_datetime(rows.iloc[idx]["timestamp"], utc=True)
    week = ts.dt.tz_localize(None).dt.to_period("W").astype(str)
    month = ts.dt.strftime("%Y-%m")
    weekly = pd.Series(pnl).groupby(week.reset_index(drop=True)).sum().to_numpy(dtype=float)
    monthly = pd.Series(pnl).groupby(month.reset_index(drop=True)).sum().to_numpy(dtype=float)
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):]
    worst_week = float(np.min(weekly)) if len(weekly) else 0.0
    weekly_mean = float(np.mean(weekly)) if len(weekly) else 0.0
    weekly_std = float(np.std(weekly)) if len(weekly) else 0.0
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0
    objective = weekly_mean - 0.5 * weekly_std + 0.25 * worst_week - 0.10 * abs(max_dd)
    metrics = {
        "candidate_count": int(n),
        "valid_path_count": int(finite.sum()),
        "n_trades": int(len(idx)),
        "trades_per_day": float(len(idx) / max((ts.max() - ts.min()).total_seconds() / 86400.0 + 1.0, 1.0)),
        "gross_pnl_bankroll": float(np.sum(gross_pnl)),
        "fee_pnl_bankroll": float(np.sum(gross_pnl - pnl)),
        "net_pnl_bankroll": float(np.sum(pnl)),
        "mean_net_return": float(np.mean(net_return[idx])),
        "mean_gross_return": float(np.mean(gross_return[idx])),
        "hit_rate": float(np.mean(net_return[idx] > 0.0)),
        "worst_week": worst_week,
        "worst_month": float(np.min(monthly)) if len(monthly) else 0.0,
        "weekly_mean": weekly_mean,
        "weekly_std": weekly_std,
        "positive_week_fraction": float(np.mean(weekly > 0.0)) if len(weekly) else 0.0,
        "max_drawdown": max_dd,
        "objective": float(objective),
        "mean_holding_hours": float(np.mean(exit_bars[idx] + 1) * bar_minutes / 60.0),
        "p90_holding_hours": float(np.quantile(exit_bars[idx] + 1, 0.9) * bar_minutes / 60.0),
        "mean_mfe": float(np.nanmean(mfe[idx])),
        "mean_mae": float(np.nanmean(mae[idx])),
        "full_sl_rate": float(np.mean(reason[idx] == REASON_FULL_SL)),
        "adverse_exit_rate": float(np.mean(reason[idx] == REASON_ADVERSE)),
        "capital_protect_rate": float(np.mean(reason[idx] == REASON_CAPITAL)),
        "trailing_rate": float(np.mean(reason[idx] == REASON_TRAILING)),
        "timeout_rate": float(np.mean(reason[idx] == REASON_TIMEOUT)),
    }
    return metrics, selected
