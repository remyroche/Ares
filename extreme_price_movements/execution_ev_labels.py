"""Causal 12-hour execution-policy labels for the execution-EV meta head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numba import njit, prange

REASON_TIMEOUT = 0
REASON_FULL_STOP = 1
REASON_TRAILING = 2
REASON_ADVERSE = 3


@dataclass(frozen=True)
class ExecutionLabelGeometry:
    sl_mult: float
    trailing_activation_mult: float
    trailing_activation_cap_pct: float
    trailing_activation_decay_half_life_minutes: float
    trailing_activation_decay_start_minutes: float
    trailing_activation_min_mult: float
    trailing_power: float
    trailing_squash_divisor: float
    giveback_beta: float
    adverse_exit_enabled: bool
    adverse_exit_min_mae_atr: float
    adverse_exit_min_speed_per_15m: float
    adverse_exit_theta: float
    adverse_exit_fast_minutes: float
    adverse_exit_max_mfe_atr: float

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "ExecutionLabelGeometry":
        return cls(
            sl_mult=float(values.get("sl_mult", 3.0)),
            trailing_activation_mult=float(values.get("trailing_activation_mult", 2.0)),
            trailing_activation_cap_pct=float(values.get("trailing_activation_cap_pct", 0.0)),
            trailing_activation_decay_half_life_minutes=float(
                values.get("trailing_activation_decay_half_life_minutes", 0.0)
            ),
            trailing_activation_decay_start_minutes=float(
                values.get("trailing_activation_decay_start_minutes", 0.0)
            ),
            trailing_activation_min_mult=float(values.get("trailing_activation_min_mult", 1.0)),
            trailing_power=float(values.get("trailing_power", 1.5)),
            trailing_squash_divisor=float(values.get("trailing_squash_divisor", 2.0)),
            giveback_beta=float(values.get("giveback_beta", 0.5)),
            adverse_exit_enabled=bool(values.get("adverse_exit_enabled", False)),
            adverse_exit_min_mae_atr=float(values.get("adverse_exit_min_mae_atr", 1.0)),
            adverse_exit_min_speed_per_15m=float(
                values.get("adverse_exit_min_speed_per_15m", 0.3)
            ),
            adverse_exit_theta=float(values.get("adverse_exit_theta", 1e9)),
            adverse_exit_fast_minutes=float(values.get("adverse_exit_fast_minutes", 0.0)),
            adverse_exit_max_mfe_atr=float(values.get("adverse_exit_max_mfe_atr", 0.25)),
        )

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.sl_mult,
                self.trailing_activation_mult,
                self.trailing_activation_cap_pct,
                self.trailing_activation_decay_half_life_minutes,
                self.trailing_activation_decay_start_minutes,
                self.trailing_activation_min_mult,
                self.trailing_power,
                self.trailing_squash_divisor,
                self.giveback_beta,
                float(self.adverse_exit_enabled),
                self.adverse_exit_min_mae_atr,
                self.adverse_exit_min_speed_per_15m,
                self.adverse_exit_theta,
                self.adverse_exit_fast_minutes,
                self.adverse_exit_max_mfe_atr,
            ],
            dtype=np.float64,
        )


@njit(cache=True, parallel=True)
def simulate_execution_ev_12h(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    side: np.ndarray,
    atr_fraction: np.ndarray,
    cost_return: np.ndarray,
    long_params: np.ndarray,
    short_params: np.ndarray,
    bar_minutes: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve side-policy exits over an already-causal fixed-length path.

    Stop collisions are pessimistic: a stop is checked before favorable movement
    from the same aggregate candle. Costs are deducted once, after gross return.
    """

    n_rows, horizon = highs.shape
    gross = np.full(n_rows, np.nan, dtype=np.float64)
    net = np.full(n_rows, np.nan, dtype=np.float64)
    reason = np.full(n_rows, REASON_TIMEOUT, dtype=np.int8)
    exit_bar = np.full(n_rows, -1, dtype=np.int16)
    mfe = np.full(n_rows, np.nan, dtype=np.float64)
    mae = np.full(n_rows, np.nan, dtype=np.float64)
    for i in prange(n_rows):
        entry = float(opens[i, 0])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        direction = 1.0 if side[i] >= 0.0 else -1.0
        params = long_params if direction > 0.0 else short_params
        atr = entry * max(float(atr_fraction[i]), 1e-8)
        full_stop = entry - direction * max(float(params[0]), 0.05) * atr
        activation = max(float(params[1]), 0.0) * atr
        if params[2] > 0.0:
            activation = min(activation, entry * params[2])
        max_fav = 0.0
        max_adv = 0.0
        exit_price = np.nan
        completed = False
        for j in range(horizon):
            high = float(highs[i, j])
            low = float(lows[i, j])
            close = float(closes[i, j])
            if not (np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
                break

            full_hit = low <= full_stop if direction > 0.0 else high >= full_stop
            if full_hit:
                exit_price = full_stop
                reason[i] = REASON_FULL_STOP
                exit_bar[i] = j
                completed = True
                break

            elapsed_minutes = float((j + 1) * bar_minutes)
            current_fav = (
                max(high - entry, 0.0)
                if direction > 0.0
                else max(entry - low, 0.0)
            )
            current_adv = (
                max(entry - low, 0.0)
                if direction > 0.0
                else max(high - entry, 0.0)
            )
            if params[9] > 0.5 and elapsed_minutes <= params[13]:
                adverse_atr = max(max_adv, current_adv) / max(atr, 1e-12)
                favorable_atr = max(max_fav, current_fav) / max(atr, 1e-12)
                speed = adverse_atr / max(elapsed_minutes / 15.0, 1.0 / 15.0)
                adverse_score = (
                    np.log1p(0.75)
                    + np.log1p(max(adverse_atr, 0.0))
                    + np.log1p(max(speed, 0.0))
                )
                if (
                    adverse_atr >= params[10]
                    and speed >= params[11]
                    and favorable_atr <= params[14]
                    and adverse_score > params[12]
                ):
                    exit_price = close
                    reason[i] = REASON_ADVERSE
                    exit_bar[i] = j
                    completed = True
                    break

            local_activation = activation
            if params[3] > 0.0 and params[5] < 1.0 and elapsed_minutes > params[4]:
                decay = 0.5 ** ((elapsed_minutes - params[4]) / params[3])
                local_activation *= params[5] + (1.0 - params[5]) * decay
            trailing_armed = max_fav >= local_activation
            if trailing_armed:
                dynamic = (max_fav / max(atr * max(params[7], 0.05), 1e-12)) ** max(
                    params[6], 0.05
                )
                dynamic = min(max(dynamic, 0.0), 1.0)
                trail_gap = max(max_fav * max(params[8], 0.0) * (1.0 - dynamic), entry * 0.003)
                trail_stop = entry + direction * max(max_fav - trail_gap, 0.0)
                trail_hit = low <= trail_stop if direction > 0.0 else high >= trail_stop
                if trail_hit:
                    exit_price = trail_stop
                    reason[i] = REASON_TRAILING
                    exit_bar[i] = j
                    completed = True
                    break
            max_fav = max(max_fav, current_fav)
            max_adv = max(max_adv, current_adv)

        if not completed:
            last = -1
            for j in range(horizon - 1, -1, -1):
                if np.isfinite(closes[i, j]):
                    last = j
                    break
            if last < 0:
                continue
            exit_price = float(closes[i, last])
            exit_bar[i] = last
            reason[i] = REASON_TIMEOUT
        row_gross = direction * (exit_price / entry - 1.0)
        gross[i] = row_gross
        net[i] = row_gross - max(float(cost_return[i]), 0.0)
        mfe[i] = max_fav / entry
        mae[i] = max_adv / entry
    return gross, net, reason, exit_bar, mfe, mae


def policy_geometry_from_manifest(
    payload: Mapping[str, object], side: str
) -> ExecutionLabelGeometry:
    geometry = payload.get("geometry")
    if not isinstance(geometry, Mapping) or not isinstance(geometry.get(side), Mapping):
        raise ValueError(f"policy manifest is missing geometry for side {side!r}")
    return ExecutionLabelGeometry.from_mapping(geometry[side])


def reason_names(values: Sequence[int]) -> np.ndarray:
    mapping = np.asarray(["timeout", "full_stop", "trailing", "adverse_exit"], dtype=object)
    raw = np.asarray(values, dtype=np.int64)
    if np.any((raw < 0) | (raw >= len(mapping))):
        raise ValueError("execution reason contains an unknown code")
    return mapping[raw]
