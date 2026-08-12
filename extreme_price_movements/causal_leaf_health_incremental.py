"""Compiled primitives for bounded strict-prequential family health.

This module contains the hot H1 transition only.  It works on compact
per-candidate contribution arrays (family code, direction, value) and never
constructs a contribution/candidate dataframe.  The artifact-level runner
builds H2 snapshots from compact period summaries and uses these primitives
for the 424M contribution updates.

All routines observe the same invariant: score a complete feature-timestamp
block before applying any label resolved at that timestamp.
"""
from __future__ import annotations

import numpy as np

try:  # Production environments for this pipeline bundle Numba.
    from numba import njit
except ImportError as exc:  # pragma: no cover - clear dependency failure
    raise RuntimeError("causal incremental health requires numba") from exc


H1_METRIC_COUNT = 10


@njit(cache=True)
def score_h1_candidate(
    family_codes: np.ndarray,
    directions: np.ndarray,
    contributions: np.ndarray,
    family_rows: np.ndarray,
    family_successes: np.ndarray,
    family_predictions: np.ndarray,
    family_nets: np.ndarray,
    family_expecteds: np.ndarray,
    family_false_positive_losses: np.ndarray,
    family_timestamps: np.ndarray,
    family_days: np.ndarray,
    family_symbols: np.ndarray,
    global_rows: float,
    global_successes: float,
    side_rows: float,
    side_successes: float,
    global_alpha: float,
    global_beta: float,
    side_prior_strength: float,
    family_prior_strength: float,
    min_timestamp_support: float,
    min_day_support: float,
    min_symbol_support: float,
) -> np.ndarray:
    """Return contribution-weighted H1 metrics for negative then positive.

    ``directions`` uses 0 for negative and 1 for positive.  Values are read
    strictly before the corresponding candidate's resolution update.
    """

    out = np.zeros((2, H1_METRIC_COUNT + 1), dtype=np.float64)
    global_mean = (global_successes + global_alpha) / max(global_rows + global_alpha + global_beta, 1e-12)
    side_mean = (side_successes + side_prior_strength * global_mean) / max(side_rows + side_prior_strength, 1e-12)
    for index in range(len(family_codes)):
        code = family_codes[index]
        direction = directions[index]
        weight = abs(contributions[index])
        rows = family_rows[code]
        successes = family_successes[code]
        alpha = successes + family_prior_strength * side_mean
        beta = (rows - successes) + family_prior_strength * (1.0 - side_mean)
        total = max(alpha + beta, 1e-12)
        posterior = alpha / total
        lower = max(0.0, posterior - 1.96 * np.sqrt(max(posterior * (1.0 - posterior) / (total + 1.0), 0.0)))
        divisor = max(rows, 1.0)
        support = min(
            1.0,
            min(family_timestamps[code] / min_timestamp_support, min(family_days[code] / min_day_support, family_symbols[code] / min_symbol_support)),
        )
        out[direction, 0] += weight * posterior
        out[direction, 1] += weight * lower
        out[direction, 2] += weight * rows
        out[direction, 3] += weight * family_timestamps[code]
        out[direction, 4] += weight * family_days[code]
        out[direction, 5] += weight * family_symbols[code]
        out[direction, 6] += weight * support
        out[direction, 7] += weight * (successes / divisor - family_predictions[code] / divisor)
        out[direction, 8] += weight * (family_nets[code] / divisor - family_expecteds[code] / divisor)
        out[direction, 9] += weight * (family_false_positive_losses[code] / divisor)
        out[direction, H1_METRIC_COUNT] += weight
    for direction in range(2):
        denominator = out[direction, H1_METRIC_COUNT]
        if denominator > 0.0:
            for metric in range(H1_METRIC_COUNT):
                out[direction, metric] /= denominator
    return out


@njit(cache=True)
def score_h1_block(
    family_codes: np.ndarray,
    directions: np.ndarray,
    contributions: np.ndarray,
    offsets: np.ndarray,
    family_rows: np.ndarray,
    family_successes: np.ndarray,
    family_predictions: np.ndarray,
    family_nets: np.ndarray,
    family_expecteds: np.ndarray,
    family_false_positive_losses: np.ndarray,
    family_timestamps: np.ndarray,
    family_days: np.ndarray,
    family_symbols: np.ndarray,
    global_rows: float,
    global_successes: float,
    side_rows: float,
    side_successes: float,
    global_alpha: float,
    global_beta: float,
    side_prior_strength: float,
    family_prior_strength: float,
    min_timestamp_support: float,
    min_day_support: float,
    min_symbol_support: float,
) -> np.ndarray:
    """Score a complete equal-feature-time candidate block without updates.

    The block form removes per-candidate Python/Numba call overhead: the
    runner supplies one contiguous contribution buffer and ``n + 1`` offsets.
    """

    count = len(offsets) - 1
    result = np.zeros((count, 2, H1_METRIC_COUNT + 1), dtype=np.float64)
    global_mean = (global_successes + global_alpha) / max(global_rows + global_alpha + global_beta, 1e-12)
    side_mean = (side_successes + side_prior_strength * global_mean) / max(side_rows + side_prior_strength, 1e-12)
    for candidate in range(count):
        for index in range(offsets[candidate], offsets[candidate + 1]):
            code = family_codes[index]
            direction = directions[index]
            weight = abs(contributions[index])
            rows = family_rows[code]
            successes = family_successes[code]
            alpha = successes + family_prior_strength * side_mean
            beta = (rows - successes) + family_prior_strength * (1.0 - side_mean)
            total = max(alpha + beta, 1e-12)
            posterior = alpha / total
            lower = max(0.0, posterior - 1.96 * np.sqrt(max(posterior * (1.0 - posterior) / (total + 1.0), 0.0)))
            divisor = max(rows, 1.0)
            support = min(
                1.0,
                min(family_timestamps[code] / min_timestamp_support, min(family_days[code] / min_day_support, family_symbols[code] / min_symbol_support)),
            )
            result[candidate, direction, 0] += weight * posterior
            result[candidate, direction, 1] += weight * lower
            result[candidate, direction, 2] += weight * rows
            result[candidate, direction, 3] += weight * family_timestamps[code]
            result[candidate, direction, 4] += weight * family_days[code]
            result[candidate, direction, 5] += weight * family_symbols[code]
            result[candidate, direction, 6] += weight * support
            result[candidate, direction, 7] += weight * (successes / divisor - family_predictions[code] / divisor)
            result[candidate, direction, 8] += weight * (family_nets[code] / divisor - family_expecteds[code] / divisor)
            result[candidate, direction, 9] += weight * (family_false_positive_losses[code] / divisor)
            result[candidate, direction, H1_METRIC_COUNT] += weight
        for direction in range(2):
            denominator = result[candidate, direction, H1_METRIC_COUNT]
            if denominator > 0.0:
                for metric in range(H1_METRIC_COUNT):
                    result[candidate, direction, metric] /= denominator
    return result


@njit(cache=True)
def score_auxiliary_block(
    family_codes: np.ndarray,
    directions: np.ndarray,
    contributions: np.ndarray,
    offsets: np.ndarray,
    h2_values: np.ndarray,
    h3_values: np.ndarray,
) -> np.ndarray:
    """Contribution-weight H2/H3 states for a complete feature-time block.

    H2/H3 feature values are pre-frozen for the current calendar month.  The
    denominator deliberately includes every active family contribution, so a
    non-selected H3 family contributes zero rather than renormalising the
    selected subset.  The final cell is the active absolute contribution.
    """

    h2_width = h2_values.shape[1]
    h3_width = h3_values.shape[1]
    width = h2_width + h3_width + 1
    count = len(offsets) - 1
    result = np.zeros((count, 2, width), dtype=np.float64)
    for candidate in range(count):
        for index in range(offsets[candidate], offsets[candidate + 1]):
            code = family_codes[index]
            direction = directions[index]
            weight = abs(contributions[index])
            for metric in range(h2_width):
                result[candidate, direction, metric] += weight * h2_values[code, metric]
            for metric in range(h3_width):
                result[candidate, direction, h2_width + metric] += weight * h3_values[code, metric]
            result[candidate, direction, width - 1] += weight
        for direction in range(2):
            denominator = result[candidate, direction, width - 1]
            if denominator > 0.0:
                for metric in range(width - 1):
                    result[candidate, direction, metric] /= denominator
    return result


@njit(cache=True)
def update_h1_candidate(
    family_codes: np.ndarray,
    contributions: np.ndarray,
    success: float,
    prediction: float,
    net_bps: float,
    base_expected_bps: float,
    decision_timestamp_ns: np.int64,
    decision_day: np.int32,
    asset_code: np.int32,
    family_rows: np.ndarray,
    family_successes: np.ndarray,
    family_predictions: np.ndarray,
    family_nets: np.ndarray,
    family_expecteds: np.ndarray,
    family_false_positive_losses: np.ndarray,
    family_timestamps: np.ndarray,
    family_days: np.ndarray,
    family_symbols: np.ndarray,
    family_last_timestamp: np.ndarray,
    family_last_day: np.ndarray,
    family_asset_seen: np.ndarray,
) -> None:
    """Apply one resolved candidate to all of its collapsed family rows."""

    for index in range(len(family_codes)):
        code = family_codes[index]
        family_rows[code] += 1.0
        family_successes[code] += success
        family_predictions[code] += prediction
        family_nets[code] += net_bps
        family_expecteds[code] += base_expected_bps
        if prediction >= 0.5 and success <= 0.0 and net_bps < 0.0:
            family_false_positive_losses[code] += -net_bps
        if family_last_timestamp[code] != decision_timestamp_ns:
            family_timestamps[code] += 1.0
            family_last_timestamp[code] = decision_timestamp_ns
        if family_last_day[code] != decision_day:
            family_days[code] += 1.0
            family_last_day[code] = decision_day
        if not family_asset_seen[code, asset_code]:
            family_asset_seen[code, asset_code] = True
            family_symbols[code] += 1.0


@njit(cache=True)
def update_h1_block(
    family_codes: np.ndarray,
    contributions: np.ndarray,
    offsets: np.ndarray,
    successes: np.ndarray,
    predictions: np.ndarray,
    nets: np.ndarray,
    expecteds: np.ndarray,
    decision_timestamps_ns: np.ndarray,
    decision_days: np.ndarray,
    asset_codes: np.ndarray,
    family_rows: np.ndarray,
    family_successes: np.ndarray,
    family_predictions: np.ndarray,
    family_nets: np.ndarray,
    family_expecteds: np.ndarray,
    family_false_positive_losses: np.ndarray,
    family_timestamps: np.ndarray,
    family_days: np.ndarray,
    family_symbols: np.ndarray,
    family_last_timestamp: np.ndarray,
    family_last_day: np.ndarray,
    family_asset_seen: np.ndarray,
) -> None:
    """Apply all resolved candidates at one label timestamp in compiled code."""

    for candidate in range(len(offsets) - 1):
        success = successes[candidate]
        prediction = predictions[candidate]
        net_bps = nets[candidate]
        expected = expecteds[candidate]
        timestamp = decision_timestamps_ns[candidate]
        day = decision_days[candidate]
        asset = asset_codes[candidate]
        for index in range(offsets[candidate], offsets[candidate + 1]):
            code = family_codes[index]
            family_rows[code] += 1.0
            family_successes[code] += success
            family_predictions[code] += prediction
            family_nets[code] += net_bps
            family_expecteds[code] += expected
            if prediction >= 0.5 and success <= 0.0 and net_bps < 0.0:
                family_false_positive_losses[code] += -net_bps
            if family_last_timestamp[code] != timestamp:
                family_timestamps[code] += 1.0
                family_last_timestamp[code] = timestamp
            if family_last_day[code] != day:
                family_days[code] += 1.0
                family_last_day[code] = day
            if not family_asset_seen[code, asset]:
                family_asset_seen[code, asset] = True
                family_symbols[code] += 1.0


def allocate_h1_state(capacity: int, asset_count: int) -> dict[str, np.ndarray]:
    """Allocate compact, typed mutable H1 state for one side/head scope."""

    if int(capacity) <= 0 or int(asset_count) <= 0:
        raise ValueError("H1 state capacity and asset count must be positive")
    return {
        "family_rows": np.zeros(capacity, dtype=np.float64),
        "family_successes": np.zeros(capacity, dtype=np.float64),
        "family_predictions": np.zeros(capacity, dtype=np.float64),
        "family_nets": np.zeros(capacity, dtype=np.float64),
        "family_expecteds": np.zeros(capacity, dtype=np.float64),
        "family_false_positive_losses": np.zeros(capacity, dtype=np.float64),
        "family_timestamps": np.zeros(capacity, dtype=np.float64),
        "family_days": np.zeros(capacity, dtype=np.float64),
        "family_symbols": np.zeros(capacity, dtype=np.float64),
        "family_last_timestamp": np.full(capacity, np.iinfo(np.int64).min, dtype=np.int64),
        "family_last_day": np.full(capacity, np.iinfo(np.int32).min, dtype=np.int32),
        "family_asset_seen": np.zeros((capacity, asset_count), dtype=np.bool_),
    }


__all__ = [
    "H1_METRIC_COUNT", "allocate_h1_state", "score_auxiliary_block",
    "score_h1_block", "score_h1_candidate", "update_h1_block", "update_h1_candidate",
]
