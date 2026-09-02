from __future__ import annotations

import numpy as np

from extreme_price_movements.strict_r3_rich_policy import (
    RichPolicyParams,
    effective_atr_fraction,
    simulate_rich_policy,
)


def _paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.array([[101.0, 101.0, 101.0]], dtype=float)
    low = np.array([[100.0, 100.70, 100.70]], dtype=float)
    close = np.array([[100.8, 100.8, 100.8]], dtype=float)
    return high, low, close


def test_static_trailing_preserves_prior_bar_activation() -> None:
    high, low, close = _paths()
    out = simulate_rich_policy(
        entry=np.array([100.0]), atr=np.array([1.0]),
        highs=high, lows=low, closes=close,
        params=RichPolicyParams(sl_mult=3.0, trailing_activation_mult=0.5, fixed_trailing_gap_mult=0.25),
        median_atr_fraction=0.01,
    )
    # Bar 0 reaches activation, then bar 1 gives back through the 100.75 stop.
    assert out["exit_reason"][0] == "trailing"
    assert out["exit_bar"][0] == 1
    assert np.isclose(out["gross_bps"][0], 75.0)
    assert np.isclose(out["net_bps"][0], -25.0)


def test_absolute_stop_floor_binds_before_absolute_cap() -> None:
    high = np.array([[100.0]], dtype=float)
    low = np.array([[99.40]], dtype=float)
    close = np.array([[99.40]], dtype=float)
    out = simulate_rich_policy(
        entry=np.array([100.0]), atr=np.array([0.1]), highs=high, lows=low, closes=close,
        params=RichPolicyParams(sl_mult=1.0, sl_abs_floor_pct=0.005, sl_abs_cap_pct=0.004),
        median_atr_fraction=0.01,
    )
    # Invalid floor/cap ordering is deterministic: the floor wins.  The
    # resulting 0.50% stop is hit by the 0.60% intrabar decline.
    assert out["exit_reason"][0] == "stop_loss"
    assert np.isclose(out["gross_bps"][0], -50.0)


def test_separate_atr_power_changes_only_requested_geometry() -> None:
    raw = np.array([0.005, 0.02])
    legacy = effective_atr_fraction(raw, median_atr_fraction=0.01, power=1.0, multiplier=1.0)
    shaped = effective_atr_fraction(raw, median_atr_fraction=0.01, power=0.5, multiplier=1.0)
    assert shaped[0] > legacy[0]
    assert shaped[1] < legacy[1]
