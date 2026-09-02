"""Directional regression tests for the canonical short policy labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_r3_frozen_policy_labels import (
    replay_frozen_policy_15m,
)


def _bars(*, first_high: float, first_low: float, later_high: float, later_low: float) -> pd.DataFrame:
    index = pd.date_range("2025-01-01T00:00:00Z", periods=48, freq="15min")
    return pd.DataFrame(
        {
            "open": np.full(48, 100.0),
            "high": np.r_[first_high, np.full(47, later_high)],
            "low": np.r_[first_low, np.full(47, later_low)],
            "close": np.full(48, 100.0),
            "volume": np.ones(48),
        },
        index=index,
    )


def _candidate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["short-1"],
            "__decision_ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "side_name": ["short"],
            "atr_1h": [1.0],
        }
    )


def test_short_policy_stop_is_above_entry_and_has_negative_economics() -> None:
    result = replay_frozen_policy_15m(
        _candidate(),
        _bars(first_high=103.0, first_low=99.0, later_high=100.0, later_low=100.0),
    ).iloc[0]
    assert result["policy_exit_reason"] == "stop_loss"
    assert result["policy_exit_price"] == pytest.approx(103.0)
    assert result["policy_gross_bps"] == pytest.approx(-300.0)
    assert result["policy_net_bps"] == pytest.approx(-400.0)


def test_short_policy_trailing_locks_a_prior_favourable_low() -> None:
    # The first bar creates 2 ATR favourable movement but may not trigger its
    # own trail. The second bar rebounds through the 1.75 ATR locked stop.
    bars = _bars(first_high=100.1, first_low=98.0, later_high=98.25, later_low=98.2)
    result = replay_frozen_policy_15m(_candidate(), bars).iloc[0]
    assert result["policy_exit_reason"] == "trailing"
    assert result["policy_exit_price"] == pytest.approx(98.25)
    assert result["policy_gross_bps"] == pytest.approx(175.0)
    assert result["policy_net_bps"] == pytest.approx(75.0)
