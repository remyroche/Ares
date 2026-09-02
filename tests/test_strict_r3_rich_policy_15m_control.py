"""Focused properties for the research-only aggregated 15m rich control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.strict_r3_rich_policy import RichPolicyParams
from extreme_price_movements.strict_r3_rich_policy_15m_control import (
    FrozenRich15mAggregationContract,
    aggregate_exact_1m_to_15m_ohlc,
    replay_frozen_rich_policy_15m_aggregate,
)


def _paths() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.full((1, 720), 100.0),
        np.full((1, 720), 100.0),
        np.full((1, 720), 100.0),
    )


def _replay(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    params: RichPolicyParams,
) -> dict[str, np.ndarray]:
    return replay_frozen_rich_policy_15m_aggregate(
        entry=np.array([100.0]), atr=np.array([1.0]),
        highs=high, lows=low, closes=close,
        entry_timestamps=pd.DatetimeIndex(["2026-08-17T00:00:00Z"]),
        params=params, median_atr_fraction=0.01,
        contract=FrozenRich15mAggregationContract(),
    )


def _base(**overrides: object) -> RichPolicyParams:
    values: dict[str, object] = {
        "sl_mult": 5.0,
        "trailing_activation_mult": 0.5,
        "fixed_trailing_gap_mult": 0.25,
        "capital_protect_mfe_mult": 2.0,
        "capital_protect_lock_frac": 0.1,
        "adverse_exit_enabled": False,
    }
    values.update(overrides)
    return RichPolicyParams(**values)


def test_aggregation_and_flat_h12_path_use_completed_15m_end() -> None:
    high, low, close = _paths()
    high[0, 14] = 101.0
    low[0, 3] = 99.0
    close[0, 14] = 100.25
    high15, low15, close15 = aggregate_exact_1m_to_15m_ohlc(
        highs=high, lows=low, closes=close,
    )
    assert high15.shape == low15.shape == close15.shape == (1, 48)
    assert high15[0, 0] == pytest.approx(101.0)
    assert low15[0, 0] == pytest.approx(99.0)
    assert close15[0, 0] == pytest.approx(100.25)

    flat_high, flat_low, flat_close = _paths()
    result = _replay(flat_high, flat_low, flat_close, _base())
    assert result["path_valid"][0]
    assert result["exit_reason"][0] == "timeout_h12"
    assert result["exit_bar_15m"][0] == 47
    assert result["exit_price"][0] == pytest.approx(100.0)
    assert result["gross_bps"][0] == pytest.approx(0.0)
    assert result["net_bps"][0] == pytest.approx(-100.0)
    assert pd.Timestamp(result["exit_timestamp"][0], tz="UTC") == pd.Timestamp("2026-08-17T12:00:00Z")


def test_peak_in_aggregate_bar_arms_trailing_only_for_later_aggregate_bar() -> None:
    high, low, close = _paths()
    # The first 15m bar makes a +1 MFE.  It cannot arm and exit itself;
    # bar two arms from that prior aggregate peak; bar three may trail.
    high[0, 0] = 101.0
    close[0, 14] = 100.9
    low[0, 30] = 100.70
    close[0, 44] = 100.70
    result = _replay(high, low, close, _base())
    assert result["exit_reason"][0] == "trailing"
    assert result["exit_bar_15m"][0] == 2
    assert result["exit_price"][0] == pytest.approx(100.75)
    assert pd.Timestamp(result["exit_timestamp"][0], tz="UTC") == pd.Timestamp("2026-08-17T00:45:00Z")


def test_stop_then_capital_then_trailing_then_fast_adverse_precedence() -> None:
    # Stop is authoritative even if the rest of the future path would have
    # armed/triggered protection and trailing levels.
    high, low, close = _paths()
    high[0, 0] = 103.0
    low[0, 0] = 94.0
    stop = _replay(high, low, close, _base(adverse_exit_enabled=True, adverse_exit_theta=-1.0))
    assert stop["exit_reason"][0] == "stop_loss"
    assert stop["exit_price"][0] == pytest.approx(95.0)

    # A previously armed capital lock and trail are both crossed in bar three;
    # the capital lock wins under the live order of checks.
    high, low, close = _paths()
    high[0, 0] = 103.0
    close[0, 14] = 102.9
    low[0, 30] = 100.0
    capital = _replay(high, low, close, _base())
    assert capital["exit_reason"][0] == "capital_protect"
    assert capital["exit_price"][0] == pytest.approx(100.2)

    # With no earlier protective state, a fast-adverse exit is a completed-bar
    # close proxy and fires only after the preceding hard/protect/trail checks.
    high, low, close = _paths()
    low[0, 0] = 99.0
    close[0, 14] = 99.5
    fast = _replay(
        high, low, close,
        _base(
            adverse_exit_enabled=True, adverse_exit_theta=0.0,
            adverse_exit_min_mae_atr=0.2, adverse_exit_min_speed=0.1,
            adverse_exit_max_mfe_atr=0.25,
        ),
    )
    assert fast["exit_reason"][0] == "fast_adverse"
    assert fast["exit_price"][0] == pytest.approx(99.5)
