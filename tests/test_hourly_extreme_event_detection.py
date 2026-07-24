from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.hourly_extreme_event_detection import (
    HourlyEventConfig,
    assert_hourly_only,
    build_hourly_market_state,
    calendar_hourly_targets,
    causal_episode_memory,
)


def test_hourly_contract_rejects_subhour_columns() -> None:
    with pytest.raises(ValueError, match="sub-hour"):
        assert_hourly_only(["mkt_ret_1h", "mkt_ret_15m"])


def test_hourly_state_is_lagged_before_transition_transforms() -> None:
    hours = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "__ts__": np.repeat(hours, 2),
            "mkt_oi_chg_1h": np.repeat(np.arange(30, dtype=np.float32), 2),
        }
    )
    state = build_hourly_market_state(rows, feature_columns=["mkt_oi_chg_1h"])
    # At t=02:00, the last observable raw value is t=01:00.
    assert state.loc[2, "mkt_oi_chg_1h"] == 1.0
    # The transition is also calculated only after that one-hour lag.
    assert state.loc[2, "evt_mkt_oi_chg_1h__delta_1h"] == 1.0
    assert "evt_causal_change_score" in state.columns


def test_calendar_has_one_hourly_onset_per_contiguous_event_block() -> None:
    hours = pd.date_range("2026-01-01", periods=72, freq="h", tz="UTC")
    hourly = pd.DataFrame({"__ts__": hours, "day": hours.floor("D")})
    calendar = pd.DataFrame(
        {
            "day": [pd.Timestamp("2026-01-02", tz="UTC")],
            "side_name": ["short"],
            "archetype_policy_key": ["short_default"],
            "adverse_calendar_cell": [1],
        }
    )
    taxonomy = pd.DataFrame(
        {
            "event_start": [pd.Timestamp("2026-01-02", tz="UTC")],
            "event_end": [pd.Timestamp("2026-01-02", tz="UTC")],
            "onset_primary_mechanism": ["liquidation_pressure"],
        }
    )
    labels = calendar_hourly_targets(
        hourly,
        calendar,
        taxonomy,
        config=HourlyEventConfig(lead_hours=2),
    )
    assert labels["event_state"].sum() == 24
    assert labels["event_onset"].sum() == 1
    onset_index = int(np.flatnonzero(labels["event_onset"].to_numpy())[0])
    assert labels.loc[onset_index, "__ts__"] == pd.Timestamp("2026-01-02", tz="UTC")
    assert labels.loc[onset_index - 1, "event_onset_next_window"] == 1
    assert labels.loc[onset_index - 2, "event_onset_next_window"] == 1
    assert labels.loc[onset_index - 3, "event_onset_next_window"] == 0
    assert labels.loc[onset_index, "mechanism__liquidation_pressure__onset"] == 1
    assert labels.loc[onset_index - 1, "mechanism__liquidation_pressure__pre_onset_next_window"] == 1
    assert not any(name.startswith("mechanism__") for name in hourly.columns)


def test_episode_memory_uses_only_previous_prediction() -> None:
    memory = causal_episode_memory(
        np.asarray([0.0, 0.95, 0.10], dtype=np.float32),
        {"liquidation_pressure": np.asarray([0.2, 0.8, 0.1], dtype=np.float32)},
        threshold=0.9,
    )
    values = memory["event_memory__liquidation_pressure"].to_numpy()
    assert values[0] == 0.0
    assert values[1] == 0.0
    assert values[2] == pytest.approx(0.8)
