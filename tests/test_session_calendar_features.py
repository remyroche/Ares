from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.features import (
    SESSION_CALENDAR_FEATURE_KEYS,
    SESSION_MARKET_FEATURE_KEYS,
    market_session_features,
    session_calendar_features,
)


def test_session_calendar_features_respect_local_opens_and_weekends() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2026-01-05T00:00:00Z",  # Tokyo 09:00 JST
            "2026-01-05T14:30:00Z",  # New York 09:30 EST
            "2026-07-06T13:30:00Z",  # New York 09:30 EDT
            "2026-01-03T14:30:00Z",  # Saturday in Paris and New York
            "2026-01-05T21:00:00Z",  # New York 16:00 EST close
        ]
    )

    values = session_calendar_features(timestamps)

    assert tuple(values) == SESSION_CALENDAR_FEATURE_KEYS
    assert values["hours_from_asia_open_signed_log"][0] == 0.0
    assert values["hours_from_us_open_signed_log"][1] == 0.0
    assert values["hours_from_us_open_signed_log"][2] == 0.0
    assert values["is_weekend_binary"][3] == 1.0
    # Closed Europe/US sessions are represented neutrally and disambiguated by
    # the explicit weekend flag, preserving complete feature coverage.
    assert values["hours_from_europe_open_signed_log"][3] == 0.0
    assert values["us_open_proximity_hours_log"][3] == 0.0
    assert values["is_europe_us_overlap"][1] == 1.0
    assert values["is_us_active"][4] == 0.0
    assert values["hours_from_us_close_signed_log"][4] == 0.0
    assert np.isfinite(np.column_stack(list(values.values()))).all()


def test_session_calendar_features_are_prefix_invariant() -> None:
    prefix = pd.date_range("2026-03-27T10:00:00Z", periods=5, freq="15min")
    extended = prefix.append(
        pd.date_range("2026-03-27T11:15:00Z", periods=4, freq="15min")
    )

    first = session_calendar_features(prefix)
    second = session_calendar_features(extended)

    for name in SESSION_CALENDAR_FEATURE_KEYS:
        np.testing.assert_array_equal(first[name], second[name][: len(prefix)])


def test_market_session_features_are_causal_and_complete() -> None:
    index = pd.date_range("2026-01-01T00:00:00Z", periods=3_000, freq="15min")
    steps = np.arange(len(index), dtype=float)
    close = pd.DataFrame(
        {
            "A": 100.0 * np.exp(steps * 0.0001),
            "B": 50.0 * np.exp(steps * 0.0002),
        },
        index=index,
    )

    first = market_session_features(close.iloc[:2_000])
    second = market_session_features(close)

    assert tuple(first) == SESSION_MARKET_FEATURE_KEYS
    for name in SESSION_MARKET_FEATURE_KEYS:
        np.testing.assert_array_equal(
            first[name].to_numpy(), second[name].iloc[: len(close.iloc[:2_000])].to_numpy()
        )
        assert np.isfinite(second[name].to_numpy()).all()
