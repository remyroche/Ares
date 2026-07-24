from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.timestamp_contract import (
    assert_first_path_timestamp,
    causal_signal_times,
)
from scripts.run_label_first_touch_capture_proxy import _policy_rows
from extreme_price_movements.simple_policy_optimiser import _fetch_policy_paths


def test_hourly_decision_is_signal_close() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01 12:00:00+00:00", "2026-07-01 13:00:00+00:00"]
            )
        }
    )
    signal, decision = causal_signal_times(frame, timeframe="1h")
    assert signal.equals(pd.DatetimeIndex(frame["timestamp"]))
    assert decision.equals(signal + pd.Timedelta(hours=1))


def test_recorded_decision_cannot_precede_signal_close() -> None:
    frame = pd.DataFrame(
        {
            "signal_bar_ts": [pd.Timestamp("2026-07-01 12:00", tz="UTC")],
            "decision_ts": [pd.Timestamp("2026-07-01 12:59", tz="UTC")],
        }
    )
    with pytest.raises(ValueError, match="precedes signal_ts"):
        causal_signal_times(frame, timeframe="1h")


def test_first_path_timestamp_must_be_at_or_after_decision() -> None:
    signal = pd.to_datetime(["2026-07-01 12:00:00+00:00"])
    assert_first_path_timestamp(
        first_path_ts=signal + pd.Timedelta(hours=1),
        signal_ts=signal,
        timeframe="1h",
    )
    with pytest.raises(AssertionError, match="first_path_timestamp"):
        assert_first_path_timestamp(
            first_path_ts=signal,
            signal_ts=signal,
            timeframe="1h",
        )


def test_materialized_label_rows_keep_signal_and_record_decision() -> None:
    frame = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(["2026-07-01 12:00:00+00:00"]),
            "__symbol__": ["BTC/USD:USD"],
            "__barrier_pct__": np.asarray([0.02], dtype=np.float32),
        }
    )
    rows = _policy_rows(frame, side="long", timeframe="1h")
    assert rows.loc[0, "timestamp"] == pd.Timestamp("2026-07-01 12:00", tz="UTC")
    assert rows.loc[0, "signal_bar_ts"] == pd.Timestamp("2026-07-01 12:00", tz="UTC")
    assert rows.loc[0, "decision_ts"] == pd.Timestamp("2026-07-01 13:00", tz="UTC")


def test_shared_path_fetcher_records_actual_causal_first_bar() -> None:
    class Store:
        timeframe = "15m"

        def load(self, _symbol, *, start_ts, end_ts, columns=None):
            index = pd.date_range(start_ts, end_ts, freq="15min", tz="UTC")
            frame = pd.DataFrame(
                {name: np.ones(len(index), dtype=np.float32) for name in ("open", "high", "low", "close")},
                index=index,
            )
            frame["ts"] = frame.index
            return frame

    rows = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2026-07-01 12:00", tz="UTC")],
            "symbol": ["BTC/USD:USD"],
        }
    )
    paths = _fetch_policy_paths(rows, Store(), path_len=4, signal_timeframe="1h")
    assert all(np.isfinite(path).all() for path in paths)
    assert rows.loc[0, "first_path_timestamp"] == pd.Timestamp(
        "2026-07-01 13:00", tz="UTC"
    )
