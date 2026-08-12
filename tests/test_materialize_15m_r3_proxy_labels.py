"""Correctness tests for the declared coarse 15m R3 label proxy."""

import numpy as np
import pandas as pd

from scripts.materialize_15m_r3_proxy_labels import _path_labels


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "__ts__": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "side_name": ["long"],
            "atr_bps": [100.0],
        }
    )


def _bars(close: np.ndarray, high: np.ndarray | None = None, low: np.ndarray | None = None):
    n = len(close)
    high = close.copy() if high is None else high
    low = close.copy() if low is None else low
    ts = pd.date_range("2025-01-01T01:00:00Z", periods=n, freq="15min").asi8
    return ts, close.copy(), high, low, close.copy()


def test_no_touch_is_timeout_not_adverse() -> None:
    close = np.full(48, 100.0)
    ts, op, hi, lo, cl = _bars(close)
    out = _path_labels(_rows(), ts=ts, opens=op, highs=hi, lows=lo, closes=cl)
    assert int(out.t2_tp6_sl4_event_proxy_15m.iloc[0]) == 1
    assert float(out.gross_bps_proxy_15m.iloc[0]) == 0.0
    assert int(out.lower_touch_minute_proxy_15m.iloc[0]) == -1


def test_adverse_and_upper_first_touch_are_distinct() -> None:
    close = np.full(48, 100.0)
    high = close.copy()
    low = close.copy()
    # 4 ATR adverse touch at bar 2.
    low[2] = 96.0
    ts, op, hi, lo, cl = _bars(close, high, low)
    out = _path_labels(_rows(), ts=ts, opens=op, highs=hi, lows=lo, closes=cl)
    assert int(out.t2_tp6_sl4_event_proxy_15m.iloc[0]) == 0
    assert float(out.gross_bps_proxy_15m.iloc[0]) == -400.0

    high[2] = 106.0
    low[2] = 100.0
    ts, op, hi, lo, cl = _bars(close, high, low)
    out = _path_labels(_rows(), ts=ts, opens=op, highs=hi, lows=lo, closes=cl)
    assert int(out.t2_tp6_sl4_event_proxy_15m.iloc[0]) == 2
    assert float(out.gross_bps_proxy_15m.iloc[0]) == 600.0


def test_cost_is_applied_exactly_once() -> None:
    close = np.full(48, 100.0)
    close[-1] = 101.5
    ts, op, hi, lo, cl = _bars(close)
    out = _path_labels(_rows(), ts=ts, opens=op, highs=hi, lows=lo, closes=cl)
    assert np.isclose(
        float(out.gross_bps_proxy_15m.iloc[0]) - float(out.net_bps_proxy_15m.iloc[0]),
        100.0,
    )
