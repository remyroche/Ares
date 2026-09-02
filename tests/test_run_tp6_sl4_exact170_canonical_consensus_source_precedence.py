from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_tp6_sl4_exact170_canonical_consensus.py"
SPEC = importlib.util.spec_from_file_location("canonical_consensus_source_precedence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_phase_zero_source_precedence_is_horizon_invariant(
    monkeypatch,
) -> None:
    """A later coarse-bar gap must not alter prior raw/cache source choice.

    The raw 15-minute source has complete OHLCV over the short window but no
    mark field.  A later raw gap used to make the all-window shortcut open the
    canonical cache, retroactively adding early mark values.  The fixed
    per-cell source chain must expose the same early vector in both horizons.
    """
    symbol = "A/USD:USD"
    index = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
    raw = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, np.nan, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, np.nan, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, np.nan, 103.0, 104.0],
            "close": [100.5, 101.5, 102.5, np.nan, 104.5, 105.5],
            "volume": [10.0, 11.0, 12.0, np.nan, 14.0, 15.0],
        },
        index=index,
    )
    cache = pd.DataFrame(
        {
            "open": [200.0] * 6,
            "high": [201.0] * 6,
            "low": [199.0] * 6,
            "close": [200.5] * 6,
            "volume": [20.0] * 6,
            "mark_price": [300.0, 301.0, 302.0, 303.0, 304.0, 305.0],
        },
        index=index,
    )

    def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return frame.loc[(frame.index >= start) & (frame.index < end)].copy()

    monkeypatch.setattr(MODULE, "_source_map", lambda _symbols: {symbol: None})
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda _sym, start, end, **_kwargs: _window(raw, start, end))
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda _sym, start, end: _window(cache, start, end))
    monkeypatch.setattr(MODULE, "_read_official_trade_hourly", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda *_args, **_kwargs: None)

    start = index[0]
    short_end = index[3]
    long_end = index[-1] + pd.Timedelta(hours=1)
    short, _ = MODULE._make_panel([symbol], start, short_end)
    long, _ = MODULE._make_panel([symbol], start, long_end)

    for field in ("open", "high", "low", "close", "volume", "mark_price"):
        pd.testing.assert_series_equal(
            short[field][symbol], long[field].loc[short[field].index, symbol],
            check_names=False,
        )
    # Downloaded 15-minute OHLCV wins where present; the cache fills the
    # later missing row and provides its additional causal mark primitive.
    assert short["close"].loc[index[1], symbol] == 101.5
    assert short["mark_price"].loc[index[1], symbol] == 301.0
    assert long["close"].loc[index[3], symbol] == 200.5


def test_partial_official_archive_cannot_suppress_older_hourly_fallback(
    monkeypatch,
) -> None:
    """A later official archive must not erase an earlier fallback prefix."""
    symbol = "A/USD:USD"
    index = pd.date_range("2025-01-01", periods=6, freq="1h", tz="UTC")
    legacy = pd.DataFrame(
        {
            "open": np.arange(100.0, 106.0),
            "high": np.arange(101.0, 107.0),
            "low": np.arange(99.0, 105.0),
            "close": np.arange(100.5, 106.5),
            "volume": np.arange(10.0, 16.0),
        },
        index=index,
    )
    official = legacy.iloc[3:].copy()
    official.loc[:, "close"] = official["close"] + 100.0

    def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
        result = frame.loc[(frame.index >= start) & (frame.index < end)].copy()
        return result if not result.empty else None

    monkeypatch.setattr(MODULE, "_source_map", lambda _symbols: {symbol: "A_USD"})
    monkeypatch.setattr(MODULE, "_read_downloaded_15m_hourly", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "_read_canonical_input_cache", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(MODULE, "_read_official_trade_hourly", lambda _sym, start, end: _window(official, start, end))
    monkeypatch.setattr(MODULE, "_read_hourly_source", lambda _source, start, end: _window(legacy, start, end))

    short_end = index[3]
    long_end = index[-1] + pd.Timedelta(hours=1)
    short, _ = MODULE._make_panel([symbol], index[0], short_end)
    long, _ = MODULE._make_panel([symbol], index[0], long_end)
    pd.testing.assert_series_equal(
        short["close"][symbol], long["close"].loc[short["close"].index, symbol],
        check_names=False,
    )
    # The newer official source wins where it actually exists, while the
    # prior legacy prefix remains available in the longer materialisation.
    assert long["close"].loc[index[1], symbol] == legacy.loc[index[1], "close"]
    assert long["close"].loc[index[4], symbol] == official.loc[index[4], "close"]
