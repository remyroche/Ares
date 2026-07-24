import threading
import time

import pandas as pd

from extreme_price_movements.raw_market_data_contract import (
    RAW_MARKET_DATA_CONTRACT_VERSION,
    load_raw_market_panel,
    refresh_raw_market_history,
    refresh_raw_market_rows,
    repair_hourly_from_complete_15m,
)


class _MemoryStore:
    def __init__(self, frames=None):
        self.frames = dict(frames or {})
        self.saved = []

    def load(self, symbol, columns=None, start_ts=None, end_ts=None):
        frame = self.frames.get(symbol, pd.DataFrame()).copy()
        if start_ts is not None and not frame.empty:
            frame = frame.loc[frame.index >= pd.Timestamp(start_ts)]
        if end_ts is not None and not frame.empty:
            frame = frame.loc[frame.index <= pd.Timestamp(end_ts)]
        return frame

    def save_partitioned(self, symbol, df, defer_compact=False):
        self.saved.append((symbol, df.copy()))
        current = self.frames.get(symbol, pd.DataFrame())
        self.frames[symbol] = pd.concat([current, df]).sort_index()


def test_hourly_repair_requires_four_candles_and_never_overwrites():
    index = pd.date_range("2026-07-01T00:00:00Z", periods=7, freq="15min")
    frame = pd.DataFrame(
        {
            "open": range(7),
            "high": range(1, 8),
            "low": range(7),
            "close": range(1, 8),
            "volume": [1.0] * 7,
        },
        index=index,
    )
    existing = pd.DataFrame(
        {
            "open": [99.0],
            "high": [100.0],
            "low": [98.0],
            "close": [99.5],
            "volume": [10.0],
        },
        index=pd.DatetimeIndex([index[0]]),
    )
    store = _MemoryStore({"BTC/USD:USD": existing})

    repaired = repair_hourly_from_complete_15m(
        store=store,
        symbol="BTC/USD:USD",
        frame_15m=frame,
    )

    assert repaired.empty
    assert store.saved == []

    complete = pd.concat(
        [
            frame,
            pd.DataFrame(
                {
                    "open": [7],
                    "high": [8],
                    "low": [7],
                    "close": [8],
                    "volume": [1.0],
                },
                index=pd.DatetimeIndex([pd.Timestamp("2026-07-01T01:45:00Z")]),
            ),
        ]
    )
    repaired = repair_hourly_from_complete_15m(
        store=store,
        symbol="BTC/USD:USD",
        frame_15m=complete,
    )

    assert repaired.index.tolist() == [pd.Timestamp("2026-07-01T01:00:00Z")]
    assert len(store.saved) == 1
    assert store.frames["BTC/USD:USD"].loc[index[0], "open"] == 99.0


def test_live_refresh_contract_is_read_only_when_requested():
    class _Fetcher:
        def fetch_hourly_universe_once(self, *args, **kwargs):
            raise AssertionError("read-only refresh must not fetch")

    result = refresh_raw_market_rows(
        fetcher=_Fetcher(),
        symbols=["ETH/USD:USD", "BTC/USD:USD"],
        read_only=True,
        max_workers=8,
    )

    assert result.read_only is True
    assert result.updated_symbols == ()
    assert result.max_workers == 2
    assert result.contract["version"] == RAW_MARKET_DATA_CONTRACT_VERSION


def test_live_refresh_forwards_bounded_microdata_lookback():
    captured = {}

    class _Fetcher:
        def fetch_hourly_universe_once(self, *args, **kwargs):
            captured.update(kwargs)
            return {}

    refresh_raw_market_rows(
        fetcher=_Fetcher(),
        symbols=["BTC/USD:USD"],
        microdata_lookback_hours=72,
        microdata_allow_live_snapshot=False,
    )

    assert captured["microdata_lookback_hours"] == 72
    assert captured["microdata_allow_live_snapshot"] is False


def test_historical_refresh_uses_bounded_concurrent_workers():
    class _Store:
        def __init__(self):
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def update_symbol_perp(self, exchange, symbol, since_ms, spot_exchange=None):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.lock:
                self.active -= 1

    store = _Store()
    result = refresh_raw_market_history(
        store=store,
        exchange=object(),
        symbols=["C/USD:USD", "A/USD:USD", "B/USD:USD"],
        since_ts=pd.Timestamp("2026-01-01T00:00:00Z"),
        max_workers=3,
    )

    assert result.updated_symbols == ("A/USD:USD", "B/USD:USD", "C/USD:USD")
    assert result.max_workers == 3
    assert store.max_active > 1


def test_panel_loader_is_utc_deterministic_and_merges_microdata():
    naive = pd.date_range("2026-07-01", periods=2, freq="1h")
    frames = {
        "B/USD:USD": pd.DataFrame({"close": [2.0, 3.0]}, index=naive),
        "A/USD:USD": pd.DataFrame({"close": [1.0, 2.0]}, index=naive),
    }
    store = _MemoryStore(frames)

    panel = load_raw_market_panel(
        store=store,
        symbols=["B/USD:USD", "A/USD:USD"],
        panel_fields=["close"],
        max_workers=2,
        microdata_loader=lambda symbols, start, end: {
            "open_interest": pd.DataFrame(
                {symbol: [10.0, 11.0] for symbol in symbols},
                index=pd.DatetimeIndex(naive).tz_localize("UTC"),
            )
        },
    )

    assert panel["close"].columns.tolist() == ["A/USD:USD", "B/USD:USD"]
    assert str(panel["close"].index.tz) == "UTC"
    assert panel["open_interest"].columns.tolist() == [
        "A/USD:USD",
        "B/USD:USD",
    ]
