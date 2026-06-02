from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements import data_store


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.requests.append(
            {
                "url": url,
                "params": params,
                "timeout": timeout,
                "headers": headers,
            }
        )
        return _FakeResponse(self.payload)


class _FakeExchange:
    def market(self, symbol):
        return {"id": "PF_CYBERUSD"}


def test_kraken_futures_1m_charts_preserve_zero_volume_carry_candles(monkeypatch):
    start = pd.Timestamp("2026-05-22T00:00:00Z")
    payload = {
        "candles": [
            {
                "time": int((start + pd.Timedelta(minutes=i)).value // 10**6),
                "open": "0.4705",
                "high": "0.4705",
                "low": "0.4705",
                "close": "0.4705",
                "volume": "0",
            }
            for i in range(3)
        ]
    }
    session = _FakeSession(payload)
    monkeypatch.setattr(data_store, "_public_data_session", lambda: session)

    out = data_store._fetch_kraken_futures_charts_ohlcv(
        _FakeExchange(),
        "CYBER/USD:USD",
        int(start.value // 10**6),
        int((start + pd.Timedelta(minutes=3)).value // 10**6),
        timeframe="1m",
        tick_type="trade",
    )

    assert len(out) == 3
    assert list(out.index) == list(pd.date_range(start, periods=3, freq="1min"))
    assert out["volume"].sum() == 0.0
    assert out["close"].tolist() == pytest.approx([0.4705, 0.4705, 0.4705])
