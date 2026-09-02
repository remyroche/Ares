from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.inference.strict_r3_live_execution import (
    _fetch_exchange_positions_for_monitor,
)


def _tracked_position() -> dict:
    return {
        "candidate_id": "TEST/USD:USD|long|2026-08-20T00:00:00Z",
        "exchange_symbol": "TEST/USD:USD",
        "side": "long",
        "amount": 10.0,
        "entry_ts": "2026-08-20T00:00:00Z",
        "entry_fill_ts": "2026-08-20T00:00:01Z",
        "entry_order_id": "entry-1",
    }


class _OpenPositions503WithFills:
    def fetch_positions(self):
        raise RuntimeError(
            "krakenfutures GET https://futures.kraken.com/derivatives/api/v3/"
            "openpositions 503 Service Unavailable"
        )

    def fetch_my_trades(self, symbol, since, limit):
        assert symbol is None
        assert since is None
        assert limit == 100
        return [{
            "symbol": "TEST/USD:USD",
            "side": "buy",
            "amount": 10.0,
            "timestamp": int(pd.Timestamp("2026-08-20T00:00:01Z").timestamp() * 1000),
            "order": "entry-1",
        }]


def test_openpositions_503_reconstructs_a_fully_covered_tracked_position():
    position = _tracked_position()
    positions, provenance = _fetch_exchange_positions_for_monitor(
        _OpenPositions503WithFills(),
        tracked_positions={position["candidate_id"]: position},
        allow_503_fills_fallback=True,
    )

    assert list(positions) == ["TEST/USD:USD"]
    assert positions["TEST/USD:USD"]["contracts"] == 10.0
    assert provenance["source"] == "kraken_private_fills_503_fallback"
    assert provenance["fallback_used"] is True
    assert provenance["tracked_positions"][0]["status"] == "open"


def test_openpositions_503_rejects_an_incomplete_fill_ledger():
    class Exchange(_OpenPositions503WithFills):
        def fetch_my_trades(self, symbol, since, limit):
            return []

    position = _tracked_position()
    with pytest.raises(ValueError, match="cannot prove entry coverage"):
        _fetch_exchange_positions_for_monitor(
            Exchange(),
            tracked_positions={position["candidate_id"]: position},
            allow_503_fills_fallback=True,
        )


def test_non_503_openpositions_error_never_uses_the_fills_fallback():
    class Exchange(_OpenPositions503WithFills):
        def fetch_positions(self):
            raise RuntimeError("krakenfutures openpositions 429 Too Many Requests")

    position = _tracked_position()
    with pytest.raises(RuntimeError, match="429"):
        _fetch_exchange_positions_for_monitor(
            Exchange(),
            tracked_positions={position["candidate_id"]: position},
            allow_503_fills_fallback=True,
        )


def test_503_fallback_rejects_a_partial_or_mismatched_position():
    class Exchange(_OpenPositions503WithFills):
        def fetch_my_trades(self, symbol, since, limit):
            return [{
                "symbol": "TEST/USD:USD",
                "side": "buy",
                "amount": 10.0,
                "timestamp": int(pd.Timestamp("2026-08-20T00:00:01Z").timestamp() * 1000),
                "order": "entry-1",
            }, {
                "symbol": "TEST/USD:USD",
                "side": "sell",
                "amount": 3.0,
                "timestamp": int(pd.Timestamp("2026-08-20T00:01:01Z").timestamp() * 1000),
                "order": "manual-partial",
            }]

    position = _tracked_position()
    with pytest.raises(ValueError, match="ambiguous signed amount"):
        _fetch_exchange_positions_for_monitor(
            Exchange(),
            tracked_positions={position["candidate_id"]: position},
            allow_503_fills_fallback=True,
        )
