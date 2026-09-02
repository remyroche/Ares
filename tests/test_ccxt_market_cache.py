"""Focused safety checks for the opt-in live CCXT market-definition cache."""

from __future__ import annotations

import json
import time

from extreme_price_movements import data_store


class _Exchange:
    def __init__(self) -> None:
        self.markets = {}
        self.currencies = {}

    def set_markets(self, markets, currencies=None) -> None:
        self.markets = dict(markets)
        self.currencies = dict(currencies or {})


def test_ccxt_market_cache_restores_only_matching_fresh_metadata(monkeypatch, tmp_path):
    path = tmp_path / "kraken_markets.json"
    path.write_text(json.dumps({
        "schema": "epm_ccxt_market_cache_v1",
        "exchange_id": "kraken",
        "created_at_unix": time.time(),
        "markets": {
            "BTC/USD:USD": {
                "id": "PF_XBTUSD",
                "symbol": "BTC/USD:USD",
                "contractSize": 1.0,
                "precision": {"amount": 0},
            },
        },
        "currencies": {"USD": {"id": "USD"}},
    }))
    monkeypatch.setenv("EPM_CCXT_MARKETS_CACHE", str(path))
    monkeypatch.setenv("EPM_CCXT_MARKETS_CACHE_MAX_AGE_SECONDS", "60")
    exchange = _Exchange()

    assert data_store._restore_ccxt_market_cache(exchange, exchange_id="kraken")
    assert exchange.markets["BTC/USD:USD"]["id"] == "PF_XBTUSD"
    assert exchange.currencies["USD"]["id"] == "USD"


def test_ccxt_market_cache_rejects_stale_or_wrong_exchange(monkeypatch, tmp_path):
    path = tmp_path / "kraken_markets.json"
    path.write_text(json.dumps({
        "schema": "epm_ccxt_market_cache_v1",
        "exchange_id": "kraken",
        "created_at_unix": time.time() - 120.0,
        "markets": {"BTC/USD:USD": {"symbol": "BTC/USD:USD"}},
        "currencies": {},
    }))
    monkeypatch.setenv("EPM_CCXT_MARKETS_CACHE", str(path))
    monkeypatch.setenv("EPM_CCXT_MARKETS_CACHE_MAX_AGE_SECONDS", "60")

    assert not data_store._restore_ccxt_market_cache(_Exchange(), exchange_id="kraken")
    assert not data_store._restore_ccxt_market_cache(_Exchange(), exchange_id="okx")
