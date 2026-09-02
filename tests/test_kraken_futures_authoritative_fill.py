from __future__ import annotations

import pytest

from extreme_price_movements.inference.trade_executor import (
    _enrich_order_from_exchange,
    _extract_order_fill,
)


class _KrakenIocClosedOrderExchange:
    """Kraken Futures shape: closed order price is not the executed fill."""

    id = "krakenfutures"

    def fetch_my_trades(self, symbol, since=None, limit=None):
        assert symbol == "XMR/USD:USD"
        return [
            {
                "id": "fill-1",
                "order": "close-order",
                "symbol": symbol,
                "side": "sell",
                "price": 420.24,
                "amount": 0.04,
                # A zero fee is enough to reproduce the former early-return
                # path that skipped the authoritative fill lookup.
                "fee": {"cost": 0.0, "currency": "USD"},
                "info": {"order_id": "close-order"},
            }
        ]


def test_kraken_futures_private_fills_override_ioc_limit_price_even_with_fee() -> None:
    order = _enrich_order_from_exchange(
        _KrakenIocClosedOrderExchange(),
        {
            "id": "close-order",
            "symbol": "XMR/USD:USD",
            "side": "sell",
            "type": "market",
            "status": "closed",
            # This is Kraken's IOC limitPrice field, not the executed VWAP.
            "average": 416.03,
            "price": 416.03,
            "filled": 0.04,
            "fee": {"cost": 0.0, "currency": "USD"},
        },
        symbol="XMR/USD:USD",
        config={"market_mode": "perps"},
        price=420.0,
    )

    fill_price, filled, partial = _extract_order_fill(order, 420.0)

    assert order["fee_source_order_fetch"] == "fetch_my_trades"
    assert order["average"] == pytest.approx(420.24)
    assert fill_price == pytest.approx(420.24)
    assert filled == pytest.approx(0.04)
    assert partial is False
