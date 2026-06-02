from extreme_price_movements.data_store import scoped_data_root


def test_scoped_data_root_is_idempotent_for_exchange_scoped_perp_root():
    cfg = {
        "data_root": "data_perp/exchanges/krakenfutures",
        "exchange_id": "krakenfutures",
        "market_mode": "perps",
    }

    assert scoped_data_root(cfg) == "data_perp/exchanges/krakenfutures"


def test_scoped_data_root_scopes_plain_perp_root():
    cfg = {
        "data_root": "data_perp",
        "exchange_id": "krakenfutures",
        "market_mode": "perps",
    }

    assert scoped_data_root(cfg) == "data_perp/exchanges/krakenfutures"
