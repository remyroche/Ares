from extreme_price_movements.universe import (
    _binance_get_json,
    _is_supported_training_symbol,
    _normalize_symbol,
    apply_hardcoded_universe_exclusions,
    filter_symbols_without_perp_support,
    margin_pairs_to_spot_symbols,
)


def test_normalize_symbol_handles_underscore_and_compact_forms():
    assert _normalize_symbol("ETH_USDT") == "ETH/USDT"
    assert _normalize_symbol("ETHUSDT") == "ETH/USDT"
    assert _normalize_symbol("ETH/USDT") == "ETH/USDT"


def test_supported_training_symbol_rejects_unsupported_quotes():
    assert _is_supported_training_symbol("ETH/USDC")
    assert not _is_supported_training_symbol("ETH_USDT")
    assert not _is_supported_training_symbol("BTC/USD1")
    assert not _is_supported_training_symbol("BNBFDUSD")
    assert not _is_supported_training_symbol("AAVE/BTC")


def test_apply_hardcoded_universe_exclusions_filters_aliases_and_unsupported_quotes():
    out = apply_hardcoded_universe_exclusions(
        [
            "ETHUSDC",
            "ETH_USDC",
            "ETH/USDC",
            "币安人生/USDC",
            "BTC/USD1",
            "BNBFDUSD",
            "CHESSUSDC",
            "FRAX_USDC",
        ]
    )
    assert out == ["ETH/USDC"]


def test_margin_pairs_to_spot_symbols_defaults_to_usdc():
    pairs = [{"symbol": "BTCUSDT"}, {"symbol": "BTCUSDC"}, {"symbol": "ETHUSDC"}]

    out = margin_pairs_to_spot_symbols(pairs)

    assert out == ["BTC/USDC", "ETH/USDC"]


def test_filter_symbols_without_perp_support_matches_by_base(monkeypatch):
    import extreme_price_movements.universe as universe

    monkeypatch.setattr(
        universe,
        "get_available_perp_spot_symbols",
        lambda: {"BTC/USDT", "ETH/USDC", "DOGE/USDC"},
    )

    out = filter_symbols_without_perp_support(
        ["BTC/USDC", "ETH/USDC", "XRP/USDC", "DOGE/USDT", "USDC/BNB"]
    )

    assert out == ["BTC/USDC", "DOGE/USDT", "ETH/USDC"]


def test_binance_get_json_retries_after_rate_limit(monkeypatch):
    import requests

    import extreme_price_movements.universe as universe

    calls = []
    sleeps = []

    class _Response:
        def __init__(self, status_code, payload, headers=None):
            self.status_code = status_code
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                exc = requests.HTTPError(f"{self.status_code} error")
                exc.response = self
                raise exc

        def json(self):
            return self._payload

    def _fake_get(url, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            return _Response(429, {}, {"Retry-After": "0"})
        return _Response(200, {"ok": True})

    monkeypatch.setattr(universe.requests, "get", _fake_get)
    monkeypatch.setattr(universe.time, "sleep", lambda seconds: sleeps.append(seconds))

    assert _binance_get_json("/api/v3/test") == {"ok": True}
    assert len(calls) == 2
    assert sleeps == [0.0]
