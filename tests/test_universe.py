from extreme_price_movements.universe import (
    _is_supported_training_symbol,
    _normalize_symbol,
    apply_hardcoded_universe_exclusions,
)


def test_normalize_symbol_handles_underscore_and_compact_forms():
    assert _normalize_symbol("ETH_USDT") == "ETH/USDT"
    assert _normalize_symbol("ETHUSDT") == "ETH/USDT"
    assert _normalize_symbol("ETH/USDT") == "ETH/USDT"


def test_supported_training_symbol_rejects_unsupported_quotes():
    assert _is_supported_training_symbol("ETH/USDT")
    assert _is_supported_training_symbol("ETH_USDC")
    assert not _is_supported_training_symbol("BTC/USD1")
    assert not _is_supported_training_symbol("BNBFDUSD")
    assert not _is_supported_training_symbol("AAVE/BTC")


def test_apply_hardcoded_universe_exclusions_filters_aliases_and_unsupported_quotes():
    out = apply_hardcoded_universe_exclusions(
        [
            "ETHUSDT",
            "ETH_USDT",
            "ETH/USDT",
            "BTC/USD1",
            "BNBFDUSD",
            "CHESSUSDT",
            "FRAX_USDT",
        ]
    )
    assert out == ["ETH/USDT"]
