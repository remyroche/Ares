from __future__ import annotations

from extreme_price_movements.pipeline_steps import _feature_subset_context_symbols


def test_strict_feature_subset_never_adds_external_context_symbols() -> None:
    requested = ["BTC/USDT", "ETH/USDT"]
    available = ["BTC/USD:BTC", "ETH/USD:ETH", "BTC/USD:USD"]
    output = ["BTC/USD:USD"]

    assert (
        _feature_subset_context_symbols(
            requested,
            available,
            output,
            strict_subset=True,
        )
        == []
    )


def test_default_feature_subset_preserves_available_context_expansion() -> None:
    requested = ["BTC/USDT", "ETH/USDT"]
    available = ["BTC/USD:BTC", "ETH/USD:ETH", "BTC/USD:USD"]
    output = ["BTC/USD:USD"]

    assert _feature_subset_context_symbols(
        requested,
        available,
        output,
        strict_subset=False,
    ) == ["BTC/USD:BTC", "ETH/USD:ETH"]
