from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.kraken_actual_data import safe_symbol
from scripts.rebuild_v11_residual_market_context import _load_feature_store_context


def test_frozen_context_loader_uses_timestamp_index_and_market_median(tmp_path) -> None:
    index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    paths = {}
    raw_map = {}
    symbols = ["AAA_USD:USD", "BBB_USD:USD", "CCC_USD:USD"]
    for offset, symbol in enumerate(symbols):
        path = tmp_path / f"symbol={symbol}.parquet"
        pd.DataFrame(
            {
                "ts": index,
                "short_covering_score_market": [1.0 + offset, 2.0 + offset],
                "funding_confirmed_long_flush": [0.1 + offset, 0.2 + offset],
            }
        ).to_parquet(path, index=False)
        key = safe_symbol(symbol)
        paths[key] = path
        raw_map[key] = symbol

    context = _load_feature_store_context(
        paths,
        index=index,
        raw_symbol_map=raw_map,
        required_symbols={"AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD"},
    )

    assert context["short_covering_score_market"].tolist() == [2.0, 3.0]
    assert context["funding_confirmed_long_flush"].tolist() == pytest.approx([1.1, 1.2])
