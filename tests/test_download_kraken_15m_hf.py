from __future__ import annotations

import pandas as pd

from scripts.download_kraken_15m_hf import (
    _partition_symbols,
    _regularize_15m_candles,
)


def test_partition_membership_does_not_depend_on_order() -> None:
    symbols = [f"S{i}" for i in range(11)]
    for partition_id in range(4):
        ascending = set(
            _partition_symbols(
                symbols,
                partition_count=4,
                partition_id=partition_id,
                order="alpha_asc",
            )
        )
        descending = set(
            _partition_symbols(
                symbols,
                partition_count=4,
                partition_id=partition_id,
                order="alpha_desc",
            )
        )
        assert ascending == descending


def test_regularization_materializes_flat_zero_volume_candles() -> None:
    index = pd.to_datetime(
        ["2026-01-01 00:00:00Z", "2026-01-01 00:30:00Z"]
    )
    source = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "volume": [3.0, 4.0],
        },
        index=index,
    )
    result = _regularize_15m_candles(source)
    assert result.index[1] == pd.Timestamp("2026-01-01 00:15:00Z")
    assert result.iloc[1][["open", "high", "low", "close"]].eq(1.0).all()
    assert result.iloc[1]["volume"] == 0.0
