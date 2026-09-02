from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.p8u_canonical_single_timestamp_runtime import (
    extract_one_timestamp_snapshot,
)


def _panel(timestamp: pd.Timestamp, symbols: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    values = np.arange(len(symbols), dtype=np.float32)[None, :]
    return {
        "close": pd.DataFrame(values, index=[timestamp], columns=symbols),
        "volume": pd.DataFrame(values + 1.0, index=[timestamp], columns=symbols),
    }


def test_extract_one_timestamp_snapshot_preserves_complete_target_free_universe() -> None:
    timestamp = pd.Timestamp("2026-08-30T10:00:00Z")
    symbols = tuple(f"S{index:03d}/USD:USD" for index in range(160))
    snapshot = extract_one_timestamp_snapshot(
        _panel(timestamp, symbols), timestamp=timestamp, symbols=symbols
    )
    assert set(snapshot) == {"close", "volume"}
    assert snapshot["close"].index.tolist() == [timestamp]
    assert snapshot["close"].columns.tolist() == list(symbols)


def test_extract_one_timestamp_snapshot_rejects_outcome_like_source() -> None:
    timestamp = pd.Timestamp("2026-08-30T10:00:00Z")
    symbols = tuple(f"S{index:03d}/USD:USD" for index in range(160))
    panel = _panel(timestamp, symbols)
    panel["policy_net_bps"] = panel["close"]
    with pytest.raises(ValueError, match="not target-free"):
        extract_one_timestamp_snapshot(panel, timestamp=timestamp, symbols=symbols)
