from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.market_microstructure_features import (
    NativeL2FeatureContractError,
    materialize_native_l2_continuation_features,
    summarize_native_l2_snapshot_rows,
)


def _snapshots() -> pd.DataFrame:
    ts = pd.to_datetime(
        ["2026-07-11 12:00:00Z", "2026-07-11 13:00:00Z", "2026-07-11 16:00:00Z"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "snapshot_ts": ts,
            "best_bid": [99.0, 100.0, 101.0],
            "best_ask": [101.0, 102.0, 103.0],
            "mid": [100.0, 101.0, 102.0],
            "bid_qty_1": [10.0, 12.0, 11.0],
            "ask_qty_1": [20.0, 12.0, 10.0],
            "cum_bid_qty_l10": [100.0, 120.0, 110.0],
            "cum_ask_qty_l10": [200.0, 120.0, 100.0],
            "cum_bid_qty_l20": [150.0, 180.0, 160.0],
            "cum_ask_qty_l20": [260.0, 180.0, 140.0],
            "l2_bid_notional_l20": [15000.0, 18180.0, 16320.0],
            "l2_ask_notional_l20": [26000.0, 18180.0, 14280.0],
            "source": ["kraken_futures_l2_snapshot"] * 3,
        }
    )


def test_native_l2_features_are_causal_and_gap_aware() -> None:
    baseline = materialize_native_l2_continuation_features(_snapshots())
    changed = _snapshots().copy()
    changed.loc[2, "bid_qty_1"] = 9_999.0
    changed_out = materialize_native_l2_continuation_features(changed)
    # The first two rows cannot depend on a later snapshot.
    pd.testing.assert_frame_equal(baseline.iloc[:2].reset_index(drop=True), changed_out.iloc[:2].reset_index(drop=True))
    assert np.isfinite(baseline.loc[1, "l2_depth_imbalance_delta_prev_snapshot"])
    # The three-hour gap intentionally invalidates lagged changes.
    assert pd.isna(baseline.loc[2, "l2_depth_imbalance_delta_prev_snapshot"])
    assert pd.isna(baseline.loc[2, "l2_spread_widening_prev_snapshot"])
    assert pd.isna(baseline.loc[2, "l2_depth_depletion_prev_snapshot"])


def test_proxy_sources_are_rejected() -> None:
    frame = _snapshots().assign(source="local_ohlcv_summary")
    with pytest.raises(NativeL2FeatureContractError, match="non-native source"):
        materialize_native_l2_continuation_features(frame)


def test_duplicate_symbol_snapshot_is_rejected() -> None:
    frame = pd.concat([_snapshots(), _snapshots().iloc[[0]]], ignore_index=True)
    with pytest.raises(NativeL2FeatureContractError, match="duplicate"):
        materialize_native_l2_continuation_features(frame)


def test_raw_native_snapshot_aggregation_is_exact_source_and_schema() -> None:
    rows = []
    for level in range(1, 21):
        rows.append(
            {
                "observed_ts": "2026-07-11T00:00:01Z",
                "timestamp": "2026-07-11T00:00:00Z",
                "symbol": "AAA/USD:USD",
                "side": "bid",
                "level": level,
                "price": 100.0 - level * 0.1,
                "qty": float(level),
                "source": "kraken_futures_l2_snapshot",
            }
        )
        rows.append(
            {
                "observed_ts": "2026-07-11T00:00:01Z",
                "timestamp": "2026-07-11T00:00:00Z",
                "symbol": "AAA/USD:USD",
                "side": "ask",
                "level": level,
                "price": 100.0 + level * 0.1,
                "qty": float(level + 1),
                "source": "kraken_futures_l2_snapshot",
            }
        )
    summary = summarize_native_l2_snapshot_rows(pd.DataFrame(rows))
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["source"] == "kraken_futures_l2_snapshot"
    assert row["best_bid"] == 99.9
    assert row["best_ask"] == 100.1
    assert row["cum_bid_qty_l10"] == 55.0
    assert row["cum_ask_qty_l20"] == sum(range(2, 22))
