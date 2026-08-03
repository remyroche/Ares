from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.execution_ev_market_state import (
    MARKET_STATE_COLUMNS,
    MARKET_STATE_FAMILIES,
    UNAVAILABLE_HISTORICAL_FAMILIES,
    attach_decision_time_market_state,
    feature_store_filename,
)


def _write_store(
    root, symbol: str, timestamps: list[str], **columns: list[float]
) -> None:
    source = pd.DataFrame(columns, index=pd.DatetimeIndex(timestamps, tz="UTC", name="ts"))
    source.to_parquet(root / feature_store_filename(symbol))


def _candidates(times: list[str], symbol: str = "BTC/USD:USD") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "execution_decision_utc": pd.to_datetime(times, utc=True),
            "__symbol__": symbol,
        }
    )


def test_completed_hour_join_is_backward_and_lagged_without_future_rows(tmp_path) -> None:
    """A decision at 10:30 may use the completed 09:00 bar, never 10:00."""
    symbol = "BTC/USD:USD"
    _write_store(
        tmp_path,
        symbol,
        ["2026-07-01 09:00", "2026-07-01 10:00"],
        volatility_of_volatility_48=[1.0, 99.0],
    )
    result = attach_decision_time_market_state(
        _candidates(["2026-07-01 10:30"], symbol), feature_store_root=tmp_path
    ).frame.iloc[0]

    assert result["mkt_state_source_utc"] == pd.Timestamp("2026-07-01 09:00", tz="UTC")
    assert result["mkt_state__volatility_of_volatility_48"] == 1.0
    assert result["mkt_state_source_utc"] <= (
        result["execution_decision_utc"] - pd.Timedelta(hours=1)
    )
    assert result["mkt_state_source_age_seconds"] == 30 * 60


def test_stale_source_remains_missing_instead_of_using_a_future_bar(tmp_path) -> None:
    symbol = "BTC/USD:USD"
    _write_store(
        tmp_path,
        symbol,
        ["2026-07-01 07:00", "2026-07-01 11:00"],
        efficiency_ratio_20=[2.0, 88.0],
    )
    result = attach_decision_time_market_state(
        _candidates(["2026-07-01 10:30"], symbol),
        feature_store_root=tmp_path,
        max_staleness=pd.Timedelta("90min"),
    ).frame.iloc[0]

    assert pd.isna(result["mkt_state_source_utc"])
    assert np.isnan(result["mkt_state__efficiency_ratio_20"])
    assert np.isnan(result["mkt_state_source_age_seconds"])
    # A source file was present; absence is due to staleness, not an invented zero.
    assert bool(result["mkt_state_source_file_found"])


def test_source_availability_is_explicit_and_unavailable_families_are_not_synthesized(tmp_path) -> None:
    symbol = "BTC/USD:USD"
    _write_store(
        tmp_path,
        symbol,
        ["2026-07-01 09:00"],
        market_breadth_4h=[0.7],
        # This tempting proxy-like column is deliberately not in the declared
        # state schema and must not be surfaced as spread/depth data.
        orderbook_depth_proxy=[123.0],
    )
    joined = attach_decision_time_market_state(
        _candidates(["2026-07-01 10:00"], symbol), feature_store_root=tmp_path
    )
    row = joined.frame.iloc[0]
    audit = joined.source_audit.iloc[0]

    assert row["mkt_state__market_breadth_4h"] == 0.7
    assert "market_breadth_4h" in audit["available_source_columns"]
    assert "orderbook_depth_proxy" not in audit["available_source_columns"]
    assert "spread_depth" in UNAVAILABLE_HISTORICAL_FAMILIES
    assert "observed_liquidations" in UNAVAILABLE_HISTORICAL_FAMILIES
    assert all("spread" not in column and "depth" not in column for column in MARKET_STATE_COLUMNS)
    assert "spread_depth" not in MARKET_STATE_FAMILIES
    assert "observed_liquidations" not in MARKET_STATE_FAMILIES
