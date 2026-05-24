from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.kraken_actual_data import (
    aggregate_trades_to_hourly,
    overlay_actual_volume_sidecar,
    plan_symbol_coverage,
    write_actual_volume_sidecar,
)


def _hourly_frame() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC", name="ts")
    return pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0, 11.0, 12.0, 12.0],
            "high": [10.0, 10.0, 10.0, 11.0, 12.5, 12.0],
            "low": [10.0, 10.0, 10.0, 11.0, 11.5, 12.0],
            "close": [10.0, 10.0, 10.0, 11.0, 12.2, 12.0],
            "volume": [5.0, 0.0, 0.0, 0.0, 0.0, float("nan")],
            "open_interest": [100.0, float("nan"), 102.0, 103.0, 104.0, 105.0],
            "funding_rate": [0.0, 0.0, float("nan"), 0.0, 0.0, 0.0],
        },
        index=idx,
    )


def test_plan_symbol_coverage_uses_sidecars_and_flags_linked_zero_volume():
    raw = _hourly_frame()
    oi_sidecar = pd.Series([101.0], index=[raw.index[1]], name="open_interest")
    funding_sidecar = pd.Series([0.0001], index=[raw.index[2]], name="funding_rate")
    actual_volume = pd.DataFrame(
        {
            "volume": [2.0],
            "quote_volume": [20.0],
            "trade_count": [3],
            "vwap": [10.0],
            "source": ["unit"],
            "coverage_status": ["actual_trades"],
        },
        index=[raw.index[1]],
    )

    coverage, oi_ranges, volume_ranges = plan_symbol_coverage(
        symbol_key="BTC_USD:USD",
        ohlcv=raw,
        oi_sidecar=oi_sidecar,
        actual_volume_sidecar=actual_volume,
        funding_sidecar=funding_sidecar,
        max_gap_hours=24,
    )

    assert coverage.price_rows == 6
    assert coverage.missing_oi == 0
    assert coverage.missing_funding == 0
    assert coverage.linked_zero_carry == 2
    assert coverage.actual_trades == 1
    assert coverage.missing_volume == 4
    assert oi_ranges == []
    assert volume_ranges == [
        (raw.index[2], raw.index[6 - 1] + pd.Timedelta(hours=1)),
    ]


def test_aggregate_trades_to_hourly_marks_actual_and_confirmed_empty_hours():
    start = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2026-01-01 03:00", tz="UTC")
    trades = [
        {"timestamp": int((start + pd.Timedelta(minutes=5)).value // 10**6), "price": 10.0, "amount": 2.0},
        {"timestamp": int((start + pd.Timedelta(minutes=20)).value // 10**6), "price": 12.0, "amount": 1.0},
        {"timestamp": int((start + pd.Timedelta(hours=2, minutes=1)).value // 10**6), "price": 20.0, "amount": 1.5},
    ]

    out = aggregate_trades_to_hourly(
        trades,
        start_ts=start,
        end_ts=end,
        source="unit",
        fill_empty_hours=True,
    )

    assert list(out["coverage_status"]) == [
        "actual_trades",
        "confirmed_no_trades",
        "actual_trades",
    ]
    assert out.iloc[0]["volume"] == pytest.approx(3.0)
    assert out.iloc[0]["quote_volume"] == pytest.approx(32.0)
    assert out.iloc[0]["vwap"] == pytest.approx(32.0 / 3.0)
    assert out.iloc[1]["volume"] == pytest.approx(0.0)
    assert out.iloc[1]["trade_count"] == 0


def test_overlay_actual_volume_sidecar_only_applies_valid_statuses(tmp_path):
    root = tmp_path / "krakenfutures"
    sidecar_path = root / "actual_volume_hourly" / "BTC_USD_USD.parquet"
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC", name="ts")
    sidecar = pd.DataFrame(
        {
            "volume": [9.0, 0.0, 99.0],
            "quote_volume": [90.0, 0.0, 990.0],
            "trade_count": [2, 0, 10],
            "vwap": [10.0, float("nan"), 10.0],
            "source": ["unit", "unit", "unit"],
            "coverage_status": ["actual_trades", "confirmed_no_trades", "unavailable"],
        },
        index=idx,
    )
    write_actual_volume_sidecar(sidecar_path, sidecar)
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 10.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 10.0, 10.0],
            "volume": [1.0, 0.0, 3.0],
        },
        index=idx,
    )

    merged = overlay_actual_volume_sidecar(raw, root_dir=root, symbol="BTC_USD:USD")

    assert list(merged["volume"]) == [9.0, 0.0, 3.0]
    assert merged.iloc[0]["trade_count"] == pytest.approx(2.0)
    assert merged.iloc[1]["trade_count"] == pytest.approx(0.0)
    assert pd.isna(merged.iloc[2].get("trade_count"))


def test_confirmed_no_trades_requires_flat_nonpositive_chart_candle(tmp_path):
    root = tmp_path / "krakenfutures"
    sidecar_path = root / "actual_volume_hourly" / "BTC_USD_USD.parquet"
    idx = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC", name="ts")
    sidecar = pd.DataFrame(
        {
            "volume": [0.0, 0.0, 0.0],
            "quote_volume": [0.0, 0.0, 0.0],
            "trade_count": [0, 0, 0],
            "vwap": [float("nan"), float("nan"), float("nan")],
            "source": ["unit", "unit", "unit"],
            "coverage_status": ["confirmed_no_trades"] * 3,
        },
        index=idx,
    )
    write_actual_volume_sidecar(sidecar_path, sidecar)
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 11.0, 10.0],
            "low": [10.0, 10.0, 10.0],
            "close": [10.0, 11.0, 10.0],
            "volume": [0.0, 0.0, 5.0],
        },
        index=idx,
    )

    merged = overlay_actual_volume_sidecar(raw, root_dir=root, symbol="BTC_USD:USD")

    assert list(merged["volume"]) == [0.0, 0.0, 5.0]
    assert merged.iloc[0]["trade_count"] == pytest.approx(0.0)
    assert pd.isna(merged.iloc[1].get("trade_count"))
    assert pd.isna(merged.iloc[2].get("trade_count"))
