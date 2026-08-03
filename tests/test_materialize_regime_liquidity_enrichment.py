from pathlib import Path

import pandas as pd

from scripts.materialize_regime_liquidity_enrichment import (
    ASSET_FIELDS,
    MARKET_FIELDS,
    build_liquidity_sidecar,
)


def _write(path: Path, values: float) -> None:
    ts = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    data = {c: [values, values + 1, values + 2] for c in ASSET_FIELDS}
    data.update({c: [10 + values] * 3 for c in MARKET_FIELDS})
    pd.DataFrame(data, index=pd.Index(ts, name="ts")).to_parquet(path)


def test_exact_observed_liquidity_aggregation(tmp_path: Path) -> None:
    a = tmp_path / "symbol=A.parquet"
    b = tmp_path / "symbol=B.parquet"
    _write(a, 1.0)
    _write(b, 3.0)
    calendar = pd.DataFrame(
        {
            "source_utc": pd.date_range(
                "2024-01-01", periods=3, freq="h", tz="UTC"
            ),
            "calendar_segment_id": [1, 1, 1],
        }
    )
    out, manifest = build_liquidity_sidecar(
        calendar=calendar,
        feature_paths=[a, b],
        market_reference_path=a,
    )
    assert out["liquidity_xs__amihud_illiq__mean"].tolist() == [2.0, 3.0, 4.0]
    assert out["liquidity_xs__amihud_illiq__coverage"].eq(1.0).all()
    assert out["liquidity_market__median_spread_bps"].eq(11.0).all()
    assert manifest["counts"]["feature_files_used"] == 2


def test_no_asof_fill_across_missing_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "symbol=A.parquet"
    _write(source, 1.0)
    calendar = pd.DataFrame(
        {
            "source_utc": [
                pd.Timestamp("2024-01-01 00:00", tz="UTC"),
                pd.Timestamp("2024-01-01 00:30", tz="UTC"),
            ],
            "calendar_segment_id": [1, 2],
        }
    )
    out, _ = build_liquidity_sidecar(
        calendar=calendar,
        feature_paths=[source],
        market_reference_path=source,
    )
    assert pd.isna(out.loc[1, "liquidity_xs__amihud_illiq__mean"])
    assert pd.isna(out.loc[1, "liquidity_market__median_spread_bps"])
