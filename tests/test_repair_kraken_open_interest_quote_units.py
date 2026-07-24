import pandas as pd

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.repair_kraken_open_interest_quote_units import _repair_symbol


def test_repair_updates_sidecar_and_only_native_embedded_rows(tmp_path):
    root = tmp_path / "krakenfutures"
    sidecar_dir = root / "open_interest_hourly"
    part_dir = root / "ohlcv" / "symbol=SOL_USD:USD" / "year=2026"
    sidecar_dir.mkdir(parents=True)
    part_dir.mkdir(parents=True)
    index = pd.date_range("2026-07-13 21:00", periods=5, freq="h", tz="UTC")
    sidecar_path = sidecar_dir / "SOL_USD_USD.parquet"
    pd.DataFrame(
        {"open_interest": [1_000.0, 10.0, 10.0, 10.0, 1_000.0]}, index=index
    ).to_parquet(sidecar_path)

    native_part = part_dir / "part-1783976400-1783990800.parquet"
    pd.DataFrame(
        {
            "ts": index,
            "close": 100.0,
            "open_interest": [1_000.0, 10.0, 10.0, 10.0, 1_000.0],
        }
    ).to_parquet(native_part, index=False)
    quote_part = part_dir / "compact-1783976400-1783990800.parquet"
    pd.DataFrame(
        {"ts": index, "close": 100.0, "open_interest": 1_000.0}
    ).to_parquet(quote_part, index=False)

    record = _repair_symbol(
        sidecar_path=sidecar_path,
        price_store=PartitionedOHLCVStore(root_dir=str(root), timeframe="1h"),
        start=index[1],
        end=index[3],
        minimum_log_improvement=0.35,
        apply=True,
        backup_dir=tmp_path / "backup",
    )

    assert record["converted_rows"] == 3
    assert record["embedded_converted_rows"] == 3
    repaired_sidecar = pd.read_parquet(sidecar_path)
    assert repaired_sidecar["open_interest"].tolist() == [1_000.0] * 5
    repaired_native = pd.read_parquet(native_part)
    assert repaired_native["open_interest"].tolist() == [1_000.0] * 5
    pd.testing.assert_frame_equal(pd.read_parquet(quote_part), pd.DataFrame(
        {"ts": index, "close": 100.0, "open_interest": 1_000.0}
    ))
