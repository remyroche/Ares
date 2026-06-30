from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_t1_feature_store_sample_ledger import STRATEGY_IDS, build_sample_ledger


def _write_feature_file(root: Path, name: str, timestamps: list[str]) -> None:
    ts = pd.to_datetime(timestamps, utc=True)
    frame = pd.DataFrame({"feature_a": [1.0] * len(ts)}, index=pd.Index(ts, name="ts"))
    frame.to_parquet(root / f"symbol={name}.parquet")


def test_build_sample_ledger_filters_time_and_maps_active_heads(tmp_path: Path) -> None:
    feature_root = tmp_path / "features"
    feature_root.mkdir()
    _write_feature_file(
        feature_root,
        "BTC_USD:USD",
        ["2026-06-23T08:00:00Z", "2026-06-23T09:00:00Z", "2026-06-23T10:00:00Z"],
    )
    _write_feature_file(
        feature_root,
        "ETH_USD:USD",
        ["2026-06-23T09:00:00Z", "2026-06-23T10:00:00Z"],
    )

    ledger, summary = build_sample_ledger(
        feature_store_dir=feature_root,
        start=pd.Timestamp("2026-06-23T09:00:00Z"),
        end=pd.Timestamp("2026-06-23T10:00:00Z"),
        heads=("short_asset", "short_boll"),
    )

    assert len(ledger) == 8
    assert set(ledger["symbol"]) == {"BTC/USD:USD", "ETH/USD:USD"}
    assert set(ledger["head"]) == {"short_asset", "short_boll"}
    assert set(ledger["strategy_id"]) == {
        STRATEGY_IDS["short_asset"],
        STRATEGY_IDS["short_boll"],
    }
    assert ledger["timestamp"].min() == pd.Timestamp("2026-06-23T09:00:00Z")
    assert ledger["timestamp"].max() == pd.Timestamp("2026-06-23T10:00:00Z")
    assert summary["rows"] == 8
    assert summary["timestamp_count"] == 2
