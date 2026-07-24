from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements import hf_data_loader
from extreme_price_movements.data_store import (
    canonical_hf_ohlcv_dir,
    canonical_market_data_root,
    ensure_hf_ohlcv_store_contract,
    make_ohlcv_store,
)
from extreme_price_movements.inference.data_fetcher import DataFetcher
from extreme_price_movements.inference.run_inference import _resolve_live_data_root
from extreme_price_movements.simple_policy_optimiser import _PerpPolicy15mReplayStore


def _kraken_perps_cfg(root: Path) -> dict[str, object]:
    return {
        "data_root": str(root),
        "exchange_id": "kraken",
        "market_mode": "perps",
        "use_perps": True,
        "exchange_scoped_data": True,
    }


def test_hourly_and_precise_roots_share_one_exchange_scoped_contract(tmp_path):
    cfg = _kraken_perps_cfg(tmp_path / "data_perp")
    market_root = canonical_market_data_root(cfg)

    hourly = make_ohlcv_store(cfg)
    precise_15m = ensure_hf_ohlcv_store_contract(cfg, timeframe="15m")
    precise_5m = ensure_hf_ohlcv_store_contract(cfg, timeframe="5m")

    assert Path(hourly.root_dir) == market_root
    assert precise_15m == market_root / "raw" / "ohlcv_15m"
    assert precise_5m == market_root / "raw" / "ohlcv_5m"
    manifest = json.loads((precise_15m / "_store_contract.json").read_text())
    assert manifest["exchange_data_component"] == "krakenfutures"
    assert manifest["market_mode"] == "perps"
    assert manifest["timeframe"] == "15m"
    assert manifest["timestamp_timezone"] == "UTC"


def test_live_fetcher_does_not_rescope_or_split_precise_cache(tmp_path):
    class _Exchange:
        id = "krakenfutures"

    artifact_root = tmp_path / "data_perp"
    expected_market_root = artifact_root / "exchanges" / "krakenfutures"
    fetcher = DataFetcher(
        exchange=_Exchange(), data_root=str(artifact_root), market_mode="perps"
    )

    assert Path(fetcher.ohlcv_store.root_dir) == expected_market_root
    assert fetcher.hf_data_dir == expected_market_root / "raw" / "ohlcv_15m"
    assert fetcher.hf_data_dir_5m == expected_market_root / "raw" / "ohlcv_5m"
    assert hf_data_loader.HF_DATA_DIR == fetcher.hf_data_dir
    assert hf_data_loader.HF_DATA_DIR_5M == fetcher.hf_data_dir_5m


def test_canonical_hf_root_is_idempotent_for_live_exchange_root(tmp_path):
    market_root = tmp_path / "data_perp" / "exchanges" / "krakenfutures"
    cfg = {
        "data_root": str(market_root),
        "exchange_id": "krakenfutures",
        "market_mode": "perps",
        "use_perps": True,
    }

    assert canonical_hf_ohlcv_dir(cfg, timeframe="15m") == (
        market_root / "raw" / "ohlcv_15m"
    )


def test_policy_replay_uses_canonical_precise_store_without_override(tmp_path, monkeypatch):
    monkeypatch.delenv("EPM_HF_DATA_DIR", raising=False)
    monkeypatch.setenv("EPM_EXCHANGE", "kraken")
    root = tmp_path / "data_perp"

    store = _PerpPolicy15mReplayStore(root, "perps")

    assert store.hf_15m_root == (
        root / "exchanges" / "krakenfutures" / "raw" / "ohlcv_15m"
    )


def test_live_root_rejects_a_noncanonical_raw_store_by_default(tmp_path, monkeypatch):
    class _Exchange:
        id = "krakenfutures"

    root = tmp_path / "data_perp"
    expected = root / "exchanges" / "krakenfutures"
    assert _resolve_live_data_root(
        artifact_data_root=str(root), exchange=_Exchange(), market_mode="perps"
    ) == str(expected)

    with pytest.raises(ValueError, match="canonical exchange-scoped store"):
        _resolve_live_data_root(
            artifact_data_root=str(root),
            exchange=_Exchange(),
            market_mode="perps",
            explicit_live_data_root=str(tmp_path / "other_live_store"),
        )

    monkeypatch.setenv("EPM_ALLOW_SEPARATE_LIVE_DATA_STORE", "1")
    assert _resolve_live_data_root(
        artifact_data_root=str(root),
        exchange=_Exchange(),
        market_mode="perps",
        explicit_live_data_root=str(tmp_path / "other_live_store"),
    ) == str(tmp_path / "other_live_store")


def test_kraken_1h_and_15m_round_trip_through_shared_store(tmp_path, monkeypatch):
    """Exercise the actual downloader/persistence/read path without network I/O."""

    class _Kraken:
        id = "krakenfutures"

        def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None, params=None):
            start = pd.Timestamp(int(since), unit="ms", tz="UTC")
            if timeframe == "1h":
                return [
                    [
                        int(start.timestamp() * 1000),
                        100.0,
                        103.0,
                        99.0,
                        102.0,
                        50.0,
                    ]
                ]
            if timeframe == "15m":
                start = start.floor("15min")
                return [
                    [
                        int((start + pd.Timedelta(minutes=15 * i)).timestamp() * 1000),
                        100.0 + i,
                        101.0 + i,
                        99.0 + i,
                        100.5 + i,
                        10.0 + i,
                    ]
                    for i in range(4)
                ]
            raise AssertionError(f"unexpected timeframe={timeframe}")

    # DataFetcher configures the module-global precise cache in production.
    # Preserve it here so this integration test cannot affect another test.
    monkeypatch.setattr(hf_data_loader, "HF_DATA_DIR", hf_data_loader.HF_DATA_DIR)
    monkeypatch.setattr(hf_data_loader, "HF_DATA_DIR_5M", hf_data_loader.HF_DATA_DIR_5M)
    monkeypatch.setenv("EPM_HF_DATA_DIR", str(hf_data_loader.HF_DATA_DIR))
    monkeypatch.setenv("EPM_HF_DATA_DIR_5M", str(hf_data_loader.HF_DATA_DIR_5M))

    artifact_root = tmp_path / "data_perp"
    target_hour = pd.Timestamp("2026-07-10 12:00:00+00:00")
    symbol = "BTC/USD:USD"
    fetcher = DataFetcher(
        exchange=_Kraken(), data_root=str(artifact_root), market_mode="perps"
    )

    # Live hourly download writes the canonical exchange-scoped hourly store.
    written_hourly = fetcher.fetch_latest_hourly_symbol(symbol, target_hour=target_hour)
    assert written_hourly.index.tolist() == [target_hour]
    train_store = make_ohlcv_store(_kraken_perps_cfg(artifact_root))
    training_read = train_store.load(symbol, start_ts=target_hour, end_ts=target_hour)
    assert training_read.index.tolist() == [target_hour]
    np.testing.assert_allclose(training_read["close"].to_numpy(), [102.0])

    # The same exchange root owns the precise cache.  This is the exact data
    # path used by labels, historical replay, and live hourly-gap repair.
    precise = hf_data_loader.sync_15m_ohlcv_range(
        _Kraken(),
        symbol,
        target_hour,
        target_hour + pd.Timedelta(minutes=45),
        full_backfill=False,
    )
    expected_15m_path = fetcher.hf_data_dir / "btcusd:usd_15m.parquet"
    assert expected_15m_path.exists()
    assert precise.index.tolist() == list(
        pd.date_range(target_hour, periods=4, freq="15min", tz="UTC")
    )

    # Policy replay consumes the exact persisted 15m file, not an alternate
    # cache or the hourly partitions.
    replay_store = _PerpPolicy15mReplayStore(artifact_root, "perps")
    replay_read = replay_store.load(
        symbol,
        start_ts=target_hour,
        end_ts=target_hour + pd.Timedelta(minutes=45),
    )
    assert replay_read.index.tolist() == precise.index.tolist()
    np.testing.assert_allclose(
        replay_read["close"].to_numpy(), precise["close"].to_numpy()
    )
