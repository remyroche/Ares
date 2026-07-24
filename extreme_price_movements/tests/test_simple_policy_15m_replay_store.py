import pandas as pd

from extreme_price_movements.simple_policy_optimiser import _PerpPolicy15mReplayStore


def test_replay_store_prefers_dedicated_15m_cache(tmp_path, monkeypatch):
    cache = tmp_path / "15m"
    cache.mkdir()
    index = pd.date_range("2026-07-10", periods=8, freq="15min", tz="UTC")
    source = pd.DataFrame(
        {
            "open": range(8),
            "high": range(1, 9),
            "low": range(8),
            "close": range(1, 9),
            "volume": range(8),
        },
        index=index,
    )
    source.to_parquet(cache / "btcusd:usd_15m.parquet")
    monkeypatch.setenv("EPM_HF_DATA_DIR", str(cache))
    monkeypatch.setenv("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "0")

    store = _PerpPolicy15mReplayStore(tmp_path / "data", "perps")
    loaded = store.load(
        "BTC/USD:USD",
        start_ts=index[2],
        end_ts=index[5],
    )

    assert loaded.index.tolist() == index[2:6].tolist()
    assert loaded["close"].tolist() == [3.0, 4.0, 5.0, 6.0]
    assert store.downloaded_15m_rows == 0
