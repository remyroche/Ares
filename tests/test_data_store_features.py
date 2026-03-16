import pandas as pd
import numpy as np

from extreme_price_movements.data_store import (
    _ensure_feature_frame_index,
    _write_feature_metadata,
    append_symbol_features,
    load_features_selected,
)
from extreme_price_movements.pipeline_steps import _enforce_feature_snapshot_completeness


def test_ensure_feature_frame_index_restores_ts_column():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"),
            "feat_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        }
    )
    out, reason = _ensure_feature_frame_index(df)

    assert reason == "ts_column_indexed"
    assert isinstance(out.index, pd.DatetimeIndex)
    assert "ts" not in out.columns


def test_append_symbol_features_persists_datetime_index(tmp_path):
    path = tmp_path / "symbol=ETH_USDT.parquet"
    df = pd.DataFrame(
        {"feat_a": np.array([1.0, 2.0], dtype=np.float32)},
        index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    )

    append_symbol_features(str(path), "ETH/USDT", df)
    loaded = pd.read_parquet(path)

    assert "__symbol__" in loaded.columns
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_features_selected_restores_ts_column_cache(tmp_path):
    root = tmp_path
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
            "__symbol__": ["ETH/USDT", "ETH/USDT"],
            "feat_a": np.array([1.0, 2.0], dtype=np.float32),
        }
    )
    df.to_parquet(feature_dir / "symbol=ETH_USDT.parquet", index=False)

    feats = load_features_selected(
        pd.Timestamp("2026-01-01 00:00:00"),
        str(root),
        feature_keys=["feat_a"],
        symbols=["ETH/USDT"],
    )

    feat_df = feats["feat_a"]
    assert isinstance(feat_df.index, pd.DatetimeIndex)
    assert list(feat_df.columns) == ["ETH/USDT"]


def test_load_features_selected_recovers_numeric_index_from_metadata(tmp_path):
    root = tmp_path
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    path = feature_dir / "symbol=ETH_USDT.parquet"
    df = pd.DataFrame(
        {
            "__symbol__": ["ETH/USDT", "ETH/USDT"],
            "feat_a": np.array([1.0, 2.0], dtype=np.float32),
        }
    )
    df.to_parquet(path, index=False)
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    _write_feature_metadata(str(path), "ETH/USDT", idx)

    feats = load_features_selected(
        pd.Timestamp("2026-01-01 00:00:00"),
        str(root),
        feature_keys=["feat_a"],
        symbols=["ETH/USDT"],
    )

    feat_df = feats["feat_a"]
    assert isinstance(feat_df.index, pd.DatetimeIndex)
    assert feat_df.index[0] == idx[0].tz_localize(None)


def test_enforce_feature_snapshot_completeness_adds_missing_columns_and_rows(tmp_path):
    root = tmp_path
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)

    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    panel_close = pd.DataFrame(
        {
            "ETH/USDT": [1.0, 2.0, 3.0],
            "BTC/USDT": [1.0, 2.0, 3.0],
        },
        index=idx,
    )

    partial = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0], dtype=np.float32),
            "__symbol__": ["ETH/USDT", "ETH/USDT"],
        },
        index=idx[:2],
    )
    partial.to_parquet(feature_dir / "symbol=ETH_USDT.parquet")

    stats = _enforce_feature_snapshot_completeness(
        ts_sig=ts,
        data_root=str(root),
        expected_keys={"feat_a", "feat_b"},
        panel_close=panel_close,
    )

    assert stats["normalized_symbols"] == 2

    eth = pd.read_parquet(feature_dir / "symbol=ETH_USDT.parquet")
    btc = pd.read_parquet(feature_dir / "symbol=BTC_USDT.parquet")

    assert list(eth.drop(columns=["__symbol__"]).columns) == ["feat_a", "feat_b"]
    assert len(eth) == 3
    assert pd.isna(eth["feat_b"]).all()

    assert list(btc.drop(columns=["__symbol__"]).columns) == ["feat_a", "feat_b"]
    assert len(btc) == 3
    assert pd.isna(btc["feat_a"]).all()
    assert pd.isna(btc["feat_b"]).all()
