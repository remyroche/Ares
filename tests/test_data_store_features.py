import pandas as pd
import numpy as np

from extreme_price_movements.data_store import (
    _ensure_feature_frame_index,
    _write_feature_metadata,
    append_symbol_features,
    load_features_selected,
    PartitionedOHLCVStore,
    save_features,
)
from extreme_price_movements.pipeline_steps import (
    _cap_panel_rows,
    _enforce_feature_snapshot_completeness,
    _expected_feature_keys_from_cfg,
    _feature_quality_issues_for_keys,
    _feature_snapshot_health_issues,
    _missing_requested_feature_keys,
    _scan_feature_cache_light,
    _derive_symbol_backfill_keys,
    _validate_feature_snapshot_completeness,
)
from extreme_price_movements.config import CFG
from extreme_price_movements.features import (
    add_regime_gates,
    compute_features_hourly,
    compute_market_features,
)
from extreme_price_movements.universe import _normalize_symbol


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


def test_save_features_supports_bounded_parallel_symbol_writes(tmp_path):
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    cols = ["ETH/USDT", "BTC/USDT"]
    feats = {
        "feat_a": np.array(
            [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0], [4.0, 13.0]],
            dtype=np.float32,
        ),
        "feat_b": np.array(
            [[5.0, 14.0], [6.0, 15.0], [7.0, 16.0], [8.0, 17.0]],
            dtype=np.float32,
        ),
        "feat_market": np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float32),
    }

    save_features(
        feats,
        ts,
        str(tmp_path),
        feat_index=idx,
        feat_columns=cols,
        save_workers=2,
    )

    out_dir = tmp_path / "features" / "20260101_000000"
    eth = pd.read_parquet(out_dir / "symbol=ETH_USDT.parquet")
    btc = pd.read_parquet(out_dir / "symbol=BTC_USDT.parquet")

    assert list(eth["feat_a"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(btc["feat_b"]) == [14.0, 15.0, 16.0, 17.0]
    assert list(eth["feat_market"]) == [100.0, 101.0, 102.0, 103.0]
    assert list(btc["feat_market"]) == [100.0, 101.0, 102.0, 103.0]


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


def test_load_features_selected_with_start_ts_preserves_index_backed_values(tmp_path):
    root = tmp_path
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    path = feature_dir / "symbol=ETH_USDT.parquet"
    df = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "__symbol__": ["ETH/USDT"] * 4,
        },
        index=idx,
    )
    df.to_parquet(path)

    feats = load_features_selected(
        pd.Timestamp("2026-01-01 00:00:00"),
        str(root),
        feature_keys=["feat_a"],
        symbols=["ETH/USDT"],
        start_ts=pd.Timestamp("2026-01-01 02:00:00", tz="UTC"),
    )

    feat_df = feats["feat_a"]
    assert isinstance(feat_df.index, pd.DatetimeIndex)
    assert list(feat_df.columns) == ["ETH/USDT"]
    assert list(feat_df.index) == [idx[2].tz_localize(None), idx[3].tz_localize(None)]
    assert list(feat_df["ETH/USDT"].astype(float)) == [3.0, 4.0]


def test_load_features_selected_accepts_symbol_alias_without_slash(tmp_path):
    root = tmp_path
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0], dtype=np.float32),
            "__symbol__": ["ETH/USDT", "ETH/USDT"],
        },
        index=idx,
    )
    df.to_parquet(feature_dir / "symbol=ETH_USDT.parquet")

    feats = load_features_selected(
        pd.Timestamp("2026-01-01 00:00:00"),
        str(root),
        feature_keys=["feat_a"],
        symbols=["ETHUSDT"],
    )

    feat_df = feats["feat_a"]
    assert list(feat_df.columns) == ["ETH/USDT"]
    assert list(feat_df["ETH/USDT"].astype(float)) == [1.0, 2.0]


def test_partitioned_ohlcv_store_load_resolves_existing_alias_dir(tmp_path):
    store = PartitionedOHLCVStore(root_dir=str(tmp_path), timeframe="1h")
    sym_dir = tmp_path / "ohlcv" / "symbol=ETHUSDT" / "year=2026" / "month=01"
    sym_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": idx,
            "open": np.array([1.0, 2.0], dtype=np.float32),
            "high": np.array([1.1, 2.1], dtype=np.float32),
            "low": np.array([0.9, 1.9], dtype=np.float32),
            "close": np.array([1.05, 2.05], dtype=np.float32),
            "volume": np.array([10.0, 20.0], dtype=np.float32),
        }
    )
    df.to_parquet(sym_dir / "data-1704067200-1704070800.parquet", index=False)

    loaded = store.load("ETH/USDT")
    assert len(loaded) == 2
    assert np.allclose(loaded["close"].astype(float).to_numpy(), [1.05, 2.05])


def test_normalize_symbol_restores_missing_quote_separator():
    assert _normalize_symbol("ETHUSDT") == "ETH/USDT"
    assert _normalize_symbol("BTC_USDC") == "BTC/USDC"
    assert _normalize_symbol("AAVE/USDT") == "AAVE/USDT"


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


def test_feature_snapshot_health_issues_flags_constant_or_missing_critical_keys():
    healthy = {
        "loc_range_pos_24": pd.DataFrame(
            {"A": np.array([0.0, 1.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "loc_vwap_dev_z_24": pd.DataFrame(
            {"A": np.array([1.0, 2.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "loc_pullback_depth_24": pd.DataFrame(
            {"A": np.array([2.0, 3.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "dist_ema50_atr": pd.DataFrame(
            {"A": np.array([3.0, 4.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "ema50_slope": pd.DataFrame(
            {"A": np.array([4.0, 5.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "prior_volatility": pd.DataFrame(
            {"A": np.array([5.0, 6.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "trend_acceleration": pd.DataFrame(
            {"A": np.array([7.0, 8.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
    }

    assert _feature_snapshot_health_issues(healthy) == []

    unhealthy = dict(healthy)
    unhealthy["loc_range_pos_24"] = pd.DataFrame(
        {"A": np.array([0.0, 0.0], dtype=np.float32)},
        index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
    )

    issues = _feature_snapshot_health_issues(unhealthy)
    assert any(issue.startswith("constant:loc_range_pos_24") for issue in issues)


def test_missing_requested_feature_keys_reports_absent_batch_outputs():
    features = {
        "feat_a": pd.DataFrame(
            {"A": np.array([1.0, 2.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "feat_b": pd.DataFrame(
            {"A": np.array([3.0, 4.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
    }

    missing = _missing_requested_feature_keys(features, ["feat_a", "feat_c"])

    assert missing == ["feat_c"]


def test_feature_quality_issues_for_keys_reports_noncritical_nan_and_constant():
    features = {
        "feat_ok": pd.DataFrame(
            {"A": np.array([1.0, 2.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "feat_nan": pd.DataFrame(
            {"A": np.array([np.nan, np.nan], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
        "feat_const": pd.DataFrame(
            {"A": np.array([5.0, 5.0], dtype=np.float32)},
            index=pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC"),
        ),
    }

    issues = _feature_quality_issues_for_keys(
        features,
        ["feat_ok", "feat_nan", "feat_const", "feat_missing"],
    )

    assert issues == [
        "all_nan:feat_nan",
        "constant:feat_const",
        "missing:feat_missing",
    ]


def test_append_symbol_features_preserves_existing_columns_on_partial_append(tmp_path):
    path = tmp_path / "symbol=BTC_USDT.parquet"
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")

    first = pd.DataFrame(
        {"feat_a": np.array([1.0, 2.0, 3.0], dtype=np.float32)},
        index=idx,
    )
    second = pd.DataFrame(
        {"feat_b": np.array([4.0, 5.0, 6.0], dtype=np.float32)},
        index=idx,
    )

    append_symbol_features(str(path), "BTC/USDT", first)
    append_symbol_features(str(path), "BTC/USDT", second)

    out = pd.read_parquet(path).drop(columns=["__symbol__"])
    assert out["feat_a"].tolist() == [1.0, 2.0, 3.0]
    assert out["feat_b"].tolist() == [4.0, 5.0, 6.0]


def test_scan_feature_cache_light_treats_all_nan_columns_as_partial(tmp_path):
    root = tmp_path
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")

    eth = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "feat_b": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            "__symbol__": ["ETH/USDT"] * 3,
        },
        index=idx,
    )
    btc = pd.DataFrame(
        {
            "feat_a": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            "feat_b": np.array([7.0, 8.0, 9.0], dtype=np.float32),
            "__symbol__": ["BTC/USDT"] * 3,
        },
        index=idx,
    )
    eth.to_parquet(feature_dir / "symbol=ETH_USDT.parquet")
    btc.to_parquet(feature_dir / "symbol=BTC_USDT.parquet")

    panel_close = pd.DataFrame(
        {"ETH/USDT": [1.0, 2.0, 3.0], "BTC/USDT": [1.0, 2.0, 3.0]},
        index=idx,
    )

    scan = _scan_feature_cache_light(
        ts_sig=ts,
        data_root=str(root),
        expected_keys={"feat_a", "feat_b"},
        panel_close=panel_close,
    )

    assert scan is not None
    assert "feat_b" in scan["partial_keys"]
    assert "ETH/USDT" in scan["stale_symbols"]
    assert scan["all_nan_symbol_keys"]["ETH/USDT"] == ["feat_b"]


def test_derive_symbol_backfill_keys_includes_all_nan_existing_columns(tmp_path):
    root = tmp_path
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")

    eth = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "feat_b": np.array([np.nan, np.nan, np.nan], dtype=np.float32),
            "__symbol__": ["ETH/USDT"] * 3,
        },
        index=idx,
    )
    eth.to_parquet(feature_dir / "symbol=ETH_USDT.parquet")

    keys = _derive_symbol_backfill_keys(
        ts_sig=ts,
        data_root=str(root),
        expected_keys={"feat_a", "feat_b"},
        symbols=["ETH/USDT"],
        full_rewrite_symbols=set(),
    )

    assert keys == ["feat_b"]


def test_expected_feature_keys_excludes_nonpersisted_intraday_rule_names():
    keys = _expected_feature_keys_from_cfg(CFG)

    assert "loc_range_pos_24" in keys
    assert "LOC_01_AboveEMA" not in keys
    assert "LONG_01_WideBullBody" not in keys
    assert "SHORT_04_EMATagCloseBelow" not in keys


def test_compute_features_hourly_emits_offline_volume_z_keys():
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    base = np.linspace(100.0, 120.0, len(idx), dtype=np.float32)
    close = pd.DataFrame(
        {
            "ETH/USDT": base,
            "BTC/USDT": base * np.float32(1.1),
        },
        index=idx,
    )
    open_ = close * np.float32(0.999)
    high = close * np.float32(1.002)
    low = close * np.float32(0.998)
    volume = pd.DataFrame(
        {
            "ETH/USDT": np.linspace(1000.0, 1500.0, len(idx), dtype=np.float32),
            "BTC/USDT": np.linspace(1200.0, 1700.0, len(idx), dtype=np.float32),
        },
        index=idx,
    )
    panel = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
    cfg = dict(CFG)
    cfg["enable_gated_features"] = False
    cfg["use_regime_features"] = False
    mkt = compute_market_features(panel, cfg["market_basket"])
    gates = add_regime_gates(
        mkt,
        cfg["gate_vol_lookback_hours"],
        cfg["gate_trend_thr"],
    )

    requested = [
        "volume_z_12",
        "volume_z_24",
    ]
    feats, _, _ = compute_features_hourly(
        {k: v.copy() for k, v in panel.items()},
        gates,
        cfg,
        requested_feature_keys=requested,
    )

    assert set(requested).issubset(feats.keys())


def test_validate_feature_snapshot_completeness_detects_partial_schema(tmp_path):
    root = tmp_path
    ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
    feature_dir = root / "features" / "20260101_000000"
    feature_dir.mkdir(parents=True)

    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    panel_close = pd.DataFrame(
        {"ETH/USDT": [1.0, 2.0, 3.0]},
        index=idx,
    )

    partial = pd.DataFrame(
        {
            "feat_a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "__symbol__": ["ETH/USDT", "ETH/USDT", "ETH/USDT"],
        },
        index=idx,
    )
    partial.to_parquet(feature_dir / "symbol=ETH_USDT.parquet")

    try:
        _validate_feature_snapshot_completeness(
            ts_sig=ts,
            data_root=str(root),
            expected_keys={"feat_a", "feat_b"},
            panel_close=panel_close,
        )
        assert False, "expected completeness validation to fail on partial schema"
    except RuntimeError as exc:
        assert "missing_keys=1" in str(exc)


def test_cap_panel_rows_is_noop():
    idx = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    panel = {
        "close": pd.DataFrame(
            {
                "A": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "B": np.array([4.0, 5.0, 6.0], dtype=np.float32),
            },
            index=idx,
        )
    }

    out = _cap_panel_rows(panel, 1)

    assert out is panel
    pd.testing.assert_frame_equal(out["close"], panel["close"])
