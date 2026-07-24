import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.config import CFG
import extreme_price_movements.features as feature_impl
from extreme_price_movements.static_feature_store import (
    STATIC_FEATURE_ENDPOINT_VERSION,
    append_static_features,
    compare_static_feature_block_sources,
    compute_static_features,
    compute_static_market_context,
    configure_static_feature_runtime,
    materialize_static_feature_blocks,
    read_static_features,
    read_static_feature_blocks,
    resolve_static_feature_save_workers,
)


def test_static_feature_workers_are_shared_and_concurrent(monkeypatch):
    monkeypatch.delenv("EPM_STATIC_FEATURE_SAVE_WORKERS", raising=False)
    assert resolve_static_feature_save_workers({}) == 4
    assert resolve_static_feature_save_workers({"feature_save_workers": 6}) == 6
    assert resolve_static_feature_save_workers({"feature_save_workers": 1}) == 2
    monkeypatch.setenv("EPM_STATIC_FEATURE_SAVE_WORKERS", "7")
    assert resolve_static_feature_save_workers({"feature_save_workers": 3}) == 7


def test_static_endpoint_configures_shared_incremental_state(tmp_path):
    cfg = configure_static_feature_runtime(
        {"data_root": str(tmp_path)},
        data_root=tmp_path,
        feature_store_id="20260716_120000",
        requested_feature_keys=["ret_1h", "mkt_ret_1h"],
        incremental=True,
        state_scope="feature_store=20260716_120000",
    )

    assert cfg["static_feature_endpoint_version"] == STATIC_FEATURE_ENDPOINT_VERSION
    assert cfg["feature_causal_transform_state_enabled"] is True
    assert cfg["feature_raw_rolling_state_enabled"] is True
    assert cfg["feature_raw_rolling_state_container_enabled"] is True
    assert "features/20260716_120000/_static_state" in cfg[
        "feature_causal_transform_state_path"
    ]
    assert "features/20260716_120000/_static_state" in cfg[
        "feature_raw_rolling_state_container_path"
    ]
    assert cfg["feature_causal_transform_requested_hash"] != "all"

    live_state = tmp_path / "explicit-live-state.npz"
    live_cfg = configure_static_feature_runtime(
        {
            "live_causal_transform_state_path": str(live_state),
            "static_feature_allow_legacy_live_state_path": True,
        },
        data_root=tmp_path,
        feature_store_id="run_b",
        requested_feature_keys=["ret_1h"],
        incremental=True,
    )
    assert live_cfg["feature_causal_transform_state_path"] == str(live_state)

    static_cfg = configure_static_feature_runtime(
        {"live_causal_transform_state_path": str(live_state)},
        data_root=tmp_path,
        feature_store_id="run_c",
        requested_feature_keys=["ret_1h"],
        incremental=True,
    )
    assert static_cfg["feature_causal_transform_state_path"] != str(live_state)
    assert "features/run_c/_static_state" in static_cfg[
        "feature_causal_transform_state_path"
    ]


def test_static_incremental_contract_bypasses_legacy_tail_fast_path():
    """A narrow future contract cannot silently skip persisted rolling state."""

    result = feature_impl._compute_live_lgbm_mask_features_fast(
        panel={},
        cfg={
            "static_feature_endpoint_version": STATIC_FEATURE_ENDPOINT_VERSION,
            "feature_raw_rolling_state_enabled": True,
        },
        requested_feature_keys={"rolling_range_20"},
    )
    assert result is None


def test_static_endpoint_uses_features_module_and_excludes_model_state(tmp_path, monkeypatch):
    index = pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC")
    panel = {"close": pd.DataFrame({"BTC/USD:USD": [1.0, 2.0]}, index=index)}
    gates = pd.DataFrame(index=index)
    captured = {}

    def fake_compute(panel_arg, gates_arg, cfg_arg, requested_feature_keys=None):
        captured["cfg"] = cfg_arg
        captured["requested"] = requested_feature_keys
        return (
            {"ret_1h": pd.DataFrame({"BTC/USD:USD": [0.0, 1.0]}, index=index)},
            index,
            ["BTC/USD:USD"],
        )

    monkeypatch.setattr(feature_impl, "compute_features_hourly", fake_compute)
    result = compute_static_features(
        panel,
        gates,
        {"data_root": str(tmp_path)},
        requested_feature_keys=["ret_1h"],
        data_root=tmp_path,
        feature_store_id="run_a",
        incremental=True,
    )

    assert captured["requested"] == ["ret_1h"]
    assert captured["cfg"]["static_feature_endpoint_version"] == STATIC_FEATURE_ENDPOINT_VERSION
    assert result.columns == ["BTC/USD:USD"]
    assert list(result.features) == ["ret_1h"]

    monkeypatch.setattr(
        feature_impl,
        "compute_features_hourly",
        lambda *args, **kwargs: (
            {"aegmm_cluster_id": pd.DataFrame({"BTC/USD:USD": [0.0, 1.0]}, index=index)},
            index,
            ["BTC/USD:USD"],
        ),
    )
    with pytest.raises(RuntimeError, match="AE/GMM"):
        compute_static_features(
            panel,
            gates,
            {"data_root": str(tmp_path)},
            data_root=tmp_path,
            feature_store_id="run_a",
        )


def test_static_market_context_uses_the_shared_causal_implementation(monkeypatch):
    index = pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC")
    panel = {
        "close": pd.DataFrame({"BTC/USD:USD": [1.0, 2.0]}, index=index),
        "high": pd.DataFrame({"BTC/USD:USD": [1.0, 2.0]}, index=index),
        "low": pd.DataFrame({"BTC/USD:USD": [1.0, 2.0]}, index=index),
        "volume": pd.DataFrame({"BTC/USD:USD": [1.0, 2.0]}, index=index),
    }
    captured = {}

    def fake_market(panel_arg, basket_arg, trend_sma_hours):
        captured["basket"] = basket_arg
        captured["trend"] = trend_sma_hours
        return pd.DataFrame({"mkt_ret": [0.0, 0.1]}, index=index)

    def fake_gates(market_arg, gate_vol_lookback_hours, gate_trend_thr):
        captured["gate_window"] = gate_vol_lookback_hours
        captured["gate_threshold"] = gate_trend_thr
        return market_arg.assign(G_VOL=0)

    monkeypatch.setattr(feature_impl, "compute_market_features", fake_market)
    monkeypatch.setattr(feature_impl, "add_regime_gates", fake_gates)
    result = compute_static_market_context(
        panel,
        ["BTC/USD:USD"],
        trend_sma_hours=48,
        gate_vol_lookback_hours=24,
        gate_trend_thr=0.1,
    )

    assert captured == {
        "basket": ["__static_market__"],
        "trend": 48,
        "gate_window": 24,
        "gate_threshold": 0.1,
    }
    assert "G_VOL" in result.regime_gates


def test_incremental_market_state_matches_full_market_context(tmp_path):
    rng = np.random.default_rng(7)
    index = pd.date_range("2026-01-01", periods=160, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD", "CCC/USD:USD"]
    close = pd.DataFrame(
        100.0 + rng.normal(0.0, 0.5, size=(len(index), len(columns))).cumsum(axis=0),
        index=index,
        columns=columns,
        dtype=np.float32,
    )
    panel = {
        "close": close,
        "high": (close + 0.5).astype(np.float32),
        "low": (close - 0.5).astype(np.float32),
        "volume": pd.DataFrame(
            rng.uniform(1.0, 5.0, size=(len(index), len(columns))),
            index=index,
            columns=columns,
            dtype=np.float32,
        ),
    }
    kwargs = {
        "trend_sma_hours": 48,
        "gate_vol_lookback_hours": 24,
        "gate_trend_thr": 0.0,
    }
    full = compute_static_market_context(panel, columns, **kwargs)
    state_cfg = {"static_feature_state_root": str(tmp_path)}
    first_panel = {key: value.iloc[:110] for key, value in panel.items()}
    compute_static_market_context(
        first_panel,
        columns,
        cfg=state_cfg,
        data_root=tmp_path,
        feature_store_id="run_a",
        incremental=True,
        **kwargs,
    )
    tail_panel = {key: value.iloc[86:] for key, value in panel.items()}
    incremental = compute_static_market_context(
        tail_panel,
        columns,
        cfg=state_cfg,
        data_root=tmp_path,
        feature_store_id="run_a",
        incremental=True,
        **kwargs,
    )

    assert incremental.used_state is True
    assert incremental.state_path is not None
    assert pd.Timestamp(incremental.regime_gates.index.min()) == index[86]
    pd.testing.assert_frame_equal(
        full.market_features.loc[index[86:]],
        incremental.market_features,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )
    pd.testing.assert_frame_equal(
        full.regime_gates.loc[index[86:]],
        incremental.regime_gates,
        check_exact=False,
        rtol=1e-6,
        atol=1e-6,
    )


def test_static_parity_blocks_are_timestamp_symbol_keyed_and_coalesce(tmp_path):
    index = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    features = {
        "ret_1h": pd.DataFrame(
            np.arange(6, dtype=np.float32).reshape(3, 2), index=index, columns=columns
        ),
        "mkt_ret_1h": pd.DataFrame(
            np.full((3, 2), 0.25, dtype=np.float32), index=index, columns=columns
        ),
    }
    manifest = materialize_static_feature_blocks(
        features,
        index=index,
        columns=columns,
        data_root=tmp_path,
        feature_store_id="run_a",
        source="pipeline",
        max_timestamps=None,
    )
    assert manifest["rows"] == 6
    assert manifest["model_state_excluded"] == ["AE", "GMM"]

    repaired = {"ret_1h": features["ret_1h"].copy()}
    repaired["ret_1h"].iloc[-1, 0] = 99.0
    materialize_static_feature_blocks(
        repaired,
        index=index[-1:],
        columns=columns,
        data_root=tmp_path,
        feature_store_id="run_a",
        source="inference",
        max_timestamps=None,
    )
    loaded = read_static_feature_blocks(
        data_root=tmp_path,
        feature_store_id="run_a",
        feature_keys=["ret_1h", "mkt_ret_1h"],
        start_ts=index.min(),
        end_ts=index.max(),
    )

    assert len(loaded) == 6
    final = loaded.loc[
        (loaded["ts"] == index[-1]) & (loaded["symbol"] == "AAA/USD:USD")
    ].iloc[0]
    assert final["ret_1h"] == 99.0
    assert final["mkt_ret_1h"] == 0.25


def test_static_parity_blocks_accept_zero_as_an_explicit_opt_out(tmp_path):
    index = pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC")
    columns = ["AAA/USD:USD"]
    manifest = materialize_static_feature_blocks(
        {
            "ret_1h": pd.DataFrame(
                [[0.1], [0.2]], index=index, columns=columns, dtype=np.float32
            )
        },
        index=index,
        columns=columns,
        data_root=tmp_path,
        feature_store_id="run_a",
        source="test",
        max_timestamps=0,
    )

    assert manifest["rows"] == 0
    assert not (tmp_path / "features" / "run_a" / "_static_feature_blocks").exists()


def test_static_parity_block_reader_prunes_unrequested_date_partitions(
    tmp_path, monkeypatch
):
    import extreme_price_movements.static_feature_store as static_store

    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-01 23:00", tz="UTC"),
            pd.Timestamp("2026-07-02 00:00", tz="UTC"),
        ]
    )
    columns = ["AAA/USD:USD"]
    materialize_static_feature_blocks(
        {
            "ret_1h": pd.DataFrame(
                [[0.1], [0.2]], index=index, columns=columns, dtype=np.float32
            )
        },
        index=index,
        columns=columns,
        data_root=tmp_path,
        feature_store_id="run_a",
        source="pipeline",
        max_timestamps=None,
    )

    original_read = static_store.pd.read_parquet
    read_paths: list[str] = []

    def _tracked_read(path, *args, **kwargs):
        read_paths.append(str(path))
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(static_store.pd, "read_parquet", _tracked_read)
    loaded = read_static_feature_blocks(
        data_root=tmp_path,
        feature_store_id="run_a",
        feature_keys=["ret_1h"],
        start_ts=index[-1],
        end_ts=index[-1],
    )

    assert loaded["ts"].tolist() == [index[-1]]
    assert read_paths
    assert all("date=2026-07-02" in path for path in read_paths)


def test_static_parity_blocks_compare_pipeline_and_inference_sources(tmp_path):
    index = pd.date_range("2026-07-01", periods=2, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    features = {
        "ret_1h": pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=index,
            columns=columns,
            dtype=np.float32,
        )
    }
    for source in ("pipeline", "inference_raw_tail"):
        materialize_static_feature_blocks(
            features,
            index=index,
            columns=columns,
            data_root=tmp_path,
            feature_store_id="run_a",
            source=source,
            max_timestamps=None,
        )

    report = compare_static_feature_block_sources(
        data_root=tmp_path,
        feature_store_id="run_a",
        feature_keys=["ret_1h"],
        left_source="pipeline",
        right_source="inference_raw_tail",
    )

    assert report["overlap_rows"] == 4
    assert report["all_within_tolerance"] is True
    assert report["per_feature"]["ret_1h"]["max_abs_diff"] == 0.0


def test_shared_static_append_read_uses_parquet_base_and_duckdb_tail(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    monkeypatch.setenv("EPM_FEATURE_DELTA_DUCKDB", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    index = pd.date_range("2026-07-01", periods=4, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    store_ts = pd.Timestamp("2026-07-01 12:00", tz="UTC")
    base = {
        "ret_1h": pd.DataFrame(
            [[0.1, 0.2], [0.3, 0.4]],
            index=index[:2],
            columns=columns,
            dtype=np.float32,
        )
    }
    tail = {
        "ret_1h": pd.DataFrame(
            [[0.5, 0.6], [0.7, 0.8]],
            index=index[2:],
            columns=columns,
            dtype=np.float32,
        )
    }

    append_static_features(
        base,
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index[:2],
        columns=columns,
        source="pipeline",
    )
    append_static_features(
        tail,
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index[2:],
        columns=columns,
        min_timestamp_by_symbol={symbol: index[1] for symbol in columns},
        source="inference_raw_tail",
    )

    store_id = store_ts.strftime("%Y%m%d_%H%M%S")
    assert list((tmp_path / "features" / store_id).glob("*.parquet.deltas.duckdb"))
    loaded = read_static_features(
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_keys=["ret_1h"],
        symbols=columns,
        start_ts=index.min(),
        end_ts=index.max() + pd.Timedelta(microseconds=1),
    )
    assert loaded is not None
    pd.testing.assert_frame_equal(
        loaded["ret_1h"].reindex(index=index, columns=columns),
        pd.concat([base["ret_1h"], tail["ret_1h"]]),
    )


def test_read_static_features_symbol_frame_matches_panel_layout(tmp_path):
    index = pd.date_range("2026-07-01", periods=3, freq="h", tz="UTC")
    symbols = ["AAA/USD:USD", "BBB/USD:USD"]
    store_ts = pd.Timestamp("2026-07-01 12:00", tz="UTC")
    features = {
        "ret_1h": pd.DataFrame(
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            index=index,
            columns=symbols,
            dtype=np.float32,
        ),
        "rv_24h": pd.DataFrame(
            [[1.1, 1.4], [1.2, 1.5], [1.3, 1.6]],
            index=index,
            columns=symbols,
            dtype=np.float32,
        ),
    }
    append_static_features(
        features,
        feature_store_ts=store_ts,
        data_root=tmp_path,
        index=index,
        columns=symbols,
        source="pipeline",
    )
    panels = read_static_features(
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_keys=list(features),
        symbols=[symbols[0]],
    )
    symbol_frame = read_static_features(
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_keys=list(features),
        symbols=[symbols[0]],
        output_layout="symbol_frame",
    )
    assert isinstance(symbol_frame, pd.DataFrame)
    for key in features:
        np.testing.assert_allclose(
            symbol_frame[key].to_numpy(),
            panels[key][symbols[0]].to_numpy(),
            rtol=0.0,
            atol=0.0,
        )


def test_real_static_incremental_endpoint_matches_full_and_materializes_parity(
    tmp_path, monkeypatch
):
    """Exercise the actual static formulas, persisted state, and store merge.

    The tail deliberately overlaps the seeded prefix.  This is the production
    shape: the live caller supplies enough raw warmup for non-stateful feature
    families while the persisted NumPy/Numba primitives update only new bars.
    """

    pytest.importorskip("duckdb")
    monkeypatch.setenv("EPM_FEATURE_DELTA_DUCKDB", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    rng = np.random.default_rng(23)
    index = pd.date_range("2026-01-01", periods=144, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    close = pd.DataFrame(
        100.0 + rng.normal(0.0, 0.25, size=(len(index), len(columns))).cumsum(axis=0),
        index=index,
        columns=columns,
        dtype=np.float32,
    )
    panel = {
        "open": (close - 0.05).astype(np.float32),
        "high": (close + 0.15).astype(np.float32),
        "low": (close - 0.15).astype(np.float32),
        "close": close,
        "volume": pd.DataFrame(
            rng.uniform(10.0, 20.0, size=(len(index), len(columns))),
            index=index,
            columns=columns,
            dtype=np.float32,
        ),
    }
    feature_keys = ["ret1h", "rv_24h"]
    market_kwargs = {
        "trend_sma_hours": 48,
        "gate_vol_lookback_hours": 24,
        "gate_trend_thr": 0.0,
    }
    static_cfg = {
        **CFG,
        "data_root": str(tmp_path),
        # The fast deployment-mask shortcut has no raw rolling-state contract.
        # Static incremental generation therefore exercises the stateful core.
        "live_lgbm_mask_feature_fast_path_enabled": False,
    }
    store_ts = pd.Timestamp("2026-01-01 12:00", tz="UTC")
    store_id = store_ts.strftime("%Y%m%d_%H%M%S")

    full_context = compute_static_market_context(
        panel, columns, cfg=static_cfg, incremental=False, **market_kwargs
    )
    full = compute_static_features(
        panel,
        full_context.regime_gates,
        static_cfg,
        requested_feature_keys=feature_keys,
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=False,
    )
    seed_end = 96
    append_static_features(
        {key: value.iloc[:seed_end] for key, value in full.features.items()},
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_store_id=store_id,
        index=full.index[:seed_end],
        columns=full.columns,
        source="pipeline",
        block_max_timestamps=None,
    )
    materialize_static_feature_blocks(
        full.features,
        index=full.index,
        columns=full.columns,
        data_root=tmp_path,
        feature_store_id=store_id,
        source="pipeline",
        feature_keys=feature_keys,
        max_timestamps=None,
    )

    seed_panel = {key: value.iloc[:seed_end] for key, value in panel.items()}
    seed_context = compute_static_market_context(
        seed_panel,
        columns,
        cfg={**static_cfg, "feature_raw_rolling_state_container_enabled": False},
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
        **market_kwargs,
    )
    compute_static_features(
        seed_panel,
        seed_context.regime_gates,
        {**static_cfg, "feature_raw_rolling_state_container_enabled": False},
        requested_feature_keys=feature_keys,
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
    )
    state_root = tmp_path / "features" / store_id / "_static_state"
    assert list(state_root.glob("raw_rolling_state.*.npz"))

    # The full production tail warmup covers long FFD/transform dependencies
    # (normally 720+ hours).  This synthetic panel is shorter than that, so
    # hand the incremental call its complete raw history while state skips the
    # already persisted prefix and updates only the final 48 bars.
    tail_start = 0
    tail_panel = {key: value.iloc[tail_start:] for key, value in panel.items()}
    incremental_context = compute_static_market_context(
        tail_panel,
        columns,
        cfg=static_cfg,
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
        **market_kwargs,
    )
    incremental = compute_static_features(
        tail_panel,
        incremental_context.regime_gates,
        static_cfg,
        requested_feature_keys=feature_keys,
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
    )
    expected = {
        key: frame.reindex(index=index[seed_end:], columns=columns)
        for key, frame in full.features.items()
        if key in feature_keys
    }
    for key in feature_keys:
        pd.testing.assert_frame_equal(
            incremental.features[key]
            .reindex(index=index[seed_end:], columns=columns),
            expected[key],
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
            obj=f"incremental static feature {key}",
        )

    assert (state_root / "market_transform_state.npz").exists()
    assert (state_root / "causal_transform_state.container.sqlite").exists()
    assert (state_root / "raw_rolling_state.container.sqlite").exists()

    append_static_features(
        incremental.features,
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_store_id=store_id,
        index=incremental.index,
        columns=incremental.columns,
        min_timestamp_by_symbol={symbol: index[seed_end - 1] for symbol in columns},
        source="inference_raw_tail",
        block_max_timestamps=None,
    )
    stored = read_static_features(
        feature_store_ts=store_ts,
        data_root=tmp_path,
        feature_keys=feature_keys,
        symbols=columns,
        start_ts=index[seed_end],
        end_ts=index[-1] + pd.Timedelta(microseconds=1),
    )
    assert stored is not None
    for key in feature_keys:
        pd.testing.assert_frame_equal(
            stored[key].reindex(index=index[seed_end:], columns=columns),
            full.features[key].reindex(index=index[seed_end:], columns=columns),
            check_exact=False,
            rtol=1e-6,
            atol=1e-6,
            obj=f"stored static feature {key}",
        )
    assert list((tmp_path / "features" / store_id).glob("*.parquet.deltas.duckdb"))

    block_report = compare_static_feature_block_sources(
        data_root=tmp_path,
        feature_store_id=store_id,
        feature_keys=feature_keys,
        left_source="pipeline",
        right_source="inference_raw_tail",
        start_ts=index[seed_end],
        end_ts=index[-1],
    )
    assert block_report["overlap_rows"] == (len(index) - seed_end) * len(columns)
    assert block_report["all_within_tolerance"] is True


def test_repeated_short_tail_does_not_regress_raw_rolling_state(tmp_path):
    rng = np.random.default_rng(91)
    index = pd.date_range("2026-01-01", periods=96, freq="h", tz="UTC")
    columns = ["AAA/USD:USD", "BBB/USD:USD"]
    close = pd.DataFrame(
        100.0 + rng.normal(0.0, 0.2, size=(len(index), len(columns))).cumsum(axis=0),
        index=index,
        columns=columns,
        dtype=np.float32,
    )
    panel = {
        "open": close - 0.05,
        "high": close + 0.10,
        "low": close - 0.10,
        "close": close,
        "volume": pd.DataFrame(
            rng.uniform(10.0, 20.0, size=close.shape),
            index=index,
            columns=columns,
            dtype=np.float32,
        ),
    }
    store_id = "20260101_000000"
    cfg = {
        **CFG,
        "data_root": str(tmp_path),
        "live_lgbm_mask_feature_fast_path_enabled": False,
        "feature_raw_rolling_state_container_enabled": False,
    }
    market_kwargs = {
        "trend_sma_hours": 48,
        "gate_vol_lookback_hours": 24,
        "gate_trend_thr": 0.0,
    }
    context = compute_static_market_context(panel, columns, cfg=cfg, **market_kwargs)
    first = compute_static_features(
        panel,
        context.regime_gates,
        cfg,
        requested_feature_keys=["rv_24h"],
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
    )
    state_root = tmp_path / "features" / store_id / "_static_state"
    raw_states = sorted(state_root.glob("raw_rolling_state.*.npz"))
    assert raw_states
    before = {path.name: path.read_bytes() for path in raw_states}

    tail_panel = {key: value.iloc[-48:] for key, value in panel.items()}
    tail_context = compute_static_market_context(
        tail_panel, columns, cfg=cfg, **market_kwargs
    )
    repeated = compute_static_features(
        tail_panel,
        tail_context.regime_gates,
        cfg,
        requested_feature_keys=["ret1h", "rv_24h"],
        data_root=tmp_path,
        feature_store_id=store_id,
        incremental=True,
    )

    for name, payload in before.items():
        assert (state_root / name).read_bytes() == payload
    assert "rv_24h" in repeated.features
