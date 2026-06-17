import os

import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG
from extreme_price_movements.inference.model_orchestrator import ModelOrchestrator
from extreme_price_movements.training_utils import get_meta_feature_keys
import extreme_price_movements.training as tr
import extreme_price_movements.data_store as data_store
from extreme_price_movements.data_store import (
    _feature_delta_dir,
    _write_feature_metadata,
    append_symbol_features,
    get_feature_bounds,
    load_features_selected,
    read_symbol_features,
    save_features,
)
from extreme_price_movements.pipeline_steps import (
    _build_tail_only_backfill_cutoffs,
    _chunked_partial_backfill_is_fully_covered,
    _check_feature_source_target_coverage,
    _feature_scan_has_broad_target_gap,
    _feature_time_coverage_backfill_keys,
    _filter_requested_feature_keys_for_runtime_sources,
    _scan_feature_cache_light,
    _validate_feature_snapshot_completeness,
)


def test_meta_policy_slice_availability_exempts_model_derived_lgbm_features(monkeypatch):
    df = pd.DataFrame(
        {
            "raw_good": [1.0, 2.0, 3.0, 4.0],
            "base_prob_x_vol_regime": [np.nan, np.nan, np.nan, np.nan],
            "pred_H10": [np.nan, np.nan, np.nan, np.nan],
            "feature_drift_psi_core": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    def fake_feature_store_availability_matrix(feature_cols, *, cfg):
        assert list(feature_cols) == ["raw_good", "base_prob_x_vol_regime"]
        finite = np.asarray(
            [
                [True, False],
                [True, False],
                [True, False],
                [True, False],
            ],
            dtype=bool,
        )
        return finite, int(finite.shape[0]), "fake policy slice"

    monkeypatch.setattr(
        tr,
        "_feature_store_availability_matrix",
        fake_feature_store_availability_matrix,
    )

    kept = tr._recent_feature_availability_filter(
        df,
        list(df.columns),
        cfg={"lgbm_feature_recent_min_coverage": 0.85, "lgbm_feature_recent_min_rows": 1},
        context="LGBM model race meta_demo_clf",
        exempt_features={
            c for c in df.columns if tr._is_lgbm_model_derived_meta_feature(c)
        },
    )

    assert kept == ["raw_good", "pred_H10", "feature_drift_psi_core"]
    assert "base_prob_x_vol_regime" not in kept


def test_meta_performance_feature_groups_survive_portable_config():
    clf_keys = set(get_meta_feature_keys("clf", CFG))

    assert "recent_global_rolling_ic_5d" in clf_keys
    assert "recent_global_confidence_surprise_5d" in clf_keys
    assert "recent_global_top15_hit_rate_5d" in clf_keys
    assert "base_model_score" in clf_keys
    assert "prob_error" in clf_keys
    assert "recent_hit_rate_20" in clf_keys


def test_base_lgbm_raw_state_drift_features_are_meta_candidates():
    clf_keys = set(get_meta_feature_keys("clf", CFG))

    assert "BASE_LGBM_RAW_STATE_DRIFT_FEATURE_KEYS" in CFG["meta_shared_feature_keys"]
    assert "base_lgbm_feature_drift_cov_shift" in clf_keys
    assert "base_lgbm_raw_state_knn_distance" in clf_keys
    assert "base_lgbm_raw_state_mahalanobis" in clf_keys
    assert "base_lgbm_raw_state_transition_mahalanobis" in clf_keys


def test_inference_materializes_base_performance_meta_features():
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H10",
                "base_model_score",
                "base_model_score_pct",
                "base_model_margin",
                "prob_error",
                "recent_prob_error_20",
                "recent_hit_rate_20",
                "base_model_abs_error_roll20",
                "recent_global_rolling_ic_5d",
            ]
        },
    )()
    orch = ModelOrchestrator(
        {}, runtime_cfg={"meta_trade_rank_window": 20, "strict_feature_parity": False}
    )
    features = pd.DataFrame({"pred_H10": [0.2, 0.8]})

    out = orch._materialize_meta_model_derived_features(
        features,
        model,
        side="long",
        kind="demo",
    )

    assert out["base_model_score"].tolist() == [0.2, 0.8]
    assert out["base_model_score_pct"].tolist() == [0.5, 0.5]
    assert np.allclose(out["base_model_margin"].to_numpy(), [0.3, 0.3])
    assert out["prob_error"].tolist() == [0.5, 0.5]
    assert out["recent_prob_error_20"].tolist() == [0.5, 0.5]
    assert out["recent_hit_rate_20"].tolist() == [0.5, 0.5]
    assert out["base_model_abs_error_roll20"].tolist() == [0.5, 0.5]
    assert out["recent_global_rolling_ic_5d"].tolist() == [0.0, 0.0]


def test_strict_inference_does_not_neutralize_non_causal_meta_features():
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H10",
                "prob_error",
                "recent_hit_rate_20",
                "recent_global_rolling_ic_5d",
            ]
        },
    )()
    orch = ModelOrchestrator(
        {}, runtime_cfg={"meta_trade_rank_window": 20, "strict_feature_parity": True}
    )
    features = pd.DataFrame({"pred_H10": [0.2, 0.8]})

    out = orch._materialize_meta_model_derived_features(
        features,
        model,
        side="long",
        kind="demo",
    )

    assert "prob_error" not in out.columns
    assert "recent_hit_rate_20" not in out.columns
    assert "recent_global_rolling_ic_5d" not in out.columns


def test_inference_materializes_drift_aliases_from_artifact_state(monkeypatch):
    model = type(
        "MetaModelStub",
        (),
        {
            "feature_columns": [
                "pred_H5_regime_centroid_similarity_train",
                "base_H5_feature_drift_psi_core_80",
                "pred_demo_H5_mahalanobis_mean_shift",
                "pred_demo_H5_reg_rare_leaf_low_support_score",
                "feature_drift_psi_core",
            ],
            "model_drift_state_": {"enabled": True},
        },
    )()
    orch = ModelOrchestrator({}, runtime_cfg={"strict_feature_parity": True})
    features = pd.DataFrame({"pred_H5": [0.2, 0.8]})

    def fake_transform_model_drift_features(*args, **kwargs):
        return pd.DataFrame(
            {
                "regime_centroid_similarity_train": [0.91, 0.82],
                "feature_drift_psi_core_80": [0.11, 0.22],
                "mahalanobis_mean_shift": [1.5, 2.5],
                "rare_leaf_low_support_score": [0.03, 0.04],
            },
            index=features.index,
        )

    monkeypatch.setattr(
        "extreme_price_movements.inference.model_orchestrator.transform_model_drift_features",
        fake_transform_model_drift_features,
    )

    out = orch._materialize_meta_model_drift_features(features, model)

    assert np.allclose(out["pred_H5_regime_centroid_similarity_train"], [0.91, 0.82])
    assert np.allclose(out["base_H5_feature_drift_psi_core_80"], [0.11, 0.22])
    assert np.allclose(out["pred_demo_H5_mahalanobis_mean_shift"], [1.5, 2.5])
    assert np.allclose(out["pred_demo_H5_reg_rare_leaf_low_support_score"], [0.03, 0.04])
    assert np.allclose(out["feature_drift_psi_core"], [0.11, 0.22])


def test_feature_generation_guard_fails_when_source_lags_target():
    close = pd.DataFrame(
        {"BTC/USD:USD": [1.0, 2.0]},
        index=pd.DatetimeIndex(
            ["2026-06-03 09:00:00+00:00", "2026-06-03 10:00:00+00:00"],
            name="ts",
        ),
    )

    try:
        _check_feature_source_target_coverage(
            close,
            pd.Timestamp("2026-06-03 11:00:00", tz="UTC"),
            require_target_end=True,
            label="test",
        )
    except RuntimeError as exc:
        assert "source OHLCV only covers" in str(exc)
        assert "2026-06-03T10:00:00+00:00" in str(exc)
        assert "2026-06-03T11:00:00+00:00" in str(exc)
    else:
        raise AssertionError("expected stale source coverage to fail")


def test_feature_cache_scan_detects_per_column_stale_tail(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.DatetimeIndex(
        [
            "2026-06-02 21:00:00+00:00",
            "2026-06-03 10:00:00+00:00",
            "2026-06-03 11:00:00+00:00",
        ],
        name="ts",
    )
    cfx_path = feature_dir / "symbol=CFX_USD:USD.parquet"
    pd.DataFrame(
        {
            "stale_feature": [1.0, np.nan, np.nan],
            "current_feature": [1.0, 2.0, 3.0],
            "__symbol__": ["CFX/USD:USD"] * 3,
        },
        index=idx,
    ).to_parquet(cfx_path)
    _write_feature_metadata(str(cfx_path), "CFX/USD:USD", idx)
    btc_path = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {
            "stale_feature": [1.0, 2.0, 3.0],
            "current_feature": [1.0, 2.0, 3.0],
            "__symbol__": ["BTC/USD:USD"] * 3,
        },
        index=idx,
    ).to_parquet(btc_path)
    _write_feature_metadata(str(btc_path), "BTC/USD:USD", idx)
    close = pd.DataFrame(
        {"CFX/USD:USD": [1.0], "BTC/USD:USD": [1.0]},
        index=idx[-1:],
    )

    scan = _scan_feature_cache_light(
        ts_sig=ts,
        data_root=str(tmp_path),
        expected_keys={"stale_feature", "current_feature"},
        panel_close=close,
    )
    cutoffs, stats = _build_tail_only_backfill_cutoffs(
        ts_sig=ts,
        data_root=str(tmp_path),
        panel_close=close,
        backfill_keys=["stale_feature", "current_feature"],
    )

    assert scan is not None
    assert "CFX/USD:USD" in scan["stale_symbols"]
    assert scan["partial_keys"] == ["stale_feature"]
    assert cutoffs["CFX/USD:USD"] == pd.Timestamp("2026-06-02 21:00:00")
    assert stats["eligible_tail_only"] == 1


def test_tail_cutoff_uses_file_bound_for_pure_tail_gap(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    feature_idx = pd.DatetimeIndex(
        [
            "2026-06-03 09:00:00+00:00",
            "2026-06-03 10:00:00+00:00",
            "2026-06-03 11:00:00+00:00",
        ],
        name="ts",
    )
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {
            "feat_a": [1.0, np.nan, np.nan],
            "feat_b": [1.0, 2.0, 3.0],
            "__symbol__": ["BTC/USD:USD"] * 3,
        },
        index=feature_idx,
    ).to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", feature_idx)
    close = pd.DataFrame(
        {"BTC/USD:USD": [1.0, 1.0]},
        index=pd.DatetimeIndex(
            ["2026-06-03 11:00:00+00:00", "2026-06-03 12:00:00+00:00"],
            name="ts",
        ),
    )

    cutoffs, stats = _build_tail_only_backfill_cutoffs(
        ts_sig=ts,
        data_root=str(tmp_path),
        panel_close=close,
        backfill_keys=["feat_a", "feat_b"],
    )

    assert cutoffs["BTC/USD:USD"] == pd.Timestamp("2026-06-03 11:00:00")
    assert stats["tail_bound_fastpath"] == 1


def test_save_features_slices_rows_before_symbol_payload_materialization(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    idx = pd.DatetimeIndex(
        [
            "2026-06-03 09:00:00+00:00",
            "2026-06-03 10:00:00+00:00",
            "2026-06-03 11:00:00+00:00",
            "2026-06-03 12:00:00+00:00",
        ],
        name="ts",
    )
    feats = {
        "feat_a": np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32),
        "feat_b": np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32),
    }

    save_features(
        feats,
        ts,
        str(tmp_path),
        min_timestamp_by_symbol={
            "BTC/USD:USD": pd.Timestamp("2026-06-03 10:00:00+00:00")
        },
        feat_index=idx,
        feat_columns=["BTC/USD:USD"],
        save_workers=1,
    )

    run_id = ts.strftime("%Y%m%d_%H%M%S")
    fpath = tmp_path / "features" / run_id / "symbol=BTC_USD:USD.parquet"
    saved = read_symbol_features(str(fpath))

    assert list(saved.index) == list(idx[2:])
    assert saved["feat_a"].tolist() == [3.0, 4.0]
    assert saved["feat_b"].tolist() == [30.0, 40.0]


def test_tail_cutoff_uses_target_row_fastpath_for_wide_sparse_target(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.DatetimeIndex(
        ["2026-06-03 11:00:00+00:00", "2026-06-03 12:00:00+00:00"],
        name="ts",
    )
    feature_cols = {f"feat_{i}": [float(i), np.nan] for i in range(128)}
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {
            **feature_cols,
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
        },
        index=idx,
    ).to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", idx)
    close = pd.DataFrame({"BTC/USD:USD": [1.0]}, index=idx[-1:])

    cutoffs, stats = _build_tail_only_backfill_cutoffs(
        ts_sig=ts,
        data_root=str(tmp_path),
        panel_close=close,
        backfill_keys=list(feature_cols),
    )

    assert cutoffs["BTC/USD:USD"] == pd.Timestamp("2026-06-03 11:59:59.999999")
    assert stats["target_row_fastpath"] == 1


def test_feature_cache_scan_counts_only_required_symbols(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.DatetimeIndex(
        ["2026-06-03 10:00:00+00:00"],
        name="ts",
    )
    pd.DataFrame(
        {
            "feat_a": [np.nan],
            "feat_b": [2.0],
            "__symbol__": ["BTC/USD:USD"],
        },
        index=idx,
    ).to_parquet(feature_dir / "symbol=BTC_USD:USD.parquet")
    pd.DataFrame(
        {
            "feat_a": [1.0],
            "feat_b": [2.0],
            "__symbol__": ["ETH/USD:USD"],
        },
        index=idx,
    ).to_parquet(feature_dir / "symbol=ETH_USD:USD.parquet")
    _write_feature_metadata(
        str(feature_dir / "symbol=BTC_USD:USD.parquet"),
        "BTC/USD:USD",
        idx,
    )
    _write_feature_metadata(
        str(feature_dir / "symbol=ETH_USD:USD.parquet"),
        "ETH/USD:USD",
        idx,
    )
    close = pd.DataFrame({"BTC/USD:USD": [1.0]}, index=idx)

    scan = _scan_feature_cache_light(
        ts_sig=ts,
        data_root=str(tmp_path),
        expected_keys={"feat_a", "feat_b"},
        panel_close=close,
    )

    assert scan is not None
    assert scan["missing_keys"] == ["feat_a"]
    assert scan["partial_keys"] == []
    assert scan["available_key_count"] == 1


def test_feature_cache_scan_manifest_reuses_clean_scan(tmp_path, monkeypatch):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.DatetimeIndex(
        ["2026-06-03 10:00:00+00:00"],
        name="ts",
    )
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {
            "feat_a": [1.0],
            "feat_b": [2.0],
            "__symbol__": ["BTC/USD:USD"],
        },
        index=idx,
    ).to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", idx)
    close = pd.DataFrame({"BTC/USD:USD": [1.0]}, index=idx)

    first = _scan_feature_cache_light(
        ts_sig=ts,
        data_root=str(tmp_path),
        expected_keys={"feat_a", "feat_b"},
        panel_close=close,
    )

    def fail_read_symbol_features(*args, **kwargs):
        raise AssertionError("manifest hit should avoid parquet target-row reads")

    monkeypatch.setattr(
        "extreme_price_movements.pipeline_steps.read_symbol_features",
        fail_read_symbol_features,
    )
    second = _scan_feature_cache_light(
        ts_sig=ts,
        data_root=str(tmp_path),
        expected_keys={"feat_a", "feat_b"},
        panel_close=close,
    )

    assert first is not None
    assert second is not None
    assert second["from_manifest"] is True
    assert second["available_key_count"] == first["available_key_count"]
    assert second["missing_keys"] == first["missing_keys"]
    assert second["partial_keys"] == first["partial_keys"]


def test_chunked_partial_backfill_only_skips_when_no_symbol_work_remains():
    assert _chunked_partial_backfill_is_fully_covered([])
    assert not _chunked_partial_backfill_is_fully_covered(["BTC/USD:USD"])


def test_feature_snapshot_validation_allows_naturally_sparse_latest_vwap(tmp_path):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    idx = pd.DatetimeIndex(
        ["2026-06-03 10:00:00+00:00"],
        name="ts",
    )
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    pd.DataFrame(
        {
            "loc_vwap_dev_z_24": [np.nan],
            "loc_range_pos_24": [0.5],
            "__symbol__": ["BTC/USD:USD"],
        },
        index=idx,
    ).to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", idx)
    close = pd.DataFrame({"BTC/USD:USD": [1.0]}, index=idx)

    _validate_feature_snapshot_completeness(
        ts_sig=ts,
        data_root=str(tmp_path),
        expected_keys={"loc_vwap_dev_z_24", "loc_range_pos_24"},
        panel_close=close,
    )


def test_feature_scan_detects_broad_sparse_target_gap():
    scan = {
        "required_symbol_count": 4,
        "available_key_count": 10,
        "partial_keys": ["f1", "f2", "f3"],
        "all_nan_symbol_keys": {
            "A": ["f1", "f2"],
            "B": ["f1", "f2"],
        },
    }

    assert _feature_scan_has_broad_target_gap(scan)


def test_time_coverage_backfill_uses_all_computable_keys():
    keys = {
        "ret24h",
        "dist_ema20_atr",
        "feature_drift_psi_core",
        "leaf_support_mean_frac",
    }

    selected = _feature_time_coverage_backfill_keys(
        keys,
        missing_keys=["feature_drift_psi_core", "leaf_support_mean_frac"],
    )

    assert selected == ["dist_ema20_atr", "ret24h"]


def test_feature_delta_append_visible_to_selected_loader(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_COMPACT_ROWS", "2")
    monkeypatch.setenv("EPM_FEATURE_DELTA_COMPRESSION", "none")
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    base_idx = pd.DatetimeIndex(
        ["2026-06-03 09:00:00+00:00", "2026-06-03 10:00:00+00:00"],
        name="ts",
    )
    base = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0],
            "feat_b": [10.0, 20.0],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
        },
        index=base_idx,
    )
    base.to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", base_idx)

    append_symbol_features(
        str(fpath),
        "BTC/USD:USD",
        pd.DataFrame(
            {"feat_a": [3.0], "feat_b": [30.0]},
            index=pd.DatetimeIndex(["2026-06-03 11:00:00+00:00"], name="ts"),
        ),
    )
    append_symbol_features(
        str(fpath),
        "BTC/USD:USD",
        pd.DataFrame(
            {"feat_a": [4.0]},
            index=pd.DatetimeIndex(["2026-06-03 10:00:00+00:00"], name="ts"),
        ),
    )

    first_ts, last_ts = get_feature_bounds(str(fpath))
    assert pd.Timestamp(first_ts) == pd.Timestamp("2026-06-03 09:00:00+00:00")
    assert pd.Timestamp(last_ts) == pd.Timestamp("2026-06-03 11:00:00+00:00")
    assert not os.path.exists(_feature_delta_dir(str(fpath)))

    loaded = load_features_selected(
        ts=ts,
        root_dir=str(tmp_path),
        feature_keys=["feat_a", "feat_b"],
        symbols=["BTC/USD:USD"],
        start_ts=pd.Timestamp("2026-06-03 09:00:00Z"),
        end_ts=pd.Timestamp("2026-06-03 12:00:00Z"),
    )
    assert loaded is not None
    feat_a = loaded["feat_a"]
    feat_b = loaded["feat_b"]
    ten = pd.Timestamp("2026-06-03 10:00:00Z")
    eleven = pd.Timestamp("2026-06-03 11:00:00Z")
    assert float(feat_a.loc[ten, "BTC/USD:USD"]) == 2.0
    assert float(feat_b.loc[ten, "BTC/USD:USD"]) == 20.0
    assert float(feat_a.loc[eleven, "BTC/USD:USD"]) == 3.0
    assert float(feat_b.loc[eleven, "BTC/USD:USD"]) == 30.0


def test_trusted_cutoff_append_keeps_overlap_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("EPM_FEATURE_DELTA_APPEND", "1")
    monkeypatch.setenv("EPM_FEATURE_DELTA_COMPACT_ROWS", "1000")
    monkeypatch.setenv("EPM_FEATURE_DELTA_TRUST_CUTOFF_APPEND", "1")

    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    base_idx = pd.DatetimeIndex(
        ["2026-06-03 09:00:00+00:00", "2026-06-03 10:00:00+00:00"],
        name="ts",
    )
    base = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0],
            "__symbol__": ["BTC/USD:USD", "BTC/USD:USD"],
        },
        index=base_idx,
    )
    base.to_parquet(fpath)
    _write_feature_metadata(str(fpath), "BTC/USD:USD", base_idx)

    feature_index = pd.DatetimeIndex(
        ["2026-06-03 10:00:00+00:00", "2026-06-03 11:00:00+00:00"],
        name="ts",
    )
    save_features(
        {"feat_a": pd.DataFrame({"BTC/USD:USD": [99.0, 3.0]}, index=feature_index)},
        ts=ts,
        root_dir=str(tmp_path),
        min_timestamp_by_symbol={"BTC/USD:USD": pd.Timestamp("2026-06-03 09:00:00Z")},
        save_workers=1,
    )

    loaded = load_features_selected(
        ts=ts,
        root_dir=str(tmp_path),
        feature_keys=["feat_a"],
        symbols=["BTC/USD:USD"],
        start_ts=pd.Timestamp("2026-06-03 09:00:00Z"),
        end_ts=pd.Timestamp("2026-06-03 12:00:00Z"),
    )
    assert loaded is not None
    feat_a = loaded["feat_a"]
    ten = pd.Timestamp("2026-06-03 10:00:00Z")
    eleven = pd.Timestamp("2026-06-03 11:00:00Z")
    assert float(feat_a.loc[ten, "BTC/USD:USD"]) == 2.0
    assert float(feat_a.loc[eleven, "BTC/USD:USD"]) == 3.0

    bounded = load_features_selected(
        ts=ts,
        root_dir=str(tmp_path),
        feature_keys=["feat_a", "feat_b"],
        symbols=["BTC/USD:USD"],
        start_ts=pd.Timestamp("2026-06-03 10:30:00Z"),
        end_ts=pd.Timestamp("2026-06-03 12:00:00Z"),
    )
    assert bounded is not None
    bounded_a = bounded["feat_a"]
    assert ten not in bounded_a.index
    assert float(bounded_a.loc[eleven, "BTC/USD:USD"]) == 3.0

    exact = load_features_selected(
        ts=ts,
        root_dir=str(tmp_path),
        feature_keys=["feat_a"],
        symbols=["BTC/USD:USD"],
        start_ts=eleven,
        end_ts=eleven,
    )
    assert exact is not None
    exact_a = exact["feat_a"]
    assert list(exact_a.index) == [eleven]
    assert float(exact_a.loc[eleven, "BTC/USD:USD"]) == 3.0


def test_selected_feature_load_preserves_tz_aware_pushdown_bounds(tmp_path, monkeypatch):
    ts = pd.Timestamp("2026-05-23 01:59:47", tz="UTC")
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    feature_dir = tmp_path / "features" / run_id
    feature_dir.mkdir(parents=True)
    fpath = feature_dir / "symbol=BTC_USD:USD.parquet"
    idx = pd.DatetimeIndex(
        ["2026-06-03 11:00:00+00:00"],
        name="ts",
    )
    pd.DataFrame(
        {
            "feat_a": [1.0],
            "__symbol__": ["BTC/USD:USD"],
        },
        index=idx,
    ).to_parquet(fpath)

    captured = {}

    def fake_read_symbol_features(
        parquet_path,
        *,
        columns=None,
        start_ts=None,
        end_ts=None,
        allowed_periods=None,
    ):
        captured["start_ts"] = start_ts
        captured["end_ts"] = end_ts
        return pd.DataFrame(
            {
                "feat_a": [1.0],
                "__symbol__": ["BTC/USD:USD"],
            },
            index=idx,
        )

    monkeypatch.setattr(data_store, "read_symbol_features", fake_read_symbol_features)

    loaded = data_store.load_features_selected(
        ts=ts,
        root_dir=str(tmp_path),
        feature_keys=["feat_a"],
        symbols=["BTC/USD:USD"],
        start_ts=pd.Timestamp("2026-06-03 11:00:00Z"),
        end_ts=pd.Timestamp("2026-06-03 11:00:00.000001Z"),
    )

    assert loaded is not None
    assert captured["start_ts"].tzinfo is not None
    assert captured["end_ts"].tzinfo is not None


def test_runtime_source_filter_accepts_saved_orderbook_side_store():
    idx = pd.DatetimeIndex(["2026-06-04 11:00:00+00:00"], name="ts")
    panel = {
        "close": pd.DataFrame({"SHIB/USD:USD": [0.000005]}, index=idx),
        "volume": pd.DataFrame({"SHIB/USD:USD": [238960992.0]}, index=idx),
    }
    cfg = {
        "feature_portability_mode": "same_exchange_perp",
        "feature_portability_strict": True,
    }

    allowed, skipped = _filter_requested_feature_keys_for_runtime_sources(
        ["ob_depth_l20_to_qv_z_7d", "ret24h"],
        cfg,
        panel,
        ["SHIB/USD:USD"],
    )

    assert "ret24h" in allowed
    assert skipped["ob_depth_l20_to_qv_z_7d"] == "missing_source:orderbook"

    saved_orderbook = pd.DataFrame(
        {"best_bid": [0.0000049], "best_ask": [0.0000051]},
        index=idx,
    )
    allowed, skipped = _filter_requested_feature_keys_for_runtime_sources(
        ["ob_depth_l20_to_qv_z_7d", "ret24h"],
        cfg,
        panel,
        ["SHIB/USD:USD"],
        orderbook_by_symbol={"SHIB/USD:USD": saved_orderbook},
    )

    assert allowed == ["ob_depth_l20_to_qv_z_7d", "ret24h"]
    assert skipped == {}
