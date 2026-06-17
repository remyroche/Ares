import numpy as np
import pandas as pd

from extreme_price_movements.candidate_drift_calibration import (
    CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS,
    CANDIDATE_DRIFT_FEATURE_COLUMNS,
    CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS,
    _transform_calibration_atlas_features,
    candidate_drift_forward_oos_feature_frame,
    compact_candidate_drift_calibrator_state,
    fit_transform_candidate_drift_calibrator,
    hydrate_candidate_drift_calibrator_state,
    transform_candidate_drift_features,
)
from extreme_price_movements.regime_adaptor import _append_candidate_drift_calibration_features
from extreme_price_movements.regime_adaptor import _append_candidate_drift_calibration_fit_blocks
from extreme_price_movements.regime_adaptor import _resolve_direct_label_regime_targets


def _synthetic_panel(n: int = 180):
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rank = np.linspace(0.05, 0.99, n)
    x = pd.DataFrame(
        {
            "feature_drift_psi_core_80": rng.normal(0.0, 1.0, n),
            "feature_drift_ks_bin_mean": rng.normal(0.0, 1.0, n),
            "mahalanobis_mean_shift": rng.normal(0.0, 1.0, n),
            "uncertainty_score": rng.random(n),
            "raw_state_knn_distance": rng.random(n),
            "max_abs_zscore": rng.random(n) * 3.0,
            "mean_abs_zscore": rng.random(n),
            "p95_PSI": rng.random(n),
            "mean_PSI": rng.random(n),
            "pca_reconstruction_error": rng.random(n),
            "missing_count": rng.integers(0, 3, n),
            "stale_feature_count": rng.integers(0, 2, n),
            "recent_base_meta_disagreement_abs_sub_mean_7d": rng.random(n),
            "recent_global_abs_top15_calibration_error_5d": rng.random(n),
        }
    )
    # Make the most similar neighborhood around the last rows profitable.
    net = rng.normal(-0.002, 0.004, n)
    net[-70:] = rng.normal(0.012, 0.003, 70)
    candidates = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": np.where(np.arange(n) % 2 == 0, "BTC/USD:USD", "ETH/USD:USD"),
            "side": "short",
            "strategy_id": "strategy_a",
            "normalized_rank_score": rank,
            "net_return": net,
            "gross_return": net + 0.002,
        }
    )
    return x, candidates, ts


def _directional_panel(n: int = 180, *, positive_high_axis: bool = False):
    rng = np.random.default_rng(123)
    ts = pd.date_range("2026-02-01", periods=n, freq="h", tz="UTC")
    axis = np.r_[np.full(n // 2, -2.0), np.full(n - n // 2, 2.0)]
    x = pd.DataFrame(
        {
            "directional_source_drift_axis": axis + rng.normal(0.0, 0.10, n),
            "feature_drift_psi_core_80": axis * 0.50 + rng.normal(0.0, 0.10, n),
            "uncertainty_score": rng.random(n),
            "rank_signal": np.linspace(0.05, 0.95, n),
        }
    )
    if positive_high_axis:
        net = np.where(axis > 0.0, 0.015, -0.012) + rng.normal(0.0, 0.001, n)
    else:
        net = np.where(axis > 0.0, -0.012, 0.015) + rng.normal(0.0, 0.001, n)
    candidates = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": "BTC/USD:USD",
            "side": "short",
            "strategy_id": "directional_strategy",
            "normalized_rank_score": np.linspace(0.05, 0.95, n),
            "net_return": net,
            "gross_return": net + 0.001,
        }
    )
    return x, candidates, ts


def _atlas_panel(n_per_regime: int = 200):
    rng = np.random.default_rng(321)
    centers = np.asarray(
        [
            [-30.0, -30.0],
            [-30.0, 0.0],
            [-30.0, 30.0],
            [0.0, -30.0],
            [0.0, 30.0],
            [30.0, -30.0],
            [30.0, 0.0],
            [30.0, 30.0],
        ],
        dtype=float,
    )
    rows = []
    regimes = []
    local_move = []
    for cid, center in enumerate(centers):
        move = rng.normal(0.0, 0.30, n_per_regime)
        orth = rng.normal(0.0, 0.05, n_per_regime)
        x = center[0] + orth
        y = center[1] + move
        rows.append(np.column_stack([x, y]))
        regimes.extend([cid] * n_per_regime)
        local_move.extend(move.tolist())
    coords = np.vstack(rows)
    regimes = np.asarray(regimes)
    local_move = np.asarray(local_move)
    perm = rng.permutation(len(coords))
    coords = coords[perm]
    regimes = regimes[perm]
    local_move = local_move[perm]
    n = len(coords)
    ts = pd.date_range("2026-03-01", periods=n, freq="h", tz="UTC")
    sign = np.where(regimes % 2 == 0, 1.0, -1.0)
    net = 0.010 * sign * local_move + rng.normal(0.0, 0.001, n)
    rank = np.clip(0.55 + 0.30 * np.tanh(sign * local_move), 0.01, 0.99)
    x = pd.DataFrame(
        {
            "feature_drift_psi_core_80": coords[:, 0],
            "raw_state_knn_distance": coords[:, 1],
            "uncertainty_score": np.abs(local_move),
            "rank_signal": rank,
            "distance_to_bad_archetype": np.maximum(0.1, 1.0 - sign * local_move),
            "distance_to_good_archetype": np.maximum(0.1, 1.0 + sign * local_move),
            "archetype_oof_bad_rate_lift": np.where(net <= 0.0, 1.3, 0.8),
        }
    )
    candidates = pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": [f"ASSET{i % 24}/USD:USD" for i in range(n)],
            "side": np.where(regimes % 2 == 0, "short", "long"),
            "strategy_id": "atlas_strategy",
            "normalized_rank_score": rank,
            "net_return": net,
            "gross_return": net + 0.001,
        }
    )
    return x, candidates, ts


def test_candidate_drift_features_are_bounded_and_complete():
    x, candidates, ts = _synthetic_panel()
    state, features, report = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=10,
        max_reference_rows=100,
    )

    assert state["enabled"] is True
    assert report["rows"] == len(x)
    assert set(CANDIDATE_DRIFT_FEATURE_COLUMNS).issubset(features.columns)
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    assert "contrib_bad_closeness_score" in CANDIDATE_DRIFT_FEATURE_COLUMNS
    assert "contrib_bad_archetype_cosine" not in CANDIDATE_DRIFT_FEATURE_COLUMNS
    for col in CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS:
        assert col in features.columns
    for col in (
        "knn_dist_pct_k50",
        "distribution_ood_score",
        "prediction_disagreement_score",
        "recent_calibration_risk_score",
        "ood_risk_score",
        "similarity_support_score",
        "max_abs_zscore_pct",
        "mean_PSI_pct",
    ):
        assert ((features[col] >= 0.0) & (features[col] <= 1.0)).all()


def test_candidate_drift_forward_oos_report_is_time_safe():
    x, candidates, ts = _synthetic_panel(260)
    state, features, report = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=120,
        include_forward_oos_report=True,
        forward_oos_folds=3,
    )

    assert state["enabled"] is True
    assert not features.empty
    forward = report["forward_oos"]
    assert forward["enabled"] is True
    assert forward["split"] == "fit_past_transform_future"
    assert forward["folds_completed"] >= 1
    for fold in forward["folds"]:
        if fold.get("enabled"):
            assert fold["train_rows"] > 0
            assert fold["validation_rows"] > 0
            assert fold["train_end_ts"] < fold["validation_start_ts"]


def test_candidate_drift_forward_oos_feature_frame_is_time_safe():
    x, candidates, ts = _synthetic_panel(260)
    features, report = candidate_drift_forward_oos_feature_frame(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=120,
        n_folds=3,
    )

    assert report["enabled"] is True
    assert report["split"] == "fit_past_transform_future"
    assert report["covered_rows"] > 0
    assert report["neutral_rows"] < len(x)
    assert set(CANDIDATE_DRIFT_FEATURE_COLUMNS).issubset(features.columns)
    assert set(CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS).issubset(features.columns)
    assert np.isfinite(features.to_numpy(dtype=float)).all()
    for fold in report["folds"]:
        if fold.get("enabled"):
            assert fold["train_end_ts"] < fold["validation_start_ts"]


def test_candidate_drift_fit_blocks_use_oos_training_and_full_final_features():
    x, candidates, ts = _atlas_panel()
    train_block, final_block, state, report = _append_candidate_drift_calibration_fit_blocks(
        x,
        candidates,
        timestamps=ts,
    )

    assert state["enabled"] is True
    assert report["training_feature_mode"] == "forward_oos_past_fit_future_transform"
    assert report["final_fit_feature_mode"] == "full_artifact_fit_for_live_parity"
    assert train_block.shape[0] == len(x)
    assert final_block.shape[0] == len(x)
    assert "nearest_regime_distance_pct_global" in train_block.columns
    assert "nearest_regime_distance_pct_global" in final_block.columns
    for col in CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS:
        assert col not in train_block.columns
        assert col not in final_block.columns
    for col in CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS:
        assert col not in train_block.columns
        assert col not in final_block.columns
    assert report["forward_oos_training_features"]["covered_rows"] > 0


def test_candidate_drift_compact_sidecar_save_load_parity(tmp_path):
    x, candidates, ts = _synthetic_panel(220)
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=120,
    )
    sidecar = tmp_path / "candidate_drift_arrays.npz"
    compact = compact_candidate_drift_calibrator_state(state, sidecar)
    assert sidecar.exists()
    assert "reference_matrix" not in compact
    assert "compact_array_sidecar" in compact

    hydrated = hydrate_candidate_drift_calibrator_state(compact, base_dir=tmp_path)
    assert "reference_matrix" in hydrated
    qx = x.iloc[-12:].copy()
    qm = candidates.iloc[-12:].copy()
    original = transform_candidate_drift_features(
        qx,
        state,
        candidate_frame=qm,
        timestamps=qm["timestamp"],
        training_mode=False,
    )
    restored = transform_candidate_drift_features(
        qx,
        hydrated,
        candidate_frame=qm,
        timestamps=qm["timestamp"],
        training_mode=False,
    )
    compare_cols = list(CANDIDATE_DRIFT_FEATURE_COLUMNS) + list(
        CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS
    )
    np.testing.assert_allclose(
        original[compare_cols].to_numpy(dtype=float),
        restored[compare_cols].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_shifted_rows_have_higher_knn_distance_percentile_and_ood_score():
    x, candidates, ts = _synthetic_panel()
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=10,
        max_reference_rows=120,
    )
    in_domain = transform_candidate_drift_features(
        x.iloc[-20:].copy(),
        state,
        candidate_frame=candidates.iloc[-20:].copy(),
        timestamps=ts[-20:],
        training_mode=False,
    )
    shifted_x = x.iloc[-20:].copy()
    for col in state["feature_columns"]:
        if col in shifted_x.columns:
            shifted_x[col] = shifted_x[col] + 6.0
    shifted = transform_candidate_drift_features(
        shifted_x,
        state,
        candidate_frame=candidates.iloc[-20:].copy(),
        timestamps=ts[-20:],
        training_mode=False,
    )

    assert shifted["knn_dist_pct_k50"].mean() > in_domain["knn_dist_pct_k50"].mean()
    assert shifted["ood_risk_score"].mean() > in_domain["ood_risk_score"].mean()


def test_local_ev_uses_historical_neighbors_and_excludes_future_rows():
    x, candidates, ts = _synthetic_panel()
    state, features, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=140,
    )

    early = features.iloc[:10]
    late = features.iloc[-20:]
    assert early["local_sample_n_k50"].max() < 50
    assert late["local_ev_shrunk_k50"].mean() > early["local_ev_shrunk_k50"].mean()
    assert late["local_hit_rate_k50"].mean() > early["local_hit_rate_k50"].mean()


def test_directional_bad_alignment_tracks_bad_policy_centroid():
    x, candidates, ts = _directional_panel(positive_high_axis=False)
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=4,
        max_reference_rows=160,
    )
    good_like = x.iloc[[5]].copy()
    bad_like = x.iloc[[-5]].copy()
    q = pd.concat([good_like, bad_like], ignore_index=True)
    meta = pd.concat([candidates.iloc[[5]], candidates.iloc[[-5]]], ignore_index=True)
    out = transform_candidate_drift_features(
        q,
        state,
        candidate_frame=meta,
        timestamps=meta["timestamp"],
        training_mode=False,
    )

    assert out.loc[1, "feature_drift_bad_cosine"] > out.loc[0, "feature_drift_bad_cosine"]
    assert out.loc[1, "unknown_direction_score"] < 0.50


def test_directional_ev_spread_rewards_historically_profitable_direction():
    x, candidates, ts = _directional_panel(positive_high_axis=True)
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=4,
        max_reference_rows=160,
    )
    low_like = x.iloc[[5]].copy()
    high_like = x.iloc[[-5]].copy()
    q = pd.concat([low_like, high_like], ignore_index=True)
    meta = pd.concat([candidates.iloc[[5]], candidates.iloc[[-5]]], ignore_index=True)
    out = transform_candidate_drift_features(
        q,
        state,
        candidate_frame=meta,
        timestamps=meta["timestamp"],
        training_mode=False,
    )

    assert out.loc[1, "directional_ev_spread_k50"] > out.loc[0, "directional_ev_spread_k50"]
    assert out.loc[1, "directional_local_ev_shrunk_k50"] > out.loc[0, "directional_local_ev_shrunk_k50"]


def test_old_candidate_calibrator_state_without_directional_spec_stays_compatible():
    x, candidates, ts = _synthetic_panel()
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=120,
    )
    state = dict(state)
    state.pop("directional_drift_spec", None)
    out = transform_candidate_drift_features(
        x.iloc[-10:].copy(),
        state,
        candidate_frame=candidates.iloc[-10:].copy(),
        timestamps=ts[-10:],
        training_mode=False,
    )

    assert set(CANDIDATE_DRIFT_FEATURE_COLUMNS).issubset(out.columns)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert (out["feature_drift_bad_cosine"] == 0.0).all()
    assert ((out["unknown_unsupported_score"] >= 0.0) & (out["unknown_unsupported_score"] <= 1.0)).all()


def test_calibration_atlas_enables_local_distance_and_support_features():
    x, candidates, ts = _atlas_panel()
    state, features, report = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=len(x),
    )

    assert state["calibration_atlas"]["enabled"] is True
    assert report["calibration_atlas_enabled"] is True
    assert state["calibration_atlas"]["k"] >= 8
    for col in (
        "nearest_regime_distance_pct_global",
        "nearest_regime_distance_pct_local",
        "regime_membership_entropy",
        "atlas_support_quality",
        "local_directional_ev_shrunk",
    ):
        assert col in features.columns
        assert np.isfinite(features[col].to_numpy(dtype=float)).all()
    for col in CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS:
        assert col in features.columns
        assert col not in CANDIDATE_DRIFT_FEATURE_COLUMNS
    assert ((features["nearest_regime_distance_pct_global"] >= 0.0) & (features["nearest_regime_distance_pct_global"] <= 1.0)).all()
    assert ((features["nearest_regime_distance_pct_local"] >= 0.0) & (features["nearest_regime_distance_pct_local"] <= 1.0)).all()
    assert ((features["atlas_support_quality"] >= 0.0) & (features["atlas_support_quality"] <= 1.0)).all()


def test_calibration_atlas_entropy_margin_and_distance_calibration():
    x, candidates, ts = _atlas_panel()
    state, _, _ = fit_transform_candidate_drift_calibrator(
        x,
        candidates,
        timestamps=ts,
        max_features=8,
        max_reference_rows=len(x),
    )
    atlas = state["calibration_atlas"]
    centers = np.asarray(atlas["centers"], dtype=float)
    comps = np.asarray(atlas["embedding_components"], dtype=float)
    emb_center = np.asarray(atlas["embedding_center"], dtype=float)
    c0 = centers[0]
    c1 = centers[1]
    direction = c0 - c1
    direction = direction / max(np.sqrt(np.sum(direction * direction)), 1e-12)
    embedded_queries = np.vstack(
        [
            c0,
            (c0 + c1) / 2.0,
            c0 + 8.0 * direction,
        ]
    )
    z_queries = emb_center.reshape(1, -1) + embedded_queries @ comps
    meta = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-05-01", periods=3, freq="h", tz="UTC"),
            "symbol": ["TEST/USD:USD"] * 3,
            "side": ["short"] * 3,
            "strategy_id": ["atlas_strategy"] * 3,
        }
    )
    out = _transform_calibration_atlas_features(
        z_queries,
        state,
        metadata=meta,
        timestamps=meta["timestamp"],
        training_mode=False,
        index=meta.index,
    )

    assert out.loc[1, "regime_membership_entropy"] >= out.loc[0, "regime_membership_entropy"]
    assert out.loc[1, "top2_regime_margin"] <= out.loc[0, "top2_regime_margin"]
    assert out.loc[2, "nearest_regime_distance_pct_global"] >= out.loc[0, "nearest_regime_distance_pct_global"]
    assert out.loc[2, "nearest_regime_distance_pct_local"] >= out.loc[0, "nearest_regime_distance_pct_local"]


def test_calibration_atlas_support_quality_is_monotone():
    from extreme_price_movements.candidate_drift_calibration import _atlas_support_quality

    weak = _atlas_support_quality(
        support_n=20,
        effective_n=2,
        time_span_days=3,
        asset_count=1,
        neighbor_age_days=180,
        membership_concentration=0.25,
        k=8,
    )
    strong = _atlas_support_quality(
        support_n=800,
        effective_n=50,
        time_span_days=180,
        asset_count=20,
        neighbor_age_days=2,
        membership_concentration=0.95,
        k=8,
    )

    assert 0.0 <= weak <= 1.0
    assert 0.0 <= strong <= 1.0
    assert strong > weak


def test_calibration_atlas_diagnostics_are_not_appended_to_regime_model_features():
    x, candidates, ts = _atlas_panel()
    augmented, state, report = _append_candidate_drift_calibration_features(
        x,
        candidates,
        timestamps=ts,
        fit=True,
    )

    assert state["calibration_atlas"]["enabled"] is True
    assert report["calibration_atlas_enabled"] is True
    assert "contrib_bad_closeness_score" in augmented.columns
    for col in CANDIDATE_DRIFT_DIAGNOSTIC_COLUMNS:
        assert col not in augmented.columns
    for col in CANDIDATE_DRIFT_LEGACY_ALIAS_COLUMNS:
        assert col not in augmented.columns


def test_direct_label_resolution_uses_supplied_labels_not_pnl_correctness():
    frame = pd.DataFrame(
        {
            "meta_tbm_soft_label": [0.9, 0.8, 0.2, 0.1] * 80,
            "meta_tbm_label": [1, 1, 0, 0] * 80,
            "sample_weight": [2.0, 1.0, 1.0, 2.0] * 80,
        }
    )
    # Deliberately conflict with labels: positive labels have negative pnl.
    pnl = np.array([-0.01, -0.02, 0.03, 0.04] * 80, dtype=float)
    meta = np.linspace(0.2, 0.9, len(frame))

    y_soft, y_hard, sw, diag = _resolve_direct_label_regime_targets(
        frame,
        meta_scores=meta,
        pnl=pnl,
    )

    assert diag["enabled"] is True
    assert diag["soft_label_column"] == "meta_tbm_soft_label"
    assert diag["hard_label_column"] == "meta_tbm_label"
    assert y_soft is not None and y_hard is not None and sw is not None
    assert y_hard[:4].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert float(y_soft[0]) > float(y_soft[2])
