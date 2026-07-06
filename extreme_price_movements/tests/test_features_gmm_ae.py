from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    AE_GMM_LATENT_FEATURE_COLUMNS,
    _cluster_stability,
    _smooth_probabilities,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from extreme_price_movements.lgbm_pipeline import LGBMStabilityModel
from extreme_price_movements.optional_model_features import (
    is_optional_generated_model_feature_key,
)


def test_cluster_stability_matches_rolling_loop_semantics():
    labels = np.asarray([0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1], dtype=np.int32)
    window = 4

    age, stability, flips = _cluster_stability(labels, window=window)

    expected_age = []
    expected_stability = []
    expected_flips = []
    last_change = 0
    for i in range(len(labels)):
        if i > 0 and labels[i] != labels[i - 1]:
            last_change = i
        start = max(0, i - window + 1)
        recent = labels[start : i + 1]
        expected_age.append(float(i - last_change))
        expected_stability.append(float(np.mean(recent == labels[i])))
        expected_flips.append(float(np.sum(recent[1:] != recent[:-1])) if len(recent) > 1 else 0.0)

    np.testing.assert_allclose(age, np.asarray(expected_age, dtype=np.float32))
    np.testing.assert_allclose(stability, np.asarray(expected_stability, dtype=np.float32))
    np.testing.assert_allclose(flips, np.asarray(expected_flips, dtype=np.float32))


def test_smooth_probabilities_matches_original_recurrence():
    prob = np.asarray(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.20, 0.20, 0.60],
            [0.50, 0.25, 0.25],
        ],
        dtype=np.float32,
    )
    lam = 0.8
    expected = np.empty_like(prob, dtype=np.float32)
    prev = prob[0].astype(np.float32)
    expected[0] = prev
    for i in range(1, len(prob)):
        prev = (float(lam) * prev) + ((1.0 - float(lam)) * prob[i])
        total = float(np.sum(prev))
        if total > 0.0:
            prev = prev / total
        expected[i] = prev.astype(np.float32)

    np.testing.assert_allclose(_smooth_probabilities(prob, lam), expected, rtol=1e-6, atol=1e-7)


def test_ae_gmm_features_are_finite_and_schema_stable():
    rng = np.random.default_rng(123)
    x = pd.DataFrame(
        rng.normal(size=(260, 8)).astype("float32"),
        columns=[f"f{i}" for i in range(8)],
    )
    edge = (x["f0"] - 0.3 * x["f1"] + rng.normal(scale=0.2, size=len(x))).to_numpy(
        dtype=np.float32,
    )
    state = fit_ae_gmm_state(
        x,
        economic_targets={
            "returns": edge,
            "target": (edge > 0.0).astype(np.float32),
            "bad_mae_1r": (edge < -0.15).astype(np.float32),
            "timeout": (np.abs(edge) < 0.05).astype(np.float32),
            "clean_positive": (edge > 0.25).astype(np.float32),
            "dirty_positive": ((edge > 0.0) & (edge <= 0.25)).astype(np.float32),
        },
        random_state=17,
        max_train_rows=260,
        ae_max_iter=4,
    )

    features = transform_ae_gmm_features(x, state)

    assert list(features.columns) == list(AE_GMM_FEATURE_COLUMNS)
    assert features.shape == (len(x), len(AE_GMM_FEATURE_COLUMNS))
    assert np.isfinite(features.to_numpy(dtype=np.float32)).all()
    selected_config = state.get("selected_config", {})
    assert "path_cleanliness_score" in selected_config
    assert "clean_positive_contrast" in selected_config
    assert "dirty_positive_contrast" in selected_config
    assert "bad_mae_contrast" in selected_config
    assert "temporal_concentration_score" in selected_config
    for col in (
        "gmm_posterior_max",
        "gmm_posterior_margin",
        "gmm_posterior_delta_1",
        "cluster_entropy_accel_1",
        "dae_reconstruction_error_delta_1",
        "latent_speed",
        "latent_acceleration",
    ):
        assert col in features.columns
        assert is_optional_generated_model_feature_key(col)


def test_ae_gmm_max_train_rows_zero_means_unbounded():
    rng = np.random.default_rng(124)
    x = pd.DataFrame(
        rng.normal(size=(260, 8)).astype("float32"),
        columns=[f"f{i}" for i in range(8)],
    )

    state = fit_ae_gmm_state(
        x,
        random_state=19,
        max_train_rows=0,
        ae_max_iter=4,
    )

    assert state.get("enabled") is True
    assert state.get("gmm_n_components", 0) >= 2


def test_ae_gmm_model_features_default_to_continuous_not_hard_cluster_ids():
    import extreme_price_movements.lgbm_pipeline as lp

    names = lp._ae_gmm_model_feature_names_for_objective("train_meta")

    assert "gmm_prob_0" in names
    assert "gmm_entropy" in names
    assert "mahalanobis_distance" in names
    assert "gmm_cluster_id" not in names
    assert "cluster_t" not in names


def test_lgbm_model_frame_regenerates_ae_gmm_features_from_selected_inputs():
    rng = np.random.default_rng(321)
    x = pd.DataFrame(
        rng.normal(size=(240, 6)).astype("float32"),
        columns=[f"f{i}" for i in range(6)],
    )
    state = fit_ae_gmm_state(
        x,
        random_state=23,
        max_train_rows=240,
        ae_max_iter=4,
    )
    model = LGBMStabilityModel()
    model.selected_features = list(x.columns) + list(AE_GMM_FEATURE_COLUMNS)
    model.input_feature_names = list(x.columns)
    model.ae_gmm_input_features = list(x.columns)
    model.ae_gmm_feature_names = list(AE_GMM_FEATURE_COLUMNS)
    model.ae_gmm_state = state

    frame = model._frame(x.iloc[:7])

    assert list(frame.columns) == model.selected_features
    assert set(AE_GMM_FEATURE_COLUMNS).issubset(frame.columns)
    assert np.isfinite(frame.to_numpy(dtype=np.float32)).all()
    assert "gmm_posterior_max" in frame.columns
    assert "latent_acceleration" in frame.columns


def test_model_uses_no_same_layer_ae_gmm_features_but_exports_context():
    rng = np.random.default_rng(777)
    x = pd.DataFrame(
        rng.normal(size=(240, 6)).astype("float32"),
        columns=[f"f{i}" for i in range(6)],
    )
    state = fit_ae_gmm_state(
        x,
        random_state=29,
        max_train_rows=240,
        ae_max_iter=4,
    )
    model = LGBMStabilityModel()
    model.selected_features = list(x.columns)
    model.input_feature_names = list(x.columns)
    model.ae_gmm_input_features = list(x.columns)
    model.ae_gmm_feature_names = []
    model.ae_gmm_context_feature_names = list(AE_GMM_FEATURE_COLUMNS)
    model.ae_gmm_state = state

    frame = model._frame(x.iloc[:9])

    assert list(frame.columns) == model.selected_features
    assert not set(AE_GMM_LATENT_FEATURE_COLUMNS).intersection(frame.columns)
    assert not any(str(col).startswith("gmm_") for col in frame.columns)
    assert "cluster_entropy_norm" not in frame.columns

    meta = model.transform_internal_model_metrics(x.iloc[:9])

    assert set(AE_GMM_FEATURE_COLUMNS).issubset(meta.columns)
    assert np.isfinite(meta[list(AE_GMM_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)).all()


def test_ae_gmm_context_export_is_optional_when_transform_fails(monkeypatch):
    import extreme_price_movements.lgbm_pipeline as lp

    model = LGBMStabilityModel()
    model.selected_features = ["ret24h"]
    model.input_feature_names = ["ret24h"]
    model.ae_gmm_input_features = ["ret24h"]
    model.ae_gmm_context_feature_names = [
        "dae_b16_00",
        "gmm_prob_0",
        "cluster_entropy_norm",
    ]
    model.ae_gmm_state = {"enabled": True}

    def fail_transform(*args, **kwargs):
        raise RuntimeError("simulated optional transform failure")

    monkeypatch.setattr(lp, "transform_ae_gmm_features", fail_transform)

    meta = model.transform_internal_model_metrics(
        pd.DataFrame({"ret24h": [0.1, 0.2]}, index=["AAA/USDC", "BBB/USDC"])
    )

    assert list(meta[model.ae_gmm_context_feature_names].columns) == (
        model.ae_gmm_context_feature_names
    )
    assert (
        meta[model.ae_gmm_context_feature_names]
        .to_numpy(dtype=np.float32)
        .sum()
        == pytest.approx(0.0)
    )
