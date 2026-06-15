from __future__ import annotations

import numpy as np
import pandas as pd

from extreme_price_movements.features_gmm_ae import (
    AE_GMM_FEATURE_COLUMNS,
    AE_GMM_LATENT_FEATURE_COLUMNS,
    fit_ae_gmm_state,
    transform_ae_gmm_features,
)
from extreme_price_movements.lgbm_pipeline import LGBMStabilityModel


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
        },
        random_state=17,
        max_train_rows=260,
        ae_max_iter=4,
    )

    features = transform_ae_gmm_features(x, state)

    assert list(features.columns) == list(AE_GMM_FEATURE_COLUMNS)
    assert features.shape == (len(x), len(AE_GMM_FEATURE_COLUMNS))
    assert np.isfinite(features.to_numpy(dtype=np.float32)).all()


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
