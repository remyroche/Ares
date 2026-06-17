import numpy as np
import pandas as pd

from extreme_price_movements.config import CFG, CURRENT_REGIME_AE_FEATURE_KEYS
from extreme_price_movements.regime_ae_features import (
    CURRENT_REGIME_AE_FEATURE_COLUMNS,
    fit_transform_current_regime_ae_features,
    fit_transform_current_regime_ae_features_walk_forward,
    fit_current_regime_ae_state,
    transform_current_regime_ae_features,
)
from extreme_price_movements.training_utils import get_base_feature_keys, get_meta_feature_keys


def test_current_regime_ae_fit_transform_outputs_expected_columns():
    n = 32
    ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            **{
                f"feature_{i}": np.sin(np.arange(n) / (i + 2.0)) + i * 0.01
                for i in range(6)
            },
            "calibrated_score": np.linspace(0.2, 0.9, n),
        }
    )
    frame.loc[3:5, "feature_2"] = np.nan

    features, state = fit_transform_current_regime_ae_features(
        frame,
        frame,
        feature_columns=[f"feature_{i}" for i in range(6)],
        score_target=frame["calibrated_score"].to_numpy(dtype=np.float32),
        cfg={
            "regime_ae_min_rows": 10,
            "regime_ae_min_features": 4,
            "regime_ae_max_features": 8,
            "regime_ae_max_train_rows": 64,
            "regime_ae_max_epochs": 1,
            "regime_ae_batch_size": 16,
        },
    )

    assert state["enabled"] is True
    assert list(features.columns) == list(CURRENT_REGIME_AE_FEATURE_COLUMNS)
    assert np.isfinite(features.to_numpy(dtype=np.float32)).all()
    assert features["ae_reconstruction_error_percentile"].between(0.0, 1.0).all()
    assert features["ae_latent_distance_percentile"].between(0.0, 1.0).all()

    live = transform_current_regime_ae_features(frame.tail(5), state)
    assert list(live.columns) == list(CURRENT_REGIME_AE_FEATURE_COLUMNS)
    assert np.isfinite(live.to_numpy(dtype=np.float32)).all()


def test_current_regime_ae_features_are_regime_adaptor_only():
    ae = set(CURRENT_REGIME_AE_FEATURE_KEYS)
    assert ae == set(CURRENT_REGIME_AE_FEATURE_COLUMNS)
    assert ae.issubset(set(CFG["REGIME_ADAPTOR_FEATURE_ORDER"]))
    assert not ae.intersection(get_base_feature_keys("long", CFG))
    assert not ae.intersection(get_meta_feature_keys("reg", CFG))


def test_current_regime_ae_requires_recent_rows_by_default():
    n = 24
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="30D", tz="UTC"),
            **{f"feature_{i}": np.arange(n, dtype=np.float32) + i for i in range(4)},
        }
    )
    state = fit_current_regime_ae_state(
        frame,
        feature_columns=[f"feature_{i}" for i in range(4)],
        cfg={
            "regime_ae_lookback_days": 10,
            "regime_ae_min_rows": 10,
            "regime_ae_min_features": 4,
        },
    )
    assert state["enabled"] is False
    assert state["reason"] == "insufficient_recent_rows"


def test_current_regime_ae_walk_forward_uses_prior_only_blocks():
    n = 48
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
            **{
                f"feature_{i}": np.sin(np.arange(n) / (i + 2.0)) + i * 0.1
                for i in range(6)
            },
            "calibrated_score": np.linspace(0.2, 0.8, n),
        }
    )
    features, state = fit_transform_current_regime_ae_features_walk_forward(
        frame,
        frame,
        feature_columns=[f"feature_{i}" for i in range(6)],
        score_target=frame["calibrated_score"].to_numpy(dtype=np.float32),
        cfg={
            "regime_ae_min_rows": 10,
            "regime_ae_walk_forward_min_prior_rows": 20,
            "regime_ae_min_features": 4,
            "regime_ae_max_features": 8,
            "regime_ae_max_train_rows": 64,
            "regime_ae_max_epochs": 1,
            "regime_ae_batch_size": 16,
            "regime_ae_oof_block_hours": 12,
        },
    )
    assert state["enabled"] is True
    generation = state["candidate_generation"]
    assert generation["mode"] == "walk_forward_prior_only"
    assert generation["disabled_blocks"] >= 1
    assert generation["enabled_blocks"] >= 1
    assert np.isfinite(features.to_numpy(dtype=np.float32)).all()
