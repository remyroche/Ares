import pytest
import numpy as np
import pandas as pd
from extreme_price_movements.feature_select.cv import CVConfig
from extreme_price_movements.feature_select.scoring import UtilityConfig, FeatureSelectConfig
from extreme_price_movements.feature_select.run import run_feature_selection
from extreme_price_movements.feature_selection_extreme_events import (
    linear_prescreen_enet,
    mdi_feature_selection_v3,
)
from sklearn.ensemble import ExtraTreesRegressor

def generate_synthetic_data(n_samples=2000, n_features=20, n_informative=3, seed=42):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])

    # Generate labels primarily based on the first few features
    y_raw = X.iloc[:, :n_informative].sum(axis=1) + rng.randn(n_samples) * 0.5
    y_bin = (y_raw > y_raw.median()).astype(int).values

    return X, y_bin, y_raw.values

def test_feature_selection_sanity():
    """Test 1: Sanity on synthetic data (classification)"""
    X, y, _ = generate_synthetic_data(n_samples=2000, n_features=10, n_informative=3)

    cv_config = CVConfig(n_splits=3, min_train_size=500, val_size=400)
    utility_config = UtilityConfig(utility_mode="topq_mean", topq=0.2, direction="higher_is_better")
    fs_config = FeatureSelectConfig(min_features=5, n_repeats_perm=2, utility_drop_tol=0.1)

    lgbm_params = {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 50, "early_stopping_rounds": 10}

    result = run_feature_selection(
        X=X,
        y=y,
        groups=None,
        time_index=None,
        model_kind="binary",
        quantile_alpha=None,
        cv_config=cv_config,
        lgbm_params=lgbm_params,
        utility_config=utility_config,
        fs_config=fs_config,
        random_seed=42,
        output_dir="/tmp"
    )

    selected = result.selected_features

    # Informative features should be retained (feat_0, feat_1, feat_2)
    retained_informative = sum(1 for f in ["feat_0", "feat_1", "feat_2"] if f in selected)
    assert retained_informative >= 2, "Failed to retain informative features"

def test_feature_selection_regression():
    """Test feature selection for regression"""
    X, _, y = generate_synthetic_data(n_samples=2000, n_features=10, n_informative=3)

    cv_config = CVConfig(n_splits=2, min_train_size=1000, val_size=400)
    utility_config = UtilityConfig(utility_mode="topq_mean", topq=0.2, direction="higher_is_better")
    fs_config = FeatureSelectConfig(min_features=3, n_repeats_perm=1, utility_drop_tol=0.5)

    lgbm_params = {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 50, "early_stopping_rounds": 5}

    result = run_feature_selection(
        X=X,
        y=y,
        groups=None,
        time_index=None,
        model_kind="regression",
        quantile_alpha=None,
        cv_config=cv_config,
        lgbm_params=lgbm_params,
        utility_config=utility_config,
        fs_config=fs_config,
        random_seed=42,
        output_dir="/tmp"
    )

    assert "feat_0" in result.selected_features or "feat_1" in result.selected_features


def test_linear_prescreen_enet_respects_max_drop_cap():
    X, _, y = generate_synthetic_data(n_samples=1200, n_features=100, n_informative=5)
    kept = linear_prescreen_enet(X, y, n_select=10, multiplier=4)
    assert len(kept) == 75


def test_mdi_drops_near_constant_features_before_selection():
    X, _, y = generate_synthetic_data(n_samples=800, n_features=24, n_informative=4)
    X["const_base"] = 1.0
    X["const_base_G_VOL_0"] = 0.0
    X["const_base_G_VOL_1"] = 1.0

    res = mdi_feature_selection_v3(
        X,
        y,
        base_model=ExtraTreesRegressor(
            n_estimators=50,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=1,
        ),
        end_features=8,
        min_features=5,
        n_splits=2,
        selector_target="regression",
        selector_loss="huber",
    )

    assert "const_base" not in res.kept_after_dedupe
    assert "const_base_G_VOL_0" not in res.kept_after_dedupe
    assert "const_base_G_VOL_1" not in res.kept_after_dedupe
