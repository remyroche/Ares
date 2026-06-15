import numpy as np
import pandas as pd

from extreme_price_movements import lgbm_pipeline as lp
from extreme_price_movements.lgbm_archetype_features import (
    ARCHETYPE_FEATURE_NAMES,
    BASE_ERROR_ARCHETYPE_FEATURE_NAMES,
    META_RAW_CONTRIB_SVD_FEATURE_NAMES,
    RAW_STATE_DIAGNOSTIC_FEATURE_NAMES,
    RAW_STATE_DISTRIBUTION_FEATURE_NAMES,
    RAW_STATE_SVD_FEATURE_NAMES,
    contrib_summary_frame,
    fit_raw_state_archetype_state,
    fit_residual_error_archetype_state,
    is_archetype_feature_name,
    is_raw_contrib_feature_name,
    raw_contrib_feature_mapping,
    raw_contrib_frame,
    transform_raw_state_archetype_features,
    transform_residual_error_archetype_features,
)


def test_predict_contrib_matrix_drops_lightgbm_bias_term():
    class FakeModel:
        def predict(self, X, pred_contrib=False):
            assert pred_contrib is True
            return np.array(
                [
                    [1.0, -2.0, 0.5, 999.0],
                    [-0.25, 0.75, 1.0, 999.0],
                ],
                dtype=np.float32,
            )

    X = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 2.0], "c": [2.0, 3.0]})
    contrib = lp._predict_contrib_matrix(FakeModel(), X, n_features=3)

    assert contrib.shape == (2, 3)
    assert np.max(contrib) < 999.0
    summary = contrib_summary_frame(contrib)
    assert summary["contrib_abs_sum"].iloc[0] == np.float32(3.5)
    assert summary["top_3_contrib_abs_sum"].iloc[0] == np.float32(3.5)
    assert "contrib_abs_sum" in ARCHETYPE_FEATURE_NAMES


def test_raw_state_archetype_transform_uses_frozen_selected_features():
    train = pd.DataFrame(
        {
            "f0": np.linspace(-1.0, 1.0, 12),
            "f1": np.linspace(1.0, -1.0, 12),
            "ignored": np.arange(12, dtype=np.float32),
        }
    )
    timestamps = pd.date_range("2026-01-01", periods=len(train), freq="15min", tz="UTC")
    assets = np.repeat(["BTC", "ETH"], 6)
    state = fit_raw_state_archetype_state(
        train,
        ["f0", "f1", "missing_feature"],
        timestamps=timestamps,
        assets=assets,
        random_state=7,
    )
    valid = pd.DataFrame({"f1": [2.0, -2.0], "f0": [2.0, -2.0]})
    out = transform_raw_state_archetype_features(
        valid,
        state,
        timestamps=pd.date_range("2026-01-02", periods=2, freq="15min", tz="UTC"),
        assets=["BTC", "BTC"],
    )

    expected_cols = set(RAW_STATE_SVD_FEATURE_NAMES + RAW_STATE_DIAGNOSTIC_FEATURE_NAMES)
    assert expected_cols.issubset(out.columns)
    assert set(RAW_STATE_DISTRIBUTION_FEATURE_NAMES).issubset(out.columns)
    assert set(RAW_STATE_DISTRIBUTION_FEATURE_NAMES).issubset(ARCHETYPE_FEATURE_NAMES)
    assert np.isfinite(out.to_numpy(dtype=np.float32)).all()
    assert out["raw_state_ks_mean"].between(0.0, 1.0).all()
    assert out["raw_state_svd_ks_mean"].between(0.0, 1.0).all()
    assert {"raw_state_svd_mean", "raw_state_svd_std"}.issubset(out.columns)
    assert np.isfinite(out[["raw_state_svd_mean", "raw_state_svd_std"]].to_numpy(dtype=np.float32)).all()
    assert state.feature_names == ["f0", "f1", "missing_feature"]


def test_leaf_support_target_and_model_space_proximity_are_fold_train_based():
    class FakeBooster:
        def dump_model(self):
            return {
                "tree_info": [
                    {
                        "tree_structure": {
                            "split_index": 0,
                            "internal_value": 0.0,
                            "left_child": {
                                "leaf_index": 1,
                                "leaf_count": 25,
                                "leaf_weight": 2.5,
                                "leaf_value": 0.4,
                            },
                            "right_child": {
                                "leaf_index": 2,
                                "leaf_count": 75,
                                "leaf_weight": 7.5,
                                "leaf_value": -0.2,
                            },
                        }
                    }
                ]
            }

    class FakeModel:
        booster_ = FakeBooster()
        _ares_lgbm_leaf_training_diagnostics_ = [
            {
                "leaf_ids": np.array([1, 2], dtype=np.int32),
                "train_freq": np.array([0.25, 0.75], dtype=np.float32),
                "target_mean": np.array([0.8, 0.2], dtype=np.float32),
                "target_std": np.array([0.1, 0.2], dtype=np.float32),
                "target_iqr": np.array([0.05, 0.15], dtype=np.float32),
                "target_range": np.array([0.2, 0.4], dtype=np.float32),
                "target_abs_mean": np.array([0.03, 0.08], dtype=np.float32),
                "pred_mean": np.array([0.7, 0.3], dtype=np.float32),
                "error_mean": np.array([0.1, 0.2], dtype=np.float32),
            }
        ]

        def predict(self, X, pred_leaf=False, **_kwargs):
            assert pred_leaf is True
            return np.array([[1], [2], [1]], dtype=np.int32)

    frames: dict[str, np.ndarray] = {}
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    lp._append_leaf_diagnostics(
        frames,
        [FakeModel()],
        X,
        prediction=np.array([0.9, 0.4, 0.6], dtype=np.float32),
        support_diagnostics=True,
        target_diagnostics=True,
        centroid_diagnostics=False,
    )

    assert np.allclose(frames["leaf_hit_rate_avg"], [0.8, 0.2, 0.8])
    assert np.allclose(frames["support_gap"], [0.1, 0.2, -0.2], atol=1e-6)
    assert np.allclose(frames["leaf_proximity_mean"], [0.25, 0.75, 0.25])
    assert np.allclose(frames["leaf_model_space_distance_mean"], [0.75, 0.25, 0.75])
    assert np.allclose(frames["leaf_surprisal_mean"], -np.log([0.25, 0.75, 0.25]))


def test_raw_contrib_export_names_and_meta_input_transform_are_stable():
    feature_names = ["price_ret_1h", "vol z/24", "raw state"]
    contrib = np.array(
        [
            [0.5, -1.0, 0.25],
            [-0.25, 0.75, 0.5],
            [0.1, -0.2, 0.3],
            [0.2, -0.1, 0.4],
        ],
        dtype=np.float32,
    )
    raw = raw_contrib_frame(contrib, feature_names)
    mapping = raw_contrib_feature_mapping(feature_names)

    assert list(raw.columns) == list(mapping.keys())
    assert all(is_raw_contrib_feature_name(c) for c in raw.columns)
    assert mapping[list(raw.columns)[1]] == "vol z/24"

    state = lp._fit_raw_contrib_input_state(raw, list(raw.columns), random_state=11)
    transformed = lp._raw_contrib_model_input_frame(
        raw,
        passthrough_features=[],
        raw_contrib_features=list(raw.columns),
        state=state,
        output_feature_names=META_RAW_CONTRIB_SVD_FEATURE_NAMES,
    )

    assert list(transformed.columns) == META_RAW_CONTRIB_SVD_FEATURE_NAMES
    assert np.isfinite(transformed.to_numpy(dtype=np.float32)).all()


def test_base_residual_error_archetype_state_outputs_stable_columns():
    rng = np.random.default_rng(13)
    good = rng.normal(loc=-1.5, scale=0.15, size=(80, 4)).astype(np.float32)
    bad = rng.normal(loc=1.5, scale=0.15, size=(80, 4)).astype(np.float32)
    frame = pd.DataFrame(
        np.vstack([good, bad]),
        columns=[
            "prob_uncertainty",
            "leaf_surprisal_mean",
            "feature_drift_psi_core",
            "raw_state_mahalanobis",
        ],
    )
    y_bad = np.r_[np.zeros(len(good), dtype=np.float32), np.ones(len(bad), dtype=np.float32)]

    state = fit_residual_error_archetype_state(
        frame,
        y_bad,
        min_rows=40,
        min_role_support=10,
        random_state=5,
    )
    out = transform_residual_error_archetype_features(frame, state)

    assert state.enabled
    assert list(out.columns) == BASE_ERROR_ARCHETYPE_FEATURE_NAMES
    assert np.isfinite(out.to_numpy(dtype=np.float32)).all()
    assert out["base_error_archetype_is_bad"].sum() > 0
    assert out["base_error_archetype_is_good"].sum() > 0
    assert is_archetype_feature_name("base_error_archetype_is_bad")
    assert is_archetype_feature_name("base_H5_base_error_archetype_is_bad")


def test_rank_bin_oof_features_use_training_fold_reference():
    train_pred = np.array([0.10, 0.20, 0.80, 0.90], dtype=np.float32)
    train_rank = lp._safe_rank_pct(train_pred)
    train_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    stats = lp._fit_rank_bin_stats_oof(train_y, train_rank, classifier=True, returns=train_y)

    valid_pred = np.array([0.85], dtype=np.float32)
    valid_rank = lp._rank_pct_against_reference(valid_pred, train_pred)
    frames: dict[str, np.ndarray] = {}
    lp._append_rank_bin_oof_features(frames, valid_rank, stats)

    leaky_full_rank = lp._safe_rank_pct(np.r_[train_pred, valid_pred])
    leaky_stats = lp._fit_rank_bin_stats_oof(
        np.r_[train_y, 0.0],
        leaky_full_rank,
        classifier=True,
        returns=np.r_[train_y, 0.0],
    )
    leaky_frames: dict[str, np.ndarray] = {}
    lp._append_rank_bin_oof_features(leaky_frames, leaky_full_rank[-1:], leaky_stats)

    assert valid_rank[0] == np.float32(0.75)
    assert frames["rank_bin_win_rate_oof"][0] > leaky_frames["rank_bin_win_rate_oof"][0]
    assert frames["rank_bin_lift_oof"][0] > leaky_frames["rank_bin_lift_oof"][0]


def test_lgbm_frame_uses_training_history_defaults_only_for_effectiveness_features():
    model = lp.LGBMStabilityModel(mode="classifier")
    model.selected_features = ["raw_signal", "recent_global_rank_ic_5d"]
    model.model_effectiveness_history_defaults_ = {
        "recent_global_rank_ic_5d": 1.25,
    }

    out = model._frame(pd.DataFrame({"raw_signal": [0.2, 0.4]}))

    assert np.allclose(out["recent_global_rank_ic_5d"], [1.25, 1.25])

    model.selected_features = ["raw_signal", "missing_raw_market_feature"]
    try:
        model._frame(pd.DataFrame({"raw_signal": [0.2]}))
    except ValueError as exc:
        assert "missing_raw_market_feature" in str(exc)
    else:
        raise AssertionError("missing raw features must still fail closed")

    model.selected_features = ["raw_signal", "percentile_rank_in_recent_range_5d"]
    model.model_effectiveness_history_defaults_ = {
        "percentile_rank_in_recent_range_5d": 0.9,
    }
    try:
        model._frame(pd.DataFrame({"raw_signal": [0.2]}))
    except ValueError as exc:
        assert "percentile_rank_in_recent_range_5d" in str(exc)
    else:
        raise AssertionError("raw recent market features must still fail closed")
