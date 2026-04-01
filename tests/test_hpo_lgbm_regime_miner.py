import numpy as np

from extreme_price_movements.hpo_lgbm_regime_miner import (
    MIN_GAIN_GRID,
    HPOConfig,
    build_hpo_params,
    make_support_preference_weights,
    score_rule_matrix_vectorized,
    _support_quality_score,
)


def test_support_quality_score_prefers_center_and_preferred_band():
    supports = np.array([0.15, 0.12, 0.125, 0.20], dtype=np.float64)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    centered = _support_quality_score(np.array([0.15]), np.array([1.0]))
    preferred_edge = _support_quality_score(np.array([0.125]), np.array([1.0]))
    outside_preferred = _support_quality_score(np.array([0.20]), np.array([1.0]))

    assert centered > preferred_edge
    assert preferred_edge > outside_preferred
    assert _support_quality_score(supports, weights) > 0.0


def test_score_rule_matrix_invalidates_when_too_many_rules_outside_support_band():
    y_val = np.array([0.4, 0.3, 0.2, 0.1, -0.1, -0.2, -0.3, -0.4], dtype=np.float32)
    vol_val = np.ones_like(y_val)
    rule_matrix = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    )

    score, diagnostics = score_rule_matrix_vectorized(
        y_val=y_val,
        vol_val=vol_val,
        rule_matrix=rule_matrix,
        support_min=0.30,
        support_max=0.40,
        target_support=0.15,
    )

    assert not np.isfinite(score)
    assert diagnostics["reason"] == "too_many_rules_outside_support_band"


def test_build_hpo_params_enforces_depth_aware_min_data_in_leaf_floor():
    cfg = HPOConfig(alpha=0.9, min_gain_to_split=0.001, min_leaf_frac=0.0005)
    params = build_hpo_params(
        {
            "learning_rate": 0.03,
            "objective": "quantile",
            "metric": "quantile",
            "max_depth": 6,
            "num_leaves": 64,
        },
        cfg,
        n_train_subsample=20000,
    )

    assert params["min_data_in_leaf"] >= 40


def test_support_preference_weights_favor_target_support_rows():
    x = np.array(
        [
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.float32,
    )
    weights = make_support_preference_weights(
        x,
        target_pct=0.33,
        preferred_low_pct=0.25,
        preferred_high_pct=0.40,
        strength=0.5,
    )

    assert weights[0] > weights[2]
    assert weights[1] > weights[3]


def test_min_gain_grid_includes_higher_value():
    assert 0.004 in MIN_GAIN_GRID
