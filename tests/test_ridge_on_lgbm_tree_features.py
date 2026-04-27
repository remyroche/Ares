import numpy as np

from extreme_price_movements.ridge_on_lgbm import (
    GLOBAL_TREE_FEATURE_NAMES,
    _fit_dense_tree_feature_pruner,
    _fit_ridge_uncertainty_modulator,
    _global_tree_summary_features,
    _hard_path_stats_for_tree,
    _logit,
    _soft_leaf_probability_matrix,
    _soft_leaf_stats_for_tree,
    _tree_feature_config,
)


def _tiny_tree() -> dict:
    return {
        "split_feature": 0,
        "threshold": 0.0,
        "left_child": {"leaf_value": -0.5},
        "right_child": {
            "split_feature": 1,
            "threshold": 1.0,
            "left_child": {"leaf_value": 0.25},
            "right_child": {"leaf_value": 0.75},
        },
    }


def test_hard_path_margins_are_positive_on_chosen_branches() -> None:
    x = np.asarray([[-1.0, 0.0], [2.0, 0.25], [2.0, 2.0]], dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)

    margins, paths = _hard_path_stats_for_tree(x, _tiny_tree(), scales)

    assert np.all(margins[:, 0] >= 0.0)
    assert np.all(margins[:, 2] >= 0.0)
    assert paths[:, 0].tolist() == [1.0, 2.0, 2.0]


def test_soft_leaf_probabilities_are_valid_distribution() -> None:
    x = np.asarray([[-1.0, 0.0], [2.0, 0.25], [2.0, 2.0]], dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)

    probs, values = _soft_leaf_probability_matrix(
        x, _tiny_tree(), scales, soft_tau_mult=0.25
    )
    stats = _soft_leaf_stats_for_tree(x, _tiny_tree(), scales, soft_tau_mult=0.25)

    assert values.shape == (3,)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)
    assert np.all(stats[:, 2] >= 0.0)
    assert np.all(stats[:, 3] >= 0.0)
    assert np.all(stats[:, 4] >= -1e-6)
    assert np.all(stats[:, 3] >= stats[:, 3] - stats[:, 4] - 1e-6)


def test_dense_tree_feature_pruner_drops_constant_features() -> None:
    cfg = _tree_feature_config(feature_var_threshold=1e-8)
    x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )

    keep_idx, names, meta = _fit_dense_tree_feature_pruner(
        x, ["constant", "trend", "trend_dup"], cfg
    )

    assert 0 not in keep_idx
    assert "constant" not in names
    assert meta["dropped_by_variance"] == 1


def test_global_tree_summary_features_include_requested_interactions() -> None:
    names = [
        "mdl0_tree0_min_margin",
        "mdl0_tree0_deep_margin",
        "mdl0_tree0_soft_mean",
        "mdl0_tree0_soft_var",
        "mdl0_tree0_soft_entropy",
        "mdl0_tree1_min_margin",
        "mdl0_tree1_deep_margin",
        "mdl0_tree1_soft_mean",
        "mdl0_tree1_soft_var",
        "mdl0_tree1_soft_entropy",
    ]
    block = np.asarray(
        [
            [1.0, 2.0, 0.5, 0.1, 0.2, 3.0, 4.0, 1.5, 0.3, 0.6],
            [2.0, 3.0, 1.0, 0.2, 0.4, 4.0, 5.0, 2.0, 0.4, 0.8],
        ],
        dtype=np.float32,
    )

    global_block, global_names = _global_tree_summary_features(block, names)

    assert global_names == GLOBAL_TREE_FEATURE_NAMES
    assert global_block.shape == (2, 8)
    assert np.allclose(global_block[:, 0], [2.0, 3.0])
    assert np.allclose(global_block[:, 1], [3.0, 4.0])
    assert np.allclose(global_block[:, 2], [0.2, 0.3])
    assert np.allclose(global_block[:, 3], [0.4, 0.6])
    assert np.allclose(global_block[:, 4], [2.0, 4.5])


def test_uncertainty_modulator_stays_disabled_without_improvement() -> None:
    y = np.asarray([0, 1] * 30, dtype=np.int8)
    base_prob = np.where(y == 1, 0.8, 0.2).astype(np.float32)
    base_score = _logit(base_prob)
    global_features = np.zeros(
        (len(y), len(GLOBAL_TREE_FEATURE_NAMES)), dtype=np.float32
    )

    result = _fit_ridge_uncertainty_modulator(
        base_score,
        global_features,
        y,
        sample_weight=None,
        random_state=42,
        min_improvement=0.01,
        correction_weight=0.2,
    )

    assert result["enabled"] is False
    assert np.allclose(result["prob"], base_prob)
    assert np.allclose(result["score"], base_score)
