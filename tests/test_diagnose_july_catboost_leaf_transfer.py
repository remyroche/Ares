import numpy as np

from scripts.diagnose_july_catboost_leaf_transfer import js_divergence, leaf_support_features


def test_js_divergence_is_zero_for_identical_occupancy():
    assert js_divergence(np.array([2, 3, 5]), np.array([2, 3, 5])) == 0.0


def test_leaf_support_marks_unseen_and_low_support_leaves():
    train = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    evaluation = np.array([[0, 1], [2, 0]], dtype=np.int64)
    support, drift = leaf_support_features(train, evaluation)

    assert support.loc[0, "leaf_support_min"] == 2
    assert support.loc[1, "leaf_unseen_tree_fraction"] == 0.5
    assert drift["mean_tree_unseen_fraction"] > 0
