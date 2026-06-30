import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.leaf_extraction import ExtractedLeaf
from extreme_price_movements.performance_regimes.leaf_scoring import (
    estimate_leaf_stability,
    prune_leaves,
    score_directional_edges,
    score_leaf_oof_contribution,
)


def _leaf(**kwargs):
    defaults = dict(
        leaf_uid="leaf",
        strategy="s1",
        direction="bad",
        fold_id=1,
        tree_id=0,
        leaf_id=0,
        leaf_value=0.0,
        parent_value=None,
        coverage=0.5,
        weighted_coverage=0.5,
        n_active=2,
        weighted_n_active=2.0,
        leaf_label_mean=0.8,
        global_label_mean=0.5,
        leaf_strategy_perf_mean=-1.0,
        global_strategy_perf_mean=0.0,
        directional_label_edge=0.3,
        positive_label_edge=0.3,
        label_edge_mass=0.15,
        directional_perf_edge=1.0,
        positive_perf_edge=1.0,
        perf_edge_mass=0.5,
        oof_contribution=0.1,
        contribution_share=0.2,
        stability=0.9,
        split_path_features=("a", "b"),
        split_path_thresholds=(0.0, 1.0),
        split_path_operators=("<=", ">"),
        timestamp_membership=np.array([True, True, False, False]),
    )
    defaults.update(kwargs)
    return ExtractedLeaf(**defaults)


def test_direction_aware_edge_mass_uses_positive_direction_not_squared_edge():
    bad = score_directional_edges(
        direction="bad",
        leaf_label_mean=0.8,
        global_label_mean=0.5,
        leaf_strategy_perf_mean=-2.0,
        global_strategy_perf_mean=0.0,
        weighted_coverage=0.25,
    )
    good = score_directional_edges(
        direction="good",
        leaf_label_mean=0.7,
        global_label_mean=0.5,
        leaf_strategy_perf_mean=2.0,
        global_strategy_perf_mean=0.0,
        weighted_coverage=0.25,
    )
    wrong_bad = score_directional_edges(
        direction="bad",
        leaf_label_mean=0.8,
        global_label_mean=0.5,
        leaf_strategy_perf_mean=2.0,
        global_strategy_perf_mean=0.0,
        weighted_coverage=0.25,
    )

    assert np.isclose(bad["label_edge_mass"], 0.25 * 0.3)
    assert np.isclose(bad["perf_edge_mass"], 0.25 * 2.0)
    assert np.isclose(good["perf_edge_mass"], 0.25 * 2.0)
    assert wrong_bad["perf_edge_mass"] == 0.0


def test_oof_contribution_positive_and_pruning_keeps_positive_stable_leaf():
    leaf = _leaf()
    y = pd.Series([1.0, 1.0, 0.0, 0.0])
    model = pd.Series([0.9, 0.8, 0.1, 0.2])
    baseline = pd.Series([0.5, 0.5, 0.5, 0.5])
    weight = pd.Series([1.0, 1.0, 1.0, 1.0])

    contribution = score_leaf_oof_contribution(leaf, model, baseline, y, weight)
    assert contribution > 0.0
    kept = prune_leaves([leaf], min_stability=0.5, absolute_min_coverage=0.01)
    assert [item.leaf_uid for item in kept] == ["leaf"]


def test_leaf_stability_requires_positive_contribution_and_counts_active_blocks():
    time_blocks = pd.Series([0, 0, 1, 1, 2, 2, 3, 3])
    active = np.array([True, False, True, False, True, False, True, False])
    stable = _leaf(
        timestamp_membership=active,
        active_positions=np.flatnonzero(active).astype(np.int32),
        oof_contribution=0.1,
        directional_label_edge=0.2,
    )
    unstable = _leaf(
        timestamp_membership=active,
        active_positions=np.flatnonzero(active).astype(np.int32),
        oof_contribution=0.0,
        directional_label_edge=0.2,
    )

    assert estimate_leaf_stability(stable, time_blocks=time_blocks, min_block_count=4) > 0.0
    assert estimate_leaf_stability(unstable, time_blocks=time_blocks, min_block_count=4) == 0.0
