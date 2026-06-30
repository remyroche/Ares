import numpy as np

from extreme_price_movements.performance_regimes.leaf_deduplication import (
    deduplicate_leaves_by_jaccard,
)
from extreme_price_movements.performance_regimes.leaf_extraction import ExtractedLeaf


def _leaf(uid, *, strategy="s1", direction="bad", quality=1.0):
    return ExtractedLeaf(
        leaf_uid=uid,
        strategy=strategy,
        direction=direction,
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
        label_edge_mass=quality,
        directional_perf_edge=1.0,
        positive_perf_edge=1.0,
        perf_edge_mass=quality,
        oof_contribution=quality,
        contribution_share=quality,
        stability=1.0,
        split_path_features=("a", "b"),
        split_path_thresholds=(0.0, 1.0),
        split_path_operators=("<=", ">"),
        timestamp_membership=np.array([True, True, False, False]),
    )


def test_same_direction_jaccard_keeps_higher_quality_and_cross_strategy_retains():
    low = _leaf("low", quality=0.1)
    high = _leaf("high", quality=1.0)
    other_strategy = _leaf("other", strategy="s2", quality=0.1)
    low = ExtractedLeaf(**{**low.__dict__, "active_positions": np.array([0, 1], dtype=np.int32)})
    high = ExtractedLeaf(**{**high.__dict__, "active_positions": np.array([0, 1], dtype=np.int32)})
    other_strategy = ExtractedLeaf(**{**other_strategy.__dict__, "active_positions": np.array([0, 1], dtype=np.int32)})

    kept = deduplicate_leaves_by_jaccard([low, high, other_strategy])

    assert {leaf.leaf_uid for leaf in kept} == {"high", "other"}


def test_opposite_direction_strong_contradiction_drops_both():
    bad = _leaf("bad", direction="bad", quality=1.0)
    good = _leaf("good", direction="good", quality=0.9)

    kept = deduplicate_leaves_by_jaccard([bad, good])

    assert kept == []
