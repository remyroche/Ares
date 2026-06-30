import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.archetypes import (
    ArchetypeClusteringConfig,
    build_archetype_activity_targets,
    build_archetype_activity_intensity,
    build_archetype_signed_effect_targets,
    cluster_leaves_into_archetypes,
)
from extreme_price_movements.performance_regimes.labels import (
    build_strategy_performance_labels,
)
from extreme_price_movements.performance_regimes.leaf_extraction import ExtractedLeaf


def _leaf(uid, strategy, direction, membership):
    return ExtractedLeaf(
        leaf_uid=uid,
        strategy=strategy,
        direction=direction,
        fold_id=1,
        tree_id=0,
        leaf_id=0,
        leaf_value=0.0,
        parent_value=None,
        coverage=float(np.mean(membership)),
        weighted_coverage=float(np.mean(membership)),
        n_active=int(np.sum(membership)),
        weighted_n_active=float(np.sum(membership)),
        leaf_label_mean=0.8,
        global_label_mean=0.5,
        leaf_strategy_perf_mean=-1.0,
        global_strategy_perf_mean=0.0,
        directional_label_edge=0.3,
        positive_label_edge=0.3,
        label_edge_mass=0.3,
        directional_perf_edge=1.0,
        positive_perf_edge=1.0,
        perf_edge_mass=0.5,
        oof_contribution=0.1,
        contribution_share=0.2,
        stability=0.9,
        split_path_features=("family_a__x", "family_b__y"),
        split_path_thresholds=(0.0, 1.0),
        split_path_operators=("<=", ">"),
        timestamp_membership=np.asarray(membership, dtype=bool),
    )


def test_archetype_clustering_is_separate_by_strategy_and_direction():
    leaves = [
        _leaf("s1_bad", "s1", "bad", [1, 1, 0, 0, 0]),
        _leaf("s1_good", "s1", "good", [1, 1, 0, 0, 0]),
        _leaf("s2_bad", "s2", "bad", [1, 1, 0, 0, 0]),
    ]

    bundle = cluster_leaves_into_archetypes(
        leaves,
        clustering_config=ArchetypeClusteringConfig(distance_threshold=1.0),
    )

    keys = {(a.strategy, a.direction) for a in bundle.archetypes}
    assert keys == {("s1", "bad"), ("s1", "good"), ("s2", "bad")}


def test_progressive_activity_intensity_and_targets_are_active_inactive_not_effect():
    timestamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    leaves = [_leaf("s1_bad", "s1", "bad", [1, 0, 1, 0, 0])]
    leaves = [
        ExtractedLeaf(
            **{
                **leaves[0].__dict__,
                "active_positions": np.array([0, 2], dtype=np.int32),
            }
        )
    ]
    bundle = cluster_leaves_into_archetypes(leaves)
    archetype = bundle.archetypes[0]

    intensity = build_archetype_activity_intensity(archetype, leaves, timestamps, ewma_halflife=1)
    targets = build_archetype_activity_targets(bundle.archetypes, leaves, timestamps, ewma_halflife=1)
    y = targets.activity[archetype.archetype_id]

    assert intensity.between(0.0, 1.0).all()
    assert y.between(0.0, 1.0).all()
    assert y.max() > 0.0
    assert targets.sample_weights[archetype.archetype_id].between(0.25, 10.0).all()


def test_signed_effect_target_is_separate_from_activity_target():
    timestamps = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
    leaves = [_leaf("s1_bad", "s1", "bad", [1, 0, 1, 0, 0])]
    bundle = cluster_leaves_into_archetypes(leaves)
    targets = build_archetype_activity_targets(bundle.archetypes, leaves, timestamps, ewma_halflife=1)
    trades = pd.DataFrame(
        {
            "timestamp": timestamps,
            "strategy": ["s1"] * len(timestamps),
            "performance": [-1.0, -0.5, 0.0, 0.5, 1.0],
        }
    )
    labels = build_strategy_performance_labels(
        trades,
        strategy_col="strategy",
        timestamp_col="timestamp",
        performance_col="performance",
        strategies=["s1"],
        ewma_halflife=1,
        anchor_mode="minmax",
    )

    signed = build_archetype_signed_effect_targets(bundle.archetypes, targets.activity, labels)
    archetype_id = bundle.archetypes[0].archetype_id

    assert archetype_id in signed
    assert signed[archetype_id].between(-1.0, 1.0).all()
    assert not signed[archetype_id].equals(targets.activity[archetype_id])
