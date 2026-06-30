import numpy as np
import pandas as pd

from extreme_price_movements.performance_regimes.leaf_extraction import ExtractedLeaf
from extreme_price_movements.performance_regimes.leaf_interactions import (
    extract_leaf_guided_interactions,
)
from extreme_price_movements.unsupervised_regime_learning.pipeline import (
    generate_operator_features,
)


def _leaf(uid, features):
    return ExtractedLeaf(
        leaf_uid=uid,
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
        label_edge_mass=0.3,
        directional_perf_edge=1.0,
        positive_perf_edge=1.0,
        perf_edge_mass=0.5,
        oof_contribution=0.1,
        contribution_share=0.2,
        stability=0.9,
        split_path_features=tuple(features),
        split_path_thresholds=tuple(range(len(features))),
        split_path_operators=tuple("<=" for _ in features),
        timestamp_membership=np.array([True, True, False, False, True]),
    )


def test_leaf_guided_pairs_and_triples_come_only_from_retained_leaf_features():
    seeds = extract_leaf_guided_interactions([_leaf("l1", ["a", "b", "c"])])

    assert set(seeds.pairs["feature_i"]).union(seeds.pairs["feature_j"]) <= {"a", "b", "c"}
    assert len(seeds.pairs) == 3
    assert len(seeds.triples) == 1
    assert tuple(seeds.triples.loc[0, ["feature_i", "feature_j", "feature_k"]]) == ("a", "b", "c")


def test_seeded_operator_generation_reuses_unsupervised_regime_pipeline():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="h", tz="UTC"),
            "symbol": ["AAA"] * 8,
            "a": np.linspace(0, 1, 8),
            "b": np.linspace(1, 0, 8),
            "c": np.sin(np.arange(8)),
        }
    )
    pairs = pd.DataFrame({"feature_i": ["a"], "feature_j": ["b"], "candidate_score": [1.0]})
    triples = pd.DataFrame({"feature_i": ["a"], "feature_j": ["b"], "feature_k": ["c"]})

    features = generate_operator_features(
        frame,
        primitive_features=["a", "b", "c"],
        seeded_pairs=pairs,
        seeded_triples=triples,
        mode="leaf_guided",
        cfg={"operators": {"pair_window": 3, "quantile_window": 3, "autocorr_window": 3, "min_periods": 2}},
    )

    assert any(col.startswith("cov_w3__a__b") for col in features.columns)
    assert any(col.startswith("corr_w3__a__b") for col in features.columns)
    assert any(col.startswith("triple_joint_pressure__a__b__c") for col in features.columns)
