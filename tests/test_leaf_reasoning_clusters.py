from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.leaf_reasoning_clusters import (
    LINKAGES,
    THRESHOLDS,
    LeafReasoningClusterConfig,
    LeafReasoningClusterError,
    cluster_leaf_reasoning_signatures,
    pairwise_leaf_reasoning_similarity,
    sweep_leaf_reasoning_clusters,
)


def _rule(
    rule_instance_id: str,
    fold_id: str,
    path: list[str],
    *,
    side_name: str = "long",
    head_name: str = "robust_clear",
    contribution_direction: str = "positive",
) -> dict[str, object]:
    return {
        "rule_instance_id": rule_instance_id,
        "fold_id": fold_id,
        "side_name": side_name,
        "head_name": head_name,
        "contribution_direction": contribution_direction,
        "rule_signature": f"signature:{rule_instance_id}",
        "rule_structural_path_json": json.dumps(
            [{"feature": name, "decision_type": "<=", "threshold_band_index": 2} for name in path]
        ),
        "activation_rate": 0.20,
        "contribution_signature": "contribution:shared",
        "contribution_top_features_json": json.dumps({"ret_1h": 1.0, "volatility": 0.5}),
        "economic_effect": 0.50,
        "portability_score": 0.95,
    }


def _bridge_summary() -> pd.DataFrame:
    # a-b and b-c share half their structural path; a-c shares none.  The
    # non-structural components are equal, so .70 average linkage bridges all
    # three while .70 complete linkage must retain two clusters.
    return pd.DataFrame(
        [
            _rule("a", "fold_1", ["f1"]),
            _rule("b", "fold_2", ["f1", "f2"]),
            _rule("c", "fold_3", ["f2"]),
        ]
    )


def test_similarity_uses_all_five_predeclared_components_and_never_raw_leaf_ids() -> None:
    pairwise = pairwise_leaf_reasoning_similarity(_bridge_summary())
    pair_ab = pairwise.loc[
        (pairwise["left_rule_instance_id"] == "a") & (pairwise["right_rule_instance_id"] == "b")
    ].iloc[0]
    assert pair_ab["structural_similarity"] == pytest.approx(0.5)
    assert pair_ab["activation_similarity"] == pytest.approx(1.0)
    assert pair_ab["contribution_similarity"] == pytest.approx(1.0)
    assert pair_ab["economic_similarity"] == pytest.approx(1.0)
    assert pair_ab["portability_similarity"] == pytest.approx(1.0)
    assert pair_ab["total_similarity"] == pytest.approx(0.35 * 0.5 + 0.20 + 0.20 + 0.15 + 0.10)

    raw = _bridge_summary().assign(leaf_token=[11, 12, 13])
    with pytest.raises(LeafReasoningClusterError, match="raw fold-local leaf identifiers"):
        pairwise_leaf_reasoning_similarity(raw)


def test_distinct_signatures_with_missing_optional_expansions_do_not_gain_false_similarity() -> None:
    summary = _bridge_summary().drop(columns=["rule_structural_path_json", "contribution_top_features_json"])
    summary.loc[1, "contribution_signature"] = "contribution:distinct"
    pair_ab = pairwise_leaf_reasoning_similarity(summary).iloc[0]
    assert pair_ab["structural_similarity"] == 0.0
    assert pair_ab["contribution_similarity"] == 0.0
    assert pair_ab["total_similarity"] == pytest.approx(0.20 + 0.15 + 0.10)


def test_side_head_and_direction_are_hard_cluster_boundaries() -> None:
    summary = pd.DataFrame(
        [
            _rule("long_clear_positive", "fold_1", ["f1"]),
            _rule("long_adverse_positive", "fold_1", ["f1"], head_name="adverse"),
            _rule("long_clear_negative", "fold_1", ["f1"], contribution_direction="negative"),
            _rule("short_clear_positive", "fold_1", ["f1"], side_name="short"),
        ]
    )
    result = cluster_leaf_reasoning_signatures(summary)
    assert not result.pairwise_similarity["compatible"].any()
    mismatch_text = " ".join(result.pairwise_similarity["compatibility_reason"])
    assert all(name in mismatch_text for name in ("side_name", "head_name", "contribution_direction"))
    assert result.assignments["cluster_id"].nunique() == len(summary)


def test_split_activation_and_threshold_interval_gates_are_hard_boundaries() -> None:
    """Rules in one semantic cell still need compatible transferable geometry."""

    summary = pd.DataFrame([
        _rule("opposing_left", "fold_1", ["f1"]),
        _rule("opposing_right", "fold_2", ["f1"]),
        _rule("interval_left", "fold_3", ["f2"]),
        _rule("interval_right", "fold_4", ["f2"]),
        _rule("high_activation", "fold_5", ["f3"]),
        _rule("low_activation", "fold_6", ["f3"]),
    ])
    paths = {
        "opposing_left": [{"feature": "x", "branch": "left", "threshold": 1.0}],
        "opposing_right": [{"feature": "x", "branch": "right", "threshold": 1.0}],
        "interval_left": [{"feature": "y", "branch": "left", "threshold": 1.0}],
        "interval_right": [{"feature": "y", "branch": "right", "threshold": 2.0}],
        "high_activation": [{"feature": "z", "branch": "left", "threshold": 1.0}],
        "low_activation": [{"feature": "z", "branch": "left", "threshold": 1.0}],
    }
    for rule_id, path in paths.items():
        summary.loc[summary["rule_instance_id"].eq(rule_id), "rule_structural_path_json"] = json.dumps(path)
    summary.loc[summary["rule_instance_id"].eq("high_activation"), "activation_rate"] = .90
    summary.loc[summary["rule_instance_id"].eq("low_activation"), "activation_rate"] = .10

    result = cluster_leaf_reasoning_signatures(summary)
    pairs = result.pairwise_similarity.set_index(["left_rule_instance_id", "right_rule_instance_id"])
    opposing = pairs.loc[("opposing_left", "opposing_right")]
    assert not opposing["compatible"]
    assert opposing["contradictory_defining_split"]
    assert "contradictory_defining_split" in opposing["compatibility_reason"]

    interval = pairs.loc[("interval_left", "interval_right")]
    assert not interval["compatible"]
    assert not interval["contradictory_defining_split"]
    assert "incompatible_threshold_interval" in interval["compatibility_reason"]

    activation = pairs.loc[("high_activation", "low_activation")]
    assert not activation["compatible"]
    assert activation["activation_overlap"] == pytest.approx(1.0 / 9.0)
    assert "minimum_activation_overlap" in activation["compatibility_reason"]
    # Missing pairs used to raise in linkage; hard-incompatible pairs now stay
    # as singleton clusters in their otherwise shared semantic cell.
    assert result.assignments["cluster_id"].nunique() == len(summary)


def test_average_and_complete_linkage_have_expected_bridge_behavior_at_point_seven() -> None:
    summary = _bridge_summary()
    average = cluster_leaf_reasoning_signatures(
        summary, config=LeafReasoningClusterConfig(threshold=0.70, linkage="average")
    )
    complete = cluster_leaf_reasoning_signatures(
        summary, config=LeafReasoningClusterConfig(threshold=0.70, linkage="complete")
    )
    assert average.cluster_summary["member_count"].tolist() == [3]
    assert complete.cluster_summary["member_count"].tolist() == [2, 1]
    average_cluster = average.cluster_summary.iloc[0]
    assert average_cluster["medoid_rule_instance_id"] == "b"
    assert average_cluster["fold_coverage_count"] == 3
    assert average_cluster["available_fold_count"] == 3
    assert average_cluster["fold_coverage_fraction"] == pytest.approx(1.0)


def test_predeclared_threshold_and_linkage_sweep_is_exhaustive_and_fixed() -> None:
    results = sweep_leaf_reasoning_clusters(_bridge_summary())
    assert set(results) == {(threshold, linkage) for threshold in THRESHOLDS for linkage in LINKAGES}
    assert results[(0.60, "average")].cluster_summary["member_count"].tolist() == [3]
    assert results[(0.90, "average")].cluster_summary["member_count"].tolist() == [1, 1, 1]
    with pytest.raises(LeafReasoningClusterError, match="threshold"):
        LeafReasoningClusterConfig(threshold=0.75)
    with pytest.raises(LeafReasoningClusterError, match="immutable"):
        LeafReasoningClusterConfig(structural_weight=0.40)
    with pytest.raises(LeafReasoningClusterError, match="activation-overlap"):
        LeafReasoningClusterConfig(minimum_activation_overlap=0.10)


def test_one_rule_contract_returns_an_empty_but_typed_pairwise_audit() -> None:
    result = cluster_leaf_reasoning_signatures(_bridge_summary().iloc[[0]])
    assert result.pairwise_similarity.empty
    assert "compatible" in result.pairwise_similarity
    assert result.cluster_summary.loc[0, "member_count"] == 1
