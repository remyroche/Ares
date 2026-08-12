from __future__ import annotations

import json

import pandas as pd
import pytest

from extreme_price_movements.structural_rule_families import (
    StructuralRuleFamilyConfig,
    StructuralRuleFamilyError,
    cluster_structural_rule_families,
    materialize_structural_family_posteriors,
)


def _rule(identifier: str, *, fold: str, model: str, path: list[str], version: str = "base-v1") -> dict[str, object]:
    return {
        "rule_instance_id": identifier,
        "fold_id": fold,
        "model_id": model,
        "side_name": "long",
        "head_name": "p_clear",
        "base_model_version": version,
        "model_layer": "base",
        "rule_signature": "|".join(path),
        "rule_structural_path_json": json.dumps([
            {"feature": item, "decision_type": "<=", "threshold_band_index": 2} for item in path
        ]),
        "train_leaf_frequency": 0.25,
        "ensemble_tree_contribution": 0.5,
    }


def test_structural_families_ignore_frozen_geometry_for_alignment_and_require_recurrence() -> None:
    catalogue = pd.DataFrame([
        _rule("a", fold="f1", model="m1", path=["trend", "flow"]),
        _rule("b", fold="f2", model="m2", path=["trend", "flow"]),
        _rule("c", fold="f3", model="m3", path=["volatility"]),
    ])
    catalogue.loc[catalogue.rule_instance_id.eq("b"), "ensemble_tree_contribution"] = -9_999.0
    result = cluster_structural_rule_families(
        catalogue,
        config=StructuralRuleFamilyConfig(frozen_contribution_column="ensemble_tree_contribution"),
    )
    pair = result.pairwise_similarity.loc[
        result.pairwise_similarity.left_rule_instance_id.eq("a")
        & result.pairwise_similarity.right_rule_instance_id.eq("b")
    ].iloc[0]
    assert pair.structural_similarity == pytest.approx(1.0)
    recurrent = result.family_summary.loc[result.family_summary.member_count.eq(2)].iloc[0]
    assert recurrent.is_recurrent
    assert recurrent.distinct_fold_count == 2
    assert recurrent.distinct_model_count == 2
    assert len(result.selected_cluster_ids) == 1
    assert not result.family_summary.loc[result.family_summary.member_count.eq(1), "is_selected"].any()


def test_structural_family_boundaries_keep_versions_and_layers_separate() -> None:
    catalogue = pd.DataFrame([
        _rule("a", fold="f1", model="m1", path=["trend"]),
        _rule("b", fold="f2", model="m2", path=["trend"]),
        _rule("other-version", fold="f3", model="m3", path=["trend"], version="base-v2"),
    ])
    result = cluster_structural_rule_families(catalogue)
    assert len(result.pairwise_similarity) == 1
    assert result.family_summary.groupby("base_model_version")["member_count"].sum().to_dict() == {
        "base-v1": 2, "base-v2": 1,
    }


def test_structural_alignment_rejects_outcomes_even_when_not_referenced() -> None:
    catalogue = pd.DataFrame([
        _rule("a", fold="f1", model="m1", path=["trend"]),
        _rule("b", fold="f2", model="m2", path=["trend"]),
    ]).assign(economic_effect=[10.0, -10.0])
    with pytest.raises(StructuralRuleFamilyError, match="forbidden during structural alignment"):
        cluster_structural_rule_families(catalogue)


def test_posterior_projection_emits_only_family_shares_and_unassigned_mass() -> None:
    catalogue = pd.DataFrame([
        _rule("a", fold="f1", model="m1", path=["trend"]),
        _rule("b", fold="f2", model="m2", path=["trend"]),
        _rule("c", fold="f3", model="m3", path=["flow"]),
    ])
    result = cluster_structural_rule_families(catalogue)
    activations = pd.DataFrame({
        "candidate_id": ["x", "x", "y"],
        "rule_instance_id": ["a", "c", "b"],
        "base_tree_contribution": [2.0, 2.0, -3.0],
    })
    projected = materialize_structural_family_posteriors(
        activations, result.assignments, contribution_column="base_tree_contribution",
    )
    fields = list(projected.cluster_id_to_feature.values())
    assert all("leaf" not in column and "rule_instance" not in column for column in projected.features.columns)
    x = projected.features.set_index("candidate_id").loc["x"]
    assert x[fields[0]] == pytest.approx(0.5)
    assert x["base_structural_family__unassigned_mass"] == pytest.approx(0.5)
    y = projected.features.set_index("candidate_id").loc["y"]
    assert y[fields[0]] == pytest.approx(1.0)
    assert y["base_structural_family__unassigned_mass"] == pytest.approx(0.0)
