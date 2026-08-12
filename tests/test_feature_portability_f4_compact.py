from __future__ import annotations

import pytest

from extreme_price_movements.feature_portability_f4_compact import (
    F4CompactContractError,
    compact_contract_payload,
    compact_contracts_for_ranked_groups,
    f4_transform_groups,
    selected_compact_feature_manifest,
)


def _f3_contract() -> dict[str, list[str]]:
    suffixes = (
        "__causal_rank_w90", "__causal_rank_w180", "__causal_robust_z_w90",
        "__causal_robust_z_w180", "__causal_delta_p4", "__causal_delta_p24",
    )
    return {
        side: [*sources, *(f"{source}{suffix}" for suffix in suffixes for source in sources)]
        for side, sources in (("long", ["long_a", "long_b"]), ("short", ["short_a"]))
    }


def test_builds_nested_compact_contracts_from_complete_f3_transform_families() -> None:
    groups = f4_transform_groups(_f3_contract())
    contracts = compact_contracts_for_ranked_groups(
        groups,
        ranked_transform_groups=("robust_z_w90", "rank_w180", "delta_p4", "rank_w90", "robust_z_w180", "delta_p24"),
    )
    assert set(contracts) == {"F4_compact_top01", "F4_compact_top02", "F4_compact_top03"}
    assert contracts["F4_compact_top01"]["long"] == (
        "long_a", "long_b", "long_a__causal_robust_z_w90", "long_b__causal_robust_z_w90",
    )
    assert len(contracts["F4_compact_top03"]["short"]) == 4


def test_rejects_partial_or_misaligned_f3_transform_contracts() -> None:
    bad = _f3_contract()
    bad["long"].remove("long_b__causal_delta_p24")
    with pytest.raises(F4CompactContractError, match="exactly one causal transform"):
        f4_transform_groups(bad)


def test_selected_manifest_requires_exact_cross_transport_field_lists() -> None:
    groups = f4_transform_groups(_f3_contract())
    ranking = ("robust_z_w90", "rank_w180", "delta_p4", "rank_w90", "robust_z_w180", "delta_p24")
    first = compact_contracts_for_ranked_groups(groups, ranked_transform_groups=ranking)
    payload = compact_contract_payload(
        source_representation="F3_plus_relative",
        by_transport={"a": first, "b": first}, ranking_by_transport={"a": ranking, "b": ranking},
    )
    selected = {"representation": "F4_compact_top02", "feature_count": 6}
    manifest = selected_compact_feature_manifest(selection=selected, compact_contracts=payload, required_transports=("a", "b"))
    assert manifest["status"] == "F4_TRANSPORT_SELECTED_COMPACT_FEATURE_MANIFEST"
    assert manifest["base_control_verified"] is True
    assert manifest["full_f3_control_verified"] is True

    second = compact_contracts_for_ranked_groups(
        groups,
        ranked_transform_groups=("delta_p4", "rank_w180", "robust_z_w90", "rank_w90", "robust_z_w180", "delta_p24"),
    )
    unstable = compact_contract_payload(
        source_representation="F3_plus_relative",
        by_transport={"a": first, "b": second}, ranking_by_transport={"a": ranking, "b": ("delta_p4", "rank_w180", "robust_z_w90", "rank_w90", "robust_z_w180", "delta_p24")},
    )
    rejected = selected_compact_feature_manifest(selection=selected, compact_contracts=unstable, required_transports=("a", "b"))
    assert rejected["status"] == "F4_SELECTED_PROCEDURE_HAS_NO_STABLE_COMPACT_CONTRACT"
    assert rejected["feature_contract"] is None
