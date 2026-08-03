from __future__ import annotations

from scripts.materialize_capture_feature_expansion import (
    build_side_feature_contract,
)


def test_side_feature_contract_preserves_frozen_side_lists() -> None:
    manifests = {
        "long": {
            "features": ["raw_a", "dae_b16_02", "gmm_ood_score"],
        },
        "short": {
            "features": ["raw_b"],
        },
    }
    sides, raw, generated = build_side_feature_contract(
        manifests, ["core_score"]
    )
    assert sides["long"] == [
        "core_score",
        "capture_raw__raw_a",
        "capture_repr__dae_b16_02",
        "capture_repr__gmm_ood_score",
    ]
    assert sides["short"] == ["core_score", "capture_raw__raw_b"]
    assert raw == ["raw_a", "raw_b"]
    assert generated == ["dae_b16_02", "gmm_ood_score"]


def test_side_feature_contract_can_isolate_raw_features() -> None:
    manifests = {
        "long": {"features": ["raw_a", "dae_b16_02", "gmm_ood_score"]},
        "short": {"features": ["raw_b"]},
    }
    sides, raw, generated = build_side_feature_contract(
        manifests, ["core_score"], include_generated=False
    )
    assert sides["long"] == ["core_score", "capture_raw__raw_a"]
    assert sides["short"] == ["core_score", "capture_raw__raw_b"]
    assert raw == ["raw_a", "raw_b"]
    assert generated == []
