from __future__ import annotations

from scripts.materialize_execution_ev_forward_preentry import _timing_features


def test_timing_feature_union_is_deterministic_by_horizon() -> None:
    state = {
        "selected_features_by_horizon": {
            12: ["c", "a"],
            2: ["a", "b"],
            8: ["b", "c"],
        }
    }
    assert _timing_features(state) == ["a", "b", "c"]


def test_timing_feature_legacy_contract_is_preserved() -> None:
    assert _timing_features({"selected_features": ["b", "a"]}) == ["b", "a"]
