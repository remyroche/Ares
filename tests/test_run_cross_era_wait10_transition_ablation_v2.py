from __future__ import annotations

from scripts.run_cross_era_wait10_transition_ablation_v2 import (
    EXPANDED_TRANSITIONS,
    FEATURE_SETS,
    SUBTYPE_FEATURES,
)


def test_expanded_transition_contract_is_nested_and_preregistered() -> None:
    assert set(SUBTYPE_FEATURES).issubset(EXPANDED_TRANSITIONS)
    assert set(FEATURE_SETS["transition_common_v1"]).issubset(
        FEATURE_SETS["transition_expanded"]
    )
    assert set(FEATURE_SETS["transition_subtype_only"]) == set(SUBTYPE_FEATURES)


def test_subtype_features_cover_observed_state_differences() -> None:
    required = {
        "btc_resilience_alt_weakness_gap",
        "downside_breadth_intensity",
        "breadth_dispersion",
        "compression_quality_consistency",
        "short_default_damage_max_5d",
    }
    assert required.issubset(SUBTYPE_FEATURES)
