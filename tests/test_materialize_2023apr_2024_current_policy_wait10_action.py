from __future__ import annotations

from scripts.materialize_2023apr_2024_current_policy_wait10_action import (
    MODEL_FEATURES,
    PROVENANCE_FIELDS,
    REGIME_FEATURES,
    SCORE_FEATURES,
    TRANSITION_FEATURES,
    raw_policy_archetype,
)
import pandas as pd


def test_historical_feature_groups_are_disjoint_and_complete() -> None:
    groups = [set(SCORE_FEATURES), set(REGIME_FEATURES), set(TRANSITION_FEATURES)]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert set(MODEL_FEATURES) == set.union(*groups)


def test_unstable_regime_components_and_provenance_are_not_model_inputs() -> None:
    forbidden = {
        "regime_state_p__0",
        "regime_state_p__1",
        "regime_state_p__2",
        "regime_state_id",
        "transition_state_id",
        *PROVENANCE_FIELDS,
    }
    assert not forbidden.intersection(MODEL_FEATURES)


def test_historical_features_contain_transition_state_probabilities() -> None:
    required = {
        "transition_active_probability",
        "transition_state_p__approach",
        "transition_state_p__transition",
        "transition_state_p__settled_destination",
        "transition_state_entropy",
        "transition_state_ood_score",
    }
    assert required.issubset(MODEL_FEATURES)


def test_persisted_archetype_is_denormalized_exactly_once() -> None:
    values = pd.Series(
        [
            "policy_archetype_long__long_breakout_diagnostic_candidate",
            "historical_side_parent_fallback",
        ]
    )
    assert raw_policy_archetype(values).tolist() == [
        "long__long_breakout_diagnostic_candidate",
        "historical_side_parent_fallback",
    ]
