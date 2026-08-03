from __future__ import annotations

import pandas as pd

from scripts.materialize_pooled_transition_classification_readiness import audit_readiness


def test_readiness_fails_closed_on_missing_2023_and_global_book_contract() -> None:
    identities = {"__ts__": [pd.Timestamp("2022-12-01", tz="UTC"), pd.Timestamp("2023-02-01", tz="UTC")], "__symbol__": ["A", "A"], "side_name": ["long", "long"], "candidate_id": ["a", "b"]}
    context = pd.DataFrame({**identities, "__decision_ts__": [pd.Timestamp("2022-12-01 01:00", tz="UTC"), pd.Timestamp("2023-02-01 01:00", tz="UTC")], "transition_context_available": [True, False], "source_family": ["old", "unavailable"], "legacy_feature": [1.0, float("nan")]})
    labels = pd.DataFrame({**identities, "__decision_ts__": context["__decision_ts__"], "execution_net_ev_12h": [0.0, 0.0]})
    coverage, readiness, missing = audit_readiness(context, labels, ["context__state_mean__x"])
    assert coverage["transition_context_rows"].sum() == 1
    assert not any(item["ready"] for item in readiness)
    assert {item["id"] for item in missing} == {"historical_transition_context_through_2023", "common_decision_time_feature_contract", "causal_global_book_selection", "exact_before_after_transition_targets"}


def test_readiness_marks_completed_context_without_overclaiming_other_contracts() -> None:
    identities = {
        "__ts__": [pd.Timestamp("2023-02-01", tz="UTC")],
        "__symbol__": ["A"],
        "side_name": ["long"],
        "candidate_id": ["a"],
    }
    decision = [pd.Timestamp("2023-02-01 01:00", tz="UTC")]
    context = pd.DataFrame(
        {
            **identities,
            "__decision_ts__": decision,
            "transition_context_available": [True],
            "source_family": ["old"],
            "shared_feature": [1.0],
        }
    )
    labels = pd.DataFrame(
        {
            **identities,
            "__decision_ts__": decision,
            "execution_net_ev_12h": [0.0],
        }
    )
    _, readiness, missing = audit_readiness(
        context, labels, ["shared_feature", "current_only_feature"]
    )
    status = {item["requirement"]: item["ready"] for item in readiness}
    assert status["historical_transition_context_through_2023"]
    assert status["common_decision_time_feature_contract"]
    assert not status["causal_global_book_selection"]
    assert not status["exact_before_after_transition_targets"]
    assert {item["id"] for item in missing} == {
        "causal_global_book_selection",
        "exact_before_after_transition_targets",
    }
