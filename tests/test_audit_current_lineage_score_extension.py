from __future__ import annotations

import pandas as pd

from scripts.audit_current_lineage_score_extension import (
    MAPPED_OOF,
    MAPPED_SCORE,
    attach_transition_scores,
    build_current_score_panel,
    transition_coverage,
)


def _identity() -> dict[str, object]:
    return {
        "__ts__": pd.Timestamp("2026-06-02T17:00:00Z"),
        "__symbol__": "BTC/USD:USD",
        "side_name": "long",
        "candidate_id": "candidate-a",
    }


def test_current_panel_excludes_forward_and_preserves_each_score_layer() -> None:
    identity = _identity()
    mapped = pd.DataFrame(
        [
            {
                **identity,
                "execution_decision_utc": identity["__ts__"] + pd.Timedelta(hours=1),
                "execution_label_end_utc": identity["__ts__"] + pd.Timedelta(hours=13),
                "execution_gross_ev_12h": 0.02,
                "execution_net_ev_12h": 0.01,
                "evaluation_origin": "outer_oof",
                "catboost__residual__without_hpo__all_features": 0.2,
                MAPPED_SCORE: 0.3,
                MAPPED_OOF: True,
            },
            {
                **{**identity, "candidate_id": "forward"},
                "execution_decision_utc": identity["__ts__"] + pd.Timedelta(hours=1),
                "execution_label_end_utc": identity["__ts__"] + pd.Timedelta(hours=13),
                "execution_gross_ev_12h": 0.02,
                "execution_net_ev_12h": 0.01,
                "evaluation_origin": "forward",
                "catboost__residual__without_hpo__all_features": 0.2,
                MAPPED_SCORE: 0.3,
                MAPPED_OOF: False,
            },
        ]
    )
    base = pd.DataFrame([{**identity, "prediction": 0.5, "outer_fold": "b", "prediction_source": "oof"}])
    residual = pd.DataFrame(
        [{**identity, "prediction": 0.6, "residual_oof_fold": "r", "base_expected_ev": 0.01, "residual_delta_ev": 0.02, "residual_expected_ev": 0.03, "residual_is_oof": True}]
    )
    alpha = pd.DataFrame(
        [{**identity, "existing_alpha_ev": 0.03, "base_alpha_ev": 0.01, "alpha_prediction_uncertainty": 0.1, "alpha_leaf_support": 5.0, "oof_fold": "a"}]
    )
    panel = build_current_score_panel(mapped, base, residual, alpha)
    assert panel["candidate_id"].tolist() == ["candidate-a"]
    assert panel.loc[0, "base_31_8_oof_score"] == 0.5
    assert panel.loc[0, "residual_expected_ev"] == 0.03


def test_transition_overlap_counts_events_not_candidate_rows() -> None:
    base = pd.DataFrame(
        [
            {**_identity(), "candidate_id": "a"},
            {**_identity(), "candidate_id": "b", "side_name": "short"},
        ]
    )
    active = pd.DataFrame(
        [{"source_utc": _identity()["__ts__"], "target__event_id": "event", "target__transition_active": 1, "prediction": 0.8}]
    )
    joined = attach_transition_scores(base, active)
    coverage = transition_coverage(joined)
    assert coverage["events_overlapped"] == 1
    assert coverage["events_with_active_hours"] == 1
    assert coverage["event_rows"][0]["candidate_rows"] == 2
