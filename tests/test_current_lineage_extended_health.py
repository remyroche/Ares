from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

STAGING = Path(__file__).resolve().parents[1] / "scripts"
if str(STAGING) not in sys.path:
    sys.path.insert(0, str(STAGING))

from materialize_current_lineage_extended_health import (  # noqa: E402
    assemble_extended_rich_handoff,
)


def _rich(candidate: str, stamp: str) -> dict[str, object]:
    source = pd.Timestamp(stamp, tz="UTC")
    return {
        "candidate_id": candidate,
        "__ts__": source,
        "execution_decision_utc": source + pd.Timedelta(hours=1),
        "base_oof_score": 0.1,
        "base_margin_to_cutoff_z": 0.2,
        "catboost_entropy": 0.3,
        "alpha_prediction_uncertainty": 0.4,
    }


def test_extended_handoff_keeps_oof_and_forward_identity_separate() -> None:
    history = pd.DataFrame(
        {
            "candidate_id": ["oof", "forward"],
            "failure_first_history_role": [
                "outer_oof",
                "retired_resolved_forward_oos",
            ],
            "failure_first_score_is_strict_model_oos": [True, True],
            "causal_recent_side_isotonic_ev__is_oof": [True, False],
            "causal_recent_side_isotonic_ev__is_forward_oos": [False, True],
        }
    )
    result = assemble_extended_rich_handoff(
        history,
        pd.DataFrame([_rich("oof", "2026-01-01")]),
        pd.DataFrame([_rich("forward", "2026-01-02")]),
    )
    assert set(result["candidate_id"]) == {"oof", "forward"}
    assert len(result) == 2


def test_extended_handoff_rejects_role_provenance_disagreement() -> None:
    history = pd.DataFrame(
        {
            "candidate_id": ["forward"],
            "failure_first_history_role": ["outer_oof"],
            "failure_first_score_is_strict_model_oos": [True],
            "causal_recent_side_isotonic_ev__is_oof": [False],
            "causal_recent_side_isotonic_ev__is_forward_oos": [True],
        }
    )
    try:
        assemble_extended_rich_handoff(
            history,
            pd.DataFrame([_rich("unused", "2026-01-01")]),
            pd.DataFrame([_rich("forward", "2026-01-02")]),
        )
    except ValueError as error:
        assert "role disagrees" in str(error)
    else:
        raise AssertionError("role/provenance disagreement must fail")
