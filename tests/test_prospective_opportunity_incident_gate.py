from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "enforce_prospective_opportunity_incident_gate.py"
)
SPEC = importlib.util.spec_from_file_location("incident_gate", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _packet(identifier: str, start: str, content_hash: str = "a") -> dict:
    stamp = pd.Timestamp(start)
    return {
        "lineage": "current",
        "opportunity_incident_id": identifier,
        "incident_start_utc": stamp,
        "packet_content_sha256": content_hash,
    }


def test_current_packet_role_is_not_silently_pooled() -> None:
    packet = pd.Series(
        {
            "incident_start_utc": pd.Timestamp("2026-07-01", tz="UTC"),
            "incident_end_utc": pd.Timestamp("2026-07-01 12:00", tz="UTC"),
        }
    )
    selected = pd.DataFrame(
        {
            "__ts__": pd.to_datetime(
                ["2026-07-01 01:00", "2026-07-01 02:00"], utc=True
            ),
            "failure_first_history_role": [
                "outer_oof",
                "retired_resolved_forward_oos",
            ],
        }
    )
    contract = {
        "retrospective_roles": ["outer_oof"],
        "forward_research_roles": ["retired_resolved_forward_oos"],
    }
    role, rows, source = MODULE.classify_current_packet_role(
        packet, selected, contract
    )
    assert role == "MIXED_OR_UNKNOWN_PROVENANCE"
    assert rows == 2
    assert source == "outer_oof|retired_resolved_forward_oos"


def test_append_only_rejects_frozen_packet_rewrite() -> None:
    prior = pd.DataFrame([_packet("one", "2026-07-01", "original")])
    current = pd.DataFrame([_packet("one", "2026-07-01", "changed")])
    with pytest.raises(ValueError, match="rewrote frozen packet"):
        MODULE.assert_append_only(prior, current)


def test_append_only_rejects_retroactive_insert() -> None:
    prior = pd.DataFrame([_packet("two", "2026-07-10", "original")])
    current = pd.DataFrame(
        [
            _packet("one", "2026-07-05", "new"),
            _packet("two", "2026-07-10", "original"),
        ]
    )
    with pytest.raises(ValueError, match="behind lineage watermark"):
        MODULE.assert_append_only(prior, current)


def test_gate_requires_sixty_compatible_incidents() -> None:
    ledger = pd.DataFrame(
        {
            "detector_support_eligible": [True] * 10,
            "taxonomy_support_eligible": [True] * 7 + [False] * 3,
            "prospective_forward_support": [False] * 10,
            "incumbent_portfolio_support_eligible": [False] * 10,
        }
    )
    config = {
        "gate": {
            "minimum_independent_incidents": 60,
            "target_independent_incidents": 100,
        }
    }
    report = MODULE.gate_report(ledger, config)
    assert report["candidate_current_model_incidents"] == 10
    assert report["taxonomy_usable_current_model_incidents"] == 7
    assert report["promotion_grade_incumbent_portfolio_incidents"] == 0
    assert report["remaining_to_minimum_current_model"] == 50
    assert not report["supervised_failure_detector_training_authorized"]
    assert not report["incumbent_portfolio_promotion_authorized"]
