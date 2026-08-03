from __future__ import annotations

import importlib.util
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_july20_23_bridge_transition_readiness.py"
SPEC = importlib.util.spec_from_file_location("july_bridge_transition_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_raw_score_bridge_fails_closed_without_mapping_or_geometry() -> None:
    readiness, bounds = MODULE.audit(MODULE.DEFAULT_BRIDGE)
    assert bounds["rows"] == 5760
    assert bounds["candidate_ids"] == 5760
    assert readiness["available"].eq(False).all()
    assert readiness["missing_columns_or_state"].str.contains("mapped_execution_ev").any()
    assert readiness["minimal_request"].str.len().gt(30).all()
