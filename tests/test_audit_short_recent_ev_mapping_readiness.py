from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "audit_short_recent_ev_mapping_readiness.py"
SPEC = importlib.util.spec_from_file_location("short_recent_mapping_readiness", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_daily_reference_audit_requires_strictly_prior_label_end_and_no_identity_overlap() -> None:
    snapshot = pd.Timestamp("2025-03-02T00:00:00Z")
    frame = pd.DataFrame({
        "candidate_id": ["old", "same_day"], "candidate_month": ["2025-02", "2025-03"], "side_name": ["short", "short"],
        "__ts__": [pd.Timestamp("2025-02-28T10:00:00Z"), snapshot], "execution_label_end_utc": [snapshot - pd.Timedelta(seconds=1), snapshot + pd.Timedelta(hours=12)],
        "execution_net_ev_12h": [.01, -.01], "base_oof_score": [.3, .4], "fold_id": ["feb", "mar"],
    })
    result = MODULE.daily_reference_audit(frame)
    row = result.iloc[0]
    assert row.reference_rows == 1 and row.reference_short_rows == 1
    assert row.strict_label_end_before_snapshot
    assert row.evaluation_reference_identity_overlap == 0


def test_grid_feasibility_does_not_relax_legal_snapshot_gate() -> None:
    audit = pd.DataFrame({"reference_rows": [10_000], "reference_short_rows": [10_000], "strict_label_end_before_snapshot": [False], "evaluation_reference_identity_overlap": [0]})
    result = MODULE.grid_feasibility(audit)
    assert not result.snapshot_mapping_ready_proxy.any()


def test_short_score_contract_is_not_inferred_from_base_score() -> None:
    frame = pd.DataFrame({"candidate_month": ["2025-03"], "side_name": ["short"], "base_oof_score": [.5]})
    result = MODULE.score_contract_readiness(frame)
    assert result.loc[result.score_contract.eq("frozen_short_conversion_oof_score"), "status"].iloc[0] == "NOT_MATERIALIZED_BLOCKS_MAPPING_EVALUATION"
