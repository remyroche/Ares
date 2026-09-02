from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "materialize_strict_r3_router_single_base_target_labels_v1.py"
SPEC = importlib.util.spec_from_file_location("router_base_labels", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _part() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": pd.to_datetime(["2025-04-01T00:00:00Z", "2025-04-01T01:00:00Z"]),
        "side_name": ["long", "long"],
        "policy_net_bps": [100.0, float("nan")],
        "decision_atr_bps": [50.0, float("nan")],
        "policy_ordinal_valid": [True, False],
        "raw_magnitude_valid": [True, False],
        "normalised_magnitude_valid": [True, False],
    })


def test_resume_coverage_row_reuses_persisted_part_without_relaxing_validity() -> None:
    row = MODULE._coverage_row(_part(), pd.Timestamp("2025-04-01", tz="UTC"))
    assert row["candidate_rows"] == 2
    assert row["policy_valid_rows"] == 1
    assert row["raw_magnitude_valid_rows"] == 1
    assert row["normalised_valid_rows"] == 1


def test_resume_contract_is_explicit_and_refuses_completed_receipts() -> None:
    source = SCRIPT.read_text()
    assert 'parser.add_argument(\n        "--resume"' in source
    assert "refusing to resume a completed immutable target-label receipt" in source
    assert "existing immutable label values" in source
