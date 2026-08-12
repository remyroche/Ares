from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_canonical_walkforward.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_current_walkforward", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["one", "two"],
            "__decision_ts__": pd.to_datetime(
                ["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"],
            ),
            "final_score": [0.7, 0.8],
        }
    )


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["one", "two"],
            "policy_path_valid": [True, True],
            "policy_gross_bps": [150.0, 200.0],
            "policy_net_bps": [50.0, 100.0],
            "policy_label_available_ts": pd.to_datetime(
                ["2025-01-01T12:00:00Z", "2025-01-01T13:00:00Z"],
            ),
        }
    )


def test_outcomes_are_joined_only_after_stable_prediction_identity() -> None:
    predictions = _predictions()
    joined, columns = MODULE._attach_outcomes_after_scoring(predictions, _outcomes())

    assert joined["candidate_id"].tolist() == predictions["candidate_id"].tolist()
    assert columns == [
        "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_label_available_ts",
    ]
    assert joined["policy_net_bps"].tolist() == [50.0, 100.0]


def test_outcome_join_rejects_duplicate_identity() -> None:
    outcomes = pd.concat([_outcomes(), _outcomes().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique candidate_id"):
        MODULE._attach_outcomes_after_scoring(_predictions(), outcomes)


def test_run_lock_rejects_a_second_active_writer_and_releases_cleanly(tmp_path: Path) -> None:
    first = MODULE._acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already actively owned"):
            MODULE._acquire_run_lock(tmp_path)
    finally:
        first.close()

    second = MODULE._acquire_run_lock(tmp_path)
    second.close()


def test_walkforward_uses_only_the_canonical_physical_reference_window() -> None:
    """The wrapper must not silently load a wider legacy score population."""

    source = PATH.read_text()
    assert MODULE.REFERENCE_DAYS == 28
    assert "Timedelta(days=42)" not in source
    assert '"reference_window_days": REFERENCE_DAYS' in source
