"""Regression tests for the residual-event Bayesian correction contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_strict_r3_beta_residual_event_fold.py"
SPEC = importlib.util.spec_from_file_location("beta_residual_event", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_target_anchor_never_enters_residual_event_feature_contract() -> None:
    rows = 32
    frame = pd.DataFrame({
        "policy_net_bps": np.linspace(-250.0, 250.0, rows),
        "raw_expected_bps": np.linspace(-50.0, 50.0, rows),
    })
    for index in range(12):
        frame[f"feature_{index}"] = np.arange(rows, dtype=float) + index
    selected = MODULE._eligible_fields(frame)
    assert "raw_expected_bps" not in selected
    assert len(selected) == 12


def test_timestamp_top30_is_local_to_each_decision_timestamp() -> None:
    frame = pd.DataFrame({
        "__decision_ts__": [pd.Timestamp("2025-01-01", tz="UTC")] * 4
        + [pd.Timestamp("2025-01-01 01:00", tz="UTC")] * 4,
        "candidate_id": [str(value) for value in range(8)],
        "final_score": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })
    selected = MODULE._timestamp_top30(frame)
    assert selected.tolist() == [True, True, False, False, True, True, False, False]
