from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_strict_r3_f72_early_feature_parity_v1.py"
SPEC = importlib.util.spec_from_file_location("f72_feature_parity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["A|long|2025-04-01T00:00:00Z"],
        "__decision_ts__": pd.to_datetime(["2025-04-01T01:00:00Z"], utc=True),
        "side_name": ["long"],
        "f": [value],
    })


def test_parity_audit_requires_identity_and_exact_tolerance_alignment() -> None:
    identity, field, checks = MODULE._audit(_frame(1.0), _frame(1.0), ("f",))
    assert identity.loc[0, "both_rows"] == 1
    assert field.loc[0, "equal_fraction"] == 1.0
    assert checks == {
        "identity_exact": True,
        "all_finite_masks_match": True,
        "all_values_within_tolerance": True,
    }


def test_parity_audit_rejects_material_value_drift() -> None:
    _, field, checks = MODULE._audit(_frame(1.0), _frame(1.1), ("f",))
    assert field.loc[0, "equal_fraction"] == 0.0
    assert not checks["all_values_within_tolerance"]


def test_parity_audit_detects_finite_mask_drift() -> None:
    _, field, checks = MODULE._audit(_frame(1.0), _frame(np.nan), ("f",))
    assert field.loc[0, "finite_mismatch_rows"] == 1
    assert not checks["all_finite_masks_match"]
