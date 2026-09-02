"""Focused invariants for the strict-OOF SHAP-derived Base pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_strict_r3_p8u_fulluniverse_economic_mda_v1 as mda  # noqa: E402
import run_strict_r3_p8u_shapderived_feature_pipeline_v1 as derived  # noqa: E402


def test_conditional_mi_detects_dependency() -> None:
    x = np.arange(1_000, dtype=float)
    y = x.copy()
    condition = np.tile(np.arange(10), 100)
    assert derived._conditional_mi(x, y, condition) > .01


def test_timestamp_local_permutation_does_not_cross_queries() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [str(i) for i in range(12)],
            "__decision_ts__": pd.to_datetime(["2025-01-01"] * 6 + ["2025-01-02"] * 6, utc=True),
            "policy_net_bps": np.arange(12, dtype=float),
        }
    )
    permutation = mda._within_timestamp_permutation(frame, 1729)
    for _, part in frame.assign(__source__=permutation).groupby("__decision_ts__", sort=False):
        assert set(part.__source__).issubset(set(part.index))


def test_boundary_delta_is_zero_for_identical_scores() -> None:
    frame = pd.DataFrame(
        {
            "candidate_id": [str(i) for i in range(10)],
            "__decision_ts__": pd.to_datetime(["2025-01-01"] * 10, utc=True),
            "policy_net_bps": np.arange(10, dtype=float),
        }
    )
    score = np.arange(len(frame), dtype=float)
    assert mda._boundary_delta(frame, score, score) == 0.0


def test_structural_target_field_is_not_mistaken_for_an_outcome_field() -> None:
    # A causal field can contain the English word "target".  The guard must
    # reject actual outcome namespaces, not destroy valid structural inputs.
    assert not "reversion_target_distance".startswith(derived.PROHIBITED_SOURCE_PREFIXES)
    assert "policy_net_bps".startswith(derived.PROHIBITED_SOURCE_PREFIXES)
