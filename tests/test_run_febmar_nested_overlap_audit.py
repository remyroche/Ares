from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_febmar_nested_overlap_audit.py"
SPEC = importlib.util.spec_from_file_location("febmar_nested_overlap_audit", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _context(rows: int = 320) -> pd.DataFrame:
    value = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "candidate_id": [f"id-{idx}" for idx in range(rows)],
        "side_name": np.where(value % 2, "long", "short"),
        "__symbol__": np.where(value % 3, "BTC_USD:USD", "ETH_USD:USD"),
        "score_ventile": np.where(value % 2, "18", "19"),
        "candidate_group_size_bin": np.where(value % 4, "q1", "q2"),
        "transition_state": np.where(value % 2, "q0|q1|q2", "q1|q1|q0"),
        "liq_volume_confirmation": value / rows,
        "vol_range": (value % 17) / 17,
        "volatility": (value % 13) / 13,
        "trend": np.sin(value / 20),
        "trend_level": np.cos(value / 20),
        "transition_range": np.sin(value / 13),
        "transition_volatility": np.cos(value / 13),
        "transition_trend": np.sin(value / 11),
        "transition_jump": np.cos(value / 11),
    })


def test_nested_sets_are_predeclared_and_strictly_nested() -> None:
    core, lvt, transition = MODULE._nested_configs()
    assert core[1] == ()
    assert set(core[2]).issubset(lvt[2])
    assert set(lvt[1]).issubset(transition[1])
    assert "transition_state" in transition[2]


def test_outcome_blind_support_reports_coverage_ess_and_balance() -> None:
    source, target = _context(), _context()
    supported_source, supported_target, odds, overlap_source, overlap_target, summary, balance = MODULE.fit_support(
        source, target, continuous=("liq_volume_confirmation", "vol_range"),
        categorical=("side_name", "__symbol__", "score_ventile", "candidate_group_size_bin"),
    )
    assert summary["common_support_pass"]
    assert summary["target_support_coverage"] == 1.0
    assert summary["weight_ess"] >= MODULE.MIN_EFFECTIVE_ROWS
    assert summary["max_abs_smd_after"] <= MODULE.MAX_WEIGHTED_SMD
    assert len(supported_source) == len(odds) == len(overlap_source)
    assert len(supported_target) == len(overlap_target)
    assert not balance.empty


def test_excluded_cohorts_marks_observed_support_removal() -> None:
    frame = _context(4)
    cohorts = MODULE._cohorts(frame, np.array([True, False, True, False]), role="source", covariate_set="core")
    assert cohorts.loc[cohorts.excluded_by_common_support, "rows"].sum() == 2
