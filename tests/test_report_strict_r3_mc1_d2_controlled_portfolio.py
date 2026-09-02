from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "strict_r3_controlled_report",
    ROOT / "scripts" / "report_strict_r3_mc1_d2_controlled_portfolio.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _prediction() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["valid", "invalid"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
        "__symbol__": ["A/USD:USD", "B/USD:USD"],
        "final_score": [.9, .8],
        "mc1_expected_bps": [100.0, 100.0],
    })


def _policy() -> pd.DataFrame:
    return pd.DataFrame({
        "candidate_id": ["valid", "invalid"],
        "policy_path_valid": [True, False],
        "policy_net_bps": [50.0, float("nan")],
        "policy_gross_bps": [150.0, float("nan")],
        "policy_exit_bar_15m": [4, float("nan")],
        "policy_entry_price": [1.0, float("nan")],
        "policy_exit_price": [1.01, float("nan")],
        "policy_exit_reason": ["trailing", "invalid_path"],
    })


def test_exclude_mode_does_not_create_capacity_reserving_pseudo_trade() -> None:
    result = MODULE._candidate_table(
        _prediction(), _policy(), 50.0, invalid_outcome_mode="exclude",
    )
    assert result.candidate_id.tolist() == ["valid"]
    assert result.policy_outcome_available.tolist() == [True]


def test_reserve_mode_retains_legacy_invalid_row_for_explicit_legacy_replay() -> None:
    result = MODULE._candidate_table(
        _prediction(), _policy(), 50.0, invalid_outcome_mode="reserve",
    )
    assert result.candidate_id.tolist() == ["valid", "invalid"]
    assert result.policy_outcome_available.tolist() == [True, False]
