from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "replay_strict_r3_forward_portfolio.py"
SPEC = importlib.util.spec_from_file_location("strict_r3_forward_portfolio", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_current_schema_defaults_to_selected_pre2025_policy() -> None:
    values, engine, period = MODULE._resolve_policy("current-v5", None)
    assert values["sl_mult"] == pytest.approx(4.1520006)
    assert values["trailing_activation_mult"] == pytest.approx(2.3262249)
    assert values["fixed_trailing_gap_mult"] == pytest.approx(0.1023720)
    assert engine == "simple_policy_optimiser_pre2025_winner"
    assert "pre-2025" in period


def test_current_schema_rejects_a_policy_target_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "wrong_policy.json"
    path.write_text(
        '{"winner":{"sl_mult":3,"trailing_activation_mult":0.5,'
        '"fixed_trailing_gap_mult":0.25}}'
    )
    with pytest.raises(ValueError, match="retrain for another policy"):
        MODULE._resolve_policy("current-v5", path)


def test_auction_does_not_future_qualify_on_policy_path_availability() -> None:
    timestamp = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": ["missing", "resolved"],
            "__decision_ts__": [timestamp, timestamp],
            "__symbol__": ["AAA/USDT", "BBB/USDT"],
            "side_name": ["long", "long"],
            "causal_21d_side_admitted_ge_50bps": [True, True],
            "causal_21d_side_expected_net_bps": [150.0, 100.0],
            "policy_path_valid": [False, True],
            "policy_net_bps": [np.nan, 75.0],
            "policy_gross_bps": [np.nan, 175.0],
            "policy_exit_bar_15m": [np.nan, 3.0],
            "policy_entry_price": [np.nan, 10.0],
            "policy_exit_price": [np.nan, 10.175],
            "policy_exit_reason": [None, "TRAILING"],
            "portfolio_size_multiplier": [0.4, 1.6],
        }
    )
    output = MODULE._auction_candidates(frame).set_index("candidate_id")
    assert set(output.index) == {"missing", "resolved"}
    assert bool(output.loc["missing", "policy_outcome_proxy_for_constraints"])
    assert not bool(output.loc["missing", "policy_outcome_available"])
    assert output.loc["missing", "holding_bars"] == 48
    assert output.loc["missing", "net_return"] == 0.0
    assert output.loc["missing", "mapped_expected_net_bps"] == 150.0
    assert output.loc["missing", "normalized_rank_score"] == 1.0
    assert output.loc["missing", "portfolio_size_multiplier"] == pytest.approx(0.4)
    assert output.loc["resolved", "portfolio_size_multiplier"] == pytest.approx(1.6)


def test_unavailable_nonadmitted_row_never_enters_auction() -> None:
    timestamp = pd.Timestamp("2026-01-01 01:00", tz="UTC")
    frame = pd.DataFrame(
        {
            "candidate_id": ["missing"],
            "__decision_ts__": [timestamp],
            "__symbol__": ["AAA/USDT"],
            "side_name": ["long"],
            "causal_21d_side_admitted_ge_50bps": [False],
            "causal_21d_side_expected_net_bps": [25.0],
            "policy_path_valid": [False],
            "policy_net_bps": [np.nan],
            "policy_gross_bps": [np.nan],
            "policy_exit_bar_15m": [np.nan],
            "policy_entry_price": [np.nan],
            "policy_exit_price": [np.nan],
            "policy_exit_reason": [None],
        }
    )
    output = MODULE._auction_candidates(frame)
    assert output.empty
