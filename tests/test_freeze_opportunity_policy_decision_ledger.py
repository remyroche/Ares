from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "freeze_opportunity_policy_decision_ledger.py"
)
SPEC = importlib.util.spec_from_file_location("policy_ledger", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.to_datetime(
        ["2026-07-01 00:00", "2026-07-01 01:00"], utc=True
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "timestamp": signal + pd.Timedelta(hours=1),
            "source_score_timestamp": signal,
            "symbol": ["A", "B"],
            "side": ["long", "short"],
            "normalized_rank_score": [0.9, 0.8],
            "oof_fold": [1, 1],
            "net_return": [0.01, -0.02],
            "gross_return": [0.02, -0.01],
            "exit_timestamp": signal + pd.Timedelta(hours=13),
            "simple_policy_exit_reason": ["timeout", "full_stop"],
        }
    )
    decisions = pd.DataFrame(
        {
            "candidate_index": [1, 0],
            "accepted": [False, True],
            "rejection_reason": ["capacity", "accepted"],
            "position_size": [0.0, 1.0],
            "open_positions_before": [1, 0],
            "open_positions_after": [1, 1],
            "wallet_before": [1.0, 1.0],
            "wallet_after": [1.0, 1.0],
            "position_exit_timestamp": [
                candidates.loc[1, "exit_timestamp"],
                candidates.loc[0, "exit_timestamp"],
            ],
            "position_net_return": [-0.02, 0.01],
            "position_gross_return": [-0.01, 0.02],
            "position_exit_reason": ["full_stop", "timeout"],
        }
    )
    return candidates, decisions


def test_decision_ledger_is_identity_stable_and_index_aligned() -> None:
    candidates, decisions = _frames()
    ledger = MODULE.build_decision_ledger(candidates, decisions, "policy")
    assert ledger.candidate_id.tolist() == ["a", "b"]
    assert ledger.portfolio_accepted.tolist() == [True, False]
    assert ledger.decision_id.nunique() == 2
    assert ledger.decision_utc.eq(
        ledger.signal_utc + pd.Timedelta(hours=1)
    ).all()


def test_decision_ledger_rejects_incomplete_portfolio_coverage() -> None:
    candidates, decisions = _frames()
    with pytest.raises(ValueError, match="cover every candidate"):
        MODULE.build_decision_ledger(
            candidates, decisions.iloc[:1], "policy"
        )


def test_policy_hash_is_key_order_invariant() -> None:
    assert MODULE.canonical_hash({"a": 1, "b": 2}) == MODULE.canonical_hash(
        {"b": 2, "a": 1}
    )
