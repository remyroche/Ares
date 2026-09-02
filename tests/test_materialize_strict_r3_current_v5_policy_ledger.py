from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.materialize_strict_r3_current_v5_policy_ledger import (
    POLICY_COLUMNS,
    materialize_policy_contract,
)


def _policy_row(candidate_id: str, *, valid: bool, net: float | None) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "policy_path_valid": valid,
        "policy_gross_bps": None if net is None else net + 100.0,
        "policy_net_bps": net,
        "policy_exit_bar_15m": None if net is None else 8.0,
        "policy_entry_price": None if net is None else 1.0,
        "policy_exit_price": None if net is None else 1.02,
        "policy_exit_reason": "timeout" if valid else "unavailable",
        "policy_label_available_ts": "2026-01-02T00:00:00Z" if valid else None,
        "policy_outcome_source": "coarse_15m" if valid else "unavailable",
        "policy_cost_bps": None if net is None else 100.0,
    }


def test_authoritative_contract_replaces_stale_validity_without_touching_scores() -> None:
    scores = pd.DataFrame({
        "candidate_id": ["a", "b", "missing"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"] * 3),
        "final_score": [0.8, 0.5, 0.2],
        "policy_path_valid": [False, False, False],
        "policy_net_bps": [12.0, 99.0, -5.0],
    }, index=[11, 23, 41])
    policy = pd.DataFrame([_policy_row("a", valid=True, net=42.0), _policy_row("b", valid=False, net=None)])
    output, audit = materialize_policy_contract(scores, policy)
    assert output["final_score"].tolist() == [0.8, 0.5, 0.2]
    assert output["policy_path_valid"].tolist() == [True, False, False]
    assert output.loc[0, "policy_net_bps"] == pytest.approx(42.0)
    assert output.loc[1:, "policy_net_bps"].isna().all()
    assert output.loc[1:, "policy_label_available_ts"].isna().all()
    assert audit == {
        "score_rows": 3,
        "canonical_policy_rows": 2,
        "candidate_ids_found_in_policy": 2,
        "valid_policy_rows": 1,
        "invalid_or_missing_policy_rows": 2,
    }
    assert set(POLICY_COLUMNS).issubset(output.columns)


def test_rejects_a_valid_outcome_available_at_decision() -> None:
    scores = pd.DataFrame({
        "candidate_id": ["a"],
        "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "final_score": [0.8],
    })
    policy = pd.DataFrame([_policy_row("a", valid=True, net=42.0)])
    policy.loc[0, "policy_label_available_ts"] = "2026-01-01T00:00:00Z"
    with pytest.raises(AssertionError, match="available at or before"):
        materialize_policy_contract(scores, policy)
