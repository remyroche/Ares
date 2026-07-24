from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.diagnostics.execution_factorial import (
    assert_fixed_stage_keys,
    exit_transition_matrix,
    interaction_delta,
    mfe_capture_ratio,
    paired_variant_summary,
)


def _frame(variant: str, gross: list[float], reasons: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-05-01", "2026-05-02"], utc=True),
        "symbol": ["A", "B"],
        "side_name": ["long", "short"],
        "variant": variant,
        "gross_return": gross,
        "net_return": np.asarray(gross) - 0.003,
        "exit_type": reasons,
        "mfe": [0.02, 0.01],
        "mae": [0.01, 0.02],
        "holding_minutes": [60, 120],
    })


def test_fixed_stage_keys_accept_equal_populations() -> None:
    ledger = pd.concat([
        _frame("reference", [0.01, -0.01], ["trailing", "full_stop"]),
        _frame("candidate", [0.02, -0.02], ["trailing", "timeout"]),
    ], ignore_index=True)
    assert_fixed_stage_keys(
        ledger, key_columns=["timestamp", "symbol", "side_name"]
    )


def test_fixed_stage_keys_reject_attrition() -> None:
    ledger = pd.concat([
        _frame("reference", [0.01, -0.01], ["trailing", "full_stop"]),
        _frame("candidate", [0.02, -0.02], ["trailing", "timeout"]).iloc[:1],
    ], ignore_index=True)
    with pytest.raises(ValueError, match="fixed row population"):
        assert_fixed_stage_keys(
            ledger, key_columns=["timestamp", "symbol", "side_name"]
        )


def test_paired_summary_and_transition_preserve_exit_taxonomy() -> None:
    reference = _frame("reference", [0.01, -0.01], ["trailing", "full_stop"])
    candidate = _frame("candidate", [0.02, -0.02], ["capital_exit", "timeout"])
    summary = paired_variant_summary(candidate, reference)
    assert summary["ev_delta"] == pytest.approx(0.0)
    assert summary["exit_type_changes"] == 2
    assert summary["return_change_gt_50bps"] == 2
    transitions = exit_transition_matrix(candidate, reference, variant="candidate")
    assert transitions.loc[
        transitions["reference_exit_type"].eq("trailing")
        & transitions["candidate_exit_type"].eq("capital_exit"),
        "count",
    ].iloc[0] == 1


def test_mfe_capture_is_signed_and_interaction_formula_is_exact() -> None:
    ratio = mfe_capture_ratio(np.asarray([0.01, -0.01]), np.asarray([0.02, 0.01]))
    assert ratio.tolist() == pytest.approx([0.5, -1.0])
    assert interaction_delta(0.7, 0.9, 0.8, 1.0) == pytest.approx(0.0)
