from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_source_utility_risk_gate_diagnostic import (  # noqa: E402
    _assert_gate_columns_causal,
    _gate_mask,
    _gate_specs_by_name,
)


def test_recovery_support_gates_are_registered_and_causal() -> None:
    gates = _gate_specs_by_name(
        [
            "high_recovery_excess_q50",
            "high_recovery_convexity_q50",
            "recovery_excess_convexity_q40",
            "recovery_support_q40_barrier_q50",
        ]
    )

    _assert_gate_columns_causal(gates)
    assert [gate.name for gate in gates] == [
        "high_recovery_excess_q50",
        "high_recovery_convexity_q50",
        "recovery_excess_convexity_q40",
        "recovery_support_q40_barrier_q50",
    ]


def test_recovery_support_gate_uses_train_quantiles_only() -> None:
    frame = pd.DataFrame(
        {
            "excess_12h": [1.0, 2.0, 3.0, 4.0],
            "convexity_t": [1.0, 2.0, 3.0, 4.0],
            "barrier_pressure_score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    train_mask = pd.Series([True, True, False, False])
    gate = _gate_specs_by_name(["recovery_support_q40_barrier_q50"])[0]

    mask, report = _gate_mask(frame, train_mask, gate)

    assert mask.tolist() == [False, True, True, True]
    assert report["missing_gate_columns"] == []
    assert "excess_12h_ge_q0.4" in report["thresholds"]
    assert "convexity_t_ge_q0.4" in report["thresholds"]
    assert "barrier_pressure_score_le_q0.5" in report["thresholds"]
