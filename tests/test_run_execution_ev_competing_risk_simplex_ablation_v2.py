from __future__ import annotations

from scripts.run_execution_ev_competing_risk_simplex_ablation_v2 import (
    CLASS_NAMES,
    MIN_CONDITIONAL_PAYOFF_ROWS,
    SCHEMA,
    joint_hpo_combinations,
)


def test_v2_declares_bounded_conditional_support_without_changing_hpo_surface() -> None:
    assert SCHEMA == "execution_ev_competing_risk_simplex_ablation_v2"
    assert MIN_CONDITIONAL_PAYOFF_ROWS == 200
    combinations = joint_hpo_combinations(
        ({"classifier": 0}, {"classifier": 1}),
        {name: ({"payoff": 0}, {"payoff": 1}) for name in CLASS_NAMES},
    )
    assert len(combinations) == 16
    assert {
        tuple(sorted(candidate["payoff_params"]))
        for candidate in combinations
    } == {tuple(sorted(CLASS_NAMES))}
