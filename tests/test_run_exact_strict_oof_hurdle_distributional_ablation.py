from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_exact_strict_oof_hurdle_distributional_ablation import (
    add_distributional_targets,
    compose_scores,
    _select_indices,
    _tie_diagnostics,
)


def test_distributional_targets_are_exact_net_of_cost_and_exit_simplex() -> None:
    frame = pd.DataFrame(
        {
            "execution_gross_ev_12h": [0.03, 0.004, -0.02],
            "execution_cost_return": [0.01, 0.01, 0.01],
            "execution_net_ev_12h": [0.02, -0.006, -0.03],
            "execution_exit_reason": ["trailing", "timeout", "full_sl"],
        }
    )
    add_distributional_targets(frame)
    assert frame["target_gross_exceeds_cost"].tolist() == [1, 0, 0]
    assert frame["target_timeout_exit"].tolist() == [0, 1, 0]
    assert frame["target_full_stop"].tolist() == [0, 0, 1]
    assert frame[["target_full_stop", "target_timeout_exit", "target_other_exit"]].sum(axis=1).tolist() == [1, 1, 1]


def test_distributional_targets_reject_cost_reconciliation_failure() -> None:
    frame = pd.DataFrame(
        {
            "execution_gross_ev_12h": [0.03],
            "execution_cost_return": [0.01],
            "execution_net_ev_12h": [0.021],
            "execution_exit_reason": ["timeout"],
        }
    )
    with pytest.raises(ValueError, match="reconcile"):
        add_distributional_targets(frame)


def test_composition_uses_net_payoff_once_and_normalizes_exit_mass() -> None:
    raw = {
        "direct_net": np.array([0.01]),
        "positive_magnitude": np.array([0.04]),
        "loss_magnitude": np.array([0.02]),
        "full_stop_payoff": np.array([-0.05]),
        "timeout_payoff": np.array([-0.01]),
        "other_exit_payoff": np.array([0.03]),
        "joint_multitask_direct_primary": np.array([0.012]),
    }
    probability = {
        "gross_exceeds_cost": np.array([0.75]),
        "full_stop": np.array([0.2]),
        "timeout_exit": np.array([0.3]),
    }
    result = compose_scores(raw, probability)
    assert np.isclose(result["gross_cost_hurdle_ev"][0], 0.75 * 0.04 - 0.25 * 0.02)
    # other has residual 50%; no additional fee is subtracted.
    assert np.isclose(result["exit_policy_mixture_ev"][0], 0.2 * -0.05 + 0.3 * -0.01 + 0.5 * 0.03)
    assert np.isclose(result["direct_exit_blend_050"][0], 0.5 * (0.01 + result["exit_policy_mixture_ev"][0]))


def test_pooled_selection_breaks_cutoff_ties_by_candidate_id() -> None:
    rows = pd.DataFrame(
        {
            "__ts__": [3, 2, 1],
            "__symbol__": ["BTC", "BTC", "BTC"],
            "side_name": ["long", "short", "long"],
            "candidate_id": ["z", "a", "b"],
        }
    )
    score = np.array([1.0, 1.0, 0.2])
    selected = _select_indices(rows, score, 0.34)
    assert selected.tolist() == [1, 0]
    audit = _tie_diagnostics(score, selected)
    assert audit["cutoff_tie_rows"] == 2
    assert audit["cutoff_tie_selected_rows"] == 2
