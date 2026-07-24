from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme_price_movements.inference.run_inference import (
    _apply_runtime_sizing_overlay,
)
from extreme_price_movements.simple_policy_winner import (
    WINNER_POLICY_PATHWAY_ID,
    apply_raw_bayesian_sizing_state,
    fit_raw_bayesian_sizing_state,
)


class _Executor:
    def __init__(self, params):
        self.params = params

    def resolve_simple_policy_strategy_id(self, *_args):
        return "long_base"

    def get_simple_policy_stop_params(self, strategy_id):
        assert strategy_id == "long_base"
        return self.params


def _state_and_decision():
    n = 120
    rows = pd.DataFrame(
        {
            "side": ["long"] * n,
            "policy_archetype": ["continuation"] * 60 + ["compression"] * 60,
            "rank_pct": np.linspace(0.9, 1.0, n),
            "expected_net_ev_after_1pct": np.linspace(-0.01, 0.03, n),
            "meta_hit_probability_uncertainty_p1mp": np.linspace(0.25, 0.02, n),
            "gmm_ood_score": np.linspace(2.0, 0.0, n),
        }
    )
    state = fit_raw_bayesian_sizing_state(
        rows,
        np.linspace(-0.02, 0.04, n),
        strength=3.0,
        ood_weight=0.5,
    )
    decision = {
        "policy_archetype": "compression",
        "chain_results": {
            "policy_archetype": "compression",
            "threshold_basis_corrected_expected_ev_rank": 0.97,
            "expected_net_ev_after_1pct": 0.025,
            "meta_hit_probability_uncertainty_p1mp": 0.03,
            "gmm_ood_score": 0.1,
            "uncertainty_ev_size_multiplier": 0.2,
        },
    }
    return state, decision


def test_live_raw_bayesian_sizing_matches_frozen_state_exactly():
    state, decision = _state_and_decision()
    params = {
        "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "raw_bayesian_sizing_state": state,
    }
    executor = _Executor(params)
    expected_row = pd.DataFrame(
        [
            {
                "side": "long",
                "policy_archetype": "compression",
                "rank_pct": 0.97,
                "expected_net_ev_after_1pct": 0.025,
                "meta_hit_probability_uncertainty_p1mp": 0.03,
                "gmm_ood_score": 0.1,
            }
        ]
    )
    expected_multiplier = apply_raw_bayesian_sizing_state(expected_row, state)[0]

    sized, audit = _apply_runtime_sizing_overlay(
        100.0,
        decision=decision,
        executor=executor,
        strategy_id="long_base",
        side="long",
        remaining_total_notional=1_000.0,
    )

    assert sized == pytest.approx(100.0 * expected_multiplier)
    assert audit["raw_bayesian_size_multiplier"] == pytest.approx(expected_multiplier)
    assert audit["sizing_overlay_source"] == "raw_bayesian_v1_frozen_train_state"
    assert sized != pytest.approx(20.0)


def test_live_raw_bayesian_sizing_fails_closed_without_frozen_state():
    _, decision = _state_and_decision()
    executor = _Executor({"policy_pathway_id": WINNER_POLICY_PATHWAY_ID})
    with pytest.raises(RuntimeError, match="frozen raw_bayesian_sizing_state"):
        _apply_runtime_sizing_overlay(
            100.0,
            decision=decision,
            executor=executor,
            strategy_id="long_base",
            side="long",
        )


def test_live_raw_bayesian_sizing_maps_canonical_mlp_aliases_and_zero_ood():
    state, decision = _state_and_decision()
    state = dict(state)
    state["ev_column"] = "expected_net_ev_after_1pct_mlp_direct"
    state["uncertainty_column"] = "meta_hit_probability_uncertainty_p1mp"
    state["ood_columns"] = ["gmm_ood_score", "cluster_entropy_norm"]
    state["ood_weight"] = 0.0
    chain = decision["chain_results"]
    chain.pop("expected_net_ev_after_1pct", None)
    chain.pop("meta_hit_probability_uncertainty_p1mp", None)
    chain.pop("gmm_ood_score", None)
    chain["market_state_mlp_expected_net_ev_after_1pct"] = 0.025
    chain["v9_tail95_predecessor_rank"] = 0.97

    params = {
        "policy_pathway_id": WINNER_POLICY_PATHWAY_ID,
        "raw_bayesian_sizing_state": state,
    }
    executor = _Executor(params)
    expected = pd.DataFrame(
        [
            {
                "side": "long",
                "policy_archetype": "compression",
                "rank_pct": 0.97,
                "expected_net_ev_after_1pct_mlp_direct": 0.025,
                "meta_hit_probability_uncertainty_p1mp": 0.97 * 0.03,
            }
        ]
    )
    expected_multiplier = apply_raw_bayesian_sizing_state(expected, state)[0]

    sized, audit = _apply_runtime_sizing_overlay(
        100.0,
        decision=decision,
        executor=executor,
        strategy_id="long_base",
        side="long",
    )

    assert audit["raw_bayesian_size_multiplier"] == pytest.approx(
        expected_multiplier
    )
    assert sized == pytest.approx(100.0 * expected_multiplier)
