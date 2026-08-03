from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_exact_policy_capture_hurdle_ablation import (
    add_hurdle_targets,
    compose_atr_soft_scores,
    compose_capture_adverse_score,
    compose_decomposed_scores,
    compose_gross_cost_scores,
    compose_hurdle_scores,
)


def test_hurdle_targets_use_row_specific_exact_cost() -> None:
    frame = pd.DataFrame(
        {
            "execution_mfe_return_12h": [0.015, 0.015],
            "execution_gross_ev_12h": [0.012, 0.012],
            "execution_cost_return": [0.010, 0.020],
            "execution_net_ev_12h": [0.002, -0.008],
            "soft_label": [0.9, 0.1],
            "favorable_first": [1, 0],
            "adverse_first": [0, 1],
            "timeout": [0, 0],
        }
    )
    targets = add_hurdle_targets(frame)
    assert targets["target_opportunity_hard"].tolist() == [1, 0]
    assert targets["target_capture_positive"].tolist() == [1, 0]
    assert (
        targets.loc[0, "target_opportunity_soft"]
        > targets.loc[1, "target_opportunity_soft"]
    )
    assert np.allclose(
        targets[
            [
                "target_soft_timeout",
                "target_soft_adverse",
                "target_soft_favorable",
            ]
        ].sum(axis=1),
        1.0,
    )
    assert np.isclose(targets.loc[0, "target_soft_favorable"], 0.9)
    assert np.isclose(targets.loc[1, "target_soft_adverse"], 0.9)


def test_hurdle_composition_is_nested_and_capture_guarded() -> None:
    scores = compose_hurdle_scores(
        np.array([0.8]),
        np.array([0.5]),
        np.array([np.log1p(100.0)]),
        np.array([0.25]),
    )
    assert np.isclose(scores["hurdle_prob"][0], 0.4)
    assert np.isclose(scores["hurdle_ev"][0], 0.004)
    assert np.isclose(scores["hurdle_capture_guard"][0], 0.001)


def test_gross_heads_subtract_known_row_cost_exactly_once() -> None:
    scores = compose_gross_cost_scores(
        direct_gross=np.array([0.031]),
        p_opportunity=np.array([0.80]),
        p_capture=np.array([0.50]),
        positive_gross_log_bps=np.array([np.log1p(300.0)]),
        exact_cost=np.array([0.010]),
    )
    assert np.isclose(scores["direct_gross_minus_exact_cost"][0], 0.021)
    # 0.80 x 0.50 x 3.00% gross - 1.00% exact row fee.
    assert np.isclose(
        scores["capture_gross_mixture_minus_exact_cost"][0], 0.002
    )


def test_decomposed_ev_preserves_outcome_units_and_fixed_blend() -> None:
    scores = compose_decomposed_scores(
        clean_probability=np.array([0.6]),
        adverse_probability=np.array([0.3]),
        competing_probability=np.array([[0.1, 0.3, 0.6]]),
        timeout_net=np.array([-0.01]),
        adverse_net=np.array([-0.02]),
        favorable_net=np.array([0.04]),
        direct_net=np.array([0.01]),
    )
    expected = 0.1 * -0.01 + 0.3 * -0.02 + 0.6 * 0.04
    assert np.isclose(scores["binary_decomposed_ev"][0], expected)
    assert np.isclose(scores["competing_decomposed_ev"][0], expected)
    assert np.isclose(
        scores["direct_competing_blend_050"][0],
        0.5 * 0.01 + 0.5 * expected,
    )


def test_atr_soft_and_explicit_adverse_loss_scores() -> None:
    soft = compose_atr_soft_scores(
        soft_competing_probability=np.array([[0.2, 0.3, 0.5]]),
        timeout_net=np.array([-0.01]),
        adverse_net=np.array([-0.03]),
        favorable_net=np.array([0.04]),
    )
    assert np.isclose(soft["atr_soft_favorable_probability"][0], 0.5)
    assert np.isclose(
        soft["atr_soft_decomposed_ev"][0],
        0.2 * -0.01 + 0.3 * -0.03 + 0.5 * 0.04,
    )
    score = compose_capture_adverse_score(
        p_opportunity=np.array([0.8]),
        p_capture_given_opportunity=np.array([0.5]),
        positive_net_log_bps=np.array([np.log1p(200.0)]),
        p_adverse=np.array([0.25]),
        adverse_net=np.array([-0.03]),
    )
    assert np.isclose(score[0], 0.8 * 0.5 * 0.02 - 0.25 * 0.03)
