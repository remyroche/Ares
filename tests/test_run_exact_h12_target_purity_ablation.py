from __future__ import annotations

import pandas as pd
import numpy as np

from scripts.run_exact_h12_target_purity_ablation import _event, _hierarchical_state_probabilities, _paired_day_bootstrap, _persistence_event, _policy_features, _soft_terminal_net_target


def test_policy_features_reject_realised_cost_and_exit_spread_by_construction() -> None:
    frame = pd.DataFrame({
        "estimated_spread_bps": [10.0], "entry_half_spread_bps": [5.0],
        "barrier_pct": [0.02], "execution_entry_price": [100.0],
        "row_cost_bps": [100.0], "exit_half_spread_bps": [50.0],
    })
    columns = _policy_features(frame).columns.tolist()
    assert "row_cost_bps" not in columns
    assert "exit_half_spread_bps" not in columns
    assert {"estimated_spread_bps", "entry_half_spread_bps", "barrier_pct", "entry_price_log"}.issubset(columns)


def test_event_is_exact_three_state_simplex() -> None:
    frame = pd.DataFrame({"event_first": ["favorable_first", "adverse_first_or_conflict", "timeout"], "exact_h12_gross_bps": [150.0, 0.0, 0.0]})
    assert _event(frame).tolist() == ["clean", "adverse", "timeout"]


def test_post_cost_event_requires_favorable_path_and_cost_clearance() -> None:
    frame = pd.DataFrame({
        "event_first": ["favorable_first", "favorable_first", "adverse_first_or_conflict", "timeout"],
        "exact_h12_gross_bps": [101.0, 99.0, 400.0, 400.0],
    })
    assert _event(frame, post_cost_hurdle_bps=0.0).tolist() == ["clean", "timeout", "adverse", "timeout"]


def test_exact_materialised_postcost_event_is_used_without_final_gross_proxy() -> None:
    frame = pd.DataFrame({
        "event_first": ["timeout"],
        "exact_h12_gross_bps": [-500.0],
        "postcost_h0_event": ["clear_cost_first"],
    })
    assert _event(frame, exact_postcost_token="h0").tolist() == ["clean"]


def test_exact_persistence_state_is_a_four_state_simplex() -> None:
    frame = pd.DataFrame({"postcost_h0_four_state": ["clear_then_retained", "clear_then_giveback", "adverse_first_or_conflict", "timeout"]})
    assert _persistence_event(frame, "h0").tolist() == ["retained", "giveback", "adverse", "timeout"]


def test_hierarchical_persistence_probabilities_are_a_four_state_simplex() -> None:
    probabilities = _hierarchical_state_probabilities(
        p_clear=[0.8, 0.2],
        p_retain_given_clear=[0.75, 0.5],
        p_adverse_given_not_clear=[0.25, 0.6],
    )
    assert np.allclose(probabilities, [[0.60, 0.20, 0.05, 0.15], [0.10, 0.10, 0.48, 0.32]])
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_soft_terminal_target_is_cost_aware_and_monotonic() -> None:
    values = _soft_terminal_net_target(np.asarray([-100.0, 0.0, 100.0]), hurdle_bps=0.0, temperature_bps=100.0)
    assert values[0] < values[1] < values[2]
    assert values[1] == 0.5


def test_paired_day_bootstrap_uses_same_candidate_rows_for_control_and_arm() -> None:
    candidates = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "decision_ts": pd.to_datetime(["2024-08-01", "2024-08-01", "2024-08-02", "2024-08-02"], utc=True),
        "exact_h12_net_bps": [-50.0, 20.0, -10.0, 40.0],
    })
    scored = pd.concat([
        candidates.assign(arm="CONTROL_base_opportunity", calibrated_expected_net_bps=[0.0, 1.0, 2.0, 3.0]),
        candidates.assign(arm="E0_direct_net", calibrated_expected_net_bps=[3.0, 2.0, 1.0, 0.0]),
    ], ignore_index=True)
    result = _paired_day_bootstrap(scored, control_arm="CONTROL_base_opportunity", seed=1, replicates=10)
    assert result.arm.tolist() == ["E0_direct_net"]
    assert result.day_blocks.iloc[0] == 2
    assert result.replicates.iloc[0] == 10
