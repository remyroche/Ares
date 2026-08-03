from __future__ import annotations

import pandas as pd

from scripts.materialize_historical_exact_h12_postcost_persistence_labels import build_labels


def test_persistence_is_conditional_on_exact_cost_clearance() -> None:
    events = pd.DataFrame({
        "candidate_id": ["a", "b", "c"], "side": ["long", "long", "short"],
        "decision_ts": pd.to_datetime(["2024-01-01"] * 3, utc=True),
        "label_end_ts": pd.to_datetime(["2024-01-01 12:00"] * 3, utc=True),
        "label_available_ts": pd.to_datetime(["2024-01-01 12:00"] * 3, utc=True),
        "postcost_target_id": ["exact_1m_h12_postcost_barrier_first_fixed100bps_v1"] * 3,
        "execution_policy_id": ["historical_current_frozen_spread_counterfactual_h12_v1"] * 3,
        "cost_model_id": ["current_frozen_spread_counterfactual_row_cost_v1"] * 3,
        "postcost_h0_event": ["clear_cost_first", "clear_cost_first", "timeout"],
        "postcost_h25_event": ["clear_cost_first", "clear_cost_first", "timeout"],
    })
    alignment = events.loc[:, ["candidate_id", "side", "decision_ts", "label_end_ts", "label_available_ts", "execution_policy_id", "cost_model_id"]].copy()
    alignment["exact_h12_net_bps"] = [10.0, -10.0, 20.0]
    result = build_labels(events, alignment).set_index("candidate_id")
    assert result.loc["a", "postcost_h0_four_state"] == "clear_then_retained"
    assert result.loc["b", "postcost_h0_four_state"] == "clear_then_giveback"
    assert result.loc["c", "postcost_h0_persistence_target_valid"] == 0
