from __future__ import annotations

import pandas as pd

from scripts.replay_strict_r3_score_family_mc1_canonical_policy import _policy_union


def test_policy_union_is_candidate_unique_and_retains_canonical_fields() -> None:
    first = pd.DataFrame({
        "candidate_id": ["a"], "__decision_ts__": pd.to_datetime(["2026-01-01T00:00:00Z"]),
        "final_score": [0.8], "policy_path_valid": [True], "policy_gross_bps": [150.0],
        "policy_net_bps": [50.0], "policy_exit_bar_15m": [4.0], "policy_entry_price": [1.0],
        "policy_exit_price": [1.01], "policy_exit_reason": ["timeout"],
        "policy_label_available_ts": pd.to_datetime(["2026-01-01T12:00:00Z"]),
        "policy_outcome_source": ["coarse_15m"], "policy_cost_bps": [100.0],
    })
    second = first.assign(final_score=0.2)
    union = _policy_union([first, second])
    assert len(union) == 1
    assert union.loc[0, "policy_path_valid"]
    assert union.loc[0, "policy_net_bps"] == 50.0
