from __future__ import annotations

import pandas as pd

from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import _to_candidates


def test_common_agreement_candidate_uses_only_explicitly_admitted_valid_rows() -> None:
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    panel = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": [timestamp, timestamp],
        "__symbol__": ["A/USD:USD", "B/USD:USD"],
        "policy_path_valid": [True, True],
        "policy_gross_bps": [180.0, 190.0],
        "policy_net_bps": [80.0, 90.0],
        "policy_exit_bar_15m": [3, 4],
        "policy_entry_price": [1.0, 1.0],
        "policy_exit_price": [1.01, 1.02],
        "policy_exit_reason": ["trailing", "timeout"],
        "policy_cost_bps": [100.0, 100.0],
    })
    candidates = _to_candidates(
        panel,
        admission=pd.Series([True, False], index=panel.index),
        priority=pd.Series([71.0, 999.0], index=panel.index),
    )
    assert candidates["candidate_id"].tolist() == ["a"]
    assert candidates["mapped_expected_net_bps"].tolist() == [71.0]
    assert candidates["net_return"].tolist() == [0.008]
