from __future__ import annotations

import pandas as pd

from scripts.replay_strict_r3_phase_h1_pooled_dual_portfolio import (
    _aggregate,
    _join,
    _phase_hourly_admissions,
)


def _prediction(candidate_ids: list[str], *, ev: list[float]) -> pd.DataFrame:
    timestamp = pd.Timestamp("2026-05-01T00:15:00Z")
    return pd.DataFrame({
        "candidate_id": candidate_ids,
        "__decision_ts__": [timestamp] * len(candidate_ids),
        "__symbol__": [f"{value}/USD:USD" for value in candidate_ids],
        "side_name": ["long"] * len(candidate_ids),
        "final_score": [0.9, 0.8],
        "mc1_expected_bps": ev,
        "policy_path_valid": [True, False],
        "policy_net_bps": [120.0, float("nan")],
        "policy_gross_bps": [220.0, float("nan")],
        "policy_exit_bar_15m": [3.0, float("nan")],
        "policy_entry_price": [1.0, 1.0],
        "policy_exit_price": [1.02, float("nan")],
        "policy_exit_reason": ["trailing", None],
    })


def test_phase_join_freezes_dual_admission_before_outcome_join(tmp_path) -> None:
    current = _prediction(["a", "b"], ev=[80.0, 70.0])
    bcf = _prediction(["a", "b"], ev=[60.0, 40.0])
    current_path = tmp_path / "current.parquet"
    bcf_path = tmp_path / "bcf.parquet"
    current.to_parquet(current_path, index=False)
    bcf.to_parquet(bcf_path, index=False)

    joined, audit = _join(15, current_path, bcf_path, threshold=50.0)

    assert audit == {"phase_minutes": 15, "current_routed_rows": 2, "dual_admitted_target_free_rows": 1}
    assert joined.set_index("candidate_id").loc["a", "dual_admitted"]
    assert not joined.set_index("candidate_id").loc["b", "dual_admitted"]
    assert joined.set_index("candidate_id").loc["a", "mc1_expected_bps"] == 60.0
    # The invalid policy path is retained in the target-free join and is only
    # excluded later by the portfolio candidate construction.
    assert not joined.set_index("candidate_id").loc["b", "policy_path_valid"]


def test_phase_hourly_admission_report_retains_phase_provenance() -> None:
    timestamp = pd.Timestamp("2026-05-01T00:15:00Z")
    combined = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__decision_ts__": [timestamp, timestamp],
        "phase_minutes": [15, 15],
        "current_mc1_expected_bps": [80.0, 70.0],
        "bcf_mc1_expected_bps": [60.0, 40.0],
        "dual_admitted": [True, False],
        "policy_path_valid": [True, False],
    })
    decisions = pd.DataFrame({
        "candidate_id": ["a"],
        "accepted": [True],
        "position_net_return": [0.012],
    })

    hourly, summary = _phase_hourly_admissions(combined, decisions, threshold_bps=50.0)

    assert hourly.loc[0, "phase_minutes"] == 15
    assert hourly.loc[0, "dual_admitted_target_free_rows"] == 1
    assert hourly.loc[0, "dual_admitted_valid_outcome_rows"] == 1
    assert hourly.loc[0, "portfolio_accepted_rows"] == 1
    assert hourly.loc[0, "accepted_net_ev_bps"] == 120.0
    assert summary.loc[0, "portfolio_accepted_rows"] == 1


def test_empty_phase_portfolio_aggregation_is_a_valid_zero_entry_receipt() -> None:
    result = _aggregate(pd.DataFrame(), "month")

    assert result.empty
    assert result.columns.tolist() == ["month", "entries", "net_ev_bps", "net_sum_bps"]
