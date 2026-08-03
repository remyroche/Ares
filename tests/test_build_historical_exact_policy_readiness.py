from __future__ import annotations

from scripts.build_historical_exact_policy_readiness import period_gate


def test_period_gate_accepts_only_exact_reconstructed_history_after_parity() -> None:
    accepted = period_gate(
        period="2025-02",
        candidate_rows=100,
        canonical_path_rows=75,
        exact_1m_rows=75,
        minimum_exact_coverage=0.70,
        parity_pass=True,
    )
    assert accepted["new_exact_policy_labels_accepted"]
    assert accepted["exact_1m_coverage_of_original_candidates"] == 0.75

    blocked = period_gate(
        period="2025-12",
        candidate_rows=100,
        canonical_path_rows=75,
        exact_1m_rows=20,
        minimum_exact_coverage=0.70,
        parity_pass=True,
    )
    assert not blocked["new_exact_policy_labels_accepted"]
    assert "insufficient_exact_1m_coverage_of_canonical_candidates" in blocked["blockers"]


def test_period_gate_rejects_missing_canonical_path_inputs() -> None:
    blocked = period_gate(
        period="2025-01",
        candidate_rows=10,
        canonical_path_rows=0,
        exact_1m_rows=0,
        minimum_exact_coverage=0.70,
        parity_pass=True,
    )
    assert not blocked["new_exact_policy_labels_accepted"]
    assert "no_joinable_canonical_path_inputs" in blocked["blockers"]
