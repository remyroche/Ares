import pandas as pd

from scripts.run_causal_retained_transition_mechanism_extension import join_candidates


def test_join_candidates_uses_exact_signal_hour_and_fails_closed_on_missing_context():
    forward = pd.DataFrame({
        "candidate_id": ["a", "b"],
        "__ts__": pd.to_datetime(["2026-06-01T02:00:00Z", "2026-07-11T02:00:00Z"]),
        "execution_decision_utc": pd.to_datetime(["2026-06-01T03:00:00Z", "2026-07-11T03:00:00Z"]),
        "support_label_available_utc": pd.to_datetime(["2026-06-01T14:00:00Z", "2026-07-11T14:00:00Z"]),
        "window": ["may_to_june_forward_control", "later_july_forward"],
    })
    geometry = pd.DataFrame({
        "signal_context_utc": pd.to_datetime(["2026-06-01T02:00:00Z"]),
        "common_transition_context_available": [True],
    })
    joined, coverage = join_candidates(forward, geometry)
    assert joined.loc[0, "required_signal_context_utc"] == pd.Timestamp("2026-06-01T02:00:00Z")
    assert joined.loc[0, "context_joined"]
    assert not joined.loc[1, "context_joined"]
    assert not coverage.full_window_coverage.all()
