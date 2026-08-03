import pandas as pd

from scripts.run_causal_retained_transition_interactions import stable_top, tie_metrics


def test_stable_top_and_tie_metrics_are_global_and_deterministic():
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"], "score": [1.0, 1.0, 0.0], "execution_net_ev_12h": [0.01, -0.01, 0.0]})
    chosen = stable_top(frame, "score", 1 / 3)
    assert chosen.candidate_id.tolist() == ["a"]
    assert tie_metrics(frame, "score", 1 / 3)["cutoff_tie_ambiguous"]
