import pandas as pd

from scripts.run_strict_forward_transition_evaluation import causal_feature_columns, global_top10


def test_global_top10_is_pooled_and_tie_deterministic():
    frame = pd.DataFrame({"candidate_id": ["b", "a", "c"], "score": [1.0, 1.0, 0.0]})
    selected = global_top10(frame, "score")
    assert selected.sum() == 1 and selected.iloc[1]


def test_causal_feature_screen_excludes_targets_and_current_state():
    frame = pd.DataFrame({"source_utc": pd.date_range("2025-01-01", periods=3, tz="UTC", freq="h"), "causal": [1.0, 2.0, 3.0], "target__phase": [0.0, 1.0, 0.0], "state_context__current_state": [0.0, 1.0, 1.0]})
    assert causal_feature_columns(frame, frame) == ["causal"]
