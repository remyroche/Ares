import pandas as pd

from scripts.diagnose_febapr_base_target_execution_alignment import top_global


def test_top_global_is_not_timestamp_local():
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c", "d"], "hour": [1, 1, 2, 2], "score": [4.0, 3.0, 2.0, 1.0]})
    assert top_global(frame, "score")["candidate_id"].tolist() == ["a"]
