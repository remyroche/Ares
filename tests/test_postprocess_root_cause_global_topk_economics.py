import pandas as pd

from scripts.postprocess_root_cause_global_topk_economics import _global_top_k


def test_global_top_k_pools_sides_instead_of_forcing_equal_side_allocation():
    frame = pd.DataFrame(
        {
            "candidate_id": ["l1", "l2", "l3", "s1", "s2", "s3", "s4", "s5", "s6", "s7"],
            "combined_economic_prediction_bps": [100.0, 99.0, 98.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            "side_name": ["long", "long", "long", "short", "short", "short", "short", "short", "short", "short"],
        }
    )
    selected = _global_top_k(frame)
    assert selected.candidate_id.tolist() == ["l1"]


def test_global_top_k_breaks_score_ties_by_candidate_id():
    frame = pd.DataFrame(
        {
            "candidate_id": ["z", "a", "b", "c", "d", "e", "f", "g", "h", "i"],
            "combined_economic_prediction_bps": [4.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        }
    )
    assert _global_top_k(frame).candidate_id.tolist() == ["a"]
