import numpy as np
import pandas as pd

from scripts.diagnose_execution_ev_oracle_recovery import recovery_rows, select_global_topk


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "__symbol__": ["A", "B", "C", "D", "E"],
            "side_name": ["long", "short", "long", "short", "long"],
            "canonical_recent_ev_score": [5.0, 4.0, 3.0, 2.0, 1.0],
            # The future oracle winners are b and c; top-40 score picks a,b.
            "execution_net_ev_12h": [-0.010, 0.020, 0.010, 0.0, -0.020],
            "week": ["2026-05-04/2026-05-10"] * 4 + ["2026-05-11/2026-05-17"],
        }
    )


def test_selection_is_one_global_book_with_existing_stable_sort_rule() -> None:
    picked = select_global_topk(_frame(), 0.40)
    assert set(_frame().loc[picked, "candidate_id"]) == {"a", "b"}


def test_recovery_reports_event_precision_recall_and_oracle_overlap() -> None:
    rows = recovery_rows(_frame(), top_fraction=0.40, surplus_bps=0.0)
    overall = next(row for row in rows if row["grouping"] == "overall")
    assert overall["selected_rows"] == 2
    assert overall["surplus_event_rows"] == 2
    assert overall["true_positive_rows"] == 1
    assert overall["false_positive_rows"] == 1
    assert overall["missed_winner_rows"] == 1
    assert overall["precision"] == 0.5
    assert overall["surplus_recall"] == 0.5
    assert overall["oracle_topk_overlap_rows"] == 1
    assert np.isclose(overall["oracle_topk_jaccard"], 1.0 / 3.0)
    assert np.isclose(overall["false_positive_mean_shortfall_bps"], 100.0)


def test_week_breakdown_does_not_rerank_inside_week() -> None:
    rows = recovery_rows(_frame(), top_fraction=0.40, surplus_bps=0.0)
    latest = next(row for row in rows if row["grouping"] == "latest_week")
    # Candidate e is in the latest week but was not selected by the global book.
    assert latest["candidate_rows"] == 1
    assert latest["selected_rows"] == 0
