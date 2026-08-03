from __future__ import annotations

import pandas as pd

from scripts.audit_pooled_historical_current_transition_score_ties import tie_aware_top10


def test_constant_score_expected_lift_is_one_despite_arbitrary_timestamp_slice() -> None:
    frame = pd.DataFrame({"prediction": [0.2] * 10, "target": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "selected_top10": [True] + [False] * 9})
    result = tie_aware_top10(frame)
    assert result["original_timestamp_tiebreak_precision"] == 1.0
    assert result["tie_aware_expected_precision"] == 0.2
    assert result["tie_aware_expected_lift_unweighted"] == 1.0
    assert result["cutoff_is_ambiguous"]
    assert result["tie_aware_lower_precision"] == 0.0
    assert result["tie_aware_upper_precision"] == 1.0


def test_unique_cutoff_has_no_allocation_ambiguity() -> None:
    frame = pd.DataFrame({"prediction": list(range(10)), "target": [0.0] * 9 + [1.0], "selected_top10": [False] * 9 + [True]})
    result = tie_aware_top10(frame)
    assert not result["cutoff_is_ambiguous"]
    assert result["tie_aware_expected_precision"] == 1.0
    assert result["tie_aware_lower_precision"] == result["tie_aware_upper_precision"]
