from __future__ import annotations

import pandas as pd

from scripts.audit_exact_h12_score_map_ties import _selection


def test_raw_tiebreak_changes_only_equal_mapped_score_membership() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "calibrated_expected_net_bps": [10.0, 9.0, 9.0, 1.0],
        "raw_score": [0.1, 0.2, 0.9, 0.0],
        "exact_h12_net_bps": [0.0, 0.0, 0.0, 0.0],
    })
    stable, details = _selection(frame, 0.50, raw_tiebreak=False)
    resolved, _ = _selection(frame, 0.50, raw_tiebreak=True)
    assert stable.candidate_id.tolist() == ["a", "b"]
    assert resolved.candidate_id.tolist() == ["a", "c"]
    assert details["cutoff_tie_rows"] == 2
    assert details["cutoff_tie_rows_needed"] == 1


def test_raw_tiebreak_preserves_strict_mapped_order() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c"],
        "calibrated_expected_net_bps": [3.0, 2.0, 1.0],
        "raw_score": [0.0, 1.0, 2.0],
        "exact_h12_net_bps": [0.0, 0.0, 0.0],
    })
    stable, _ = _selection(frame, 2 / 3, raw_tiebreak=False)
    resolved, _ = _selection(frame, 2 / 3, raw_tiebreak=True)
    assert stable.candidate_id.tolist() == resolved.candidate_id.tolist() == ["a", "b"]
