import numpy as np

from scripts.run_unresolved_event_mechanism_discovery import _metrics


def test_metrics_rejects_constant_scores_instead_of_tie_breaking() -> None:
    result = _metrics(
        np.full(40, 0.5, dtype=np.float32),
        np.array([1, 1] + [0] * 38, dtype=np.int8),
    )
    assert result["ranking_status"] == "degenerate_score"
    assert np.isnan(result["top05_lift"])
    assert result["top05_selected_rate"] == 0.0


def test_metrics_keeps_all_top_boundary_ties() -> None:
    result = _metrics(
        np.array([0.9, 0.9, 0.8] + [0.1] * 37, dtype=np.float32),
        np.array([1, 0] + [0] * 38, dtype=np.int8),
    )
    assert result["ranking_status"] == "ok"
    assert result["top05_selected_rate"] == 0.05
    assert result["top05_recall"] == 1.0
