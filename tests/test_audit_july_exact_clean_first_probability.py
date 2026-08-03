from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_july_exact_clean_first_probability import _top_decile, probability_metrics


def test_probability_metrics_reports_exact_fixed_bin_calibration() -> None:
    result = probability_metrics(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.6, 0.2]))
    assert result["rows"] == 4
    assert result["auc"] == 1.0
    assert result["pr_auc"] == 1.0
    assert result["brier"] > 0.0
    assert result["ece_10"] > 0.0


def test_top_decile_uses_deterministic_candidate_id_tie_break_and_exact_net() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["b", "a", "c", "d", "e", "f", "g", "h", "i", "j"],
        "catboost_hard_clean_first__probability": [0.9, 0.9, *([0.1] * 8)],
        "exact_clean_first": [0, 1, *([0] * 8)],
        "exact_adverse_first_or_conflict": [1, 0, *([0] * 8)],
        "exact_timeout": [0, 0, *([1] * 8)],
        "execution_net_ev_12h": [-0.01, 0.02, *([0.0] * 8)],
    })
    result = _top_decile(frame)
    assert result["top10_rows"] == 1
    assert result["top10_exact_clean_first_precision"] == 1.0
    assert result["top10_net_ev_bps"] == 200.0
