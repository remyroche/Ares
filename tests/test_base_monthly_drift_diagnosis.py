from __future__ import annotations

import pandas as pd
import pytest

from extreme_price_movements.base_monthly_drift_diagnosis import (
    BaseMonthlyDriftDiagnosticError,
    classify_drift_attribution,
    paired_score_stability,
)


def test_paired_stability_reports_rank_error_and_deterministic_top_overlap() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["a", "b", "c", "d"],
        "frozen_score": [4.0, 3.0, 2.0, 1.0],
        "refit_score": [4.0, 1.0, 3.0, 2.0],
    })
    result = paired_score_stability(frame, top_fractions=(.05, .50))
    assert result["rows"] == 4
    assert result["score_spearman"] == pytest.approx(.4)
    assert result["score_mae"] == pytest.approx(1.0)
    assert result["top_05_overlap_fraction"] == pytest.approx(1.0)
    assert result["top_50_overlap_rows"] == 1
    assert result["top_50_jaccard"] == pytest.approx(1 / 3)


def test_paired_stability_rejects_duplicate_or_nonfinite_identities() -> None:
    duplicate = pd.DataFrame({"candidate_id": ["a", "a"], "frozen_score": [1., 2.], "refit_score": [1., 2.]})
    with pytest.raises(BaseMonthlyDriftDiagnosticError, match="unique"):
        paired_score_stability(duplicate)
    nonfinite = pd.DataFrame({"candidate_id": ["a"], "frozen_score": [float("nan")], "refit_score": [1.]})
    with pytest.raises(BaseMonthlyDriftDiagnosticError, match="finite"):
        paired_score_stability(nonfinite)


def test_attribution_classifies_mixed_evidence_without_using_rows() -> None:
    result = classify_drift_attribution({
        "frozen_rank_ic": -.02,
        "refit_rank_ic": .03,
        "score_spearman": .70,
        "top_05_overlap_fraction": .50,
        "max_feature_psi": .30,
        "max_feature_extrapolation_rate": .01,
        "train_rank_ic": .05,
        "calibration_slope_shift": .10,
    })
    assert result == "MODEL_DRIFT+INPUT_POPULATION_DRIFT+ECONOMIC_RELATIONSHIP_DRIFT"


def test_attribution_rejects_missing_required_metrics_and_can_return_no_drift() -> None:
    with pytest.raises(BaseMonthlyDriftDiagnosticError, match="lack required"):
        classify_drift_attribution({})
    result = classify_drift_attribution({
        "frozen_rank_ic": .02, "refit_rank_ic": .02,
        "score_spearman": .99, "top_05_overlap_fraction": .99,
        "max_feature_psi": .01, "max_feature_extrapolation_rate": .01,
        "train_rank_ic": .03, "calibration_slope_shift": .01,
    })
    assert result == "NO_DOMINANT_DRIFT"
