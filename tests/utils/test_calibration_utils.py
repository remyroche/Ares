"""Tests for calibration utility helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from src.training.steps.pre_training.calibration_utils import (
    compute_classification_calibration,
    evaluate_conformal_interval,
)


def test_classification_calibration_matches_brier_score() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred_proba = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    result = compute_classification_calibration(y_true, y_pred_proba, classes=[0, 1], n_bins=2)

    expected_brier = brier_score_loss(y_true, y_pred_proba[:, 1])
    assert pytest.approx(expected_brier) == result['brier_score']
    reliability = result['reliability_diagram']['1']
    assert sum(bin_info['count'] for bin_info in reliability) == len(y_true)


def test_conformal_interval_reports_coverage_and_breach() -> None:
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([0.0, 0.0])

    result = evaluate_conformal_interval(y_true, y_pred, coverage_target=0.75)

    assert result['coverage'] < result['coverage_target']
    assert result['coverage_met'] is False
    assert result['interval_width'] == pytest.approx(15.0)


def test_conformal_interval_hits_target_for_well_aligned_predictions() -> None:
    y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.0, 1.0, 2.0, 2.0, 3.0])

    result = evaluate_conformal_interval(y_true, y_pred, coverage_target=0.8)

    assert result['coverage'] >= result['coverage_target']
    assert result['coverage_met'] is True
    assert result['interval_width'] == pytest.approx(2.0)
