import numpy as np

from extreme_price_movements.gate_metrics import compute_stage_gate_metrics


def test_classifier_gate_metrics_accept_rank_scores_outside_probability_range():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    y_score = np.array([-0.15, -0.10, -0.08, -0.03, -0.02, 0.0, 0.01, 0.03, 0.04, 0.08, 0.10, 0.15])

    result = compute_stage_gate_metrics(y_true, y_score, model_type="classifier")

    assert result["Score_Is_Probability"] is False
    assert result["Probability_Metrics_Clipped"] is True
    assert np.isfinite(result["Brier"])
    assert np.isfinite(result["LogLoss"])
