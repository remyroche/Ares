import numpy as np

from extreme_price_movements.gate_metrics import compute_stage_gate_metrics


def test_meta_regression_gate_does_not_require_quantile_coverage():
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.normal(0.0, 0.2, size=n)
    y_pred = y_true + rng.normal(0.0, 0.05, size=n)
    y_ret = y_true + rng.normal(0.0, 0.05, size=n)

    out = compute_stage_gate_metrics(y_true, y_pred, y_ret=y_ret, model_type="meta_regression")

    assert "Coverage" not in out
    assert "Pass_Coverage" not in out
    assert "Pass_Robust_Loss" in out
    assert "Pass_Bias" in out
    assert "Pass_IC" in out


def test_quantile_meta_gate_keeps_coverage_metrics():
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.normal(0.0, 0.2, size=n)
    # synthetic upper-quantile-ish predictor
    y_pred = y_true + 0.2

    out = compute_stage_gate_metrics(y_true, y_pred, model_type="quantile_meta")

    assert "Coverage" in out
    assert "Pass_Coverage" in out
