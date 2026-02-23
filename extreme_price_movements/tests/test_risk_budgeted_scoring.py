import numpy as np

from extreme_price_movements.optimization import RiskBudgetConfig, score_backtest_risk_budgeted
from extreme_price_movements.sample_weights import drawdown_aware_weights


def test_score_backtest_risk_budgeted_feasible_path():
    r = np.array([0.01, 0.005, -0.002, 0.004, 0.003], dtype=float)
    x = np.array([0.2, 0.25, 0.2, 0.22, 0.21], dtype=float)
    cfg = RiskBudgetConfig(ui_max=0.20, x_min=0.05, lambda_rs=0.1, hard_fail=True)

    out = score_backtest_risk_budgeted(r, x, cfg)

    assert np.isfinite(out["score"])
    assert out["ui_violation"] == 0.0
    assert out["x_violation"] == 0.0
    assert out["xbar"] >= cfg.x_min


def test_score_backtest_risk_budgeted_hard_fail_activity():
    r = np.array([0.01, -0.01, 0.005, -0.003], dtype=float)
    x = np.array([0.0, 0.01, 0.0, 0.01], dtype=float)
    cfg = RiskBudgetConfig(ui_max=1.0, x_min=0.2, hard_fail=True)

    out = score_backtest_risk_budgeted(r, x, cfg)

    assert out["x_violation"] > 0.0
    assert out["score"] == -1e9


def test_drawdown_aware_weights_overweights_drawdown_and_early_episode():
    dd = np.array([0.0, 0.0, 0.10, 0.12, 0.08, 0.0, 0.05, 0.02], dtype=float)
    w = drawdown_aware_weights(dd, k_dd=5.0, k_early=2.0, tau=24.0)

    assert w.shape == dd.shape
    base = 1.0 + 5.0 * dd
    assert w[2] > base[2]  # early-episode bonus applies after episode start
    assert w[2] > w[1]  # underwater > non-underwater
    assert np.all(w >= 1.0)
