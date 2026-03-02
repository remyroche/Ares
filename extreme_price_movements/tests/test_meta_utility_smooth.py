import numpy as np

from extreme_price_movements.meta_training.utility_smooth import (
    smooth_utility_from_mfe_mae,
    smooth_utility_from_log_heads_standardized,
)


def test_smooth_utility_near_positive_tp_and_negative_sl():
    tp, sl, alpha = 0.02, 0.01, 25.0
    u_good = smooth_utility_from_mfe_mae(mfe=np.array([0.20]), mae=np.array([0.001]), tp=tp, sl=sl, alpha=alpha)
    u_bad = smooth_utility_from_mfe_mae(mfe=np.array([0.001]), mae=np.array([0.20]), tp=tp, sl=sl, alpha=alpha)
    assert u_good[0] > 0.7 * tp
    assert u_bad[0] < -0.2 * sl


def test_smooth_utility_gradients_flow():
    import pytest
    torch = pytest.importorskip("torch")
    tp, sl, alpha = 0.02, 0.01, 20.0
    mfe = torch.tensor([0.015], requires_grad=True)
    mae = torch.tensor([0.008], requires_grad=True)
    u = tp * torch.sigmoid(alpha * (mfe - tp)) - sl * torch.sigmoid(alpha * (mae - sl))
    u.backward()
    assert mfe.grad is not None and torch.abs(mfe.grad).item() > 0
    assert mae.grad is not None and torch.abs(mae.grad).item() > 0


def test_standardized_log_heads_preserve_directionality():
    tp, sl, alpha = 0.02, 0.01, 6.0
    # Same standardization stats for both points
    kwargs = {
        "tp": tp,
        "sl": sl,
        "alpha": alpha,
        "mfe_mean": 0.5,
        "mfe_std": 0.2,
        "mae_mean": 0.5,
        "mae_std": 0.2,
    }
    u_good = smooth_utility_from_log_heads_standardized(
        log_mfe=np.array([0.9]),
        log_mae=np.array([0.2]),
        **kwargs,
    )
    u_bad = smooth_utility_from_log_heads_standardized(
        log_mfe=np.array([0.2]),
        log_mae=np.array([0.9]),
        **kwargs,
    )
    assert u_good[0] > u_bad[0]
