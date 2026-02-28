import numpy as np

from extreme_price_movements.meta_training.utility_smooth import smooth_utility_from_mfe_mae


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
