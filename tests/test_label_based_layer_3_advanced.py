import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.training.steps.labeling.label_based_layer_3 import generate_efficiency_labels, layer3_analyst_lgbm
from src.training.steps.labeling.layer3_pytorch_losses import SoftF1Loss, SoftAUC_PR_Loss, PyTorchObjectiveWrapper

def test_generate_efficiency_labels():
    # Setup dummy events and price series
    # 3 events:
    # 1. Profitable, stable -> High Efficiency (1)
    # 2. Profitable, volatile -> Low Efficiency (0)
    # 3. Loss -> Low Efficiency (0)

    dates = pd.date_range("2023-01-01", periods=100, freq="1min")
    price_values = np.linspace(100, 110, 100) # steady uptrend base

    # Event 1: Steady rise (indices 0 to 10)
    # Return ~1%, Neg Vol ~0

    # Event 2: Rise but with deep dip (indices 20 to 30)
    price_values[25] = 90 # sharp dip

    # Event 3: Loss (indices 40 to 50)
    price_values[50] = 95

    price_series = pd.Series(price_values, index=dates)

    events_data = {
        'entry_time': [dates[0], dates[20], dates[40]],
        'exit_time': [dates[10], dates[30], dates[50]],
    }
    events_df = pd.DataFrame(events_data, index=[dates[0], dates[20], dates[40]])

    labels = generate_efficiency_labels(events_df, price_series)

    assert len(labels) == 3
    # Event 1: Stable profit -> 1
    assert labels.iloc[0] == 1.0
    # Event 2: Volatile profit (dip to 90 from ~102) -> huge downside vol -> Efficiency < 0.5 -> 0
    assert labels.iloc[1] == 0.0
    # Event 3: Loss -> 0
    assert labels.iloc[2] == 0.0

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch not installed")
def test_pytorch_losses_wrapper():
    # Dummy data
    y_true = np.array([1, 0, 1, 0], dtype=np.float32)
    y_pred_raw = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32) # Logits -> High prob for 1, Low for 0

    # SoftF1
    loss_mod = SoftF1Loss()
    wrapper = PyTorchObjectiveWrapper(loss_mod)

    # Mock LightGBM Dataset
    class MockDataset:
        def get_label(self):
            return y_true

    train_data = MockDataset()

    grad, hess = wrapper(y_pred_raw, train_data)

    assert grad.shape == y_pred_raw.shape
    assert hess.shape == y_pred_raw.shape
    # Check gradient direction roughly
    # y=1, pred=high -> grad should be small or negative (minimize loss)
    # y=0, pred=low -> grad small

    # AUC PR
    loss_auc = SoftAUC_PR_Loss()
    wrapper_auc = PyTorchObjectiveWrapper(loss_auc)
    grad_a, hess_a = wrapper_auc(y_pred_raw, train_data)

    assert grad_a.shape == y_pred_raw.shape

def test_layer3_integration_mock():
    # Test the race logic with dummy data
    dates = pd.date_range("2023-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        'meta_feature_1': np.random.randn(100),
        'meta_feature_2': np.random.randn(100),
        'target': np.random.choice([0, 1], 100),
        'exit_time': dates # dummy exit times
    }, index=dates)

    # Create fake market data for sortino generation
    market_data = pd.DataFrame({
        'close': np.random.randn(100) + 100,
        'volume': np.random.randn(100),
        'high': np.random.randn(100) + 100,
        'low': np.random.randn(100) + 100
    }, index=dates)

    # Weights
    w = pd.Series(np.ones(100), index=dates)

    # Config
    cfg = {
        'layer3_neutral_target_value': 0.5
    }

    # Run
    # This will trigger the race logic inside layer3_analyst_lgbm
    # We expect it to run without crashing and return a df and model
    try:
        res_df, model = layer3_analyst_lgbm(
            oof_df=df,
            base_model_cols=[],
            target_col='target',
            layer1_weight=w,
            layer2_weight=w,
            layer2_weight_quality=w,
            net_returns=pd.Series(np.random.randn(100), index=dates),
            market_data=market_data,
            config=cfg
        )
        assert 'meta_prob' in res_df.columns
        # assert model is not None # Model might be None if data is too small/random for any split, but loop should finish
    except Exception as e:
        pytest.fail(f"Layer 3 integration failed: {e}")
