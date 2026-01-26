
import pandas as pd
import numpy as np
import sys
import os

# Mock imports
sys.path.append(os.getcwd())

try:
    from src.feature_generation.categories.layer3_specific_features import generate_layer3_features, _compute_gate_regime_features
    from src.training.steps.labeling.label_based_layer_4 import MetaLearnerFeatures, integrate_entropy_bars_into_layer4
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_features():
    # Create dummy data
    idx = pd.date_range("2023-01-01", periods=100, freq="15min")
    df = pd.DataFrame(index=idx)
    df['close'] = np.linspace(100, 110, 100)
    df['high'] = df['close'] + 1
    df['low'] = df['close'] - 1
    df['volume'] = 1000
    df['vwap'] = df['close'] * 1.05 # VWAP is 5% higher to distinguish

    # Add some base model cols
    df['model_a'] = 0.6

    print("Testing Layer 3 Features...")
    l3_feats = generate_layer3_features(df, base_model_cols=['model_a'])

    # Check specific features to see if they used close or vwap
    # Volatility Ratio uses returns.
    # Log ret close: log(110/100) / 100 approx constant
    # Log ret vwap: log(1.05*110 / 1.05*100) = same returns!
    # Wait, if VWAP is exactly proportional to Close, returns are identical.
    # I need VWAP to have different returns.

    df['vwap'] = df['close'] + np.sin(np.arange(100)) * 5
    # Now VWAP returns will differ from Close returns.

    # Re-run
    l3_feats = generate_layer3_features(df, base_model_cols=['model_a'])

    # How to check which one was used?
    # I can check the values of specific features.
    # But I haven't modified the code yet.
    # This script is to verify I can run it and then I'll use it to verify changes.

    print("Layer 3 Features generated:", l3_feats.shape)

    print("Testing Layer 4 MetaLearnerFeatures...")
    ml_gen = MetaLearnerFeatures()
    ml_feats = ml_gen.generate(df)
    print("MetaLearner Features generated:", ml_feats.shape)

if __name__ == "__main__":
    test_features()
