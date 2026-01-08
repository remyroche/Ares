#!/usr/bin/env python3
"""
Debug sample weights issue.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Add the project root to Python path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tprint import tprint_info


def debug_sample_weights():
    """Debug sample weights implementation."""
    tprint_info("🔍 Debugging Sample Weights")

    # Create simple test data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], 100, p=[0.7, 0.3]))
    weights = pd.Series(np.where(y == 1, 2.0, 1.0))

    tprint_info(f"Data shapes: X={X.shape}, y={y.shape}, weights={weights.shape}")
    tprint_info(f"Target distribution: {y.value_counts().to_dict()}")
    tprint_info(f"Weight distribution: mean={weights.mean():.2f}, unique={weights.unique()}")

    # Test model fitting with sample weights
    try:
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y, sample_weight=weights.values)
        tprint_info("✅ Model fitting with sample weights works!")

        # Test prediction
        preds = rf.predict(X)
        proba = rf.predict_proba(X)
        tprint_info(f"✅ Predictions work: {len(preds)} predictions, proba shape: {proba.shape}")

    except Exception as e:
        tprint_info(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_sample_weights()










