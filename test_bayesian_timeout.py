#!/usr/bin/env python3
"""
Test script to validate Bayesian optimization timeout fixes
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_bayesian_timeout():
    """Test that Bayesian optimization respects timeouts."""

    try:
        from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager, BayesianOptimizationConfig

        # Create test data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # Generate synthetic time series data
        data = np.random.randn(n_samples, n_features)
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])

        # Create manager
        config = BayesianOptimizationConfig(
            n_trials=3,  # Very few trials
            timeout=10   # Very short timeout
        )

        manager = EnhancedHMMCompositeManager()
        manager.logger.info("🧪 Testing Bayesian optimization with timeout...")

        start_time = time.time()

        # This should complete quickly due to timeout
        result = manager.optimize_hmm_parameters_vectorized(df.values, config)

        elapsed = time.time() - start_time

        print(".2f"        print(f"✅ Optimization result: {result.get('success', False)}")

        if elapsed < 15:  # Should complete well before 15 seconds
            print("✅ Timeout protection working correctly!")
            return True
        else:
            print("⚠️ Optimization took longer than expected")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Bayesian optimization timeout fixes...")
    success = test_bayesian_timeout()
    sys.exit(0 if success else 1)
