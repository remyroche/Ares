#!/usr/bin/env python3
"""
Test script to verify that Bayesian optimization now uses Optuna's parameter suggestions.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager, BayesianOptimizationConfig

def test_bayesian_optimization():
    """Test that Bayesian optimization actually uses Optuna parameter suggestions."""

    print("🧪 Testing Bayesian optimization parameter suggestions...")

    # Create manager
    manager = EnhancedHMMCompositeManager()

    # Create some dummy data
    np.random.seed(42)
    data = np.random.randn(1000, 5)  # 1000 samples, 5 features

    # Create config
    config = BayesianOptimizationConfig()
    config.n_trials = 3  # Just a few trials for testing
    config.timeout = 30

    print("📊 Testing with dummy data shape:", data.shape)
    print("🎯 Running Bayesian optimization with", config.n_trials, "trials...")

    try:
        # Convert to DataFrame for the method
        import pandas as pd
        df_data = pd.DataFrame(data)

        # Run optimization
        result = manager.optimize_hmm_parameters(
            data=df_data,
            config=config,
            mode='conservative'
        )

        print("✅ Optimization completed successfully!")
        print("🏆 Best parameters found:", result.get('best_params', 'N/A'))
        print("📊 Best score:", result.get('best_score', 'N/A'))

        # Check that we got actual parameters (not empty dict)
        best_params = result.get('best_params', {})
        if best_params and len(best_params) > 0:
            print("✅ SUCCESS: Optuna provided parameter suggestions!")
            print("🔧 Parameters:", best_params)
            return True
        else:
            print("❌ FAILURE: Still getting empty parameters - Optuna not working")
            return False

    except Exception as e:
        print("❌ Test failed with error:", str(e))
        return False

if __name__ == "__main__":
    success = test_bayesian_optimization()
    sys.exit(0 if success else 1)
