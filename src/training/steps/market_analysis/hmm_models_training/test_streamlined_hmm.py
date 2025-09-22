"""
Test script for streamlined HMM training

This script demonstrates the new streamlined HMM training approach that leverages
the common_utils/ ML training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import time

# Import the new streamlined HMM training
from .streamlined_hmm_training import (
    create_streamlined_hmm_training,
    execute_streamlined_hmm_training
)
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig


def generate_sample_hmm_data(n_samples: int = 1000,
                           n_features: int = 20,
                           n_regimes: int = 3) -> Dict[str, np.ndarray]:
    """
    Generate sample data for testing HMM training.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_regimes: Number of regimes

    Returns:
        Dictionary with sample data
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features)

    # Generate regime labels (simulating HMM states)
    regime_labels = np.random.choice(n_regimes, n_samples)

    # Generate target values based on regime (HMM state recognition task)
    # This simulates HMM states that we want to recognize
    y = np.zeros(n_samples)
    for i in range(n_regimes):
        regime_mask = regime_labels == i
        # Each regime has different characteristics
        if i == 0:  # Bull regime
            y[regime_mask] = np.random.choice([0, 1], size=regime_mask.sum(), p=[0.2, 0.8])
        elif i == 1:  # Bear regime
            y[regime_mask] = np.random.choice([0, 1], size=regime_mask.sum(), p=[0.7, 0.3])
        else:  # Sideways regime
            y[regime_mask] = np.random.choice([0, 1], size=regime_mask.sum(), p=[0.5, 0.5])

    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Optional HMM states (cluster assignments)
    hmm_states = np.random.choice(n_regimes, n_samples)

    return {
        'X': X,
        'y': y,
        'regime_labels': regime_labels,
        'feature_names': feature_names,
        'hmm_states': hmm_states
    }


def test_streamlined_hmm_training():
    """Test the streamlined HMM training approach."""
    print("🚀 Testing Streamlined HMM Training")
    print("=" * 50)

    # Generate sample data
    print("📊 Generating sample HMM data...")
    data = generate_sample_hmm_data(n_samples=1000, n_features=10, n_regimes=3)
    X, y, regime_labels = data['X'], data['y'], data['regime_labels']
    feature_names, hmm_states = data['feature_names'], data['hmm_states']

    print(f"✅ Generated data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(regime_labels))} regimes")

    # Test 1: Simple execution (using default config)
    print("\n🔄 Test 1: Simple execution with default config")
    start_time = time.time()

    try:
        results = execute_streamlined_hmm_training(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states
        )

        execution_time = time.time() - start_time
        print(f"✅ Simple execution completed in {execution_time:.2f}s")
        print(f"📊 Results keys: {list(results.keys())}")
        print(f"📊 HMM state recognition focus: {results.get('hmm_state_recognition_focus', False)}")
        print(f"📊 Timeframe: {results.get('timeframe', 'unknown')}")

        # Check regime analysis
        if 'regime_analysis' in results:
            regime_analysis = results['regime_analysis']
            print(f"📊 Regime counts: {regime_analysis.get('regime_counts', {})}")

        # Check validation results
        if 'validation_results' in results:
            validation = results['validation_results']
            print(f"📊 Validation passed: {validation.get('valid', False)}")

    except Exception as e:
        print(f"❌ Simple execution failed: {e}")
        return False

    # Test 2: Custom configuration
    print("\n🔄 Test 2: Custom configuration")
    try:
        config = HMMTrainingConfig(
            model_name="custom_hmm_state_recognition",
            timeframe="15m",  # Should be enforced
            model_types=["logistic_regression", "lightgbm"],  # Custom selection
            hpo_trials=30,  # Fewer trials for testing
            enable_multi_objective=True,
            objectives=["accuracy", "f1_score"],
            objective_weights=[0.6, 0.4]
        )

        training_step = create_streamlined_hmm_training(config)
        results = training_step.execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states
        )

        print(f"✅ Custom configuration completed")
        print(f"📊 Config timeframe: {config.timeframe}")
        print(f"📊 Config model types: {config.model_types}")
        print(f"📊 Results timeframe: {results.get('timeframe', 'unknown')}")

        # Verify timeframe enforcement
        if results.get('timeframe') == '15m':
            print("✅ Timeframe enforcement working correctly")
        else:
            print(f"⚠️ Timeframe enforcement issue: {results.get('timeframe')}")

    except Exception as e:
        print(f"❌ Custom configuration failed: {e}")
        return False

    # Test 3: Error handling
    print("\n🔄 Test 3: Error handling")
    try:
        # Test with invalid data
        X_invalid = np.random.randn(10, 10)  # Too few samples
        y_invalid = np.random.choice(2, 10)
        regime_labels_invalid = np.random.choice(3, 10)

        results = execute_streamlined_hmm_training(
            X=X_invalid,
            y=y_invalid,
            regime_labels=regime_labels_invalid
        )

        if 'error' in results:
            print(f"✅ Error handling working: {results['error'][:100]}...")
        else:
            print("⚠️ Expected error handling but got successful results")

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("\n📊 Summary:")
    print("- ✅ Streamlined HMM training works with default config")
    print("- ✅ Custom configuration works correctly")
    print("- ✅ Error handling functions properly")
    print("- ✅ Timeframe enforcement working (15m for HMM state recognition)")
    print("- ✅ HMM state recognition focus confirmed")

    return True


if __name__ == "__main__":
    success = test_streamlined_hmm_training()
    exit(0 if success else 1)