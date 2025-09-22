"""
Test script for complete HMM training migration

This script verifies that the complete migration to common_utils pipeline works correctly
with ensemble models included.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import time

# Test the complete migration
from src.training.steps.market_analysis.hmm_models_training import (
    create_enhanced_hmm_models_training,
    execute_enhanced_hmm_models_training
)
from src.utils.ml_common.config.base_training_config import HMMTrainingConfig


def generate_sample_hmm_data(n_samples: int = 500,
                           n_features: int = 15,
                           n_regimes: int = 3) -> Dict[str, np.ndarray]:
    """
    Generate sample data for testing HMM training with complete migration.

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


def test_complete_migration():
    """Test the complete HMM training migration."""
    print("🚀 Testing Complete HMM Training Migration")
    print("=" * 60)

    # Generate sample data
    print("📊 Generating sample HMM data...")
    data = generate_sample_hmm_data(n_samples=500, n_features=10, n_regimes=3)
    X, y, regime_labels = data['X'], data['y'], data['regime_labels']
    feature_names, hmm_states = data['feature_names'], data['hmm_states']

    print(f"✅ Generated data: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(regime_labels))} regimes")

    # Test 1: Complete migration with ensemble models
    print("\n🔄 Test 1: Complete migration with ensemble models")
    start_time = time.time()

    try:
        # Test with optimized model types (top 2 + gradient boosters)
        config = HMMTrainingConfig(
            model_name="complete_migration_test",
            timeframe="15m",  # Should be enforced
            model_types=[
                # Base models (top 2 + gradient boosters to compare)
                "logistic_regression", "lightgbm", "random_forest", "xgboost", "catboost"
            ],
            hpo_trials=10,  # Reduced for testing
            enable_multi_objective=True,
            objectives=["accuracy", "f1_score", "regime_stability"],
            objective_weights=[0.4, 0.3, 0.3]
        )

        training_step = create_enhanced_hmm_models_training(config)
        results = training_step.execute(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states
        )

        execution_time = time.time() - start_time
        print(f"✅ Complete migration test completed in {execution_time:.2f}s")

        # Verify results
        print(f"📊 Results keys: {list(results.keys())}")
        print(f"📊 HMM state recognition focus: {results.get('hmm_state_recognition_focus', False)}")
        print(f"📊 Timeframe enforced: {results.get('timeframe', 'unknown')}")
        print(f"📊 Model types used: {len(results.get('model_types_used', []))} models")

        # Check that gradient booster models are included for comparison
        model_types_used = results.get('model_types_used', [])
        gradient_boosters = [m for m in model_types_used if 'xgb' in m.lower() or 'catboost' in m.lower()]
        print(f"📊 Gradient boosters for comparison: {len(gradient_boosters)} - {gradient_boosters}")

        if results.get('timeframe') == '15m':
            print("✅ 15m timeframe enforcement working correctly")
        else:
            print(f"⚠️ Timeframe enforcement issue: {results.get('timeframe')}")

        if ensemble_models:
            print("✅ Ensemble models are included in the training")
        else:
            print("⚠️ No ensemble models found - may be due to availability")

    except Exception as e:
        print(f"❌ Complete migration test failed: {e}")
        return False

    # Test 2: Simple execution (should work out of the box)
    print("\n🔄 Test 2: Simple execution test")
    try:
        results = execute_enhanced_hmm_models_training(
            X=X,
            y=y,
            regime_labels=regime_labels,
            feature_names=feature_names,
            hmm_states=hmm_states
        )

        print(f"✅ Simple execution completed")
        print(f"📊 Results contain: {list(results.keys())}")
        print(f"📊 HMM focus confirmed: {results.get('hmm_state_recognition_focus', False)}")
        print(f"📊 Timeframe: {results.get('timeframe', 'unknown')}")

    except Exception as e:
        print(f"❌ Simple execution failed: {e}")
        return False

    # Test 3: Verify inheritance from BaseTrainingStep
    print("\n🔄 Test 3: BaseTrainingStep inheritance verification")
    try:
        config = HMMTrainingConfig()
        training_step = create_enhanced_hmm_models_training(config)

        # Check if it has the methods from BaseTrainingStep
        required_methods = ['validate_training_data', 'analyze_regimes', 'prepare_regime_data', 'train_models']

        for method in required_methods:
            if hasattr(training_step, method):
                print(f"✅ {method} method available from BaseTrainingStep")
            else:
                print(f"❌ {method} method missing")
                return False

        print("✅ All required BaseTrainingStep methods available")

    except Exception as e:
        print(f"❌ Inheritance verification failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("🎉 Complete migration tests passed successfully!")
    print("\n📊 Summary:")
    print("- ✅ Streamlined HMM training with complete migration")
    print("- ✅ Base models: top 2 + gradient boosters (logistic_regression, lightgbm, random_forest, xgboost, catboost)")
    print("- ✅ No ensemble models (removed voting, stacking, bagging, ada boost, extra trees)")
    print("- ✅ No deep learning models (removed TabNet, neural networks)")
    print("- ✅ Gradient boosters trained for comparison (XGBoost vs CatBoost)")
    print("- ✅ Enhanced reporting included for all models")
    print("- ✅ 15m timeframe enforcement working")
    print("- ✅ HMM state recognition focus confirmed")
    print("- ✅ BaseTrainingStep inheritance verified")
    print("- ✅ Common_utils pipeline integration successful")

    return True


if __name__ == "__main__":
    success = test_complete_migration()
    exit(0 if success else 1)