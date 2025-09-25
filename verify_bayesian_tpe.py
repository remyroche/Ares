#!/usr/bin/env python3
"""
Simple verification script for Bayesian TPE optimizer.
This script checks that the module can be loaded and basic functionality works.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the Bayesian TPE optimizer can be imported."""
    try:
        # Test basic imports
        from utils.ml_common.optimization.bayesian_tpe_optimizer import (
            BayesianTPEOptimizer,
            OptimizationConfig,
            TPEConfig,
            GridConfig,
            OptimizationResult,
            optimize_hyperparameters,
            create_optimization_config
        )
        print("✅ All imports successful")

        # Test class instantiation
        config = create_optimization_config(n_trials=10)
        optimizer = BayesianTPEOptimizer(config=config, model_type='xgboost')
        print("✅ Optimizer instantiation successful")

        # Test configuration creation
        from utils.ml_common.optimization.bayesian_tpe_optimizer import OptimizationConfig
        custom_config = OptimizationConfig()
        print("✅ Configuration creation successful")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structure():
    """Test that the module structure is correct."""
    try:
        import inspect

        # Check that key classes exist
        from utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer

        # Verify key methods exist
        required_methods = [
            'optimize',
            '_coarse_grid_optimization',
            '_fine_grid_optimization',
            '_bayesian_tpe_optimization'
        ]

        for method in required_methods:
            if not hasattr(BayesianTPEOptimizer, method):
                print(f"❌ Missing method: {method}")
                return False

        print("✅ Module structure verification successful")
        return True

    except Exception as e:
        print(f"❌ Structure test failed: {e}")
        return False

def test_search_space():
    """Test search space functionality."""
    try:
        from utils.ml_common.optimization.bayesian_tpe_optimizer import BayesianTPEOptimizer

        optimizer = BayesianTPEOptimizer(model_type='xgboost')

        # Test default search space
        search_space = optimizer._get_default_search_space('xgboost')

        required_params = ['max_depth', 'learning_rate', 'n_estimators']
        for param in required_params:
            if param not in search_space:
                print(f"❌ Missing parameter in search space: {param}")
                return False

        print("✅ Search space verification successful")
        return True

    except Exception as e:
        print(f"❌ Search space test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🔍 Verifying Bayesian TPE Optimizer...")
    print("=" * 50)

    tests = [
        test_imports,
        test_structure,
        test_search_space
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 50)
    print(f"✅ Verification complete: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Bayesian TPE optimizer is ready to use.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)