#!/usr/bin/env python3
"""
Simple test for Bayesian TPE optimizer migrations.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_bayesian_tpe_import():
    """Test that we can import the Bayesian TPE optimizer."""
    try:
        from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
            BayesianTPEOptimizer,
            BayesianTPEConfig
        )
        print("✅ Bayesian TPE Optimizer import successful")
        return True
    except Exception as e:
        print(f"❌ Bayesian TPE Optimizer import failed: {e}")
        return False

def test_hpo_utils_import():
    """Test that we can import the migrated HPO utils."""
    try:
        from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
        print("✅ HPO Utils import successful")
        return True
    except Exception as e:
        print(f"❌ HPO Utils import failed: {e}")
        return False

def test_final_parameters_import():
    """Test that we can import the migrated Final Parameters Optimization."""
    try:
        from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
        print("✅ Final Parameters Optimization import successful")
        return True
    except Exception as e:
        print(f"❌ Final Parameters Optimization import failed: {e}")
        return False

def test_attention_network_import():
    """Test that we can import the migrated Attention Network Optimizer."""
    try:
        from src.training.steps.model_training.bayesian_optimization_msm import AttentionNetworkOptimizer
        print("✅ Attention Network Optimizer import successful")
        return True
    except Exception as e:
        print(f"❌ Attention Network Optimizer import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the Bayesian TPE optimizer."""
    try:
        from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
            BayesianTPEOptimizer,
            BayesianTPEConfig
        )
        
        # Create simple test data
        np.random.seed(42)
        
        # Define simple search space
        search_space = {
            'param1': {'type': 'float', 'low': 0.0, 'high': 1.0},
            'param2': {'type': 'int', 'low': 1, 'high': 10}
        }
        
        # Define simple objective function
        def objective_function(params, **kwargs):
            return params['param1'] + params['param2'] / 10.0
        
        # Configure optimizer
        config = BayesianTPEConfig(
            n_trials=2,  # Very small for testing
            timeout_seconds=10,
            enable_grid_search=False,  # Disable grid search for simple test
            backend='optuna',
            enable_parallel=False,
            log_level='WARNING'
        )
        
        # Run optimization
        optimizer = BayesianTPEOptimizer(config)
        result = optimizer.optimize(objective_function, search_space)
        
        if result.success:
            print("✅ Bayesian TPE Optimizer basic functionality test passed!")
            print(f"   Best score: {result.best_score:.4f}")
            print(f"   Best params: {result.best_params}")
            return True
        else:
            print(f"❌ Bayesian TPE Optimizer test failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Bayesian TPE Optimizer test failed with exception: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Bayesian TPE Migration Tests")
    print("=" * 50)
    
    tests = [
        ("Bayesian TPE Optimizer Import", test_bayesian_tpe_import),
        ("HPO Utils Import", test_hpo_utils_import),
        ("Final Parameters Optimization Import", test_final_parameters_import),
        ("Attention Network Optimizer Import", test_attention_network_import),
        ("Bayesian TPE Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All migration tests passed! The Bayesian TPE optimizer is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()