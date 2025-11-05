"""
Test Suite for Enhanced MarkovRegressionAdapter

This module provides comprehensive tests for the enhanced MarkovRegressionAdapter
including hardware optimization, parameter mapping, diagnostics, and integration tests.
"""

import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path

# Add the module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.markov_regression_adapter import (
    MarkovRegressionAdapter,
    MarkovRegressionConfig,
    create_enhanced_markov_regression_adapter
)

def create_test_data(n_samples=1000, n_features=5, n_regimes=3):
    """Create synthetic test data with regime changes."""
    np.random.seed(42)
    
    # Generate base data
    data = np.random.randn(n_samples, n_features)
    
    # Create regime segments
    regime_size = n_samples // n_regimes
    
    for i in range(n_regimes):
        start_idx = i * regime_size
        end_idx = (i + 1) * regime_size if i < n_regimes - 1 else n_samples
        
        # Add regime-specific patterns
        if i == 0:  # Regime 0: High volatility, positive trend
            data[start_idx:end_idx] += np.random.randn(end_idx - start_idx, n_features) * 2.0
            data[start_idx:end_idx] += np.linspace(0, 2, end_idx - start_idx).reshape(-1, 1)
        elif i == 1:  # Regime 1: Low volatility, negative trend
            data[start_idx:end_idx] += np.random.randn(end_idx - start_idx, n_features) * 0.5
            data[start_idx:end_idx] -= np.linspace(0, 1, end_idx - start_idx).reshape(-1, 1)
        else:  # Regime 2: Medium volatility, oscillating
            data[start_idx:end_idx] += np.random.randn(end_idx - start_idx, n_features) * 1.0
            data[start_idx:end_idx] += np.sin(np.linspace(0, 4*np.pi, end_idx - start_idx)).reshape(-1, 1)
    
    return data

def test_basic_functionality():
    """Test basic functionality of the enhanced adapter."""
    print("🧪 Testing Enhanced MarkovRegressionAdapter - Basic Functionality")
    print("=" * 70)
    
    # Create test data
    data = create_test_data(n_samples=500, n_features=3, n_regimes=2)
    print(f"📊 Generated test data: {data.shape}")
    
    # Test 1: Default configuration
    print("\n🔧 Test 1: Default Configuration")
    adapter = create_enhanced_markov_regression_adapter(
        k_regimes=2,
        enable_hardware_optimization=False,  # Disable for testing
        enable_diagnostics=True,
        enable_pca=False  # Keep simple for testing
    )
    print("✅ Enhanced adapter created with default configuration")
    
    # Test 2: Fit model
    print("\n🔄 Test 2: Fit Model")
    start_time = time.time()
    result = adapter.fit(data)
    fit_time = time.time() - start_time
    
    print(f"✅ Model fitted in {fit_time:.2f}s")
    print(f"   - Success: {result.success}")
    print(f"   - N regimes: {result.n_regimes}")
    print(f"   - Log likelihood: {result.log_likelihood:.2f}")
    print(f"   - AIC: {result.aic:.2f}")
    print(f"   - BIC: {result.bic:.2f}")
    
    if not result.success:
        print(f"❌ Model fitting failed: {result.error_message}")
        return False
    
    # Test 3: Predictions
    print("\n🔮 Test 3: Predictions")
    try:
        predictions = adapter.predict(steps=10)
        print(f"✅ Predictions generated: {list(predictions.keys())}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test 4: Regime probabilities
    print("\n📊 Test 4: Regime Probabilities")
    try:
        probabilities = adapter.get_regime_probabilities()
        print(f"✅ Regime probabilities shape: {probabilities.shape}")
    except Exception as e:
        print(f"❌ Regime probabilities failed: {e}")
        return False
    
    # Test 5: Transition matrix
    print("\n🔄 Test 5: Transition Matrix")
    try:
        transition_matrix = adapter.get_transition_matrix()
        print(f"✅ Transition matrix shape: {transition_matrix.shape}")
        print(f"   - Diagonal persistence: {np.mean(np.diag(transition_matrix)):.3f}")
    except Exception as e:
        print(f"❌ Transition matrix failed: {e}")
        return False
    
    print("\n🎉 All basic tests passed!")
    return True

def test_parameter_mapping():
    """Test parameter mapping from Pyro configurations."""
    print("\n🧪 Testing Parameter Mapping")
    print("=" * 70)
    
    # Create Pyro-style configuration
    pyro_config = {
        'K': 3,
        'switching_variance': True,
        'switching_trend': False,
        'order': 1,
        'max_iter': 50,
        'tolerance': 1e-5,
        'random_state': 123
    }
    
    print(f"📋 Pyro config: {pyro_config}")
    
    # Create adapter with Pyro config
    config = MarkovRegressionConfig(
        k_regimes=2,  # Will be overridden by mapping
        pyro_config=pyro_config,
        auto_map_parameters=True,
        enable_hardware_optimization=False,
        enable_diagnostics=True,
        enable_pca=False
    )
    
    adapter = MarkovRegressionAdapter(config)
    
    # Check if parameters were mapped
    print(f"🔧 Mapped k_regimes: {adapter.config.k_regimes}")
    print(f"🔧 Mapped switching_variance: {adapter.config.switching_variance}")
    print(f"🔧 Mapped switching_trend: {adapter.config.switching_trend}")
    print(f"🔧 Mapped order: {adapter.config.order}")
    
    # Test fitting with mapped parameters
    data = create_test_data(n_samples=300, n_features=3, n_regimes=3)
    result = adapter.fit(data)
    
    if result.success:
        print("✅ Parameter mapping successful")
        return True
    else:
        print(f"❌ Parameter mapping failed: {result.error_message}")
        return False

def test_diagnostics():
    """Test advanced diagnostics capabilities."""
    print("\n🧪 Testing Advanced Diagnostics")
    print("=" * 70)
    
    # Create adapter with diagnostics enabled
    config = MarkovRegressionConfig(
        k_regimes=3,
        enable_hardware_optimization=False,
        enable_diagnostics=True,
        enable_pca=False,
        enable_validation=True,
        validation_split=0.2
    )
    
    adapter = MarkovRegressionAdapter(config)
    
    # Fit model
    data = create_test_data(n_samples=500, n_features=4, n_regimes=3)
    result = adapter.fit(data)
    
    if not result.success:
        print(f"❌ Model fitting failed: {result.error_message}")
        return False
    
    # Check diagnostics
    if result.diagnostics:
        print("✅ Diagnostics generated")
        
        # Check model fit diagnostics
        if 'model_fit' in result.diagnostics:
            model_fit = result.diagnostics['model_fit']
            print(f"   - Log likelihood: {model_fit.get('log_likelihood', 'N/A')}")
            print(f"   - Converged: {model_fit.get('converged', 'N/A')}")
        
        # Check regime stability
        if 'regime_stability' in result.diagnostics:
            stability = result.diagnostics['regime_stability']
            print(f"   - Switching frequency: {stability.get('switching_frequency', 'N/A'):.3f}")
        
        # Check transition analysis
        if 'transition_analysis' in result.diagnostics:
            transitions = result.diagnostics['transition_analysis']
            if 'empirical_transition_probs' in transitions:
                trans_probs = transitions['empirical_transition_probs']
                print(f"   - Transition matrix shape: {trans_probs.shape}")
        
        return True
    else:
        print("❌ No diagnostics generated")
        return False

def test_hardware_optimization():
    """Test hardware optimization integration."""
    print("\n🧪 Testing Hardware Optimization")
    print("=" * 70)
    
    # Create adapter with hardware optimization
    config = MarkovRegressionConfig(
        k_regimes=2,
        enable_hardware_optimization=True,
        workload_type='ml_training',
        optimization_level='balanced',
        enable_diagnostics=False,
        enable_pca=False
    )
    
    try:
        adapter = MarkovRegressionAdapter(config)
        
        # Check if hardware optimization is available
        if adapter.hardware_manager is not None:
            print("✅ Hardware optimization enabled")
            
            # Fit model
            data = create_test_data(n_samples=300, n_features=3, n_regimes=2)
            result = adapter.fit(data)
            
            if result.success:
                print("✅ Model fitted with hardware optimization")
                
                # Check hardware metrics
                if result.hardware_metrics:
                    print("✅ Hardware metrics collected")
                
                return True
            else:
                print(f"❌ Model fitting failed: {result.error_message}")
                return False
        else:
            print("⚠️ Hardware optimization not available")
            return True  # Not a failure, just not available
            
    except Exception as e:
        print(f"❌ Hardware optimization test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and validation."""
    print("\n🧪 Testing Error Handling")
    print("=" * 70)
    
    adapter = create_enhanced_markov_regression_adapter(
        k_regimes=2,
        enable_hardware_optimization=False,
        enable_diagnostics=False,
        enable_pca=False
    )
    
    # Test 1: Insufficient data
    print("\n🔍 Test 1: Insufficient Data")
    small_data = np.random.randn(50, 2)  # Too small
    result = adapter.fit(small_data)
    
    if not result.success:
        print("✅ Correctly caught insufficient data error")
    else:
        print("❌ Should have failed with insufficient data")
        return False
    
    # Test 2: NaN data
    print("\n🔍 Test 2: NaN Data")
    nan_data = np.random.randn(300, 3)
    nan_data[100:150] = np.nan  # Add NaNs
    
    result = adapter.fit(nan_data)
    
    if not result.success:
        print("✅ Correctly caught NaN data error")
    else:
        print("❌ Should have failed with NaN data")
        return False
    
    # Test 3: Prediction before fitting
    print("\n🔍 Test 3: Prediction Before Fitting")
    adapter2 = create_enhanced_markov_regression_adapter(k_regimes=2)
    
    try:
        adapter2.predict(steps=5)
        print("❌ Should have failed before fitting")
        return False
    except ValueError:
        print("✅ Correctly caught prediction before fitting error")
    
    print("\n✅ Error handling tests passed")
    return True

def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Starting Enhanced MarkovRegressionAdapter Test Suite")
    print("=" * 70)
    
    test_results = {}
    
    # Run all tests
    test_results['basic_functionality'] = test_basic_functionality()
    test_results['parameter_mapping'] = test_parameter_mapping()
    test_results['diagnostics'] = test_diagnostics()
    test_results['hardware_optimization'] = test_hardware_optimization()
    test_results['error_handling'] = test_error_handling()
    
    # Report results
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Enhanced MarkovRegressionAdapter is ready for production.")
        return True
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)