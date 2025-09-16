#!/usr/bin/env python3
"""
Test script for the enhanced confidence optimization logic in final_parameters_optimization.py

This script tests the new optimal confidence calculation features:
1. Confidence level availability validation
2. Multiplicative and logarithmic operations
3. Weighted confidence calculation
4. Different combination methods
"""

import sys
import os
import numpy as np
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.training.steps.backtesting.final_parameters_optimization import FinalParametersOptimizer
    from src.utils.nonlinear_optimization_helpers import NonLinearConfig
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the workspace root directory")
    sys.exit(1)


def test_confidence_availability_validation():
    """Test confidence level availability validation."""
    print("🧪 Testing confidence availability validation...")
    
    # Test case 1: Both confidence levels available
    calibration_results_both = {
        'tactician_models': {'model1': 'test'},
        'analyst_models': {'model1': 'test'}
    }
    
    # Test case 2: Only tactician available
    calibration_results_tactician_only = {
        'tactician_models': {'model1': 'test'}
    }
    
    # Test case 3: Only analyst available
    calibration_results_analyst_only = {
        'analyst_models': {'model1': 'test'}
    }
    
    # Test case 4: Neither available
    calibration_results_none = {}
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    # Test availability checks
    assert optimizer._has_confidence_levels_available(calibration_results_both) == True
    assert optimizer._has_confidence_levels_available(calibration_results_tactician_only) == False
    assert optimizer._has_confidence_levels_available(calibration_results_analyst_only) == False
    assert optimizer._has_confidence_levels_available(calibration_results_none) == False
    
    print("✅ Confidence availability validation tests passed")


def test_confidence_calculation_methods():
    """Test different confidence calculation methods."""
    print("🧪 Testing confidence calculation methods...")
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    # Test parameters
    analyst_threshold = 0.7
    tactician_threshold = 0.8
    tactician_weight = 0.6
    analyst_weight = 0.4
    
    # Test multiplicative method
    multiplicative_conf = optimizer._calculate_multiplicative_confidence(
        analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
    )
    print(f"   Multiplicative confidence: {multiplicative_conf:.4f}")
    assert 0.0 <= multiplicative_conf <= 1.0
    
    # Test logarithmic method
    logarithmic_conf = optimizer._calculate_logarithmic_confidence(
        analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
    )
    print(f"   Logarithmic confidence: {logarithmic_conf:.4f}")
    assert 0.0 <= logarithmic_conf <= 1.0
    
    # Test harmonic method
    harmonic_conf = optimizer._calculate_harmonic_confidence(
        analyst_threshold, tactician_threshold, tactician_weight, analyst_weight
    )
    print(f"   Harmonic confidence: {harmonic_conf:.4f}")
    assert 0.0 <= harmonic_conf <= 1.0
    
    print("✅ Confidence calculation methods tests passed")


def test_optimal_confidence_calculation():
    """Test the main optimal confidence calculation."""
    print("🧪 Testing optimal confidence calculation...")
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    # Test with both confidence levels available
    calibration_results = {
        'tactician_models': {'model1': 'test'},
        'analyst_models': {'model1': 'test'},
        'tactician_confidence_weight': 0.6,
        'analyst_confidence_weight': 0.4
    }
    
    analyst_threshold = 0.7
    tactician_threshold = 0.8
    
    # Test different combination methods
    methods = ['multiplicative', 'logarithmic', 'harmonic', 'weighted_average']
    
    for method in methods:
        calibration_results['confidence_combination_method'] = method
        optimal_conf = optimizer._calculate_optimal_confidence(
            analyst_threshold, tactician_threshold, calibration_results
        )
        
        print(f"   {method} method: {optimal_conf:.4f}")
        assert optimal_conf is not None
        assert 0.0 <= optimal_conf <= 1.0
    
    # Test with missing confidence levels
    calibration_results_no_conf = {}
    optimal_conf_none = optimizer._calculate_optimal_confidence(
        analyst_threshold, tactician_threshold, calibration_results_no_conf
    )
    assert optimal_conf_none is None
    
    print("✅ Optimal confidence calculation tests passed")


def test_confidence_evaluation():
    """Test the enhanced confidence parameter evaluation."""
    print("🧪 Testing confidence parameter evaluation...")
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    # Test parameters with all new confidence parameters
    params = {
        'base_entry_threshold': 0.7,
        'analyst_confidence_threshold': 0.65,
        'tactician_confidence_threshold': 0.8,
        'tactician_confidence_weight': 0.6,
        'analyst_confidence_weight': 0.4,
        'confidence_combination_method': 'multiplicative'
    }
    
    calibration_results = {
        'tactician_models': {'model1': 'test'},
        'analyst_models': {'model1': 'test'}
    }
    
    # Test evaluation
    score = optimizer._evaluate_confidence_params(params, calibration_results)
    print(f"   Confidence evaluation score: {score:.4f}")
    assert score > 0.0
    
    # Test with missing confidence levels
    calibration_results_no_conf = {}
    score_no_conf = optimizer._evaluate_confidence_params(params, calibration_results_no_conf)
    print(f"   Score without confidence levels: {score_no_conf:.4f}")
    assert score_no_conf >= 0.0  # Should still work but with lower score
    
    print("✅ Confidence parameter evaluation tests passed")


def test_confidence_stability():
    """Test confidence stability evaluation."""
    print("🧪 Testing confidence stability evaluation...")
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    calibration_results = {}
    
    # Test good stability (reasonable thresholds)
    analyst_thresh_good = 0.7
    tactician_thresh_good = 0.8
    stability_good = optimizer._evaluate_confidence_stability(
        analyst_thresh_good, tactician_thresh_good, calibration_results
    )
    print(f"   Good stability score: {stability_good:.4f}")
    assert 0.0 <= stability_good <= 1.0
    
    # Test poor stability (thresholds too close)
    analyst_thresh_poor = 0.75
    tactician_thresh_poor = 0.76
    stability_poor = optimizer._evaluate_confidence_stability(
        analyst_thresh_poor, tactician_thresh_poor, calibration_results
    )
    print(f"   Poor stability score: {stability_poor:.4f}")
    assert 0.0 <= stability_poor <= 1.0
    assert stability_poor < stability_good  # Poor should be lower than good
    
    print("✅ Confidence stability evaluation tests passed")


def test_search_space_parameters():
    """Test that the new confidence parameters are in the search space."""
    print("🧪 Testing search space parameters...")
    
    config = {'n_trials': 10, 'timeout': 60}
    optimizer = FinalParametersOptimizer(config)
    
    confidence_space = optimizer.default_search_spaces['confidence']
    
    # Check that all new parameters are present
    required_params = [
        'base_entry_threshold',
        'analyst_confidence_threshold', 
        'tactician_confidence_threshold',
        'tactician_confidence_weight',
        'analyst_confidence_weight',
        'confidence_combination_method'
    ]
    
    for param in required_params:
        assert param in confidence_space, f"Missing parameter: {param}"
        print(f"   ✅ {param}: {confidence_space[param]}")
    
    # Check parameter types
    assert confidence_space['tactician_confidence_weight']['type'] == 'float'
    assert confidence_space['analyst_confidence_weight']['type'] == 'float'
    assert confidence_space['confidence_combination_method']['type'] == 'categorical'
    assert 'multiplicative' in confidence_space['confidence_combination_method']['choices']
    assert 'logarithmic' in confidence_space['confidence_combination_method']['choices']
    
    print("✅ Search space parameters tests passed")


def main():
    """Run all tests."""
    print("🚀 Starting confidence optimization tests...")
    print("=" * 60)
    
    try:
        test_confidence_availability_validation()
        print()
        
        test_confidence_calculation_methods()
        print()
        
        test_optimal_confidence_calculation()
        print()
        
        test_confidence_evaluation()
        print()
        
        test_confidence_stability()
        print()
        
        test_search_space_parameters()
        print()
        
        print("=" * 60)
        print("🎉 All confidence optimization tests passed!")
        print("✅ The enhanced confidence optimization logic is working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)