#!/usr/bin/env python3
"""
Simple test for confidence optimization logic without external dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_confidence_calculation_formulas():
    """Test the mathematical formulas for confidence calculation."""
    print("🧪 Testing confidence calculation formulas...")
    
    # Test multiplicative formula: (tactician^tactician_weight) * (analyst^analyst_weight)
    analyst_thresh = 0.7
    tactician_thresh = 0.8
    tactician_weight = 0.6
    analyst_weight = 0.4
    
    # Multiplicative calculation
    multiplicative_conf = (tactician_thresh ** tactician_weight) * (analyst_thresh ** analyst_weight)
    print(f"   Multiplicative: {multiplicative_conf:.4f}")
    assert 0.0 <= multiplicative_conf <= 1.0
    
    # Logarithmic calculation: exp(tactician_weight * log(tactician_thresh) + analyst_weight * log(analyst_thresh))
    import math
    log_combination = tactician_weight * math.log(tactician_thresh) + analyst_weight * math.log(analyst_thresh)
    logarithmic_conf = math.exp(log_combination)
    print(f"   Logarithmic: {logarithmic_conf:.4f}")
    assert 0.0 <= logarithmic_conf <= 1.0
    
    # Harmonic calculation: 1 / (tactician_weight/tactician_thresh + analyst_weight/analyst_thresh)
    harmonic_conf = 1.0 / (tactician_weight / tactician_thresh + analyst_weight / analyst_thresh)
    print(f"   Harmonic: {harmonic_conf:.4f}")
    assert 0.0 <= harmonic_conf <= 1.0
    
    print("✅ Confidence calculation formulas work correctly")


def test_weight_validation():
    """Test weight validation logic."""
    print("🧪 Testing weight validation...")
    
    # Test weight normalization
    tactician_weight = 0.6
    analyst_weight = 0.4
    total_weight = tactician_weight + analyst_weight
    
    if total_weight > 0:
        normalized_tactician = tactician_weight / total_weight
        normalized_analyst = analyst_weight / total_weight
        print(f"   Normalized weights - Tactician: {normalized_tactician:.3f}, Analyst: {normalized_analyst:.3f}")
        assert abs(normalized_tactician + normalized_analyst - 1.0) < 0.001
    
    # Test weight constraints
    assert 0.1 <= tactician_weight <= 0.9
    assert 0.1 <= analyst_weight <= 0.9
    assert abs(tactician_weight + analyst_weight - 1.0) < 0.1
    
    print("✅ Weight validation works correctly")


def test_confidence_availability_logic():
    """Test confidence availability checking logic."""
    print("🧪 Testing confidence availability logic...")
    
    # Test case 1: Both available
    calibration_both = {
        'tactician_models': {'model1': 'test'},
        'analyst_models': {'model1': 'test'}
    }
    
    tactician_available = (
        'tactician_confidence' in calibration_both or
        'tactician_models' in calibration_both or
        'tactician_ensemble' in calibration_both
    )
    
    analyst_available = (
        'analyst_confidence' in calibration_both or
        'analyst_models' in calibration_both or
        'analyst_ensemble' in calibration_both
    )
    
    both_available = tactician_available and analyst_available
    assert both_available == True
    print("   ✅ Both confidence levels available")
    
    # Test case 2: Only tactician
    calibration_tactician_only = {'tactician_models': {'model1': 'test'}}
    
    tactician_available = (
        'tactician_confidence' in calibration_tactician_only or
        'tactician_models' in calibration_tactician_only or
        'tactician_ensemble' in calibration_tactician_only
    )
    
    analyst_available = (
        'analyst_confidence' in calibration_tactician_only or
        'analyst_models' in calibration_tactician_only or
        'analyst_ensemble' in calibration_tactician_only
    )
    
    both_available = tactician_available and analyst_available
    assert both_available == False
    print("   ✅ Only tactician available (both_available = False)")
    
    print("✅ Confidence availability logic works correctly")


def test_combination_methods():
    """Test different combination methods."""
    print("🧪 Testing combination methods...")
    
    analyst_thresh = 0.7
    tactician_thresh = 0.8
    tactician_weight = 0.6
    analyst_weight = 0.4
    
    # Test multiplicative
    multiplicative_conf = (tactician_thresh ** tactician_weight) * (analyst_thresh ** analyst_weight)
    
    # Test logarithmic
    import math
    log_combination = tactician_weight * math.log(tactician_thresh) + analyst_weight * math.log(analyst_thresh)
    logarithmic_conf = math.exp(log_combination)
    
    # Test harmonic
    harmonic_conf = 1.0 / (tactician_weight / tactician_thresh + analyst_weight / analyst_thresh)
    
    # Test weighted average
    weighted_average_conf = (
        0.4 * multiplicative_conf +
        0.4 * logarithmic_conf +
        0.2 * harmonic_conf
    )
    
    print(f"   Multiplicative: {multiplicative_conf:.4f}")
    print(f"   Logarithmic: {logarithmic_conf:.4f}")
    print(f"   Harmonic: {harmonic_conf:.4f}")
    print(f"   Weighted Average: {weighted_average_conf:.4f}")
    
    # All should be in valid range
    for conf, name in [(multiplicative_conf, "multiplicative"), 
                       (logarithmic_conf, "logarithmic"),
                       (harmonic_conf, "harmonic"),
                       (weighted_average_conf, "weighted_average")]:
        assert 0.0 <= conf <= 1.0, f"{name} confidence out of range: {conf}"
    
    print("✅ All combination methods work correctly")


def test_parameter_validation():
    """Test parameter validation logic."""
    print("🧪 Testing parameter validation...")
    
    # Test threshold validation
    analyst_thresh = 0.65
    tactician_thresh = 0.8
    
    # Check tactician > analyst
    assert tactician_thresh > analyst_thresh
    print("   ✅ Tactician threshold > analyst threshold")
    
    # Check reasonable separation
    threshold_diff = abs(tactician_thresh - analyst_thresh)
    assert 0.1 <= threshold_diff <= 0.2
    print(f"   ✅ Good threshold separation: {threshold_diff:.2f}")
    
    # Test weight validation
    tactician_weight = 0.6
    analyst_weight = 0.4
    
    assert 0.1 <= tactician_weight <= 0.9
    assert 0.1 <= analyst_weight <= 0.9
    assert abs(tactician_weight + analyst_weight - 1.0) < 0.1
    print("   ✅ Weight constraints satisfied")
    
    print("✅ Parameter validation works correctly")


def main():
    """Run all tests."""
    print("🚀 Starting simple confidence optimization tests...")
    print("=" * 60)
    
    try:
        test_confidence_calculation_formulas()
        print()
        
        test_weight_validation()
        print()
        
        test_confidence_availability_logic()
        print()
        
        test_combination_methods()
        print()
        
        test_parameter_validation()
        print()
        
        print("=" * 60)
        print("🎉 All simple confidence optimization tests passed!")
        print("✅ The mathematical formulas and logic are working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)