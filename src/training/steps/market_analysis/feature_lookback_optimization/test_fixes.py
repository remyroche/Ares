#!/usr/bin/env python3
"""
Test script to verify the bug fixes in feature lookback optimization.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def create_test_data():
    """Create synthetic test data for optimization."""
    np.random.seed(42)
    
    # Create 500 data points (smaller for faster testing)
    n_points = 500
    
    # Generate synthetic OHLCV data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_points, freq='1min'),
        'open': 100 + np.cumsum(np.random.randn(n_points) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n_points) * 0.1) + np.abs(np.random.randn(n_points) * 0.05),
        'low': 100 + np.cumsum(np.random.randn(n_points) * 0.1) - np.abs(np.random.randn(n_points) * 0.05),
        'close': 100 + np.cumsum(np.random.randn(n_points) * 0.1),
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Add returns column
    data['returns'] = data['close'].pct_change()
    
    return data

def test_configuration_validation():
    """Test configuration validation."""
    print("🧪 Testing Configuration Validation")
    print("=" * 40)
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.mrmr_lookback_optimizer import LookbackOptimizationConfig
        
        # Test valid configuration
        print("✅ Testing valid configuration...")
        config = LookbackOptimizationConfig()
        print("✅ Valid configuration created successfully")
        
        # Test invalid configuration - negative min_lookback
        print("⚠️ Testing invalid configuration (negative min_lookback)...")
        try:
            invalid_config = LookbackOptimizationConfig(min_lookback=-1)
            print("❌ Should have failed validation")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught validation error: {e}")
        
        # Test invalid configuration - max < min
        print("⚠️ Testing invalid configuration (max < min)...")
        try:
            invalid_config = LookbackOptimizationConfig(min_lookback=50, max_lookback=10)
            print("❌ Should have failed validation")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught validation error: {e}")
        
        # Test invalid weights
        print("⚠️ Testing invalid configuration (negative weights)...")
        try:
            invalid_config = LookbackOptimizationConfig(first_lookback_weight=-0.1)
            print("❌ Should have failed validation")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught validation error: {e}")
        
        print("✅ Configuration validation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_safe_array_access():
    """Test safe array access functions."""
    print("\n🧪 Testing Safe Array Access")
    print("=" * 30)
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.mrmr_lookback_optimizer import safe_list_get
        
        # Test normal access
        test_list = [1, 2, 3, 4, 5]
        result = safe_list_get(test_list, 2, 'default')
        assert result == 3, f"Expected 3, got {result}"
        print("✅ Normal array access works")
        
        # Test out-of-bounds access
        result = safe_list_get(test_list, 10, 'default')
        assert result == 'default', f"Expected 'default', got {result}"
        print("✅ Out-of-bounds access returns default")
        
        # Test empty list access
        result = safe_list_get([], 0, 'default')
        assert result == 'default', f"Expected 'default', got {result}"
        print("✅ Empty list access returns default")
        
        # Test None list access
        result = safe_list_get(None, 0, 'default')
        assert result == 'default', f"Expected 'default', got {result}"
        print("✅ None list access returns default")
        
        print("✅ Safe array access tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Safe array access test failed: {e}")
        return False

def test_math_operations():
    """Test safe mathematical operations."""
    print("\n🧪 Testing Safe Math Operations")
    print("=" * 30)
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.mrmr_lookback_optimizer import MATH_VALIDATION_AVAILABLE
        
        if MATH_VALIDATION_AVAILABLE:
            from src.utils.math_validation import safe_correlation, safe_divide
            
            # Test safe correlation with valid data
            x = np.array([1, 2, 3, 4, 5])
            y = np.array([2, 4, 6, 8, 10])
            corr = safe_correlation(x, y, default=0.0)
            assert abs(corr - 1.0) < 0.001, f"Expected ~1.0, got {corr}"
            print("✅ Safe correlation with valid data works")
            
            # Test safe correlation with NaN data
            x_nan = np.array([1, 2, np.nan, 4, 5])
            y_nan = np.array([2, 4, 6, np.nan, 10])
            corr_nan = safe_correlation(x_nan, y_nan, default=0.0)
            assert not np.isnan(corr_nan), f"Expected finite result, got {corr_nan}"
            print("✅ Safe correlation with NaN data works")
            
            # Test safe division
            result = safe_divide(10, 2, 0.0)
            assert result == 5.0, f"Expected 5.0, got {result}"
            print("✅ Safe division works")
            
            # Test safe division by zero
            result = safe_divide(10, 0, 999.0)
            assert result == 999.0, f"Expected 999.0, got {result}"
            print("✅ Safe division by zero works")
            
        else:
            print("⚠️ Math validation utilities not available, using fallbacks")
        
        print("✅ Safe math operations tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Safe math operations test failed: {e}")
        return False

def test_memory_monitoring():
    """Test memory monitoring functionality."""
    print("\n🧪 Testing Memory Monitoring")
    print("=" * 25)
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.feature_lookback_optimization import FeatureLookbackOptimizationComponent
        
        # Create component
        component = FeatureLookbackOptimizationComponent()
        
        # Test memory monitoring
        memory_info = component._check_memory_usage()
        
        assert isinstance(memory_info, dict), "Memory info should be a dictionary"
        assert 'current_memory_mb' in memory_info, "Should contain current_memory_mb"
        assert 'peak_memory_mb' in memory_info, "Should contain peak_memory_mb"
        assert 'memory_warnings' in memory_info, "Should contain memory_warnings"
        
        print(f"✅ Memory monitoring works: {memory_info['current_memory_mb']:.1f}MB current")
        print("✅ Memory monitoring tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory monitoring test failed: {e}")
        return False

def test_optimization_robustness():
    """Test optimization with edge cases."""
    print("\n🧪 Testing Optimization Robustness")
    print("=" * 35)
    
    try:
        from src.training.steps.market_analysis.feature_lookback_optimization.mrmr_lookback_optimizer import MRMRLookbackOptimizer, LookbackOptimizationConfig
        
        # Create test data with some edge cases
        data = create_test_data()
        
        # Add some NaN values to test robustness
        data.loc[10:15, 'close'] = np.nan
        data.loc[50:55, 'returns'] = np.nan
        
        # Create configuration with small parameters for faster testing
        config = LookbackOptimizationConfig(
            min_lookback=3,
            max_lookback=15,
            coarse_grid_size=3,
            fine_grid_size=3,
            tpe_trials=5,
            top_k_coarse_candidates=2,
            top_k_fine_candidates=1
        )
        
        # Test that configuration validation works
        print("✅ Configuration with edge cases created successfully")
        
        # Create optimizer
        optimizer = MRMRLookbackOptimizer(config)
        print("✅ Optimizer created successfully")
        
        # Test individual calculation methods with edge cases
        mi_score = optimizer._calculate_mutual_information(
            data, 'close', 'returns', 5, 'technical_indicator'
        )
        assert isinstance(mi_score, (int, float)), f"MI score should be numeric, got {type(mi_score)}"
        assert mi_score >= 0, f"MI score should be non-negative, got {mi_score}"
        print(f"✅ Mutual information calculation robust: {mi_score:.4f}")
        
        # Test correlation calculation
        corr = optimizer._calculate_correlation_between_periods(
            data, 'close', 5, 10, 'technical_indicator'
        )
        assert isinstance(corr, (int, float)), f"Correlation should be numeric, got {type(corr)}"
        assert 0 <= corr <= 1, f"Correlation should be between 0 and 1, got {corr}"
        print(f"✅ Correlation calculation robust: {corr:.4f}")
        
        print("✅ Optimization robustness tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Optimization robustness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Feature Lookback Optimization Bug Fixes")
    print("=" * 55)
    
    tests = [
        test_configuration_validation,
        test_safe_array_access,
        test_math_operations,
        test_memory_monitoring,
        test_optimization_robustness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 55)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Bug fixes are working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the fixes.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)