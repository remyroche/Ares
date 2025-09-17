#!/usr/bin/env python3
"""
Test script to verify that the feature lookback optimization fixes work correctly.
This tests the critical fixes and improvements made to the codebase.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import traceback

# Add the source directory to the path
sys.path.insert(0, '/workspace/src')

def create_test_data(n_rows: int = 1000) -> pd.DataFrame:
    """Create test market data."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')
    
    # Generate realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_rows):
        change = np.random.normal(0, 0.02) * prices[-1]
        new_price = max(1.0, prices[-1] + change)  # Ensure positive prices
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Create OHLC from prices
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_rows))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_rows))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_rows),
        'returns': np.concatenate([[0], np.diff(prices) / prices[:-1]])
    })
    
    return data

def test_constants_import():
    """Test that constants can be imported correctly."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.constants import (
            OPTIMIZATION_CONSTANTS, PERFORMANCE_CONSTANTS, VALIDATION_CONSTANTS
        )
        print("✅ Constants imported successfully")
        print(f"   - Default lookback range: {OPTIMIZATION_CONSTANTS.DEFAULT_MIN_LOOKBACK}-{OPTIMIZATION_CONSTANTS.DEFAULT_MAX_LOOKBACK}")
        print(f"   - Memory limit: {PERFORMANCE_CONSTANTS.DEFAULT_MEMORY_LIMIT_GB}GB")
        print(f"   - Required columns: {VALIDATION_CONSTANTS.REQUIRED_OHLCV_COLUMNS}")
        return True
    except Exception as e:
        print(f"❌ Constants import failed: {e}")
        return False

def test_optimization_strategy():
    """Test the optimization strategy abstraction."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.optimization_strategy import (
            OptimizationStrategyFactory, OptimizationMethod, GridSearchStrategy
        )
        
        # Test factory
        strategy = OptimizationStrategyFactory.create_strategy(OptimizationMethod.GRID_SEARCH)
        print("✅ Optimization strategy factory works")
        
        # Test with data
        test_data = create_test_data(500)
        
        # Validate inputs
        is_valid, error_msg = strategy.validate_inputs(test_data, 'close', 'returns')
        if is_valid:
            print("✅ Input validation passed")
        else:
            print(f"❌ Input validation failed: {error_msg}")
            return False
        
        # Test optimization (quick test with small grid)
        strategy.update_config({'grid_size': 5, 'max_lookback': 20})
        result = strategy.optimize(test_data, 'close', 'returns')
        
        print(f"✅ Optimization completed:")
        print(f"   - Best lookback: {result.best_lookback_period}")
        print(f"   - Best score: {result.best_score:.4f}")
        print(f"   - Trials: {result.total_trials}")
        print(f"   - Time: {result.optimization_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization strategy test failed: {e}")
        traceback.print_exc()
        return False

def test_mrmr_optimizer_fixes():
    """Test that MRMR optimizer fixes work."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.mrmr_lookback_optimizer import (
            MRMRLookbackOptimizer, LookbackOptimizationConfig, optimize_lookback_periods
        )
        
        # Test class name fix
        config = LookbackOptimizationConfig(
            min_lookback=5,
            max_lookback=20,
            tpe_trials=5  # Small number for testing
        )
        
        optimizer = MRMRLookbackOptimizer(config)
        print("✅ MRMRLookbackOptimizer instantiated correctly")
        
        # Test the convenience function (this was the critical fix)
        test_data = create_test_data(200)
        
        try:
            # This should not raise an error about BayesianLookbackOptimizer anymore
            result = optimize_lookback_periods(
                data=test_data,
                feature_name='close',
                target_column='returns',
                config=config
            )
            print("✅ optimize_lookback_periods function works (class name fix verified)")
            return True
        except Exception as e:
            print(f"❌ optimize_lookback_periods failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ MRMR optimizer test failed: {e}")
        traceback.print_exc()
        return False

def test_memory_management():
    """Test memory management improvements."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.monitoring_metrics import (
            MonitoringMetrics, MetricType, MetricLevel
        )
        
        # Create monitor with small limits for testing
        monitor = MonitoringMetrics("TestComponent")
        monitor.max_metrics_memory = 100  # Small limit for testing
        monitor.cleanup_interval = 10
        
        print("✅ MonitoringMetrics instantiated")
        
        # Add many metrics to trigger cleanup
        for i in range(150):  # More than max_metrics_memory
            monitor.record_metric(
                name=f"test_metric_{i}",
                value=i,
                metric_type=MetricType.PERFORMANCE
            )
        
        # Check that cleanup occurred
        if len(monitor.metrics) <= monitor.max_metrics_memory:
            print(f"✅ Memory cleanup working: {len(monitor.metrics)} metrics retained")
            return True
        else:
            print(f"❌ Memory cleanup failed: {len(monitor.metrics)} metrics (should be <= {monitor.max_metrics_memory})")
            return False
            
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test standardized error handling."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.feature_lookback_optimization import (
            StandardizedErrorHandler
        )
        import logging
        
        # Create a test logger
        logger = logging.getLogger('test_logger')
        handler = StandardizedErrorHandler(logger, 'TestComponent')
        
        print("✅ StandardizedErrorHandler instantiated")
        
        # Test error handling
        test_error = ValueError("Test error")
        result = handler.handle_error(test_error, "test_operation", return_value="fallback")
        
        if result == "fallback":
            print("✅ Error handling returns correct fallback value")
        else:
            print(f"❌ Error handling returned unexpected value: {result}")
            return False
        
        # Test warning handling
        handler.handle_warning("Test warning", "test_operation")
        print("✅ Warning handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_data_validation_improvements():
    """Test data validation improvements."""
    try:
        from training.steps.market_analysis.feature_lookback_optimization.validation_framework import (
            ValidationFramework
        )
        
        framework = ValidationFramework()
        print("✅ ValidationFramework instantiated")
        
        # Test with good data
        good_data = create_test_data(100)
        is_valid, results, fixed_data = framework.validate_data(good_data)
        
        if is_valid:
            print("✅ Good data validation passed")
        else:
            print(f"❌ Good data validation failed unexpectedly")
            return False
        
        # Test with bad data (missing columns)
        bad_data = pd.DataFrame({'price': [1, 2, 3], 'vol': [100, 200, 300]})
        is_valid, results, fixed_data = framework.validate_data(bad_data)
        
        if not is_valid:
            print("✅ Bad data correctly identified as invalid")
        else:
            print("❌ Bad data incorrectly passed validation")
            return False
        
        # Test auto-fix
        if fixed_data is not None and len(fixed_data.columns) > len(bad_data.columns):
            print("✅ Auto-fix functionality working")
        else:
            print("⚠️ Auto-fix may not be working as expected")
        
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("🧪 Running Feature Lookback Optimization Fixes Tests")
    print("=" * 60)
    
    tests = [
        ("Constants Import", test_constants_import),
        ("Optimization Strategy", test_optimization_strategy),
        ("MRMR Optimizer Fixes", test_mrmr_optimizer_fixes),
        ("Memory Management", test_memory_management),
        ("Error Handling", test_error_handling),
        ("Data Validation", test_data_validation_improvements),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed out of {len(results)} tests")
    
    if failed == 0:
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print(f"⚠️ {failed} tests failed. Some fixes may need additional work.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)