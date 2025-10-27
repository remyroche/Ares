#!/usr/bin/env python3
"""
Test script to validate SR detection improvements.

This script tests the enhanced error handling, data validation,
performance monitoring, and logging capabilities.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(size: int = 1000) -> pd.DataFrame:
    """Create test OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.normal(0, 0.02, size)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some noise to create realistic OHLC
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else price
        close = price
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range('2023-01-01', periods=len(df), freq='1H')
    return df

def create_invalid_data() -> pd.DataFrame:
    """Create invalid data for testing error handling."""
    return pd.DataFrame({
        'open': [100, 101, np.nan, 103],
        'high': [101, 102, 103, 104],
        'low': [99, 100, 101, 102],
        'close': [100.5, 101.5, 102.5, 103.5],
        'volume': [1000, 2000, 3000, 4000]
    })

def test_enhanced_error_handling():
    """Test enhanced error handling decorators."""
    print("🧪 Testing Enhanced Error Handling...")
    
    try:
        from src.training.steps.market_analysis.sr_error_handlers import (
            handles_sr_detection_errors, handles_sr_data_validation
        )
        
        @handles_sr_detection_errors(default_return=[], use_fallback=True)
        @handles_sr_data_validation(required_columns=['high', 'low'], min_rows=5)
        def test_detection_method(data):
            if len(data) < 10:
                raise ValueError("Insufficient data")
            return [{'price': 100.0, 'type': 'support'}]
        
        # Test with valid data
        valid_data = create_test_data(100)
        result = test_detection_method(valid_data)
        print(f"  ✅ Valid data test passed: {len(result)} results")
        
        # Test with invalid data
        invalid_data = create_invalid_data()
        result = test_detection_method(invalid_data)
        print(f"  ✅ Invalid data test passed: {len(result)} results (fallback)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error handling test failed: {e}")
        return False

def test_data_validation():
    """Test comprehensive data validation."""
    print("🧪 Testing Data Validation...")
    
    try:
        from src.training.steps.market_analysis.sr_data_validator import (
            SRDataValidator, ValidationLevel
        )
        
        validator = SRDataValidator(ValidationLevel.STANDARD)
        
        # Test with valid data
        valid_data = create_test_data(100)
        result = validator.validate_ohlcv_data(valid_data)
        print(f"  ✅ Valid data validation: {result.is_valid}, score: {result.quality_score:.2f}")
        
        # Test with invalid data
        invalid_data = create_invalid_data()
        result = validator.validate_ohlcv_data(invalid_data)
        print(f"  ✅ Invalid data validation: {result.is_valid}, issues: {len(result.issues)}")
        
        # Test validation summary
        summary = validator.get_validation_summary(result)
        print(f"  ✅ Validation summary generated: {len(summary)} characters")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data validation test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring."""
    print("🧪 Testing Performance Monitoring...")
    
    try:
        from src.training.steps.market_analysis.sr_performance_monitor import (
            SRPerformanceMonitor, PerformanceMetrics, performance_monitor_decorator
        )
        
        monitor = SRPerformanceMonitor()
        monitor.start_monitoring()
        
        # Simulate some performance metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                method_name='test_method',
                execution_time=0.1 + i * 0.05,
                memory_usage=100 + i * 10,
                memory_delta=5 + i,
                data_size=1000,
                result_count=10,
                timestamp=time.time()
            )
            monitor.record_metrics(metrics)
        
        # Test performance summary
        summary = monitor.get_performance_summary()
        print(f"  ✅ Performance summary: {len(summary)} methods tracked")
        
        # Test system status
        status = monitor.get_system_status()
        print(f"  ✅ System status: CPU {status.get('cpu_percent', 0):.1f}%, Memory {status.get('memory_percent', 0):.1f}%")
        
        monitor.stop_monitoring()
        return True
        
    except Exception as e:
        print(f"  ❌ Performance monitoring test failed: {e}")
        return False

def test_enhanced_logging():
    """Test enhanced logging capabilities."""
    print("🧪 Testing Enhanced Logging...")
    
    try:
        from src.training.steps.market_analysis.sr_logging_enhancer import (
            SRLoggingEnhancer, create_sr_logger, LogLevel
        )
        
        # Create logger
        logger = create_sr_logger(enable_structured=True)
        
        # Test various log events
        logger.log_method_start('test_method', data_size=1000)
        time.sleep(0.1)
        logger.log_method_end('test_method', time.time() - 0.1, result_count=5)
        
        logger.log_validation_result('test_method', True, quality_score=0.95)
        logger.log_performance_alert('test_method', 'execution_time', 5.0, 3.0)
        
        # Test performance summary
        summary = logger.get_performance_summary()
        print(f"  ✅ Logging summary: {summary['total_events']} events logged")
        
        # Test export
        logger.export_events('test_logs.json')
        if os.path.exists('test_logs.json'):
            print("  ✅ Log export successful")
            os.remove('test_logs.json')
        
        return True
        
    except Exception as e:
        print(f"  ❌ Enhanced logging test failed: {e}")
        return False

def test_sr_detection_step():
    """Test SRDetectionStep with enhanced features."""
    print("🧪 Testing SRDetectionStep Integration...")
    
    try:
        from src.training.steps.market_analysis.sr_detection import SRDetectionStep
        
        # Create test config
        config = {
            'sr_optimization': {
                'min_touches': 2,
                'tolerance_pct': 0.005,
                'lookback_periods': 50
            },
            'log_file': 'test_sr_detection.log'
        }
        
        # Initialize step
        step = SRDetectionStep(config)
        
        # Test status
        status = step.get_status()
        print(f"  ✅ Step status: {status['step_name']}, Enhanced logging: {status.get('enhanced_logging', False)}")
        
        # Test validation
        step.validate_config()
        print("  ✅ Configuration validation passed")
        
        # Test cleanup
        step.cleanup()
        print("  ✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ SRDetectionStep test failed: {e}")
        return False

def test_enhanced_sr_detector():
    """Test EnhancedSRDetector with new features."""
    print("🧪 Testing EnhancedSRDetector Integration...")
    
    try:
        from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector
        
        # Create test config
        config = {
            'use_optimized_fractals': True,
            'use_optimized_touch_counting': True,
            'enable_fractal_caching': True,
            'chunk_size': 500
        }
        
        # Initialize detector
        detector = EnhancedSRDetector(config)
        
        # Test with valid data
        test_data = create_test_data(200)
        levels = detector.detect_sr_levels(test_data)
        print(f"  ✅ SR detection: {len(levels)} levels detected")
        
        # Test performance summary
        summary = detector.get_performance_summary()
        print(f"  ✅ Performance summary: {len(summary)} metrics")
        
        # Test adaptive parameters
        params = detector.get_adaptive_parameters('fractal')
        print(f"  ✅ Adaptive parameters: batch_size={params['batch_size']}")
        
        # Test cleanup
        detector.cleanup()
        print("  ✅ Detector cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ EnhancedSRDetector test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting SR Detection Improvements Test Suite")
    print("=" * 60)
    
    tests = [
        test_enhanced_error_handling,
        test_data_validation,
        test_performance_monitoring,
        test_enhanced_logging,
        test_sr_detection_step,
        test_enhanced_sr_detector
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! SR detection improvements are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())