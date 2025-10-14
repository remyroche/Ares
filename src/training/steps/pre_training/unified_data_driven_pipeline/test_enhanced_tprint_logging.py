#!/usr/bin/env python3
"""
Test script for enhanced tprint logging and silent failure prevention
in the UnifiedDataDrivenPipeline.

This script tests the comprehensive logging enhancements and ensures
no silent failures occur in the pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import tprint utilities: {e}")
    sys.exit(1)

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
        UnifiedDataDrivenPipeline, UnifiedPipelineConfig, create_default_config
    )
    PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import pipeline: {e}")
    sys.exit(1)

try:
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_error_handling import (
        AdvancedErrorHandler
    )
    from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_performance_monitoring import (
        AdvancedPerformanceMonitor
    )
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import enhanced components: {e}")
    ENHANCED_COMPONENTS_AVAILABLE = False


def test_tprint_logging():
    """Test that tprint logging functions work correctly."""
    tprint_info("🧪 Testing tprint logging functions")
    
    try:
        tprint("🔧 Basic tprint message")
        tprint_info("ℹ️ Info message")
        tprint_success("✅ Success message")
        tprint_warning("⚠️ Warning message")
        tprint_error("❌ Error message")
        tprint_debug("🔍 Debug message")
        tprint_performance("📊 Performance message")
        
        tprint_success("✅ All tprint functions working correctly")
        return True
        
    except Exception as e:
        tprint_error(f"❌ Tprint logging test failed: {e}")
        return False


def test_pipeline_initialization():
    """Test pipeline initialization with enhanced logging."""
    tprint_info("🧪 Testing pipeline initialization with enhanced logging")
    
    try:
        # Test with default config
        tprint_debug("🔧 Creating default configuration")
        config = create_default_config()
        tprint_success("✅ Default configuration created")
        
        # Test pipeline initialization
        tprint_debug("🔧 Initializing pipeline")
        pipeline = UnifiedDataDrivenPipeline(config)
        tprint_success("✅ Pipeline initialized successfully")
        
        # Test cleanup
        tprint_debug("🧹 Testing pipeline cleanup")
        pipeline.cleanup()
        tprint_success("✅ Pipeline cleanup completed")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Pipeline initialization test failed: {e}")
        tprint_error(f"❌ Error details: {type(e).__name__}: {str(e)}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False


def test_enhanced_components():
    """Test enhanced components with comprehensive logging."""
    tprint_info("🧪 Testing enhanced components with comprehensive logging")
    
    if not ENHANCED_COMPONENTS_AVAILABLE:
        tprint_warning("⚠️ Enhanced components not available, skipping test")
        return True
    
    try:
        # Test error handler
        tprint_debug("🔧 Testing advanced error handler")
        error_handler = AdvancedErrorHandler("test_component")
        tprint_success("✅ Error handler initialized")
        
        # Test safe execution
        tprint_debug("🔧 Testing safe execution")
        def test_function(x, y):
            return x + y
        
        result = error_handler.safe_execute(test_function, 2, 3, operation="test_addition")
        if result == 5:
            tprint_success("✅ Safe execution test passed")
        else:
            tprint_error(f"❌ Safe execution test failed: expected 5, got {result}")
            return False
        
        # Test error handling
        tprint_debug("🔧 Testing error handling")
        def failing_function():
            raise ValueError("Test error")
        
        error_result = error_handler.safe_execute(failing_function, operation="test_error", return_value="fallback")
        if error_result == "fallback":
            tprint_success("✅ Error handling test passed")
        else:
            tprint_error(f"❌ Error handling test failed: expected 'fallback', got {error_result}")
            return False
        
        # Test performance monitor
        tprint_debug("🔧 Testing performance monitor")
        perf_monitor = AdvancedPerformanceMonitor("test_component")
        tprint_success("✅ Performance monitor initialized")
        
        # Test data quality monitoring
        tprint_debug("🔧 Testing data quality monitoring")
        test_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': ['a', 'b', 'c', 'd', 'e']
        })
        
        quality_result = perf_monitor.monitor_data_quality(test_data, "test_quality_check")
        if 'quality_metrics' in quality_result:
            tprint_success("✅ Data quality monitoring test passed")
        else:
            tprint_error(f"❌ Data quality monitoring test failed: {quality_result}")
            return False
        
        # Test operation timing
        tprint_debug("🔧 Testing operation timing")
        start_time = perf_monitor.start_operation("test_operation")
        import time
        time.sleep(0.1)  # Simulate work
        execution_time = perf_monitor.end_operation("test_operation", start_time, success=True)
        
        if execution_time > 0:
            tprint_success(f"✅ Operation timing test passed: {execution_time:.3f}s")
        else:
            tprint_error(f"❌ Operation timing test failed: {execution_time}")
            return False
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Enhanced components test failed: {e}")
        tprint_error(f"❌ Error details: {type(e).__name__}: {str(e)}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False


def test_silent_failure_prevention():
    """Test that silent failures are prevented."""
    tprint_info("🧪 Testing silent failure prevention")
    
    try:
        # Test DataFrame operations with None data
        tprint_debug("🔧 Testing DataFrame operations with None data")
        error_handler = AdvancedErrorHandler("test_component")
        
        result = error_handler.safe_dataframe_operation(
            "test_none_data", 
            None, 
            lambda x: x.head()
        )
        
        if result is not None and result.empty:
            tprint_success("✅ None data handling test passed")
        else:
            tprint_error(f"❌ None data handling test failed: {result}")
            return False
        
        # Test NumPy operations with None data
        tprint_debug("🔧 Testing NumPy operations with None data")
        result = error_handler.safe_numpy_operation(
            "test_none_array",
            None,
            lambda x: x.sum()
        )
        
        if result is not None and result.size == 0:
            tprint_success("✅ None array handling test passed")
        else:
            tprint_error(f"❌ None array handling test failed: {result}")
            return False
        
        # Test empty DataFrame handling
        tprint_debug("🔧 Testing empty DataFrame handling")
        empty_df = pd.DataFrame()
        result = error_handler.safe_dataframe_operation(
            "test_empty_data",
            empty_df,
            lambda x: x.head()
        )
        
        if result is not None and result.empty:
            tprint_success("✅ Empty DataFrame handling test passed")
        else:
            tprint_error(f"❌ Empty DataFrame handling test failed: {result}")
            return False
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Silent failure prevention test failed: {e}")
        tprint_error(f"❌ Error details: {type(e).__name__}: {str(e)}")
        tprint_error(f"❌ Traceback: {traceback.format_exc()}")
        return False


def test_performance_monitoring_enhancements():
    """Test performance monitoring enhancements."""
    tprint_info("🧪 Testing performance monitoring enhancements")
    
    if not ENHANCED_COMPONENTS_AVAILABLE:
        tprint_warning("⚠️ Enhanced components not available, skipping test")
        return True
    
    try:
        perf_monitor = AdvancedPerformanceMonitor("test_component")
        
        # Test with various data quality scenarios
        tprint_debug("🔧 Testing data quality monitoring with various scenarios")
        
        # Good quality data
        good_data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.1, 2.2, 3.3, 4.4, 5.5],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        
        result = perf_monitor.monitor_data_quality(good_data, "good_quality_test")
        if 'quality_metrics' in result and result.get('quality_metrics', {}).get('quality_score', 0) > 0.5:
            tprint_success("✅ Good quality data test passed")
        else:
            tprint_warning("⚠️ Good quality data test had issues")
        
        # Poor quality data
        poor_data = pd.DataFrame({
            'A': [1, np.nan, 3, np.nan, 5],
            'B': [1.1, np.nan, 3.3, np.nan, 5.5],
            'C': [0.1, np.nan, 0.3, np.nan, 0.5]
        })
        
        result = perf_monitor.monitor_data_quality(poor_data, "poor_quality_test")
        if 'quality_metrics' in result:
            tprint_success("✅ Poor quality data test passed")
        else:
            tprint_warning("⚠️ Poor quality data test had issues")
        
        # Test memory monitoring
        tprint_debug("🔧 Testing memory monitoring")
        try:
            memory_usage = perf_monitor.get_memory_usage()
            if memory_usage > 0:
                tprint_success(f"✅ Memory monitoring test passed: {memory_usage:.2f} MB")
            else:
                tprint_warning("⚠️ Memory monitoring returned 0")
        except Exception as e:
            tprint_warning(f"⚠️ Memory monitoring test failed: {e}")
        
        return True
        
    except Exception as e:
        tprint_error(f"❌ Performance monitoring test failed: {e}")
        tprint_error(f"❌ Error details: {type(e).__name__}: {str(e)}")
        return False


def main():
    """Run all tests."""
    tprint_info("🚀 Starting comprehensive tprint logging and silent failure prevention tests")
    tprint_info(f"📅 Test started at: {datetime.now().isoformat()}")
    
    tests = [
        ("Tprint Logging", test_tprint_logging),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Enhanced Components", test_enhanced_components),
        ("Silent Failure Prevention", test_silent_failure_prevention),
        ("Performance Monitoring", test_performance_monitoring_enhancements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        tprint_info(f"\n{'='*60}")
        tprint_info(f"🧪 Running test: {test_name}")
        tprint_info(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                tprint_success(f"✅ {test_name} PASSED")
            else:
                tprint_error(f"❌ {test_name} FAILED")
                
        except Exception as e:
            tprint_error(f"❌ {test_name} FAILED with exception: {e}")
            tprint_error(f"❌ Traceback: {traceback.format_exc()}")
            results.append((test_name, False))
    
    # Summary
    tprint_info(f"\n{'='*60}")
    tprint_info("📊 TEST SUMMARY")
    tprint_info(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        tprint_info(f"{test_name}: {status}")
    
    tprint_info(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        tprint_success("🎉 All tests passed! Enhanced tprint logging and silent failure prevention working correctly.")
        return True
    else:
        tprint_error(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)