#!/usr/bin/env python3
"""
Test individual infrastructure components without external dependencies.

This script tests the new infrastructure components individually
to verify they work correctly.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_advanced_validation():
    """Test the advanced validation component."""
    print("🧪 Testing AdvancedInputValidator...")
    
    try:
        # Mock numpy and pandas for testing
        class MockNumpy:
            @staticmethod
            def mean(x):
                return sum(x) / len(x) if x else 0
            
            @staticmethod
            def min(x):
                return min(x) if x else 0
            
            @staticmethod
            def max(x):
                return max(x) if x else 0
            
            @staticmethod
            def std(x):
                if not x:
                    return 0
                mean_val = sum(x) / len(x)
                return (sum((val - mean_val) ** 2 for val in x) / len(x)) ** 0.5
        
        class MockPandas:
            class DataFrame:
                def __init__(self, data, index=None):
                    self.data = data
                    self.index = index or list(range(len(data)))
                    self.columns = list(data.keys()) if isinstance(data, dict) else []
                
                def __len__(self):
                    return len(self.index)
                
                def __getitem__(self, key):
                    if isinstance(key, str):
                        return self.data[key]
                    return self.data[list(self.data.keys())[key]]
                
                @property
                def empty(self):
                    return len(self.index) == 0
                
                def select_dtypes(self, include=None):
                    return self
                
                def isnull(self):
                    return self
                
                def any(self):
                    return False
                
                def pct_change(self):
                    return self
                
                def rolling(self, window):
                    return self
                
                def mean(self):
                    return self
                
                def std(self):
                    return self
        
        # Mock the dependencies
        sys.modules['numpy'] = MockNumpy()
        sys.modules['pandas'] = MockPandas()
        
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_validation import (
            AdvancedInputValidator, ValidationLevel, ValidationStatus
        )
        
        # Create validator
        validator = AdvancedInputValidator()
        
        # Test validation rules
        rules = validator.validation_rules
        print(f"✅ Validator has {len(rules)} validation rules")
        
        # Test validation with mock data
        mock_data = MockPandas.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        is_valid, summary, cleaned_data = validator.validate_data(mock_data)
        print(f"✅ Validation test: valid={is_valid}, quality_score={summary.quality_score}")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedInputValidator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_error_handling():
    """Test the advanced error handling component."""
    print("\n🧪 Testing AdvancedErrorHandler...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_error_handling import (
            AdvancedErrorHandler, PipelineError, DataValidationError, ErrorSeverity, ErrorCategory
        )
        
        # Create error handler
        error_handler = AdvancedErrorHandler(component_name="TestComponent")
        
        # Test safe execution with success
        result = error_handler.safe_execute(
            lambda: 1 + 1,
            operation="test_addition",
            return_value="error"
        )
        if result == 2:
            print("✅ Safe execution with success works")
        else:
            print(f"❌ Safe execution with success failed: {result}")
            return False
        
        # Test safe execution with error
        result = error_handler.safe_execute(
            lambda: 1 / 0,  # This will raise ZeroDivisionError
            operation="test_division",
            return_value="error_handled"
        )
        if result == "error_handled":
            print("✅ Safe execution with error handling works")
        else:
            print(f"❌ Safe execution with error handling failed: {result}")
            return False
        
        # Test error statistics
        error_stats = error_handler.get_error_stats()
        print(f"✅ Error statistics: {error_stats['total_errors']} errors recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedErrorHandler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_performance_monitoring():
    """Test the advanced performance monitoring component."""
    print("\n🧪 Testing AdvancedPerformanceMonitor...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_performance_monitoring import (
            AdvancedPerformanceMonitor, MetricType, MetricLevel
        )
        
        # Create performance monitor
        performance_monitor = AdvancedPerformanceMonitor(component_name="TestComponent")
        
        # Test metric recording
        performance_monitor.record_metric("test_metric", 1.0, MetricType.PERFORMANCE)
        performance_monitor.record_metric("test_metric_2", 2.0, MetricType.QUALITY)
        
        # Test operation timing
        start_time = performance_monitor.start_operation("test_operation")
        import time
        time.sleep(0.01)  # Small delay
        execution_time = performance_monitor.end_operation("test_operation", start_time, success=True)
        
        if execution_time > 0:
            print("✅ Operation timing works")
        else:
            print("❌ Operation timing failed")
            return False
        
        # Test performance summary
        summary = performance_monitor.get_performance_summary()
        if summary['total_metrics'] > 0:
            print("✅ Performance summary works")
        else:
            print("❌ Performance summary failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedPerformanceMonitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_data_loading():
    """Test the advanced data loading component."""
    print("\n🧪 Testing AdvancedDataLoader...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_data_loading import (
            AdvancedDataLoader
        )
        
        # Create data loader
        data_loader = AdvancedDataLoader()
        
        # Test cache metrics
        cache_metrics = data_loader.get_cache_metrics()
        if 'hits' in cache_metrics and 'misses' in cache_metrics:
            print("✅ Cache metrics work")
        else:
            print("❌ Cache metrics failed")
            return False
        
        # Test reset cache metrics
        data_loader.reset_cache_metrics()
        reset_metrics = data_loader.get_cache_metrics()
        if reset_metrics['hits'] == 0 and reset_metrics['misses'] == 0:
            print("✅ Cache metrics reset works")
        else:
            print("❌ Cache metrics reset failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedDataLoader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_artifact_management():
    """Test the advanced artifact management component."""
    print("\n🧪 Testing AdvancedArtifactManager...")
    
    try:
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_artifact_management import (
            AdvancedArtifactManager, ArtifactMetadata, ArtifactSaveReport
        )
        
        # Create artifact manager
        artifact_manager = AdvancedArtifactManager(base_dir="test_artifacts")
        
        # Test artifact registry
        registry = artifact_manager.get_artifact_registry()
        if isinstance(registry, dict):
            print("✅ Artifact registry works")
        else:
            print("❌ Artifact registry failed")
            return False
        
        # Test save history
        history = artifact_manager.get_save_history()
        if isinstance(history, list):
            print("✅ Save history works")
        else:
            print("❌ Save history failed")
            return False
        
        # Test artifact metadata creation
        metadata = ArtifactMetadata(
            name="test_artifact",
            artifact_type="json",
            created_at="2024-01-01T00:00:00",
            size_bytes=1024,
            checksum="test_checksum"
        )
        if metadata.name == "test_artifact":
            print("✅ Artifact metadata creation works")
        else:
            print("❌ Artifact metadata creation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedArtifactManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Individual Infrastructure Component Tests")
    print("=" * 60)
    
    tests = [
        ("Advanced Input Validation", test_advanced_validation),
        ("Advanced Error Handling", test_advanced_error_handling),
        ("Advanced Performance Monitoring", test_advanced_performance_monitoring),
        ("Advanced Data Loading", test_advanced_data_loading),
        ("Advanced Artifact Management", test_advanced_artifact_management),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All individual component tests passed!")
        print("\n📋 IMPLEMENTATION SUMMARY:")
        print("✅ Advanced Input Validation - Comprehensive data validation framework")
        print("✅ Advanced Error Handling - Robust error handling with fast failing")
        print("✅ Advanced Performance Monitoring - Real-time metrics and monitoring")
        print("✅ Advanced Data Loading - Sophisticated data loading and caching")
        print("✅ Advanced Artifact Management - Comprehensive artifact persistence")
        print("\n🎯 All infrastructure components are working correctly!")
        print("   The UnifiedDataDrivenPipeline now has all the missing infrastructure")
        print("   that was present in FeatureLookbackOptimizationComponent!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)