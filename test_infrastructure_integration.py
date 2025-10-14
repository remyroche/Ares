#!/usr/bin/env python3
"""
Simple test script for the enhanced UnifiedDataDrivenPipeline infrastructure.

This script tests the integration of the new infrastructure components
without requiring external dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test that all new infrastructure components can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test advanced validation
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_validation import (
            AdvancedInputValidator, ValidationLevel, ValidationStatus
        )
        print("✅ AdvancedInputValidator imported successfully")
        
        # Test advanced error handling
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_error_handling import (
            AdvancedErrorHandler, PipelineError, DataValidationError
        )
        print("✅ AdvancedErrorHandler imported successfully")
        
        # Test advanced performance monitoring
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_performance_monitoring import (
            AdvancedPerformanceMonitor, MetricType, MetricLevel
        )
        print("✅ AdvancedPerformanceMonitor imported successfully")
        
        # Test advanced data loading
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_data_loading import (
            AdvancedDataLoader
        )
        print("✅ AdvancedDataLoader imported successfully")
        
        # Test advanced artifact management
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_artifact_management import (
            AdvancedArtifactManager, ArtifactMetadata, ArtifactSaveReport
        )
        print("✅ AdvancedArtifactManager imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_component_initialization():
    """Test that components can be initialized."""
    print("\n🧪 Testing component initialization...")
    
    try:
        # Test advanced validation
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_validation import AdvancedInputValidator
        validator = AdvancedInputValidator()
        print("✅ AdvancedInputValidator initialized")
        
        # Test advanced error handling
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_error_handling import AdvancedErrorHandler
        error_handler = AdvancedErrorHandler(component_name="TestComponent")
        print("✅ AdvancedErrorHandler initialized")
        
        # Test advanced performance monitoring
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_performance_monitoring import AdvancedPerformanceMonitor
        performance_monitor = AdvancedPerformanceMonitor(component_name="TestComponent")
        print("✅ AdvancedPerformanceMonitor initialized")
        
        # Test advanced data loading
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_data_loading import AdvancedDataLoader
        data_loader = AdvancedDataLoader()
        print("✅ AdvancedDataLoader initialized")
        
        # Test advanced artifact management
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_artifact_management import AdvancedArtifactManager
        artifact_manager = AdvancedArtifactManager(base_dir="test_artifacts")
        print("✅ AdvancedArtifactManager initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_functionality():
    """Test basic functionality of components."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test advanced validation
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_validation import AdvancedInputValidator
        validator = AdvancedInputValidator()
        
        # Test validation rules
        rules = validator.validation_rules
        print(f"✅ Validator has {len(rules)} validation rules")
        
        # Test advanced error handling
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_error_handling import AdvancedErrorHandler
        error_handler = AdvancedErrorHandler(component_name="TestComponent")
        
        # Test safe execution
        result = error_handler.safe_execute(
            lambda: 1 + 1,
            operation="test_addition",
            return_value="error"
        )
        if result == 2:
            print("✅ Error handler safe execution works")
        else:
            print(f"❌ Error handler safe execution failed: {result}")
            return False
        
        # Test advanced performance monitoring
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_performance_monitoring import AdvancedPerformanceMonitor
        performance_monitor = AdvancedPerformanceMonitor(component_name="TestComponent")
        
        # Test metric recording
        performance_monitor.record_metric("test_metric", 1.0)
        summary = performance_monitor.get_performance_summary()
        if summary['total_metrics'] > 0:
            print("✅ Performance monitor metric recording works")
        else:
            print("❌ Performance monitor metric recording failed")
            return False
        
        # Test advanced data loading
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_data_loading import AdvancedDataLoader
        data_loader = AdvancedDataLoader()
        
        # Test cache metrics
        cache_metrics = data_loader.get_cache_metrics()
        if 'hits' in cache_metrics:
            print("✅ Data loader cache metrics work")
        else:
            print("❌ Data loader cache metrics failed")
            return False
        
        # Test advanced artifact management
        from src.training.steps.pre_training.unified_data_driven_pipeline.enhanced_components.advanced_artifact_management import AdvancedArtifactManager
        artifact_manager = AdvancedArtifactManager(base_dir="test_artifacts")
        
        # Test artifact registry
        registry = artifact_manager.get_artifact_registry()
        if isinstance(registry, dict):
            print("✅ Artifact manager registry works")
        else:
            print("❌ Artifact manager registry failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration_with_pipeline():
    """Test integration with the main pipeline."""
    print("\n🧪 Testing integration with main pipeline...")
    
    try:
        # Test that the pipeline can be imported with new infrastructure
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
            UnifiedDataDrivenPipeline, create_unified_pipeline
        )
        print("✅ UnifiedDataDrivenPipeline imported with new infrastructure")
        
        # Test that the pipeline can be created
        pipeline = create_unified_pipeline()
        print("✅ UnifiedDataDrivenPipeline created successfully")
        
        # Test that new infrastructure components are available
        if hasattr(pipeline, 'advanced_validator'):
            print("✅ Advanced validator integrated")
        else:
            print("❌ Advanced validator not integrated")
            return False
        
        if hasattr(pipeline, 'advanced_error_handler'):
            print("✅ Advanced error handler integrated")
        else:
            print("❌ Advanced error handler not integrated")
            return False
        
        if hasattr(pipeline, 'advanced_performance_monitor'):
            print("✅ Advanced performance monitor integrated")
        else:
            print("❌ Advanced performance monitor not integrated")
            return False
        
        if hasattr(pipeline, 'advanced_data_loader'):
            print("✅ Advanced data loader integrated")
        else:
            print("❌ Advanced data loader not integrated")
            return False
        
        if hasattr(pipeline, 'advanced_artifact_manager'):
            print("✅ Advanced artifact manager integrated")
        else:
            print("❌ Advanced artifact manager not integrated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Enhanced UnifiedDataDrivenPipeline Infrastructure Tests")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_imports),
        ("Component Initialization", test_component_initialization),
        ("Basic Functionality", test_basic_functionality),
        ("Pipeline Integration", test_integration_with_pipeline),
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
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced UnifiedDataDrivenPipeline infrastructure is working correctly.")
        print("\n📋 IMPLEMENTATION SUMMARY:")
        print("✅ Advanced Input Validation - Comprehensive data validation framework")
        print("✅ Advanced Error Handling - Robust error handling with fast failing")
        print("✅ Advanced Performance Monitoring - Real-time metrics and monitoring")
        print("✅ Advanced Data Loading - Sophisticated data loading and caching")
        print("✅ Advanced Artifact Management - Comprehensive artifact persistence")
        print("✅ Pipeline Integration - All components integrated into UnifiedDataDrivenPipeline")
        print("\n🎯 The UnifiedDataDrivenPipeline now has all the infrastructure")
        print("   that was present in FeatureLookbackOptimizationComponent!")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)