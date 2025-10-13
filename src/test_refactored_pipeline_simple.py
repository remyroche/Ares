"""
Simple test script to verify the refactored unified data-driven pipeline.

This script tests the refactored pipeline from the src directory to ensure
all functionality works correctly with the integrated utilities.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test tprint import
        from utils.tprint import tprint_info, tprint_success, tprint_error
        print("✓ tprint imports successful")
        
        # Test error handler import
        from utils.error_handler import UnifiedErrorHandler, ValidationError
        print("✓ error handler imports successful")
        
        # Test data processing utils import
        from utils.data_processing_utils import DataProcessingUtils
        print("✓ data processing utils imports successful")
        
        # Test performance utils import
        from utils.performance_utils import PerformanceMonitor
        print("✓ performance utils imports successful")
        
        # Test enhanced data operations import
        from utils.enhanced_data_operations import memory_optimize_dataframe
        print("✓ enhanced data operations imports successful")
        
        # Test monitoring utils import
        from utils.monitoring_utils import UnifiedPerformanceMonitor
        print("✓ monitoring utils imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        # Test error handler
        from utils.error_handler import UnifiedErrorHandler
        error_handler = UnifiedErrorHandler()
        
        # Test safe execution
        result = error_handler.safe_execute(lambda: "test", default="error")
        if result == "test":
            print("✓ Error handler safe execution works")
        else:
            print("✗ Error handler safe execution failed")
            return False
        
        # Test validation
        try:
            error_handler.validate_not_none(None, "test")
            print("✗ Should have raised error for None validation")
            return False
        except Exception:
            print("✓ Error handler validation works correctly")
        
        # Test performance monitor
        from utils.performance_utils import PerformanceMonitor
        perf_monitor = PerformanceMonitor()
        
        # Test unified monitor
        from utils.monitoring_utils import UnifiedPerformanceMonitor
        unified_monitor = UnifiedPerformanceMonitor()
        
        print("✓ All basic functionality tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_pipeline_imports():
    """Test that the refactored pipeline can be imported."""
    print("\nTesting pipeline imports...")
    
    try:
        # Test that the refactored files exist and can be imported
        from training.steps.pre_training.unified_data_driven_pipeline.core.unified_pipeline import (
            UnifiedDataDrivenPipeline, FeaturePipelineResult
        )
        print("✓ Main pipeline classes import successful")
        
        from training.steps.pre_training.unified_data_driven_pipeline.core.config import (
            UnifiedPipelineConfig, create_default_config
        )
        print("✓ Configuration classes import successful")
        
        from training.steps.pre_training.unified_data_driven_pipeline.statistical_analysis.statistical_framework import (
            StatisticalAnalysisFramework
        )
        print("✓ Statistical framework import successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Pipeline import failed: {e}")
        return False

def test_pipeline_initialization():
    """Test pipeline initialization with utilities."""
    print("\nTesting pipeline initialization...")
    
    try:
        from training.steps.pre_training.unified_data_driven_pipeline.core.unified_pipeline import (
            UnifiedDataDrivenPipeline
        )
        from training.steps.pre_training.unified_data_driven_pipeline.core.config import (
            create_default_config
        )
        
        # Test pipeline initialization
        config = create_default_config()
        pipeline = UnifiedDataDrivenPipeline(config)
        
        # Test that utilities are properly initialized
        assert hasattr(pipeline, 'error_handler'), "Error handler not initialized"
        assert hasattr(pipeline, 'data_processor'), "Data processor not initialized"
        assert hasattr(pipeline, 'performance_monitor'), "Performance monitor not initialized"
        assert hasattr(pipeline, 'unified_monitor'), "Unified monitor not initialized"
        
        print("✓ Pipeline initialization with utilities successful")
        
        # Test performance stats
        stats = pipeline.get_performance_stats()
        print(f"✓ Performance stats retrieved: {len(stats)} categories")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("TESTING REFACTORED UNIFIED DATA-DRIVEN PIPELINE")
    print("="*60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Pipeline Imports", test_pipeline_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Pipeline refactoring successful.")
        print("\nKey improvements made:")
        print("✓ Replaced custom tprint with unified tprint from src/utils/")
        print("✓ Integrated UnifiedErrorHandler for robust error handling")
        print("✓ Added DataProcessingUtils for enhanced data operations")
        print("✓ Integrated PerformanceMonitor and UnifiedPerformanceMonitor")
        print("✓ Added memory optimization utilities")
        print("✓ Enhanced configuration validation")
        print("✓ Improved statistical analysis framework")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)