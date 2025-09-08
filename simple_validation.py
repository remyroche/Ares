#!/usr/bin/env python3
"""
Simple validation script for Step05 utility integration.

This script performs basic validation without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test dependency injection
        from src.training.steps.step05_dependency_injection import (
            Step05DependencyContainer, 
            UtilityConfig, 
            initialize_step05_utilities
        )
        print("✅ Dependency injection imports successful")
        
        # Test step05
        from src.training.steps.step05_optimized_integrated import Step05OptimizedIntegrated
        print("✅ Step05 imports successful")
        
        # Test utility modules
        from src.utils.common_operations import get_current_datetime, format_datetime
        print("✅ Common operations imports successful")
        
        from src.utils.math_validation import safe_divide, validate_positive
        print("✅ Math validation imports successful")
        
        from src.utils.parquet_utils import ParquetUtils
        print("✅ Parquet utils imports successful")
        
        from src.utils.serialization_utils import JSONSerializer, PickleSerializer
        print("✅ Serialization utils imports successful")
        
        from src.utils.data_processing_utils import DataFrameValidator, DataFrameCleaner
        print("✅ Data processing utils imports successful")
        
        from src.utils.m1_gpu_utils import M1GPUManager, M1PerformanceOptimizer
        print("✅ M1 GPU utils imports successful")
        
        from src.utils.m1_memory_optimizer import M1MemoryOptimizer, M1DataManager
        print("✅ M1 memory optimizer imports successful")
        
        from src.utils.m1_cpu_optimizer import M1CPUOptimizer, M1BatchProcessor
        print("✅ M1 CPU optimizer imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_dependency_injection():
    """Test dependency injection container."""
    print("\n🔧 Testing dependency injection...")
    
    try:
        from src.training.steps.step05_dependency_injection import (
            UtilityConfig, 
            initialize_step05_utilities
        )
        
        # Create config
        config = UtilityConfig(
            enable_gpu_optimization=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_math_validation=True,
            enable_data_validation=True,
            enable_serialization=True,
            memory_limit_gb=4.0,
            max_workers=2,
            gpu_memory_threshold=0.7,
            log_level='INFO'
        )
        
        # Initialize container
        container = initialize_step05_utilities(config)
        print("✅ Container initialized successfully")
        
        # Test categories
        categories = [
            'common_operations', 'common_utilities', 'math_validation',
            'parquet_utils', 'serialization_utils', 'data_processing_utils',
            'm1_gpu_utils', 'm1_memory_utils', 'm1_cpu_utils'
        ]
        
        for category in categories:
            if container.has_category(category):
                category_utils = container.get_category(category)
                print(f"✅ {category}: {len(category_utils)} utilities")
            else:
                print(f"❌ {category}: Not found")
                return False
        
        # Test health check
        health_status = container.health_check()
        print(f"✅ Health check: {health_status['overall_health']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dependency injection test failed: {e}")
        return False


def test_utility_functions():
    """Test basic utility functions."""
    print("\n⚙️ Testing utility functions...")
    
    try:
        from src.training.steps.step05_dependency_injection import (
            UtilityConfig, 
            initialize_step05_utilities
        )
        
        config = UtilityConfig()
        container = initialize_step05_utilities(config)
        
        # Test common operations
        common_ops = container.get_category('common_operations')
        current_time = common_ops['datetime_ops']['get_current_datetime']()
        formatted_time = common_ops['datetime_ops']['format_datetime'](current_time)
        print(f"✅ Datetime operations: {formatted_time}")
        
        # Test string operations
        test_string = "Hello World"
        lower_string = common_ops['string_ops']['safe_lower'](test_string)
        upper_string = common_ops['string_ops']['safe_upper'](test_string)
        print(f"✅ String operations: '{lower_string}' -> '{upper_string}'")
        
        # Test math operations
        safe_float = common_ops['math_ops']['safe_float']("123.45", 0.0)
        safe_int = common_ops['math_ops']['safe_int']("123", 0)
        print(f"✅ Math operations: {safe_float}, {safe_int}")
        
        # Test math validation
        math_validation = container.get_category('math_validation')
        safe_divide = math_validation['safe_math_ops']['safe_divide'](10, 2, 0.0)
        print(f"✅ Math validation: 10/2 = {safe_divide}")
        
        # Test validation
        math_validation['validation_ops']['validate_positive'](5.0, "test_value")
        print("✅ Validation operations: Positive validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        return False


def test_step05_initialization():
    """Test Step05 initialization."""
    print("\n🚀 Testing Step05 initialization...")
    
    try:
        from src.training.steps.step05_optimized_integrated import Step05OptimizedIntegrated
        
        # Create minimal config
        config = {
            'SYMBOL': 'BTCUSDT',
            'EXCHANGE': 'binance',
            'TIMEFRAME': '1h',
            'DATA_DIR': '/tmp/test',
            'enable_gpu_optimization': True,
            'enable_memory_optimization': True,
            'enable_cpu_optimization': True,
            'enable_math_validation': True,
            'enable_data_validation': True,
            'enable_serialization': True,
            'memory_limit_gb': 4.0,
            'max_workers': 2,
            'gpu_memory_threshold': 0.7,
            'log_level': 'INFO'
        }
        
        # Initialize Step05
        step = Step05OptimizedIntegrated(config)
        print("✅ Step05 initialized successfully")
        
        # Test utility references
        assert hasattr(step, 'utils'), "Utils container not found"
        assert hasattr(step, 'common_ops'), "Common operations not found"
        assert hasattr(step, 'math_validation'), "Math validation not found"
        assert hasattr(step, 'parquet_utils'), "Parquet utils not found"
        assert hasattr(step, 'serialization_utils'), "Serialization utils not found"
        assert hasattr(step, 'data_processing_utils'), "Data processing utils not found"
        assert hasattr(step, 'm1_gpu_utils'), "M1 GPU utils not found"
        assert hasattr(step, 'm1_memory_utils'), "M1 memory utils not found"
        assert hasattr(step, 'm1_cpu_utils'), "M1 CPU utils not found"
        print("✅ All utility references properly set")
        
        # Test performance metrics
        assert 'gpu_operations' in step.performance_metrics
        assert 'cpu_parallel_operations' in step.performance_metrics
        assert 'math_validation_operations' in step.performance_metrics
        assert 'data_processing_operations' in step.performance_metrics
        assert 'serialization_operations' in step.performance_metrics
        print("✅ Performance metrics include utility tracking")
        
        return True
        
    except Exception as e:
        print(f"❌ Step05 initialization test failed: {e}")
        return False


def main():
    """Main validation function."""
    print("🎯 Step05 Utility Integration Validation")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Dependency Injection", test_dependency_injection),
        ("Utility Functions", test_utility_functions),
        ("Step05 Initialization", test_step05_initialization)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print(f"{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Step05 utility integration is working correctly.")
        return True
    else:
        print(f"⚠️ {total - passed} tests failed. Please check the issues above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)