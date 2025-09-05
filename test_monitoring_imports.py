#!/usr/bin/env python3
"""
Import Test Script for Comprehensive Monitoring System

This script tests all imports and dependencies to ensure they work correctly.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_import(module_path, description):
    """Test importing a module and report the result."""
    try:
        __import__(module_path)
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: Unexpected error - {e}")
        return False


def test_builtin_modules():
    """Test built-in Python modules."""
    print("🔍 Testing built-in Python modules...")
    
    modules = [
        ('asyncio', 'AsyncIO support'),
        ('datetime', 'Date and time handling'),
        ('functools', 'Function utilities'),
        ('inspect', 'Function inspection'),
        ('json', 'JSON handling'),
        ('logging', 'Logging system'),
        ('os', 'OS interface'),
        ('pathlib', 'Path handling'),
        ('re', 'Regular expressions'),
        ('sys', 'System interface'),
        ('threading', 'Threading support'),
        ('time', 'Time utilities'),
        ('traceback', 'Exception traceback'),
        ('contextlib', 'Context management'),
        ('enum', 'Enumeration support'),
        ('dataclasses', 'Data classes'),
        ('typing', 'Type hints')
    ]
    
    success_count = 0
    for module, description in modules:
        if test_import(module, description):
            success_count += 1
    
    print(f"📊 Built-in modules: {success_count}/{len(modules)} successful")
    return success_count == len(modules)


def test_optional_modules():
    """Test optional Python modules."""
    print("🔍 Testing optional Python modules...")
    
    modules = [
        ('psutil', 'System monitoring'),
        ('pandas', 'Data processing'),
        ('numpy', 'Numerical computing'),
        ('structlog', 'Structured logging'),
        ('prometheus_client', 'Prometheus metrics')
    ]
    
    success_count = 0
    available_modules = []
    
    for module, description in modules:
        if test_import(module, description):
            success_count += 1
            available_modules.append(module)
    
    print(f"📊 Optional modules: {success_count}/{len(modules)} available")
    print(f"📋 Available: {', '.join(available_modules)}")
    return success_count


def test_utils_imports():
    """Test utils module imports."""
    print("🔍 Testing utils module imports...")
    
    modules = [
        ('src.utils.function_call_monitor', 'Function call monitor'),
        ('src.utils.function_validation_framework', 'Function validation framework'),
        ('src.utils.enhanced_error_handler', 'Enhanced error handler'),
        ('src.utils.pipeline_standards', 'Pipeline standards'),
        ('src.utils.logger', 'Logger system')
    ]
    
    success_count = 0
    for module, description in modules:
        if test_import(module, description):
            success_count += 1
    
    print(f"📊 Utils modules: {success_count}/{len(modules)} successful")
    return success_count == len(modules)


def test_step01_imports():
    """Test step01 monitoring imports."""
    print("🔍 Testing step01 monitoring imports...")
    
    modules = [
        ('src.training.steps.data_collection.step01_enhanced_with_monitoring', 'Enhanced step01 monitoring'),
        ('src.training.steps.data_collection.step01_comprehensive_monitoring', 'Comprehensive step01 monitoring')
    ]
    
    success_count = 0
    for module, description in modules:
        if test_import(module, description):
            success_count += 1
    
    print(f"📊 Step01 modules: {success_count}/{len(modules)} successful")
    return success_count == len(modules)


def test_monitoring_components():
    """Test monitoring system components."""
    print("🧪 Testing monitoring system components...")
    
    try:
        # Test function call monitor
        from src.utils.function_call_monitor import (
            get_function_call_monitor, 
            monitor_basic, 
            monitor_standard, 
            monitor_comprehensive
        )
        monitor = get_function_call_monitor()
        print("✅ Function call monitor components")
        
        # Test validation framework
        from src.utils.function_validation_framework import (
            get_function_validator,
            validate_function_entry,
            validate_function_output
        )
        validator = get_function_validator()
        print("✅ Function validation framework components")
        
        # Test error handler
        from src.utils.enhanced_error_handler import (
            get_error_handler,
            handle_errors_basic,
            handle_errors_strict
        )
        error_handler = get_error_handler()
        print("✅ Enhanced error handler components")
        
        # Test comprehensive monitoring
        from src.training.steps.data_collection.step01_comprehensive_monitoring import (
            Step01ComprehensiveMonitoring,
            run_comprehensive_step01
        )
        print("✅ Comprehensive monitoring components")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring components test failed: {e}")
        traceback.print_exc()
        return False


def test_decorator_functionality():
    """Test decorator functionality."""
    print("🧪 Testing decorator functionality...")
    
    try:
        from src.utils.function_call_monitor import monitor_basic
        from src.utils.function_validation_framework import validate_function_entry
        from src.utils.enhanced_error_handler import handle_errors_basic
        
        @monitor_basic
        @validate_function_entry('data_collection')
        @handle_errors_basic
        def test_function(symbol: str, exchange: str) -> bool:
            return True
        
        result = test_function("ETHUSDT", "BINANCE")
        print(f"✅ Decorator functionality: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Decorator functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_async_functionality():
    """Test async functionality."""
    print("🧪 Testing async functionality...")
    
    try:
        import asyncio
        from src.utils.function_call_monitor import monitor_standard
        from src.utils.function_validation_framework import validate_function_entry
        from src.utils.enhanced_error_handler import handle_errors_basic
        
        @monitor_standard
        @validate_function_entry('data_collection')
        @handle_errors_basic
        async def test_async_function(symbol: str, exchange: str) -> bool:
            await asyncio.sleep(0.01)  # Simulate async work
            return True
        
        async def run_test():
            result = await test_async_function("ETHUSDT", "BINANCE")
            return result
        
        result = asyncio.run(run_test())
        print(f"✅ Async functionality: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Async functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test integration between components."""
    print("🧪 Testing component integration...")
    
    try:
        from src.training.steps.data_collection.step01_comprehensive_monitoring import Step01ComprehensiveMonitoring
        
        # Test initialization
        config = {
            'SYMBOL': 'ETHUSDT',
            'EXCHANGE': 'BINANCE',
            'TIMEFRAME': '1m',
            'DATA_DIR': 'test_data'
        }
        
        step = Step01ComprehensiveMonitoring(config)
        print("✅ Step01 comprehensive monitoring initialization")
        
        # Test monitoring instances
        from src.utils.function_call_monitor import get_function_call_monitor
        from src.utils.function_validation_framework import get_function_validator
        from src.utils.enhanced_error_handler import get_error_handler
        
        monitor = get_function_call_monitor()
        validator = get_function_validator()
        error_handler = get_error_handler()
        
        print("✅ Monitoring instances created")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("🧪 Comprehensive Monitoring System Import Test")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Built-in modules", test_builtin_modules),
        ("Optional modules", test_optional_modules),
        ("Utils imports", test_utils_imports),
        ("Step01 imports", test_step01_imports),
        ("Monitoring components", test_monitoring_components),
        ("Decorator functionality", test_decorator_functionality),
        ("Async functionality", test_async_functionality),
        ("Integration", test_integration)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 IMPORT TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed_tests += 1
    
    print("=" * 60)
    print(f"📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL IMPORT TESTS PASSED!")
        print("✅ The comprehensive monitoring system is ready to use.")
    else:
        print("❌ SOME IMPORT TESTS FAILED!")
        print("⚠️ Please check the failed tests and fix import issues.")
    
    print("=" * 60)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)