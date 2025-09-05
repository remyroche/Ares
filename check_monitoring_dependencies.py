#!/usr/bin/env python3
"""
Dependency Checker for Comprehensive Monitoring System

This script checks all dependencies and imports for the monitoring system.
"""

import sys
import importlib
from pathlib import Path


def check_builtin_modules():
    """Check built-in Python modules."""
    print("🔍 Checking built-in Python modules...")
    
    builtin_modules = [
        'asyncio', 'datetime', 'functools', 'inspect', 'json', 'logging',
        'os', 'pathlib', 're', 'sys', 'threading', 'time', 'traceback',
        'contextlib', 'enum', 'dataclasses', 'typing'
    ]
    
    all_available = True
    for module in builtin_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            all_available = False
    
    return all_available


def check_optional_modules():
    """Check optional Python modules."""
    print("🔍 Checking optional Python modules...")
    
    optional_modules = [
        ('psutil', 'psutil'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('structlog', 'structlog'),
        ('prometheus_client', 'prometheus_client')
    ]
    
    available_modules = []
    for module_name, import_name in optional_modules:
        try:
            importlib.import_module(import_name)
            print(f"  ✅ {module_name}")
            available_modules.append(module_name)
        except ImportError:
            print(f"  ⚠️ {module_name} (optional)")
    
    return available_modules


def check_monitoring_imports():
    """Check monitoring system imports."""
    print("🔍 Checking monitoring system imports...")
    
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    monitoring_imports = [
        ('src.utils.function_call_monitor', 'Function call monitor'),
        ('src.utils.function_validation_framework', 'Function validation framework'),
        ('src.utils.enhanced_error_handler', 'Enhanced error handler'),
        ('src.utils.pipeline_standards', 'Pipeline standards'),
        ('src.utils.logger', 'Logger system')
    ]
    
    all_available = True
    for module_path, description in monitoring_imports:
        try:
            importlib.import_module(module_path)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            all_available = False
    
    return all_available


def check_step01_imports():
    """Check step01 monitoring imports."""
    print("🔍 Checking step01 monitoring imports...")
    
    step01_imports = [
        ('src.training.steps.data_collection.step01_enhanced_with_monitoring', 'Enhanced step01 monitoring'),
        ('src.training.steps.data_collection.step01_comprehensive_monitoring', 'Comprehensive step01 monitoring')
    ]
    
    all_available = True
    for module_path, description in step01_imports:
        try:
            importlib.import_module(module_path)
            print(f"  ✅ {description}")
        except ImportError as e:
            print(f"  ❌ {description}: {e}")
            all_available = False
    
    return all_available


def test_monitoring_functionality():
    """Test basic monitoring functionality."""
    print("🧪 Testing monitoring functionality...")
    
    try:
        # Test function call monitor
        from src.utils.function_call_monitor import get_function_call_monitor, monitor_basic
        monitor = get_function_call_monitor()
        
        @monitor_basic
        def test_function():
            return "test"
        
        result = test_function()
        print(f"  ✅ Function call monitoring: {result}")
        
        # Test validation framework
        from src.utils.function_validation_framework import get_function_validator, validate_function_entry
        validator = get_function_validator()
        
        @validate_function_entry('data_collection')
        def test_validation(symbol: str, exchange: str):
            return True
        
        result = test_validation("ETHUSDT", "BINANCE")
        print(f"  ✅ Function validation: {result}")
        
        # Test error handler
        from src.utils.enhanced_error_handler import get_error_handler, handle_errors_basic
        error_handler = get_error_handler()
        
        @handle_errors_basic
        def test_error_handling():
            return "success"
        
        result = test_error_handling()
        print(f"  ✅ Error handling: {result}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Monitoring functionality test failed: {e}")
        return False


def check_file_structure():
    """Check if all required files exist."""
    print("🔍 Checking file structure...")
    
    required_files = [
        'src/utils/function_call_monitor.py',
        'src/utils/function_validation_framework.py',
        'src/utils/enhanced_error_handler.py',
        'src/training/steps/data_collection/step01_enhanced_with_monitoring.py',
        'src/training/steps/data_collection/step01_comprehensive_monitoring.py',
        'test_comprehensive_step01_monitoring.py',
        'STEP01_COMPREHENSIVE_MONITORING_README.md',
        'requirements_monitoring.txt',
        'setup_monitoring.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_exist = False
    
    return all_exist


def generate_dependency_report():
    """Generate a comprehensive dependency report."""
    print("📊 Generating dependency report...")
    
    report = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'builtin_modules': check_builtin_modules(),
        'optional_modules': check_optional_modules(),
        'monitoring_imports': check_monitoring_imports(),
        'step01_imports': check_step01_imports(),
        'file_structure': check_file_structure(),
        'functionality_test': test_monitoring_functionality()
    }
    
    # Save report
    import json
    with open('monitoring_dependency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("  ✅ Dependency report saved to monitoring_dependency_report.json")
    
    return report


def main():
    """Main dependency check function."""
    print("🔍 Comprehensive Monitoring System Dependency Check")
    print("=" * 60)
    
    # Check all components
    builtin_ok = check_builtin_modules()
    optional_modules = check_optional_modules()
    monitoring_ok = check_monitoring_imports()
    step01_ok = check_step01_imports()
    files_ok = check_file_structure()
    functionality_ok = test_monitoring_functionality()
    
    # Generate report
    report = generate_dependency_report()
    
    # Summary
    print("=" * 60)
    print("📋 DEPENDENCY CHECK SUMMARY")
    print("=" * 60)
    
    print(f"Python Version: {report['python_version']}")
    print(f"Built-in Modules: {'✅ OK' if builtin_ok else '❌ FAILED'}")
    print(f"Optional Modules: {len(optional_modules)} available")
    print(f"Monitoring Imports: {'✅ OK' if monitoring_ok else '❌ FAILED'}")
    print(f"Step01 Imports: {'✅ OK' if step01_ok else '❌ FAILED'}")
    print(f"File Structure: {'✅ OK' if files_ok else '❌ FAILED'}")
    print(f"Functionality Test: {'✅ OK' if functionality_ok else '❌ FAILED'}")
    
    # Overall status
    overall_ok = all([builtin_ok, monitoring_ok, step01_ok, files_ok, functionality_ok])
    
    print("=" * 60)
    if overall_ok:
        print("🎉 ALL DEPENDENCIES ARE SATISFIED!")
        print("✅ The comprehensive monitoring system is ready to use.")
    else:
        print("❌ SOME DEPENDENCIES ARE MISSING!")
        print("⚠️ Please install missing dependencies and fix import issues.")
        print("💡 Run 'python setup_monitoring.py' to install dependencies.")
    
    print("=" * 60)
    
    return overall_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)