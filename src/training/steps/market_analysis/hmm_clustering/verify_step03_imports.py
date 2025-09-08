from ..standardized_parquet_handler import standardized_parquet_handler
#!/usr/bin/env python3
"""
Import Verification Script for Step03 Enhanced Monitoring System.

This script verifies that all imports and dependencies are correctly
configured for the enhanced monitoring system.
"""

import sys
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import(module_name: str, description: str) -> bool:
    """Test importing a module and report success/failure."""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name} - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {description}: {module_name} - {e}")
        return False

def test_core_decorators():
    """Test core decorator imports."""
    print("\n🔧 Testing Core Decorator Imports...")
    print("=" * 50)
    
    tests = [
        ("src.core.decorators", "Core decorators module"),
        ("src.core.decorators.function_monitor", "Function monitoring decorator"),
        ("src.core.decorators.enhanced_error_handling", "Enhanced error handling"),
        ("src.core.decorators.compose", "Decorator composition utilities"),
        ("src.core.decorators.logging", "Logging decorators"),
        ("src.core.decorators.validate", "Validation decorators"),
        ("src.core.decorators.trace", "Tracing decorators"),
        ("src.core.decorators.errors", "Error handling decorators"),
    ]
    
    results = []
    for module, description in tests:
        results.append(test_import(module, description))
    
    return all(results)

def test_monitoring_components():
    """Test monitoring component imports."""
    print("\n📊 Testing Monitoring Component Imports...")
    print("=" * 50)
    
    tests = [
        ("src.core.reporting", "Reporting module"),
        ("src.core.reporting.step03_execution_reporter", "Step03 execution reporter"),
    ]
    
    results = []
    for module, description in tests:
        results.append(test_import(module, description))
    
    return all(results)

def test_specific_decorators():
    """Test specific decorator imports."""
    print("\n🎯 Testing Specific Decorator Imports...")
    print("=" * 50)
    
    try:
        from src.core.decorators import (
            monitor_step03_functions,
            handle_step03_errors,
            monitor_function_calls,
            handle_errors_enhanced,
            FunctionCallMonitor,
            EnhancedErrorHandler,
        )
        print("✅ All monitoring decorators imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import monitoring decorators: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Error importing monitoring decorators: {e}")
        return False

def test_reporting_components():
    """Test reporting component imports."""
    print("\n📈 Testing Reporting Component Imports...")
    print("=" * 50)
    
    try:
        from src.core.reporting import (
            Step03ExecutionReporter,
            Step03ExecutionReport,
            FunctionCallSummary,
            PerformanceMetrics,
            ErrorAnalysis,
            QualityMetrics,
            ReportFormat,
            ReportLevel,
        )
        print("✅ All reporting components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import reporting components: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Error importing reporting components: {e}")
        return False

def test_step03_imports():
    """Test step03 specific imports."""
    print("\n🚀 Testing Step03 Specific Imports...")
    print("=" * 50)
    
    tests = [
        ("src.training.steps.market_analysis.step03_hmm_clustering", "Main step03 file"),
        ("src.training.steps.market_analysis.hmm_clustering.step03_enhanced_hmm_regime_discovery", "Enhanced HMM regime discovery"),
    ]
    
    results = []
    for module, description in tests:
        results.append(test_import(module, description))
    
    return all(results)

def test_optional_dependencies():
    """Test optional dependencies."""
    print("\n🔍 Testing Optional Dependencies...")
    print("=" * 50)
    
    optional_tests = [
        ("psutil", "System monitoring (psutil)"),
        ("pandas", "Data processing (pandas)"),
        ("numpy", "Numerical computing (numpy)"),
        ("matplotlib", "Plotting (matplotlib) - optional"),
        ("seaborn", "Statistical plotting (seaborn) - optional"),
        ("plotly", "Interactive plotting (plotly) - optional"),
        ("reportlab", "PDF generation (reportlab) - optional"),
        ("jinja2", "HTML templating (jinja2) - optional"),
    ]
    
    results = []
    for module, description in optional_tests:
        results.append(test_import(module, description))
    
    return results

def main():
    """Main verification function."""
    print("🔍 Step03 Enhanced Monitoring System - Import Verification")
    print("=" * 70)
    
    all_tests = [
        ("Core Decorators", test_core_decorators),
        ("Monitoring Components", test_monitoring_components),
        ("Specific Decorators", test_specific_decorators),
        ("Reporting Components", test_reporting_components),
        ("Step03 Imports", test_step03_imports),
        ("Optional Dependencies", test_optional_dependencies),
    ]
    
    results = []
    for test_name, test_func in all_tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports and dependencies are correctly configured!")
        return True
    else:
        print("⚠️ Some imports or dependencies need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)