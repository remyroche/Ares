#!/usr/bin/env python3
"""
Step06 Import Validation Script

This script validates that all imports and dependencies are working correctly
for the step06 enhanced validation framework.
"""

import sys
import os
import importlib
from pathlib import Path
import traceback


def add_import_paths():
    """Add necessary paths to sys.path."""
    current_dir = Path(__file__).parent
    src_dir = current_dir / "src"
    training_steps_dir = src_dir / "training" / "steps"
    
    paths_to_add = [str(src_dir), str(training_steps_dir)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"✅ Added to path: {path}")


def test_core_imports():
    """Test core Python imports."""
    print("🔍 Testing core Python imports...")
    
    core_modules = [
        "pandas", "numpy", "sklearn", "logging", "asyncio", "json",
        "datetime", "pathlib", "typing", "dataclasses", "enum",
        "threading", "contextlib", "functools", "inspect", "time",
        "traceback", "collections", "sys", "os"
    ]
    
    failed_imports = []
    
    for module in core_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0


def test_step06_validation_framework():
    """Test step06 validation framework imports."""
    print("\n🔍 Testing step06 validation framework imports...")
    
    try:
        from step06_enhanced_validation_framework import (
            step06_function_validator,
            step06_function_tracker,
            step06_validation_context,
            get_step06_validation_summary,
            reset_step06_validation_tracking,
            ValidationLevel,
            FunctionStatus,
            FunctionCallContext,
            FunctionCallReport,
            Step06Validator,
            Step06Reporter
        )
        print("✅ All step06 validation framework components imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import step06 validation framework: {e}")
        traceback.print_exc()
        return False


def test_step06_components():
    """Test step06 component imports."""
    print("\n🔍 Testing step06 component imports...")
    
    components = [
        ("market_analysis.step06_feature_engineering", "FeatureInteractionEngine"),
        ("step06_labeling_components.optimized_triple_barrier_labeling", "OptimizedTripleBarrierLabeling"),
        ("data_collection.feature_engineering.step06_feature_engineering", "FeatureEngineeringStep")
    ]
    
    successful_imports = 0
    
    for module_path, class_name in components:
        try:
            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)
            print(f"✅ {module_path}.{class_name}")
            successful_imports += 1
        except ImportError as e:
            print(f"❌ {module_path}.{class_name}: {e}")
        except AttributeError as e:
            print(f"❌ {module_path}.{class_name}: {e}")
        except Exception as e:
            print(f"❌ {module_path}.{class_name}: {e}")
    
    return successful_imports == len(components)


def test_validation_orchestrator():
    """Test validation orchestrator imports."""
    print("\n🔍 Testing validation orchestrator imports...")
    
    try:
        from step06_validation_orchestrator import (
            Step06ValidationOrchestrator,
            run_step06_comprehensive_validation
        )
        print("✅ Validation orchestrator imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import validation orchestrator: {e}")
        traceback.print_exc()
        return False




def test_validation_levels():
    """Test validation levels."""
    print("\n🔍 Testing validation levels...")
    
    try:
        from step06_enhanced_validation_framework import ValidationLevel
        
        levels = [ValidationLevel.BASIC, ValidationLevel.DETAILED, ValidationLevel.COMPREHENSIVE]
        level_values = [level.value for level in levels]
        
        expected_values = ["basic", "detailed", "comprehensive"]
        
        if level_values == expected_values:
            print(f"✅ Validation levels working: {level_values}")
            return True
        else:
            print(f"❌ Validation levels mismatch: expected {expected_values}, got {level_values}")
            return False
            
    except Exception as e:
        print(f"❌ Validation levels test failed: {e}")
        return False


def test_function_status():
    """Test function status enum."""
    print("\n🔍 Testing function status...")
    
    try:
        from step06_enhanced_validation_framework import FunctionStatus
        
        statuses = [
            FunctionStatus.PENDING,
            FunctionStatus.IN_PROGRESS,
            FunctionStatus.COMPLETED,
            FunctionStatus.FAILED,
            FunctionStatus.TIMEOUT
        ]
        
        status_values = [status.value for status in statuses]
        expected_values = ["pending", "in_progress", "completed", "failed", "timeout"]
        
        if status_values == expected_values:
            print(f"✅ Function status working: {status_values}")
            return True
        else:
            print(f"❌ Function status mismatch: expected {expected_values}, got {status_values}")
            return False
            
    except Exception as e:
        print(f"❌ Function status test failed: {e}")
        return False


def test_validation_context():
    """Test validation context manager."""
    print("\n🔍 Testing validation context...")
    
    try:
        from step06_enhanced_validation_framework import step06_validation_context
        
        with step06_validation_context("test_function", "test_type"):
            print("✅ Validation context manager working")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation context test failed: {e}")
        traceback.print_exc()
        return False


def test_validation_summary():
    """Test validation summary functions."""
    print("\n🔍 Testing validation summary functions...")
    
    try:
        from step06_enhanced_validation_framework import (
            get_step06_validation_summary,
            reset_step06_validation_tracking
        )
        
        # Test reset function
        reset_step06_validation_tracking()
        print("✅ reset_step06_validation_tracking working")
        
        # Test summary function
        summary = get_step06_validation_summary()
        if isinstance(summary, dict):
            print("✅ get_step06_validation_summary working")
            return True
        else:
            print(f"❌ get_step06_validation_summary returned wrong type: {type(summary)}")
            return False
            
    except Exception as e:
        print(f"❌ Validation summary test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main validation function."""
    print("🚀 Step06 Import Validation")
    print("=" * 50)
    
    # Add import paths
    add_import_paths()
    
    # Run all tests
    tests = [
        ("Core imports", test_core_imports),
        ("Step06 validation framework", test_step06_validation_framework),
        ("Step06 components", test_step06_components),
        ("Validation orchestrator", test_validation_orchestrator),
        ("Validation levels", test_validation_levels),
        ("Function status", test_function_status),
        ("Validation context", test_validation_context),
        ("Validation summary", test_validation_summary)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All imports and dependencies are working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)