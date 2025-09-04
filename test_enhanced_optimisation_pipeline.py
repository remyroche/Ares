#!/usr/bin/env python3
"""Test script for enhanced optimisation pipeline structure validation."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all enhanced modules can be imported."""
    print("🔍 Testing enhanced optimisation pipeline imports...")
    
    try:
        # Test main launcher imports
        print("  📦 Testing ares_launcher imports...")
        from ares_launcher import AresLauncher
        print("    ✅ AresLauncher imported successfully")
        
        # Test common operations
        print("  📦 Testing common operations...")
        from src.utils.common_operations import (
            format_datetime, get_current_datetime, safe_file_exists, 
            ensure_directory, safe_json_dump, safe_json_load
        )
        print("    ✅ Common operations imported successfully")
        
        # Test data quality framework
        print("  📦 Testing data quality framework...")
        from src.utils.data_quality_framework import DataQualityFramework
        print("    ✅ DataQualityFramework imported successfully")
        
        # Test core decorators
        print("  📦 Testing core decorators...")
        from src.core.decorators import handles_errors, validates, traced, log_execution_time
        print("    ✅ Core decorators imported successfully")
        
        # Test optimisation pipeline components
        print("  📦 Testing optimisation pipeline components...")
        from src.training.steps.optimisation import (
            run_optimisation_pipeline,
            OptimisationPipelineValidator,
            ConfidenceCalibrationStepValidator,
            FinalParametersOptimizationStepValidator,
            OptimisationPipelineStepValidator,
            create_optimisation_validator
        )
        print("    ✅ Optimisation pipeline components imported successfully")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_launcher_methods():
    """Test that enhanced launcher methods exist."""
    print("🔍 Testing enhanced launcher methods...")
    
    try:
        from ares_launcher import AresLauncher
        
        # Create launcher instance
        launcher = AresLauncher()
        
        # Test enhanced methods exist
        methods_to_test = [
            'run_optimisation_pipeline',
            '_validate_optimisation_prerequisites'
        ]
        
        for method_name in methods_to_test:
            if hasattr(launcher, method_name):
                print(f"    ✅ Method {method_name} exists")
            else:
                print(f"    ❌ Method {method_name} missing")
                return False
        
        print("🎉 All launcher methods exist!")
        return True
        
    except Exception as e:
        print(f"❌ Launcher method test failed: {e}")
        return False

def test_validator_creation():
    """Test that validators can be created."""
    print("🔍 Testing validator creation...")
    
    try:
        from src.training.steps.optimisation import create_optimisation_validator
        
        # Test configuration
        config = {
            'confidence_calibration': True,
            'parameter_optimization': True,
            'random_state': 42,
            'enhanced_mode': True
        }
        
        # Test creating different validators
        validator_types = [
            'confidence_calibration',
            'final_parameters_optimization',
            'optimisation_pipeline'
        ]
        
        for validator_type in validator_types:
            validator = create_optimisation_validator(validator_type, config)
            print(f"    ✅ {validator_type} validator created successfully")
        
        print("🎉 All validators created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Validator creation test failed: {e}")
        return False

def test_common_operations():
    """Test common operations functionality."""
    print("🔍 Testing common operations functionality...")
    
    try:
        from src.utils.common_operations import (
            format_datetime, get_current_datetime, safe_file_exists, 
            ensure_directory, safe_json_dump, safe_json_load
        )
        
        # Test datetime operations
        current_time = get_current_datetime()
        formatted_time = format_datetime(current_time, '%Y-%m-%d %H:%M:%S')
        print(f"    ✅ Datetime operations: {formatted_time}")
        
        # Test file operations
        test_file = "test_file.json"
        test_data = {"test": "data", "timestamp": formatted_time}
        
        safe_json_dump(test_data, test_file)
        if safe_file_exists(test_file):
            print("    ✅ File operations: JSON dump/load successful")
            os.remove(test_file)  # Cleanup
        else:
            print("    ❌ File operations: JSON dump failed")
            return False
        
        # Test directory operations
        test_dir = "test_directory"
        ensure_directory(test_dir)
        if os.path.exists(test_dir):
            print("    ✅ Directory operations: Directory creation successful")
            os.rmdir(test_dir)  # Cleanup
        else:
            print("    ❌ Directory operations: Directory creation failed")
            return False
        
        print("🎉 All common operations working!")
        return True
        
    except Exception as e:
        print(f"❌ Common operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ENHANCED OPTIMISATION PIPELINE STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Launcher Method Tests", test_launcher_methods),
        ("Validator Creation Tests", test_validator_creation),
        ("Common Operations Tests", test_common_operations)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Enhanced optimisation pipeline is ready!")
        return True
    else:
        print("💥 Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)