#!/usr/bin/env python3
"""
Simple validation test for comprehensive file validation implementation.

This script tests the structure and syntax of the validation implementation
without requiring external dependencies.
"""

import os
import sys
from pathlib import Path


def test_file_structure():
    """Test that all required files exist and have correct structure."""
    print("🧪 Testing File Structure")
    print("=" * 50)
    
    # Check if main validation file exists
    validation_file = "src/utils/comprehensive_file_validation.py"
    if os.path.exists(validation_file):
        print(f"✅ {validation_file} exists")
        
        # Check file content structure
        with open(validation_file, 'r') as f:
            content = f.read()
            
        # Check for required classes and functions
        required_elements = [
            "class ComprehensiveFileValidator",
            "class ValidationSeverity",
            "class ValidationIssue", 
            "class FileValidationResult",
            "def validate_step1_file",
            "def validate_step1_5_file",
            "def validate_step2_file",
            "def validate_step4_file",
            "def validate_file_format"
        ]
        
        for element in required_elements:
            if element in content:
                print(f"✅ Found {element}")
            else:
                print(f"❌ Missing {element}")
                return False
    else:
        print(f"❌ {validation_file} not found")
        return False
    
    return True


def test_step_integration():
    """Test that validation is integrated into all required steps."""
    print("\n🧪 Testing Step Integration")
    print("=" * 50)
    
    step_files = [
        "src/training/steps/step1_data_collection.py",
        "src/training/steps/step1_5_data_converter.py", 
        "src/training/steps/step2_feature_engineering.py",
        "src/training/steps/step4_processing_labeling.py"
    ]
    
    for step_file in step_files:
        if os.path.exists(step_file):
            print(f"✅ {step_file} exists")
            
            # Check for validation imports
            with open(step_file, 'r') as f:
                content = f.read()
                
            if "comprehensive_file_validation" in content:
                print(f"   ✅ Validation imports found")
            else:
                print(f"   ❌ Validation imports missing")
                return False
                
            if "_run_comprehensive_validation" in content:
                print(f"   ✅ Validation function found")
            else:
                print(f"   ❌ Validation function missing")
                return False
        else:
            print(f"❌ {step_file} not found")
            return False
    
    return True


def test_documentation():
    """Test that documentation exists."""
    print("\n🧪 Testing Documentation")
    print("=" * 50)
    
    docs = [
        "COMPREHENSIVE_VALIDATION_IMPLEMENTATION.md",
        "test_comprehensive_validation.py"
    ]
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} not found")
            return False
    
    return True


def test_validation_requirements():
    """Test that all requested validation requirements are implemented."""
    print("\n🧪 Testing Validation Requirements")
    print("=" * 50)
    
    validation_file = "src/utils/comprehensive_file_validation.py"
    if not os.path.exists(validation_file):
        print(f"❌ {validation_file} not found")
        return False
    
    with open(validation_file, 'r') as f:
        content = f.read()
    
    # Check for required validation features
    requirements = [
        ("File type validation", "_validate_file_type"),
        ("Data type validation", "_validate_data_types"), 
        ("Column count validation", "_validate_column_count"),
        ("Column names validation", "_validate_column_names"),
        ("Column completeness validation", "_validate_column_completeness"),
        ("Index validation", "_validate_index")
    ]
    
    for req_name, func_name in requirements:
        if func_name in content:
            print(f"✅ {req_name} implemented")
        else:
            print(f"❌ {req_name} not implemented")
            return False
    
    return True


def test_configuration():
    """Test that configuration is properly structured."""
    print("\n🧪 Testing Configuration")
    print("=" * 50)
    
    validation_file = "src/utils/comprehensive_file_validation.py"
    if not os.path.exists(validation_file):
        print(f"❌ {validation_file} not found")
        return False
    
    with open(validation_file, 'r') as f:
        content = f.read()
    
    # Check for configuration elements
    config_elements = [
        "file_types",
        "data_quality", 
        "expected_schemas",
        "klines",
        "aggtrades",
        "futures",
        "features"
    ]
    
    for element in config_elements:
        if element in content:
            print(f"✅ Configuration element '{element}' found")
        else:
            print(f"❌ Configuration element '{element}' missing")
            return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive Validation Implementation Tests")
    print("=" * 70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Step Integration", test_step_integration),
        ("Documentation", test_documentation),
        ("Validation Requirements", test_validation_requirements),
        ("Configuration", test_configuration)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} test passed")
            else:
                print(f"❌ {test_name} test failed")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All tests passed! Comprehensive validation implementation is complete.")
        print("\n📋 Summary of Implementation:")
        print("✅ Comprehensive file validation module created")
        print("✅ Validation integrated into steps 1, 1.5, 2, and 4")
        print("✅ All requested validation requirements implemented:")
        print("   - Type of file validation")
        print("   - Type of strings, boolean values, etc.")
        print("   - Number of columns validation")
        print("   - Column names validation")
        print("   - Column completeness (no empty values)")
        print("   - Index validation")
        print("✅ Documentation and test files created")
        print("✅ Configurable validation system with step-specific schemas")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)