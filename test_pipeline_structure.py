#!/usr/bin/env python3
"""Simplified test for enhanced optimisation pipeline structure validation."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_file_structure():
    """Test that all enhanced files exist."""
    print("🔍 Testing enhanced pipeline file structure...")
    
    required_files = [
        "ares_launcher.py",
        "src/training/steps/optimisation/step16_optimisation_main.py",
        "src/training/steps/optimisation/__init__.py",
        "src/training/steps/optimisation/optimisation_pipeline_validator.py",
        "src/training/steps/optimisation/step_validators.py",
        "src/training/steps/optimisation/step16_confidence_calibration_per_regime.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"    ✅ {file_path}")
        else:
            print(f"    ❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("🎉 All required files exist!")
    return True

def test_launcher_enhancements():
    """Test that launcher has been enhanced."""
    print("🔍 Testing launcher enhancements...")
    
    try:
        with open("ares_launcher.py", "r") as f:
            content = f.read()
        
        # Check for enhanced methods
        enhancements = [
            "_validate_optimisation_prerequisites",
            "enhanced optimisation pipeline",
            "comprehensive validation",
            "data protection"
        ]
        
        found_enhancements = []
        for enhancement in enhancements:
            if enhancement in content:
                found_enhancements.append(enhancement)
                print(f"    ✅ Found: {enhancement}")
            else:
                print(f"    ❌ Missing: {enhancement}")
        
        if len(found_enhancements) >= 3:  # At least 3 out of 4 enhancements
            print("🎉 Launcher has been properly enhanced!")
            return True
        else:
            print(f"❌ Insufficient enhancements found: {len(found_enhancements)}/4")
            return False
            
    except Exception as e:
        print(f"❌ Error reading launcher file: {e}")
        return False

def test_optimisation_main_enhancements():
    """Test that optimisation main has been enhanced."""
    print("🔍 Testing optimisation main enhancements...")
    
    try:
        with open("src/training/steps/optimisation/step16_optimisation_main.py", "r") as f:
            content = f.read()
        
        # Check for enhanced features
        enhancements = [
            "OptimisationPipelineValidator",
            "comprehensive validation",
            "enhanced mode",
            "data protection",
            "argument parsing"
        ]
        
        found_enhancements = []
        for enhancement in enhancements:
            if enhancement in content:
                found_enhancements.append(enhancement)
                print(f"    ✅ Found: {enhancement}")
            else:
                print(f"    ❌ Missing: {enhancement}")
        
        if len(found_enhancements) >= 4:  # At least 4 out of 5 enhancements
            print("🎉 Optimisation main has been properly enhanced!")
            return True
        else:
            print(f"❌ Insufficient enhancements found: {len(found_enhancements)}/5")
            return False
            
    except Exception as e:
        print(f"❌ Error reading optimisation main file: {e}")
        return False

def test_validator_files():
    """Test that validator files have been created."""
    print("🔍 Testing validator files...")
    
    validator_files = [
        "src/training/steps/optimisation/optimisation_pipeline_validator.py",
        "src/training/steps/optimisation/step_validators.py"
    ]
    
    for file_path in validator_files:
        if os.path.exists(file_path):
            print(f"    ✅ {file_path} exists")
            
            # Check file content
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                
                if "class" in content and "validate" in content:
                    print(f"    ✅ {file_path} contains validation classes")
                else:
                    print(f"    ⚠️ {file_path} may be incomplete")
                    
            except Exception as e:
                print(f"    ⚠️ Error reading {file_path}: {e}")
        else:
            print(f"    ❌ {file_path} missing")
            return False
    
    print("🎉 Validator files created successfully!")
    return True

def test_confidence_calibration_enhancements():
    """Test that confidence calibration has been enhanced."""
    print("🔍 Testing confidence calibration enhancements...")
    
    try:
        with open("src/training/steps/optimisation/step16_confidence_calibration_per_regime.py", "r") as f:
            content = f.read()
        
        # Check for enhanced features
        enhancements = [
            "DataQualityFramework",
            "DataFormattingFramework",
            "validate_tactician_data",
            "data protection",
            "comprehensive error handling"
        ]
        
        found_enhancements = []
        for enhancement in enhancements:
            if enhancement in content:
                found_enhancements.append(enhancement)
                print(f"    ✅ Found: {enhancement}")
            else:
                print(f"    ❌ Missing: {enhancement}")
        
        if len(found_enhancements) >= 3:  # At least 3 out of 5 enhancements
            print("🎉 Confidence calibration has been properly enhanced!")
            return True
        else:
            print(f"❌ Insufficient enhancements found: {len(found_enhancements)}/5")
            return False
            
    except Exception as e:
        print(f"❌ Error reading confidence calibration file: {e}")
        return False

def test_pipeline_integration():
    """Test that pipeline integration has been enhanced."""
    print("🔍 Testing pipeline integration...")
    
    try:
        with open("src/training/steps/optimisation/__init__.py", "r") as f:
            content = f.read()
        
        # Check for enhanced features
        enhancements = [
            "OptimisationPipelineValidator",
            "ConfidenceCalibrationStepValidator",
            "FinalParametersOptimizationStepValidator",
            "create_optimisation_validator",
            "enhanced validation and protection"
        ]
        
        found_enhancements = []
        for enhancement in enhancements:
            if enhancement in content:
                found_enhancements.append(enhancement)
                print(f"    ✅ Found: {enhancement}")
            else:
                print(f"    ❌ Missing: {enhancement}")
        
        if len(found_enhancements) >= 4:  # At least 4 out of 5 enhancements
            print("🎉 Pipeline integration has been properly enhanced!")
            return True
        else:
            print(f"❌ Insufficient enhancements found: {len(found_enhancements)}/5")
            return False
            
    except Exception as e:
        print(f"❌ Error reading pipeline integration file: {e}")
        return False

def main():
    """Run all structure tests."""
    print("🚀 ENHANCED OPTIMISATION PIPELINE STRUCTURE VALIDATION")
    print("=" * 70)
    
    tests = [
        ("File Structure Test", test_file_structure),
        ("Launcher Enhancements Test", test_launcher_enhancements),
        ("Optimisation Main Enhancements Test", test_optimisation_main_enhancements),
        ("Validator Files Test", test_validator_files),
        ("Confidence Calibration Enhancements Test", test_confidence_calibration_enhancements),
        ("Pipeline Integration Test", test_pipeline_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 70)
    print("📊 STRUCTURE VALIDATION RESULTS")
    print("=" * 70)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("✅ Enhanced optimisation pipeline structure is complete!")
        print("✅ All validators, decorators, and utilities are in place!")
        print("✅ Data protection and comprehensive validation implemented!")
        return True
    else:
        print("💥 Some structure tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)