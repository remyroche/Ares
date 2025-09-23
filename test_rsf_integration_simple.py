#!/usr/bin/env python3
"""
Simple test script for Random Survival Forest integration structure.

This script tests the code structure and integration without requiring
ML libraries to be installed.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_file_structure():
    """Test that all required files exist."""
    files_to_check = [
        "src/training/steps/model_training/random_survival_forest_tactician.py",
        "src/utils/ml_common/config/base_training_config.py",
        "src/training/steps/model_training/tactician_models_training_refactored.py"
    ]
    
    all_exist = True
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_config_integration():
    """Test that RandomSurvivalForest is in the config."""
    try:
        # Read the config file and check for RandomSurvivalForest
        config_file = Path("src/utils/ml_common/config/base_training_config.py")
        if not config_file.exists():
            print("❌ Config file not found")
            return False
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        if "RandomSurvivalForest" in content:
            print("✅ RandomSurvivalForest found in config file")
            return True
        else:
            print("❌ RandomSurvivalForest not found in config file")
            return False
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        return False

def test_tactician_integration():
    """Test that RandomSurvivalForest is integrated in tactician training."""
    try:
        # Read the tactician training file and check for RandomSurvivalForest
        tactician_file = Path("src/training/steps/model_training/tactician_models_training_refactored.py")
        if not tactician_file.exists():
            print("❌ Tactician training file not found")
            return False
        
        with open(tactician_file, 'r') as f:
            content = f.read()
        
        if "RandomSurvivalForest" in content:
            print("✅ RandomSurvivalForest found in tactician training file")
            return True
        else:
            print("❌ RandomSurvivalForest not found in tactician training file")
            return False
    except Exception as e:
        print(f"❌ Tactician integration test failed: {e}")
        return False

def test_rsf_file_structure():
    """Test that the Random Survival Forest file has the expected structure."""
    try:
        rsf_file = Path("src/training/steps/model_training/random_survival_forest_tactician.py")
        if not rsf_file.exists():
            print("❌ Random Survival Forest file not found")
            return False
        
        with open(rsf_file, 'r') as f:
            content = f.read()
        
        # Check for key classes and methods
        required_elements = [
            "class RandomSurvivalForestTactician",
            "class SurvivalAnalysisConfig",
            "class MultiHorizonRandomSurvivalForest",
            "def fit(",
            "def predict(",
            "def _prepare_survival_data(",
            "def _prepare_features("
        ]
        
        all_found = True
        for element in required_elements:
            if element in content:
                print(f"✅ Found {element}")
            else:
                print(f"❌ Missing {element}")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"❌ RSF file structure test failed: {e}")
        return False

def test_model_types_integration():
    """Test that model_types includes RandomSurvivalForest."""
    try:
        tactician_file = Path("src/training/steps/model_training/tactician_models_training_refactored.py")
        if not tactician_file.exists():
            print("❌ Tactician training file not found")
            return False
        
        with open(tactician_file, 'r') as f:
            content = f.read()
        
        # Check for model_types list that includes RandomSurvivalForest
        if 'model_types=["XGBOOST", "LIGHTGBM", "DEEPSCALER_1M", "FINANCIAL_RESNET", "RandomSurvivalForest"]' in content:
            print("✅ RandomSurvivalForest found in model_types list")
            return True
        else:
            print("❌ RandomSurvivalForest not found in model_types list")
            return False
    except Exception as e:
        print(f"❌ Model types integration test failed: {e}")
        return False

def test_hpo_configuration():
    """Test that HPO configuration includes RandomSurvivalForest."""
    try:
        config_file = Path("src/utils/ml_common/config/base_training_config.py")
        if not config_file.exists():
            print("❌ Config file not found")
            return False
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for HPO search spaces for RandomSurvivalForest
        if "'RandomSurvivalForest':" in content and "n_estimators" in content:
            print("✅ RandomSurvivalForest HPO configuration found")
            return True
        else:
            print("❌ RandomSurvivalForest HPO configuration not found")
            return False
    except Exception as e:
        print(f"❌ HPO configuration test failed: {e}")
        return False

def main():
    """Run all tests for Random Survival Forest integration."""
    print("🚀 Testing Random Survival Forest integration structure...")
    print("=" * 70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Config Integration", test_config_integration),
        ("Tactician Integration", test_tactician_integration),
        ("RSF File Structure", test_rsf_file_structure),
        ("Model Types Integration", test_model_types_integration),
        ("HPO Configuration", test_hpo_configuration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n📈 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Random Survival Forest integration structure is correct.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)