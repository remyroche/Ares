#!/usr/bin/env python3
"""
Simple Integration Test for Enhanced Training

This script tests the basic integration without requiring external dependencies.
"""

import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the enhanced training utilities can be imported."""
    print("🧪 Testing Enhanced Training Utilities Imports...")
    
    try:
        # Test enhanced training utilities import
        from src.utils.ml_common.training.enhanced_training_utils import (
            EnhancedTrainingUtils,
            EarlyStoppingConfig,
            PurgedCVConfig,
            OverfittingMonitorConfig,
            RegularizationConfig
        )
        print("✅ Enhanced training utilities imported successfully")
        
        # Test training integration import
        from src.utils.ml_common.training.training_integration import (
            TrainingStepEnhancer,
            TrainingIntegrationConfig
        )
        print("✅ Training integration imported successfully")
        
        # Test quick integration import
        from src.utils.ml_common.training.quick_integration import (
            enhance_training_step,
            enhance_ensemble_training,
            validate_temporal_data
        )
        print("✅ Quick integration imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing File Structure...")
    
    required_files = [
        "src/utils/ml_common/training/enhanced_training_utils.py",
        "src/utils/ml_common/training/training_integration.py",
        "src/utils/ml_common/training/quick_integration.py",
        "src/utils/ml_common/training/integration_examples.py",
        "src/utils/ml_common/training/UPDATE_GUIDE.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def test_analyst_integration():
    """Test that Analyst training has been updated."""
    print("\n🧪 Testing Analyst Training Integration...")
    
    try:
        analyst_file = "src/training/steps/model_training/analyst_models_training_refactored.py"
        
        if not os.path.exists(analyst_file):
            print(f"❌ {analyst_file} not found")
            return False
        
        with open(analyst_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced training utilities import
        if "EnhancedTrainingUtils" in content:
            print("✅ Enhanced training utilities imported in Analyst training")
        else:
            print("❌ Enhanced training utilities not imported in Analyst training")
            return False
        
        # Check for enhanced training initialization
        if "_initialize_enhanced_training_utilities" in content:
            print("✅ Enhanced training initialization method found in Analyst training")
        else:
            print("❌ Enhanced training initialization method not found in Analyst training")
            return False
        
        # Check for enhanced training execution
        if "_execute_enhanced_training" in content:
            print("✅ Enhanced training execution method found in Analyst training")
        else:
            print("❌ Enhanced training execution method not found in Analyst training")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Analyst integration test failed: {e}")
        return False

def test_tactician_integration():
    """Test that Tactician training has been updated."""
    print("\n🧪 Testing Tactician Training Integration...")
    
    try:
        tactician_file = "src/training/steps/model_training/tactician_models_training_refactored.py"
        
        if not os.path.exists(tactician_file):
            print(f"❌ {tactician_file} not found")
            return False
        
        with open(tactician_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced training utilities import
        if "EnhancedTrainingUtils" in content:
            print("✅ Enhanced training utilities imported in Tactician training")
        else:
            print("❌ Enhanced training utilities not imported in Tactician training")
            return False
        
        # Check for enhanced training initialization
        if "_initialize_enhanced_training_utilities" in content:
            print("✅ Enhanced training initialization method found in Tactician training")
        else:
            print("❌ Enhanced training initialization method not found in Tactician training")
            return False
        
        # Check for enhanced training execution
        if "_execute_enhanced_tactician_training" in content:
            print("✅ Enhanced training execution method found in Tactician training")
        else:
            print("❌ Enhanced training execution method not found in Tactician training")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Tactician integration test failed: {e}")
        return False

def test_ensemble_integration():
    """Test that Ensemble training has been updated."""
    print("\n🧪 Testing Ensemble Training Integration...")
    
    try:
        ensemble_file = "src/training/steps/model_training/tactician_ensemble_training.py"
        
        if not os.path.exists(ensemble_file):
            print(f"❌ {ensemble_file} not found")
            return False
        
        with open(ensemble_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced training utilities import
        if "EnhancedTrainingUtils" in content:
            print("✅ Enhanced training utilities imported in Ensemble training")
        else:
            print("❌ Enhanced training utilities not imported in Ensemble training")
            return False
        
        # Check for enhanced training initialization
        if "_initialize_enhanced_training_utilities" in content:
            print("✅ Enhanced training initialization method found in Ensemble training")
        else:
            print("❌ Enhanced training initialization method not found in Ensemble training")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble integration test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Starting Enhanced Training Integration Tests")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_imports,
        test_analyst_integration,
        test_tactician_integration,
        test_ensemble_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced training integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)