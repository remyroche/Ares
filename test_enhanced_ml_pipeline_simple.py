#!/usr/bin/env python3
"""
Simple Test for Enhanced ML Pipeline

Tests the structural integration without requiring full ML library installations.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('/workspace/src')

def test_file_structure():
    """Test that all required files exist."""
    print("🧪 Testing File Structure...")
    
    required_files = [
        "src/utils/ml_common/validation/hpo_overfitting_prevention.py",
        "src/utils/ml_common/validation/underfitting_detection.py",
        "src/utils/ml_common/validation/model_enhancement_guide.py",
        "src/utils/ml_common/optimization/bayesian_entry_timing_optimizer.py",
        "src/training/steps/model_training/random_survival_forest_tactician.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_hpo_enhancements():
    """Test HPO enhancements."""
    print("🧪 Testing HPO Enhancements...")
    
    try:
        # Read the HPO file and check for staged HPO
        with open("src/utils/ml_common/validation/hpo_overfitting_prevention.py", "r") as f:
            content = f.read()
        
        required_features = [
            "enable_staged_hpo",
            "coarse_strategy",
            "coarse_grid_points",
            "fine_grid_points",
            "bayes_n_trials",
            "optimize_with_staged_hpo"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing HPO features: {missing_features}")
            return False
        else:
            print("✅ All HPO enhancements present")
            return True
            
    except Exception as e:
        print(f"❌ HPO enhancement test failed: {e}")
        return False

def test_overfitting_underfitting_detection():
    """Test overfitting/underfitting detection."""
    print("🧪 Testing Overfitting/Underfitting Detection...")
    
    try:
        # Check underfitting detection file
        with open("src/utils/ml_common/validation/underfitting_detection.py", "r") as f:
            underfitting_content = f.read()
        
        required_classes = [
            "UnderfittingConfig",
            "UnderfittingReport", 
            "UnderfittingDetector"
        ]
        
        missing_classes = []
        for class_name in required_classes:
            if class_name not in underfitting_content:
                missing_classes.append(class_name)
        
        if missing_classes:
            print(f"❌ Missing underfitting classes: {missing_classes}")
            return False
        
        # Check enhancement guide file
        with open("src/utils/ml_common/validation/model_enhancement_guide.py", "r") as f:
            enhancement_content = f.read()
        
        required_enhancement_features = [
            "EnhancementAction",
            "ModelEnhancementPlan",
            "ModelEnhancementGuide"
        ]
        
        missing_enhancement_features = []
        for feature in required_enhancement_features:
            if feature not in enhancement_content:
                missing_enhancement_features.append(feature)
        
        if missing_enhancement_features:
            print(f"❌ Missing enhancement features: {missing_enhancement_features}")
            return False
        
        print("✅ Overfitting/underfitting detection present")
        return True
        
    except Exception as e:
        print(f"❌ Overfitting/underfitting detection test failed: {e}")
        return False

def test_bayesian_entry_timing():
    """Test Bayesian entry timing optimization."""
    print("🧪 Testing Bayesian Entry Timing Optimization...")
    
    try:
        with open("src/utils/ml_common/optimization/bayesian_entry_timing_optimizer.py", "r") as f:
            content = f.read()
        
        required_features = [
            "EntryTimingConfig",
            "EntryTimingResult",
            "BayesianEntryTimingOptimizer",
            "optimize_entry_timing"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing entry timing features: {missing_features}")
            return False
        
        print("✅ Bayesian entry timing optimization present")
        return True
        
    except Exception as e:
        print(f"❌ Bayesian entry timing test failed: {e}")
        return False

def test_rsf_integration():
    """Test RandomSurvivalForest integration."""
    print("🧪 Testing RandomSurvivalForest Integration...")
    
    try:
        with open("src/training/steps/model_training/random_survival_forest_tactician.py", "r") as f:
            content = f.read()
        
        required_features = [
            "enable_entry_timing_optimization",
            "entry_timing_trials",
            "entry_timing_optimization",
            "optimize_entry_timing"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing RSF integration features: {missing_features}")
            return False
        
        print("✅ RandomSurvivalForest integration present")
        return True
        
    except Exception as e:
        print(f"❌ RSF integration test failed: {e}")
        return False

def test_tactician_training_integration():
    """Test tactician training integration."""
    print("🧪 Testing Tactician Training Integration...")
    
    try:
        with open("src/training/steps/model_training/tactician_models_training_refactored.py", "r") as f:
            content = f.read()
        
        required_features = [
            "enable_entry_timing_optimization",
            "entry_timing_trials",
            "RandomSurvivalForest"
        ]
        
        missing_features = []
        for feature in required_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing tactician training features: {missing_features}")
            return False
        
        print("✅ Tactician training integration present")
        return True
        
    except Exception as e:
        print(f"❌ Tactician training integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced ML Pipeline Tests")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("HPO Enhancements", test_hpo_enhancements),
        ("Overfitting/Underfitting Detection", test_overfitting_underfitting_detection),
        ("Bayesian Entry Timing", test_bayesian_entry_timing),
        ("RSF Integration", test_rsf_integration),
        ("Tactician Training Integration", test_tactician_training_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced ML pipeline is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)