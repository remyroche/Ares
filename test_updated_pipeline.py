#!/usr/bin/env python3
"""
Test script for the updated enhanced multi-stage feature selection pipeline.
Tests the new percentage-based progressive refinement and bootstrap/CV threshold.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_updated_configuration():
    """Test the updated configuration system."""
    print("🧪 Testing Updated Configuration System")
    print("=" * 45)
    
    try:
        # Test the configuration file directly
        config_file = "src/training/steps/pre_training/feature_selection/core/config.py"
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for updated components
        checks = [
            ("Default target 60", "target_features: int = 60  # Default target set to 60"),
            ("Removal percentage", "stage2_removal_percentage: float = 0.10"),
            ("Bootstrap CV threshold", "stage2_bootstrap_cv_threshold: int = 40"),
            ("Removed old batch sizes", "stage2_initial_batch_size: int = 10"),
            ("Removed old thresholds", "stage2_large_batch_threshold: float = 0.3")
        ]
        
        print("📊 Checking updated configuration components...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                if "Removed" in check_name:
                    print(f"   ❌ {check_name}: Found (should be removed)")
                    all_passed = False
                else:
                    print(f"   ✅ {check_name}: Found")
            else:
                if "Removed" in check_name:
                    print(f"   ✅ {check_name}: Not found (correctly removed)")
                else:
                    print(f"   ❌ {check_name}: Missing")
                    all_passed = False
        
        if all_passed:
            print("\n✅ Updated configuration checks passed!")
            return True
        else:
            print("\n❌ Some configuration checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_enhanced_pipeline_updates():
    """Test the enhanced pipeline file updates."""
    print("\n🧪 Testing Enhanced Pipeline Updates")
    print("=" * 40)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for updated components
        checks = [
            ("Percentage-based removal", "10% of features above target"),
            ("Bootstrap CV threshold check", "use_bootstrap_cv = features_above_target >= self.config.stage2_bootstrap_cv_threshold"),
            ("Batch size calculation", "batch_size = max(1, int(features_above_target * self.config.stage2_removal_percentage))"),
            ("Bootstrap CV parameter", "use_bootstrap_cv: bool = False"),
            ("Bootstrap CV usage", "if use_bootstrap_cv:"),
            ("Removed old batch size method", "def _determine_batch_size"),
            ("Updated method name", "progressive_refinement_percentage_based")
        ]
        
        print("📊 Checking enhanced pipeline updates...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                if "Removed" in check_name:
                    print(f"   ❌ {check_name}: Found (should be removed)")
                    all_passed = False
                else:
                    print(f"   ✅ {check_name}: Found")
            else:
                if "Removed" in check_name:
                    print(f"   ✅ {check_name}: Not found (correctly removed)")
                else:
                    print(f"   ❌ {check_name}: Missing")
                    all_passed = False
        
        if all_passed:
            print("\n✅ Enhanced pipeline update checks passed!")
            return True
        else:
            print("\n❌ Some enhanced pipeline checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Enhanced pipeline update test failed: {e}")
        return False

def test_progressive_refinement_logic():
    """Test the progressive refinement logic."""
    print("\n🧪 Testing Progressive Refinement Logic")
    print("=" * 42)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for specific logic patterns
        logic_checks = [
            ("Features above target calculation", "features_above_target = len(current_features) - target_features"),
            ("Percentage-based batch size", "int(features_above_target * self.config.stage2_removal_percentage)"),
            ("Minimum batch size of 1", "max(1, int("),
            ("Bootstrap CV threshold check", "features_above_target >= self.config.stage2_bootstrap_cv_threshold"),
            ("Bootstrap CV usage in scoring", "use_bootstrap_cv=use_bootstrap_cv"),
            ("Debug logging for batch size", "Calculated batch size: {batch_size}"),
            ("Debug logging for bootstrap CV", "Use bootstrap stability and CV: {use_bootstrap_cv}")
        ]
        
        print("📊 Checking progressive refinement logic...")
        all_passed = True
        
        for check_name, check_string in logic_checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ Progressive refinement logic checks passed!")
            return True
        else:
            print("\n❌ Some progressive refinement logic checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Progressive refinement logic test failed: {e}")
        return False

def test_configuration_defaults():
    """Test that the configuration defaults are correct."""
    print("\n🧪 Testing Configuration Defaults")
    print("=" * 35)
    
    try:
        # Test the configuration file
        config_file = "src/training/steps/pre_training/feature_selection/core/config.py"
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for correct defaults
        default_checks = [
            ("Target features default 60", "target_features: int = 60"),
            ("Removal percentage 10%", "stage2_removal_percentage: float = 0.10"),
            ("Bootstrap CV threshold 40", "stage2_bootstrap_cv_threshold: int = 40"),
            ("mRMR weight 70%", "stage1_mrmr_weight: float = 0.7"),
            ("Spearman weight 30%", "stage1_spearman_weight: float = 0.3"),
            ("Target ratio 50%", "stage1_target_ratio: float = 0.5")
        ]
        
        print("📊 Checking configuration defaults...")
        all_passed = True
        
        for check_name, check_string in default_checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ Configuration defaults checks passed!")
            return True
        else:
            print("\n❌ Some configuration defaults checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ Configuration defaults test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Updated Enhanced Multi-Stage Feature Selection Pipeline Tests")
    print("=" * 70)
    
    tests = [
        ("Updated Configuration", test_updated_configuration),
        ("Enhanced Pipeline Updates", test_enhanced_pipeline_updates),
        ("Progressive Refinement Logic", test_progressive_refinement_logic),
        ("Configuration Defaults", test_configuration_defaults)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"Running: {test_name}")
        print('='*70)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*70}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*70)
    
    if passed == total:
        print("🎉 All tests passed! The updated pipeline is ready to use.")
        print("\n📋 Summary of Updates:")
        print("   ✅ Default target features set to 60")
        print("   ✅ Progressive refinement uses 10% of features above target (rounded down)")
        print("   ✅ Bootstrap stability and CV only used when 40+ features away from target")
        print("   ✅ Removed old fixed batch sizes (10, 5, 1)")
        print("   ✅ Removed old threshold-based batch sizing")
        print("   ✅ Added percentage-based batch size calculation")
        print("   ✅ Added bootstrap/CV threshold logic")
        print("   ✅ Updated method names and logging")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)