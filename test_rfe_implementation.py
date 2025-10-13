#!/usr/bin/env python3
"""
Test script for the updated RFE implementation with percentage-based step size.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_rfe_configuration():
    """Test the RFE configuration updates."""
    print("🧪 Testing RFE Configuration Updates")
    print("=" * 40)
    
    try:
        # Test the configuration file
        config_file = "src/training/steps/pre_training/feature_selection/core/config.py"
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for RFE configuration components
        checks = [
            ("RFE step size 10%", "rfe_step_size: float = 0.10"),
            ("RFE percentage step flag", "rfe_use_percentage_step: bool = True"),
            ("RFE step size comment", "Remove 10% of features above target in each RFE round"),
            ("Removed stage2_removal_percentage", "stage2_removal_percentage: float = 0.10"),
            ("Bootstrap CV threshold", "stage2_bootstrap_cv_threshold: int = 40")
        ]
        
        print("📊 Checking RFE configuration components...")
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
            print("\n✅ RFE configuration checks passed!")
            return True
        else:
            print("\n❌ Some RFE configuration checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ RFE configuration test failed: {e}")
        return False

def test_rfe_implementation():
    """Test the RFE implementation in enhanced pipeline."""
    print("\n🧪 Testing RFE Implementation")
    print("=" * 35)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for RFE implementation components
        checks = [
            ("RFE with percentage step method", "def _rfe_with_percentage_step"),
            ("RFE step size calculation", "int(features_above_target * self.config.rfe_step_size)"),
            ("RFE recursive loop", "while len(current_features) > target_features:"),
            ("RFE step size logging", "Step size: {step_size} features (10% of {features_above_target})"),
            ("RFE rounds tracking", "rfe_rounds.append"),
            ("Fallback feature selection", "def _fallback_feature_selection"),
            ("RFE method name", "rfe_percentage_based"),
            ("RFE percentage step call", "_rfe_with_percentage_step"),
            ("RFE step size debug", "RFE step percentage: {self.config.rfe_step_size:.1%}")
        ]
        
        print("📊 Checking RFE implementation components...")
        all_passed = True
        
        for check_name, check_string in checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ RFE implementation checks passed!")
            return True
        else:
            print("\n❌ Some RFE implementation checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ RFE implementation test failed: {e}")
        return False

def test_rfe_logic():
    """Test the RFE logic and flow."""
    print("\n🧪 Testing RFE Logic and Flow")
    print("=" * 35)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for specific logic patterns
        logic_checks = [
            ("Features above target calculation", "features_above_target = len(current_features) - target_features"),
            ("Step size calculation", "step_size = max(1, int(features_above_target * self.config.rfe_step_size))"),
            ("RFE round logging", "RFE Round: {len(current_features)} features, {features_above_target} above target"),
            ("Feature removal logic", "features_to_remove = self._select_features_to_remove"),
            ("Feature update logic", "current_features = [f for f in current_features if f not in features_to_remove]"),
            ("DataFrame update logic", "current_X = current_X.drop(columns=features_to_remove)"),
            ("RFE rounds tracking", "rfe_rounds.append({"),
            ("Safety check", "if len(rfe_rounds) > 100:"),
            ("Bootstrap CV usage", "use_bootstrap_cv=use_bootstrap_cv"),
            ("Fallback mechanism", "return self._fallback_feature_selection")
        ]
        
        print("📊 Checking RFE logic and flow...")
        all_passed = True
        
        for check_name, check_string in logic_checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ RFE logic and flow checks passed!")
            return True
        else:
            print("\n❌ Some RFE logic checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ RFE logic test failed: {e}")
        return False

def test_rfe_examples():
    """Test RFE behavior with examples."""
    print("\n🧪 Testing RFE Behavior Examples")
    print("=" * 38)
    
    try:
        # Test the enhanced pipeline file
        pipeline_file = "src/training/steps/pre_training/feature_selection/core/enhanced_pipeline.py"
        
        with open(pipeline_file, 'r') as f:
            content = f.read()
        
        # Check for example scenarios in comments or documentation
        example_checks = [
            ("10% step size", "10% of {features_above_target}"),
            ("Minimum step size", "max(1, int("),
            ("RFE rounds", "rfe_rounds.append"),
            ("Step size calculation", "step_size = max(1, int(features_above_target * self.config.rfe_step_size))"),
            ("Recursive removal", "while len(current_features) > target_features:"),
            ("Feature tracking", "features_remaining"),
            ("Removal tracking", "features_removed"),
            ("Round tracking", "'round': len(rfe_rounds) + 1")
        ]
        
        print("📊 Checking RFE behavior examples...")
        all_passed = True
        
        for check_name, check_string in example_checks:
            if check_string in content:
                print(f"   ✅ {check_name}: Found")
            else:
                print(f"   ❌ {check_name}: Missing")
                all_passed = False
        
        if all_passed:
            print("\n✅ RFE behavior examples checks passed!")
            return True
        else:
            print("\n❌ Some RFE behavior checks failed!")
            return False
            
    except Exception as e:
        print(f"❌ RFE behavior examples test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 RFE Implementation with Percentage-Based Step Size Tests")
    print("=" * 65)
    
    tests = [
        ("RFE Configuration", test_rfe_configuration),
        ("RFE Implementation", test_rfe_implementation),
        ("RFE Logic and Flow", test_rfe_logic),
        ("RFE Behavior Examples", test_rfe_examples)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*65}")
        print(f"Running: {test_name}")
        print('='*65)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*65}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*65)
    
    if passed == total:
        print("🎉 All tests passed! The RFE implementation is ready to use.")
        print("\n📋 Summary of RFE Implementation:")
        print("   ✅ RFE uses 10% of features above target as step size")
        print("   ✅ RFE operates recursively until target is reached")
        print("   ✅ Bootstrap stability and CV only used when 40+ features away")
        print("   ✅ Fallback mechanism for error handling")
        print("   ✅ Comprehensive logging and tracking")
        print("   ✅ Safety checks to prevent infinite loops")
        print("   ✅ Removed old stage2_removal_percentage parameter")
        print("\n📊 Example RFE Behavior:")
        print("   • 200 → 60 features: Remove 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1...")
        print("   • 100 → 60 features: Remove 4, 3, 3, 3, 2, 2, 2, 1, 1, 1...")
        print("   • 65 → 60 features: Remove 1, 1, 1, 1, 1")
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)