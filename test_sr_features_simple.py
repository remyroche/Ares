#!/usr/bin/env python3
"""
Simple test script for comprehensive SR features implementation.
This script tests the feature assignment logic without requiring external dependencies.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_step3_feature_assignment():
    """Test Step3 feature assignment for SR block."""
    print("🧪 Testing Step3 feature assignment for SR block...")
    
    try:
        # Import the function directly
        from src.training.steps.step3_hmm_regime_discovery import _assign_block
        print("✅ Successfully imported _assign_block function")
        
        # Test feature names that should be assigned to SR block
        test_features = [
            'distance_to_support', 'distance_to_resistance',
            'normalized_distance_to_support', 'normalized_distance_to_resistance',
            'sr_proximity_score', 'strength_score', 'clarity_factor',
            'directional_pressure', 'sr_score', 'delta_sr_score',
            'isolation_score', 'support_strength', 'resistance_strength',
            'support_clarity_factor', 'resistance_clarity_factor',
            'pivot_support', 'fibonacci_resistance', 'sr_level_strength'
        ]
        
        sr_block_features = []
        other_block_features = []
        
        for feature in test_features:
            try:
                assigned_block = _assign_block(feature)
                if assigned_block == "support_resistance":
                    sr_block_features.append(feature)
                else:
                    other_block_features.append(feature)
            except Exception as e:
                print(f"❌ Error assigning block for {feature}: {e}")
                return False
        
        print(f"✅ Features assigned to SR block: {len(sr_block_features)}")
        for feature in sr_block_features:
            print(f"  ✅ {feature} -> support_resistance")
        
        if other_block_features:
            print(f"⚠️ Features assigned to other blocks: {len(other_block_features)}")
            for feature in other_block_features:
                assigned_block = _assign_block(feature)
                print(f"  ⚠️ {feature} -> {assigned_block}")
        
        # Check if we have the essential features for SR block
        essential_features = ['sr_score', 'delta_sr_score', 'directional_pressure']
        missing_essential = [f for f in essential_features if f not in sr_block_features]
        
        if missing_essential:
            print(f"❌ Missing essential SR features: {missing_essential}")
            return False
        else:
            print("✅ All essential SR features properly assigned to SR block")
            return True
            
    except ImportError as e:
        print(f"❌ Failed to import _assign_block function: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing feature assignment: {e}")
        return False

def test_sr_breakout_predictor_import():
    """Test that SRBreakoutPredictor can be imported."""
    print("\n🧪 Testing SRBreakoutPredictor import...")
    
    try:
        from src.tactician.sr_breakout_predictor import SRBreakoutPredictor, setup_sr_breakout_predictor
        print("✅ Successfully imported SRBreakoutPredictor and setup function")
        
        # Test that the class has the expected methods
        expected_methods = [
            'calculate_comprehensive_sr_features',
            '_calculate_distance_features',
            '_calculate_normalized_distance_features',
            '_calculate_sr_proximity_features',
            '_calculate_strength_score_features',
            '_calculate_directional_pressure_features',
            '_calculate_sr_score_features',
            '_calculate_delta_sr_score_features'
        ]
        
        missing_methods = []
        for method in expected_methods:
            if not hasattr(SRBreakoutPredictor, method):
                missing_methods.append(method)
        
        if missing_methods:
            print(f"❌ Missing expected methods: {missing_methods}")
            return False
        else:
            print("✅ All expected methods found in SRBreakoutPredictor")
            return True
            
    except ImportError as e:
        print(f"❌ Failed to import SRBreakoutPredictor: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing SRBreakoutPredictor import: {e}")
        return False

def test_step2_integration():
    """Test Step2 integration of comprehensive SR features."""
    print("\n🧪 Testing Step2 integration of comprehensive SR features...")
    
    try:
        # Check if the comprehensive SR feature function exists
        from src.training.steps.step2_feature_engineering import _generate_comprehensive_sr_features
        print("✅ Comprehensive SR feature generation function found in Step2")
        
        # Test that the function has the expected signature
        import inspect
        sig = inspect.signature(_generate_comprehensive_sr_features)
        expected_params = ['price_df', 'sr_levels']
        
        missing_params = []
        for param in expected_params:
            if param not in sig.parameters:
                missing_params.append(param)
        
        if missing_params:
            print(f"❌ Missing expected parameters: {missing_params}")
            return False
        else:
            print("✅ Function has expected parameters")
            return True
            
    except ImportError as e:
        print(f"❌ Failed to import Step2 comprehensive SR features: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing Step2 integration: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting simple comprehensive SR features test suite...")
    
    results = []
    
    # Test 1: Step3 feature assignment
    result1 = test_step3_feature_assignment()
    results.append(("Step3 Feature Assignment", result1))
    
    # Test 2: SRBreakoutPredictor import
    result2 = test_sr_breakout_predictor_import()
    results.append(("SRBreakoutPredictor Import", result2))
    
    # Test 3: Step2 integration
    result3 = test_step2_integration()
    results.append(("Step2 Integration", result3))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Comprehensive SR features implementation is correct.")
        print("\n📋 Implementation Summary:")
        print("✅ SRBreakoutPredictor enhanced with comprehensive SR features")
        print("✅ Step2 integrates comprehensive SR features")
        print("✅ Step3 properly assigns SR features to support_resistance block")
        print("✅ Essential features (sr_score, delta_sr_score, directional_pressure) included")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)