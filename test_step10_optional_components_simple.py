#!/usr/bin/env python3
"""
Simple test script to verify step10 handles missing optional components gracefully.
This script tests the import handling and fallback mechanisms.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_import_fallbacks():
    """Test that import fallbacks work correctly."""
    print("🧪 Testing import fallbacks...")
    
    try:
        # Test the safe import mechanism
        from src.utils.pipeline_standards import PipelineStandards
        
        # Test importing a non-existent module
        non_existent = PipelineStandards.safe_import("non.existent.module", None)
        assert non_existent is None, "Non-existent module should return None"
        print("✅ Safe import correctly returns None for non-existent modules")
        
        # Test importing with fallback
        fallback_value = PipelineStandards.safe_import("non.existent.module", "fallback")
        assert fallback_value == "fallback", "Should return fallback value"
        print("✅ Safe import correctly returns fallback value")
        
    except Exception as e:
        print(f"❌ Import fallback test failed: {e}")
        return False
    
    return True

def test_common_operations_functions():
    """Test that common operations functions exist and work."""
    print("\n🧪 Testing common operations functions...")
    
    try:
        from src.utils.common_operations import (
            ensure_directory, 
            safe_json_dump, 
            standardize_price_action_probabilities,
            safe_json_load,
            safe_read_parquet
        )
        
        # Test ensure_directory
        test_dir = "test_directory"
        result = ensure_directory(test_dir)
        assert result is True, "ensure_directory should return True"
        assert os.path.exists(test_dir), "Directory should be created"
        print("✅ ensure_directory function works correctly")
        
        # Test safe_json_dump
        test_data = {"test": "data"}
        test_file = "test_file.json"
        result = safe_json_dump(test_data, test_file)
        assert result is True, "safe_json_dump should return True"
        assert os.path.exists(test_file), "JSON file should be created"
        print("✅ safe_json_dump function works correctly")
        
        # Test safe_json_load
        loaded_data = safe_json_load(test_file)
        assert loaded_data == test_data, "Loaded data should match original"
        print("✅ safe_json_load function works correctly")
        
        # Test standardize_price_action_probabilities
        test_probs = {
            "triple_barrier_probability": 0.3,
            "direction_probability": 0.2,
            "magnitude_probability": 0.1,
            "barrier_avoidance_probability": 0.4
        }
        standardized = standardize_price_action_probabilities(test_probs)
        assert isinstance(standardized, dict), "Should return a dictionary"
        assert abs(sum(standardized.values()) - 1.0) < 0.001, "Probabilities should sum to 1"
        print("✅ standardize_price_action_probabilities function works correctly")
        
        # Cleanup
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        if os.path.exists(test_file):
            os.remove(test_file)
        
    except Exception as e:
        print(f"❌ Common operations test failed: {e}")
        return False
    
    return True

def test_step10_import_structure():
    """Test that step10 can be imported without crashing."""
    print("\n🧪 Testing step10 import structure...")
    
    try:
        # Test that we can import the main components
        from src.training.steps.model_training.step10_unified_regime_intelligence import (
            MultiTimeframeHMMEncoder,
            UnifiedRegimeIntelligenceStep
        )
        print("✅ Main step10 classes can be imported")
        
        # Test that fallback functions are available
        from src.training.steps.model_training.step10_unified_regime_intelligence import (
            create_fallback_logger,
            create_fallback_decorator
        )
        assert callable(create_fallback_logger), "create_fallback_logger should be callable"
        assert callable(create_fallback_decorator), "create_fallback_decorator should be callable"
        print("✅ Fallback functions are available")
        
    except Exception as e:
        print(f"❌ Step10 import structure test failed: {e}")
        return False
    
    return True

def test_validator_import_structure():
    """Test that validator can be imported without crashing."""
    print("\n🧪 Testing validator import structure...")
    
    try:
        from src.training.steps.model_training.step10_unified_regime_intelligence_validator import (
            UnifiedRegimeIntelligenceValidator
        )
        print("✅ Validator class can be imported")
        
    except Exception as e:
        print(f"❌ Validator import structure test failed: {e}")
        return False
    
    return True

def test_optional_component_handling():
    """Test the optional component handling logic."""
    print("\n🧪 Testing optional component handling logic...")
    
    try:
        # Test the safe import mechanism for optional components
        from src.utils.pipeline_standards import PipelineStandards
        
        # Test each optional component
        optional_components = [
            "src.training.enhanced_lm_optimizer",
            "src.tactician.sr_breakout_predictor", 
            "src.training.model_specific_pruning",
            "src.utils.warning_symbols",
            "src.utils.error_handler"
        ]
        
        for component in optional_components:
            result = PipelineStandards.safe_import(component, None)
            # Should not crash, even if module doesn't exist
            print(f"✅ {component} handled gracefully (result: {result is not None})")
        
        print("✅ All optional components handled without crashing")
        
    except Exception as e:
        print(f"❌ Optional component handling test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests for optional component handling."""
    print("🚀 Testing Step10 Optional Component Handling (Simple)")
    print("=" * 60)
    
    tests = [
        test_import_fallbacks,
        test_common_operations_functions,
        test_step10_import_structure,
        test_validator_import_structure,
        test_optional_component_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All optional component tests passed!")
        print("✅ Step10 handles missing optional components gracefully")
    else:
        print("⚠️ Some tests failed - check the output above")

if __name__ == "__main__":
    main()