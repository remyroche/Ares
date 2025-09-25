#!/usr/bin/env python3
"""
Simple test script to verify the completed implementations can be imported.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that the completed files can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test pure_tree_nas imports
        print("Testing pure_tree_nas imports...")
        from src.utils.ml_common.optimization.pure_tree_nas import (
            PureTreeNAS, 
            PureTreeNASConfig, 
            TreeArchitectureCandidate,
            NODEModel,
            ObliviousTreeModel,
            RotationForestModel,
            HistogramGradientBoostingModel
        )
        print("✅ Pure Tree NAS imports successful")
        
        # Test hybrid_nas_system imports
        print("Testing hybrid_nas_system imports...")
        from src.utils.ml_common.optimization.hybrid_nas_system import (
            HybridNASSystem,
            HybridNASConfig,
            HybridArchitectureCandidate,
            search_hybrid_architecture,
            search_tree_only_architecture,
            search_neural_only_architecture
        )
        print("✅ Hybrid NAS System imports successful")
        
        # Test utility imports
        print("Testing utility imports...")
        from src.utils.common_operations import safe_divide, safe_log, safe_sqrt
        from src.utils.math_validation import validate_finite, validate_positive
        from src.utils.serialization_utils import JSONSerializer, PickleSerializer
        from src.utils.tprint import tprint, tprint_info, tprint_success
        print("✅ Utility imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test math utilities
        from src.utils.common_operations import safe_divide, safe_log, safe_sqrt
        
        result = safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Math utilities working")
        
        # Test validation utilities
        from src.utils.math_validation import validate_finite, validate_positive
        
        validated = validate_finite(42.0)
        assert validated == 42.0, f"Expected 42.0, got {validated}"
        print("✅ Validation utilities working")
        
        # Test tprint utilities
        from src.utils.tprint import tprint, tprint_info, tprint_success
        
        tprint("Test message")
        tprint_info("Test info message")
        tprint_success("Test success message")
        print("✅ TPrint utilities working")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting simple implementation verification tests...\n")
    
    tests = [
        test_imports,
        test_basic_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All implementations completed successfully!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementations.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)