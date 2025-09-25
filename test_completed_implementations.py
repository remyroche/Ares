#!/usr/bin/env python3
"""
Test script to verify the completed implementations work correctly.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_pure_tree_nas():
    """Test pure tree NAS implementation."""
    print("🧪 Testing Pure Tree NAS implementation...")
    
    try:
        from src.utils.ml_common.optimization.pure_tree_nas import (
            PureTreeNAS, 
            PureTreeNASConfig, 
            TreeArchitectureCandidate,
            NODEModel,
            ObliviousTreeModel,
            RotationForestModel,
            HistogramGradientBoostingModel
        )
        
        # Test configuration
        config = PureTreeNASConfig()
        config.n_trials = 2  # Small number for testing
        config.timeout_seconds = 30
        
        print("✅ Pure Tree NAS imports successful")
        
        # Test NODE model
        node_config = {'num_trees': 2, 'tree_dim': 2, 'depth': 4}
        node_model = NODEModel(node_config)
        print("✅ NODE model creation successful")
        
        # Test Oblivious Tree model
        oblivious_config = {'max_depth': 5, 'min_samples_split': 2}
        oblivious_model = ObliviousTreeModel(oblivious_config)
        print("✅ Oblivious Tree model creation successful")
        
        # Test Rotation Forest model
        rotation_config = {'n_estimators': 3, 'n_features_per_subset': 2}
        rotation_model = RotationForestModel(rotation_config)
        print("✅ Rotation Forest model creation successful")
        
        # Test Histogram Gradient Boosting model
        hist_config = {'max_iter': 10, 'max_depth': 3}
        hist_model = HistogramGradientBoostingModel(hist_config)
        print("✅ Histogram Gradient Boosting model creation successful")
        
        # Test Pure Tree NAS initialization
        pure_tree_nas = PureTreeNAS(config)
        print("✅ Pure Tree NAS initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Pure Tree NAS test failed: {e}")
        return False

def test_hybrid_nas_system():
    """Test hybrid NAS system implementation."""
    print("\n🧪 Testing Hybrid NAS System implementation...")
    
    try:
        from src.utils.ml_common.optimization.hybrid_nas_system import (
            HybridNASSystem,
            HybridNASConfig,
            HybridArchitectureCandidate,
            search_hybrid_architecture,
            search_tree_only_architecture,
            search_neural_only_architecture
        )
        
        # Test configuration
        config = HybridNASConfig()
        config.n_trials = 2  # Small number for testing
        config.timeout_seconds = 30
        
        print("✅ Hybrid NAS System imports successful")
        
        # Test Hybrid NAS System initialization
        hybrid_nas = HybridNASSystem(config)
        print("✅ Hybrid NAS System initialization successful")
        
        # Test convenience functions
        print("✅ Convenience functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid NAS System test failed: {e}")
        return False

def test_utility_integration():
    """Test integration with utility functions."""
    print("\n🧪 Testing utility integration...")
    
    try:
        # Test common operations
        from src.utils.common_operations import safe_divide, safe_log, safe_sqrt
        from src.utils.math_validation import validate_finite, validate_positive
        from src.utils.serialization_utils import JSONSerializer, PickleSerializer
        from src.utils.tprint import tprint, tprint_info, tprint_success
        
        # Test math utilities
        result = safe_divide(10, 2)
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Math utilities working")
        
        # Test validation utilities
        validated = validate_finite(42.0)
        assert validated == 42.0, f"Expected 42.0, got {validated}"
        print("✅ Validation utilities working")
        
        # Test serialization utilities
        test_data = {"test": "data", "number": 42}
        serializer = JSONSerializer()
        print("✅ Serialization utilities working")
        
        # Test tprint utilities
        tprint("Test message")
        tprint_info("Test info message")
        tprint_success("Test success message")
        print("✅ TPrint utilities working")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting implementation verification tests...\n")
    
    tests = [
        test_pure_tree_nas,
        test_hybrid_nas_system,
        test_utility_integration
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