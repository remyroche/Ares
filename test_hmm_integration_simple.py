#!/usr/bin/env python3
"""
Simple test script for HMM Common Utilities Integration

This script tests the integration without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all common utilities can be imported."""
    print("🔄 Testing imports...")
    
    try:
        # Test common utilities imports
        from src.utils.common_operations import (
            safe_dataframe_operation,
            validate_dataframe_columns,
            calculate_data_quality_metrics,
            get_m1_gpu_manager,
            get_m1_memory_optimizer,
            get_m1_cpu_optimizer
        )
        print("✅ Common operations imported successfully")
        
        from src.utils.math_validation import (
            safe_divide,
            validate_finite,
            validate_numeric_array,
            safe_log,
            safe_sqrt
        )
        print("✅ Math validation imported successfully")
        
        from src.utils.serialization_utils import (
            JSONSerializer,
            PickleSerializer
        )
        print("✅ Serialization utilities imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_math_validation():
    """Test math validation functions."""
    print("\n🔄 Testing math validation...")
    
    try:
        from src.utils.math_validation import safe_divide, validate_finite, safe_sqrt
        
        # Test safe division
        result = safe_divide(10, 2, 0.0)
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Safe division works correctly")
        
        # Test division by zero
        result = safe_divide(10, 0, 0.0)
        assert result == 0.0, f"Expected 0.0, got {result}"
        print("✅ Safe division by zero works correctly")
        
        # Test validate finite
        result = validate_finite(5.0, "test_value")
        assert result == 5.0, f"Expected 5.0, got {result}"
        print("✅ Validate finite works correctly")
        
        # Test safe sqrt
        result = safe_sqrt(16.0, 0.0)
        assert result == 4.0, f"Expected 4.0, got {result}"
        print("✅ Safe sqrt works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Math validation error: {e}")
        return False

def test_serialization():
    """Test serialization utilities."""
    print("\n🔄 Testing serialization utilities...")
    
    try:
        from src.utils.serialization_utils import JSONSerializer, PickleSerializer
        
        # Test data
        test_data = {
            'model_name': 'test_model',
            'accuracy': 0.85,
            'timestamp': '2024-01-01T00:00:00'
        }
        
        # Test JSON serialization
        json_path = "test_metadata.json"
        success = JSONSerializer.save(test_data, json_path)
        assert success, "JSON serialization failed"
        print("✅ JSON serialization works correctly")
        
        # Test JSON loading
        loaded_data = JSONSerializer.load(json_path)
        assert loaded_data is not None, "JSON loading failed"
        assert loaded_data['model_name'] == 'test_model', "JSON data mismatch"
        print("✅ JSON loading works correctly")
        
        # Cleanup
        os.remove(json_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization error: {e}")
        return False

def test_file_structure():
    """Test that the HMM training file has been updated correctly."""
    print("\n🔄 Testing file structure...")
    
    try:
        hmm_file = Path("src/training/steps/market_analysis/hmm_models_training/hmm_models_training_enhanced.py")
        
        if not hmm_file.exists():
            print("❌ HMM training file not found")
            return False
        
        # Read the file and check for common utilities integration
        content = hmm_file.read_text()
        
        # Check for common utilities imports
        if "from src.utils.common_operations import" in content:
            print("✅ Common operations imports found")
        else:
            print("❌ Common operations imports not found")
            return False
        
        if "from src.utils.math_validation import" in content:
            print("✅ Math validation imports found")
        else:
            print("❌ Math validation imports not found")
            return False
        
        if "from src.utils.serialization_utils import" in content:
            print("✅ Serialization imports found")
        else:
            print("❌ Serialization imports not found")
            return False
        
        if "_initialize_hardware_optimizers" in content:
            print("✅ Hardware optimizer initialization found")
        else:
            print("❌ Hardware optimizer initialization not found")
            return False
        
        if "safe_divide" in content:
            print("✅ Safe math operations found")
        else:
            print("❌ Safe math operations not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test error: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting HMM Common Utilities Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_math_validation,
        test_serialization,
        test_file_structure
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
        print("🎉 All tests passed! HMM Common Utilities Integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the integration.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)