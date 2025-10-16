#!/usr/bin/env python3
"""
Test script to verify circular import fixes.
"""

import sys
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test importing key modules to check for circular imports."""
    print("🧪 Testing circular import fixes...")
    
    # Test base utilities
    try:
        from src.utils.base_utilities import validate_file_path, create_directory_safe
        print("✅ Base utilities imported successfully")
    except Exception as e:
        print(f"❌ Base utilities import failed: {e}")
        return False
    
    # Test base matrix operations
    try:
        from src.utils.base_matrix_operations import safe_correlation_matrix, safe_matrix_rank
        print("✅ Base matrix operations imported successfully")
    except Exception as e:
        print(f"❌ Base matrix operations import failed: {e}")
        return False
    
    # Test lazy imports
    try:
        from src.utils.lazy_module_loader import get_validate_file_path, get_safe_correlation_matrix
        print("✅ Lazy module loader imported successfully")
    except Exception as e:
        print(f"❌ Lazy module loader import failed: {e}")
        return False
    
    # Test common operations (should not have circular imports now)
    try:
        from src.utils.common_operations import validate_file_path as co_validate_file_path
        print("✅ Common operations imported successfully")
    except Exception as e:
        print(f"❌ Common operations import failed: {e}")
        return False
    
    # Test matrix operations (should not have circular imports now)
    try:
        from src.utils.matrix_operations import safe_correlation_matrix as mo_safe_correlation_matrix
        print("✅ Matrix operations imported successfully")
    except Exception as e:
        print(f"❌ Matrix operations import failed: {e}")
        return False
    
    # Test lazy imports functionality
    try:
        validate_file_path_func = get_validate_file_path()
        safe_correlation_func = get_safe_correlation_matrix()
        print("✅ Lazy import functions work correctly")
    except Exception as e:
        print(f"❌ Lazy import functions failed: {e}")
        return False
    
    print("🎉 All circular import tests passed!")
    return True

def test_functionality():
    """Test that the imported functions work correctly."""
    print("\n🔧 Testing functionality...")
    
    try:
        from src.utils.base_utilities import validate_file_path, safe_divide, validate_finite
        
        # Test validate_file_path
        test_path = Path(__file__)
        result = validate_file_path(test_path)
        print(f"✅ validate_file_path works: {result}")
        
        # Test safe_divide
        result = safe_divide(10, 2)
        print(f"✅ safe_divide works: {result}")
        
        # Test validate_finite
        result = validate_finite(5.0)
        print(f"✅ validate_finite works: {result}")
        
        print("🎉 All functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting circular import tests...")
    
    success = True
    success &= test_imports()
    success &= test_functionality()
    
    if success:
        print("\n🎉 All tests passed! Circular imports have been resolved.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Circular imports may still exist.")
        sys.exit(1)