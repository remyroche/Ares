#!/usr/bin/env python3
"""
Verify that the test setup is working correctly in the new location.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_structure():
    """Verify the test structure is correct."""
    code_quality_dir = Path(__file__).parent
    tests_dir = code_quality_dir / "tests"
    test_file = tests_dir / "test_common_operations.py"
    runner_file = code_quality_dir / "run_common_operations_tests.py"
    
    print("🔍 Verifying test structure...")
    print(f"   Code quality dir: {code_quality_dir}")
    print(f"   Tests dir exists: {tests_dir.exists()}")
    print(f"   Test file exists: {test_file.exists()}")
    print(f"   Runner file exists: {runner_file.exists()}")
    
    # Try to import common_operations
    try:
        from src.utils import common_operations
        print("✅ Successfully imported common_operations module")
        print(f"   Module has {len(dir(common_operations))} attributes")
    except ImportError as e:
        print(f"❌ Failed to import common_operations: {e}")
        return False
    
    # Try to import the test module
    try:
        sys.path.insert(0, str(code_quality_dir))
        from tests import test_common_operations
        print("✅ Successfully imported test module")
        
        # Count test classes and methods
        test_classes = [obj for obj in dir(test_common_operations) 
                       if obj.startswith('Test')]
        print(f"   Found {len(test_classes)} test classes")
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return False
    
    print("\n✅ Test setup verified successfully!")
    print("\nTo run tests:")
    print("  python code_quality/run_common_operations_tests.py")
    print("  python code_quality/run_common_operations_tests.py --coverage")
    
    return True

if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)