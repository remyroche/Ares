#!/usr/bin/env python3
"""
Simple test for timestamped prints to verify basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_timestamped_prints():
    """Test basic timestamped print functionality."""
    print("🧪 Testing timestamped print functionality...")
    
    try:
        from src.utils.logger import timestamped_print
        
        print("✅ Successfully imported timestamped_print")
        
        # Test the timestamped_print function
        print("\n--- Testing timestamped_print function ---")
        timestamped_print("This is a test message with timestamp")
        timestamped_print("Multiple arguments", "should work", "correctly")
        
        print("\n✅ Basic timestamped print test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_enhanced_logger():
    """Test enhanced logger."""
    print("\n🧪 Testing enhanced logger...")
    
    try:
        from src.utils.enhanced_simple_logger import enhanced_system_logger
        
        print("✅ Successfully imported enhanced logger")
        
        # Test logging
        logger = enhanced_system_logger
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        print("✅ Enhanced logger test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_artifact_naming():
    """Test artifact naming."""
    print("\n🧪 Testing artifact naming...")
    
    try:
        from src.utils.artifact_naming import create_outcome_filename
        
        filename = create_outcome_filename("data_collection", "data_download", "aresv1")
        print(f"✅ Generated filename: {filename}")
        
        # Check if filename contains expected components
        assert "data_collection" in filename
        assert "data_download" in filename
        assert "outcome" in filename
        assert "aresv1" in filename
        assert filename.endswith(".json")
        
        print("✅ Artifact naming test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting simple timestamp tests...")
    print("="*50)
    
    tests = [
        test_timestamped_prints,
        test_enhanced_logger,
        test_artifact_naming
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED - The new print system is working!")
    else:
        print("❌ Some tests failed")