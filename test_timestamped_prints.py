#!/usr/bin/env python3
"""
Test script to demonstrate the new timestamped print functionality.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import enable_timestamped_prints, disable_timestamped_prints, setup_logging, get_logger

def test_timestamped_prints():
    """Test the timestamped print functionality."""
    
    print("=" * 60)
    print("Testing Timestamped Print Functionality")
    print("=" * 60)
    
    # Test 1: Regular prints (should have timestamps after enabling)
    print("\n1. Testing regular prints:")
    print("This is a regular print statement")
    print("Another print statement")
    
    # Test 2: Enable timestamped prints
    print("\n2. Enabling timestamped prints...")
    success = enable_timestamped_prints(include_relative=True, include_session=True)
    print(f"Timestamped prints enabled: {success}")
    
    # Test 3: Print statements with timestamps
    print("\n3. Print statements with timestamps:")
    print("This print should now have a timestamp!")
    print("Another timestamped print")
    
    # Test 4: Wait a bit and print again to see session duration
    print("\n4. Waiting 2 seconds to see session duration...")
    time.sleep(2)
    print("This print should show session duration")
    
    # Test 5: Test logging as well
    print("\n5. Testing logging with enhanced timestamps:")
    logger = get_logger("TestLogger")
    logger.info("This is a log message with enhanced timestamp")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test 6: Disable timestamped prints
    print("\n6. Disabling timestamped prints...")
    success = disable_timestamped_prints()
    print(f"Timestamped prints disabled: {success}")
    
    # Test 7: Regular prints again (should not have timestamps)
    print("\n7. Regular prints again (no timestamps):")
    print("This print should NOT have a timestamp")
    print("Another regular print")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_timestamped_prints()
