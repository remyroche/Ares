#!/usr/bin/env python3
from src.utils.tprint import tprint

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
    
    tprint("=" * 60)
    tprint("Testing Timestamped Print Functionality")
    tprint("=" * 60)
    
    # Test 1: Regular prints (should have timestamps after enabling)
    tprint("\n1. Testing regular prints:")
    tprint("This is a regular print statement")
    tprint("Another print statement")
    
    # Test 2: Enable timestamped prints
    tprint("\n2. Enabling timestamped prints...")
    success = enable_timestamped_prints(include_relative=True, include_session=True)
    tprint(f"Timestamped prints enabled: {success}")
    
    # Test 3: Print statements with timestamps
    tprint("\n3. Print statements with timestamps:")
    tprint("This print should now have a timestamp!")
    tprint("Another timestamped print")
    
    # Test 4: Wait a bit and print again to see session duration
    tprint("\n4. Waiting 2 seconds to see session duration...")
    time.sleep(2)
    tprint("This print should show session duration")
    
    # Test 5: Test logging as well
    tprint("\n5. Testing logging with enhanced timestamps:")
    logger = get_logger("TestLogger")
    logger.info("This is a log message with enhanced timestamp")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test 6: Disable timestamped prints
    tprint("\n6. Disabling timestamped prints...")
    success = disable_timestamped_prints()
    tprint(f"Timestamped prints disabled: {success}")
    
    # Test 7: Regular prints again (should not have timestamps)
    tprint("\n7. Regular prints again (no timestamps):")
    tprint("This print should NOT have a timestamp")
    tprint("Another regular print")
    
    tprint("\n" + "=" * 60)
    tprint("Test completed!")
    tprint("=" * 60)

if __name__ == "__main__":
    test_timestamped_prints()
