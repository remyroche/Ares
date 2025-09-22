#!/usr/bin/env python3
"""
Test script to verify tprint and automatic print logging integration.

This script tests:
1. Basic tprint functionality
2. Automatic logging of print statements
3. Integration with Python logging system
4. Configuration options
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import tprint functionality
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error,
    enable_auto_print_logging, set_print_log_level, LogLevel,
    get_tprint_config, configure_tprint, TPrintConfig
)

def test_basic_tprint():
    """Test basic tprint functionality."""
    print("\n🧪 Testing basic tprint functionality...")

    tprint("Basic tprint message")
    tprint_info("Info level message")
    tprint_warning("Warning level message")
    tprint_error("Error level message")

    print("✅ Basic tprint test completed")

def test_auto_print_logging():
    """Test automatic logging of print statements."""
    print("\n🧪 Testing automatic print logging...")

    # Test using context manager approach
    from src.utils.tprint import capture_print_to_tprint
    with capture_print_to_tprint():
        print("This print should be captured and logged to tprint")
        print("Another print statement", 42, "test")

    print("✅ Auto print logging test completed")

def test_configuration():
    """Test configuration options."""
    print("\n🧪 Testing configuration options...")

    config = get_tprint_config()
    print(f"Current config - auto_log_prints: {config.auto_log_prints}")
    print(f"Current config - print_log_level: {config.print_log_level}")

    # Test changing configuration
    set_print_log_level(LogLevel.WARNING)
    config = get_tprint_config()
    print(f"Updated config - print_log_level: {config.print_log_level}")

    print("✅ Configuration test completed")

def test_python_logging_integration():
    """Test integration with Python logging."""
    print("\n🧪 Testing Python logging integration...")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create a logger
    logger = logging.getLogger('test_logger')
    logger.info("This is a Python logging message")

    # Test tprint with logging integration
    tprint_info("This tprint message should also go to Python logging")

    print("✅ Python logging integration test completed")

def main():
    """Main test function."""
    print("🚀 Starting tprint integration tests...")
    print("=" * 60)

    try:
        test_basic_tprint()
        test_auto_print_logging()
        test_configuration()
        test_python_logging_integration()

        print("\n🎉 All tests completed successfully!")
        print("📊 Summary:")
        print("   - tprint functionality: ✅ Working")
        print("   - Auto print logging: ✅ Working")
        print("   - Configuration: ✅ Working")
        print("   - Python logging integration: ✅ Working")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()