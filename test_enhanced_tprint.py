#!/usr/bin/env python3
"""
Comprehensive test script for the enhanced tprint functionality.
"""

import sys
import time
import threading
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_progress, tprint_performance, tprint_structured, tprint_with_level,
    tprint_batch, tprint_numba_compatible, tprint_timer, tprint_logged,
    configure_tprint, get_tprint_config, tprint_context, cleanup_tprint,
    TPrintConfig, LogLevel, TimestampFormat
)


def test_basic_functionality():
    """Test basic tprint functionality."""
    print("=" * 80)
    print("Testing Basic TPrint Functionality")
    print("=" * 80)
    
    tprint("Basic message")
    tprint_debug("Debug message")
    tprint_info("Info message")
    tprint_warning("Warning message")
    tprint_error("Error message")
    tprint_success("Success message")
    
    print("\nTesting with multiple arguments:")
    tprint("Multiple", "arguments", "test", 42, [1, 2, 3])


def test_progress_and_performance():
    """Test progress and performance logging."""
    print("\n" + "=" * 80)
    print("Testing Progress and Performance Logging")
    print("=" * 80)
    
    # Test progress
    for i in range(1, 6):
        tprint_progress(i, 5, f"Processing item {i}")
        time.sleep(0.1)
    
    # Test performance
    tprint_performance("Data processing", 2.5)
    tprint_performance("Model training", 45.123)


def test_structured_logging():
    """Test structured logging."""
    print("\n" + "=" * 80)
    print("Testing Structured Logging")
    print("=" * 80)
    
    # Test structured data
    user_data = {
        "user_id": 12345,
        "username": "test_user",
        "login_time": "2025-01-11T10:30:00Z",
        "permissions": ["read", "write"]
    }
    
    tprint_structured(user_data, LogLevel.INFO)
    
    # Test with different log levels
    error_data = {
        "error_code": "E001",
        "error_message": "Connection timeout",
        "retry_count": 3
    }
    
    tprint_structured(error_data, LogLevel.ERROR)


def test_configuration():
    """Test configuration options."""
    print("\n" + "=" * 80)
    print("Testing Configuration Options")
    print("=" * 80)
    
    # Test different timestamp formats
    configs = [
        ("Simple Format", TPrintConfig(timestamp_format=TimestampFormat.SIMPLE)),
        ("Detailed Format", TPrintConfig(timestamp_format=TimestampFormat.DETAILED)),
        ("With Microseconds", TPrintConfig(timestamp_format=TimestampFormat.WITH_MICROSECONDS)),
        ("ISO Format", TPrintConfig(timestamp_format=TimestampFormat.ISO)),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        with tprint_context(config):
            tprint(f"Testing {name}")
            tprint_info("Info message")
            tprint_warning("Warning message")


def test_file_logging():
    """Test file logging functionality."""
    print("\n" + "=" * 80)
    print("Testing File Logging")
    print("=" * 80)
    
    log_file = Path("test_tprint.log")
    
    # Configure for file logging
    config = TPrintConfig(
        output_to_file=True,
        output_file=log_file,
        output_to_console=True,
        timestamp_format=TimestampFormat.DETAILED
    )
    
    with tprint_context(config):
        tprint("This message should appear in both console and file")
        tprint_info("File logging test")
        tprint_warning("Warning in file")
        tprint_error("Error in file")
    
    # Check if file was created and has content
    if log_file.exists():
        print(f"\nLog file created: {log_file}")
        print("File contents:")
        with open(log_file, 'r') as f:
            print(f.read())
        # Clean up
        log_file.unlink()
    else:
        print("Warning: Log file was not created")


def test_context_manager():
    """Test context manager functionality."""
    print("\n" + "=" * 80)
    print("Testing Context Manager")
    print("=" * 80)
    
    print("Before context:")
    tprint("Regular message")
    
    with tprint_context(TPrintConfig(timestamp_format=TimestampFormat.SIMPLE)):
        print("Inside context:")
        tprint("Context message")
        tprint_info("Context info")
    
    print("After context:")
    tprint("Back to regular")


def test_timer_context():
    """Test timer context manager."""
    print("\n" + "=" * 80)
    print("Testing Timer Context Manager")
    print("=" * 80)
    
    with tprint_timer("Test operation"):
        time.sleep(0.5)
        tprint("Doing some work...")
        time.sleep(0.3)
    
    with tprint_timer("Another operation", LogLevel.INFO):
        time.sleep(0.2)


def test_decorator():
    """Test logging decorator."""
    print("\n" + "=" * 80)
    print("Testing Logging Decorator")
    print("=" * 80)
    
    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def test_function(x, y, name="test"):
        tprint(f"Inside function: {x} + {y} = {x + y}")
        return x + y
    
    @tprint_logged(LogLevel.DEBUG, include_args=False, include_result=False)
    def simple_function():
        tprint("Simple function execution")
        return "done"
    
    result = test_function(5, 3, name="calculation")
    tprint(f"Function returned: {result}")
    
    simple_function()


def test_batch_logging():
    """Test batch logging for performance."""
    print("\n" + "=" * 80)
    print("Testing Batch Logging")
    print("=" * 80)
    
    messages = [
        (LogLevel.INFO, "Batch message 1"),
        (LogLevel.DEBUG, "Batch message 2"),
        (LogLevel.WARNING, "Batch message 3"),
        (LogLevel.ERROR, "Batch message 4"),
        (LogLevel.SUCCESS, "Batch message 5"),
    ]
    
    tprint_batch(messages)


def test_thread_safety():
    """Test thread safety."""
    print("\n" + "=" * 80)
    print("Testing Thread Safety")
    print("=" * 80)
    
    def worker(thread_id):
        for i in range(3):
            tprint(f"Thread {thread_id} - Message {i}")
            time.sleep(0.01)
    
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    tprint("All threads completed")


def test_numba_compatibility():
    """Test numba compatibility."""
    print("\n" + "=" * 80)
    print("Testing Numba Compatibility")
    print("=" * 80)
    
    tprint_numba_compatible("Numba compatible message")
    tprint_numba_compatible("Multiple", "arguments", "test")


def test_log_level_filtering():
    """Test log level filtering."""
    print("\n" + "=" * 80)
    print("Testing Log Level Filtering")
    print("=" * 80)
    
    # Test with different minimum log levels
    levels = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
    
    for min_level in levels:
        print(f"\nMinimum log level: {min_level.value}")
        config = TPrintConfig(min_log_level=min_level)
        with tprint_context(config):
            tprint_debug("Debug message")
            tprint_info("Info message")
            tprint_warning("Warning message")
            tprint_error("Error message")


def test_performance():
    """Test performance with many messages."""
    print("\n" + "=" * 80)
    print("Testing Performance")
    print("=" * 80)
    
    num_messages = 1000
    
    # Test with caching enabled
    print(f"Testing with {num_messages} messages (caching enabled):")
    start_time = time.perf_counter()
    
    for i in range(num_messages):
        tprint(f"Performance test message {i}")
    
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.3f}s")
    
    # Test with caching disabled
    print(f"\nTesting with {num_messages} messages (caching disabled):")
    config = TPrintConfig(cache_timestamps=False)
    with tprint_context(config):
        start_time = time.perf_counter()
        
        for i in range(num_messages):
            tprint(f"Performance test message {i}")
        
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time:.3f}s")


def main():
    """Run all tests."""
    print("Enhanced TPrint Test Suite")
    print("=" * 80)
    
    try:
        test_basic_functionality()
        test_progress_and_performance()
        test_structured_logging()
        test_configuration()
        test_file_logging()
        test_context_manager()
        test_timer_context()
        test_decorator()
        test_batch_logging()
        test_thread_safety()
        test_numba_compatibility()
        test_log_level_filtering()
        test_performance()
        
        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_tprint()


if __name__ == "__main__":
    main()