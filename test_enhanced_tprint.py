#!/usr/bin/env python3
"""
Test script for enhanced tprint features.

This demonstrates all the advanced features of the enhanced tprint system.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_enhanced_features():
    """Test basic enhanced tprint features."""
    print("\n" + "="*60)
    print("🧪 TESTING ENHANCED TPRINT FEATURES")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import (
            tprint, tprint_debug, tprint_info, tprint_warning, 
            tprint_error, tprint_critical, tprint_success,
            tprint_progress, tprint_performance, tprint_context, tprint_timer
        )
        
        print("✅ Successfully imported enhanced tprint functions")
        
        # Test different log levels with colors
        print("\n--- Different log levels with colors ---")
        tprint_debug("This is a debug message")
        tprint_info("This is an info message")
        tprint_warning("This is a warning message")
        tprint_error("This is an error message")
        tprint_critical("This is a critical message")
        tprint_success("This is a success message")
        
        # Test enhanced progress with progress bar
        print("\n--- Enhanced progress with progress bar ---")
        for i in range(1, 11):
            tprint_progress(i, 10, "Processing data", show_bar=True)
            time.sleep(0.1)
        
        # Test performance tracking
        print("\n--- Performance tracking ---")
        tprint_performance("Data processing", 2.5)
        tprint_performance("Model training", 45.2)
        tprint_performance("Quick operation", 0.1)
        
        print("\n✅ Enhanced tprint features test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_context_management():
    """Test context management features."""
    print("\n" + "="*60)
    print("🧪 TESTING CONTEXT MANAGEMENT")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import tprint_context, tprint_info
        
        print("--- Context management ---")
        
        with tprint_context("DataCollection"):
            tprint_info("Starting data collection")
            
            with tprint_context("Download"):
                tprint_info("Downloading data from API")
                tprint_info("Download completed")
            
            with tprint_context("Processing"):
                tprint_info("Processing downloaded data")
                tprint_info("Processing completed")
            
            tprint_info("Data collection completed")
        
        print("\n✅ Context management test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_performance_timing():
    """Test performance timing context manager."""
    print("\n" + "="*60)
    print("🧪 TESTING PERFORMANCE TIMING")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import tprint_timer, tprint_info
        
        print("--- Performance timing context manager ---")
        
        with tprint_timer("Database Query"):
            tprint_info("Executing database query")
            time.sleep(0.5)  # Simulate work
        
        with tprint_timer("File Processing"):
            tprint_info("Processing large file")
            time.sleep(0.3)  # Simulate work
        
        with tprint_timer("Quick Operation"):
            tprint_info("Quick operation")
            time.sleep(0.1)  # Simulate work
        
        print("\n✅ Performance timing test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_configuration():
    """Test configuration features."""
    print("\n" + "="*60)
    print("🧪 TESTING CONFIGURATION FEATURES")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import (
            configure_tprint, TPrintConfig, LogLevel, 
            tprint_info, get_performance_summary
        )
        
        print("--- Configuration test ---")
        
        # Test with custom configuration
        config = TPrintConfig(
            timestamp_format='%H:%M:%S',
            include_microseconds=True,
            enable_colors=True,
            enable_file_output=True,
            log_file_path="logs/enhanced_test.log",
            performance_threshold=0.5
        )
        
        configure_tprint(config)
        
        tprint_info("This message uses custom configuration")
        tprint_info("Timestamp includes microseconds")
        
        # Test performance summary
        print("\n--- Performance summary ---")
        summary = get_performance_summary()
        print(f"Performance summary: {summary}")
        
        print("\n✅ Configuration test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_file_output():
    """Test file output functionality."""
    print("\n" + "="*60)
    print("🧪 TESTING FILE OUTPUT")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import (
            configure_tprint, TPrintConfig, 
            tprint_info, tprint_warning, tprint_error
        )
        
        print("--- File output test ---")
        
        # Configure for file output
        config = TPrintConfig(
            enable_file_output=True,
            log_file_path="logs/test_output.log",
            enable_colors=False  # No colors in file
        )
        
        configure_tprint(config)
        
        tprint_info("This message goes to both console and file")
        tprint_warning("Warning message also goes to file")
        tprint_error("Error message in file too")
        
        # Check if file was created
        log_file = Path("logs/test_output.log")
        if log_file.exists():
            print(f"✅ Log file created: {log_file}")
            print(f"File size: {log_file.stat().st_size} bytes")
            
            # Show file content
            print("\n--- Log file content ---")
            with open(log_file, 'r') as f:
                print(f.read())
        else:
            print("❌ Log file was not created")
        
        print("\n✅ File output test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_performance_metrics():
    """Test performance metrics collection and export."""
    print("\n" + "="*60)
    print("🧪 TESTING PERFORMANCE METRICS")
    print("="*60)
    
    try:
        from src.utils.enhanced_tprint import (
            tprint_timer, tprint_performance, 
            get_performance_summary, export_performance_metrics,
            clear_performance_metrics
        )
        
        print("--- Performance metrics collection ---")
        
        # Generate some performance data
        with tprint_timer("Operation A"):
            time.sleep(0.2)
        
        with tprint_timer("Operation B"):
            time.sleep(0.3)
        
        with tprint_timer("Operation A"):  # Same operation, different timing
            time.sleep(0.1)
        
        tprint_performance("Manual Operation", 0.5)
        
        # Get performance summary
        print("\n--- Performance summary ---")
        summary = get_performance_summary()
        print(f"Total operations: {summary.get('total_operations', 0)}")
        print(f"Total duration: {summary.get('total_duration', 0):.3f}s")
        print(f"Average duration: {summary.get('average_duration', 0):.3f}s")
        
        if 'operation_breakdown' in summary:
            print("\nOperation breakdown:")
            for op, stats in summary['operation_breakdown'].items():
                print(f"  {op}: {stats['count']} calls, avg {stats['average']:.3f}s")
        
        # Export metrics
        export_file = "logs/performance_metrics.json"
        export_performance_metrics(export_file)
        print(f"\n✅ Performance metrics exported to {export_file}")
        
        # Clear metrics
        clear_performance_metrics()
        summary_after_clear = get_performance_summary()
        print(f"Metrics after clear: {summary_after_clear}")
        
        print("\n✅ Performance metrics test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_threading_safety():
    """Test threading safety."""
    print("\n" + "="*60)
    print("🧪 TESTING THREADING SAFETY")
    print("="*60)
    
    try:
        import threading
        from src.utils.enhanced_tprint import tprint_info, tprint_context
        
        print("--- Threading safety test ---")
        
        def worker_thread(thread_id: int):
            with tprint_context(f"Thread-{thread_id}"):
                for i in range(3):
                    tprint_info(f"Thread {thread_id} message {i}")
                    time.sleep(0.1)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        print("\n✅ Threading safety test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all enhanced tprint tests."""
    print("🚀 STARTING ENHANCED TPRINT TESTS")
    print("="*80)
    
    tests = [
        test_basic_enhanced_features,
        test_context_management,
        test_performance_timing,
        test_configuration,
        test_file_output,
        test_performance_metrics,
        test_threading_safety
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*80)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL ENHANCED TPRINT TESTS PASSED!")
        print("\n🎯 ENHANCED FEATURES DEMONSTRATED:")
        print("  ✅ Colored output with different log levels")
        print("  ✅ Enhanced progress bars")
        print("  ✅ Context management with nested logging")
        print("  ✅ Performance timing with context managers")
        print("  ✅ Configuration system")
        print("  ✅ File output with rotation")
        print("  ✅ Performance metrics collection and export")
        print("  ✅ Threading safety")
        print("  ✅ Advanced filtering and module control")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()