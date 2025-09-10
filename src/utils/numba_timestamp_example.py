#!/usr/bin/env python3
"""
Example Usage of Numba-Friendly Timestamps

This file demonstrates how to use the numba-friendly timestamp utilities
in various scenarios, including numba-compiled functions.
"""

import numpy as np
import time
from typing import List

# Import the numba-friendly timestamp utilities
from .numba_timestamps import (
    numba_print_with_timestamp,
    numba_print_detailed,
    numba_print_simple,
    numba_print_progress,
    numba_print_performance,
    numba_print_error,
    numba_print_warning,
    numba_print_info,
    numba_print_debug,
    get_numba_timestamp_string,
    get_numba_detailed_timestamp_string,
    get_numba_simple_timestamp_string,
    numba_timer_start,
    numba_timer_elapsed,
    numba_print_timing,
    NUMBA_AVAILABLE
)

try:
    import numba
    from numba import njit
except ImportError:
    numba = None
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def example_basic_usage():
    """Example of basic numba-friendly timestamp usage."""
    print("=== Basic Numba-Friendly Timestamp Usage ===")
    
    # Basic timestamped printing
    numba_print_info("Starting basic example")
    numba_print_warning("This is a warning message")
    numba_print_error("This is an error message")
    numba_print_debug("This is a debug message")
    
    # Detailed timestamps
    numba_print_detailed("This message has detailed timestamp")
    
    # Simple timestamps
    numba_print_simple("This message has simple timestamp")
    
    print()


def example_progress_tracking():
    """Example of progress tracking with timestamps."""
    print("=== Progress Tracking Example ===")
    
    @njit
    def process_data_with_progress(data: List[float]) -> List[float]:
        """Process data with progress tracking."""
        numba_print_info("Starting data processing")
        
        result = []
        total_items = len(data)
        
        for i, item in enumerate(data):
            # Log progress every 25% of completion
            if i % (total_items // 4) == 0 or i == total_items - 1:
                numba_print_progress(i + 1, total_items, f"Processing item {i + 1}")
            
            # Some processing
            processed_item = item * 2.0 + 1.0
            result.append(processed_item)
        
        numba_print_info("Data processing completed")
        return result
    
    # Test the function
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    result = process_data_with_progress(test_data)
    print(f"Processed {len(result)} items")
    print()


def example_performance_monitoring():
    """Example of performance monitoring with timestamps."""
    print("=== Performance Monitoring Example ===")
    
    @njit
    def matrix_operations_with_timing(size: int) -> float:
        """Perform matrix operations with timing."""
        numba_print_info(f"Starting matrix operations for size {size}")
        
        # Start timing
        start_time = numba_timer_start()
        
        # Create matrices
        numba_print_info("Creating matrices")
        matrix_a = np.random.random((size, size))
        matrix_b = np.random.random((size, size))
        
        # Matrix multiplication
        numba_print_info("Performing matrix multiplication")
        result = np.dot(matrix_a, matrix_b)
        
        # Calculate sum
        numba_print_info("Calculating sum")
        total_sum = np.sum(result)
        
        # Print timing
        numba_print_timing("Matrix operations", start_time)
        
        return total_sum
    
    # Test the function
    result = matrix_operations_with_timing(100)
    print(f"Matrix operations result: {result}")
    print()


def example_error_handling():
    """Example of error handling with timestamps."""
    print("=== Error Handling Example ===")
    
    @njit
    def safe_division_with_logging(a: float, b: float) -> float:
        """Safe division with error logging."""
        if b == 0.0:
            numba_print_error(f"Cannot divide {a} by zero")
            return 0.0
        
        if b < 0.0:
            numba_print_warning(f"Dividing by negative number: {b}")
        
        result = a / b
        numba_print_info(f"Division result: {a} / {b} = {result}")
        return result
    
    # Test the function
    print("Testing safe division:")
    result1 = safe_division_with_logging(10.0, 2.0)
    result2 = safe_division_with_logging(10.0, 0.0)
    result3 = safe_division_with_logging(10.0, -2.0)
    print(f"Results: {result1}, {result2}, {result3}")
    print()


def example_timestamp_strings():
    """Example of getting timestamp strings in numba functions."""
    print("=== Timestamp Strings Example ===")
    
    @njit
    def work_with_timestamps() -> str:
        """Work with timestamp strings."""
        # Get different types of timestamps
        simple_ts = get_numba_timestamp_string()
        detailed_ts = get_numba_detailed_timestamp_string()
        micro_ts = get_numba_simple_timestamp_string()
        
        # Create a message with timestamps
        message = f"Simple: {simple_ts}, Detailed: {detailed_ts}, Micro: {micro_ts}"
        
        numba_print_info(f"Timestamp comparison: {message}")
        
        return message
    
    # Test the function
    result = work_with_timestamps()
    print(f"Timestamp result: {result}")
    print()


def example_complex_workflow():
    """Example of a complex workflow with comprehensive logging."""
    print("=== Complex Workflow Example ===")
    
    @njit
    def complex_data_processing(data: List[float], threshold: float) -> dict:
        """Complex data processing with comprehensive logging."""
        numba_print_info("Starting complex data processing workflow")
        
        # Phase 1: Data validation
        numba_print_info("Phase 1: Data validation")
        start_time = numba_timer_start()
        
        valid_count = 0
        for item in data:
            if item > 0.0:
                valid_count += 1
        
        numba_print_timing("Data validation", start_time)
        numba_print_info(f"Valid items: {valid_count}/{len(data)}")
        
        if valid_count == 0:
            numba_print_error("No valid data found")
            return {"error": "No valid data"}
        
        # Phase 2: Data filtering
        numba_print_info("Phase 2: Data filtering")
        start_time = numba_timer_start()
        
        filtered_data = []
        for item in data:
            if item > threshold:
                filtered_data.append(item)
        
        numba_print_timing("Data filtering", start_time)
        numba_print_info(f"Filtered items: {len(filtered_data)}")
        
        # Phase 3: Statistical analysis
        numba_print_info("Phase 3: Statistical analysis")
        start_time = numba_timer_start()
        
        if len(filtered_data) > 0:
            total = sum(filtered_data)
            mean = total / len(filtered_data)
            
            # Calculate variance
            variance_sum = 0.0
            for item in filtered_data:
                diff = item - mean
                variance_sum += diff * diff
            variance = variance_sum / len(filtered_data)
            
            numba_print_timing("Statistical analysis", start_time)
            numba_print_info(f"Mean: {mean:.4f}, Variance: {variance:.4f}")
            
            return {
                "count": len(filtered_data),
                "mean": mean,
                "variance": variance,
                "threshold": threshold
            }
        else:
            numba_print_warning("No data passed threshold filter")
            return {"error": "No data passed threshold"}
    
    # Test the function
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 4.5]
    result = complex_data_processing(test_data, 2.0)
    print(f"Complex workflow result: {result}")
    print()


def main():
    """Run all examples."""
    print("Numba-Friendly Timestamps Examples")
    print("=" * 50)
    print(f"Numba available: {NUMBA_AVAILABLE}")
    print()
    
    try:
        example_basic_usage()
        example_progress_tracking()
        example_performance_monitoring()
        example_error_handling()
        example_timestamp_strings()
        example_complex_workflow()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
