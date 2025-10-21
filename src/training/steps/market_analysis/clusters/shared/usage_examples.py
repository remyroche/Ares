"""
Usage examples for shared utilities.

This module demonstrates how to use the shared utilities to eliminate
code duplication across the clustering codebase.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .hardware_initializer import HardwareInitializer, HardwareContext
from .validation_utils import ClusteringValidationUtils, ValidationResult
from .common_utils import (
    ClusteringCommonUtils, 
    clustering_operation, 
    memory_optimized,
    performance_tracked,
    safe_execution
)


def example_hardware_initialization():
    """Example of using centralized hardware initialization."""
    
    # Method 1: Direct initialization
    hardware_components = HardwareInitializer.initialize_hardware_components(
        "example_component", verbose=True
    )
    
    gpu_manager = hardware_components.get('gpu_manager')
    memory_manager = hardware_components.get('memory_manager')
    cpu_optimizer = hardware_components.get('cpu_optimizer')
    
    # Method 2: Context manager for automatic cleanup
    with HardwareContext("example_component") as hw:
        if hw['initialization_successful']:
            print("Hardware initialized successfully")
            # Use hardware components
            pass
        # Automatic cleanup happens here


def example_validation_patterns():
    """Example of using centralized validation."""
    
    # Create sample data
    features = np.random.randn(100, 10)
    assignments = np.random.randint(0, 3, 100)
    market_data = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
    
    # Feature validation
    feature_result = ClusteringValidationUtils.validate_features(
        features, 
        min_samples=50,
        min_features=5
    )
    
    if not feature_result.is_valid:
        print(f"Feature validation failed: {feature_result.errors}")
    else:
        print(f"Feature validation passed: {feature_result.get_summary()}")
    
    # Assignment validation
    assignment_result = ClusteringValidationUtils.validate_clustering_assignments(
        assignments, 
        expected_length=100,
        min_clusters=2
    )
    
    print(f"Assignment validation: {assignment_result.get_summary()}")
    
    # Market data validation
    market_result = ClusteringValidationUtils.validate_market_data(
        market_data,
        required_columns=['A', 'B', 'C']
    )
    
    print(f"Market data validation: {market_result.get_summary()}")


def example_common_utilities():
    """Example of using common utilities."""
    
    # Safe mathematical operations
    result1 = ClusteringCommonUtils.safe_divide(10, 0, default=0.0)
    result2 = ClusteringCommonUtils.safe_log(-5, default=0.0)
    result3 = ClusteringCommonUtils.safe_sqrt(-4, default=0.0)
    
    print(f"Safe operations: {result1}, {result2}, {result3}")
    
    # Memory management
    large_array = np.random.randn(1000, 1000)
    memory_usage = ClusteringCommonUtils.get_memory_usage_mb(large_array)
    print(f"Memory usage: {ClusteringCommonUtils.format_memory_size(memory_usage)}")
    
    # Cleanup
    ClusteringCommonUtils.memory_cleanup(large_array)


@clustering_operation("example_operation", verbose=True)
@memory_optimized("moderate")
@performance_tracked("example_operation")
def example_decorated_function(data: np.ndarray) -> np.ndarray:
    """Example function using decorators."""
    
    # This function is automatically:
    # - Memory optimized
    # - Performance tracked
    # - Error handled with logging
    
    result = data * 2
    return result


@safe_execution("Data processing failed", verbose=True)
def example_safe_execution(data: np.ndarray) -> np.ndarray:
    """Example of safe execution with error handling."""
    
    # This function will automatically handle errors and cleanup
    if data is None:
        raise ValueError("Data cannot be None")
    
    return data * 3


def example_chunked_processing():
    """Example of chunked processing for large datasets."""
    
    # Create large dataset
    large_data = np.random.randn(10000, 100)
    
    def process_chunk(chunk: np.ndarray) -> float:
        """Process a single chunk."""
        return np.mean(chunk)
    
    # Process in chunks
    results = ClusteringCommonUtils.chunked_processing(
        large_data,
        chunk_size=1000,
        process_func=process_chunk
    )
    
    print(f"Processed {len(results)} chunks")
    return results


def example_comprehensive_validation():
    """Example of comprehensive validation workflow."""
    
    # Create sample clustering data
    features = np.random.randn(200, 15)
    assignments = np.random.randint(0, 4, 200)
    market_data = pd.DataFrame(np.random.randn(200, 8))
    
    # Comprehensive validation
    validation_results = {}
    
    # Feature validation
    validation_results['features'] = ClusteringValidationUtils.safe_validate_with_logging(
        ClusteringValidationUtils.validate_features,
        features,
        min_samples=100,
        min_features=10
    )
    
    # Assignment validation
    validation_results['assignments'] = ClusteringValidationUtils.safe_validate_with_logging(
        ClusteringValidationUtils.validate_clustering_assignments,
        assignments,
        expected_length=200,
        min_clusters=2
    )
    
    # Market data validation
    validation_results['market_data'] = ClusteringValidationUtils.safe_validate_with_logging(
        ClusteringValidationUtils.validate_market_data,
        market_data,
        min_rows=100
    )
    
    # Summary
    all_valid = all(result.is_valid for result in validation_results.values())
    print(f"Overall validation: {'✅ PASSED' if all_valid else '❌ FAILED'}")
    
    for name, result in validation_results.items():
        print(f"{name}: {result.get_summary()}")


if __name__ == "__main__":
    """Run all examples."""
    
    print("=== Hardware Initialization Examples ===")
    example_hardware_initialization()
    
    print("\n=== Validation Pattern Examples ===")
    example_validation_patterns()
    
    print("\n=== Common Utilities Examples ===")
    example_common_utilities()
    
    print("\n=== Decorated Function Examples ===")
    data = np.random.randn(100, 10)
    result = example_decorated_function(data)
    print(f"Decorated function result shape: {result.shape}")
    
    print("\n=== Safe Execution Examples ===")
    try:
        result = example_safe_execution(data)
        print(f"Safe execution result shape: {result.shape}")
    except Exception as e:
        print(f"Safe execution caught error: {e}")
    
    print("\n=== Chunked Processing Examples ===")
    chunk_results = example_chunked_processing()
    print(f"Chunked processing completed: {len(chunk_results)} results")
    
    print("\n=== Comprehensive Validation Examples ===")
    example_comprehensive_validation()
    
    print("\n=== All Examples Completed ===")