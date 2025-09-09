#!/usr/bin/env python3
"""
Example script demonstrating automatic memory skimming functionality.

This script shows how to use the auto memory skimming features to automatically
free unused memory whenever more memory is needed.
"""

import numpy as np
import pandas as pd
import time
from src.utils.m1_memory_optimizer import (
    auto_skim_memory,
    smart_memory_allocation,
    memory_skim_decorator,
    auto_memory_skim_decorator,
    auto_memory_skim_context,
    smart_memory_context,
    get_m1_memory_optimizer
)

def demonstrate_basic_skimming():
    """Demonstrate basic auto memory skimming."""
    print("🔍 === Basic Auto Memory Skimming ===")
    
    # Simulate needing 1GB of memory
    result = auto_skim_memory(required_memory_mb=1024, operation_type="data_processing")
    
    print(f"Skimming needed: {result['skimming_needed']}")
    print(f"Memory freed: {result['memory_freed_mb']:.1f}MB")
    print(f"Available memory: {result['available_mb']:.1f}MB")
    print(f"Required memory: {result['required_mb']:.1f}MB")
    
    if result['skimming_needed']:
        print("📊 Skimming steps:")
        for i, step in enumerate(result['skimming_steps'], 1):
            print(f"  {i}. {step['cleanup_type']}: {step['memory_freed_mb']:.1f}MB freed")
            print(f"     Operations: {', '.join(step['operations'])}")
    
    print()

def demonstrate_smart_allocation():
    """Demonstrate smart memory allocation."""
    print("🧠 === Smart Memory Allocation ===")
    
    # Simulate needing 2GB of memory
    allocation = smart_memory_allocation(required_memory_mb=2048, operation_type="neural_net")
    
    print(f"Allocation successful: {allocation['allocation_successful']}")
    print(f"Skimming performed: {allocation['skimming_performed']}")
    print(f"Available memory: {allocation['available_mb']:.1f}MB")
    print(f"Required memory: {allocation['required_mb']:.1f}MB")
    
    if allocation['skimming_performed']:
        print(f"Total memory freed: {allocation['skimming_results']['memory_freed_mb']:.1f}MB")
    
    print()

def demonstrate_context_managers():
    """Demonstrate context managers for automatic memory management."""
    print("🎯 === Context Managers ===")
    
    # Auto memory skim context
    print("Using auto_memory_skim_context:")
    with auto_memory_skim_context(required_memory_mb=1500, operation_type="matrix_mult") as context:
        if context['skimming_performed']:
            print(f"  🧹 Skimming performed: {context['skimming_results']['memory_freed_mb']:.1f}MB freed")
        else:
            print("  ✅ Sufficient memory available")
        
        # Simulate some work
        print("  🔄 Performing matrix operations...")
        time.sleep(0.1)
    
    print()
    
    # Smart memory context
    print("Using smart_memory_context:")
    with smart_memory_context(required_memory_mb=1000, operation_type="data_processing") as allocation:
        if allocation['allocation_successful']:
            print("  ✅ Memory allocation successful")
            if allocation['skimming_performed']:
                print(f"  🧹 Skimming freed: {allocation['skimming_results']['memory_freed_mb']:.1f}MB")
        else:
            print("  ⚠️ Insufficient memory")
        
        # Simulate some work
        print("  🔄 Processing data...")
        time.sleep(0.1)
    
    print()

def demonstrate_decorators():
    """Demonstrate decorators for automatic memory management."""
    print("🏷️ === Decorators ===")
    
    # Memory skim decorator
    @memory_skim_decorator(required_memory_mb=800, operation_type="data_processing")
    def process_large_data():
        """Simulate processing large data."""
        print("  🔄 Processing large dataset...")
        time.sleep(0.1)
        return "Data processed successfully"
    
    print("Using memory_skim_decorator:")
    result = process_large_data()
    print(f"  Result: {result}")
    print()
    
    # Auto memory skim decorator
    @auto_memory_skim_decorator(operation_type="neural_net")
    def train_model(data_size):
        """Simulate model training with automatic memory estimation."""
        print(f"  🔄 Training model with {data_size} samples...")
        time.sleep(0.1)
        return f"Model trained on {data_size} samples"
    
    print("Using auto_memory_skim_decorator:")
    result = train_model(10000)
    print(f"  Result: {result}")
    print()

def demonstrate_memory_monitoring():
    """Demonstrate memory monitoring capabilities."""
    print("📊 === Memory Monitoring ===")
    
    optimizer = get_m1_memory_optimizer()
    
    # Get current memory usage
    memory_info = optimizer.get_memory_usage()
    print(f"Current memory usage: {memory_info['rss_gb']:.1f}GB")
    print(f"Available memory: {memory_info['available_gb']:.1f}GB")
    print(f"Memory percentage: {memory_info['percentage']:.1f}%")
    
    # Check if chunking is needed
    should_chunk = optimizer.should_chunk_data(data_size_mb=1000, operation_type="general")
    print(f"Chunking needed for 1GB operation: {should_chunk}")
    
    # Calculate optimal chunk size
    optimal_size = optimizer.calculate_optimal_chunk_size((10000, 100), "general")
    print(f"Optimal chunk size for (10000, 100) data: {optimal_size}")
    
    print()

def demonstrate_custom_cleanup():
    """Demonstrate custom memory cleanup levels."""
    print("🧹 === Custom Memory Cleanup ===")
    
    optimizer = get_m1_memory_optimizer()
    
    # Light cleanup
    print("Performing light cleanup:")
    light_result = optimizer._light_memory_cleanup()
    print(f"  Memory freed: {light_result['memory_freed_mb']:.1f}MB")
    print(f"  Operations: {', '.join(light_result['operations'])}")
    
    # Moderate cleanup
    print("Performing moderate cleanup:")
    moderate_result = optimizer._moderate_memory_cleanup()
    print(f"  Memory freed: {moderate_result['memory_freed_mb']:.1f}MB")
    print(f"  Operations: {', '.join(moderate_result['operations'])}")
    
    # Aggressive cleanup
    print("Performing aggressive cleanup:")
    aggressive_result = optimizer._aggressive_memory_cleanup()
    print(f"  Memory freed: {aggressive_result['memory_freed_mb']:.1f}MB")
    print(f"  Operations: {', '.join(aggressive_result['operations'])}")
    
    print()

def main():
    """Main demonstration function."""
    print("🚀 Auto Memory Skimming Demonstration")
    print("=" * 50)
    print()
    
    try:
        demonstrate_basic_skimming()
        demonstrate_smart_allocation()
        demonstrate_context_managers()
        demonstrate_decorators()
        demonstrate_memory_monitoring()
        demonstrate_custom_cleanup()
        
        print("✅ All demonstrations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
