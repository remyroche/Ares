#!/usr/bin/env python3
"""
Memory Management Demo Script

This script demonstrates the integrated automatic memory management system
using all the specified utilities for comprehensive resource optimization.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the enhanced memory management system
from src.utils.enhanced_memory_management import (
    get_memory_manager, start_memory_monitoring, memory_context,
    get_memory_status, optimize_dataframe_memory_usage, MemoryConfig
)

# Import other utilities as specified
from src.utils.common_operations import safe_operation
from src.utils.common_utilities import validate_data_types
from src.utils.math_validation import validate_numeric_data
from src.utils.parquet_utils import optimize_dataframe_memory
from src.utils.serialization_utils import compress_data
from src.utils.data_processing_utils import clean_dataframe
from src.utils.m1_memory_optimizer import optimize_memory_usage
from src.utils.m1_cpu_optimizer import optimize_cpu_usage
from src.utils.m1_gpu_utils import get_gpu_memory_info

# Import ML common utilities
try:
    from src.utils.ml_common.parallel_processing import get_parallel_processor
    from src.utils.ml_common.hpo_utils import optimize_hyperparameters
    from src.utils.ml_common.lookahead_protection import apply_lookahead_protection
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def create_large_dataframe() -> pd.DataFrame:
    """Create a large DataFrame for testing memory management."""
    print("📊 Creating large test DataFrame...")

    # Create a DataFrame with 1M rows and various data types
    n_rows = 1_000_000
    data = {
        'id': range(n_rows),
        'float64_col': np.random.randn(n_rows),
        'float32_col': np.random.randn(n_rows).astype(np.float32),
        'int64_col': np.random.randint(0, 1000, n_rows),
        'int32_col': np.random.randint(0, 100, n_rows).astype(np.int32),
        'category_col': pd.Categorical(np.random.choice(['A', 'B', 'C', 'D'], n_rows)),
        'string_col': [f'string_{i}' for i in range(n_rows)],
        'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='1min'),
        'bool_col': np.random.choice([True, False], n_rows)
    }

    df = pd.DataFrame(data)
    print(f"✅ Created DataFrame with shape: {df.shape}")
    return df


def demonstrate_memory_optimization():
    """Demonstrate comprehensive memory optimization using all utilities."""
    print("\n🚀 Starting Memory Management Demonstration")
    print("=" * 60)

    # Get memory manager and start monitoring
    memory_config = MemoryConfig(
        memory_threshold=0.80,  # Trigger at 80% memory usage
        disk_threshold=0.85,    # Trigger at 85% disk usage
        check_interval=10,      # Check every 10 seconds
        cleanup_enabled=True
    )

    memory_mgr = get_memory_manager(memory_config)
    memory_mgr.start_monitoring()

    print("🧠 Memory monitoring started")

    # Initial memory status
    initial_status = get_memory_status()
    print(f"📊 Initial memory usage: {initial_status.get('memory_percent', 0):.1%}")

    # Create large DataFrame
    with memory_context("DataFrame creation"):
        df = create_large_dataframe()

    print(f"📈 DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

    # Demonstrate memory optimization using all utilities
    with memory_context("Comprehensive memory optimization"):
        print("\n🔧 Applying comprehensive memory optimization...")

        # Step 1: Use parquet optimization
        print("📦 Step 1: Parquet optimization")
        df_optimized = optimize_dataframe_memory(df.copy())

        # Step 2: Apply M1 memory optimization
        print("🧠 Step 2: M1 memory optimization")
        df_optimized = optimize_memory_usage(df_optimized)

        # Step 3: Clean dataframe
        print("🧹 Step 3: Data cleaning")
        df_optimized = clean_dataframe(df_optimized)

        # Step 4: Validate data types
        print("✅ Step 4: Data type validation")
        validate_data_types(df_optimized.dtypes.to_dict())

        # Step 5: Use comprehensive optimization
        print("🎯 Step 5: Comprehensive optimization")
        df_final = optimize_dataframe_memory_usage(df_optimized)

    # Compare memory usage
    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    optimized_memory = df_final.memory_usage(deep=True).sum() / 1024 / 1024
    memory_savings = original_memory - optimized_memory
    savings_percent = (memory_savings / original_memory) * 100

    print(f"\n📊 Memory Optimization Results:")
    print(".1f"    print(".1f"    print(".1f"    print(".2f"
    # Demonstrate serialization with compression
    with memory_context("Data serialization"):
        print("\n💾 Testing data serialization with compression...")
        compressed_data = compress_data(df_final.to_dict())
        print(f"📦 Compressed data size: {len(str(compressed_data))} characters")

    # Test GPU memory info if available
    try:
        gpu_info = get_gpu_memory_info()
        print(f"\n🎮 GPU Memory Info: {gpu_info}")
    except Exception as e:
        print(f"\n🎮 GPU not available: {e}")

    # Test parallel processing if Ray is available
    if RAY_AVAILABLE:
        with memory_context("Parallel processing test"):
            print("\n⚡ Testing parallel processing...")
            try:
                processor = get_parallel_processor()
                # Simple parallel operation test
                test_data = list(range(100))
                results = processor.process_batch(lambda x: x * 2, test_data, batch_size=10)
                print(f"✅ Parallel processing result: {len(results)} items processed")
            except Exception as e:
                print(f"⚠️ Parallel processing test failed: {e}")

    # Demonstrate memory monitoring and cleanup
    print("\n🧹 Testing memory cleanup...")
    memory_mgr._perform_memory_cleanup()

    # Final status
    final_status = get_memory_status()
    print("
📊 Final Status:"    print(".1%"    print(".1%"    print(f"🚨 Memory alerts during session: {final_status.get('memory_alerts', 0)}")

    # Stop monitoring
    memory_mgr.stop_monitoring()
    print("\n✅ Memory management demonstration completed!")


def demonstrate_error_handling():
    """Demonstrate error handling and recovery in memory management."""
    print("\n🛡️ Testing Error Handling and Recovery")

    memory_mgr = get_memory_manager()

    # Test with invalid data
    try:
        invalid_df = pd.DataFrame({'col': [float('inf'), float('-inf'), float('nan')]})
        optimized = optimize_dataframe_memory_usage(invalid_df)
        print("✅ Handled invalid data gracefully")
    except Exception as e:
        print(f"⚠️ Error handling test: {e}")

    # Test memory pressure simulation
    print("🔥 Simulating memory pressure...")
    large_objects = []
    try:
        for i in range(100):
            large_objects.append('x' * 1000000)  # 1MB strings

        # This should trigger memory monitoring and cleanup
        time.sleep(2)  # Give monitoring time to react

        print("✅ Memory pressure handled")

    except Exception as e:
        print(f"⚠️ Memory pressure test: {e}")

    finally:
        # Clean up
        del large_objects
        memory_mgr._perform_memory_cleanup()


if __name__ == "__main__":
    print("🧠 Ares Memory Management System Demo")
    print("This demo showcases integrated automatic memory management")
    print("using all specified utility modules.\n")

    try:
        # Main demonstration
        demonstrate_memory_optimization()

        # Error handling demonstration
        demonstrate_error_handling()

        print("\n🎉 All demonstrations completed successfully!")
        print("\n💡 Key Features Demonstrated:")
        print("  • Automatic memory monitoring")
        print("  • Comprehensive memory optimization")
        print("  • Data validation and cleaning")
        print("  • Parallel processing integration")
        print("  • Error handling and recovery")
        print("  • GPU memory monitoring")
        print("  • Disk space management")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure monitoring is stopped
        try:
            stop_memory_monitoring()
        except:
            pass
