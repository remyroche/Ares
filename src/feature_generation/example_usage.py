#!/usr/bin/env python3
"""
Example Usage of New Feature Generation System

This script demonstrates how to use the new consolidated feature generation system
with the new utility mixins, factory pattern, and optimized base classes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the new features
from src.feature_generation import (
    VectorizedFeatureGenerator,
    OptimizationMixin,
    RollingOperationsMixin,
    VectorBTOptimizationMixin,
    GeneratorFactory,
    create_generator,
    FeatureConfig,
    FeatureCategory
)

def create_sample_data() -> pd.DataFrame:
    """Create sample financial data for demonstration."""
    np.random.seed(42)
    n_points = 1000
    
    # Generate sample OHLCV data
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n_points) * 0.1),
        'high': 100 + np.cumsum(np.random.randn(n_points) * 0.1) + np.abs(np.random.randn(n_points) * 0.5),
        'low': 100 + np.cumsum(np.random.randn(n_points) * 0.1) - np.abs(np.random.randn(n_points) * 0.5),
        'close': 100 + np.cumsum(np.random.randn(n_points) * 0.1),
        'volume': np.random.randint(1000, 10000, n_points)
    })
    
    # Ensure high >= low and high >= close >= low
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def example_1_basic_usage():
    """Example 1: Basic usage with VectorizedFeatureGenerator."""
    print("🚀 Example 1: Basic Usage with VectorizedFeatureGenerator")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    print(f"📊 Sample data shape: {data.shape}")
    
    # Create feature config
    config = FeatureConfig(
        name="sma_20",
        category=FeatureCategory.CUSTOM,
        description="Simple Moving Average with 20-period window",
        required_columns=["close"],
        default_lookback=20
    )
    
    # Create a simple generator
    class SimpleMovingAverageGenerator(VectorizedFeatureGenerator):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            return data['close'].rolling(window=20).mean()
    
    # Create generator instance
    generator = SimpleMovingAverageGenerator(config)
    
    # Generate feature
    result = generator.generate(data)
    
    print(f"✅ Feature generated successfully: {result.success}")
    print(f"⏱️ Computation time: {result.computation_time:.3f}s")
    print(f"📈 Feature data length: {len(result.data)}")
    print(f"📊 First 5 values: {result.data.head().values}")
    print()

def example_2_with_mixins():
    """Example 2: Using utility mixins for enhanced functionality."""
    print("🚀 Example 2: Using Utility Mixins")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Create feature config
    config = FeatureConfig(
        name="enhanced_sma",
        category=FeatureCategory.CUSTOM,
        description="Enhanced SMA with optimization and rolling operations",
        required_columns=["close"],
        default_lookback=20
    )
    
    # Create generator with mixins
    class EnhancedSMAGenerator(VectorizedFeatureGenerator, OptimizationMixin, RollingOperationsMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Use optimization mixin
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Use rolling operations mixin
            return self.rolling_mean(optimized_data['close'], window=20)
    
    # Create generator instance
    generator = EnhancedSMAGenerator(config)
    
    # Generate feature
    result = generator.generate(data)
    
    print(f"✅ Enhanced feature generated: {result.success}")
    print(f"⏱️ Computation time: {result.computation_time:.3f}s")
    
    # Check optimization stats
    opt_stats = generator.get_optimization_stats()
    print(f"🔧 Memory optimizations: {opt_stats['memory_optimizations']}")
    print(f"💾 Memory saved: {opt_stats['memory_saved_mb']:.2f}MB")
    
    # Check rolling stats
    rolling_stats = generator.get_rolling_stats()
    print(f"📊 Rolling operations: {rolling_stats['operations_count']}")
    print(f"⚡ VectorBT usage: {rolling_stats['vectorbt_usage_percentage']:.1f}%")
    print()

def example_3_factory_pattern():
    """Example 3: Using the factory pattern for generator creation."""
    print("🚀 Example 3: Factory Pattern Usage")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Get factory instance
    factory = GeneratorFactory()
    
    # Create a vectorized generator using factory
    generator = factory.create_vectorized_generator(
        name="factory_sma",
        category=FeatureCategory.CUSTOM,
        required_columns=["close"],
        window=20
    )
    
    if generator:
        print(f"✅ Generator created via factory: {generator.__class__.__name__}")
        
        # Create an optimized generator
        optimized_generator = factory.create_optimized_generator(
            name="optimized_sma",
            category=FeatureCategory.CUSTOM,
            required_columns=["close"]
        )
        
        if optimized_generator:
            print(f"✅ Optimized generator created: {optimized_generator.__class__.__name__}")
            print(f"🔧 Has optimization mixin: {hasattr(optimized_generator, 'optimize_dataframe_processing')}")
            print(f"📊 Has rolling mixin: {hasattr(optimized_generator, 'rolling_mean')}")
    else:
        print("❌ Failed to create generator via factory")
    print()

def example_4_batch_operations():
    """Example 4: Batch operations with rolling operations mixin."""
    print("🚀 Example 4: Batch Operations")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Create feature config
    config = FeatureConfig(
        name="batch_features",
        category=FeatureCategory.CUSTOM,
        description="Batch rolling operations example",
        required_columns=["close", "volume"],
        default_lookback=20
    )
    
    # Create generator with rolling operations mixin
    class BatchFeatureGenerator(VectorizedFeatureGenerator, RollingOperationsMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Define batch operations
            operations = [
                {'column': 'close', 'operation': 'mean', 'window': 20, 'name': 'close_sma_20'},
                {'column': 'close', 'operation': 'std', 'window': 20, 'name': 'close_std_20'},
                {'column': 'volume', 'operation': 'mean', 'window': 20, 'name': 'volume_sma_20'},
                {'column': 'volume', 'operation': 'sum', 'window': 20, 'name': 'volume_sum_20'}
            ]
            
            # Perform batch operations
            result_df = self.batch_rolling_operations(data, operations)
            
            # Return the first feature as the main result
            return result_df['close_sma_20']
    
    # Create generator instance
    generator = BatchFeatureGenerator(config)
    
    # Generate feature
    result = generator.generate(data)
    
    print(f"✅ Batch operations completed: {result.success}")
    print(f"⏱️ Computation time: {result.computation_time:.3f}s")
    
    # Check rolling stats
    rolling_stats = generator.get_rolling_stats()
    print(f"📊 Total rolling operations: {rolling_stats['operations_count']}")
    print(f"⚡ VectorBT operations: {rolling_stats['vectorbt_operations']}")
    print(f"🐼 Pandas operations: {rolling_stats['pandas_operations']}")
    print()

def example_5_performance_comparison():
    """Example 5: Performance comparison between old and new approaches."""
    print("🚀 Example 5: Performance Comparison")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data()
    
    # Old approach (simulated)
    import time
    
    start_time = time.time()
    old_result = data['close'].rolling(window=20).mean()
    old_time = time.time() - start_time
    
    # New approach with optimization
    config = FeatureConfig(
        name="performance_test",
        category=FeatureCategory.CUSTOM,
        description="Performance test",
        required_columns=["close"],
        default_lookback=20
    )
    
    class PerformanceTestGenerator(VectorizedFeatureGenerator, OptimizationMixin, RollingOperationsMixin):
        def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
            # Use optimized approach
            optimized_data = self.optimize_dataframe_processing(data)
            return self.rolling_mean(optimized_data['close'], window=20)
    
    generator = PerformanceTestGenerator(config)
    result = generator.generate(data)
    
    print(f"📊 Data size: {len(data):,} rows")
    print(f"⏱️ Old approach time: {old_time:.3f}s")
    print(f"⏱️ New approach time: {result.computation_time:.3f}s")
    
    if result.computation_time < old_time:
        speedup = old_time / result.computation_time
        print(f"🚀 Speedup: {speedup:.2f}x faster")
    else:
        slowdown = result.computation_time / old_time
        print(f"⚠️ Slowdown: {slowdown:.2f}x slower")
    
    # Show optimization stats
    opt_stats = generator.get_optimization_stats()
    rolling_stats = generator.get_rolling_stats()
    
    print(f"🔧 Memory optimizations: {opt_stats['memory_optimizations']}")
    print(f"📊 Rolling operations: {rolling_stats['operations_count']}")
    print()

def main():
    """Run all examples."""
    print("🎯 Feature Generation System - New Features Demo")
    print("=" * 80)
    print()
    
    try:
        example_1_basic_usage()
        example_2_with_mixins()
        example_3_factory_pattern()
        example_4_batch_operations()
        example_5_performance_comparison()
        
        print("🎉 All examples completed successfully!")
        print("=" * 80)
        print("📚 Key Benefits of New System:")
        print("  ✅ Eliminated 100+ duplicate methods")
        print("  ✅ Enhanced base class with VectorBT optimization")
        print("  ✅ Utility mixins for common functionality")
        print("  ✅ Factory pattern for easy generator creation")
        print("  ✅ Comprehensive documentation and examples")
        print("  ✅ Backward compatibility maintained")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()