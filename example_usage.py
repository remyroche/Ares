#!/usr/bin/env python3
"""
Simple example demonstrating the improved DataDrivenPeriodSelector.

This script shows how to use the enhanced DataDrivenPeriodSelector
with VectorBT optimizations enabled by default.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.data_driven_periods import (
    DataDrivenPeriodSelector, 
    get_data_driven_periods,
    get_data_driven_periods_with_stats
)

def create_sample_data(n_points: int = 5000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    print(f"📊 Creating sample data with {n_points} points...")
    
    # Create datetime index
    start_date = datetime.now() - timedelta(minutes=n_points * 15)
    dates = pd.date_range(start=start_date, periods=n_points, freq='15min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    }, index=dates)
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    print(f"✅ Sample data created: {data.shape}")
    return data

def basic_example():
    """Basic usage example."""
    print("\n" + "="*50)
    print("🚀 Basic Usage Example")
    print("="*50)
    
    # Create sample data
    data = create_sample_data(3000)
    
    # Simple usage - VectorBT optimizations enabled by default
    print("\n📊 Analyzing data with VectorBT optimizations...")
    periods = get_data_driven_periods(data, target_timeframe="15m", max_periods=6)
    
    print(f"✅ Optimal periods: {periods}")
    print("💡 These periods are optimized for cross-timeframe feature generation")

def advanced_example():
    """Advanced usage with performance monitoring."""
    print("\n" + "="*50)
    print("⚡ Advanced Usage with Performance Monitoring")
    print("="*50)
    
    # Create larger dataset
    data = create_sample_data(10000)
    
    # Create selector instance for detailed control
    selector = DataDrivenPeriodSelector(max_periods=8)
    
    print("\n📊 Running analysis with performance monitoring...")
    
    # Analyze with performance tracking
    result = selector.select_optimal_periods(data, target_timeframe="15m")
    
    # Get performance statistics
    stats = selector.get_performance_stats()
    
    print(f"\n✅ Analysis Results:")
    print(f"   Optimal periods: {result.optimal_periods}")
    print(f"   Confidence score: {result.confidence_score:.2f}")
    print(f"   Period categories: {result.period_categories}")
    
    print(f"\n📊 Performance Statistics:")
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   VectorBT operations: {stats['vectorbt_operations']}")
    print(f"   Batch operations: {stats['batch_operations']}")
    print(f"   Memory optimizations: {stats['memory_optimizations']}")
    print(f"   VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.1%}")
    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")

def convenience_functions_example():
    """Demonstrate convenience functions."""
    print("\n" + "="*50)
    print("🔧 Convenience Functions Example")
    print("="*50)
    
    data = create_sample_data(2000)
    
    # Method 1: Simple function call
    print("\n📊 Method 1: Simple function call")
    periods = get_data_driven_periods(data, max_periods=5)
    print(f"   Periods: {periods}")
    
    # Method 2: Function with performance stats
    print("\n📊 Method 2: Function with performance stats")
    periods, stats = get_data_driven_periods_with_stats(data, max_periods=5)
    print(f"   Periods: {periods}")
    print(f"   VectorBT usage: {stats.get('vectorbt_usage_rate', 0):.1%}")
    
    # Method 3: Custom configuration
    print("\n📊 Method 3: Custom configuration")
    selector = DataDrivenPeriodSelector(
        max_periods=6,
        chunk_size=500,  # Smaller chunks for memory-constrained environments
        memory_efficient=True  # Explicitly enable memory optimization
    )
    result = selector.select_optimal_periods(data)
    print(f"   Periods: {result.optimal_periods}")

def memory_optimization_example():
    """Demonstrate memory optimization features."""
    print("\n" + "="*50)
    print("💾 Memory Optimization Example")
    print("="*50)
    
    # Create large dataset
    data = create_sample_data(50000)
    original_memory = data.memory_usage(deep=True).sum() / (1024**2)
    print(f"📊 Original data memory usage: {original_memory:.1f} MB")
    
    # Create selector with memory optimization
    selector = DataDrivenPeriodSelector(memory_efficient=True, chunk_size=5000)
    
    # Optimize data
    optimized_data = selector.optimize_for_large_datasets(data)
    optimized_memory = optimized_data.memory_usage(deep=True).sum() / (1024**2)
    print(f"📊 Optimized data memory usage: {optimized_memory:.1f} MB")
    print(f"💾 Memory savings: {((original_memory - optimized_memory) / original_memory * 100):.1f}%")
    
    # Analyze optimized data
    result = selector.select_optimal_periods(optimized_data)
    print(f"✅ Analysis completed with {len(result.optimal_periods)} optimal periods")

def main():
    """Run all examples."""
    print("🚀 DataDrivenPeriodSelector Examples")
    print("VectorBT optimizations enabled by default")
    print("="*60)
    
    try:
        # Run examples
        basic_example()
        advanced_example()
        convenience_functions_example()
        memory_optimization_example()
        
        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("="*60)
        
        print("\n📋 Key Benefits Demonstrated:")
        print("✅ VectorBT optimizations enabled by default")
        print("✅ Simple, clean API")
        print("✅ Performance monitoring and statistics")
        print("✅ Memory optimization for large datasets")
        print("✅ Multiple usage patterns supported")
        print("✅ Backward compatible with existing code")
        
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)