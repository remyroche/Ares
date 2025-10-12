#!/usr/bin/env python3
"""
Test script for the improved DataDrivenPeriodSelector with VectorBT optimizations.

This script demonstrates the performance improvements and new features
of the enhanced DataDrivenPeriodSelector.
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.pre_training.interaction_feature_generator.feature_interaction_generation.data_driven_periods import (
    DataDrivenPeriodSelector, 
    get_data_driven_periods,
    get_data_driven_periods_with_stats,
    benchmark_period_selector
)

def create_sample_data(n_points: int = 10000, timeframe_minutes: int = 15) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    print(f"📊 Creating sample data with {n_points} points...")
    
    # Create datetime index
    start_date = datetime.now() - timedelta(minutes=n_points * timeframe_minutes)
    dates = pd.date_range(start=start_date, periods=n_points, freq=f'{timeframe_minutes}min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, n_points)  # 2% daily volatility
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
    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
    print(f"💰 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    return data

def test_basic_functionality():
    """Test basic functionality of the improved DataDrivenPeriodSelector."""
    print("\n" + "="*60)
    print("🧪 Testing Basic Functionality")
    print("="*60)
    
    # Create sample data
    data = create_sample_data(n_points=5000)
    
    # Test with VectorBT optimizations (enabled by default)
    print("\n🚀 Testing with VectorBT optimizations (default configuration)...")
    selector = DataDrivenPeriodSelector(max_periods=6)
    
    start_time = time.time()
    result = selector.select_optimal_periods(data, target_timeframe="15m")
    execution_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {execution_time:.3f}s")
    print(f"📊 Optimal periods: {result.optimal_periods}")
    print(f"📈 Confidence score: {result.confidence_score:.2f}")
    print(f"🏷️ Period categories: {result.period_categories}")
    
    # Get performance stats
    stats = selector.get_performance_stats()
    print(f"\n📊 Performance Statistics:")
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   VectorBT operations: {stats['vectorbt_operations']}")
    print(f"   Batch operations: {stats['batch_operations']}")
    print(f"   Memory optimizations: {stats['memory_optimizations']}")
    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
    
    return result, stats

def test_performance_comparison():
    """Compare performance between different configurations."""
    print("\n" + "="*60)
    print("⚡ Performance Comparison")
    print("="*60)
    
    # Create larger dataset for meaningful comparison
    data = create_sample_data(n_points=20000)
    
    # Run benchmark
    print("🔄 Running performance benchmark...")
    benchmark_results = benchmark_period_selector(data, trials=3)
    
    print("\n📊 Benchmark Results:")
    print("-" * 50)
    for config_name, results in benchmark_results.items():
        print(f"{config_name:20} | {results['avg_time']:.3f}s ± {results['std_time']:.3f}s | {results['trials_completed']} trials")
    
    # Find best configuration
    best_config = min(benchmark_results.items(), key=lambda x: x[1]['avg_time'])
    print(f"\n🏆 Best configuration: {best_config[0]} ({best_config[1]['avg_time']:.3f}s)")
    
    return benchmark_results

def test_memory_efficiency():
    """Test memory efficiency features."""
    print("\n" + "="*60)
    print("💾 Memory Efficiency Test")
    print("="*60)
    
    # Create large dataset
    data = create_sample_data(n_points=50000)
    print(f"📊 Original data memory usage: {data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Test with memory optimization (enabled by default)
    selector = DataDrivenPeriodSelector(chunk_size=5000)
    
    # Optimize data
    optimized_data = selector.optimize_for_large_datasets(data)
    print(f"📊 Optimized data memory usage: {optimized_data.memory_usage(deep=True).sum() / (1024**2):.1f} MB")
    
    # Test analysis
    start_time = time.time()
    result = selector.select_optimal_periods(optimized_data)
    execution_time = time.time() - start_time
    
    print(f"✅ Analysis completed in {execution_time:.3f}s")
    print(f"📊 Optimal periods: {result.optimal_periods}")
    
    return result

def test_convenience_functions():
    """Test the new convenience functions."""
    print("\n" + "="*60)
    print("🔧 Convenience Functions Test")
    print("="*60)
    
    data = create_sample_data(n_points=3000)
    
    # Test basic convenience function
    print("🔄 Testing basic convenience function...")
    periods = get_data_driven_periods(data, target_timeframe="15m", max_periods=5)
    print(f"✅ Basic function result: {periods}")
    
    # Test function with stats
    print("\n🔄 Testing function with performance stats...")
    periods, stats = get_data_driven_periods_with_stats(
        data, 
        target_timeframe="15m", 
        max_periods=5
    )
    print(f"✅ Periods: {periods}")
    print(f"📊 VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.1%}")
    print(f"📊 Batch usage rate: {stats.get('batch_usage_rate', 0):.1%}")
    
    return periods, stats

def test_caching():
    """Test caching functionality."""
    print("\n" + "="*60)
    print("💾 Caching Test")
    print("="*60)
    
    data = create_sample_data(n_points=2000)
    
    selector = DataDrivenPeriodSelector()
    
    # First run (cache miss)
    print("🔄 First run (should be cache miss)...")
    start_time = time.time()
    result1 = selector.select_optimal_periods(data)
    time1 = time.time() - start_time
    stats1 = selector.get_performance_stats()
    
    # Second run (cache hit)
    print("🔄 Second run (should be cache hit)...")
    start_time = time.time()
    result2 = selector.select_optimal_periods(data)
    time2 = time.time() - start_time
    stats2 = selector.get_performance_stats()
    
    print(f"✅ First run: {time1:.3f}s, Cache hits: {stats1['cache_hits']}")
    print(f"✅ Second run: {time2:.3f}s, Cache hits: {stats2['cache_hits']}")
    print(f"📊 Speedup: {time1/time2:.2f}x")
    print(f"📊 Cache hit rate: {stats2.get('cache_hit_rate', 0):.1f}%")
    
    # Verify results are identical
    assert result1.optimal_periods == result2.optimal_periods, "Cached results should be identical"
    print("✅ Cache consistency verified")
    
    return result1, result2

def main():
    """Run all tests."""
    print("🚀 Testing Improved DataDrivenPeriodSelector with VectorBT Optimizations")
    print("=" * 80)
    
    try:
        # Test basic functionality
        result, stats = test_basic_functionality()
        
        # Test performance comparison
        benchmark_results = test_performance_comparison()
        
        # Test memory efficiency
        memory_result = test_memory_efficiency()
        
        # Test convenience functions
        periods, convenience_stats = test_convenience_functions()
        
        # Test caching
        cache_result1, cache_result2 = test_caching()
        
        print("\n" + "="*80)
        print("🎉 All tests completed successfully!")
        print("="*80)
        
        # Summary
        print("\n📊 Summary of Improvements:")
        print("✅ VectorBT rolling operations integration")
        print("✅ Unified vectorization manager for batch processing")
        print("✅ Memory optimization for large datasets")
        print("✅ Performance monitoring and caching")
        print("✅ Parallel processing capabilities")
        print("✅ Enhanced convenience functions")
        print("✅ Comprehensive benchmarking tools")
        
        print(f"\n🏆 Best performance configuration: {min(benchmark_results.items(), key=lambda x: x[1]['avg_time'])[0]}")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)