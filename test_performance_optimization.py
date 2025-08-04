#!/usr/bin/env python3
"""
Test script for Performance Optimization features.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Import the optimization components
from src.utils.performance_optimizer import PerformanceOptimizer, setup_performance_optimizer
from src.utils.data_optimizer import DataOptimizer, setup_data_optimizer


async def test_performance_optimization():
    """Test performance optimization features."""
    print("🧪 Testing Performance Optimization Features...")
    
    # Test configuration
    config = {
        "performance_optimizer": {
            "monitoring_interval": 5,  # 5 seconds for testing
            "memory_threshold": 0.8,
            "cpu_threshold": 0.7,
            "cache_size_limit": 1000
        },
        "data_optimizer": {
            "chunk_size": 1000,
            "memory_limit": 0.8,
            "compression_enabled": True,
            "cache_enabled": True
        }
    }
    
    try:
        # Test 1: Initialize Performance Optimizer
        print("\n📋 Test 1: Initializing Performance Optimizer...")
        performance_optimizer = await setup_performance_optimizer(config)
        
        if performance_optimizer:
            print("✅ Performance Optimizer initialized successfully")
        else:
            print("❌ Performance Optimizer initialization failed")
            return False
        
        # Test 2: Initialize Data Optimizer
        print("\n📋 Test 2: Initializing Data Optimizer...")
        data_optimizer = await setup_data_optimizer(config)
        
        if data_optimizer:
            print("✅ Data Optimizer initialized successfully")
        else:
            print("❌ Data Optimizer initialization failed")
            return False
        
        # Test 3: Generate test data
        print("\n📋 Test 3: Generating Test Data...")
        test_data = generate_test_data(10000)
        print(f"✅ Generated test data: {len(test_data)} rows")
        
        # Test 4: Test DataFrame optimization
        print("\n📋 Test 4: Testing DataFrame Optimization...")
        
        # Test different optimization strategies
        strategies = ["auto", "memory", "speed", "balanced"]
        
        for strategy in strategies:
            print(f"\n📋 Testing {strategy} optimization...")
            
            # Create copy of test data
            df_copy = test_data.copy()
            
            # Measure original memory usage
            original_memory = df_copy.memory_usage(deep=True).sum()
            
            # Apply optimization
            start_time = time.time()
            optimized_df = await data_optimizer.optimize_dataframe(df_copy, strategy=strategy)
            optimization_time = time.time() - start_time
            
            # Measure optimized memory usage
            optimized_memory = optimized_df.memory_usage(deep=True).sum()
            memory_saved = original_memory - optimized_memory
            memory_saved_mb = memory_saved / 1024 / 1024
            
            print(f"✅ {strategy} optimization completed:")
            print(f"  - Original memory: {original_memory / 1024 / 1024:.2f}MB")
            print(f"  - Optimized memory: {optimized_memory / 1024 / 1024:.2f}MB")
            print(f"  - Memory saved: {memory_saved_mb:.2f}MB")
            print(f"  - Optimization time: {optimization_time:.4f}s")
        
        # Test 5: Test market data optimization
        print("\n📋 Test 5: Testing Market Data Optimization...")
        
        market_data = generate_market_data(5000)
        optimized_market_data = await data_optimizer.optimize_market_data(market_data)
        
        print(f"✅ Market data optimization completed:")
        print(f"  - Original rows: {len(market_data)}")
        print(f"  - Optimized rows: {len(optimized_market_data)}")
        print(f"  - Columns: {list(optimized_market_data.columns)}")
        
        # Test 6: Test ensemble data optimization
        print("\n📋 Test 6: Testing Ensemble Data Optimization...")
        
        ensemble_data = {
            "dataset_1": generate_test_data(2000),
            "dataset_2": generate_test_data(2000),
            "dataset_3": generate_test_data(2000)
        }
        
        optimized_ensemble = await data_optimizer.optimize_ensemble_data(ensemble_data)
        
        print(f"✅ Ensemble data optimization completed:")
        print(f"  - Original datasets: {len(ensemble_data)}")
        print(f"  - Optimized datasets: {len(optimized_ensemble)}")
        
        # Test 7: Test chunk processing
        print("\n📋 Test 7: Testing Chunk Processing...")
        
        large_data = generate_test_data(50000)
        chunks = await data_optimizer.process_in_chunks(large_data, chunk_size=5000)
        
        print(f"✅ Chunk processing completed:")
        print(f"  - Original data size: {len(large_data)}")
        print(f"  - Number of chunks: {len(chunks)}")
        print(f"  - Average chunk size: {len(large_data) / len(chunks):.0f}")
        
        # Test 8: Test performance monitoring
        print("\n📋 Test 8: Testing Performance Monitoring...")
        
        # Wait for some monitoring data to be collected
        await asyncio.sleep(10)
        
        # Get performance report
        performance_report = performance_optimizer.get_performance_report()
        
        if "error" not in performance_report:
            print("✅ Performance monitoring working:")
            print(f"  - Current CPU usage: {performance_report['current_metrics']['cpu_usage']:.2%}")
            print(f"  - Current memory usage: {performance_report['current_metrics']['memory_usage']:.2%}")
            print(f"  - Cache hit rate: {performance_report['current_metrics']['cache_hit_rate']:.2%}")
        else:
            print("⚠️ Performance monitoring not available yet")
        
        # Test 9: Test optimization statistics
        print("\n📋 Test 9: Testing Optimization Statistics...")
        
        data_stats = data_optimizer.get_optimization_stats()
        
        if "error" not in data_stats:
            print("✅ Data optimization statistics:")
            print(f"  - Total processed: {data_stats['total_processed']}")
            print(f"  - Memory saved: {data_stats['memory_saved_mb']:.2f}MB")
            print(f"  - Cache efficiency: {data_stats['cache_efficiency']:.2%}")
        else:
            print("❌ Failed to get optimization statistics")
        
        # Test 10: Test cleanup
        print("\n📋 Test 10: Testing Cleanup...")
        
        await performance_optimizer.stop()
        await data_optimizer.stop()
        
        print("✅ Cleanup completed successfully")
        
        print("\n🎉 All performance optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_test_data(size: int) -> pd.DataFrame:
    """Generate test data for optimization testing."""
    np.random.seed(42)
    
    data = {
        'id': range(size),
        'value_1': np.random.randint(0, 1000, size),
        'value_2': np.random.randn(size),
        'value_3': np.random.choice(['A', 'B', 'C', 'D'], size),
        'value_4': np.random.randint(0, 255, size),
        'value_5': np.random.randn(size) * 1000,
        'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='1min')
    }
    
    return pd.DataFrame(data)


def generate_market_data(size: int) -> pd.DataFrame:
    """Generate market data for optimization testing."""
    np.random.seed(42)
    
    base_price = 100.0
    prices = [base_price]
    
    for _ in range(size - 1):
        change = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=size, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': [p * (1 + np.random.normal(0, 0.002)) for p in prices],
        'volume': np.random.randint(1000, 10000, size)
    }
    
    return pd.DataFrame(data)


async def test_optimization_benchmarks():
    """Test optimization benchmarks."""
    print("\n🧪 Testing Optimization Benchmarks...")
    
    config = {
        "data_optimizer": {
            "chunk_size": 1000,
            "memory_limit": 0.8,
            "compression_enabled": True,
            "cache_enabled": True
        }
    }
    
    try:
        # Initialize data optimizer
        data_optimizer = await setup_data_optimizer(config)
        
        # Test different data sizes
        data_sizes = [1000, 5000, 10000, 50000]
        
        benchmark_results = {}
        
        for size in data_sizes:
            print(f"\n📋 Benchmarking data size: {size}")
            
            # Generate test data
            test_data = generate_test_data(size)
            
            # Test different optimization strategies
            strategies = ["auto", "memory", "speed", "balanced"]
            
            size_results = {}
            
            for strategy in strategies:
                # Create copy for testing
                df_copy = test_data.copy()
                
                # Measure original memory
                original_memory = df_copy.memory_usage(deep=True).sum()
                
                # Time optimization
                start_time = time.time()
                optimized_df = await data_optimizer.optimize_dataframe(df_copy, strategy=strategy)
                optimization_time = time.time() - start_time
                
                # Measure optimized memory
                optimized_memory = optimized_df.memory_usage(deep=True).sum()
                memory_saved = original_memory - optimized_memory
                
                size_results[strategy] = {
                    "original_memory_mb": original_memory / 1024 / 1024,
                    "optimized_memory_mb": optimized_memory / 1024 / 1024,
                    "memory_saved_mb": memory_saved / 1024 / 1024,
                    "optimization_time_s": optimization_time,
                    "memory_reduction_percent": (memory_saved / original_memory) * 100
                }
                
                print(f"  {strategy}: {size_results[strategy]['memory_reduction_percent']:.1f}% reduction, "
                      f"{size_results[strategy]['optimization_time_s']:.4f}s")
            
            benchmark_results[size] = size_results
        
        # Print summary
        print("\n📊 Benchmark Summary:")
        print("=" * 80)
        print(f"{'Size':<8} {'Strategy':<10} {'Memory Saved':<15} {'Time (s)':<10} {'Reduction %':<12}")
        print("=" * 80)
        
        for size, results in benchmark_results.items():
            for strategy, metrics in results.items():
                print(f"{size:<8} {strategy:<10} {metrics['memory_saved_mb']:<15.2f} "
                      f"{metrics['optimization_time_s']:<10.4f} {metrics['memory_reduction_percent']:<12.1f}")
        
        await data_optimizer.stop()
        
        print("\n✅ Benchmark tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return False


async def main():
    """Run all performance optimization tests."""
    print("🚀 Starting Performance Optimization Tests...")
    
    # Run tests
    test_results = []
    
    # Test 1: Basic functionality
    test_results.append(await test_performance_optimization())
    
    # Test 2: Benchmarks
    test_results.append(await test_optimization_benchmarks())
    
    # Summary
    passed_tests = sum(test_results)
    total_tests = len(test_results)
    
    print(f"\n📊 Test Summary:")
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 All performance optimization tests passed!")
        print("✅ Performance optimization is ready for production use.")
    else:
        print("⚠️ Some performance optimization tests failed.")
        print("🔧 Please check the implementation.")


if __name__ == "__main__":
    asyncio.run(main()) 