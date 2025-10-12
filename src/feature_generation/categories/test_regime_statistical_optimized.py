"""
Test script for the optimized regime statistical feature generator.

This script validates the optimized implementation and demonstrates
the performance improvements over the original implementation.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_data(size: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create test data for validation."""
    np.random.seed(seed)
    
    # Generate realistic price data
    returns = np.random.normal(0, 0.02, size)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate volume data
    volume = np.random.lognormal(10, 1, size)
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, size))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, size))),
        'open': np.roll(prices, 1),
        'volume': volume
    })
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data

def test_optimized_generator():
    """Test the optimized regime statistical generator."""
    print("Testing Optimized Regime Statistical Feature Generator...")
    
    try:
        from regime_statistical_optimized import OptimizedRegimeStatisticalFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        print(f"Created test data with {len(data)} rows")
        
        # Create generator
        generator = OptimizedRegimeStatisticalFeatureGenerator()
        print("Created optimized generator")
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features_optimized(data)
        end_time = time.time()
        
        print(f"Generated {len(features)} features in {end_time - start_time:.4f} seconds")
        
        # Print feature names
        print("Generated features:")
        for i, (name, values) in enumerate(features.items()):
            print(f"  {i+1:2d}. {name}: {len(values)} values, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Get performance stats
        stats = generator.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  VectorBT usage: {stats['vectorbt_usage_percentage']:.1f}%")
        print(f"  Batch operations: {stats['batch_usage_percentage']:.1f}%")
        print(f"  Chunked operations: {stats['chunked_usage_percentage']:.1f}%")
        print(f"  GPU operations: {stats['gpu_usage_percentage']:.1f}%")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1f}%")
        print(f"  Average operation time: {stats['average_operation_time']:.4f}s")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error testing optimized generator: {e}")
        return False

def test_original_generator():
    """Test the original regime statistical generator for comparison."""
    print("\nTesting Original Regime Statistical Feature Generator...")
    
    try:
        from regime_statistical import RegimeStatisticalFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        print(f"Created test data with {len(data)} rows")
        
        # Create generator
        generator = RegimeStatisticalFeatureGenerator()
        print("Created original generator")
        
        # Generate features
        start_time = time.time()
        features = generator.generate_features(data)
        end_time = time.time()
        
        print(f"Generated {len(features)} features in {end_time - start_time:.4f} seconds")
        
        # Print feature names
        print("Generated features:")
        for i, (name, values) in enumerate(features.items()):
            print(f"  {i+1:2d}. {name}: {len(values)} values, mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Get performance stats if available
        if hasattr(generator, 'get_performance_stats'):
            stats = generator.get_performance_stats()
            print(f"\nPerformance Statistics:")
            print(f"  Total operations: {stats.get('total_operations', 'N/A')}")
            print(f"  VectorBT usage: {stats.get('vectorbt_usage_percentage', 0):.1f}%")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error testing original generator: {e}")
        return False

def compare_implementations():
    """Compare original and optimized implementations."""
    print("\n" + "="*60)
    print("COMPARING ORIGINAL VS OPTIMIZED IMPLEMENTATIONS")
    print("="*60)
    
    # Test data sizes
    data_sizes = [100, 500, 1000, 2000]
    
    results = {
        'original': {},
        'optimized': {},
        'speedup': {}
    }
    
    for size in data_sizes:
        print(f"\nTesting with data size: {size}")
        print("-" * 30)
        
        data = create_test_data(size)
        
        # Test original
        try:
            from regime_statistical import RegimeStatisticalFeatureGenerator
            generator_orig = RegimeStatisticalFeatureGenerator()
            
            start_time = time.time()
            features_orig = generator_orig.generate_features(data)
            orig_time = time.time() - start_time
            
            results['original'][size] = {
                'time': orig_time,
                'features': len(features_orig),
                'success': True
            }
            print(f"Original:  {orig_time:.4f}s, {len(features_orig)} features")
            
        except Exception as e:
            results['original'][size] = {'success': False, 'error': str(e)}
            print(f"Original:  FAILED - {e}")
            orig_time = None
        
        # Test optimized
        try:
            from regime_statistical_optimized import OptimizedRegimeStatisticalFeatureGenerator
            generator_opt = OptimizedRegimeStatisticalFeatureGenerator()
            
            start_time = time.time()
            features_opt = generator_opt.generate_features_optimized(data)
            opt_time = time.time() - start_time
            
            results['optimized'][size] = {
                'time': opt_time,
                'features': len(features_opt),
                'success': True
            }
            print(f"Optimized: {opt_time:.4f}s, {len(features_opt)} features")
            
            # Calculate speedup
            if orig_time and opt_time:
                speedup = orig_time / opt_time
                results['speedup'][size] = speedup
                print(f"Speedup:   {speedup:.2f}x")
            
        except Exception as e:
            results['optimized'][size] = {'success': False, 'error': str(e)}
            print(f"Optimized: FAILED - {e}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['speedup']:
        print("Speedup Ratios:")
        for size, speedup in results['speedup'].items():
            print(f"  Data size {size:4d}: {speedup:.2f}x speedup")
        
        avg_speedup = np.mean(list(results['speedup'].values()))
        print(f"\nAverage speedup: {avg_speedup:.2f}x")
    
    return results

def main():
    """Run all tests."""
    print("Regime Statistical VectorBT Optimization Test Suite")
    print("=" * 60)
    
    # Test optimized generator
    optimized_success = test_optimized_generator()
    
    # Test original generator
    original_success = test_original_generator()
    
    # Compare implementations if both are available
    if optimized_success and original_success:
        compare_implementations()
    else:
        print("\nSkipping comparison due to import errors")
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()