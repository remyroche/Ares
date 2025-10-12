#!/usr/bin/env python3
"""
Test script for VectorBT integration in feature generation.

This script tests the VectorBT-optimized feature generators to ensure they work correctly
and provide performance improvements over the legacy implementations.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_points: int = 10000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with some trend and volatility
    base_price = 100
    returns = np.random.normal(0, 0.02, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_points))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_points)
    })
    
    # Ensure high >= low and high/low contain open/close
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    return data

def test_vectorbt_availability():
    """Test if VectorBT is available and working."""
    print("Testing VectorBT availability...")
    
    try:
        import vectorbt as vbt
        print(f"✓ VectorBT version: {vbt.__version__}")
        
        # Test basic VectorBT functionality
        test_data = pd.Series([1, 2, 3, 4, 5])
        result = vbt.generic.rolling_mean(test_data, window=3)
        print(f"✓ VectorBT rolling operations working: {len(result)} results")
        
        return True
    except ImportError as e:
        print(f"✗ VectorBT not available: {e}")
        return False
    except Exception as e:
        print(f"✗ VectorBT error: {e}")
        return False

def test_feature_generators():
    """Test the VectorBT-optimized feature generators."""
    print("\nTesting VectorBT feature generators...")
    
    # Create sample data
    data = create_sample_data(1000)
    print(f"Created sample data with {len(data)} points")
    
    # Test each category
    categories = [
        ('Advanced Statistical', 'src.feature_generation.categories.advanced_statistical'),
        ('Support/Resistance', 'src.feature_generation.categories.support_resistance'),
        ('Legacy', 'src.feature_generation.categories.legacy'),
        ('Acceleration', 'src.feature_generation.categories.acceleration'),
        ('Order Flow', 'src.feature_generation.categories.order_flow')
    ]
    
    results = {}
    
    for category_name, module_path in categories:
        print(f"\n--- Testing {category_name} Features ---")
        
        try:
            # Import the module
            module = __import__(module_path, fromlist=['create_default_vectorbt_advanced_statistical_generators', 
                                                      'create_default_vectorbt_support_resistance_generators',
                                                      'create_default_vectorbt_legacy_generators',
                                                      'create_default_vectorbt_acceleration_generators',
                                                      'create_default_vectorbt_order_flow_generators'])
            
            # Get the appropriate function name
            if 'advanced_statistical' in module_path:
                create_func = getattr(module, 'create_default_vectorbt_advanced_statistical_generators', None)
            elif 'support_resistance' in module_path:
                create_func = getattr(module, 'create_default_vectorbt_support_resistance_generators', None)
            elif 'legacy' in module_path:
                create_func = getattr(module, 'create_default_vectorbt_legacy_generators', None)
            elif 'acceleration' in module_path:
                create_func = getattr(module, 'create_default_vectorbt_acceleration_generators', None)
            elif 'order_flow' in module_path:
                create_func = getattr(module, 'create_default_vectorbt_order_flow_generators', None)
            else:
                create_func = None
            
            if create_func:
                # Create generators
                generators = create_func()
                print(f"✓ Created {len(generators)} {category_name} generators")
                
                # Test a few generators
                test_generators = generators[:3]  # Test first 3 generators
                for i, generator in enumerate(test_generators):
                    try:
                        start_time = time.time()
                        result = generator.generate(data)
                        end_time = time.time()
                        
                        print(f"  ✓ Generator {i+1} ({generator.config.name}): {len(result)} features in {end_time - start_time:.4f}s")
                        
                        # Check if result is valid
                        if hasattr(result, 'values') and len(result.values) > 0:
                            valid_values = np.isfinite(result.values).sum()
                            total_values = len(result.values)
                            print(f"    Valid values: {valid_values}/{total_values} ({valid_values/total_values*100:.1f}%)")
                        
                    except Exception as e:
                        print(f"  ✗ Generator {i+1} failed: {e}")
                
                results[category_name] = {
                    'total_generators': len(generators),
                    'tested_generators': len(test_generators),
                    'status': 'success'
                }
            else:
                print(f"✗ No VectorBT generators found for {category_name}")
                results[category_name] = {'status': 'no_generators'}
                
        except Exception as e:
            print(f"✗ Error testing {category_name}: {e}")
            results[category_name] = {'status': 'error', 'error': str(e)}
    
    return results

def test_performance_comparison():
    """Compare performance between VectorBT and legacy implementations."""
    print("\n--- Performance Comparison ---")
    
    # Create larger dataset for performance testing
    data = create_sample_data(5000)
    print(f"Testing with {len(data)} data points")
    
    # Test a simple feature (SMA) with both implementations
    try:
        from src.feature_generation.categories.legacy import LegacySMAGenerator
        from src.feature_generation.categories.vectorbt_legacy import VectorBTLegacySMAGenerator
        
        # Test legacy implementation
        legacy_generator = LegacySMAGenerator(period=20)
        start_time = time.time()
        legacy_result = legacy_generator.generate(data)
        legacy_time = time.time() - start_time
        
        # Test VectorBT implementation
        vectorbt_generator = VectorBTLegacySMAGenerator(period=20)
        start_time = time.time()
        vectorbt_result = vectorbt_generator.generate(data)
        vectorbt_time = time.time() - start_time
        
        print(f"Legacy SMA: {legacy_time:.4f}s")
        print(f"VectorBT SMA: {vectorbt_time:.4f}s")
        print(f"Speedup: {legacy_time/vectorbt_time:.2f}x")
        
        # Check if results are similar
        if len(legacy_result) == len(vectorbt_result):
            correlation = np.corrcoef(legacy_result.dropna(), vectorbt_result.dropna())[0, 1]
            print(f"Correlation between results: {correlation:.4f}")
        
    except Exception as e:
        print(f"Performance comparison failed: {e}")

def test_vectorbt_rolling_optimizer():
    """Test the VectorBT rolling optimizer."""
    print("\n--- Testing VectorBT Rolling Optimizer ---")
    
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
        
        # Create optimizer
        optimizer = VectorBTRollingOptimizer(enable_gpu=False, enable_parallel=True)
        
        # Test with sample data
        data = create_sample_data(1000)
        close = data['close']
        
        # Test various operations
        operations = ['mean', 'std', 'min', 'max', 'sum']
        windows = [10, 20, 50]
        
        for operation in operations:
            for window in windows:
                try:
                    result = getattr(optimizer, f'rolling_{operation}')(close, window)
                    print(f"  ✓ {operation} (window={window}): {len(result)} results")
                except Exception as e:
                    print(f"  ✗ {operation} (window={window}): {e}")
        
        # Get performance stats
        stats = optimizer.get_performance_stats()
        print(f"Performance stats: {stats}")
        
    except Exception as e:
        print(f"VectorBT Rolling Optimizer test failed: {e}")

def main():
    """Main test function."""
    print("VectorBT Integration Test")
    print("=" * 50)
    
    # Test VectorBT availability
    vectorbt_available = test_vectorbt_availability()
    
    if not vectorbt_available:
        print("\nVectorBT is not available. Please install it with: pip install vectorbt")
        return
    
    # Test feature generators
    results = test_feature_generators()
    
    # Test performance comparison
    test_performance_comparison()
    
    # Test VectorBT rolling optimizer
    test_vectorbt_rolling_optimizer()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    for category, result in results.items():
        if result['status'] == 'success':
            print(f"✓ {category}: {result['total_generators']} generators, {result['tested_generators']} tested")
        elif result['status'] == 'no_generators':
            print(f"⚠ {category}: No VectorBT generators found")
        else:
            print(f"✗ {category}: {result.get('error', 'Unknown error')}")
    
    print("\nVectorBT integration test completed!")

if __name__ == "__main__":
    main()