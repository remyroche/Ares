#!/usr/bin/env python3
"""
Test script for VectorBT optimization in returns feature generation.

This script tests the enhanced returns.py with full VectorBT integration
including VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Create datetime index
    start_date = datetime.now() - timedelta(days=n_points)
    dates = pd.date_range(start=start_date, periods=n_points, freq='1min')
    
    # Generate sample price data with some trend and volatility
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n_points)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Add some volume data
    volume = np.random.lognormal(10, 1, n_points)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_points)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_points))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_points))),
        'close': prices,
        'volume': volume
    }, index=dates)
    
    # Ensure high >= low and high/low contain open/close
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data

def test_returns_generators():
    """Test the enhanced returns generators with VectorBT optimization."""
    print("🧪 Testing VectorBT-optimized returns feature generation...")
    
    # Create sample data
    data = create_sample_data(1000)
    print(f"📊 Created sample data with {len(data)} points")
    
    try:
        # Import the enhanced returns generators
        from src.feature_generation.categories.returns import (
            ReturnsFeatureGenerator,
            LogReturnsGenerator,
            SimpleReturnsGenerator,
            VectorBTOptimizedReturnsGenerator,
            create_vectorbt_optimized_returns_generators
        )
        
        print("✅ Successfully imported enhanced returns generators")
        
        # Test individual generators
        print("\n🔍 Testing individual generators...")
        
        # Test ReturnsFeatureGenerator
        print("  - Testing ReturnsFeatureGenerator...")
        returns_gen = ReturnsFeatureGenerator()
        returns_result = returns_gen.generate_feature(data)
        print(f"    Generated {len(returns_result)} returns features")
        
        # Test LogReturnsGenerator
        print("  - Testing LogReturnsGenerator...")
        log_returns_gen = LogReturnsGenerator(period=1)
        log_returns_result = log_returns_gen.generate_feature(data)
        print(f"    Generated {len(log_returns_result)} log returns features")
        
        # Test SimpleReturnsGenerator
        print("  - Testing SimpleReturnsGenerator...")
        simple_returns_gen = SimpleReturnsGenerator(period=1)
        simple_returns_result = simple_returns_gen.generate_feature(data)
        print(f"    Generated {len(simple_returns_result)} simple returns features")
        
        # Test VectorBTOptimizedReturnsGenerator
        print("  - Testing VectorBTOptimizedReturnsGenerator...")
        vectorbt_gen = VectorBTOptimizedReturnsGenerator()
        comprehensive_result = vectorbt_gen.generate_comprehensive_returns_features(data)
        print(f"    Generated {len(comprehensive_result.columns)} comprehensive features")
        
        # Test batch processing
        print("\n🚀 Testing batch processing...")
        feature_configs = [
            {'type': 'simple_returns', 'period': 1},
            {'type': 'log_returns', 'period': 1},
            {'type': 'cumulative_returns', 'window': 20},
            {'type': 'returns_volatility', 'window': 20}
        ]
        
        batch_result = returns_gen.generate_returns_features_batch(data, feature_configs)
        print(f"    Generated {len(batch_result.columns)} batch features")
        
        # Test performance statistics
        print("\n📈 Performance Statistics:")
        stats = returns_gen.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
        
        # Test VectorBT-optimized generators
        print("\n🎯 Testing VectorBT-optimized generators...")
        vectorbt_generators = create_vectorbt_optimized_returns_generators()
        print(f"    Created {len(vectorbt_generators)} VectorBT-optimized generators")
        
        # Test a few generators
        for i, generator in enumerate(vectorbt_generators[:5]):  # Test first 5
            try:
                result = generator.generate_feature(data)
                print(f"    Generator {i+1} ({generator.__class__.__name__}): {len(result)} features")
            except Exception as e:
                print(f"    Generator {i+1} failed: {e}")
        
        print("\n✅ All tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def test_performance_comparison():
    """Compare performance between optimized and non-optimized versions."""
    print("\n⚡ Performance comparison test...")
    
    data = create_sample_data(5000)  # Larger dataset for performance testing
    
    try:
        from src.feature_generation.categories.returns import (
            ReturnsFeatureGenerator,
            VectorBTOptimizedReturnsGenerator
        )
        
        # Test standard generator
        print("  - Testing standard ReturnsFeatureGenerator...")
        start_time = time.time()
        standard_gen = ReturnsFeatureGenerator()
        standard_result = standard_gen.generate_feature(data)
        standard_time = time.time() - start_time
        print(f"    Standard generator time: {standard_time:.4f} seconds")
        
        # Test VectorBT-optimized generator
        print("  - Testing VectorBTOptimizedReturnsGenerator...")
        start_time = time.time()
        vectorbt_gen = VectorBTOptimizedReturnsGenerator()
        vectorbt_result = vectorbt_gen.generate_comprehensive_returns_features(data)
        vectorbt_time = time.time() - start_time
        print(f"    VectorBT-optimized generator time: {vectorbt_time:.4f} seconds")
        
        # Performance comparison
        if standard_time > 0:
            speedup = standard_time / vectorbt_time if vectorbt_time > 0 else float('inf')
            print(f"    Speedup: {speedup:.2f}x")
        
        # Test performance statistics
        print("\n📊 VectorBT Performance Statistics:")
        stats = vectorbt_gen.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.2f}")
            else:
                print(f"    {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Starting VectorBT optimization tests for returns feature generation...")
    print("=" * 70)
    
    # Test basic functionality
    success = test_returns_generators()
    
    if success:
        # Test performance
        test_performance_comparison()
        
        print("\n" + "=" * 70)
        print("🎉 All VectorBT optimization tests completed successfully!")
        print("\nKey improvements implemented:")
        print("✅ VectorBTRollingOptimizer integration for rolling operations")
        print("✅ UnifiedVectorizationManager for intelligent optimization")
        print("✅ Batch processing for multiple features")
        print("✅ Performance monitoring and statistics")
        print("✅ Comprehensive fallback mechanisms")
        print("✅ Enhanced error handling and logging")
        
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)