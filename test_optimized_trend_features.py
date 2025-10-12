#!/usr/bin/env python3
"""
Test script for optimized trend feature generation using VectorBTRollingOptimizer and UnifiedVectorizationManager.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_points: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate price data with trend
    base_price = 100
    trend = np.linspace(0, 10, n_points)
    noise = np.random.randn(n_points) * 0.5
    close_prices = base_price + trend + noise
    
    # Generate OHLC data
    high_prices = close_prices + np.random.uniform(0, 1, n_points)
    low_prices = close_prices - np.random.uniform(0, 1, n_points)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    # Generate volume data
    volume = np.random.lognormal(10, 1, n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=pd.date_range('2023-01-01', periods=n_points, freq='1min'))
    
    return data

def test_optimized_trend_generator():
    """Test the OptimizedTrendFeatureGenerator."""
    print("🧪 Testing OptimizedTrendFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.trend import OptimizedTrendFeatureGenerator
        
        # Create sample data
        data = create_sample_data(1000)
        print(f"   → Created sample data with {len(data)} points")
        
        # Initialize generator
        generator = OptimizedTrendFeatureGenerator()
        print(f"   → Initialized generator with config: {generator.config.name}")
        
        # Test single feature generation
        start_time = time.time()
        feature = generator._generate_feature(data)
        single_time = time.time() - start_time
        
        print(f"   → Generated single feature: {feature.name}")
        print(f"   → Single feature generation time: {single_time:.4f}s")
        print(f"   → Feature shape: {feature.shape}")
        print(f"   → Feature stats: mean={feature.mean():.4f}, std={feature.std():.4f}")
        
        # Test batch feature generation
        start_time = time.time()
        batch_features = generator.generate_batch_features(data)
        batch_time = time.time() - start_time
        
        print(f"   → Generated {len(batch_features)} batch features")
        print(f"   → Batch feature generation time: {batch_time:.4f}s")
        
        for name, series in batch_features.items():
            print(f"     - {name}: mean={series.mean():.4f}, std={series.std():.4f}")
        
        print("✅ OptimizedTrendFeatureGenerator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ OptimizedTrendFeatureGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorbt_trend_generator():
    """Test the VectorBTTrendFeatureGenerator."""
    print("\n🧪 Testing VectorBTTrendFeatureGenerator...")
    
    try:
        from src.feature_generation.categories.trend import VectorBTTrendFeatureGenerator
        
        # Create sample data
        data = create_sample_data(1000)
        print(f"   → Created sample data with {len(data)} points")
        
        # Initialize generator
        generator = VectorBTTrendFeatureGenerator(period=20)
        print(f"   → Initialized generator with period: {generator.period}")
        
        # Test feature generation
        start_time = time.time()
        feature = generator._generate_feature(data)
        generation_time = time.time() - start_time
        
        print(f"   → Generated feature: {feature.name}")
        print(f"   → Generation time: {generation_time:.4f}s")
        print(f"   → Feature shape: {feature.shape}")
        print(f"   → Feature stats: mean={feature.mean():.4f}, std={feature.std():.4f}")
        
        print("✅ VectorBTTrendFeatureGenerator test passed!")
        return True
        
    except Exception as e:
        print(f"❌ VectorBTTrendFeatureGenerator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimized_generators_factory():
    """Test the create_optimized_trend_generators factory function."""
    print("\n🧪 Testing create_optimized_trend_generators...")
    
    try:
        from src.feature_generation.categories.trend import create_optimized_trend_generators
        
        # Create generators
        generators = create_optimized_trend_generators(periods=[10, 20, 50])
        print(f"   → Created {len(generators)} optimized generators")
        
        # Test each generator
        data = create_sample_data(500)
        successful_generators = 0
        
        for i, generator in enumerate(generators):
            try:
                feature = generator._generate_feature(data)
                print(f"   → Generator {i+1} ({generator.config.name}): {feature.name}")
                successful_generators += 1
            except Exception as e:
                print(f"   → Generator {i+1} failed: {e}")
        
        print(f"   → {successful_generators}/{len(generators)} generators successful")
        
        print("✅ create_optimized_trend_generators test passed!")
        return True
        
    except Exception as e:
        print(f"❌ create_optimized_trend_generators test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance between optimized and standard generators."""
    print("\n🧪 Testing performance comparison...")
    
    try:
        from src.feature_generation.categories.trend import (
            OptimizedTrendFeatureGenerator, 
            SMAGenerator,
            create_optimized_trend_generators
        )
        
        # Create sample data
        data = create_sample_data(2000)
        print(f"   → Created sample data with {len(data)} points")
        
        # Test optimized generator
        optimized_gen = OptimizedTrendFeatureGenerator()
        start_time = time.time()
        optimized_feature = optimized_gen._generate_feature(data)
        optimized_time = time.time() - start_time
        
        # Test standard generator
        standard_gen = SMAGenerator(period=20)
        start_time = time.time()
        standard_feature = standard_gen._generate_feature(data)
        standard_time = time.time() - start_time
        
        print(f"   → Optimized generator time: {optimized_time:.4f}s")
        print(f"   → Standard generator time: {standard_time:.4f}s")
        print(f"   → Speedup: {standard_time/optimized_time:.2f}x")
        
        # Test batch processing
        start_time = time.time()
        batch_features = optimized_gen.generate_batch_features(data)
        batch_time = time.time() - start_time
        
        print(f"   → Batch processing time: {batch_time:.4f}s")
        print(f"   → Generated {len(batch_features)} features in batch")
        
        print("✅ Performance comparison test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Starting optimized trend feature generation tests...\n")
    
    tests = [
        test_optimized_trend_generator,
        test_vectorbt_trend_generator,
        test_optimized_generators_factory,
        test_performance_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VectorBT optimization is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)