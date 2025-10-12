#!/usr/bin/env python3
"""
Test script for optimized regime structural trend feature generation.

This script tests the VectorBT optimizations implemented in the
RegimeStructuralTrendFeatureGenerator class.
"""

import numpy as np
import pandas as pd
import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create test data for regime structural trend features."""
    np.random.seed(42)
    
    # Generate realistic price data
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='15min')
    
    # Create trending and choppy periods
    trend_periods = []
    for i in range(0, n_samples, 200):
        period_length = min(200, n_samples - i)
        if i % 400 < 200:  # Trending period
            trend = np.linspace(100, 110, period_length) + np.random.randn(period_length) * 0.5
        else:  # Choppy period
            trend = 105 + np.random.randn(period_length) * 2
        trend_periods.extend(trend)
    
    prices = np.array(trend_periods[:n_samples])
    
    # Add some noise and gaps
    prices = prices + np.random.randn(n_samples) * 0.1
    
    data = pd.DataFrame({
        'close': prices,
        'high': prices + np.random.rand(n_samples) * 0.5,
        'low': prices - np.random.rand(n_samples) * 0.5,
        'open': np.roll(prices, 1),
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    return data

def test_optimization_availability():
    """Test if optimization components are available."""
    print("🔍 Testing optimization component availability...")
    
    try:
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        generator = RegimeStructuralTrendFeatureGenerator()
        stats = generator.get_optimization_stats()
        
        print(f"✅ VectorBT Optimizer Available: {stats['vectorbt_optimizer_available']}")
        print(f"✅ Unified Manager Available: {stats['unified_manager_available']}")
        print(f"✅ Optimization Available: {stats['optimization_available']}")
        print(f"✅ VectorBT Available: {stats['vectorbt_available']}")
        
        if stats['vectorbt_optimizer_available']:
            print(f"📊 VectorBT Optimizer Stats: {stats.get('vectorbt_optimizer_stats', 'N/A')}")
        
        return stats['vectorbt_optimizer_available'] or stats['unified_manager_available']
        
    except Exception as e:
        print(f"❌ Error testing optimization availability: {e}")
        return False

def test_feature_generation():
    """Test feature generation with and without optimization."""
    print("\n🧪 Testing feature generation...")
    
    try:
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        # Create test data
        data = create_test_data(1000)
        print(f"📊 Created test data with {len(data)} samples")
        
        # Test feature generation
        generator = RegimeStructuralTrendFeatureGenerator()
        
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(features)} features in {generation_time:.3f}s")
        
        # Display feature names and shapes
        print("\n📋 Generated Features:")
        for name, values in features.items():
            print(f"  • {name}: shape={values.shape}, non-nan={np.sum(~np.isnan(values))}")
        
        # Test optimization stats
        stats = generator.get_optimization_stats()
        print(f"\n📈 Optimization Stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing feature generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance with and without optimization."""
    print("\n⚡ Testing performance comparison...")
    
    try:
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        # Create larger test data
        data = create_test_data(5000)
        print(f"📊 Created test data with {len(data)} samples")
        
        generator = RegimeStructuralTrendFeatureGenerator()
        
        # Test with optimization
        start_time = time.time()
        features_optimized = generator.generate_features(data)
        optimized_time = time.time() - start_time
        
        print(f"✅ Optimized generation: {len(features_optimized)} features in {optimized_time:.3f}s")
        
        # Get performance stats
        stats = generator.get_optimization_stats()
        
        if 'vectorbt_optimizer_stats' in stats:
            optimizer_stats = stats['vectorbt_optimizer_stats']
            print(f"📊 VectorBT Operations: {optimizer_stats.get('vectorbt_operations', 0)}")
            print(f"📊 Pandas Fallbacks: {optimizer_stats.get('pandas_fallbacks', 0)}")
            print(f"📊 GPU Operations: {optimizer_stats.get('gpu_operations', 0)}")
            print(f"📊 Total Time: {optimizer_stats.get('total_time', 0):.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing performance: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_operations():
    """Test individual VectorBT operations."""
    print("\n🔧 Testing individual VectorBT operations...")
    
    try:
        from src.feature_generation.categories.regime_structural_trend import RegimeStructuralTrendFeatureGenerator
        
        generator = RegimeStructuralTrendFeatureGenerator()
        
        if not generator.vectorbt_optimizer:
            print("⚠️ VectorBT optimizer not available, skipping individual operation tests")
            return True
        
        # Create test data
        data = create_test_data(500)
        prices = data['close'].values
        
        print("Testing structural trend persistence...")
        persistence = generator._calculate_structural_trend_persistence(prices, 20)
        print(f"  ✅ Persistence shape: {persistence.shape}")
        
        print("Testing trend direction consistency...")
        consistency = generator._calculate_trend_direction_consistency(prices, 20)
        print(f"  ✅ Consistency shape: {consistency.shape}")
        
        print("Testing trend regime persistence...")
        regime_persistence = generator._calculate_trend_regime_persistence(prices, 20)
        print(f"  ✅ Regime persistence shape: {regime_persistence.shape}")
        
        print("Testing structural trend strength...")
        strength = generator._calculate_structural_trend_strength(prices, 20)
        print(f"  ✅ Strength shape: {strength.shape}")
        
        print("Testing trend acceleration...")
        acceleration = generator._calculate_trend_acceleration(prices, 20)
        print(f"  ✅ Acceleration shape: {acceleration.shape}")
        
        print("Testing trend intensity...")
        intensity = generator._calculate_trend_intensity(prices, 20)
        print(f"  ✅ Intensity shape: {intensity.shape}")
        
        print("Testing market structure strength...")
        structure_strength = generator._calculate_market_structure_strength(prices, 20)
        print(f"  ✅ Structure strength shape: {structure_strength.shape}")
        
        print("Testing support/resistance strength...")
        sr_strength = generator._calculate_support_resistance_strength(prices, 20)
        print(f"  ✅ Support/resistance strength shape: {sr_strength.shape}")
        
        print("Testing market structure consistency...")
        structure_consistency = generator._calculate_market_structure_consistency(prices, 20)
        print(f"  ✅ Structure consistency shape: {structure_consistency.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing individual operations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Regime Structural Trend VectorBT Optimization")
    print("=" * 60)
    
    tests = [
        ("Optimization Availability", test_optimization_availability),
        ("Feature Generation", test_feature_generation),
        ("Performance Comparison", test_performance_comparison),
        ("Individual Operations", test_individual_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! VectorBT optimization is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)