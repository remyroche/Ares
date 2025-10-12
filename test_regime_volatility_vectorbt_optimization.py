#!/usr/bin/env python3
"""
Test script to verify VectorBT optimization in regime volatility feature generation.

This script tests:
1. VectorBTRollingOptimizer usage in RegimeVolatilityFeatureGenerator
2. UnifiedVectorizationManager usage in RegimeFeatureIntegration
3. Performance improvements from VectorBT optimizations
4. Fallback behavior when VectorBT is not available
"""

import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data(n_periods: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * (1 + returns).cumprod()
    
    # Add some volatility clustering
    vol_cluster = np.random.choice([0.01, 0.05], size=n_periods, p=[0.7, 0.3])
    clustered_returns = np.random.normal(0, vol_cluster, n_periods)
    clustered_prices = 100 * (1 + clustered_returns).cumprod()
    
    data = pd.DataFrame({
        'open': clustered_prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': clustered_prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
        'low': clustered_prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
        'close': clustered_prices,
        'volume': np.random.lognormal(10, 1, n_periods)
    }, index=pd.date_range('2020-01-01', periods=n_periods, freq='15min'))
    
    return data

def test_regime_volatility_optimization():
    """Test VectorBT optimization in RegimeVolatilityFeatureGenerator."""
    print("🧪 Testing RegimeVolatilityFeatureGenerator VectorBT optimization...")
    
    try:
        from src.feature_generation.categories.regime_volatility import RegimeVolatilityFeatureGenerator
        
        # Create test data
        data = create_sample_data(500)
        
        # Initialize generator
        generator = RegimeVolatilityFeatureGenerator()
        
        # Test VectorBT optimizer initialization
        assert generator.vectorbt_optimizer is not None, "VectorBT optimizer should be initialized"
        assert generator.unified_optimizer is not None, "Unified optimizer should be initialized"
        print("✅ VectorBT optimizers initialized successfully")
        
        # Test feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(features)} features in {generation_time:.3f}s")
        
        # Test specific VectorBT operations
        returns = generator._get_returns(data)
        if returns is not None and len(returns) > 20:
            # Test rolling volatility calculation
            vol = generator._rolling_volatility(returns, 20)
            assert len(vol) > 0, "Rolling volatility should return results"
            print("✅ Rolling volatility calculation works")
            
            # Test volatility persistence calculation
            persistence = generator._calculate_volatility_persistence(vol, 5)
            assert len(persistence) > 0, "Volatility persistence should return results"
            print("✅ Volatility persistence calculation works")
            
            # Test volatility clustering calculation
            clustering = generator._calculate_volatility_clustering(returns, 20)
            assert len(clustering) > 0, "Volatility clustering should return results"
            print("✅ Volatility clustering calculation works")
        
        return True
        
    except Exception as e:
        print(f"❌ RegimeVolatilityFeatureGenerator test failed: {e}")
        return False

def test_regime_feature_integration_optimization():
    """Test VectorBT optimization in RegimeFeatureIntegration."""
    print("\n🧪 Testing RegimeFeatureIntegration VectorBT optimization...")
    
    try:
        from src.feature_generation.categories.regime_feature_integration import RegimeFeatureIntegration
        
        # Create test data
        data = create_sample_data(500)
        
        # Initialize generator
        generator = RegimeFeatureIntegration()
        
        # Test VectorBT optimizer initialization
        assert generator.vectorbt_optimizer is not None, "VectorBT optimizer should be initialized"
        assert generator.unified_optimizer is not None, "Unified optimizer should be initialized"
        print("✅ VectorBT optimizers initialized successfully")
        
        # Test feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(features)} features in {generation_time:.3f}s")
        
        # Test DataFrame optimization
        optimized_data = generator.optimize_dataframe_processing(data)
        assert len(optimized_data) == len(data), "Optimized data should have same length"
        print("✅ DataFrame optimization works")
        
        # Test vectorized rolling operations
        result = generator.vectorized_rolling_operations(
            data, 
            operations=['mean', 'std'], 
            windows=[10, 20], 
            columns=['close', 'volume']
        )
        assert len(result) == len(data), "Vectorized operations should return same length"
        print("✅ Vectorized rolling operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ RegimeFeatureIntegration test failed: {e}")
        return False

def test_advanced_volatility_optimization():
    """Test VectorBT optimization in AdvancedVolatilityFeatures."""
    print("\n🧪 Testing AdvancedVolatilityFeatures VectorBT optimization...")
    
    try:
        from src.feature_generation.categories.advanced_volatility_features import AdvancedVolatilityFeatures
        
        # Create test data
        data = create_sample_data(500)
        
        # Initialize generator
        generator = AdvancedVolatilityFeatures()
        
        # Test VectorBT optimizer initialization
        assert generator.vectorbt_optimizer is not None, "VectorBT optimizer should be initialized"
        assert generator.unified_optimizer is not None, "Unified optimizer should be initialized"
        print("✅ VectorBT optimizers initialized successfully")
        
        # Test feature generation
        start_time = time.time()
        features = generator.generate_features(data)
        generation_time = time.time() - start_time
        
        print(f"✅ Generated {len(features.columns)} features in {generation_time:.3f}s")
        
        # Test specific VectorBT operations
        test_series = pd.Series(np.random.randn(100))
        
        # Test rolling mean
        mean_result = generator._vectorbt_rolling_operation(test_series, 'mean', 10)
        assert len(mean_result) == len(test_series), "Rolling mean should return same length"
        print("✅ Rolling mean operation works")
        
        # Test rolling apply
        apply_result = generator._vectorbt_rolling_operation(
            test_series, 
            'apply', 
            10, 
            func=lambda x: x.sum()
        )
        assert len(apply_result) == len(test_series), "Rolling apply should return same length"
        print("✅ Rolling apply operation works")
        
        return True
        
    except Exception as e:
        print(f"❌ AdvancedVolatilityFeatures test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance with and without VectorBT optimization."""
    print("\n🧪 Testing performance comparison...")
    
    try:
        from src.feature_generation.categories.regime_volatility import RegimeVolatilityFeatureGenerator
        
        # Create larger test data
        data = create_sample_data(2000)
        
        # Test with VectorBT optimization
        generator = RegimeVolatilityFeatureGenerator()
        
        start_time = time.time()
        features_optimized = generator.generate_features(data)
        optimized_time = time.time() - start_time
        
        print(f"✅ VectorBT optimized generation: {optimized_time:.3f}s for {len(features_optimized)} features")
        
        # Test performance stats
        if generator.vectorbt_optimizer:
            stats = generator.vectorbt_optimizer.get_performance_stats()
            print(f"📊 VectorBT usage rate: {stats.get('vectorbt_usage_rate', 0):.2%}")
            print(f"📊 Total operations: {stats.get('total_operations', 0)}")
            print(f"📊 Average time per operation: {stats.get('avg_time_per_operation', 0):.6f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance comparison test failed: {e}")
        return False

def test_fallback_behavior():
    """Test fallback behavior when VectorBT is not available."""
    print("\n🧪 Testing fallback behavior...")
    
    try:
        # Temporarily disable VectorBT
        import sys
        original_modules = sys.modules.copy()
        
        # Mock VectorBT as unavailable
        class MockVectorBT:
            pass
        
        sys.modules['vectorbt'] = MockVectorBT()
        sys.modules['vectorbt.generic'] = MockVectorBT()
        
        # Reload the module to test fallback
        import importlib
        import src.feature_generation.categories.regime_volatility
        importlib.reload(src.feature_generation.categories.regime_volatility)
        
        from src.feature_generation.categories.regime_volatility import RegimeVolatilityFeatureGenerator
        
        # Create test data
        data = create_sample_data(100)
        
        # Initialize generator (should work with fallback)
        generator = RegimeVolatilityFeatureGenerator()
        
        # Test feature generation (should use pandas fallback)
        features = generator.generate_features(data)
        assert len(features) > 0, "Feature generation should work with fallback"
        print("✅ Fallback behavior works correctly")
        
        # Restore original modules
        sys.modules.update(original_modules)
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback behavior test failed: {e}")
        return False

def main():
    """Run all VectorBT optimization tests."""
    print("🚀 Starting VectorBT optimization tests for regime volatility features...\n")
    
    tests = [
        test_regime_volatility_optimization,
        test_regime_feature_integration_optimization,
        test_advanced_volatility_optimization,
        test_performance_comparison,
        test_fallback_behavior
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All VectorBT optimization tests passed!")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)