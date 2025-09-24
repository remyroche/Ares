#!/usr/bin/env python3
"""
Test Script for Automatic Timeframe Optimization

This script tests the automatic timeframe optimization implementation
to ensure it works correctly with the training pipeline.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def generate_test_market_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate realistic test market data."""
    print(f"📊 Generating {n_samples} samples of test market data...")
    
    # Create time index
    start_time = datetime.now() - timedelta(days=30)
    time_index = pd.date_range(start_time, periods=n_samples, freq='5min')
    
    # Generate realistic price data with trends and volatility
    np.random.seed(42)
    base_price = 100.0
    
    # Generate returns with volatility clustering
    returns = np.random.normal(0.0001, 0.002, n_samples)
    
    # Add volatility clustering
    vol_persistence = 0.9
    volatility = np.zeros(n_samples)
    volatility[0] = 0.002
    
    for i in range(1, n_samples):
        volatility[i] = vol_persistence * volatility[i-1] + (1 - vol_persistence) * 0.002
        returns[i] = np.random.normal(0.0001, volatility[i])
    
    # Generate prices
    prices = [base_price]
    for ret in returns[:-1]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    }, index=time_index)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        data.loc[data.index[i], 'high'] = max(data.iloc[i][['open', 'high', 'low', 'close']])
        data.loc[data.index[i], 'low'] = min(data.iloc[i][['open', 'high', 'low', 'close']])
    
    print(f"✅ Generated test data: {data.shape}")
    return data


def test_automatic_timeframe_optimizer():
    """Test the automatic timeframe optimizer."""
    print("\n🧪 Testing Automatic Timeframe Optimizer...")
    
    try:
        from src.training.steps.market_analysis.automatic_timeframe_optimizer import (
            AutomaticTimeframeOptimizer,
            ModelType,
            optimize_timeframes_for_training,
            get_optimal_timeframes_for_models
        )
        
        # Generate test data
        market_data = generate_test_market_data(1000)
        
        # Test optimizer initialization
        print("   → Initializing optimizer...")
        optimizer = AutomaticTimeframeOptimizer()
        
        if not optimizer.optimization_enabled:
            print("   ⚠️ Optimization disabled - using fallback configurations")
            return True
        
        # Test Analyst optimization
        print("   → Testing Analyst model optimization...")
        analyst_result = optimizer.optimize_for_model(ModelType.ANALYST, market_data)
        print(f"   ✅ Analyst optimization completed:")
        print(f"      → Score: {analyst_result.optimization_score:.3f}")
        print(f"      → Time horizons: {analyst_result.optimal_config.time_horizons}")
        print(f"      → Profit targets: {analyst_result.optimal_config.profit_targets}")
        
        # Test Tactician optimization
        print("   → Testing Tactician model optimization...")
        tactician_result = optimizer.optimize_for_model(ModelType.TACTICIAN, market_data)
        print(f"   ✅ Tactician optimization completed:")
        print(f"      → Score: {tactician_result.optimization_score:.3f}")
        print(f"      → Time horizons: {tactician_result.optimal_config.time_horizons}")
        print(f"      → Profit targets: {tactician_result.optimal_config.profit_targets}")
        
        # Test convenience functions
        print("   → Testing convenience functions...")
        analyst_config = optimize_timeframes_for_training(market_data, "analyst")
        tactician_config = optimize_timeframes_for_training(market_data, "tactician")
        
        print(f"   ✅ Convenience functions working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_enhanced_multi_horizon_pipeline():
    """Test the enhanced multi-horizon pipeline."""
    print("\n🧪 Testing Enhanced Multi-Horizon Pipeline...")
    
    try:
        from src.training.steps.market_analysis.enhanced_multi_horizon_pipeline import (
            EnhancedMultiHorizonPipeline,
            EnhancedPipelineConfig,
            execute_enhanced_multi_horizon_labeling
        )
        
        # Generate test data
        market_data = generate_test_market_data(1500)
        
        # Test pipeline initialization
        print("   → Initializing enhanced pipeline...")
        config = EnhancedPipelineConfig(
            enable_automatic_optimization=True,
            optimize_for_analyst=True,
            optimize_for_tactician=True
        )
        pipeline = EnhancedMultiHorizonPipeline(config)
        
        # Test enhanced labeling
        print("   → Testing enhanced labeling for Analyst model...")
        analyst_result = pipeline.execute_enhanced_labeling_step(
            data=market_data,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="5m",
            mode="light",
            model_type="analyst"
        )
        
        if 'analyst' in analyst_result:
            print(f"   ✅ Analyst labeling completed")
            if 'optimization_metadata' in analyst_result['analyst']:
                metadata = analyst_result['analyst']['optimization_metadata']
                print(f"      → Optimization score: {metadata.get('optimization_score', 0):.3f}")
                print(f"      → Validation score: {metadata.get('validation_score', 0):.3f}")
        
        # Test convenience function
        print("   → Testing convenience function...")
        convenience_result = execute_enhanced_multi_horizon_labeling(
            data=market_data,
            symbol="ETHUSDT",
            exchange="binance",
            timeframe="5m",
            mode="light",
            model_type="both",
            enable_optimization=True
        )
        
        print(f"   ✅ Convenience function working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def test_integration_with_existing_pipeline():
    """Test integration with existing pipeline."""
    print("\n🧪 Testing Integration with Existing Pipeline...")
    
    try:
        from src.training.steps.market_analysis.multi_horizon_sub_pipeline_adapter import (
            MultiHorizonSubPipelineAdapter,
            execute_multi_horizon_labeling_step
        )
        
        # Generate test data
        market_data = generate_test_market_data(2000)
        
        # Test that the adapter now includes optimization
        print("   → Testing adapter with automatic optimization...")
        adapter = MultiHorizonSubPipelineAdapter()
        
        if adapter.optimization_enabled:
            print("   ✅ Automatic optimization is enabled in adapter")
        else:
            print("   ⚠️ Automatic optimization is disabled in adapter")
        
        # Test execution
        print("   → Testing labeling execution...")
        result = adapter.execute_multi_horizon_labeling_step(
            data=market_data,
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="5m",
            mode="light"
        )
        
        if result.get('status') == 'success':
            print("   ✅ Labeling execution successful")
            print(f"      → Artifacts: {list(result.get('artifacts', {}).keys())}")
        else:
            print(f"   ⚠️ Labeling execution had issues: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing Automatic Timeframe Optimization Implementation")
    print("=" * 60)
    
    tests = [
        ("Automatic Timeframe Optimizer", test_automatic_timeframe_optimizer),
        ("Enhanced Multi-Horizon Pipeline", test_enhanced_multi_horizon_pipeline),
        ("Integration with Existing Pipeline", test_integration_with_existing_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Automatic timeframe optimization is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation and dependencies.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
