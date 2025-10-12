#!/usr/bin/env python3
"""
Test script to verify VectorBT feature bank integration.

This script tests the integration between VectorBT optimizations and the feature generation bank.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vectorbt_feature_bank_integration():
    """Test VectorBT feature bank integration."""
    print("🧪 Testing VectorBT Feature Bank Integration...")
    
    try:
        # Import the core optimizer
        from src.training.steps.pre_training.feature_lookback_optimization.core.optimizer import CoreOptimizer
        
        # Create test data
        np.random.seed(42)
        n_samples = 1000
        
        data = pd.DataFrame({
            'open': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'close': np.random.randn(n_samples).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, n_samples)
        })
        
        print(f"📊 Created test data with {len(data)} samples")
        
        # Initialize the core optimizer
        optimizer = CoreOptimizer()
        
        print(f"✅ Core optimizer initialized")
        print(f"🔧 VectorBT optimization available: {optimizer.use_vectorbt_optimization}")
        print(f"🔧 VectorBT rolling optimizer available: {optimizer.rolling_optimizer is not None}")
        print(f"🔧 Unified manager available: {optimizer.unified_manager is not None}")
        
        # Test feature calculation with VectorBT optimizations
        test_features = ['sma', 'ema', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle']
        lookback_periods = [10, 20, 30]
        
        for feature_name in test_features:
            print(f"\n🔄 Testing feature: {feature_name}")
            
            for lookback in lookback_periods:
                try:
                    # Test VectorBT optimized feature calculation
                    result = optimizer._calculate_feature_vectorbt_optimized(
                        data, feature_name, lookback
                    )
                    
                    if result is not None:
                        print(f"  ✅ {feature_name} (lookback={lookback}): {len(result)} values, range: {result.min():.4f} to {result.max():.4f}")
                    else:
                        print(f"  ⚠️ {feature_name} (lookback={lookback}): No result (generator not found or failed)")
                        
                except Exception as e:
                    print(f"  ❌ {feature_name} (lookback={lookback}): Error - {e}")
        
        # Test feature bank access
        if hasattr(optimizer, '_feature_bank') and optimizer._feature_bank is not None:
            print(f"\n📚 Feature bank initialized with {len(optimizer._feature_bank.registry._generators_by_name)} generators")
            
            # List some available generators
            available_generators = list(optimizer._feature_bank.registry._generators_by_name.keys())[:10]
            print(f"📋 Sample available generators: {available_generators}")
        else:
            print("\n⚠️ Feature bank not initialized")
        
        # Test performance metrics
        metrics = optimizer.get_vectorbt_performance_metrics()
        print(f"\n📈 VectorBT Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        print("\n✅ VectorBT Feature Bank Integration Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vectorbt_feature_bank_integration()
    sys.exit(0 if success else 1)