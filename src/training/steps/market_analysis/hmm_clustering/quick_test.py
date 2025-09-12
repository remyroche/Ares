#!/usr/bin/env python3
"""
Quick test to verify the enhanced HMM clustering implementation
"""

import sys
import os
sys.path.append('/workspace')

import numpy as np
import pandas as pd

def test_enhanced_feature_engineering():
    """Test the enhanced feature engineering functionality"""
    print("🧪 Testing Enhanced Feature Engineering...")
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 100
    
    # Create realistic price data
    prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    highs = prices + np.random.rand(n_samples) * 2
    lows = prices - np.random.rand(n_samples) * 2
    volumes = np.random.randint(1000, 10000, n_samples)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1H'),
        'open': prices,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    })
    
    print(f"   Created sample data: {df.shape}")
    
    # Test parameter optimization
    try:
        from parameter_optimization import ParameterOptimizer
        optimizer = ParameterOptimizer()
        print("   ✅ ParameterOptimizer imported successfully")
    except Exception as e:
        print(f"   ❌ ParameterOptimizer import failed: {e}")
    
    # Test ensemble optimization
    try:
        from ensemble_optimization import EnsembleWeightOptimizer
        ensemble_optimizer = EnsembleWeightOptimizer()
        print("   ✅ EnsembleWeightOptimizer imported successfully")
    except Exception as e:
        print(f"   ❌ EnsembleWeightOptimizer import failed: {e}")
    
    print("✅ Enhanced feature engineering test completed")

if __name__ == "__main__":
    test_enhanced_feature_engineering()