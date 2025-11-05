#!/usr/bin/env python3
"""
Direct test of clustering functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 500
    
    # Generate price data
    price = 100.0
    prices = [price]
    
    for i in range(1, n_samples):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        price = price * (1 + change)
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    # Add datetime index
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    data.index = dates
    
    print(f"✅ Sample data created")
    print(f"📊 Data shape: {data.shape}")
    print(f"📅 Date range: {data.index.min()} to {data.index.max()}")
    
    return data

def test_direct_clustering():
    """Test clustering directly."""
    try:
        # Import only what we need
        from src.training.steps.market_analysis.statsmodel_clustering.core import (
            create_enhanced_markov_regression_adapter
        )
        
        # Create sample data
        data = create_sample_data()
        
        # Prepare features
        features = pd.DataFrame(index=data.index)
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volatility'] = features['returns'].rolling(20).std()
        features['volatility_ratio'] = features['volatility'] / features['volatility'].rolling(50).mean()
        features['sma_5'] = data['close'].rolling(5).mean()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_ratio'] = features['sma_5'] / features['sma_20']
        
        # Remove NaN values
        features = features.dropna()
        
        print(f"📊 Features shape: {features.shape}")
        
        # Create adapter
        print("🔄 Creating clustering adapter...")
        adapter = create_enhanced_markov_regression_adapter(
            k_regimes=3,
            enable_pca=False,
            enable_diagnostics=True,
            enable_hardware_optimization=False
        )
        
        print("✅ Clustering adapter created")
        
        # Fit model
        print("🔄 Fitting clustering model...")
        result = adapter.fit(features)
        
        if result.success:
            print("✅ Clustering completed successfully")
            print(f"📊 Number of regimes: {result.n_regimes}")
            print(f"📊 Log likelihood: {result.log_likelihood:.2f}")
            print(f"📊 AIC: {result.aic:.2f}")
            print(f"📊 BIC: {result.bic:.2f}")
            print(f"📊 Processing time: {result.processing_time:.2f}s")
            
            if len(result.cluster_labels) > 0:
                print(f"📊 Cluster labels shape: {result.cluster_labels.shape}")
                print(f"📊 Unique regimes: {np.unique(result.cluster_labels)}")
            
            return True
        else:
            print(f"❌ Clustering failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Direct clustering test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run direct clustering test."""
    print("🧪 Running direct clustering test...\n")
    
    result = test_direct_clustering()
    
    if result:
        print("\n🎉 Direct clustering test passed!")
        return 0
    else:
        print("\n❌ Direct clustering test failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)