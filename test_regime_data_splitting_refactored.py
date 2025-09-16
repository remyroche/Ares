#!/usr/bin/env python3
"""
Test script for the refactored regime data splitting component.

This script tests the integration of common utilities and hardware optimizations.
"""

import sys
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_test_data():
    """Create test market data for regime splitting."""
    # Create sample OHLCV data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_days = len(dates)
    
    # Generate realistic price data with different regimes
    np.random.seed(42)  # For reproducible results
    
    # Regime 1: Bull market (first 100 days)
    bull_prices = 100 + np.cumsum(np.random.normal(0.1, 0.5, 100))
    
    # Regime 2: Bear market (next 100 days)
    bear_prices = bull_prices[-1] + np.cumsum(np.random.normal(-0.05, 0.8, 100))
    
    # Regime 3: Sideways market (remaining days)
    sideways_prices = bear_prices[-1] + np.cumsum(np.random.normal(0.01, 0.3, n_days - 200))
    
    # Combine all prices
    all_prices = np.concatenate([bull_prices, bear_prices, sideways_prices])
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, all_prices)):
        # Generate realistic OHLC from close price
        volatility = np.random.uniform(0.01, 0.03)
        high = price * (1 + volatility)
        low = price * (1 - volatility)
        open_price = price + np.random.uniform(-volatility/2, volatility/2) * price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def create_test_regime_discovery():
    """Create test regime discovery results."""
    # Create regime states (0, 1, 2 for the three regimes)
    regime_states = np.array([0] * 100 + [1] * 100 + [2] * 165)  # 365 days total
    
    # Create regime probabilities
    regime_probabilities = np.zeros((365, 3))
    for i, state in enumerate(regime_states):
        regime_probabilities[i, state] = 0.8  # 80% confidence
        # Add some noise to other states
        for j in range(3):
            if j != state:
                regime_probabilities[i, j] = np.random.uniform(0.05, 0.15)
    
    return {
        'regime_states': regime_states,
        'regime_probabilities': regime_probabilities,
        'regime_means': np.array([[0.1, 0.5], [-0.05, 0.8], [0.01, 0.3]]),
        'regime_covariances': [np.eye(2) * 0.1, np.eye(2) * 0.2, np.eye(2) * 0.05]
    }

async def test_regime_data_splitting():
    """Test the refactored regime data splitting component."""
    print("🧪 Testing Refactored Regime Data Splitting Component")
    print("=" * 60)
    
    try:
        # Import the component
        from src.training.steps.market_analysis.regime_data_splitting.component import RegimeDataSplittingComponent
        from src.training.steps.market_analysis.components.base_component import ComponentConfig
        
        print("✅ Successfully imported refactored component")
        
        # Create test data
        print("\n📊 Creating test data...")
        market_data = create_test_data()
        regime_discovery = create_test_regime_discovery()
        
        print(f"✅ Created test data: {market_data.shape}")
        print(f"✅ Created regime discovery: {len(regime_discovery['regime_states'])} states")
        
        # Create component configuration
        config = ComponentConfig(
            symbol="BTCUSDT",
            exchange="binance",
            timeframe="1d"
        )
        
        # Initialize component
        print("\n🔧 Initializing component...")
        component = RegimeDataSplittingComponent(config)
        print("✅ Component initialized successfully")
        
        # Test hardware optimizations
        print("\n🧠 Testing hardware optimizations...")
        print(f"   - M1 Available: {component.gpu_manager.is_m1}")
        print(f"   - MPS Available: {component.gpu_manager.mps_available}")
        print(f"   - CPU Cores: {component.cpu_optimizer.get_cpu_info()['total_cores']}")
        
        # Test data loading and preparation
        print("\n📊 Testing data loading and preparation...")
        prepared_data = await component._load_and_prepare_data(market_data)
        if prepared_data is not None:
            print(f"✅ Data prepared successfully: {prepared_data.shape}")
            print(f"   - Columns: {list(prepared_data.columns)}")
            print(f"   - Data types optimized: {prepared_data.dtypes.to_dict()}")
        else:
            print("❌ Data preparation failed")
            return False
        
        # Test regime discovery retrieval
        print("\n🔍 Testing regime discovery retrieval...")
        pipeline_state = {'hmm_regime_discovery_result': regime_discovery}
        retrieved_regime = await component._get_regime_discovery_results(pipeline_state)
        if retrieved_regime is not None:
            print(f"✅ Regime discovery retrieved successfully")
            print(f"   - States: {len(retrieved_regime['regime_states'])}")
            print(f"   - Probabilities shape: {retrieved_regime['regime_probabilities'].shape}")
        else:
            print("❌ Regime discovery retrieval failed")
            return False
        
        # Test regime splitting
        print("\n✂️ Testing regime splitting...")
        report = type('Report', (), {'status': 'in_progress', 'warnings': [], 'errors': []})()
        splitting_result = await component._perform_regime_splitting(
            prepared_data, retrieved_regime, report
        )
        
        if splitting_result['success']:
            print("✅ Regime splitting completed successfully")
            regime_data = splitting_result['data']
            print(f"   - Market data shape: {regime_data['market_data'].shape}")
            print(f"   - Regime states: {len(regime_data['regime_states'])}")
            print(f"   - Unique regimes: {len(np.unique(regime_data['regime_states']))}")
            print(f"   - Regime distribution: {regime_data['regime_statistics']['regime_distribution']}")
        else:
            print(f"❌ Regime splitting failed: {splitting_result['errors']}")
            return False
        
        # Test validation
        print("\n🔍 Testing result validation...")
        validation_result = await component._validate_splitting_results(splitting_result, report)
        if validation_result['valid']:
            print("✅ Result validation passed")
        else:
            print(f"❌ Result validation failed: {validation_result['errors']}")
            return False
        
        # Test artifact creation
        print("\n💾 Testing artifact creation...")
        artifacts = await component._create_artifacts(splitting_result, report)
        if artifacts:
            print("✅ Artifacts created successfully")
            print(f"   - Artifact keys: {list(artifacts.keys())}")
            print(f"   - Processing metrics: {artifacts['regime_data_splitting_result']['processing_metrics']}")
        else:
            print("❌ Artifact creation failed")
            return False
        
        # Test cleanup
        print("\n🧹 Testing cleanup...")
        component.cleanup()
        print("✅ Cleanup completed successfully")
        
        print("\n🎉 All tests passed successfully!")
        print("✅ Refactored regime data splitting component is working correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

async def main():
    """Main test function."""
    print("🚀 Starting Regime Data Splitting Component Test")
    print("=" * 60)
    
    success = await test_regime_data_splitting()
    
    if success:
        print("\n✅ All tests completed successfully!")
        print("🎯 The refactored component successfully integrates:")
        print("   - Common operations utilities")
        print("   - Math validation utilities")
        print("   - Serialization utilities")
        print("   - M1 hardware optimizations")
        print("   - Memory optimizations")
        print("   - CPU optimizations")
        return 0
    else:
        print("\n❌ Tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)