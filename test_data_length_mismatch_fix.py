#!/usr/bin/env python3
"""
Test script to verify the data length mismatch fix.
This script tests the improved regime classification and data splitting logic.
"""

import asyncio
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.steps.step2_market_regime_classification import MarketRegimeClassificationStep
from src.training.steps.step3_regime_data_splitting import RegimeDataSplittingStep
from src.utils.logger import system_logger

async def test_data_length_mismatch_fix():
    """Test the data length mismatch fix."""
    
    print("🧪 Testing Data Length Mismatch Fix...")
    
    # Create test configuration
    config = {
        "analyst": {
            "unified_regime_classifier": {
                "n_states": 4,
                "n_iter": 50,  # Reduced for testing
                "random_state": 42,
                "target_timeframe": "1h",
                "volatility_period": 10,
                "min_data_points": 500,  # Reduced for testing
            }
        }
    }
    
    # Create synthetic test data (similar to real market data)
    print("📊 Creating synthetic test data...")
    
    # Generate 1000 data points (similar to the real scenario)
    n_points = 1000
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq='1H')
    
    # Create realistic OHLCV data
    np.random.seed(42)
    base_price = 2000
    price_changes = np.random.normal(0, 0.02, n_points)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Create OHLCV data
    test_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_points)
    })
    
    print(f"✅ Created test data with {len(test_data)} records")
    
    # Test Step 2: Market Regime Classification
    print("\n🔄 Testing Step 2: Market Regime Classification...")
    
    step2 = MarketRegimeClassificationStep(config)
    await step2.initialize()
    
    training_input = {
        "symbol": "ETHUSDT",
        "exchange": "BINANCE",
        "data_dir": "data/training"
    }
    
    pipeline_state = {}
    
    try:
        step2_result = await step2.execute(training_input, pipeline_state)
        
        if step2_result.get("status") == "SUCCESS":
            print("✅ Step 2 completed successfully")
            
            # Check the regime sequence length
            regime_results = step2_result.get("regime_classification", {})
            regime_sequence = regime_results.get("regime_sequence", [])
            
            print(f"📈 Original data length: {len(test_data)}")
            print(f"📊 Regime sequence length: {len(regime_sequence)}")
            print(f"📊 Regime distribution: {regime_results.get('regime_distribution', {})}")
            
            # Test Step 3: Regime Data Splitting
            print("\n🔄 Testing Step 3: Regime Data Splitting...")
            
            step3 = RegimeDataSplittingStep(config)
            await step3.initialize()
            
            # Create a mock prepared data DataFrame
            prepared_data = test_data.copy()
            prepared_data['target'] = np.random.choice([0, 1], size=len(prepared_data))
            
            # Test the regime splitting logic directly
            regime_data = await step3._split_data_by_regimes(
                prepared_data, regime_results, "ETHUSDT", "BINANCE"
            )
            
            print("✅ Step 3 completed successfully")
            
            # Verify the results
            total_regime_records = sum(len(df) for df in regime_data.values())
            print(f"📊 Total records in regime data: {total_regime_records}")
            print(f"📊 Original data length: {len(prepared_data)}")
            
            if total_regime_records == len(prepared_data):
                print("✅ Data length mismatch fix successful! All records preserved.")
            else:
                print(f"⚠️  Data length mismatch still exists: {total_regime_records} vs {len(prepared_data)}")
            
            # Print regime breakdown
            for regime, df in regime_data.items():
                print(f"  - {regime}: {len(df)} records")
                
        else:
            print(f"❌ Step 2 failed: {step2_result}")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_data_length_mismatch_fix()) 