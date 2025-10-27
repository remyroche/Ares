#!/usr/bin/env python3
"""
Test script to verify SR Detection artifact integration with BaseStep.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.market_analysis.sr_detection import SRDetectionStep

def create_sample_data():
    """Create sample market data for testing."""
    # Generate 1000 data points
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='15min')
    
    # Create realistic price data with some support/resistance levels
    base_price = 2000
    price_changes = np.random.normal(0, 0.02, 1000)  # 2% volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # Add some clear support/resistance levels
    prices = np.array(prices)
    prices[200:210] = 2100  # Resistance level
    prices[400:410] = 1900  # Support level
    prices[600:610] = 2200  # Another resistance level
    prices[800:810] = 1800  # Another support level
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 1000)
    })
    
    return data

async def test_sr_detection_artifact_integration():
    """Test the SR Detection step with artifact integration."""
    print("🧪 Testing SR Detection Artifact Integration")
    print("=" * 50)
    
    # Create SR Detection step
    sr_step = SRDetectionStep(step_name="test_sr_detection")
    
    # Create sample data
    print("📊 Creating sample market data...")
    data = create_sample_data()
    print(f"   Created {len(data)} data points")
    
    # Test configuration
    config = {
        'symbol': 'ETHUSDT',
        'exchange': 'binance',
        'timeframe': '15m',
        'direction': 'longs',
        'execution_mode': 'light',
        'dataframe': data
    }
    
    print("\n🎯 Testing SR Detection execution with artifact integration...")
    
    try:
        # Execute the step
        result = await sr_step.execute(config)
        
        print(f"\n✅ Execution completed!")
        print(f"   Success: {result['success']}")
        print(f"   Execution time: {result['execution_time']:.2f} seconds")
        print(f"   Artifacts saved: {len(result.get('artifacts', []))}")
        
        if result['success']:
            metrics = result.get('metrics', {})
            print(f"\n📈 Metrics:")
            print(f"   Total levels: {metrics.get('total_levels', 0)}")
            print(f"   Support levels: {metrics.get('support_levels', 0)}")
            print(f"   Resistance levels: {metrics.get('resistance_levels', 0)}")
            print(f"   Data rows: {metrics.get('data_rows', 0)}")
            print(f"   Data columns: {metrics.get('data_columns', 0)}")
            
            # Test loading from artifacts
            print(f"\n🔄 Testing artifact loading...")
            loaded_sr_levels = sr_step.load_sr_levels_from_artifacts(
                symbol='ETHUSDT',
                exchange='binance',
                direction='longs'
            )
            
            if loaded_sr_levels:
                print(f"   ✅ Successfully loaded SR levels from artifacts")
                print(f"   Loaded levels: {len(loaded_sr_levels.get('all_levels', []))}")
            else:
                print(f"   ❌ Failed to load SR levels from artifacts")
        
        else:
            print(f"   ❌ Execution failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Starting SR Detection Artifact Integration Tests")
    print("=" * 60)
    
    # Run tests
    asyncio.run(test_sr_detection_artifact_integration())
    
    print("\n🎉 All tests completed!")
