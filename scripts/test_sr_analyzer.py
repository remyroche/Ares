#!/usr/bin/env python3
"""
Test script for the SR Analyzer.

Usage:
    python scripts/test_sr_analyzer.py --symbol ETHUSDT
    python scripts/test_sr_analyzer.py --symbol BTCUSDT --use-real-data
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.config import CONFIG


async def test_sr_analyzer(symbol: str, use_real_data: bool = False):
    """Test the SR analyzer functionality."""
    print(f"🧪 Testing SR Analyzer for {symbol}")
    
    # Initialize SR analyzer
    sr_analyzer = SRLevelAnalyzer(CONFIG)
    await sr_analyzer.initialize()
    
    # Generate or load test data
    if use_real_data:
        # Load real data from data directory
        data_file = f"data/{symbol}_1h.csv"
        if os.path.exists(data_file):
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        else:
            print(f"❌ Real data file not found: {data_file}")
            return
    else:
        # Generate synthetic data
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        np.random.seed(42)
        
        # Create realistic price data with support/resistance levels
        base_price = 2000 if symbol == 'ETHUSDT' else 50000
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Add some trend and volatility
            trend = np.sin(i / 100) * 0.01  # Reduced trend
            noise = np.random.normal(0, 0.005)  # Reduced noise
            current_price *= (1 + trend + noise)
            # Ensure price stays reasonable
            current_price = max(current_price, base_price * 0.5)
            current_price = min(current_price, base_price * 2.0)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(prices))
        }, index=dates)
    
    print(f"📊 Data shape: {df.shape}")
    print(f"📊 Date range: {df.index.min()} to {df.index.max()}")
    
    # Test SR analysis
    print("\n🔍 Running SR analysis...")
    try:
        sr_results = await sr_analyzer.analyze(df)
        
        print(f"✅ SR analysis completed successfully")
        print(f"📈 Support levels found: {len(sr_results.get('support_levels', []))}")
        print(f"📉 Resistance levels found: {len(sr_results.get('resistance_levels', []))}")
        
        # Display some results
        if sr_results.get('support_levels'):
            print("\n📈 Top Support Levels:")
            for i, level in enumerate(sr_results['support_levels'][:3]):
                print(f"  {i+1}. Price: {level.get('price', 'N/A'):.2f}, "
                      f"Strength: {level.get('strength', 'N/A'):.3f}, "
                      f"Touches: {level.get('touch_count', 'N/A')}")
        
        if sr_results.get('resistance_levels'):
            print("\n📉 Top Resistance Levels:")
            for i, level in enumerate(sr_results['resistance_levels'][:3]):
                print(f"  {i+1}. Price: {level.get('price', 'N/A'):.2f}, "
                      f"Strength: {level.get('strength', 'N/A'):.3f}, "
                      f"Touches: {level.get('touch_count', 'N/A')}")
        
        # Test SR zone proximity detection
        print("\n🎯 Testing SR zone proximity detection...")
        current_price = df['close'].iloc[-1]
        proximity = sr_analyzer.detect_sr_zone_proximity(
            current_price
        )
        
        print(f"💰 Current price: {current_price:.2f}")
        print(f"🎯 SR zone proximity: {proximity}")
        
        return sr_results
        
    except Exception as e:
        print(f"❌ Error during SR analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test SR Analyzer")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to test")
    parser.add_argument("--use-real-data", action="store_true", help="Use real data instead of synthetic")
    
    args = parser.parse_args()
    
    asyncio.run(test_sr_analyzer(args.symbol, args.use_real_data))


if __name__ == "__main__":
    main() 