#!/usr/bin/env python3
"""
Test script for the Candle Analyzer.

Usage:
    python scripts/test_candle_analyzer.py --symbol ETHUSDT
    python scripts/test_candle_analyzer.py --symbol BTCUSDT --use-real-data
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

from src.analyst.candle_analyzer import CandleAnalyzer
from src.config import CONFIG


async def test_candle_analyzer(symbol: str, use_real_data: bool = False):
    """Test the candle analyzer functionality."""
    print(f"🧪 Testing Candle Analyzer for {symbol}")
    
    # Initialize candle analyzer
    candle_analyzer = CandleAnalyzer(CONFIG)
    await candle_analyzer.initialize()
    
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
        # Generate synthetic data with various candle patterns
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1h')
        np.random.seed(42)
        
        # Create realistic price data with different candle patterns
        base_price = 2000 if symbol == 'ETHUSDT' else 50000
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Add some trend and volatility
            trend = np.sin(i / 100) * 0.01
            noise = np.random.normal(0, 0.005)
            current_price *= (1 + trend + noise)
            
            # Ensure price stays reasonable
            current_price = max(current_price, base_price * 0.8)
            current_price = min(current_price, base_price * 1.2)
            prices.append(current_price)
        
        # Create OHLCV data with various patterns
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(prices))
        }, index=dates)
        
        # Add some specific patterns
        # Large candle at index 1000
        df.iloc[1000, df.columns.get_loc('high')] = df.iloc[1000]['close'] * 1.05
        df.iloc[1000, df.columns.get_loc('low')] = df.iloc[1000]['open'] * 0.95
        
        # Doji pattern at index 2000
        df.iloc[2000, df.columns.get_loc('close')] = df.iloc[2000]['open'] * 1.001
        
        # Hammer pattern at index 3000
        df.iloc[3000, df.columns.get_loc('low')] = df.iloc[3000]['open'] * 0.97
        df.iloc[3000, df.columns.get_loc('close')] = df.iloc[3000]['open'] * 1.001
        df.iloc[3000, df.columns.get_loc('high')] = df.iloc[3000]['open'] * 1.002
    
    print(f"📊 Data shape: {df.shape}")
    print(f"📊 Date range: {df.index.min()} to {df.index.max()}")
    
    # Test candle analysis
    print("\n🔍 Running candle analysis...")
    try:
        candle_results = await candle_analyzer.analyze(df)
        
        print(f"✅ Candle analysis completed successfully")
        print(f"📈 Candle patterns found: {len(candle_results.get('candle_patterns', []))}")
        print(f"📉 Large candles found: {len(candle_results.get('large_candles', []))}")
        
        # Display pattern results
        if candle_results.get('candle_patterns'):
            print("\n📈 Candle Patterns Found:")
            pattern_counts = {}
            for pattern in candle_results['candle_patterns']:
                pattern_type = pattern.get('pattern', 'unknown')
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            for pattern_type, count in pattern_counts.items():
                print(f"  - {pattern_type}: {count}")
        
        # Display large candle results
        if candle_results.get('large_candles'):
            print("\n📉 Large Candles Found:")
            for i, candle in enumerate(candle_results['large_candles'][:5]):
                print(f"  {i+1}. {candle.get('size_class', 'unknown')} candle at {candle.get('timestamp', 'N/A')}")
                print(f"     Range ratio: {candle.get('range_ratio', 'N/A'):.2f}x average")
                print(f"     Confidence: {candle.get('confidence', 'N/A'):.3f}")
        
        # Display size classification
        if candle_results.get('candle_sizes', {}).get('statistics'):
            stats = candle_results['candle_sizes']['statistics']
            print(f"\n📊 Size Classification:")
            print(f"  - Small: {stats.get('small_count', 0)}")
            print(f"  - Normal: {stats.get('normal_count', 0)}")
            print(f"  - Large: {stats.get('large_count', 0)}")
            print(f"  - Huge: {stats.get('huge_count', 0)}")
            print(f"  - Extreme: {stats.get('extreme_count', 0)}")
        
        # Test large candle detection
        print("\n🎯 Testing large candle detection...")
        current_candle = {
            'open': df['open'].iloc[-1],
            'high': df['high'].iloc[-1],
            'low': df['low'].iloc[-1],
            'close': df['close'].iloc[-1],
            'volume': df['volume'].iloc[-1]
        }
        
        detection_result = candle_analyzer.detect_large_candle(current_candle)
        print(f"💰 Current candle analysis:")
        print(f"   Is large: {detection_result.get('is_large', False)}")
        print(f"   Size class: {detection_result.get('size_class', 'unknown')}")
        print(f"   Confidence: {detection_result.get('confidence', 0):.3f}")
        print(f"   Range ratio: {detection_result.get('range_ratio', 0):.2f}x average")
        print(f"   Reason: {detection_result.get('reason', 'N/A')}")
        
        # Display volatility analysis
        if candle_results.get('volatility_analysis'):
            vol_analysis = candle_results['volatility_analysis']
            print(f"\n📈 Volatility Analysis:")
            print(f"   Current volatility: {vol_analysis.get('current_volatility', 0):.4f}")
            print(f"   Average volatility: {vol_analysis.get('avg_volatility', 0):.4f}")
            print(f"   Volatility percentile: {vol_analysis.get('volatility_percentile', 0):.1f}%")
            print(f"   Volatility trend: {vol_analysis.get('volatility_trend', 'unknown')}")
        
        # Display statistical analysis
        if candle_results.get('statistical_analysis'):
            stat_analysis = candle_results['statistical_analysis']
            print(f"\n📊 Statistical Analysis:")
            if 'range_statistics' in stat_analysis:
                range_stats = stat_analysis['range_statistics']
                print(f"   Range outliers: {range_stats.get('outliers_count', 0)}")
                print(f"   Range mean: {range_stats.get('mean', 0):.4f}")
                print(f"   Range std: {range_stats.get('std', 0):.4f}")
        
        return candle_results
        
    except Exception as e:
        print(f"❌ Error during candle analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test Candle Analyzer")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to test")
    parser.add_argument("--use-real-data", action="store_true", help="Use real data instead of synthetic")
    
    args = parser.parse_args()
    
    asyncio.run(test_candle_analyzer(args.symbol, args.use_real_data))


if __name__ == "__main__":
    main() 