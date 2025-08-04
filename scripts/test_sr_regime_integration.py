#!/usr/bin/env python3
"""
Test script for SR Analyzer integration with Unified Regime Classifier.

Usage:
    python scripts/test_sr_regime_integration.py --symbol ETHUSDT
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

from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.analyst.sr_analyzer import SRLevelAnalyzer
from src.config import CONFIG


async def test_sr_regime_integration(symbol: str):
    """Test the integration between SR analyzer and regime classifier."""
    print(f"🧪 Testing SR Analyzer + Regime Classifier Integration for {symbol}")
    
    # Initialize both components
    regime_classifier = UnifiedRegimeClassifier(CONFIG)
    sr_analyzer = SRLevelAnalyzer(CONFIG)
    
    # Initialize SR analyzer
    await sr_analyzer.initialize()
    print("✅ SR Analyzer initialized")
    
    # Generate test data with clear support/resistance levels
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1h')
    np.random.seed(42)
    
    # Create realistic price data with clear SR levels
    base_price = 2000 if symbol == 'ETHUSDT' else 50000
    prices = []
    current_price = base_price
    
    # Define some SR levels
    support_levels = [base_price * 0.9, base_price * 0.95]
    resistance_levels = [base_price * 1.05, base_price * 1.1]
    
    for i in range(len(dates)):
        # Add some trend and volatility
        trend = np.sin(i / 100) * 0.005
        noise = np.random.normal(0, 0.002)
        current_price *= (1 + trend + noise)
        
        # Ensure price stays reasonable
        current_price = max(current_price, base_price * 0.8)
        current_price = min(current_price, base_price * 1.2)
        prices.append(current_price)
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, len(prices))
    }, index=dates)
    
    print(f"📊 Generated test data: {df.shape}")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test SR analysis
    print("\n🔍 Running SR analysis...")
    sr_results = await sr_analyzer.analyze(df)
    
    print(f"📈 Support levels found: {len(sr_results.get('support_levels', []))}")
    print(f"📉 Resistance levels found: {len(sr_results.get('resistance_levels', []))}")
    
    # Display SR levels
    if sr_results.get('support_levels'):
        print("\n📈 Support Levels:")
        for i, level in enumerate(sr_results['support_levels'][:3]):
            print(f"  {i+1}. Price: ${level.get('price', 'N/A'):.2f}, "
                  f"Strength: {level.get('strength', 'N/A'):.3f}")
    
    if sr_results.get('resistance_levels'):
        print("\n📉 Resistance Levels:")
        for i, level in enumerate(sr_results['resistance_levels'][:3]):
            print(f"  {i+1}. Price: ${level.get('price', 'N/A'):.2f}, "
                  f"Strength: {level.get('strength', 'N/A'):.3f}")
    
    # Test regime classifier training
    print("\n🎯 Training regime classifier...")
    training_success = await regime_classifier.train_complete_system(df)
    
    if not training_success:
        print("❌ Regime classifier training failed")
        return
    
    print("✅ Regime classifier trained successfully")
    
    # Test predictions at different price levels
    print("\n🔮 Testing predictions at different price levels...")
    
    test_prices = [
        df['close'].iloc[-1],  # Current price
        df['close'].iloc[-1] * 0.95,  # Near support
        df['close'].iloc[-1] * 1.05,  # Near resistance
        df['close'].iloc[-1] * 0.9,   # At support
        df['close'].iloc[-1] * 1.1,   # At resistance
    ]
    
    for i, test_price in enumerate(test_prices):
        # Create test data with the specific price
        test_df = df.copy()
        test_df.iloc[-1, test_df.columns.get_loc('close')] = test_price
        test_df.iloc[-1, test_df.columns.get_loc('high')] = test_price * 1.001
        test_df.iloc[-1, test_df.columns.get_loc('low')] = test_price * 0.999
        
        # Get regime prediction
        regime, confidence, additional_info = regime_classifier.predict_regime(test_df)
        
        # Check SR zone proximity
        sr_proximity = sr_analyzer.detect_sr_zone_proximity(test_price)
        
        print(f"\n  Test {i+1}: Price ${test_price:.2f}")
        print(f"    Regime: {regime} (confidence: {confidence:.3f})")
        print(f"    SR Zone: {sr_proximity.get('in_zone', False)}")
        
        if sr_proximity.get('in_zone', False):
            print(f"    SR Info: {sr_proximity.get('level_type', 'N/A')} at ${sr_proximity.get('nearest_level', 'N/A')}")
        
        if 'sr_zone_info' in additional_info:
            print(f"    SR Zone Detected: {additional_info['sr_zone_info']}")
    
    # Test the integration specifically
    print("\n🎯 Testing SR-Zone-Action detection...")
    
    # Create a scenario where price is near SR level
    near_sr_df = df.copy()
    if sr_results.get('support_levels'):
        near_price = sr_results['support_levels'][0]['price'] * 1.01  # 1% above support
        near_sr_df.iloc[-1, near_sr_df.columns.get_loc('close')] = near_price
        
        regime, confidence, additional_info = regime_classifier.predict_regime(near_sr_df)
        
        print(f"  Price near support (${near_price:.2f}):")
        print(f"    Predicted regime: {regime}")
        print(f"    Confidence: {confidence:.3f}")
        print(f"    Additional info: {additional_info}")
    
    print("\n✅ SR Analyzer + Regime Classifier integration test completed!")


def main():
    parser = argparse.ArgumentParser(description="Test SR Analyzer + Regime Classifier Integration")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to test")
    
    args = parser.parse_args()
    
    asyncio.run(test_sr_regime_integration(args.symbol))


if __name__ == "__main__":
    main() 