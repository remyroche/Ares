#!/usr/bin/env python3
"""
Test script for the Unified Regime Classifier.

Usage:
    python scripts/test_unified_regime_classifier.py --symbol ETHUSDT
    python scripts/test_unified_regime_classifier.py --symbol BTCUSDT --use-real-data
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
from src.config import CONFIG


async def test_unified_regime_classifier(symbol: str, use_real_data: bool = False):
    """Test the unified regime classifier."""
    print(f"🧪 Testing Unified Regime Classifier for {symbol}")
    
    # Initialize classifier
    classifier = UnifiedRegimeClassifier(CONFIG)
    
    if use_real_data:
        # Load real data
        print("📊 Loading real market data...")
        data_path = f"data_cache/klines_BINANCE_{symbol}_1m_consolidated.csv"
        
        if not os.path.exists(data_path):
            print(f"❌ No data found at {data_path}")
            print("💡 Please download data first using: python ares_launcher.py load --symbol ETHUSDT --exchange BINANCE")
            return False
        
        # Load and resample data
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Filter to last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Resample to 1h
        df = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        print(f"✅ Loaded {len(df)} 1h records from {start_date.date()} to {end_date.date()}")
        
    else:
        # Generate synthetic data
        print("📊 Generating synthetic market data...")
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='1H')
        
        # Create realistic price movements
        np.random.seed(42)
        base_price = 2000 if symbol == 'ETHUSDT' else 50000
        
        # Generate price series with different regimes
        prices = [base_price]
        for i in range(len(dates) - 1):
            # Different volatility regimes
            if i < len(dates) // 3:
                # Bull market
                change = np.random.normal(0.001, 0.02)
            elif i < 2 * len(dates) // 3:
                # Bear market
                change = np.random.normal(-0.001, 0.02)
            else:
                # Sideways market
                change = np.random.normal(0, 0.01)
            
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # Prevent negative prices
        
        # Create OHLCV data
        df = pd.DataFrame({
            'open': prices[:-1],
            'close': prices[1:],
            'high': [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(prices[:-1], prices[1:])],
            'low': [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(prices[:-1], prices[1:])],
            'volume': np.random.lognormal(10, 1, len(prices) - 1)
        }, index=dates[:-1])
        
        print(f"✅ Generated {len(df)} synthetic 1h records")
    
    # Test training
    print("\n🎯 Testing training pipeline...")
    
    # Step 1: Train HMM labeler
    print("   Step 1: Training HMM labeler...")
    success = await classifier.train_hmm_labeler(df)
    if not success:
        print("❌ HMM labeler training failed")
        return False
    print("✅ HMM labeler trained successfully")
    
    # Step 2: Train basic ensemble
    print("   Step 2: Training basic ensemble...")
    success = await classifier.train_basic_ensemble(df)
    if not success:
        print("❌ Basic ensemble training failed")
        return False
    print("✅ Basic ensemble trained successfully")
    
    # Step 3: Train advanced classifier
    print("   Step 3: Training advanced classifier...")
    success = await classifier.train_advanced_classifier(df)
    if not success:
        print("❌ Advanced classifier training failed")
        return False
    print("✅ Advanced classifier trained successfully")
    
    # Test complete system training
    print("\n🚀 Testing complete system training...")
    classifier2 = UnifiedRegimeClassifier(CONFIG)
    success = await classifier2.train_complete_system(df)
    if not success:
        print("❌ Complete system training failed")
        return False
    print("✅ Complete system training successful")
    
    # Test prediction
    print("\n🔮 Testing prediction...")
    recent_data = df.tail(100)  # Use last 100 periods for prediction
    
    regime, confidence, info = classifier2.predict_regime(recent_data)
    print(f"   Predicted regime: {regime}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Basic regime: {info.get('basic_regime', 'N/A')}")
    print(f"   Advanced regime: {info.get('advanced_regime', 'N/A')}")
    
    # Test model saving/loading
    print("\n💾 Testing model persistence...")
    classifier2.save_models()
    
    classifier3 = UnifiedRegimeClassifier(CONFIG)
    success = classifier3.load_models()
    if not success:
        print("❌ Model loading failed")
        return False
    print("✅ Model loading successful")
    
    # Test prediction with loaded models
    regime2, confidence2, info2 = classifier3.predict_regime(recent_data)
    print(f"   Loaded model prediction: {regime2} (confidence: {confidence2:.3f})")
    
    # Test system status
    print("\n📊 System status:")
    status = classifier3.get_system_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print("\n✅ All tests passed! Unified regime classifier is working correctly.")
    return True


def main():
    parser = argparse.ArgumentParser(description='Test Unified Regime Classifier')
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--use-real-data', action='store_true', help='Use real market data instead of synthetic')
    
    args = parser.parse_args()
    
    # Run test
    success = asyncio.run(test_unified_regime_classifier(args.symbol, args.use_real_data))
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 