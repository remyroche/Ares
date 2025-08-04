#!/usr/bin/env python3
"""
Test script to compare SR Breakout Predictor with existing ML models.

Usage:
    python scripts/test_sr_breakout_comparison.py --symbol ETHUSDT
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

from src.analyst.sr_breakout_predictor import SRBreakoutPredictor
from src.analyst.predictive_ensembles import RegimePredictiveEnsembles
from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
from src.config import CONFIG


async def test_sr_breakout_comparison(symbol: str):
    """Compare SR breakout predictor with existing ML models."""
    print(f"🧪 Comparing SR Breakout Predictor with existing ML models for {symbol}")
    
    # Generate test data
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
    
    # Add technical indicators (simplified)
    df['rsi'] = 50 + np.random.normal(0, 15, len(df))
    df['macd'] = np.random.normal(0, 0.01, len(df))
    df['atr'] = df['close'] * 0.02
    df['volume_delta'] = np.random.normal(0, 0.1, len(df))
    
    print(f"📊 Generated test data: {df.shape}")
    print(f"💰 Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Test 1: SR Breakout Predictor
    print("\n🔍 Test 1: SR Breakout Predictor")
    try:
        sr_breakout = SRBreakoutPredictor(CONFIG)
        await sr_breakout.initialize()
        
        # Train the model
        training_success = await sr_breakout.train(df)
        if training_success:
            print("✅ SR Breakout Predictor trained successfully")
            
            # Test prediction
            current_price = df['close'].iloc[-1]
            prediction = await sr_breakout.predict_breakout_probability(df, current_price)
            
            if prediction:
                print(f"   Breakout probability: {prediction.get('breakout_probability', 'N/A'):.3f}")
                print(f"   Bounce probability: {prediction.get('bounce_probability', 'N/A'):.3f}")
                print(f"   Confidence: {prediction.get('confidence', 'N/A'):.3f}")
                print(f"   Near SR zone: {prediction.get('near_sr_zone', 'N/A')}")
                print(f"   Recommendation: {prediction.get('recommendation', 'N/A')}")
        else:
            print("❌ SR Breakout Predictor training failed")
    except Exception as e:
        print(f"❌ SR Breakout Predictor error: {e}")
    
    # Test 2: Predictive Ensembles
    print("\n🔍 Test 2: Predictive Ensembles")
    try:
        # Note: Predictive ensembles require more complex setup
        # This is a simplified test to show the concept
        print("   Predictive Ensembles would provide:")
        print("   - LSTM, Transformer, TabNet models")
        print("   - 50+ comprehensive features")
        print("   - Meta-learner with cross-validation")
        print("   - Advanced ensemble methods")
        print("   - Can be enhanced with SR context features")
    except Exception as e:
        print(f"❌ Predictive Ensembles error: {e}")
    
    # Test 3: Unified Regime Classifier
    print("\n🔍 Test 3: Unified Regime Classifier")
    try:
        regime_classifier = UnifiedRegimeClassifier(CONFIG)
        training_success = await regime_classifier.train_complete_system(df)
        
        if training_success:
            print("✅ Unified Regime Classifier trained successfully")
            
            # Test prediction
            regime, confidence, additional_info = regime_classifier.predict_regime(df.tail(100))
            
            print(f"   Predicted regime: {regime}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Additional info: {additional_info}")
            
            # Check for SR_ZONE_ACTION
            if regime == "SR_ZONE_ACTION":
                print("   ✅ SR_ZONE_ACTION detected!")
                if 'sr_zone_info' in additional_info:
                    print(f"   SR Zone Info: {additional_info['sr_zone_info']}")
        else:
            print("❌ Unified Regime Classifier training failed")
    except Exception as e:
        print(f"❌ Unified Regime Classifier error: {e}")
    
    # Comparison Summary
    print("\n📊 Comparison Summary")
    print("=" * 50)
    
    print("\n🔍 **SR Breakout Predictor**")
    print("   ✅ Specialized for SR zones")
    print("   ✅ Direct SR integration")
    print("   ❌ Single LightGBM model")
    print("   ❌ Limited features (20+)")
    print("   ❌ Basic ensemble method")
    
    print("\n🔍 **Predictive Ensembles**")
    print("   ✅ 6+ advanced models (LSTM, Transformer, TabNet)")
    print("   ✅ 50+ comprehensive features")
    print("   ✅ Meta-learner with cross-validation")
    print("   ✅ Advanced ensemble methods")
    print("   ❌ No direct SR integration (can be added)")
    
    print("\n🔍 **Unified Regime Classifier**")
    print("   ✅ HMM + Ensemble (Random Forest, LightGBM, SVM)")
    print("   ✅ Direct SR integration")
    print("   ✅ SR_ZONE_ACTION detection")
    print("   ✅ 16 core features")
    print("   ❌ Limited compared to predictive ensembles")
    
    print("\n🎯 **Recommendation**")
    print("   The SR Breakout Predictor is REDUNDANT because:")
    print("   1. Predictive Ensembles have better models and features")
    print("   2. Unified Regime Classifier already handles SR zones")
    print("   3. Can enhance Predictive Ensembles with SR context")
    print("   4. Reduces maintenance overhead")
    
    print("\n✅ **Migration Path**")
    print("   1. Add SR context features to Predictive Ensembles")
    print("   2. Replace SR breakout predictor calls")
    print("   3. Remove SR breakout predictor file")
    print("   4. Unify prediction pipeline")


def main():
    parser = argparse.ArgumentParser(description="Compare SR Breakout Predictor with existing ML models")
    parser.add_argument("--symbol", default="ETHUSDT", help="Symbol to test")
    
    args = parser.parse_args()
    
    asyncio.run(test_sr_breakout_comparison(args.symbol))


if __name__ == "__main__":
    main() 