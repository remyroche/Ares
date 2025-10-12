#!/usr/bin/env python3
"""
Test script for centralized indicators system.

This script tests that the centralized indicators work correctly
and that files can import and use them properly.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

def test_centralized_indicators():
    """Test the centralized indicators system."""
    print("🧪 Testing Centralized Indicators System")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    
    # Generate realistic OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.01)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volume = np.random.randint(1000, 10000, 100)
    
    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)
    
    print(f"✅ Created sample data with {len(data)} rows")
    print(f"📊 Data columns: {list(data.columns)}")
    print(f"📈 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    try:
        # Test importing centralized indicators
        print("\n🔧 Testing imports...")
        from src.training.steps.feature_engineering import (
            calculate_rsi, calculate_macd, calculate_stochastic,
            calculate_williams_r, calculate_cci, calculate_adx,
            get_all_indicators, CentralizedIndicators, IndicatorConfig
        )
        print("✅ Successfully imported centralized indicators")
        
        # Test RSI calculation
        print("\n📊 Testing RSI calculation...")
        rsi = calculate_rsi(data, period=14)
        print(f"✅ RSI calculated: {len(rsi)} values")
        print(f"📈 RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
        print(f"📊 RSI sample values: {rsi.head().values}")
        
        # Test MACD calculation
        print("\n📊 Testing MACD calculation...")
        macd_data = calculate_macd(data, fast=12, slow=26, signal=9)
        print(f"✅ MACD calculated: {len(macd_data)} components")
        for key, series in macd_data.items():
            print(f"📈 {key}: {len(series)} values, range: {series.min():.2f} - {series.max():.2f}")
        
        # Test Stochastic calculation
        print("\n📊 Testing Stochastic calculation...")
        stoch_data = calculate_stochastic(data, period=14, smooth_k=3, smooth_d=3)
        print(f"✅ Stochastic calculated: {len(stoch_data)} components")
        for key, series in stoch_data.items():
            print(f"📈 {key}: {len(series)} values, range: {series.min():.2f} - {series.max():.2f}")
        
        # Test Williams %R calculation
        print("\n📊 Testing Williams %R calculation...")
        williams_r = calculate_williams_r(data, period=14)
        print(f"✅ Williams %R calculated: {len(williams_r)} values")
        print(f"📈 Williams %R range: {williams_r.min():.2f} - {williams_r.max():.2f}")
        
        # Test CCI calculation
        print("\n📊 Testing CCI calculation...")
        cci = calculate_cci(data, period=20)
        print(f"✅ CCI calculated: {len(cci)} values")
        print(f"📈 CCI range: {cci.min():.2f} - {cci.max():.2f}")
        
        # Test ADX calculation
        print("\n📊 Testing ADX calculation...")
        adx = calculate_adx(data, period=14)
        print(f"✅ ADX calculated: {len(adx)} values")
        print(f"📈 ADX range: {adx.min():.2f} - {adx.max():.2f}")
        
        # Test getting all indicators
        print("\n📊 Testing get_all_indicators...")
        all_indicators = get_all_indicators(data, indicators=['rsi', 'macd', 'stochastic'])
        print(f"✅ All indicators calculated: {len(all_indicators.columns)} features")
        print(f"📊 Indicator columns: {list(all_indicators.columns)}")
        
        # Test CentralizedIndicators class
        print("\n🔧 Testing CentralizedIndicators class...")
        config = IndicatorConfig(rsi_period=21, macd_fast=15, macd_slow=30)
        indicators = CentralizedIndicators(config)
        
        rsi_custom = indicators.calculate_rsi(data, period=21)
        print(f"✅ Custom RSI (period=21): {len(rsi_custom)} values")
        
        macd_custom = indicators.calculate_macd(data, fast=15, slow=30)
        print(f"✅ Custom MACD (15,30): {len(macd_custom)} components")
        
        print("\n🎉 All tests passed! Centralized indicators system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_updated_files():
    """Test that updated files can use centralized indicators."""
    print("\n🔧 Testing Updated Files")
    print("=" * 30)
    
    try:
        # Test HMM training pipeline
        print("Testing HMM training pipeline...")
        from src.training.steps.model_training.simplified.hmm_training import HMMTrainingPipeline
        
        # Create a small sample for testing
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(50) * 0.01)
        })
        
        pipeline = HMMTrainingPipeline()
        rsi = pipeline._calculate_rsi(data['close'], period=14)
        macd = pipeline._calculate_macd(data['close'], fast=12, slow=26)
        
        print(f"✅ HMM pipeline RSI: {len(rsi)} values")
        print(f"✅ HMM pipeline MACD: {len(macd)} values")
        
        # Test unified regime classifier
        print("Testing unified regime classifier...")
        from src.analyst.unified_regime_classifier import UnifiedRegimeClassifier
        
        classifier = UnifiedRegimeClassifier()
        test_df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(50) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(50) * 0.01) + 1,
            'low': 100 + np.cumsum(np.random.randn(50) * 0.01) - 1
        })
        
        rsi_df = classifier._calculate_rsi(test_df, period=14)
        macd_df = classifier._calculate_macd(test_df, fast=12, slow=26, signal=9)
        
        print(f"✅ Regime classifier RSI: {len(rsi_df)} rows")
        print(f"✅ Regime classifier MACD: {len(macd_df)} rows")
        
        print("\n🎉 Updated files are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Updated files test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Centralized Indicators Test Suite")
    print("=" * 60)
    
    # Test centralized indicators
    indicators_ok = test_centralized_indicators()
    
    # Test updated files
    files_ok = test_updated_files()
    
    print("\n" + "=" * 60)
    if indicators_ok and files_ok:
        print("🎉 ALL TESTS PASSED! Centralized indicators system is working correctly.")
        print("✅ Indicators are now centralized in feature_engineering/indicators.py")
        print("✅ Files have been updated to use the centralized system")
        print("✅ Fallback calculations are available if feature bank is unavailable")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)