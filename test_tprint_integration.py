#!/usr/bin/env python3
"""
Test script to demonstrate tprint_data_preview integration in exchanges standardization logic.

This script shows how tprint_data_preview is now integrated into the exchange data
standardization process, providing helpful data previews during processing.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set environment variable to enable data preview
os.environ['ENABLE_DATA_PREVIEW'] = 'true'

from exchanges.shared.unified_exchange_standardizer import (
    UnifiedExchangeStandardizer, ExchangeType, DataQualityLevel
)
from exchanges.shared.unified_ohlcv_standardizer import (
    UnifiedOHLCVStandardizer
)
from exchanges.binance.klines_adapter import BinanceKlinesAdapter
from exchanges.bingx.klines_adapter import BingXKlinesAdapter
from exchanges.okx.klines_adapter import OkxKlinesAdapter
from exchanges.mexc.klines_adapter import MexcKlinesAdapter
from exchanges.gateio.klines_adapter import GateioKlinesAdapter
from exchanges.phemex.klines_adapter import PhemexKlinesAdapter
from src.utils.tprint import tprint, tprint_info, tprint_success


def create_sample_binance_data():
    """Create sample Binance-style klines data for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    sample_data = []
    for i in range(10):
        timestamp = base_time + timedelta(minutes=i*5)
        open_price = 50000 + i * 10 + np.random.normal(0, 50)
        high_price = open_price + np.random.uniform(10, 100)
        low_price = open_price - np.random.uniform(10, 100)
        close_price = open_price + np.random.normal(0, 30)
        volume = np.random.uniform(100, 1000)
        
        sample_data.append({
            'openTime': int(timestamp.timestamp() * 1000),
            'closeTime': int((timestamp + timedelta(minutes=5)).timestamp() * 1000),
            'open': f"{open_price:.2f}",
            'high': f"{high_price:.2f}",
            'low': f"{low_price:.2f}",
            'close': f"{close_price:.2f}",
            'volume': f"{volume:.2f}",
            'quoteVolume': f"{volume * close_price:.2f}",
            'trades': np.random.randint(50, 200),
            'takerBuyBase': f"{volume * 0.6:.2f}",
            'takerBuyQuote': f"{volume * close_price * 0.6:.2f}"
        })
    
    return sample_data


def create_sample_okx_data():
    """Create sample OKX-style klines data for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    sample_data = []
    for i in range(10):
        timestamp = base_time + timedelta(minutes=i*5)
        open_price = 50000 + i * 10 + np.random.normal(0, 50)
        high_price = open_price + np.random.uniform(10, 100)
        low_price = open_price - np.random.uniform(10, 100)
        close_price = open_price + np.random.normal(0, 30)
        volume = np.random.uniform(100, 1000)
        
        sample_data.append([
            str(int(timestamp.timestamp() * 1000)),
            f"{open_price:.2f}",
            f"{high_price:.2f}",
            f"{low_price:.2f}",
            f"{close_price:.2f}",
            f"{volume:.2f}",
            f"{volume * close_price:.2f}",
            str(np.random.randint(50, 200))
        ])
    
    return sample_data


def test_unified_exchange_standardizer():
    """Test UnifiedExchangeStandardizer with tprint_data_preview integration."""
    tprint_info("🧪 Testing UnifiedExchangeStandardizer with tprint_data_preview")
    
    # Create sample data
    binance_data = create_sample_binance_data()
    okx_data = create_sample_okx_data()
    
    # Initialize standardizer
    standardizer = UnifiedExchangeStandardizer(DataQualityLevel.STANDARD)
    
    # Test with Binance data
    tprint_info("📊 Testing with Binance data...")
    standardized_binance = standardizer.standardize_data(
        binance_data, 
        ExchangeType.BINANCE, 
        "BTCUSDT", 
        "5m",
        enable_data_preview=True
    )
    
    # Test with OKX data
    tprint_info("📊 Testing with OKX data...")
    standardized_okx = standardizer.standardize_data(
        okx_data, 
        ExchangeType.OKX, 
        "BTCUSDT", 
        "5m",
        enable_data_preview=True
    )
    
    # Test DataFrame conversion
    tprint_info("📊 Testing DataFrame conversion...")
    df_binance = standardizer.standardize_to_dataframe(
        binance_data, 
        ExchangeType.BINANCE, 
        "BTCUSDT", 
        "5m",
        enable_data_preview=True
    )
    
    tprint_success(f"✅ UnifiedExchangeStandardizer test completed - {len(standardized_binance)} Binance records, {len(standardized_okx)} OKX records")


def test_unified_ohlcv_standardizer():
    """Test UnifiedOHLCVStandardizer with tprint_data_preview integration."""
    tprint_info("🧪 Testing UnifiedOHLCVStandardizer with tprint_data_preview")
    
    # Create sample data
    binance_data = create_sample_binance_data()
    
    # Initialize standardizer
    standardizer = UnifiedOHLCVStandardizer(DataQualityLevel.STANDARD)
    
    # Test standardization
    tprint_info("📊 Testing OHLCV standardization...")
    standardized_data = standardizer.standardize_data(
        binance_data, 
        ExchangeType.BINANCE, 
        "BTCUSDT", 
        "5m",
        enable_data_preview=True
    )
    
    tprint_success(f"✅ UnifiedOHLCVStandardizer test completed - {len(standardized_data)} records")


def test_klines_adapters():
    """Test klines adapters with tprint_data_preview integration."""
    tprint_info("🧪 Testing Klines Adapters with tprint_data_preview")
    
    # Test Binance adapter
    tprint_info("📊 Testing Binance adapter...")
    binance_adapter = BinanceKlinesAdapter()
    
    # Test BingX adapter
    tprint_info("📊 Testing BingX adapter...")
    bingx_adapter = BingXKlinesAdapter()
    
    # Test OKX adapter
    tprint_info("📊 Testing OKX adapter...")
    okx_adapter = OkxKlinesAdapter()
    
    # Test MEXC adapter
    tprint_info("📊 Testing MEXC adapter...")
    mexc_adapter = MexcKlinesAdapter()
    
    # Test GateIO adapter
    tprint_info("📊 Testing GateIO adapter...")
    gateio_adapter = GateioKlinesAdapter()
    
    # Test Phemex adapter
    tprint_info("📊 Testing Phemex adapter...")
    phemex_adapter = PhemexKlinesAdapter()
    
    tprint_success("✅ All klines adapters initialized successfully")


def test_data_preview_features():
    """Test various data preview features."""
    tprint_info("🧪 Testing tprint_data_preview features")
    
    # Test with different data types
    sample_data = create_sample_binance_data()
    
    # Test with list of dicts
    tprint_info("📊 Testing with list of dictionaries...")
    from src.utils.tprint import tprint_data_preview
    tprint_data_preview(sample_data, "Sample Binance Data (List of Dicts)", max_rows=3)
    
    # Test with DataFrame
    tprint_info("📊 Testing with DataFrame...")
    df = pd.DataFrame(sample_data)
    tprint_data_preview(df, "Sample Binance Data (DataFrame)", max_rows=3)
    
    # Test with numpy array
    tprint_info("📊 Testing with numpy array...")
    arr = np.random.rand(5, 4)
    tprint_data_preview(arr, "Sample Numpy Array", max_rows=3)
    
    # Test with dictionary
    tprint_info("📊 Testing with dictionary...")
    sample_dict = {
        'symbol': 'BTCUSDT',
        'interval': '5m',
        'data_count': len(sample_data),
        'sample_record': sample_data[0]
    }
    tprint_data_preview(sample_dict, "Sample Dictionary", max_rows=3)
    
    tprint_success("✅ Data preview features test completed")


def main():
    """Main test function."""
    tprint_info("🚀 Starting tprint_data_preview integration test")
    tprint_info("=" * 60)
    
    try:
        # Test data preview features
        test_data_preview_features()
        tprint_info("")
        
        # Test unified standardizers
        test_unified_exchange_standardizer()
        tprint_info("")
        
        test_unified_ohlcv_standardizer()
        tprint_info("")
        
        # Test klines adapters
        test_klines_adapters()
        tprint_info("")
        
        tprint_success("🎉 All tests completed successfully!")
        tprint_info("=" * 60)
        tprint_info("✅ tprint_data_preview is now fully integrated into exchanges standardization logic")
        tprint_info("📊 Data previews will be shown during data processing when enable_data_preview=True")
        tprint_info("🔧 Use ENABLE_DATA_PREVIEW environment variable to control preview display")
        
    except Exception as e:
        tprint(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()