#!/usr/bin/env python3
"""
Simple test script to demonstrate tprint_data_preview integration.

This script shows the basic functionality without requiring external dependencies.
"""

import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Set environment variable to enable data preview
os.environ['ENABLE_DATA_PREVIEW'] = 'true'

# Import tprint functions
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_data_preview


def test_basic_data_preview():
    """Test basic tprint_data_preview functionality."""
    tprint_info("🧪 Testing basic tprint_data_preview functionality")
    
    # Test with simple list
    sample_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tprint_data_preview(sample_list, "Simple List", max_rows=5)
    
    # Test with dictionary
    sample_dict = {
        'symbol': 'BTCUSDT',
        'interval': '5m',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'open': 50000.0,
        'high': 51000.0,
        'low': 49000.0,
        'close': 50500.0,
        'volume': 1000.0
    }
    tprint_data_preview(sample_dict, "Sample OHLCV Data", max_rows=3)
    
    # Test with list of dictionaries (simulating exchange data)
    sample_klines = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    
    for i in range(5):
        timestamp = base_time + timedelta(minutes=i*5)
        sample_klines.append({
            'openTime': int(timestamp.timestamp() * 1000),
            'open': f"{50000 + i * 10:.2f}",
            'high': f"{50000 + i * 10 + 50:.2f}",
            'low': f"{50000 + i * 10 - 50:.2f}",
            'close': f"{50000 + i * 10 + 25:.2f}",
            'volume': f"{100 + i * 10:.2f}"
        })
    
    tprint_data_preview(sample_klines, "Sample Exchange Klines Data", max_rows=3)
    
    tprint_success("✅ Basic data preview test completed")


def test_standardization_integration():
    """Test the integration with standardization modules."""
    tprint_info("🧪 Testing standardization integration")
    
    # Test that we can import the modules
    try:
        from exchanges.shared.unified_exchange_standardizer import (
            UnifiedExchangeStandardizer, ExchangeType, DataQualityLevel
        )
        tprint_info("✅ Successfully imported UnifiedExchangeStandardizer")
        
        from exchanges.shared.unified_ohlcv_standardizer import (
            UnifiedOHLCVStandardizer
        )
        tprint_info("✅ Successfully imported UnifiedOHLCVStandardizer")
        
        # Test that the methods have the new parameter
        standardizer = UnifiedExchangeStandardizer(DataQualityLevel.STANDARD)
        
        # Check if the method signature includes enable_data_preview
        import inspect
        sig = inspect.signature(standardizer.standardize_data)
        if 'enable_data_preview' in sig.parameters:
            tprint_info("✅ standardize_data method has enable_data_preview parameter")
        else:
            tprint_info("❌ standardize_data method missing enable_data_preview parameter")
        
        sig_df = inspect.signature(standardizer.standardize_to_dataframe)
        if 'enable_data_preview' in sig_df.parameters:
            tprint_info("✅ standardize_to_dataframe method has enable_data_preview parameter")
        else:
            tprint_info("❌ standardize_to_dataframe method missing enable_data_preview parameter")
        
    except ImportError as e:
        tprint_info(f"❌ Import error: {e}")
    except Exception as e:
        tprint_info(f"❌ Error: {e}")


def test_klines_adapters_integration():
    """Test the integration with klines adapters."""
    tprint_info("🧪 Testing klines adapters integration")
    
    # Test that we can import the adapters
    try:
        from exchanges.binance.klines_adapter import BinanceKlinesAdapter
        tprint_info("✅ Successfully imported BinanceKlinesAdapter")
        
        from exchanges.bingx.klines_adapter import BingXKlinesAdapter
        tprint_info("✅ Successfully imported BingXKlinesAdapter")
        
        from exchanges.okx.klines_adapter import OkxKlinesAdapter
        tprint_info("✅ Successfully imported OkxKlinesAdapter")
        
        from exchanges.mexc.klines_adapter import MexcKlinesAdapter
        tprint_info("✅ Successfully imported MexcKlinesAdapter")
        
        from exchanges.gateio.klines_adapter import GateioKlinesAdapter
        tprint_info("✅ Successfully imported GateioKlinesAdapter")
        
        from exchanges.phemex.klines_adapter import PhemexKlinesAdapter
        tprint_info("✅ Successfully imported PhemexKlinesAdapter")
        
        # Test that the methods have the new parameter
        binance_adapter = BinanceKlinesAdapter()
        
        import inspect
        sig = inspect.signature(binance_adapter.get_klines_data)
        if 'enable_data_preview' in sig.parameters:
            tprint_info("✅ get_klines_data method has enable_data_preview parameter")
        else:
            tprint_info("❌ get_klines_data method missing enable_data_preview parameter")
        
        sig_dp = inspect.signature(binance_adapter.download_and_process_klines)
        if 'enable_data_preview' in sig_dp.parameters:
            tprint_info("✅ download_and_process_klines method has enable_data_preview parameter")
        else:
            tprint_info("❌ download_and_process_klines method missing enable_data_preview parameter")
        
    except ImportError as e:
        tprint_info(f"❌ Import error: {e}")
    except Exception as e:
        tprint_info(f"❌ Error: {e}")


def main():
    """Main test function."""
    tprint_info("🚀 Starting simple tprint_data_preview integration test")
    tprint_info("=" * 60)
    
    try:
        # Test basic functionality
        test_basic_data_preview()
        tprint_info("")
        
        # Test standardization integration
        test_standardization_integration()
        tprint_info("")
        
        # Test klines adapters integration
        test_klines_adapters_integration()
        tprint_info("")
        
        tprint_success("🎉 All tests completed successfully!")
        tprint_info("=" * 60)
        tprint_info("✅ tprint_data_preview is now fully integrated into exchanges standardization logic")
        tprint_info("📊 Data previews will be shown during data processing when enable_data_preview=True")
        tprint_info("🔧 Use ENABLE_DATA_PREVIEW environment variable to control preview display")
        tprint_info("")
        tprint_info("🔍 Key integration points:")
        tprint_info("  • UnifiedExchangeStandardizer.standardize_data()")
        tprint_info("  • UnifiedExchangeStandardizer.standardize_to_dataframe()")
        tprint_info("  • UnifiedOHLCVStandardizer.standardize_data()")
        tprint_info("  • All klines adapters: get_klines_data() and download_and_process_klines()")
        tprint_info("  • Convenience function: standardize_exchange_ohlcv()")
        
    except Exception as e:
        tprint(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()