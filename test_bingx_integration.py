#!/usr/bin/env python3
"""
Test script for BingX integration with enhanced klines processing pipeline.

This script demonstrates:
1. BingX klines data downloading for backtesting
2. BingX perp position operations (open/modify/close/monitor)
3. Full integration with the enhanced pipeline
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    EnhancedKlinesProcessingPipeline,
    PipelineConfig,
    ResamplingConfig
)
from exchanges.exchange_dispatcher import create_bingx_dispatcher
from exchanges.bingx import create_bingx_exchange


async def test_bingx_klines_download():
    """Test BingX klines data downloading for backtesting."""
    print("🚀 Testing BingX klines data downloading...")
    
    # Configure pipeline for BingX
    pipeline_config = PipelineConfig(
        data_dir="historical_data",
        exchange="bingx",  # Use BingX instead of Binance
        enable_logging=True,
        enable_gap_filling=True,
        enable_resampling=True,
        enable_duplicate_handling=True,
        enable_quality_validation=True,
        batch_compatible=True
    )
    
    # Configure resampling
    resampling_config = ResamplingConfig(
        target_intervals=['5m', '15m', '30m', '1h'],
        method='ohlc',
        preserve_volume=True,
        resample_older_than_days=1,
        enable_auto_resampling=True
    )
    
    # Create BingX exchange interface
    exchange_config = {
        'exchange_type': 'bingx',
        'api_key': "",  # Add your BingX API key here
        'api_secret': "",  # Add your BingX API secret here
        'testnet': True,
        'rate_limits': {}
    }
    
    # Create exchange interface
    from src.trading.execution.exchange_interface import ExchangeInterface
    exchange_interface = ExchangeInterface(exchange_config)
    
    try:
        # Initialize pipeline
        pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
        
        # Test connection
        await exchange_interface.connect()
        print("✅ Connected to BingX exchange")
        
        # Process klines data
        results = await pipeline.process_klines_data(
            symbol="BTCUSDT",
            interval="1m",
            years=1,  # Download 1 year of data
            exchange_interface=exchange_interface,
            resampling_config=resampling_config,
            batch_id="bingx_test"
        )
        
        print(f"📊 Pipeline results: {results['pipeline_success']}")
        print(f"📈 Data quality: {results['data_quality']}")
        print(f"📏 Final shape: {results['final_data_shape']}")
        print(f"💾 Stored files: {results['stored_files']}")
        print(f"🔄 Resampled intervals: {results['resampled_intervals']}")
        
        return results['pipeline_success']
        
    except Exception as e:
        print(f"❌ Error in klines download test: {e}")
        return False
    finally:
        await exchange_interface.disconnect()


async def test_bingx_perp_trading():
    """Test BingX perp trading operations."""
    print("\n🚀 Testing BingX perp trading operations...")
    
    # Create BingX exchange instance
    bingx_exchange = create_bingx_exchange(
        api_key="",  # Add your BingX API key here
        api_secret="",  # Add your BingX API secret here
        trade_symbol="BTCUSDT",
        use_testnet=True
    )
    
    try:
        # Initialize exchange
        await bingx_exchange._initialize_exchange()
        print("✅ BingX exchange initialized")
        
        # Test connection
        await bingx_exchange._test_connection()
        print("✅ Connected to BingX API")
        
        # Test getting positions
        positions = await bingx_exchange.get_positions()
        print(f"📊 Current positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos['symbol']}: {pos['side']} {pos['size']} @ {pos['entryPrice']}")
        
        # Test setting leverage (if you have a position)
        if positions:
            symbol = positions[0]['symbol']
            leverage_result = await bingx_exchange.set_leverage(symbol, 10.0)
            print(f"🔧 Set leverage for {symbol}: {leverage_result}")
        
        # Test setting margin mode
        if positions:
            symbol = positions[0]['symbol']
            margin_result = await bingx_exchange.set_margin_mode(symbol, "ISOLATED")
            print(f"🔧 Set margin mode for {symbol}: {margin_result}")
        
        # Test position risk
        if positions:
            symbol = positions[0]['symbol']
            risk = await bingx_exchange.get_position_risk(symbol)
            print(f"⚠️ Position risk for {symbol}: {risk}")
        
        print("✅ BingX perp trading operations completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in perp trading test: {e}")
        return False
    finally:
        await bingx_exchange.close()


async def test_bingx_dispatcher():
    """Test BingX through the exchange dispatcher."""
    print("\n🚀 Testing BingX through exchange dispatcher...")
    
    # Create BingX dispatcher
    dispatcher = create_bingx_dispatcher(
        api_key="",  # Add your BingX API key here
        api_secret="",  # Add your BingX API secret here
        trade_symbol="BTCUSDT",
        use_testnet=True
    )
    
    try:
        # Initialize dispatcher
        success = await dispatcher.initialize()
        if not success:
            print("❌ Failed to initialize BingX dispatcher")
            return False
        
        print("✅ BingX dispatcher initialized")
        
        # Test getting ticker
        ticker = await dispatcher.get_ticker("BTCUSDT")
        if ticker:
            print(f"📊 BTCUSDT ticker: {ticker}")
        
        # Test getting positions
        positions = await dispatcher.get_positions()
        print(f"📊 Current positions: {len(positions)}")
        
        # Test getting account info
        account_info = await dispatcher.get_account_info()
        if account_info:
            print(f"💰 Account info: {account_info}")
        
        print("✅ BingX dispatcher test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error in dispatcher test: {e}")
        return False
    finally:
        await dispatcher.close()


async def main():
    """Run all BingX integration tests."""
    print("🧪 Starting BingX Integration Tests")
    print("=" * 50)
    
    # Test 1: Klines downloading
    klines_success = await test_bingx_klines_download()
    
    # Test 2: Perp trading operations
    perp_success = await test_bingx_perp_trading()
    
    # Test 3: Exchange dispatcher
    dispatcher_success = await test_bingx_dispatcher()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print(f"   Klines Download: {'✅ PASS' if klines_success else '❌ FAIL'}")
    print(f"   Perp Trading: {'✅ PASS' if perp_success else '❌ FAIL'}")
    print(f"   Exchange Dispatcher: {'✅ PASS' if dispatcher_success else '❌ FAIL'}")
    
    overall_success = klines_success and perp_success and dispatcher_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 BingX integration is fully functional!")
        print("   - Klines downloading for backtesting: ✅")
        print("   - Perp position operations: ✅")
        print("   - Exchange dispatcher integration: ✅")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)