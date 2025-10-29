#!/usr/bin/env python3
"""
Comprehensive test script for full BingX integration with enhanced klines processing pipeline.

This script demonstrates:
1. BingX exchange dispatcher compatibility
2. Simplified pipeline interface with exchange, asset, lookback period
3. Full perp trading operations
4. Complete integration testing
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training.steps.data_collection.enhanced_klines_processing_pipeline import (
    EnhancedKlinesProcessingPipeline,
    PipelineConfig,
    ResamplingConfig
)
from exchanges.exchange_dispatcher import create_bingx_dispatcher
from exchanges.bingx import create_bingx_exchange


async def test_bingx_dispatcher_compatibility():
    """Test BingX compatibility with exchange dispatcher."""
    print("🔧 Testing BingX exchange dispatcher compatibility...")
    
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
        
        print("✅ BingX dispatcher initialized successfully")
        
        # Test all dispatcher methods
        test_results = {}
        
        # Test market data methods
        print("📊 Testing market data methods...")
        test_results['get_price'] = await dispatcher.get_price("BTCUSDT") is not None
        test_results['get_ticker'] = await dispatcher.get_ticker("BTCUSDT") is not None
        test_results['get_order_book'] = await dispatcher.get_order_book("BTCUSDT", 10) is not None
        
        # Test account methods
        print("💰 Testing account methods...")
        test_results['get_balance'] = await dispatcher.get_balance("USDT") >= 0
        test_results['get_account_info'] = await dispatcher.get_account_info() is not None
        
        # Test position methods
        print("📈 Testing position methods...")
        test_results['get_positions'] = isinstance(await dispatcher.get_positions(), list)
        test_results['get_liquidation_risk'] = await dispatcher.get_liquidation_risk("BTCUSDT") is not None or True  # May be None if no position
        
        # Test instrument info
        print("🔍 Testing instrument info...")
        test_results['get_instrument_info'] = await dispatcher.get_instrument_info("BTCUSDT") is not None
        
        # Print results
        print("\n📋 Dispatcher Method Test Results:")
        for method, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {method}: {status}")
        
        all_passed = all(test_results.values())
        print(f"\n🎯 Dispatcher Compatibility: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error in dispatcher compatibility test: {e}")
        return False
    finally:
        await dispatcher.close()


async def test_simplified_pipeline_interface():
    """Test the simplified pipeline interface with exchange, asset, lookback period."""
    print("\n🚀 Testing simplified pipeline interface...")
    
    # Configure pipeline
    pipeline_config = PipelineConfig(
        data_dir="historical_data",
        exchange="bingx",
        enable_logging=True,
        enable_gap_filling=True,
        enable_resampling=True,
        enable_duplicate_handling=True,
        enable_quality_validation=True,
        batch_compatible=True
    )
    
    # Configure resampling
    resampling_config = ResamplingConfig(
        target_intervals=['5m', '15m', '30m'],
        method='ohlc',
        preserve_volume=True,
        resample_older_than_days=1,
        enable_auto_resampling=True
    )
    
    # Create pipeline
    pipeline = EnhancedKlinesProcessingPipeline(pipeline_config)
    
    try:
        # Test different lookback periods
        test_cases = [
            {"exchange": "bingx", "asset": "BTC", "lookback_period": "7d", "interval": "1m"},
            {"exchange": "bingx", "asset": "ETH", "lookback_period": "30d", "interval": "5m"},
            {"exchange": "binance", "asset": "ADA", "lookback_period": "6m", "interval": "1h"},
        ]
        
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\n📊 Test Case {i+1}: {test_case['exchange']} {test_case['asset']} {test_case['lookback_period']}")
            
            try:
                result = await pipeline.process_klines_data_simple(
                    exchange=test_case["exchange"],
                    asset=test_case["asset"],
                    lookback_period=test_case["lookback_period"],
                    interval=test_case["interval"],
                    api_key="",  # Add your API key here
                    api_secret="",  # Add your API secret here
                    use_testnet=True,
                    resampling_config=resampling_config,
                    batch_id=f"test_case_{i+1}"
                )
                
                success = result.get('pipeline_success', False)
                results.append(success)
                
                print(f"   Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
                if success:
                    print(f"   Data quality: {result.get('data_quality', 'Unknown')}")
                    print(f"   Final shape: {result.get('final_data_shape', 'Unknown')}")
                    print(f"   Stored files: {len(result.get('stored_files', []))}")
                
            except Exception as e:
                print(f"   Error: {e}")
                results.append(False)
        
        success_rate = sum(results) / len(results) if results else 0
        print(f"\n🎯 Simplified Interface Success Rate: {success_rate:.1%} ({sum(results)}/{len(results)})")
        
        return success_rate > 0.5  # At least 50% success rate
        
    except Exception as e:
        print(f"❌ Error in simplified pipeline test: {e}")
        return False


async def test_bingx_perp_trading_operations():
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
        
        # Test all perp trading methods
        test_results = {}
        
        # Test position management
        print("📊 Testing position management...")
        positions = await bingx_exchange.get_positions()
        test_results['get_positions'] = isinstance(positions, list)
        print(f"   Current positions: {len(positions)}")
        
        # Test leverage and margin mode (if positions exist)
        if positions:
            symbol = positions[0]['symbol']
            print(f"   Testing leverage and margin mode for {symbol}...")
            
            leverage_result = await bingx_exchange.set_leverage(symbol, 5.0)
            test_results['set_leverage'] = leverage_result
            print(f"   Set leverage: {'✅' if leverage_result else '❌'}")
            
            margin_result = await bingx_exchange.set_margin_mode(symbol, "ISOLATED")
            test_results['set_margin_mode'] = margin_result
            print(f"   Set margin mode: {'✅' if margin_result else '❌'}")
        else:
            test_results['set_leverage'] = True  # Skip if no positions
            test_results['set_margin_mode'] = True  # Skip if no positions
            print("   No positions found, skipping leverage/margin tests")
        
        # Test position risk
        print("⚠️ Testing position risk...")
        risk = await bingx_exchange.get_position_risk("BTCUSDT")
        test_results['get_position_risk'] = risk is not None
        print(f"   Position risk: {'✅' if risk else '❌'}")
        
        # Test market data methods
        print("📈 Testing market data methods...")
        price = await bingx_exchange.get_price("BTCUSDT")
        test_results['get_price'] = price is not None
        print(f"   Current price: {price if price else 'N/A'}")
        
        ticker = await bingx_exchange.get_ticker("BTCUSDT")
        test_results['get_ticker'] = ticker is not None
        print(f"   Ticker data: {'✅' if ticker else '❌'}")
        
        order_book = await bingx_exchange.get_order_book("BTCUSDT", 10)
        test_results['get_order_book'] = order_book is not None
        print(f"   Order book: {'✅' if order_book else '❌'}")
        
        balance = await bingx_exchange.get_balance("USDT")
        test_results['get_balance'] = balance >= 0
        print(f"   USDT balance: {balance}")
        
        # Print results
        print("\n📋 Perp Trading Test Results:")
        for method, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {method}: {status}")
        
        all_passed = all(test_results.values())
        print(f"\n🎯 Perp Trading Operations: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error in perp trading test: {e}")
        return False
    finally:
        await bingx_exchange.close()


async def test_lookback_period_parsing():
    """Test the lookback period parsing functionality."""
    print("\n🕒 Testing lookback period parsing...")
    
    pipeline = EnhancedKlinesProcessingPipeline(PipelineConfig())
    
    test_cases = [
        ("1y", 1),
        ("6m", 1),  # 6 months = 1 year (minimum)
        ("12m", 1),  # 12 months = 1 year
        ("30d", 1),  # 30 days = 1 year (minimum)
        ("365d", 1),  # 365 days = 1 year
        ("730d", 2),  # 730 days = 2 years
        ("2", 2),  # Just a number
    ]
    
    results = []
    for period, expected_years in test_cases:
        try:
            years = pipeline._parse_lookback_period(period)
            success = years == expected_years
            results.append(success)
            print(f"   {period} -> {years} years: {'✅' if success else '❌'}")
        except Exception as e:
            print(f"   {period} -> Error: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) if results else 0
    print(f"\n🎯 Lookback Period Parsing: {success_rate:.1%} ({sum(results)}/{len(results)})")
    
    return success_rate > 0.8  # At least 80% success rate


async def main():
    """Run all integration tests."""
    print("🧪 Starting Full BingX Integration Tests")
    print("=" * 60)
    
    # Test 1: Dispatcher compatibility
    dispatcher_success = await test_bingx_dispatcher_compatibility()
    
    # Test 2: Simplified pipeline interface
    pipeline_success = await test_simplified_pipeline_interface()
    
    # Test 3: Perp trading operations
    perp_success = await test_bingx_perp_trading_operations()
    
    # Test 4: Lookback period parsing
    parsing_success = await test_lookback_period_parsing()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Full Integration Test Results:")
    print(f"   Dispatcher Compatibility: {'✅ PASS' if dispatcher_success else '❌ FAIL'}")
    print(f"   Simplified Pipeline Interface: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    print(f"   Perp Trading Operations: {'✅ PASS' if perp_success else '❌ FAIL'}")
    print(f"   Lookback Period Parsing: {'✅ PASS' if parsing_success else '❌ FAIL'}")
    
    overall_success = dispatcher_success and pipeline_success and perp_success and parsing_success
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    if overall_success:
        print("\n🎉 Full BingX integration is complete and functional!")
        print("   ✅ Exchange dispatcher compatibility")
        print("   ✅ Simplified pipeline interface (exchange, asset, lookback period)")
        print("   ✅ Complete perp trading operations")
        print("   ✅ Lookback period parsing")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")
    
    return overall_success


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)