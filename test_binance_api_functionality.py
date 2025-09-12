#!/usr/bin/env python3
"""
Binance API Functionality Test

This script tests the Binance API implementation to ensure it's fully functional
for data collection and trading operations.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.exchange.binance import BinanceExchange
from src.utils.logger import system_logger

logger = system_logger.getChild('BinanceAPITest')

class BinanceAPITester:
    """Comprehensive Binance API functionality tester."""
    
    def __init__(self):
        self.logger = logger.getChild('BinanceAPITester')
        self.exchange = None
        self.test_results = {
            'connection': False,
            'server_time': False,
            'klines': False,
            'ticker': False,
            'order_book': False,
            'agg_trades': False,
            'futures_funding': False,
            'account_info': False,
            'position_risk': False
        }
    
    async def run_all_tests(self):
        """Run all Binance API tests."""
        self.logger.info("🚀 Starting Binance API functionality tests...")
        
        # Test configuration
        config = {
            'binance_exchange': {
                'use_testnet': True,  # Use testnet for safety
                'timeout': 30,
                'max_retries': 3,
                'rate_limit_enabled': True,
                'rate_limit_requests': 1200,
                'rate_limit_window': 60
            }
        }
        
        # Initialize exchange
        self.exchange = BinanceExchange(config)
        
        try:
            # Test 1: Connection and initialization
            await self.test_connection()
            
            # Test 2: Server time
            await self.test_server_time()
            
            # Test 3: Public endpoints
            await self.test_public_endpoints()
            
            # Test 4: Historical data
            await self.test_historical_data()
            
            # Test 5: Account endpoints (if credentials available)
            await self.test_account_endpoints()
            
            # Print results
            self.print_test_results()
            
        except Exception as e:
            self.logger.error(f"❌ Test suite failed: {e}")
        finally:
            if self.exchange:
                await self.exchange.stop()
    
    async def test_connection(self):
        """Test exchange connection and initialization."""
        self.logger.info("🔌 Testing connection...")
        try:
            success = await self.exchange.initialize()
            self.test_results['connection'] = success
            if success:
                self.logger.info("✅ Connection test passed")
            else:
                self.logger.error("❌ Connection test failed")
        except Exception as e:
            self.logger.error(f"❌ Connection test error: {e}")
    
    async def test_server_time(self):
        """Test server time endpoint."""
        self.logger.info("⏰ Testing server time...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping server time test")
                return
            
            server_time = await self.exchange._get_server_time()
            if server_time:
                self.test_results['server_time'] = True
                self.logger.info(f"✅ Server time test passed: {server_time}")
            else:
                self.logger.error("❌ Server time test failed")
        except Exception as e:
            self.logger.error(f"❌ Server time test error: {e}")
    
    async def test_public_endpoints(self):
        """Test public API endpoints."""
        self.logger.info("📊 Testing public endpoints...")
        
        # Test klines
        await self.test_klines()
        
        # Test ticker
        await self.test_ticker()
        
        # Test order book
        await self.test_order_book()
    
    async def test_klines(self):
        """Test klines endpoint."""
        self.logger.info("📈 Testing klines endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping klines test")
                return
            
            klines = await self.exchange.get_klines('BTCUSDT', '1m', 10)
            if klines and len(klines) > 0:
                self.test_results['klines'] = True
                self.logger.info(f"✅ Klines test passed: {len(klines)} records")
            else:
                self.logger.error("❌ Klines test failed")
        except Exception as e:
            self.logger.error(f"❌ Klines test error: {e}")
    
    async def test_ticker(self):
        """Test ticker endpoint."""
        self.logger.info("💰 Testing ticker endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping ticker test")
                return
            
            ticker = await self.exchange.get_ticker('BTCUSDT')
            if ticker and 'symbol' in ticker:
                self.test_results['ticker'] = True
                self.logger.info(f"✅ Ticker test passed: {ticker['symbol']}")
            else:
                self.logger.error("❌ Ticker test failed")
        except Exception as e:
            self.logger.error(f"❌ Ticker test error: {e}")
    
    async def test_order_book(self):
        """Test order book endpoint."""
        self.logger.info("📚 Testing order book endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping order book test")
                return
            
            order_book = await self.exchange.get_order_book('BTCUSDT', 10)
            if order_book and 'bids' in order_book and 'asks' in order_book:
                self.test_results['order_book'] = True
                self.logger.info(f"✅ Order book test passed: {len(order_book['bids'])} bids, {len(order_book['asks'])} asks")
            else:
                self.logger.error("❌ Order book test failed")
        except Exception as e:
            self.logger.error(f"❌ Order book test error: {e}")
    
    async def test_historical_data(self):
        """Test historical data endpoints."""
        self.logger.info("📊 Testing historical data endpoints...")
        
        # Test aggregate trades
        await self.test_agg_trades()
        
        # Test futures funding rates
        await self.test_futures_funding()
    
    async def test_agg_trades(self):
        """Test aggregate trades endpoint."""
        self.logger.info("🔄 Testing aggregate trades endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping agg trades test")
                return
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=1)
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)
            
            agg_trades = await self.exchange.get_aggregate_trades('BTCUSDT', start_time_ms, end_time_ms)
            if agg_trades is not None:
                self.test_results['agg_trades'] = True
                self.logger.info(f"✅ Aggregate trades test passed: {len(agg_trades)} records")
            else:
                self.logger.error("❌ Aggregate trades test failed")
        except Exception as e:
            self.logger.error(f"❌ Aggregate trades test error: {e}")
    
    async def test_futures_funding(self):
        """Test futures funding rates endpoint."""
        self.logger.info("🏦 Testing futures funding rates endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping futures funding test")
                return
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            start_time_ms = int(start_time.timestamp() * 1000)
            end_time_ms = int(end_time.timestamp() * 1000)
            
            funding_rates = await self.exchange.futures_funding_rate('BTCUSDT', start_time_ms, end_time_ms)
            if funding_rates is not None:
                self.test_results['futures_funding'] = True
                self.logger.info(f"✅ Futures funding test passed: {len(funding_rates)} records")
            else:
                self.logger.error("❌ Futures funding test failed")
        except Exception as e:
            self.logger.error(f"❌ Futures funding test error: {e}")
    
    async def test_account_endpoints(self):
        """Test account-related endpoints (requires API credentials)."""
        self.logger.info("👤 Testing account endpoints...")
        
        # Check if API credentials are available
        if not self.exchange.api_key or not self.exchange.api_secret:
            self.logger.warning("⚠️ No API credentials available, skipping account tests")
            return
        
        # Test account info
        await self.test_account_info()
        
        # Test position risk
        await self.test_position_risk()
    
    async def test_account_info(self):
        """Test account info endpoint."""
        self.logger.info("👤 Testing account info endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping account info test")
                return
            
            account_info = await self.exchange.get_account_info()
            if account_info and 'accountType' in account_info:
                self.test_results['account_info'] = True
                self.logger.info(f"✅ Account info test passed: {account_info['accountType']}")
            else:
                self.logger.error("❌ Account info test failed")
        except Exception as e:
            self.logger.error(f"❌ Account info test error: {e}")
    
    async def test_position_risk(self):
        """Test position risk endpoint."""
        self.logger.info("⚠️ Testing position risk endpoint...")
        try:
            if not self.exchange.is_connected:
                self.logger.warning("⚠️ Exchange not connected, skipping position risk test")
                return
            
            position_risk = await self.exchange.get_position_risk()
            if position_risk is not None:
                self.test_results['position_risk'] = True
                self.logger.info(f"✅ Position risk test passed: {len(position_risk)} positions")
            else:
                self.logger.error("❌ Position risk test failed")
        except Exception as e:
            self.logger.error(f"❌ Position risk test error: {e}")
    
    def print_test_results(self):
        """Print comprehensive test results."""
        self.logger.info("\n" + "="*60)
        self.logger.info("📊 BINANCE API FUNCTIONALITY TEST RESULTS")
        self.logger.info("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        success_rate = (passed_tests / total_tests) * 100
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            self.logger.info(f"{test_name.upper().replace('_', ' '):<25} {status}")
        
        self.logger.info("-"*60)
        self.logger.info(f"TOTAL TESTS: {total_tests}")
        self.logger.info(f"PASSED: {passed_tests}")
        self.logger.info(f"FAILED: {total_tests - passed_tests}")
        self.logger.info(f"SUCCESS RATE: {success_rate:.1f}%")
        
        if success_rate >= 80:
            self.logger.info("🎉 BINANCE API IS FULLY FUNCTIONAL!")
        elif success_rate >= 60:
            self.logger.info("⚠️ BINANCE API IS PARTIALLY FUNCTIONAL")
        else:
            self.logger.info("❌ BINANCE API HAS SIGNIFICANT ISSUES")
        
        self.logger.info("="*60)

async def main():
    """Main test function."""
    tester = BinanceAPITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())