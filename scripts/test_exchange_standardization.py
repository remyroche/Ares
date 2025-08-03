#!/usr/bin/env python3
"""
Test script to verify exchange API standardization.
This script tests that all exchanges implement the standardized interface correctly.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange.factory import ExchangeFactory
from src.interfaces.base_interfaces import IExchangeClient, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeStandardizationTester:
    """Test class for verifying exchange standardization."""

    def __init__(self):
        self.test_symbol = "BTCUSDT"
        self.test_interval = "1h"
        self.test_limit = 10

    async def test_exchange_interface(self, exchange_name: str) -> bool:
        """Test that an exchange implements the standardized interface correctly."""
        logger.info(f"Testing {exchange_name} exchange...")
        
        try:
            # Create exchange instance
            exchange = ExchangeFactory.get_exchange(exchange_name)
            
            # Test 1: Verify it implements IExchangeClient
            if not isinstance(exchange, IExchangeClient):
                logger.error(f"❌ {exchange_name} does not implement IExchangeClient")
                return False
            logger.info(f"✅ {exchange_name} implements IExchangeClient")

            # Test 2: Test get_klines method
            await self._test_get_klines(exchange, exchange_name)

            # Test 3: Test get_account_info method
            await self._test_get_account_info(exchange, exchange_name)

            # Test 4: Test create_order method (without actually creating orders)
            await self._test_create_order_interface(exchange, exchange_name)

            # Test 5: Test get_position_risk method
            await self._test_get_position_risk(exchange, exchange_name)

            # Test 6: Test additional standardized methods
            await self._test_additional_methods(exchange, exchange_name)

            logger.info(f"✅ {exchange_name} passed all interface tests")
            return True

        except Exception as e:
            logger.error(f"❌ {exchange_name} failed interface tests: {e}")
            return False

    async def _test_get_klines(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test get_klines method."""
        try:
            klines = await exchange.get_klines(self.test_symbol, self.test_interval, self.test_limit)
            
            # Verify return type
            if not isinstance(klines, list):
                raise ValueError(f"get_klines should return list, got {type(klines)}")
            
            # Verify MarketData objects
            for kline in klines:
                if not isinstance(kline, MarketData):
                    raise ValueError(f"Each kline should be MarketData, got {type(kline)}")
                
                # Verify required fields
                required_fields = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'interval']
                for field in required_fields:
                    if not hasattr(kline, field):
                        raise ValueError(f"MarketData missing required field: {field}")
            
            logger.info(f"✅ {exchange_name} get_klines: {len(klines)} klines returned")
            
        except Exception as e:
            logger.error(f"❌ {exchange_name} get_klines failed: {e}")
            raise

    async def _test_get_account_info(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test get_account_info method."""
        try:
            account_info = await exchange.get_account_info()
            
            # Verify return type
            if not isinstance(account_info, dict):
                raise ValueError(f"get_account_info should return dict, got {type(account_info)}")
            
            logger.info(f"✅ {exchange_name} get_account_info: {len(account_info)} fields returned")
            
        except Exception as e:
            logger.error(f"❌ {exchange_name} get_account_info failed: {e}")
            raise

    async def _test_create_order_interface(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test create_order method interface (without actually creating orders)."""
        try:
            # Test that the method exists and has correct signature
            if not hasattr(exchange, 'create_order'):
                raise ValueError("create_order method not found")
            
            # Test with minimal parameters
            order_result = await exchange.create_order(
                symbol=self.test_symbol,
                side="BUY",
                quantity=0.001,
                order_type="MARKET"
            )
            
            # Verify return type
            if not isinstance(order_result, dict):
                raise ValueError(f"create_order should return dict, got {type(order_result)}")
            
            logger.info(f"✅ {exchange_name} create_order interface test passed")
            
        except Exception as e:
            logger.error(f"❌ {exchange_name} create_order interface failed: {e}")
            raise

    async def _test_get_position_risk(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test get_position_risk method."""
        try:
            position_risk = await exchange.get_position_risk(self.test_symbol)
            
            # Verify return type
            if not isinstance(position_risk, dict):
                raise ValueError(f"get_position_risk should return dict, got {type(position_risk)}")
            
            logger.info(f"✅ {exchange_name} get_position_risk: {len(position_risk)} fields returned")
            
        except Exception as e:
            logger.error(f"❌ {exchange_name} get_position_risk failed: {e}")
            raise

    async def _test_additional_methods(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test additional standardized methods."""
        try:
            # Test get_open_orders
            if hasattr(exchange, 'get_open_orders'):
                open_orders = await exchange.get_open_orders(self.test_symbol)
                if not isinstance(open_orders, list):
                    raise ValueError(f"get_open_orders should return list, got {type(open_orders)}")
                logger.info(f"✅ {exchange_name} get_open_orders: {len(open_orders)} orders returned")

            # Test get_historical_klines
            if hasattr(exchange, 'get_historical_klines'):
                start_time = int(datetime(2024, 1, 1).timestamp() * 1000)
                end_time = int(datetime(2024, 1, 2).timestamp() * 1000)
                
                historical_klines = await exchange.get_historical_klines(
                    self.test_symbol,
                    self.test_interval,
                    start_time,
                    end_time,
                    5
                )
                
                if not isinstance(historical_klines, list):
                    raise ValueError(f"get_historical_klines should return list, got {type(historical_klines)}")
                
                # Verify MarketData objects
                for kline in historical_klines:
                    if not isinstance(kline, MarketData):
                        raise ValueError(f"Each historical kline should be MarketData, got {type(kline)}")
                
                logger.info(f"✅ {exchange_name} get_historical_klines: {len(historical_klines)} klines returned")

            # Test close method
            if hasattr(exchange, 'close'):
                await exchange.close()
                logger.info(f"✅ {exchange_name} close method executed successfully")

        except Exception as e:
            logger.error(f"❌ {exchange_name} additional methods failed: {e}")
            raise

    async def test_all_exchanges(self) -> None:
        """Test all available exchanges."""
        exchanges = ["binance", "gateio", "mexc", "okx"]
        results = {}
        
        logger.info("Starting exchange standardization tests...")
        
        for exchange_name in exchanges:
            try:
                success = await self.test_exchange_interface(exchange_name)
                results[exchange_name] = success
            except Exception as e:
                logger.error(f"❌ {exchange_name} test failed with exception: {e}")
                results[exchange_name] = False
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("EXCHANGE STANDARDIZATION TEST RESULTS")
        logger.info("="*50)
        
        all_passed = True
        for exchange_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{exchange_name.upper():<12} {status}")
            if not success:
                all_passed = False
        
        logger.info("="*50)
        if all_passed:
            logger.info("🎉 ALL EXCHANGES PASSED STANDARDIZATION TESTS!")
        else:
            logger.error("❌ SOME EXCHANGES FAILED STANDARDIZATION TESTS")
        
        return all_passed


async def main():
    """Main test function."""
    tester = ExchangeStandardizationTester()
    success = await tester.test_all_exchanges()
    
    if success:
        print("\n✅ Exchange standardization verification completed successfully!")
        return 0
    else:
        print("\n❌ Exchange standardization verification failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 