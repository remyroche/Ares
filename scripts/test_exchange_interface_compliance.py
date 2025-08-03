#!/usr/bin/env python3
"""
Test script to verify exchange interface compliance.
This script tests that all exchanges implement the standardized interface correctly
without requiring real API calls or credentials.
"""

import asyncio
import logging
import inspect
from typing import Any, get_type_hints

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange.factory import ExchangeFactory
from src.interfaces.base_interfaces import IExchangeClient, MarketData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeInterfaceComplianceTester:
    """Test class for verifying exchange interface compliance."""

    def __init__(self):
        self.exchanges = ["binance", "gateio", "mexc", "okx"]

    def test_exchange_interface_compliance(self, exchange_name: str) -> bool:
        """Test that an exchange implements the standardized interface correctly."""
        logger.info(f"Testing {exchange_name} exchange interface compliance...")
        
        try:
            # Create exchange instance
            exchange = ExchangeFactory.get_exchange(exchange_name)
            
            # Test 1: Verify it implements IExchangeClient
            if not isinstance(exchange, IExchangeClient):
                logger.error(f"❌ {exchange_name} does not implement IExchangeClient")
                return False
            logger.info(f"✅ {exchange_name} implements IExchangeClient")

            # Test 2: Verify required methods exist
            required_methods = [
                'get_klines',
                'get_account_info', 
                'create_order',
                'get_position_risk'
            ]
            
            for method_name in required_methods:
                if not hasattr(exchange, method_name):
                    logger.error(f"❌ {exchange_name} missing required method: {method_name}")
                    return False
                logger.info(f"✅ {exchange_name} has method: {method_name}")

            # Test 3: Verify method signatures
            self._test_method_signatures(exchange, exchange_name)

            # Test 4: Verify additional standardized methods
            additional_methods = [
                'get_historical_klines',
                'get_historical_agg_trades',
                'get_open_orders',
                'cancel_order',
                'get_order_status',
                'close'
            ]
            
            for method_name in additional_methods:
                if hasattr(exchange, method_name):
                    logger.info(f"✅ {exchange_name} has additional method: {method_name}")
                else:
                    logger.warning(f"⚠️ {exchange_name} missing optional method: {method_name}")

            # Test 5: Verify abstract methods are implemented
            self._test_abstract_methods(exchange, exchange_name)

            logger.info(f"✅ {exchange_name} passed all interface compliance tests")
            return True

        except Exception as e:
            logger.error(f"❌ {exchange_name} failed interface compliance tests: {e}")
            return False

    def _test_method_signatures(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test that method signatures match the interface."""
        try:
            # Define expected method signatures for all exchanges
            expected_signatures = {
                'get_klines': {
                    'required_params': ['symbol', 'interval', 'limit'],
                    'return_type': list[MarketData],
                    'description': 'Get historical kline data'
                },
                'get_account_info': {
                    'required_params': [],
                    'return_type': dict[str, Any],
                    'description': 'Get account information'
                },
                'create_order': {
                    'required_params': ['symbol', 'side', 'quantity', 'price', 'order_type'],
                    'return_type': dict[str, Any],
                    'description': 'Create a trading order'
                },
                'get_position_risk': {
                    'required_params': ['symbol'],
                    'return_type': dict[str, Any],
                    'description': 'Get position risk information'
                },
                'get_historical_klines': {
                    'required_params': ['symbol', 'interval', 'start_time_ms', 'end_time_ms', 'limit'],
                    'return_type': list[MarketData],
                    'description': 'Get historical kline data for time range'
                },
                'get_historical_agg_trades': {
                    'required_params': ['symbol', 'start_time_ms', 'end_time_ms', 'limit'],
                    'return_type': list[dict[str, Any]],
                    'description': 'Get historical aggregated trades'
                },
                'get_open_orders': {
                    'required_params': ['symbol'],
                    'return_type': list[dict[str, Any]],
                    'description': 'Get open orders'
                },
                'cancel_order': {
                    'required_params': ['symbol', 'order_id'],
                    'return_type': dict[str, Any],
                    'description': 'Cancel an order'
                },
                'get_order_status': {
                    'required_params': ['symbol', 'order_id'],
                    'return_type': dict[str, Any],
                    'description': 'Get order status'
                }
            }

            # Test each method signature
            for method_name, expected in expected_signatures.items():
                if hasattr(exchange, method_name):
                    method = getattr(exchange, method_name)
                    sig = inspect.signature(method)
                    params = list(sig.parameters.keys())
                    
                    logger.info(f"🔍 {exchange_name} {method_name} actual signature: {params}")
                    
                    # Check that required parameters exist
                    required_params = expected['required_params']
                    if not all(param in params for param in required_params):
                        logger.error(f"❌ {exchange_name} {method_name} missing required parameters: {required_params}")
                        raise ValueError(f"Missing required parameters {required_params}, got {params}")
                    
                    # Check return type hint
                    hints = get_type_hints(method)
                    expected_return = expected['return_type']
                    if 'return' in hints and hints['return'] != expected_return:
                        logger.warning(f"⚠️ {exchange_name} {method_name} return type hint: {hints['return']} (expected {expected_return})")
                    
                    logger.info(f"✅ {exchange_name} {method_name} signature is correct")
                else:
                    logger.warning(f"⚠️ {exchange_name} missing method: {method_name}")

        except Exception as e:
            logger.error(f"❌ {exchange_name} method signature test failed: {e}")
            raise

    def _test_abstract_methods(self, exchange: IExchangeClient, exchange_name: str) -> None:
        """Test that all abstract methods from BaseExchange are implemented."""
        try:
            # Test abstract methods from BaseExchange
            abstract_methods = [
                '_initialize_exchange',
                '_convert_to_market_data',
                '_get_market_id',
                '_get_klines_raw',
                '_get_account_info_raw',
                '_create_order_raw',
                '_get_position_risk_raw',
                '_get_historical_klines_raw',
                '_get_historical_agg_trades_raw',
                '_get_open_orders_raw',
                '_cancel_order_raw',
                '_get_order_status_raw'
            ]
            
            for method_name in abstract_methods:
                if not hasattr(exchange, method_name):
                    logger.error(f"❌ {exchange_name} missing abstract method: {method_name}")
                    raise ValueError(f"Missing abstract method: {method_name}")
                logger.info(f"✅ {exchange_name} has abstract method: {method_name}")

        except Exception as e:
            logger.error(f"❌ {exchange_name} abstract method test failed: {e}")
            raise

    def test_all_exchanges(self) -> bool:
        """Test all available exchanges."""
        results = {}
        
        logger.info("Starting exchange interface compliance tests...")
        
        for exchange_name in self.exchanges:
            try:
                success = self.test_exchange_interface_compliance(exchange_name)
                results[exchange_name] = success
            except Exception as e:
                logger.error(f"❌ {exchange_name} test failed with exception: {e}")
                results[exchange_name] = False
        
        # Test cross-exchange consistency
        logger.info("\n" + "="*60)
        logger.info("CROSS-EXCHANGE CONSISTENCY TEST")
        logger.info("="*60)
        
        consistency_success = self._test_cross_exchange_consistency()
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("EXCHANGE INTERFACE COMPLIANCE TEST RESULTS")
        logger.info("="*60)
        
        all_passed = True
        for exchange_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{exchange_name.upper():<12} {status}")
            if not success:
                all_passed = False
        
        logger.info("="*60)
        if all_passed and consistency_success:
            logger.info("🎉 ALL EXCHANGES PASSED INTERFACE COMPLIANCE TESTS!")
        else:
            logger.error("❌ SOME EXCHANGES FAILED INTERFACE COMPLIANCE TESTS")
        
        return all_passed and consistency_success

    def _test_cross_exchange_consistency(self) -> bool:
        """Test that all exchanges have the same function names, inputs, and outputs."""
        try:
            logger.info("Testing cross-exchange consistency...")
            
            # Get all exchange instances
            exchanges = {}
            for exchange_name in self.exchanges:
                exchanges[exchange_name] = ExchangeFactory.get_exchange(exchange_name)
            
            # Test 1: Same function names
            logger.info("🔍 Testing function name consistency...")
            all_methods = set()
            for exchange_name, exchange in exchanges.items():
                methods = set(dir(exchange))
                # Filter out private methods and built-ins
                public_methods = {m for m in methods if not m.startswith('_') and not m.startswith('__')}
                all_methods.update(public_methods)
            
            # Check that all exchanges have the same core methods
            core_methods = {
                'get_klines', 'get_account_info', 'create_order', 'get_position_risk',
                'get_historical_klines', 'get_historical_agg_trades', 'get_open_orders',
                'cancel_order', 'get_order_status', 'close'
            }
            
            for exchange_name, exchange in exchanges.items():
                exchange_methods = set(dir(exchange))
                exchange_methods = {m for m in exchange_methods if not m.startswith('_') and not m.startswith('__')}
                
                missing_methods = core_methods - exchange_methods
                if missing_methods:
                    logger.error(f"❌ {exchange_name} missing core methods: {missing_methods}")
                    return False
            
            logger.info("✅ All exchanges have the same core function names")
            
            # Test 2: Same input parameters
            logger.info("🔍 Testing input parameter consistency...")
            for method_name in core_methods:
                signatures = {}
                for exchange_name, exchange in exchanges.items():
                    if hasattr(exchange, method_name):
                        method = getattr(exchange, method_name)
                        sig = inspect.signature(method)
                        params = list(sig.parameters.keys())
                        signatures[exchange_name] = params
                
                # Check that all exchanges have the same core parameters for this method
                # Allow for optional parameters that some exchanges might have
                core_params = set()
                for params in signatures.values():
                    # Filter out optional parameters that might vary between exchanges
                    if method_name == 'create_order':
                        # For create_order, allow time_in_force as optional
                        core_params.add(tuple([p for p in params if p not in ['time_in_force']]))
                    else:
                        core_params.add(tuple(params))
                
                if len(core_params) > 1:
                    logger.error(f"❌ Inconsistent core parameters for {method_name}:")
                    for exchange_name, params in signatures.items():
                        logger.error(f"   {exchange_name}: {params}")
                    return False
                else:
                    logger.info(f"✅ {method_name} has consistent core parameters across all exchanges")
            
            # Test 3: Same output format
            logger.info("🔍 Testing output format consistency...")
            for method_name in core_methods:
                return_types = {}
                for exchange_name, exchange in exchanges.items():
                    if hasattr(exchange, method_name):
                        method = getattr(exchange, method_name)
                        hints = get_type_hints(method)
                        return_types[exchange_name] = hints.get('return', 'No type hint')
                
                # Check that all exchanges have the same return type for this method
                unique_return_types = set(str(rt) for rt in return_types.values())
                if len(unique_return_types) > 1:
                    logger.warning(f"⚠️ Inconsistent return types for {method_name}:")
                    for exchange_name, return_type in return_types.items():
                        logger.warning(f"   {exchange_name}: {return_type}")
                else:
                    logger.info(f"✅ {method_name} has consistent return types across all exchanges")
            
            logger.info("✅ Cross-exchange consistency test passed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cross-exchange consistency test failed: {e}")
            return False


def main():
    """Main test function."""
    tester = ExchangeInterfaceComplianceTester()
    success = tester.test_all_exchanges()
    
    if success:
        print("\n✅ Exchange interface compliance verification completed successfully!")
        return 0
    else:
        print("\n❌ Exchange interface compliance verification failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 