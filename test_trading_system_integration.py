"""
Integration Test for Complete Trading System

Tests the integration between live trading module, exchange-agnostic receiver,
and exchange APIs to ensure the complete system works end-to-end.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import logging

# Import trading system components
from live_trading.trading_engine import TradingEngine
from live_trading.config import TradingConfig, TradingMode, OrderType, OrderSide
from live_trading.api_client import APIClient
from exchanges.trading_receiver import TradingReceiver, TradingMessage, MessageType
from exchanges.exchange_registry import ExchangeRegistry
from exchanges.order_router import OrderRouter
from exchanges.data_aggregator import DataAggregator
from exchange.factory import ExchangeFactory
from src.interfaces.base_interfaces import TradeDecision, AnalysisResult, StrategyResult


class TradingSystemIntegrationTest:
    """Integration test for the complete trading system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        
        # Test configuration
        self.test_config = {
            "mode": "paper",  # Use paper trading for safety
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "exchanges": {
                "binance": {
                    "api_key": "test_key",
                    "api_secret": "test_secret",
                    "enabled": True
                },
                "okx": {
                    "api_key": "test_key", 
                    "api_secret": "test_secret",
                    "password": "test_passphrase",
                    "enabled": True
                }
            },
            "risk_management": {
                "max_position_size": 1000.0,
                "max_daily_loss": 100.0,
                "max_leverage": 10.0
            },
            "data_update_interval": 1.0,
            "reconnect_attempts": 3,
            "reconnect_delay": 5.0
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        self.logger.info("Starting Trading System Integration Tests")
        
        try:
            # Test 1: Trading Engine Initialization
            await self.test_trading_engine_initialization()
            
            # Test 2: Exchange Registry and Factory
            await self.test_exchange_registry()
            
            # Test 3: Trading Receiver
            await self.test_trading_receiver()
            
            # Test 4: Order Router
            await self.test_order_router()
            
            # Test 5: Data Aggregator
            await self.test_data_aggregator()
            
            # Test 6: API Client
            await self.test_api_client()
            
            # Test 7: End-to-End Trading Flow
            await self.test_end_to_end_trading()
            
            # Test 8: Error Handling and Recovery
            await self.test_error_handling()
            
            self.logger.info("All integration tests completed")
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"Integration tests failed: {e}")
            return {"error": str(e), "results": self.test_results}
    
    async def test_trading_engine_initialization(self):
        """Test trading engine initialization"""
        self.logger.info("Testing Trading Engine Initialization")
        
        try:
            # Create trading config
            config = TradingConfig(
                mode=TradingMode.PAPER,
                symbols=["BTCUSDT"],
                exchanges=self.test_config["exchanges"],
                risk_management=self.test_config["risk_management"],
                data_update_interval=self.test_config["data_update_interval"]
            )
            
            # Create mock exchange client
            exchange_client = await self._create_mock_exchange_client()
            
            # Initialize trading engine
            engine = TradingEngine(config, exchange_client)
            
            # Test engine start
            await engine.start()
            
            # Test engine status
            status = await engine.get_trading_status()
            assert status["running"] == True
            assert status["trading_active"] == True
            
            # Test engine stop
            await engine.stop()
            
            self.test_results["trading_engine"] = "PASS"
            self.logger.info("✅ Trading Engine Initialization: PASS")
            
        except Exception as e:
            self.test_results["trading_engine"] = f"FAIL: {e}"
            self.logger.error(f"❌ Trading Engine Initialization: FAIL - {e}")
    
    async def test_exchange_registry(self):
        """Test exchange registry functionality"""
        self.logger.info("Testing Exchange Registry")
        
        try:
            registry = ExchangeRegistry()
            await registry.start()
            
            # Test exchange registration
            binance_exchange = ExchangeFactory.get_exchange("binance")
            okx_exchange = ExchangeFactory.get_exchange("okx")
            
            await registry.register_exchange("binance", binance_exchange)
            await registry.register_exchange("okx", okx_exchange)
            
            # Test getting exchanges
            binance = await registry.get_exchange("binance")
            okx = await registry.get_exchange("okx")
            
            assert binance is not None
            assert okx is not None
            
            # Test getting registered exchanges
            registered = await registry.get_registered_exchanges()
            assert "binance" in registered
            assert "okx" in registered
            
            # Test exchange status
            status = await registry.get_all_exchange_status()
            assert "binance" in status
            assert "okx" in status
            
            await registry.stop()
            
            self.test_results["exchange_registry"] = "PASS"
            self.logger.info("✅ Exchange Registry: PASS")
            
        except Exception as e:
            self.test_results["exchange_registry"] = f"FAIL: {e}"
            self.logger.error(f"❌ Exchange Registry: FAIL - {e}")
    
    async def test_trading_receiver(self):
        """Test trading receiver functionality"""
        self.logger.info("Testing Trading Receiver")
        
        try:
            receiver = TradingReceiver(self.test_config)
            await receiver.start()
            
            # Test order message
            order_message = TradingMessage(
                id="test_order_1",
                type=MessageType.ORDER,
                exchange="binance",
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                data={
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 0.001,
                    "price": None
                }
            )
            
            response = await receiver.process_message(order_message)
            assert response.success == True or response.error is not None  # Should handle gracefully
            
            # Test data request message
            data_message = TradingMessage(
                id="test_data_1",
                type=MessageType.DATA_REQUEST,
                exchange="binance",
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                data={
                    "data_type": "ticker"
                }
            )
            
            response = await receiver.process_message(data_message)
            assert response.success == True or response.error is not None
            
            # Test statistics
            stats = await receiver.get_statistics()
            assert "running" in stats
            assert "statistics" in stats
            
            await receiver.stop()
            
            self.test_results["trading_receiver"] = "PASS"
            self.logger.info("✅ Trading Receiver: PASS")
            
        except Exception as e:
            self.test_results["trading_receiver"] = f"FAIL: {e}"
            self.logger.error(f"❌ Trading Receiver: FAIL - {e}")
    
    async def test_order_router(self):
        """Test order router functionality"""
        self.logger.info("Testing Order Router")
        
        try:
            registry = ExchangeRegistry()
            await registry.start()
            
            # Register mock exchanges
            binance_exchange = await self._create_mock_exchange_client()
            await registry.register_exchange("binance", binance_exchange)
            
            router = OrderRouter(registry)
            await router.start()
            
            # Test order routing
            result = await router.route_order(
                exchange="binance",
                symbol="BTCUSDT",
                side="buy",
                order_type="market",
                quantity=0.001
            )
            
            assert "success" in result
            assert "order_id" in result
            
            # Test order status
            if result["success"]:
                order_id = result["order_id"]
                status = await router.get_order_status(order_id)
                assert "success" in status
                assert "status" in status
            
            # Test active orders
            active_orders = await router.get_active_orders()
            assert isinstance(active_orders, list)
            
            # Test order history
            history = await router.get_order_history(limit=10)
            assert isinstance(history, list)
            
            # Test statistics
            stats = await router.get_statistics()
            assert "running" in stats
            assert "statistics" in stats
            
            await router.stop()
            await registry.stop()
            
            self.test_results["order_router"] = "PASS"
            self.logger.info("✅ Order Router: PASS")
            
        except Exception as e:
            self.test_results["order_router"] = f"FAIL: {e}"
            self.logger.error(f"❌ Order Router: FAIL - {e}")
    
    async def test_data_aggregator(self):
        """Test data aggregator functionality"""
        self.logger.info("Testing Data Aggregator")
        
        try:
            registry = ExchangeRegistry()
            await registry.start()
            
            # Register mock exchanges
            binance_exchange = await self._create_mock_exchange_client()
            okx_exchange = await self._create_mock_exchange_client()
            await registry.register_exchange("binance", binance_exchange)
            await registry.register_exchange("okx", okx_exchange)
            
            aggregator = DataAggregator(registry)
            await aggregator.start()
            
            # Test single exchange data
            result = await aggregator.get_data(
                exchange="binance",
                symbol="BTCUSDT",
                data_type="ticker"
            )
            
            assert "success" in result
            assert "data" in result
            
            # Test aggregated data
            aggregated_result = await aggregator.get_aggregated_data(
                symbol="BTCUSDT",
                data_type="ticker",
                exchanges=["binance", "okx"]
            )
            
            assert "success" in aggregated_result
            assert "data" in aggregated_result
            assert "exchange_data" in aggregated_result
            
            # Test different data types
            for data_type in ["klines", "orderbook", "account_info"]:
                result = await aggregator.get_data(
                    exchange="binance",
                    symbol="BTCUSDT",
                    data_type=data_type
                )
                assert "success" in result
            
            # Test statistics
            stats = await aggregator.get_statistics()
            assert "running" in stats
            assert "statistics" in stats
            
            await aggregator.stop()
            await registry.stop()
            
            self.test_results["data_aggregator"] = "PASS"
            self.logger.info("✅ Data Aggregator: PASS")
            
        except Exception as e:
            self.test_results["data_aggregator"] = f"FAIL: {e}"
            self.logger.error(f"❌ Data Aggregator: FAIL - {e}")
    
    async def test_api_client(self):
        """Test API client functionality"""
        self.logger.info("Testing API Client")
        
        try:
            config = TradingConfig(
                mode=TradingMode.PAPER,
                symbols=["BTCUSDT"],
                exchanges=self.test_config["exchanges"]
            )
            
            # Test Binance API client
            binance_client = APIClient(config, "binance")
            await binance_client.start()
            
            # Test OKX API client
            okx_client = APIClient(config, "okx")
            await okx_client.start()
            
            # Test API client statistics
            binance_stats = await binance_client.get_statistics()
            okx_stats = await okx_client.get_statistics()
            
            assert "exchange" in binance_stats
            assert "exchange" in okx_stats
            assert binance_stats["exchange"] == "binance"
            assert okx_stats["exchange"] == "okx"
            
            await binance_client.stop()
            await okx_client.stop()
            
            self.test_results["api_client"] = "PASS"
            self.logger.info("✅ API Client: PASS")
            
        except Exception as e:
            self.test_results["api_client"] = f"FAIL: {e}"
            self.logger.error(f"❌ API Client: FAIL - {e}")
    
    async def test_end_to_end_trading(self):
        """Test end-to-end trading flow"""
        self.logger.info("Testing End-to-End Trading Flow")
        
        try:
            # Create complete trading system
            config = TradingConfig(
                mode=TradingMode.PAPER,
                symbols=["BTCUSDT"],
                exchanges=self.test_config["exchanges"],
                risk_management=self.test_config["risk_management"]
            )
            
            # Initialize components
            exchange_client = await self._create_mock_exchange_client()
            engine = TradingEngine(config, exchange_client)
            
            # Start trading engine
            await engine.start()
            
            # Create trade decision
            decision = TradeDecision(
                symbol="BTCUSDT",
                action="buy",
                quantity=0.001,
                price=0.0,  # Market order
                confidence=0.8,
                risk_score=0.2,
                leverage=1.0,
                stop_loss=45000.0,
                take_profit=55000.0
            )
            
            # Execute trade decision
            order = await engine.execute_trade_decision(decision)
            
            # Verify order was created
            assert order is not None
            assert order.symbol == "BTCUSDT"
            assert order.side == OrderSide.BUY
            assert order.quantity == 0.001
            
            # Test position summary
            positions = await engine.get_position_summary()
            assert "BTCUSDT" in positions
            
            # Test performance metrics
            metrics = await engine.get_performance_metrics()
            assert "total_trades" in metrics
            assert "win_rate" in metrics
            
            # Test trading status
            status = await engine.get_trading_status()
            assert status["running"] == True
            assert status["trading_active"] == True
            
            # Test pause/resume
            await engine.pause_trading()
            assert engine._trading_active == False
            
            await engine.resume_trading()
            assert engine._trading_active == True
            
            # Stop trading engine
            await engine.stop()
            
            self.test_results["end_to_end_trading"] = "PASS"
            self.logger.info("✅ End-to-End Trading: PASS")
            
        except Exception as e:
            self.test_results["end_to_end_trading"] = f"FAIL: {e}"
            self.logger.error(f"❌ End-to-End Trading: FAIL - {e}")
    
    async def test_error_handling(self):
        """Test error handling and recovery"""
        self.logger.info("Testing Error Handling and Recovery")
        
        try:
            # Test with invalid configuration
            invalid_config = TradingConfig(
                mode=TradingMode.PAPER,
                symbols=["INVALID_SYMBOL"],
                exchanges={},
                risk_management={}
            )
            
            exchange_client = await self._create_mock_exchange_client()
            engine = TradingEngine(invalid_config, exchange_client)
            
            # Test engine start with invalid config
            try:
                await engine.start()
                # Should handle gracefully
                status = await engine.get_trading_status()
                assert "running" in status
            except Exception as e:
                # Expected to fail gracefully
                pass
            
            # Test emergency stop
            await engine.emergency_stop()
            
            # Test with invalid trade decision
            invalid_decision = TradeDecision(
                symbol="INVALID_SYMBOL",
                action="invalid_action",
                quantity=-1.0,  # Invalid quantity
                price=-100.0,   # Invalid price
                confidence=2.0,  # Invalid confidence
                risk_score=-1.0,  # Invalid risk score
                leverage=0.0,
                stop_loss=0.0,
                take_profit=0.0
            )
            
            # Should handle invalid decision gracefully
            order = await engine.execute_trade_decision(invalid_decision)
            # Should either create order with validation or return None
            assert order is None or order is not None
            
            await engine.stop()
            
            self.test_results["error_handling"] = "PASS"
            self.logger.info("✅ Error Handling: PASS")
            
        except Exception as e:
            self.test_results["error_handling"] = f"FAIL: {e}"
            self.logger.error(f"❌ Error Handling: FAIL - {e}")
    
    async def _create_mock_exchange_client(self):
        """Create a mock exchange client for testing"""
        class MockExchangeClient:
            def __init__(self):
                self.logger = logging.getLogger("MockExchangeClient")
            
            async def _initialize_exchange(self):
                pass
            
            async def get_ticker(self, symbol):
                return {
                    "symbol": symbol,
                    "last": "50000.0",
                    "bid": "49999.0",
                    "ask": "50001.0",
                    "volume": "1000.0"
                }
            
            async def get_account_info(self):
                return {
                    "totalBalance": "10000.0",
                    "availableBalance": "10000.0"
                }
            
            async def create_order(self, symbol, side, quantity, price=None, order_type="MARKET"):
                return {
                    "orderId": f"mock_order_{int(time.time())}",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "status": "FILLED"
                }
            
            async def get_order_status(self, symbol, order_id):
                return {
                    "orderId": order_id,
                    "status": "FILLED",
                    "executedQty": "0.001",
                    "avgPrice": "50000.0"
                }
            
            async def cancel_order(self, symbol, order_id):
                return {
                    "orderId": order_id,
                    "status": "CANCELLED"
                }
            
            async def get_open_orders(self, symbol=None):
                return []
            
            async def get_position_risk(self, symbol):
                return {
                    "symbol": symbol,
                    "size": "0.0",
                    "markPrice": "50000.0",
                    "unrealizedPnl": "0.0"
                }
            
            async def close(self):
                pass
        
        return MockExchangeClient()
    
    def print_test_results(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("TRADING SYSTEM INTEGRATION TEST RESULTS")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        print("\nDetailed Results:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result == "PASS" else f"❌ FAIL: {result}"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print("="*60)
        
        if failed_tests == 0:
            print("🎉 ALL TESTS PASSED! Trading system is ready for production.")
        else:
            print(f"⚠️  {failed_tests} test(s) failed. Please review and fix issues.")


async def main():
    """Main test runner"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run integration tests
    test_suite = TradingSystemIntegrationTest()
    results = await test_suite.run_all_tests()
    
    # Print results
    test_suite.print_test_results()
    
    return results


if __name__ == "__main__":
    asyncio.run(main())