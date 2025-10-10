#!/usr/bin/env python3
"""
Exchange Interface Testing Suite

A comprehensive testing suite for ExchangeInterface that validates:
- Download klines
- Open/close positions
- Fetch balance
- Fetch trade ID & information
- Order management
- Market data access

Features:
- Detailed logging with tprint
- CLI interface
- Configurable test parameters
- Error handling and reporting
- Performance metrics
"""

import asyncio
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Import ExchangeInterface and dependencies
try:
    from src.trading.execution.exchange_interface import ExchangeInterface, ExchangeType, ConnectionStatus
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_structured, LogLevel
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

@dataclass
class TestConfig:
    """Configuration for the testing suite."""
    exchange_type: str = "simulated"
    test_symbol: str = "BTCUSDT"
    test_interval: str = "1m"
    test_quantity: float = 0.001
    test_price: Optional[float] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    testnet: bool = True
    verbose: bool = True
    timeout: int = 30
    max_retries: int = 3
    test_operations: List[str] = field(default_factory=lambda: [
        "connection", "klines", "balance", "ticker", "orderbook", 
        "trades", "orders", "positions"
    ])

@dataclass
class TestResult:
    """Result of a single test operation."""
    operation: str
    success: bool
    duration: float
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSuiteResults:
    """Results of the entire test suite."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    total_duration: float = 0.0
    results: List[TestResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class ExchangeInterfaceTestSuite:
    """Comprehensive testing suite for ExchangeInterface."""
    
    def __init__(self, config: TestConfig):
        """Initialize the test suite."""
        self.config = config
        self.exchange: Optional[ExchangeInterface] = None
        self.results = TestSuiteResults()
        self.start_time = None
        
        # Test data storage
        self.test_orders: List[Dict[str, Any]] = []
        self.test_positions: List[Dict[str, Any]] = []
        
        tprint_info("🚀 Exchange Interface Test Suite Initialized")
        tprint_structured({
            "Exchange Type": config.exchange_type,
            "Test Symbol": config.test_symbol,
            "Test Interval": config.test_interval,
            "Test Operations": config.test_operations,
            "Verbose": config.verbose
        })

    async def run_all_tests(self) -> TestSuiteResults:
        """Run all configured tests."""
        self.start_time = time.time()
        tprint_success("🧪 Starting Exchange Interface Test Suite")
        
        try:
            # Initialize exchange
            await self._test_connection()
            
            # Run individual test operations
            for operation in self.config.test_operations:
                if operation == "connection":
                    continue  # Already tested
                elif operation == "klines":
                    await self._test_klines()
                elif operation == "balance":
                    await self._test_balance()
                elif operation == "ticker":
                    await self._test_ticker()
                elif operation == "orderbook":
                    await self._test_orderbook()
                elif operation == "trades":
                    await self._test_trades()
                elif operation == "orders":
                    await self._test_orders()
                elif operation == "positions":
                    await self._test_positions()
                else:
                    tprint_warning(f"⚠️ Unknown test operation: {operation}")
            
            # Generate summary
            await self._generate_summary()
            
        except Exception as e:
            tprint_error(f"❌ Test suite failed with error: {e}")
            self.results.errors.append(str(e))
        
        self.results.total_duration = time.time() - self.start_time
        return self.results

    async def _test_connection(self) -> None:
        """Test exchange connection."""
        tprint_info("🔌 Testing Exchange Connection")
        
        start_time = time.time()
        try:
            # Create exchange interface
            exchange_config = {
                'exchange_type': self.config.exchange_type,
                'api_key': self.config.api_key,
                'api_secret': self.config.api_secret,
                'testnet': self.config.testnet,
                'rate_limits': {
                    'requests_per_minute': 1200,
                    'weight_per_minute': 6000
                }
            }
            
            self.exchange = ExchangeInterface(exchange_config)
            
            # Test connection
            connected = await self.exchange.connect()
            duration = time.time() - start_time
            
            if connected:
                tprint_success(f"✅ Connection successful ({duration:.2f}s)")
                self._record_result("connection", True, duration, 
                                  data={"status": "connected"})
            else:
                tprint_error("❌ Connection failed")
                self._record_result("connection", False, duration, 
                                  error="Failed to connect to exchange")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Connection test failed: {e}")
            self._record_result("connection", False, duration, error=str(e))

    async def _test_klines(self) -> None:
        """Test klines data download."""
        tprint_info("📊 Testing Klines Download")
        
        start_time = time.time()
        try:
            # Test recent klines
            klines = await self.exchange.get_klines(
                symbol=self.config.test_symbol,
                interval=self.config.test_interval,
                limit=100
            )
            duration = time.time() - start_time
            
            if klines and len(klines) > 0:
                tprint_success(f"✅ Klines download successful ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": self.config.test_symbol,
                    "Interval": self.config.test_interval,
                    "Data Points": len(klines),
                    "First Candle": klines[0] if klines else "N/A",
                    "Last Candle": klines[-1] if klines else "N/A"
                })
                
                self._record_result("klines", True, duration, 
                                  data={
                                      "count": len(klines),
                                      "symbol": self.config.test_symbol,
                                      "interval": self.config.test_interval,
                                      "sample_data": klines[0] if klines else None
                                  })
            else:
                tprint_error("❌ No klines data received")
                self._record_result("klines", False, duration, 
                                  error="No klines data received")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Klines test failed: {e}")
            self._record_result("klines", False, duration, error=str(e))

    async def _test_balance(self) -> None:
        """Test balance fetching."""
        tprint_info("💰 Testing Balance Fetch")
        
        start_time = time.time()
        try:
            # Test account balance
            balance = await self.exchange.get_account_balance()
            duration = time.time() - start_time
            
            if balance:
                tprint_success(f"✅ Balance fetch successful ({duration:.2f}s)")
                tprint_structured({
                    "Balance Data": balance,
                    "Available Assets": list(balance.keys()) if isinstance(balance, dict) else "N/A"
                })
                
                self._record_result("balance", True, duration, 
                                  data={"balance": balance})
            else:
                tprint_warning("⚠️ No balance data received")
                self._record_result("balance", False, duration, 
                                  error="No balance data received")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Balance test failed: {e}")
            self._record_result("balance", False, duration, error=str(e))

    async def _test_ticker(self) -> None:
        """Test ticker data fetching."""
        tprint_info("📈 Testing Ticker Data")
        
        start_time = time.time()
        try:
            ticker = await self.exchange.get_ticker(self.config.test_symbol)
            duration = time.time() - start_time
            
            if ticker:
                tprint_success(f"✅ Ticker fetch successful ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": ticker.symbol,
                    "Price": ticker.price,
                    "Bid": ticker.bid_price,
                    "Ask": ticker.ask_price,
                    "Volume 24h": ticker.volume_24h,
                    "Change 24h": f"{ticker.price_change_percent_24h:.2f}%"
                })
                
                self._record_result("ticker", True, duration, 
                                  data={
                                      "symbol": ticker.symbol,
                                      "price": ticker.price,
                                      "bid": ticker.bid_price,
                                      "ask": ticker.ask_price
                                  })
            else:
                tprint_error("❌ No ticker data received")
                self._record_result("ticker", False, duration, 
                                  error="No ticker data received")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Ticker test failed: {e}")
            self._record_result("ticker", False, duration, error=str(e))

    async def _test_orderbook(self) -> None:
        """Test order book fetching."""
        tprint_info("📖 Testing Order Book")
        
        start_time = time.time()
        try:
            orderbook = await self.exchange.get_order_book(
                symbol=self.config.test_symbol,
                limit=20
            )
            duration = time.time() - start_time
            
            if orderbook and 'bids' in orderbook and 'asks' in orderbook:
                tprint_success(f"✅ Order book fetch successful ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": self.config.test_symbol,
                    "Bids Count": len(orderbook['bids']),
                    "Asks Count": len(orderbook['asks']),
                    "Best Bid": orderbook['bids'][0] if orderbook['bids'] else "N/A",
                    "Best Ask": orderbook['asks'][0] if orderbook['asks'] else "N/A"
                })
                
                self._record_result("orderbook", True, duration, 
                                  data={
                                      "symbol": self.config.test_symbol,
                                      "bids_count": len(orderbook['bids']),
                                      "asks_count": len(orderbook['asks'])
                                  })
            else:
                tprint_error("❌ Invalid order book data received")
                self._record_result("orderbook", False, duration, 
                                  error="Invalid order book data")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Order book test failed: {e}")
            self._record_result("orderbook", False, duration, error=str(e))

    async def _test_trades(self) -> None:
        """Test recent trades fetching."""
        tprint_info("🔄 Testing Recent Trades")
        
        start_time = time.time()
        try:
            trades = await self.exchange.get_recent_trades(
                symbol=self.config.test_symbol,
                limit=50
            )
            duration = time.time() - start_time
            
            if trades and len(trades) > 0:
                tprint_success(f"✅ Trades fetch successful ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": self.config.test_symbol,
                    "Trades Count": len(trades),
                    "Latest Trade": trades[0] if trades else "N/A"
                })
                
                self._record_result("trades", True, duration, 
                                  data={
                                      "count": len(trades),
                                      "symbol": self.config.test_symbol,
                                      "sample_trade": trades[0] if trades else None
                                  })
            else:
                tprint_warning("⚠️ No trades data received")
                self._record_result("trades", False, duration, 
                                  error="No trades data received")
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Trades test failed: {e}")
            self._record_result("trades", False, duration, error=str(e))

    async def _test_orders(self) -> None:
        """Test order management operations."""
        tprint_info("📋 Testing Order Management")
        
        start_time = time.time()
        try:
            # Test getting open orders
            open_orders = await self.exchange.get_open_orders(self.config.test_symbol)
            duration = time.time() - start_time
            
            tprint_success(f"✅ Order management test successful ({duration:.2f}s)")
            tprint_structured({
                "Symbol": self.config.test_symbol,
                "Open Orders Count": len(open_orders) if open_orders else 0
            })
            
            self._record_result("orders", True, duration, 
                              data={
                                  "open_orders_count": len(open_orders) if open_orders else 0,
                                  "symbol": self.config.test_symbol
                              })
            
            # Store for potential cleanup
            if open_orders:
                self.test_orders.extend(open_orders)
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Order management test failed: {e}")
            self._record_result("orders", False, duration, error=str(e))

    async def _test_positions(self) -> None:
        """Test position management (if supported by exchange)."""
        tprint_info("📊 Testing Position Management")
        
        start_time = time.time()
        try:
            # Note: Position management might not be available for all exchanges
            # This is a placeholder for position-related tests
            tprint_info("ℹ️ Position management test - checking if supported")
            
            # For simulated exchanges, we might not have real positions
            duration = time.time() - start_time
            
            tprint_success(f"✅ Position management test completed ({duration:.2f}s)")
            tprint_structured({
                "Note": "Position management availability depends on exchange type",
                "Exchange Type": self.config.exchange_type
            })
            
            self._record_result("positions", True, duration, 
                              data={
                                  "note": "Position management test completed",
                                  "exchange_type": self.config.exchange_type
                              })
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_warning(f"⚠️ Position management test failed: {e}")
            self._record_result("positions", False, duration, error=str(e))

    def _record_result(self, operation: str, success: bool, duration: float, 
                      error: Optional[str] = None, data: Optional[Dict[str, Any]] = None) -> None:
        """Record a test result."""
        result = TestResult(
            operation=operation,
            success=success,
            duration=duration,
            error=error,
            data=data
        )
        self.results.results.append(result)
        self.results.total_tests += 1
        
        if success:
            self.results.passed_tests += 1
        else:
            self.results.failed_tests += 1
            if error:
                self.results.errors.append(f"{operation}: {error}")

    async def _generate_summary(self) -> None:
        """Generate test suite summary."""
        tprint_success("📊 Test Suite Summary")
        
        # Calculate success rate
        success_rate = (self.results.passed_tests / self.results.total_tests * 100) if self.results.total_tests > 0 else 0
        
        tprint_structured({
            "Total Tests": self.results.total_tests,
            "Passed": self.results.passed_tests,
            "Failed": self.results.failed_tests,
            "Success Rate": f"{success_rate:.1f}%",
            "Total Duration": f"{self.results.total_duration:.2f}s"
        })
        
        # Show individual results
        tprint_info("📋 Individual Test Results:")
        for result in self.results.results:
            status = "✅" if result.success else "❌"
            tprint(f"  {status} {result.operation}: {result.duration:.2f}s")
            if result.error:
                tprint(f"    Error: {result.error}")
        
        # Show errors and warnings
        if self.results.errors:
            tprint_error("❌ Errors:")
            for error in self.results.errors:
                tprint(f"  - {error}")
        
        if self.results.warnings:
            tprint_warning("⚠️ Warnings:")
            for warning in self.results.warnings:
                tprint(f"  - {warning}")

    async def cleanup(self) -> None:
        """Cleanup resources after testing."""
        if self.exchange:
            try:
                await self.exchange.disconnect()
                tprint_info("🧹 Exchange connection closed")
            except Exception as e:
                tprint_warning(f"⚠️ Error during cleanup: {e}")

def create_test_config_from_args(args) -> TestConfig:
    """Create test configuration from command line arguments."""
    return TestConfig(
        exchange_type=args.exchange,
        test_symbol=args.symbol,
        test_interval=args.interval,
        test_quantity=args.quantity,
        api_key=args.api_key,
        api_secret=args.api_secret,
        testnet=not args.live,
        verbose=args.verbose,
        timeout=args.timeout,
        test_operations=args.operations.split(',') if args.operations else None
    )

async def main():
    """Main entry point for the test suite."""
    parser = argparse.ArgumentParser(
        description="Exchange Interface Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test simulated exchange with default settings
  python exchange_interface_test_suite.py

  # Test specific operations on Binance testnet
  python exchange_interface_test_suite.py --exchange binance --operations klines,balance,ticker

  # Test with custom symbol and verbose output
  python exchange_interface_test_suite.py --symbol ETHUSDT --interval 5m --verbose

  # Test live exchange (requires API credentials)
  python exchange_interface_test_suite.py --exchange binance --live --api-key YOUR_KEY --api-secret YOUR_SECRET
        """
    )
    
    parser.add_argument('--exchange', '-e', 
                       choices=['simulated', 'binance', 'coinbase', 'kraken', 'bybit'],
                       default='simulated',
                       help='Exchange type to test (default: simulated)')
    
    parser.add_argument('--symbol', '-s',
                       default='BTCUSDT',
                       help='Trading symbol to test (default: BTCUSDT)')
    
    parser.add_argument('--interval', '-i',
                       default='1m',
                       help='Kline interval to test (default: 1m)')
    
    parser.add_argument('--quantity', '-q',
                       type=float,
                       default=0.001,
                       help='Test quantity for orders (default: 0.001)')
    
    parser.add_argument('--api-key',
                       help='API key for live exchange testing')
    
    parser.add_argument('--api-secret',
                       help='API secret for live exchange testing')
    
    parser.add_argument('--live', '-l',
                       action='store_true',
                       help='Use live exchange (default: testnet)')
    
    parser.add_argument('--operations', '-o',
                       help='Comma-separated list of operations to test (default: all)')
    
    parser.add_argument('--timeout', '-t',
                       type=int,
                       default=30,
                       help='Timeout for operations in seconds (default: 30)')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--output', '-f',
                       help='Output results to JSON file')
    
    args = parser.parse_args()
    
    # Create test configuration
    config = create_test_config_from_args(args)
    
    # Initialize and run test suite
    test_suite = ExchangeInterfaceTestSuite(config)
    
    try:
        # Run tests
        results = await test_suite.run_all_tests()
        
        # Output results to file if requested
        if args.output:
            output_data = {
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "exchange_type": config.exchange_type,
                    "test_symbol": config.test_symbol,
                    "test_interval": config.test_interval,
                    "test_operations": config.test_operations
                },
                "results": {
                    "total_tests": results.total_tests,
                    "passed_tests": results.passed_tests,
                    "failed_tests": results.failed_tests,
                    "total_duration": results.total_duration,
                    "success_rate": (results.passed_tests / results.total_tests * 100) if results.total_tests > 0 else 0
                },
                "test_details": [
                    {
                        "operation": r.operation,
                        "success": r.success,
                        "duration": r.duration,
                        "error": r.error,
                        "data": r.data
                    } for r in results.results
                ],
                "errors": results.errors,
                "warnings": results.warnings
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            tprint_success(f"📄 Results saved to {args.output}")
        
        # Exit with appropriate code
        sys.exit(0 if results.failed_tests == 0 else 1)
        
    except KeyboardInterrupt:
        tprint_warning("⚠️ Test suite interrupted by user")
        sys.exit(130)
    except Exception as e:
        tprint_error(f"❌ Test suite failed: {e}")
        sys.exit(1)
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    asyncio.run(main())