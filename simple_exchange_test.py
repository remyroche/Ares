#!/usr/bin/env python3
"""
Simple Exchange Interface Test

A minimal testing script that works without heavy dependencies.
"""

import asyncio
import sys
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Simple tprint implementation
def tprint_info(msg: str) -> None:
    print(f"ℹ️ {msg}")

def tprint_success(msg: str) -> None:
    print(f"✅ {msg}")

def tprint_error(msg: str) -> None:
    print(f"❌ {msg}")

def tprint_warning(msg: str) -> None:
    print(f"⚠️ {msg}")

def tprint_structured(data: Dict[str, Any]) -> None:
    for key, value in data.items():
        print(f"   {key}: {value}")

@dataclass
class SimpleTestResult:
    """Simple test result."""
    operation: str
    success: bool
    duration: float
    error: Optional[str] = None
    data: Optional[Dict[str, Any]] = None

class SimpleExchangeTester:
    """Simple exchange interface tester."""
    
    def __init__(self):
        self.results: List[SimpleTestResult] = []
        self.exchange = None
    
    async def test_connection(self) -> SimpleTestResult:
        """Test basic connection functionality."""
        tprint_info("🔌 Testing Exchange Connection")
        
        start_time = time.time()
        try:
            # Try to import and create a basic exchange interface
            # This is a simplified version that doesn't require all dependencies
            tprint_info("Creating simulated exchange interface...")
            
            # Simulate connection time
            await asyncio.sleep(0.5)
            
            duration = time.time() - start_time
            tprint_success(f"✅ Simulated connection successful ({duration:.2f}s)")
            
            return SimpleTestResult(
                operation="connection",
                success=True,
                duration=duration,
                data={"status": "simulated_connected"}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Connection test failed: {e}")
            return SimpleTestResult(
                operation="connection",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_klines_simulation(self) -> SimpleTestResult:
        """Test klines data simulation."""
        tprint_info("📊 Testing Klines Data Simulation")
        
        start_time = time.time()
        try:
            # Simulate klines data
            await asyncio.sleep(0.3)
            
            # Create mock klines data
            mock_klines = [
                [1640995200000, "50000.00", "51000.00", "49500.00", "50500.00", "100.5", 1640995259999, "5075000.00", 1000, "50.25", "2537500.00", "0"],
                [1640995260000, "50500.00", "51500.00", "50000.00", "51000.00", "120.3", 1640995319999, "6120000.00", 1200, "60.15", "3060000.00", "0"],
                [1640995320000, "51000.00", "52000.00", "50500.00", "51500.00", "95.7", 1640995379999, "4920000.00", 950, "47.85", "2460000.00", "0"]
            ]
            
            duration = time.time() - start_time
            tprint_success(f"✅ Klines simulation successful ({duration:.2f}s)")
            tprint_structured({
                "Data Points": len(mock_klines),
                "Symbol": "BTCUSDT",
                "Interval": "1m",
                "Sample": mock_klines[0] if mock_klines else "N/A"
            })
            
            return SimpleTestResult(
                operation="klines",
                success=True,
                duration=duration,
                data={
                    "count": len(mock_klines),
                    "symbol": "BTCUSDT",
                    "interval": "1m",
                    "sample_data": mock_klines[0]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Klines test failed: {e}")
            return SimpleTestResult(
                operation="klines",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_balance_simulation(self) -> SimpleTestResult:
        """Test balance data simulation."""
        tprint_info("💰 Testing Balance Data Simulation")
        
        start_time = time.time()
        try:
            # Simulate balance fetch
            await asyncio.sleep(0.2)
            
            # Create mock balance data
            mock_balance = {
                "USDT": 10000.0,
                "BTC": 0.5,
                "ETH": 10.0,
                "BNB": 100.0
            }
            
            duration = time.time() - start_time
            tprint_success(f"✅ Balance simulation successful ({duration:.2f}s)")
            tprint_structured({
                "Available Assets": list(mock_balance.keys()),
                "USDT Balance": mock_balance["USDT"],
                "BTC Balance": mock_balance["BTC"]
            })
            
            return SimpleTestResult(
                operation="balance",
                success=True,
                duration=duration,
                data={"balance": mock_balance}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Balance test failed: {e}")
            return SimpleTestResult(
                operation="balance",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_ticker_simulation(self) -> SimpleTestResult:
        """Test ticker data simulation."""
        tprint_info("📈 Testing Ticker Data Simulation")
        
        start_time = time.time()
        try:
            # Simulate ticker fetch
            await asyncio.sleep(0.25)
            
            # Create mock ticker data
            mock_ticker = {
                "symbol": "BTCUSDT",
                "price": 50000.0,
                "bid_price": 49995.0,
                "ask_price": 50005.0,
                "volume_24h": 1000000.0,
                "price_change_24h": 2000.0,
                "price_change_percent_24h": 4.17
            }
            
            duration = time.time() - start_time
            tprint_success(f"✅ Ticker simulation successful ({duration:.2f}s)")
            tprint_structured({
                "Symbol": mock_ticker["symbol"],
                "Price": mock_ticker["price"],
                "Bid": mock_ticker["bid_price"],
                "Ask": mock_ticker["ask_price"],
                "Volume 24h": mock_ticker["volume_24h"],
                "Change 24h": f"{mock_ticker['price_change_percent_24h']:.2f}%"
            })
            
            return SimpleTestResult(
                operation="ticker",
                success=True,
                duration=duration,
                data=mock_ticker
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Ticker test failed: {e}")
            return SimpleTestResult(
                operation="ticker",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_orderbook_simulation(self) -> SimpleTestResult:
        """Test order book data simulation."""
        tprint_info("📖 Testing Order Book Data Simulation")
        
        start_time = time.time()
        try:
            # Simulate order book fetch
            await asyncio.sleep(0.2)
            
            # Create mock order book data
            mock_orderbook = {
                "bids": [
                    [49995.0, 1.5],
                    [49990.0, 2.0],
                    [49985.0, 1.0]
                ],
                "asks": [
                    [50005.0, 1.2],
                    [50010.0, 1.8],
                    [50015.0, 0.9]
                ]
            }
            
            duration = time.time() - start_time
            tprint_success(f"✅ Order book simulation successful ({duration:.2f}s)")
            tprint_structured({
                "Symbol": "BTCUSDT",
                "Bids Count": len(mock_orderbook["bids"]),
                "Asks Count": len(mock_orderbook["asks"]),
                "Best Bid": mock_orderbook["bids"][0],
                "Best Ask": mock_orderbook["asks"][0]
            })
            
            return SimpleTestResult(
                operation="orderbook",
                success=True,
                duration=duration,
                data=mock_orderbook
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Order book test failed: {e}")
            return SimpleTestResult(
                operation="orderbook",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def run_all_tests(self) -> None:
        """Run all simulation tests."""
        tprint_success("🧪 Starting Simple Exchange Interface Test Suite")
        print("=" * 60)
        
        total_start = time.time()
        
        # Run individual tests
        tests = [
            self.test_connection(),
            self.test_klines_simulation(),
            self.test_balance_simulation(),
            self.test_ticker_simulation(),
            self.test_orderbook_simulation()
        ]
        
        for test_coro in tests:
            result = await test_coro
            self.results.append(result)
            print()  # Add spacing between tests
        
        # Calculate summary
        total_duration = time.time() - total_start
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        success_rate = (passed / len(self.results) * 100) if self.results else 0
        
        # Display summary
        tprint_success("📊 Test Suite Summary")
        print("=" * 60)
        tprint_structured({
            "Total Tests": len(self.results),
            "Passed": passed,
            "Failed": failed,
            "Success Rate": f"{success_rate:.1f}%",
            "Total Duration": f"{total_duration:.2f}s"
        })
        
        # Show individual results
        print("\n📋 Individual Test Results:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.operation}: {result.duration:.2f}s")
            if result.error:
                print(f"    Error: {result.error}")
        
        # Show any errors
        errors = [r for r in self.results if not r.success]
        if errors:
            print(f"\n❌ {len(errors)} test(s) failed:")
            for result in errors:
                print(f"  - {result.operation}: {result.error}")
        
        print(f"\n{'✅ All tests passed!' if failed == 0 else f'❌ {failed} test(s) failed'}")

async def main():
    """Main entry point."""
    print("🚀 Simple Exchange Interface Test Suite")
    print("This is a simulation test that doesn't require external dependencies")
    print()
    
    tester = SimpleExchangeTester()
    await tester.run_all_tests()
    
    # Return success status
    failed_tests = sum(1 for r in tester.results if not r.success)
    return failed_tests == 0

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        sys.exit(1)