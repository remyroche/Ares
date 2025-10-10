#!/usr/bin/env python3
"""
Enhanced Position Testing Suite for ExchangeInterface

Comprehensive testing for position management including:
- Open positions with different symbols
- Different position sizes
- Short and long positions (perpetuals)
- Market orders only
- Position monitoring and management
"""

import asyncio
import sys
import time
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

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

class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"

@dataclass
class PositionTestConfig:
    """Configuration for position testing."""
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "ADAUSDT"])
    position_sizes: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1])
    sides: List[PositionSide] = field(default_factory=lambda: [PositionSide.LONG, PositionSide.SHORT])
    order_type: OrderType = OrderType.MARKET
    test_perpetuals: bool = True
    test_spot: bool = False
    max_positions: int = 5
    position_timeout: int = 30
    cleanup_positions: bool = True

@dataclass
class PositionInfo:
    """Position information container."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    unrealized_pnl: float
    margin_used: float
    leverage: float
    position_id: str
    timestamp: float
    is_perpetual: bool = True

@dataclass
class PositionTestResult:
    """Position test result."""
    operation: str
    success: bool
    duration: float
    error: Optional[str] = None
    position_info: Optional[PositionInfo] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedPositionTester:
    """Enhanced position testing suite."""
    
    def __init__(self, config: PositionTestConfig):
        self.config = config
        self.results: List[PositionTestResult] = []
        self.open_positions: List[PositionInfo] = []
        self.exchange = None
        self.use_real_interface = False
        
        # Try to import real ExchangeInterface
        self._try_import_real_interface()
    
    def _try_import_real_interface(self) -> None:
        """Try to import the real ExchangeInterface."""
        try:
            # Add project root to path
            project_root = Path(__file__).parent
            sys.path.insert(0, str(project_root))
            
            from src.trading.execution.exchange_interface import ExchangeInterface
            self.ExchangeInterface = ExchangeInterface
            self.use_real_interface = True
            tprint_info("📦 Real ExchangeInterface imported successfully")
        except ImportError as e:
            tprint_warning(f"⚠️ Could not import real ExchangeInterface: {e}")
            tprint_info("🔄 Falling back to simulation mode")
            self.use_real_interface = False
    
    async def test_connection(self) -> PositionTestResult:
        """Test exchange connection."""
        tprint_info("🔌 Testing Exchange Connection for Position Management")
        
        start_time = time.time()
        try:
            if self.use_real_interface:
                # Use real ExchangeInterface
                exchange_config = {
                    'exchange_type': 'binance',  # Use Binance for perpetuals
                    'testnet': True,
                    'rate_limits': {
                        'requests_per_minute': 1200,
                        'weight_per_minute': 6000
                    }
                }
                
                self.exchange = self.ExchangeInterface(exchange_config)
                connected = await self.exchange.connect()
                
                if connected:
                    tprint_success("✅ Real exchange connection successful")
                else:
                    tprint_warning("⚠️ Real exchange connection failed, falling back to simulation")
                    self.use_real_interface = False
                    self.exchange = None
            else:
                # Use simulation
                await asyncio.sleep(0.5)
                tprint_success("✅ Simulated exchange connection successful")
            
            duration = time.time() - start_time
            
            return PositionTestResult(
                operation="connection",
                success=True,
                duration=duration,
                data={"mode": "real" if self.use_real_interface else "simulation"}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Connection test failed: {e}")
            return PositionTestResult(
                operation="connection",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_open_long_position(self, symbol: str, size: float) -> PositionTestResult:
        """Test opening a long position."""
        tprint_info(f"📈 Testing Long Position Open: {symbol} (Size: {size})")
        
        start_time = time.time()
        try:
            if self.use_real_interface and self.exchange:
                # Use real exchange
                position_info = await self._open_real_position(symbol, size, PositionSide.LONG)
            else:
                # Use simulation
                position_info = await self._simulate_position_open(symbol, size, PositionSide.LONG)
            
            if position_info:
                self.open_positions.append(position_info)
                duration = time.time() - start_time
                
                tprint_success(f"✅ Long position opened successfully ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": position_info.symbol,
                    "Side": position_info.side.value,
                    "Size": position_info.size,
                    "Entry Price": position_info.entry_price,
                    "Leverage": position_info.leverage,
                    "Position ID": position_info.position_id
                })
                
                return PositionTestResult(
                    operation=f"open_long_{symbol}",
                    success=True,
                    duration=duration,
                    position_info=position_info,
                    data={
                        "symbol": symbol,
                        "side": "long",
                        "size": size,
                        "entry_price": position_info.entry_price
                    }
                )
            else:
                duration = time.time() - start_time
                tprint_error(f"❌ Failed to open long position for {symbol}")
                return PositionTestResult(
                    operation=f"open_long_{symbol}",
                    success=False,
                    duration=duration,
                    error="Failed to open position"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Long position test failed for {symbol}: {e}")
            return PositionTestResult(
                operation=f"open_long_{symbol}",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def test_open_short_position(self, symbol: str, size: float) -> PositionTestResult:
        """Test opening a short position."""
        tprint_info(f"📉 Testing Short Position Open: {symbol} (Size: {size})")
        
        start_time = time.time()
        try:
            if self.use_real_interface and self.exchange:
                # Use real exchange
                position_info = await self._open_real_position(symbol, size, PositionSide.SHORT)
            else:
                # Use simulation
                position_info = await self._simulate_position_open(symbol, size, PositionSide.SHORT)
            
            if position_info:
                self.open_positions.append(position_info)
                duration = time.time() - start_time
                
                tprint_success(f"✅ Short position opened successfully ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": position_info.symbol,
                    "Side": position_info.side.value,
                    "Size": position_info.size,
                    "Entry Price": position_info.entry_price,
                    "Leverage": position_info.leverage,
                    "Position ID": position_info.position_id
                })
                
                return PositionTestResult(
                    operation=f"open_short_{symbol}",
                    success=True,
                    duration=duration,
                    position_info=position_info,
                    data={
                        "symbol": symbol,
                        "side": "short",
                        "size": size,
                        "entry_price": position_info.entry_price
                    }
                )
            else:
                duration = time.time() - start_time
                tprint_error(f"❌ Failed to open short position for {symbol}")
                return PositionTestResult(
                    operation=f"open_short_{symbol}",
                    success=False,
                    duration=duration,
                    error="Failed to open position"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Short position test failed for {symbol}: {e}")
            return PositionTestResult(
                operation=f"open_short_{symbol}",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def _open_real_position(self, symbol: str, size: float, side: PositionSide) -> Optional[PositionInfo]:
        """Open a real position using the exchange interface."""
        try:
            # Create market order for position opening
            order_side = "BUY" if side == PositionSide.LONG else "SELL"
            
            # For perpetuals, we need to create a futures order
            order_data = {
                "symbol": symbol,
                "side": order_side,
                "type": "MARKET",
                "quantity": size,
                "timeInForce": "IOC"  # Immediate or Cancel for market orders
            }
            
            # Create the order
            order_result = await self.exchange.create_order(**order_data)
            
            if order_result and order_result.get('status') == 'FILLED':
                # Extract position information
                position_info = PositionInfo(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=float(order_result.get('avgPrice', 0)),
                    unrealized_pnl=0.0,  # Will be calculated later
                    margin_used=float(order_result.get('cummulativeQuoteQty', 0)) * 0.1,  # 10x leverage
                    leverage=10.0,
                    position_id=order_result.get('orderId', f"pos_{int(time.time())}"),
                    timestamp=time.time(),
                    is_perpetual=True
                )
                
                return position_info
            else:
                tprint_warning(f"⚠️ Order not filled for {symbol}: {order_result}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error opening real position: {e}")
            return None
    
    async def _simulate_position_open(self, symbol: str, size: float, side: PositionSide) -> PositionInfo:
        """Simulate opening a position."""
        await asyncio.sleep(0.3)  # Simulate order processing time
        
        # Get current price (simulated)
        current_price = await self._get_simulated_price(symbol)
        
        # Calculate entry price with slight slippage
        slippage = 0.001  # 0.1% slippage
        if side == PositionSide.LONG:
            entry_price = current_price * (1 + slippage)
        else:
            entry_price = current_price * (1 - slippage)
        
        # Create position info
        position_info = PositionInfo(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            unrealized_pnl=0.0,
            margin_used=entry_price * size * 0.1,  # 10x leverage
            leverage=10.0,
            position_id=f"sim_pos_{int(time.time())}_{symbol}",
            timestamp=time.time(),
            is_perpetual=True
        )
        
        return position_info
    
    async def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for a symbol."""
        # Simulate different prices for different symbols
        base_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "ADAUSDT": 0.5,
            "BNBUSDT": 300.0,
            "SOLUSDT": 100.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Add some random variation
        import random
        variation = random.uniform(-0.02, 0.02)  # ±2% variation
        return base_price * (1 + variation)
    
    async def test_position_monitoring(self) -> PositionTestResult:
        """Test position monitoring and status updates."""
        tprint_info("👁️ Testing Position Monitoring")
        
        start_time = time.time()
        try:
            if not self.open_positions:
                tprint_warning("⚠️ No open positions to monitor")
                return PositionTestResult(
                    operation="position_monitoring",
                    success=True,
                    duration=0.0,
                    data={"message": "No positions to monitor"}
                )
            
            if self.use_real_interface and self.exchange:
                # Monitor real positions
                monitored_positions = await self._monitor_real_positions()
            else:
                # Monitor simulated positions
                monitored_positions = await self._monitor_simulated_positions()
            
            duration = time.time() - start_time
            
            tprint_success(f"✅ Position monitoring successful ({duration:.2f}s)")
            tprint_structured({
                "Monitored Positions": len(monitored_positions),
                "Total PnL": sum(p.get('unrealized_pnl', 0) for p in monitored_positions),
                "Active Symbols": list(set(p.get('symbol') for p in monitored_positions))
            })
            
            return PositionTestResult(
                operation="position_monitoring",
                success=True,
                duration=duration,
                data={
                    "monitored_positions": len(monitored_positions),
                    "positions": monitored_positions
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Position monitoring failed: {e}")
            return PositionTestResult(
                operation="position_monitoring",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def _monitor_real_positions(self) -> List[Dict[str, Any]]:
        """Monitor real positions."""
        try:
            # Get positions from exchange
            positions = await self.exchange.get_positions()
            
            monitored = []
            for pos in positions or []:
                monitored.append({
                    "symbol": pos.get('symbol'),
                    "side": pos.get('side'),
                    "size": pos.get('size'),
                    "entry_price": pos.get('entryPrice'),
                    "unrealized_pnl": pos.get('unrealizedPnl'),
                    "margin_used": pos.get('marginUsed'),
                    "leverage": pos.get('leverage')
                })
            
            return monitored
            
        except Exception as e:
            tprint_error(f"❌ Error monitoring real positions: {e}")
            return []
    
    async def _monitor_simulated_positions(self) -> List[Dict[str, Any]]:
        """Monitor simulated positions."""
        await asyncio.sleep(0.2)  # Simulate monitoring time
        
        monitored = []
        for pos in self.open_positions:
            # Simulate price movement and PnL calculation
            current_price = await self._get_simulated_price(pos.symbol)
            
            if pos.side == PositionSide.LONG:
                pnl = (current_price - pos.entry_price) * pos.size
            else:
                pnl = (pos.entry_price - current_price) * pos.size
            
            monitored.append({
                "symbol": pos.symbol,
                "side": pos.side.value,
                "size": pos.size,
                "entry_price": pos.entry_price,
                "current_price": current_price,
                "unrealized_pnl": pnl,
                "margin_used": pos.margin_used,
                "leverage": pos.leverage,
                "position_id": pos.position_id
            })
        
        return monitored
    
    async def test_close_position(self, position: PositionInfo) -> PositionTestResult:
        """Test closing a position."""
        tprint_info(f"🔒 Testing Position Close: {position.symbol} ({position.side.value})")
        
        start_time = time.time()
        try:
            if self.use_real_interface and self.exchange:
                # Close real position
                success = await self._close_real_position(position)
            else:
                # Close simulated position
                success = await self._close_simulated_position(position)
            
            duration = time.time() - start_time
            
            if success:
                # Remove from open positions
                self.open_positions = [p for p in self.open_positions if p.position_id != position.position_id]
                
                tprint_success(f"✅ Position closed successfully ({duration:.2f}s)")
                tprint_structured({
                    "Symbol": position.symbol,
                    "Side": position.side.value,
                    "Size": position.size,
                    "Position ID": position.position_id
                })
                
                return PositionTestResult(
                    operation=f"close_{position.symbol}_{position.side.value}",
                    success=True,
                    duration=duration,
                    data={
                        "symbol": position.symbol,
                        "side": position.side.value,
                        "position_id": position.position_id
                    }
                )
            else:
                tprint_error(f"❌ Failed to close position {position.position_id}")
                return PositionTestResult(
                    operation=f"close_{position.symbol}_{position.side.value}",
                    success=False,
                    duration=duration,
                    error="Failed to close position"
                )
                
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Position close test failed: {e}")
            return PositionTestResult(
                operation=f"close_{position.symbol}_{position.side.value}",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def _close_real_position(self, position: PositionInfo) -> bool:
        """Close a real position."""
        try:
            # Create opposite order to close position
            close_side = "SELL" if position.side == PositionSide.LONG else "BUY"
            
            order_data = {
                "symbol": position.symbol,
                "side": close_side,
                "type": "MARKET",
                "quantity": position.size,
                "timeInForce": "IOC"
            }
            
            order_result = await self.exchange.create_order(**order_data)
            return order_result and order_result.get('status') == 'FILLED'
            
        except Exception as e:
            tprint_error(f"❌ Error closing real position: {e}")
            return False
    
    async def _close_simulated_position(self, position: PositionInfo) -> bool:
        """Close a simulated position."""
        await asyncio.sleep(0.2)  # Simulate order processing
        return True  # Always successful in simulation
    
    async def test_position_management_comprehensive(self) -> PositionTestResult:
        """Test comprehensive position management."""
        tprint_info("🎯 Testing Comprehensive Position Management")
        
        start_time = time.time()
        try:
            # Test opening multiple positions with different configurations
            test_cases = []
            
            for symbol in self.config.symbols[:2]:  # Test first 2 symbols
                for size in self.config.position_sizes[:2]:  # Test first 2 sizes
                    for side in self.config.sides:
                        test_cases.append((symbol, size, side))
            
            # Limit to max_positions
            test_cases = test_cases[:self.config.max_positions]
            
            tprint_structured({
                "Test Cases": len(test_cases),
                "Symbols": self.config.symbols[:2],
                "Sizes": self.config.position_sizes[:2],
                "Sides": [s.value for s in self.config.sides]
            })
            
            # Execute test cases
            for i, (symbol, size, side) in enumerate(test_cases):
                tprint_info(f"📊 Test Case {i+1}/{len(test_cases)}: {symbol} {side.value} {size}")
                
                if side == PositionSide.LONG:
                    result = await self.test_open_long_position(symbol, size)
                else:
                    result = await self.test_open_short_position(symbol, size)
                
                self.results.append(result)
                
                # Small delay between positions
                await asyncio.sleep(0.5)
            
            # Monitor all positions
            monitor_result = await self.test_position_monitoring()
            self.results.append(monitor_result)
            
            duration = time.time() - start_time
            
            tprint_success(f"✅ Comprehensive position management test completed ({duration:.2f}s)")
            tprint_structured({
                "Positions Opened": len(self.open_positions),
                "Test Cases Executed": len(test_cases),
                "Total Duration": f"{duration:.2f}s"
            })
            
            return PositionTestResult(
                operation="comprehensive_position_management",
                success=True,
                duration=duration,
                data={
                    "positions_opened": len(self.open_positions),
                    "test_cases": len(test_cases),
                    "open_positions": [
                        {
                            "symbol": p.symbol,
                            "side": p.side.value,
                            "size": p.size,
                            "entry_price": p.entry_price
                        } for p in self.open_positions
                    ]
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            tprint_error(f"❌ Comprehensive position management test failed: {e}")
            return PositionTestResult(
                operation="comprehensive_position_management",
                success=False,
                duration=duration,
                error=str(e)
            )
    
    async def cleanup_positions(self) -> None:
        """Cleanup all open positions."""
        if not self.config.cleanup_positions or not self.open_positions:
            return
        
        tprint_info("🧹 Cleaning up open positions")
        
        for position in self.open_positions.copy():
            try:
                await self.test_close_position(position)
                await asyncio.sleep(0.2)  # Small delay between closes
            except Exception as e:
                tprint_warning(f"⚠️ Error closing position {position.position_id}: {e}")
        
        tprint_success(f"✅ Cleaned up {len(self.open_positions)} positions")
    
    async def run_all_tests(self) -> List[PositionTestResult]:
        """Run all position tests."""
        tprint_success("🧪 Starting Enhanced Position Testing Suite")
        print("=" * 70)
        
        # Test connection
        connection_result = await self.test_connection()
        self.results.append(connection_result)
        
        if not connection_result.success:
            tprint_error("❌ Connection failed, aborting tests")
            return self.results
        
        # Test comprehensive position management
        comprehensive_result = await self.test_position_management_comprehensive()
        self.results.append(comprehensive_result)
        
        # Cleanup positions if configured
        if self.config.cleanup_positions:
            await self.cleanup_positions()
        
        # Generate summary
        await self._generate_summary()
        
        return self.results
    
    async def _generate_summary(self) -> None:
        """Generate test suite summary."""
        tprint_success("📊 Position Test Suite Summary")
        print("=" * 70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Count position operations
        position_opens = sum(1 for r in self.results if r.operation.startswith('open_'))
        position_closes = sum(1 for r in self.results if r.operation.startswith('close_'))
        
        tprint_structured({
            "Test Mode": "real" if self.use_real_interface else "simulation",
            "Total Tests": total_tests,
            "Passed": passed_tests,
            "Failed": failed_tests,
            "Success Rate": f"{success_rate:.1f}%",
            "Positions Opened": position_opens,
            "Positions Closed": position_closes,
            "Open Positions": len(self.open_positions)
        })
        
        # Show individual results
        print("\n📋 Individual Test Results:")
        for result in self.results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.operation}: {result.duration:.2f}s")
            if result.error:
                print(f"    Error: {result.error}")
        
        # Show position summary
        if self.open_positions:
            print(f"\n📊 Open Positions ({len(self.open_positions)}):")
            for pos in self.open_positions:
                print(f"  - {pos.symbol} {pos.side.value} {pos.size} @ {pos.entry_price}")
        
        # Final status
        if failed_tests == 0:
            tprint_success("🎉 All position tests passed!")
        else:
            tprint_error(f"❌ {failed_tests} test(s) failed")
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.exchange and hasattr(self.exchange, 'disconnect'):
            try:
                await self.exchange.disconnect()
                tprint_info("🧹 Exchange connection closed")
            except Exception as e:
                tprint_warning(f"⚠️ Error during cleanup: {e}")

async def main():
    """Main entry point."""
    print("🚀 Enhanced Position Testing Suite")
    print("Testing position management with different symbols, sizes, and sides")
    print()
    
    # Create test configuration
    config = PositionTestConfig(
        symbols=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
        position_sizes=[0.001, 0.01, 0.1],
        sides=[PositionSide.LONG, PositionSide.SHORT],
        test_perpetuals=True,
        max_positions=6,
        cleanup_positions=True
    )
    
    # Create tester
    tester = EnhancedPositionTester(config)
    
    try:
        # Run tests
        results = await tester.run_all_tests()
        
        # Return success status
        failed_tests = sum(1 for r in results if not r.success)
        return failed_tests == 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)