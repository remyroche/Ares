#!/usr/bin/env python3
"""
Trade Launcher

Provides a unified launcher for different trading modes:
- TRADE mode: Live trading with real exchange orders
- PAPER mode: Paper trading with simulated orders and positions

This launcher integrates with ExchangeInterface to provide seamless
switching between live and paper trading modes.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Union
from enum import Enum

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)

# Import ExchangeInterface
from src.trading.execution.exchange_interface import ExchangeInterface, create_exchange_interface

class TradingMode(Enum):
    """Trading mode enumeration."""
    TRADE = "trade"      # Live trading with real orders
    PAPER = "paper"      # Paper trading with simulated orders

class TradeLauncher:
    """
    Trade launcher that supports both live and paper trading modes.
    
    In TRADE mode:
    - ExchangeInterface works normally with real exchange orders
    - All order operations (open/close positions) are executed on real exchanges
    
    In PAPER mode:
    - ExchangeInterface fetches real market data normally
    - Order operations are redirected to a simulator
    - Simulator tracks positions, calculates P&L, and handles slippage/fees
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade launcher.
        
        Args:
            config: Configuration dictionary containing:
                - trading_mode: "trade" or "paper"
                - exchange_config: Exchange configuration
                - paper_trading_config: Paper trading specific settings
        """
        self.config = config
        self.logger = system_logger.getChild('TradeLauncher')
        
        # Trading mode
        self.trading_mode = TradingMode(config.get('trading_mode', 'paper'))
        
        # Exchange interface
        self.exchange_interface: Optional[ExchangeInterface] = None
        
        # Paper trading simulator (only used in PAPER mode)
        self.paper_simulator: Optional['PaperTradingSimulator'] = None
        
        # Launcher state
        self.is_initialized = False
        self.is_running = False
        
        # Configuration
        self.exchange_config = config.get('exchange_config', {})
        self.paper_trading_config = config.get('paper_trading_config', {})
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        self.logger.info(f"TradeLauncher initialized in {self.trading_mode.value} mode")
    
    @handles_errors(default_return=False)
    @traced
    @log_execution_time
    async def initialize(self) -> bool:
        """
        Initialize trade launcher and exchange interface.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info(f"Initializing TradeLauncher in {self.trading_mode.value} mode...")
            
            # Initialize exchange interface
            await self._initialize_exchange_interface()
            
            # Initialize paper trading simulator if in PAPER mode
            if self.trading_mode == TradingMode.PAPER:
                await self._initialize_paper_simulator()
            
            self.is_initialized = True
            self.logger.info(f"✅ TradeLauncher initialized successfully in {self.trading_mode.value} mode")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize TradeLauncher: {e}")
            return False
    
    @handles_errors(default_return=None)
    async def _initialize_exchange_interface(self) -> None:
        """Initialize exchange interface."""
        try:
            # Create exchange interface with appropriate configuration
            exchange_config = self.exchange_config.copy()
            
            # Add trading mode to exchange config for internal use
            exchange_config['trading_mode'] = self.trading_mode.value
            
            self.exchange_interface = create_exchange_interface(exchange_config)
            
            # Connect to exchange
            connected = await self.exchange_interface.connect()
            if not connected:
                raise Exception("Failed to connect to exchange")
            
            self.logger.info("✅ Exchange interface initialized and connected")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize exchange interface: {e}")
            raise
    
    @handles_errors(default_return=None)
    async def _initialize_paper_simulator(self) -> None:
        """Initialize paper trading simulator."""
        try:
            from src.trading.simulation.paper_trading_simulator import PaperTradingSimulator
            
            # Create simulator with configuration
            simulator_config = self.paper_trading_config.copy()
            simulator_config['exchange_interface'] = self.exchange_interface
            
            self.paper_simulator = PaperTradingSimulator(simulator_config)
            await self.paper_simulator.initialize()
            
            self.logger.info("✅ Paper trading simulator initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize paper simulator: {e}")
            raise
    
    @handles_errors(default_return=False)
    @traced
    async def start(self) -> bool:
        """
        Start the trade launcher.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        try:
            if not self.is_initialized:
                self.logger.error("❌ Launcher not initialized")
                return False
            
            self.is_running = True
            self.start_time = datetime.now()
            
            if self.trading_mode == TradingMode.PAPER:
                await self.paper_simulator.start()
                tprint_success("🚀 Paper trading launcher started")
            else:
                tprint_success("🚀 Live trading launcher started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start launcher: {e}")
            return False
    
    @handles_errors(default_return=False)
    @traced
    async def stop(self) -> bool:
        """
        Stop the trade launcher.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running:
                self.logger.warning("⚠️ Launcher not running")
                return True
            
            # Stop paper simulator if running
            if self.trading_mode == TradingMode.PAPER and self.paper_simulator:
                await self.paper_simulator.stop()
            
            # Disconnect from exchange
            if self.exchange_interface:
                await self.exchange_interface.disconnect()
            
            self.is_running = False
            
            # Generate final report
            await self._generate_final_report()
            
            tprint_success("✅ Trade launcher stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to stop launcher: {e}")
            return False
    
    @handles_errors(default_return=False)
    @traced
    async def execute_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute an order through the appropriate channel.
        
        In TRADE mode: Executes real orders on the exchange
        In PAPER mode: Simulates order execution with realistic pricing
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', etc.)
            quantity: Order quantity
            price: Order price (for limit orders)
            **kwargs: Additional order parameters
            
        Returns:
            Dict containing order execution result
        """
        try:
            if not self.is_running:
                raise Exception("Launcher not running")
            
            if self.trading_mode == TradingMode.TRADE:
                # Live trading: execute real orders
                result = await self.exchange_interface.create_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )
                
                if result and 'error' not in result:
                    self.successful_trades += 1
                    tprint_success(f"✅ Live order executed: {side} {quantity} {symbol}")
                else:
                    self.failed_trades += 1
                    tprint_error(f"❌ Live order failed: {result.get('error', 'Unknown error')}")
                
            elif self.trading_mode == TradingMode.PAPER:
                # Paper trading: simulate order execution
                if not self.paper_simulator:
                    raise Exception("Paper simulator not available")
                
                result = await self.paper_simulator.execute_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )
                
                if result and 'error' not in result:
                    self.successful_trades += 1
                    tprint_success(f"📄 Paper order executed: {side} {quantity} {symbol}")
                else:
                    self.failed_trades += 1
                    tprint_error(f"❌ Paper order failed: {result.get('error', 'Unknown error')}")
            
            self.total_trades += 1
            return result or {}
            
        except Exception as e:
            self.failed_trades += 1
            self.logger.error(f"❌ Order execution failed: {e}")
            return {'error': str(e)}
    
    @handles_errors(default_return={})
    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        try:
            if self.trading_mode == TradingMode.PAPER and self.paper_simulator:
                return await self.paper_simulator.get_balance(asset)
            else:
                return await self.exchange_interface.get_account_balance(asset)
        except Exception as e:
            self.logger.error(f"❌ Failed to get account balance: {e}")
            return {}
    
    @handles_errors(default_return=[])
    async def get_positions(self, symbol: Optional[str] = None) -> list:
        """Get current positions."""
        try:
            if self.trading_mode == TradingMode.PAPER and self.paper_simulator:
                return await self.paper_simulator.get_positions(symbol)
            else:
                # For live trading, this would need to be implemented in ExchangeInterface
                return []
        except Exception as e:
            self.logger.error(f"❌ Failed to get positions: {e}")
            return []
    
    @handles_errors(default_return={})
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            metrics = {
                'trading_mode': self.trading_mode.value,
                'is_running': self.is_running,
                'total_trades': self.total_trades,
                'successful_trades': self.successful_trades,
                'failed_trades': self.failed_trades,
                'success_rate': (self.successful_trades / max(self.total_trades, 1)) * 100,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            }
            
            # Add mode-specific metrics
            if self.trading_mode == TradingMode.PAPER and self.paper_simulator:
                paper_metrics = await self.paper_simulator.get_performance_metrics()
                metrics.update(paper_metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get performance metrics: {e}")
            return {}
    
    @handles_errors(default_return=None)
    async def _generate_final_report(self) -> None:
        """Generate final performance report."""
        try:
            metrics = await self.get_performance_metrics()
            
            tprint_structured("📊 Final Performance Report", {
                'Trading Mode': metrics.get('trading_mode', 'unknown'),
                'Total Trades': metrics.get('total_trades', 0),
                'Successful Trades': metrics.get('successful_trades', 0),
                'Failed Trades': metrics.get('failed_trades', 0),
                'Success Rate': f"{metrics.get('success_rate', 0):.2f}%",
                'Uptime': f"{metrics.get('uptime_seconds', 0):.2f} seconds"
            })
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate final report: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get launcher status."""
        return {
            'trading_mode': self.trading_mode.value,
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'exchange_connected': self.exchange_interface.is_connected() if self.exchange_interface else False,
            'paper_simulator_active': self.paper_simulator.is_running if self.paper_simulator else False,
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'failed_trades': self.failed_trades
        }

# Factory function for creating trade launcher
def create_trade_launcher(config: Dict[str, Any]) -> TradeLauncher:
    """
    Create trade launcher instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TradeLauncher: Configured launcher instance
    """
    return TradeLauncher(config)

# Convenience functions for different modes
def create_paper_trading_launcher(exchange_config: Dict[str, Any], paper_config: Dict[str, Any] = None) -> TradeLauncher:
    """Create paper trading launcher."""
    config = {
        'trading_mode': 'paper',
        'exchange_config': exchange_config,
        'paper_trading_config': paper_config or {}
    }
    return create_trade_launcher(config)

def create_live_trading_launcher(exchange_config: Dict[str, Any]) -> TradeLauncher:
    """Create live trading launcher."""
    config = {
        'trading_mode': 'trade',
        'exchange_config': exchange_config
    }
    return create_trade_launcher(config)