"""
Cross-Asset Trading Manager

This module provides a unified trading manager that coordinates trading across
multiple cryptocurrencies while ensuring only one trade executes at a time.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from src.utils.logger import system_logger
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)

from .trading_orchestrator import (
    TradingOrchestrator, create_trading_orchestrator,
    TradingMode, TradingDecision, TradingSession
)

logger = system_logger.getChild('CrossAssetTradingManager')

class TradeStatus(Enum):
    """Trade execution status."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class CrossAssetTrade:
    """Cross-asset trade information."""
    trade_id: str
    symbol: str
    action: str
    quantity: float
    price: float
    confidence: float
    timestamp: datetime
    status: TradeStatus = TradeStatus.PENDING
    execution_time: Optional[datetime] = None
    actual_pnl: Optional[float] = None
    error_message: Optional[str] = None
    orchestrator_id: Optional[str] = None

@dataclass
class CrossAssetSession:
    """Cross-asset trading session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    orchestrators: Dict[str, TradingOrchestrator] = field(default_factory=dict)
    trade_queue: List[CrossAssetTrade] = field(default_factory=list)
    executed_trades: List[CrossAssetTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    max_drawdown: float = 0.0

class CrossAssetTradingManager:
    """
    Cross-Asset Trading Manager that coordinates trading across multiple cryptocurrencies.

    Features:
    - Manages multiple TradingOrchestrators for different symbols
    - Ensures only one trade executes at a time (trade semaphore)
    - Provides consolidated performance reporting
    - Handles cross-asset risk management
    - Real-time monitoring of all assets
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the cross-asset trading manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('CrossAssetTradingManager')

        # Core components
        self.current_session: Optional[CrossAssetSession] = None
        self.orchestrators: Dict[str, TradingOrchestrator] = {}
        self.symbol_configs: Dict[str, Dict[str, Any]] = {}

        # Trade execution control
        self.trade_semaphore = asyncio.Semaphore(1)  # Only one trade at a time
        self.is_running = False
        self.execution_task: Optional[asyncio.Task] = None

        # Configuration
        self.symbols = config.get('symbols', ['ETHUSDT', 'BTCUSDT'])
        self.primary_symbol = config.get('primary_symbol', 'ETHUSDT')
        self.exchange = config.get('exchange', 'binance')
        self.trading_mode = TradingMode(config.get('trading_mode', 'paper'))
        self.account_balance = config.get('account_balance', 10000.0)

        # Cross-asset parameters
        self.max_concurrent_symbols = config.get('max_concurrent_symbols', 3)
        self.rebalance_interval = config.get('rebalance_interval_minutes', 60)
        self.risk_per_trade = config.get('risk_per_trade', 0.02)

        # Performance tracking
        self.performance_metrics = {
            'total_sessions': 0,
            'total_trades': 0,
            'cross_asset_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'cross_correlation': 0.0
        }

    async def initialize(self) -> bool:
        """
        Initialize all components.

        Returns:
            bool: True if initialization successful
        """
        try:
            tprint_info("🚀 Initializing Cross-Asset Trading Manager...")

            # Create symbol configurations
            await self._create_symbol_configurations()

            # Initialize orchestrators for each symbol
            await self._initialize_orchestrators()

            # Set up cross-asset monitoring
            await self._setup_cross_asset_monitoring()

            tprint_success("✅ Cross-Asset Trading Manager initialized successfully")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to initialize Cross-Asset Trading Manager: {e}")
            return False

    async def _create_symbol_configurations(self):
        """Create configuration for each symbol."""
        try:
            tprint_info("🔄 Creating symbol configurations...")

            # Base configuration template
            base_config = {
                'exchange': self.exchange,
                'trading_mode': self.trading_mode.value,
                'account_balance': self.account_balance / len(self.symbols),  # Split balance
                'analyst': {
                    'confidence_threshold': 0.6
                },
                'tactician': {
                    'confidence_threshold': 0.6,
                    'risk_per_trade': self.risk_per_trade,
                    'max_leverage': 2.0
                },
                'signal_combiner': {
                    'analyst_weight': 0.6,
                    'tactician_weight': 0.4,
                    'confidence_threshold': 0.6
                }
            }

            for symbol in self.symbols:
                # Symbol-specific configuration
                symbol_config = base_config.copy()
                symbol_config.update({
                    'symbol': symbol,
                    'symbol_specific_params': {
                        'volatility_adjustment': self._calculate_volatility_adjustment(symbol),
                        'liquidity_factor': self._calculate_liquidity_factor(symbol),
                        'correlation_factor': self._calculate_correlation_factor(symbol)
                    }
                })

                self.symbol_configs[symbol] = symbol_config
                tprint_info(f"📊 Created config for {symbol}")

            tprint_success(f"✅ Created configurations for {len(self.symbols)} symbols")

        except Exception as e:
            tprint_error(f"❌ Failed to create symbol configurations: {e}")
            raise

    def _calculate_volatility_adjustment(self, symbol: str) -> float:
        """Calculate volatility adjustment for symbol."""
        # This would typically come from historical data
        # For now, return default values based on symbol type
        volatility_adjustments = {
            'BTCUSDT': 1.0,
            'ETHUSDT': 1.2,
            'BNBUSDT': 0.8,
            'ADAUSDT': 1.5,
            'SOLUSDT': 1.3
        }
        return volatility_adjustments.get(symbol, 1.0)

    def _calculate_liquidity_factor(self, symbol: str) -> float:
        """Calculate liquidity factor for symbol."""
        # This would typically come from order book analysis
        liquidity_factors = {
            'BTCUSDT': 1.0,
            'ETHUSDT': 0.9,
            'BNBUSDT': 0.7,
            'ADAUSDT': 0.5,
            'SOLUSDT': 0.6
        }
        return liquidity_factors.get(symbol, 0.5)

    def _calculate_correlation_factor(self, symbol: str) -> float:
        """Calculate correlation factor for symbol."""
        # This would come from correlation analysis
        correlation_factors = {
            'BTCUSDT': 1.0,
            'ETHUSDT': 0.8,
            'BNBUSDT': 0.6,
            'ADAUSDT': 0.4,
            'SOLUSDT': 0.7
        }
        return correlation_factors.get(symbol, 0.5)

    async def _initialize_orchestrators(self):
        """Initialize trading orchestrators for each symbol."""
        try:
            tprint_info("🔄 Initializing trading orchestrators...")

            for symbol, config in self.symbol_configs.items():
                tprint_info(f"🎯 Initializing orchestrator for {symbol}")

                orchestrator = create_trading_orchestrator(config)
                success = await orchestrator.initialize()

                if success:
                    self.orchestrators[symbol] = orchestrator
                    tprint_success(f"✅ Orchestrator initialized for {symbol}")
                else:
                    tprint_error(f"❌ Failed to initialize orchestrator for {symbol}")

            tprint_success(f"✅ Initialized {len(self.orchestrators)} orchestrators")

        except Exception as e:
            tprint_error(f"❌ Failed to initialize orchestrators: {e}")
            raise

    async def _setup_cross_asset_monitoring(self):
        """Set up cross-asset monitoring and data collection."""
        try:
            tprint_info("🔄 Setting up cross-asset monitoring...")

            # This would integrate with enhanced monitoring for cross-asset tracking
            # For now, we create a basic monitoring setup

            tprint_success("✅ Cross-asset monitoring setup complete")

        except Exception as e:
            tprint_error(f"❌ Failed to setup cross-asset monitoring: {e}")
            raise

    async def start_trading_session(self) -> bool:
        """
        Start a new cross-asset trading session.

        Returns:
            bool: True if session started successfully
        """
        try:
            if self.is_running:
                tprint_warning("⚠️ Trading session already running")
                return False

            tprint_info("🚀 Starting cross-asset trading session...")

            # Create new session
            session_id = f"cross_asset_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_session = CrossAssetSession(
                session_id=session_id,
                start_time=datetime.now(),
                symbols=self.symbols,
                orchestrators=self.orchestrators
            )

            # Start all orchestrators
            await self._start_all_orchestrators()

            # Set up cross-asset callbacks
            await self._setup_cross_asset_callbacks()

            # Start trade execution manager
            self.is_running = True
            self.execution_task = asyncio.create_task(self._trade_execution_manager())

            self.performance_metrics['total_sessions'] += 1

            tprint_success(f"✅ Cross-asset trading session {session_id} started")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to start cross-asset trading session: {e}")
            self.is_running = False
            return False

    async def _start_all_orchestrators(self):
        """Start all trading orchestrators."""
        try:
            tprint_info("🔄 Starting all trading orchestrators...")

            for symbol, orchestrator in self.orchestrators.items():
                tprint_info(f"🎯 Starting orchestrator for {symbol}")
                success = await orchestrator.start_trading_session()

                if success:
                    tprint_success(f"✅ Started orchestrator for {symbol}")
                else:
                    tprint_error(f"❌ Failed to start orchestrator for {symbol}")

            tprint_success("✅ All orchestrators started")

        except Exception as e:
            tprint_error(f"❌ Failed to start orchestrators: {e}")
            raise

    async def _setup_cross_asset_callbacks(self):
        """Set up cross-asset callbacks for all orchestrators."""
        try:
            tprint_info("🔄 Setting up cross-asset callbacks...")

            for symbol, orchestrator in self.orchestrators.items():
                # Add trade decision callback
                orchestrator.add_trade_decision_callback(
                    lambda decision, sym=symbol: self._on_trade_decision(decision, sym)
                )

            tprint_success("✅ Cross-asset callbacks setup complete")

        except Exception as e:
            tprint_error(f"❌ Failed to setup cross-asset callbacks: {e}")
            raise

    async def _on_trade_decision(self, decision: TradingDecision, symbol: str):
        """Handle trade decision from orchestrator."""
        try:
            # Create cross-asset trade
            trade = CrossAssetTrade(
                trade_id=f"cross_{decision.metadata.get('trade_id', 'unknown')}",
                symbol=symbol,
                action=decision.action,
                quantity=decision.quantity,
                price=decision.price,
                confidence=decision.confidence,
                timestamp=decision.timestamp,
                orchestrator_id=id(self.orchestrators[symbol])
            )

            # Add to trade queue
            self.current_session.trade_queue.append(trade)

            tprint_info(f"📥 Trade decision queued: {symbol} {decision.action} {decision.quantity:.4f} @ {decision.price:.2f}")

        except Exception as e:
            tprint_error(f"❌ Failed to handle trade decision: {e}")

    async def _trade_execution_manager(self):
        """Manage trade execution with semaphore control."""
        while self.is_running:
            try:
                # Check if there are trades in queue
                if not self.current_session.trade_queue:
                    await asyncio.sleep(1)  # Brief pause
                    continue

                # Get next trade from queue
                trade = self.current_session.trade_queue[0]

                # Check if we can execute this trade
                can_execute = await self._can_execute_trade(trade)

                if can_execute:
                    # Execute the trade
                    async with self.trade_semaphore:  # Ensure only one trade at a time
                        success = await self._execute_cross_asset_trade(trade)

                        if success:
                            # Remove from queue and add to executed
                            self.current_session.trade_queue.pop(0)
                            self.current_session.executed_trades.append(trade)
                            self.performance_metrics['cross_asset_trades'] += 1
                        else:
                            # Mark as failed
                            trade.status = TradeStatus.FAILED
                            trade.error_message = "Execution failed"
                            self.current_session.trade_queue.pop(0)
                            self.current_session.executed_trades.append(trade)
                else:
                    # Cannot execute, wait before checking again
                    await asyncio.sleep(5)

            except Exception as e:
                tprint_error(f"❌ Trade execution manager error: {e}")
                await asyncio.sleep(5)

    async def _can_execute_trade(self, trade: CrossAssetTrade) -> bool:
        """Check if a trade can be executed."""
        try:
            # Check if we're already executing a trade
            if self.trade_semaphore.locked():
                return False

            # Check symbol-specific limits
            symbol_trades_today = len([
                t for t in self.current_session.executed_trades
                if t.symbol == trade.symbol and t.timestamp.date() == datetime.now().date()
            ])

            max_trades_per_symbol = 10  # Configurable limit
            if symbol_trades_today >= max_trades_per_symbol:
                tprint_warning(f"⚠️ Daily trade limit reached for {trade.symbol}")
                return False

            # Check cross-asset risk limits
            if not await self._check_cross_asset_risk_limits(trade):
                return False

            return True

        except Exception as e:
            tprint_error(f"❌ Error checking trade execution: {e}")
            return False

    async def _check_cross_asset_risk_limits(self, trade: CrossAssetTrade) -> bool:
        """Check cross-asset risk limits."""
        try:
            # Check portfolio exposure
            total_exposure = sum([
                t.quantity * t.price
                for t in self.current_session.executed_trades
                if t.status == TradeStatus.COMPLETED
            ])

            max_exposure = self.account_balance * 0.5  # 50% max exposure
            trade_exposure = trade.quantity * trade.price

            if total_exposure + trade_exposure > max_exposure:
                tprint_warning(f"⚠️ Portfolio exposure limit reached")
                return False

            # Check symbol concentration
            symbol_exposure = sum([
                t.quantity * t.price
                for t in self.current_session.executed_trades
                if t.symbol == trade.symbol and t.status == TradeStatus.COMPLETED
            ])

            max_symbol_exposure = self.account_balance * 0.2  # 20% per symbol
            if symbol_exposure + trade_exposure > max_symbol_exposure:
                tprint_warning(f"⚠️ Symbol concentration limit reached for {trade.symbol}")
                return False

            return True

        except Exception as e:
            tprint_error(f"❌ Error checking cross-asset risk: {e}")
            return False

    async def _execute_cross_asset_trade(self, trade: CrossAssetTrade) -> bool:
        """Execute a cross-asset trade."""
        try:
            tprint_info(f"🔄 Executing cross-asset trade: {trade.symbol} {trade.action}")

            trade.status = TradeStatus.EXECUTING
            trade.execution_time = datetime.now()

            # Get the orchestrator for this symbol
            orchestrator = self.orchestrators.get(trade.symbol)
            if not orchestrator:
                raise ValueError(f"No orchestrator found for {trade.symbol}")

            # Find the original trade decision
            # This would typically come from the orchestrator's trading decisions
            # For now, we'll simulate execution

            # Simulate execution delay
            await asyncio.sleep(0.5)

            # Simulate execution success (95% success rate)
            import random
            execution_success = random.random() > 0.05

            if execution_success:
                trade.status = TradeStatus.COMPLETED
                # Simulate PnL calculation
                trade.actual_pnl = (random.random() - 0.5) * trade.quantity * trade.price * 0.1

                # Update session PnL
                self.current_session.total_pnl += trade.actual_pnl

                # Update performance metrics
                self.performance_metrics['total_trades'] += 1
                if trade.actual_pnl > 0:
                    self.performance_metrics['successful_trades'] += 1
                else:
                    self.performance_metrics['failed_trades'] += 1

                tprint_success(f"✅ Cross-asset trade executed: {trade.symbol} {trade.action} PnL: {trade.actual_pnl:+.2f}")
            else:
                trade.status = TradeStatus.FAILED
                trade.error_message = "Simulated execution failure"
                tprint_error(f"❌ Cross-asset trade failed: {trade.symbol} {trade.action}")

            return execution_success

        except Exception as e:
            trade.status = TradeStatus.FAILED
            trade.error_message = str(e)
            tprint_error(f"❌ Cross-asset trade execution error: {e}")
            return False

    async def stop_trading_session(self) -> bool:
        """
        Stop the cross-asset trading session.

        Returns:
            bool: True if session stopped successfully
        """
        try:
            if not self.is_running:
                return True

            tprint_info("🛑 Stopping cross-asset trading session...")

            self.is_running = False

            # Stop all orchestrators
            await self._stop_all_orchestrators()

            # End current session
            if self.current_session:
                self.current_session.end_time = datetime.now()
                self._update_session_metrics()

            # Wait for execution task to complete
            if self.execution_task:
                await self.execution_task

            tprint_success("✅ Cross-asset trading session stopped")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to stop cross-asset trading session: {e}")
            return False

    async def _stop_all_orchestrators(self):
        """Stop all trading orchestrators."""
        try:
            tprint_info("🔄 Stopping all trading orchestrators...")

            for symbol, orchestrator in self.orchestrators.items():
                tprint_info(f"🛑 Stopping orchestrator for {symbol}")
                success = await orchestrator.stop_trading_session()

                if success:
                    tprint_success(f"✅ Stopped orchestrator for {symbol}")
                else:
                    tprint_error(f"❌ Failed to stop orchestrator for {symbol}")

            tprint_success("✅ All orchestrators stopped")

        except Exception as e:
            tprint_error(f"❌ Failed to stop orchestrators: {e}")
            raise

    def _update_session_metrics(self):
        """Update session performance metrics."""
        if not self.current_session:
            return

        # Calculate cross-asset metrics
        session = self.current_session

        # Update performance metrics
        self.performance_metrics['total_pnl'] += session.total_pnl

        # Calculate max drawdown
        if session.total_pnl < 0:
            self.performance_metrics['max_drawdown'] = min(
                self.performance_metrics['max_drawdown'],
                session.total_pnl
            )

        # Calculate cross-correlation (simplified)
        if len(session.executed_trades) > 1:
            # This would be calculated from actual price data
            self.performance_metrics['cross_correlation'] = 0.6

    def get_manager_stats(self) -> Dict[str, Any]:
        """Get cross-asset trading manager statistics."""
        return {
            'is_running': self.is_running,
            'current_session': {
                'session_id': self.current_session.session_id if self.current_session else None,
                'start_time': self.current_session.start_time.isoformat() if self.current_session else None,
                'symbols': self.symbols,
                'total_pnl': self.current_session.total_pnl if self.current_session else 0.0,
                'trades_in_queue': len(self.current_session.trade_queue) if self.current_session else 0,
                'executed_trades': len(self.current_session.executed_trades) if self.current_session else 0
            },
            'performance_metrics': self.performance_metrics,
            'orchestrator_stats': {
                symbol: orchestrator.get_orchestrator_stats()
                for symbol, orchestrator in self.orchestrators.items()
            }
        }

    async def generate_consolidated_report(self, report_type: str = "session") -> Dict[str, Any]:
        """Generate consolidated cross-asset performance report."""
        try:
            tprint_info(f"📊 Generating consolidated {report_type} report...")

            report = {
                'report_type': report_type,
                'session_info': {
                    'session_id': self.current_session.session_id if self.current_session else None,
                    'start_time': self.current_session.start_time.isoformat() if self.current_session else None,
                    'end_time': self.current_session.end_time.isoformat() if self.current_session and self.current_session.end_time else None,
                    'symbols_traded': self.symbols,
                    'total_symbols': len(self.symbols)
                },
                'cross_asset_metrics': {
                    'total_trades': self.performance_metrics['cross_asset_trades'],
                    'total_pnl': self.performance_metrics['total_pnl'],
                    'success_rate': (
                        self.performance_metrics['successful_trades'] / self.performance_metrics['total_trades']
                        if self.performance_metrics['total_trades'] > 0 else 0.0
                    ),
                    'max_drawdown': self.performance_metrics['max_drawdown'],
                    'cross_correlation': self.performance_metrics['cross_correlation']
                },
                'symbol_performance': {}
            }

            # Add symbol-specific performance
            for symbol, orchestrator in self.orchestrators.items():
                symbol_stats = orchestrator.get_orchestrator_stats()
                report['symbol_performance'][symbol] = {
                    'status': symbol_stats.get('status', 'unknown'),
                    'total_trades': symbol_stats.get('performance_metrics', {}).get('total_trades', 0),
                    'total_pnl': symbol_stats.get('performance_metrics', {}).get('total_pnl', 0.0),
                    'session_info': symbol_stats.get('current_session')
                }

            # Add executed trades
            report['executed_trades'] = [
                {
                    'trade_id': trade.trade_id,
                    'symbol': trade.symbol,
                    'action': trade.action,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'confidence': trade.confidence,
                    'status': trade.status.value,
                    'actual_pnl': trade.actual_pnl,
                    'execution_time': trade.execution_time.isoformat() if trade.execution_time else None
                }
                for trade in self.current_session.executed_trades if self.current_session
            ]

            tprint_success(f"✅ Generated consolidated {report_type} report")
            return report

        except Exception as e:
            tprint_error(f"❌ Failed to generate consolidated report: {e}")
            return {}

    async def export_consolidated_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> bool:
        """Export consolidated report to file."""
        try:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"cross_asset_report_{timestamp}.json"

            filepath = Path('cross_asset_reports') / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            tprint_success(f"💾 Consolidated report exported to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"❌ Failed to export consolidated report: {e}")
            return False

# Convenience functions

def create_cross_asset_trading_manager(config: Dict[str, Any]) -> CrossAssetTradingManager:
    """Create a configured cross-asset trading manager."""
    return CrossAssetTradingManager(config)

async def start_cross_asset_trading(
    symbols: List[str],
    exchange: str = "binance",
    trading_mode: str = "paper",
    account_balance: float = 10000.0
) -> CrossAssetTradingManager:
    """Start cross-asset trading with default settings."""

    config = {
        'symbols': symbols,
        'exchange': exchange,
        'trading_mode': trading_mode,
        'account_balance': account_balance,
        'max_concurrent_symbols': min(len(symbols), 5),
        'rebalance_interval_minutes': 60,
        'risk_per_trade': 0.02
    }

    manager = create_cross_asset_trading_manager(config)

    # Initialize manager
    success = await manager.initialize()
    if not success:
        raise RuntimeError("Failed to initialize cross-asset trading manager")

    # Start trading session
    success = await manager.start_trading_session()
    if not success:
        raise RuntimeError("Failed to start cross-asset trading session")

    return manager

# Example usage
if __name__ == "__main__":
    async def main():
        """Example main function."""
        symbols = ['ETHUSDT', 'BTCUSDT', 'ADAUSDT']

        config = {
            'symbols': symbols,
            'exchange': 'binance',
            'trading_mode': 'paper',
            'account_balance': 10000.0
        }

        manager = create_cross_asset_trading_manager(config)
        success = await manager.initialize()

        if success:
            success = await manager.start_trading_session()

            if success:
                # Run for 10 minutes
                await asyncio.sleep(600)

                # Generate consolidated report
                report = await manager.generate_consolidated_report()
                await manager.export_consolidated_report(report)

                # Get stats
                stats = manager.get_manager_stats()
                tprint_structured(stats, LogLevel.INFO)

                # Stop manager
                await manager.stop_trading_session()

    # Run example
    asyncio.run(main())