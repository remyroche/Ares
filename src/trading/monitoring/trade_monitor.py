"""
Trade Monitor

This module provides real-time monitoring of trade execution,
performance tracking, and alert management for the trading system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
# comprehensive_trade_monitor import removed to avoid circular imports

logger = system_logger.getChild('TradeMonitor')

class TradeStatus(Enum):
    """Trade status enumeration."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class AlertLevel(Enum):
    """Alert level enumeration."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Trade:
    """Trade information."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    status: TradeStatus
    order_id: Optional[str] = None
    fees: float = 0.0
    pnl: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alert information."""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    trade_id: Optional[str] = None
    symbol: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    current_drawdown: float = 0.0
    peak_balance: float = 0.0
    current_balance: float = 0.0

class TradeMonitor:
    """
    Trade Monitor for real-time trade execution monitoring and performance tracking.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trade monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger.getChild('TradeMonitor')

        # Trade tracking
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.max_history = config.get('max_history', 10000)

        # Performance tracking
        self.performance_metrics = PerformanceMetrics()
        self.daily_pnl: List[float] = []
        self.balance_history: List[Dict[str, Any]] = []

        # Alert system
        self.alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.max_alerts = config.get('max_alerts', 1000)

        # Monitoring parameters
        self.alert_thresholds = {
            'max_drawdown': config.get('max_drawdown_alert', 0.05),  # 5%
            'min_win_rate': config.get('min_win_rate_alert', 0.4),   # 40%
            'max_loss_per_trade': config.get('max_loss_per_trade_alert', 0.02),  # 2%
            'max_daily_loss': config.get('max_daily_loss_alert', 0.05)  # 5%
        }

        # State
        self.is_monitoring = False
        self.monitoring_start_time: Optional[datetime] = None

    async def initialize(self) -> bool:
        """
        Initialize the trade monitor.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("✅ Trade Monitor initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Trade Monitor: {e}")
            return False

    async def start_monitoring(self) -> bool:
        """
        Start trade monitoring.

        Returns:
            bool: True if monitoring started successfully
        """
        try:
            if self.is_monitoring:
                tprint_warning("⚠️ Trade monitoring already running")
                return False

            self.is_monitoring = True
            self.monitoring_start_time = datetime.now()

            # Start monitoring loop
            asyncio.create_task(self._monitoring_loop())

            tprint_success("✅ Trade monitoring started")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to start trade monitoring: {e}")
            self.is_monitoring = False
            return False

    async def stop_monitoring(self) -> bool:
        """
        Stop trade monitoring.

        Returns:
            bool: True if monitoring stopped successfully
        """
        try:
            if not self.is_monitoring:
                return True

            self.is_monitoring = False
            tprint_success("✅ Trade monitoring stopped")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to stop trade monitoring: {e}")
            return False

    @handles_errors
    @traced(span_name="trade_monitoring")
    async def record_trade(self, trade: Trade) -> bool:
        """
        Record a new trade.

        Args:
            trade: Trade to record

        Returns:
            bool: True if trade recorded successfully
        """
        try:
            # Add to active trades if not completed
            if trade.status in [TradeStatus.PENDING, TradeStatus.PARTIALLY_FILLED]:
                self.active_trades[trade.trade_id] = trade

            # Add to history
            self.trade_history.append(trade)

            # Maintain history size
            if len(self.trade_history) > self.max_history:
                self.trade_history.pop(0)

            # Update performance metrics
            await self._update_performance_metrics(trade)

            # Check for alerts
            await self._check_trade_alerts(trade)

            tprint_info(f"📊 Trade recorded: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to record trade: {e}")
            return False

    async def update_trade_status(self, trade_id: str, status: TradeStatus, **kwargs) -> bool:
        """
        Update trade status.

        Args:
            trade_id: Trade ID to update
            status: New status
            **kwargs: Additional trade information

        Returns:
            bool: True if update successful
        """
        try:
            if trade_id in self.active_trades:
                trade = self.active_trades[trade_id]
                trade.status = status

                # Update additional fields
                for key, value in kwargs.items():
                    if hasattr(trade, key):
                        setattr(trade, key, value)

                # Remove from active trades if completed
                if status in [TradeStatus.FILLED, TradeStatus.CANCELLED, TradeStatus.REJECTED, TradeStatus.EXPIRED]:
                    del self.active_trades[trade_id]

                # Update performance metrics
                await self._update_performance_metrics(trade)

                # Check for alerts
                await self._check_trade_alerts(trade)

                tprint_info(f"📊 Trade status updated: {trade_id} -> {status.value}")
                return True
            else:
                tprint_warning(f"⚠️ Trade not found: {trade_id}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Failed to update trade status: {e}")
            return False

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Check for stale trades
                await self._check_stale_trades()

                # Update performance metrics
                await self._update_daily_metrics()

                # Check system alerts
                await self._check_system_alerts()

                # Brief pause
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                self.logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(5)

    async def _update_performance_metrics(self, trade: Trade):
        """Update performance metrics based on trade."""
        try:
            # Update basic metrics
            self.performance_metrics.total_trades += 1
            self.performance_metrics.total_pnl += trade.pnl
            self.performance_metrics.total_fees += trade.fees

            # Update win/loss counts
            if trade.pnl > 0:
                self.performance_metrics.winning_trades += 1
            elif trade.pnl < 0:
                self.performance_metrics.losing_trades += 1

            # Calculate derived metrics
            if self.performance_metrics.total_trades > 0:
                self.performance_metrics.win_rate = (
                    self.performance_metrics.winning_trades / self.performance_metrics.total_trades
                )

            # Calculate average win/loss
            if self.performance_metrics.winning_trades > 0:
                winning_trades = [t for t in self.trade_history if t.pnl > 0]
                self.performance_metrics.avg_win = np.mean([t.pnl for t in winning_trades])

            if self.performance_metrics.losing_trades > 0:
                losing_trades = [t for t in self.trade_history if t.pnl < 0]
                self.performance_metrics.avg_loss = abs(np.mean([t.pnl for t in losing_trades]))

            # Calculate profit factor
            if self.performance_metrics.avg_loss > 0:
                self.performance_metrics.profit_factor = (
                    self.performance_metrics.avg_win / self.performance_metrics.avg_loss
                )

            # Update balance
            self.performance_metrics.current_balance += trade.pnl - trade.fees

            # Update peak balance and drawdown
            if self.performance_metrics.current_balance > self.performance_metrics.peak_balance:
                self.performance_metrics.peak_balance = self.performance_metrics.current_balance
                self.performance_metrics.current_drawdown = 0.0
            else:
                self.performance_metrics.current_drawdown = (
                    (self.performance_metrics.peak_balance - self.performance_metrics.current_balance)
                    / self.performance_metrics.peak_balance
                )

                if self.performance_metrics.current_drawdown > self.performance_metrics.max_drawdown:
                    self.performance_metrics.max_drawdown = self.performance_metrics.current_drawdown

        except Exception as e:
            self.logger.error(f"❌ Performance metrics update failed: {e}")

    async def _check_trade_alerts(self, trade: Trade):
        """Check for trade-specific alerts."""
        try:
            # Check for large loss
            if trade.pnl < -self.alert_thresholds['max_loss_per_trade']:
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    message=f"Large loss on trade {trade.trade_id}: {trade.pnl:.4f}",
                    trade_id=trade.trade_id,
                    symbol=trade.symbol
                )

            # Check for unusual trade size
            if trade.quantity > 1000:  # Example threshold
                await self._create_alert(
                    level=AlertLevel.INFO,
                    message=f"Large trade size: {trade.quantity} {trade.symbol}",
                    trade_id=trade.trade_id,
                    symbol=trade.symbol
                )

        except Exception as e:
            self.logger.error(f"❌ Trade alert check failed: {e}")

    async def _check_system_alerts(self):
        """Check for system-wide alerts."""
        try:
            # Check drawdown
            if self.performance_metrics.current_drawdown > self.alert_thresholds['max_drawdown']:
                await self._create_alert(
                    level=AlertLevel.CRITICAL,
                    message=f"High drawdown: {self.performance_metrics.current_drawdown:.2%}"
                )

            # Check win rate
            if (self.performance_metrics.total_trades > 10 and
                self.performance_metrics.win_rate < self.alert_thresholds['min_win_rate']):
                await self._create_alert(
                    level=AlertLevel.WARNING,
                    message=f"Low win rate: {self.performance_metrics.win_rate:.2%}"
                )

            # Check daily loss
            if self.daily_pnl:
                daily_loss = sum(self.daily_pnl[-24:])  # Last 24 hours
                if daily_loss < -self.alert_thresholds['max_daily_loss']:
                    await self._create_alert(
                        level=AlertLevel.ERROR,
                        message=f"High daily loss: {daily_loss:.4f}"
                    )

        except Exception as e:
            self.logger.error(f"❌ System alert check failed: {e}")

    async def _check_stale_trades(self):
        """Check for stale trades that need attention."""
        try:
            current_time = datetime.now()
            stale_threshold = timedelta(minutes=30)  # 30 minutes

            for trade_id, trade in list(self.active_trades.items()):
                if current_time - trade.timestamp > stale_threshold:
                    await self._create_alert(
                        level=AlertLevel.WARNING,
                        message=f"Stale trade: {trade_id} pending for {current_time - trade.timestamp}",
                        trade_id=trade_id,
                        symbol=trade.symbol
                    )

        except Exception as e:
            self.logger.error(f"❌ Stale trade check failed: {e}")

    async def _update_daily_metrics(self):
        """Update daily performance metrics."""
        try:
            current_time = datetime.now()

            # Add current PnL to daily tracking
            if self.performance_metrics.current_balance > 0:
                self.daily_pnl.append(self.performance_metrics.current_balance)

                # Keep only last 7 days of data
                if len(self.daily_pnl) > 168:  # 7 days * 24 hours
                    self.daily_pnl.pop(0)

            # Update balance history
            self.balance_history.append({
                'timestamp': current_time,
                'balance': self.performance_metrics.current_balance,
                'drawdown': self.performance_metrics.current_drawdown
            })

            # Keep only last 24 hours of balance history
            if len(self.balance_history) > 1440:  # 24 hours * 60 minutes
                self.balance_history.pop(0)

        except Exception as e:
            self.logger.error(f"❌ Daily metrics update failed: {e}")

    async def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        trade_id: Optional[str] = None,
        symbol: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Create and process an alert."""
        try:
            alert = Alert(
                alert_id=f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                level=level,
                message=message,
                timestamp=datetime.now(),
                trade_id=trade_id,
                symbol=symbol,
                metadata=metadata or {}
            )

            # Add to alerts list
            self.alerts.append(alert)

            # Maintain alerts size
            if len(self.alerts) > self.max_alerts:
                self.alerts.pop(0)

            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    self.logger.error(f"❌ Alert callback failed: {e}")

            # Log alert
            log_level = {
                AlertLevel.INFO: self.logger.info,
                AlertLevel.WARNING: self.logger.warning,
                AlertLevel.ERROR: self.logger.error,
                AlertLevel.CRITICAL: self.logger.critical
            }

            log_level[level](f"🚨 {level.value.upper()}: {message}")

        except Exception as e:
            self.logger.error(f"❌ Alert creation failed: {e}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert callback."""
        self.alert_callbacks.append(callback)

    def get_trade_history(self, n: int = 100) -> List[Trade]:
        """Get recent trade history."""
        return self.trade_history[-n:] if len(self.trade_history) >= n else self.trade_history.copy()

    def get_active_trades(self) -> Dict[str, Trade]:
        """Get active trades."""
        return self.active_trades.copy()

    def get_alerts(self, n: int = 100) -> List[Alert]:
        """Get recent alerts."""
        return self.alerts[-n:] if len(self.alerts) >= n else self.alerts.copy()

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self.performance_metrics

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'is_monitoring': self.is_monitoring,
            'monitoring_start_time': self.monitoring_start_time,
            'active_trades_count': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'total_alerts': len(self.alerts),
            'performance_metrics': self.performance_metrics.__dict__,
            'uptime_seconds': (
                (datetime.now() - self.monitoring_start_time).total_seconds()
                if self.monitoring_start_time else 0
            )
        }

# Convenience functions

def create_trade_monitor(config: Dict[str, Any]) -> TradeMonitor:
    """Create a configured trade monitor."""
    return TradeMonitor(config)

async def start_trade_monitoring(
    trade_monitor: TradeMonitor,
    alert_callback: Optional[Callable[[Alert], None]] = None
) -> bool:
    """Start trade monitoring with optional alert callback."""
    if alert_callback:
        trade_monitor.add_alert_callback(alert_callback)

    return await trade_monitor.start_monitoring()
