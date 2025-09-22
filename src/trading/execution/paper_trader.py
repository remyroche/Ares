from src.utils.tprint import tprint
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from ..utils.error_handling import (
    ExecutionError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config, validate_market_data

# src/paper_trader.py
"""
PaperTrader for training and testnet trading.
Uses Binance testnet via BinanceExchange for all operations.
"""
from datetime import datetime

# Removed trading_decorators imports - using core decorators instead
from enum import Enum
from typing import Any, Dict, List
from dataclasses import dataclass

# Simple trade tracker stub
@dataclass
class TradeTracker:
    """Simple trade tracker for paper trading."""
    trades: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

def get_trade_tracker() -> TradeTracker:
    """Get a trade tracker instance."""
    return TradeTracker()

# Constants for paper trading
DEFAULT_COMMISSION_RATE = 0.001  # 0.1%
DEFAULT_INITIAL_BALANCE = 10000.0
DEFAULT_MAX_POSITION_SIZE = 0.25  # 25%
DEFAULT_SLIPPAGE_RATE = 0.001  # 0.1%
import logging
import os
import pandas as pd
import time

class ExecutionMode(Enum):
    """Execution mode enumeration."""

    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"
    SIMULATION = "simulation"

class PaperTrader:
    """
    Enhanced paper trader with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize paper trader with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PaperTrader")

        # Trading state
        self.is_trading: bool = False
        self.positions: dict[str, dict[str, Any]] = {}
        self.trade_history: list[dict[str, Any]] = []
        self.balance: float = 10000.0  # Starting balance
        self.equity_history: list[float] = []
        self.prices: dict[str, float] = {}

        # Configuration
        self.trader_config: dict[str, Any] = self.config.get("paper_trader", {})
        self.initial_balance: float = self.trader_config.get(
            "initial_balance",
            DEFAULT_INITIAL_BALANCE,
        )
        self.max_position_size: float = self.trader_config.get(
            "max_position_size",
            DEFAULT_MAX_POSITION_SIZE,
        )
        self.commission_rate: float = self.trader_config.get(
            "commission_rate",
            DEFAULT_COMMISSION_RATE,
        )
        self.slippage_rate: float = self.trader_config.get(
            "slippage_rate",
            DEFAULT_SLIPPAGE_RATE,
        )

        # Trade tracking
        self.trade_tracker = get_trade_tracker()
        
        # Enhanced monitoring integration
        self.enhanced_monitoring: EnhancedMonitoringOrchestrator | None = None

    @trading_error_handler(
        error_types=(ValueError, AttributeError, KeyError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def initialize(self) -> bool:
        """
        Initialize paper trader with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Paper Trader...")

            # Load trader configuration
            await self._load_trader_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for paper trader"))
                return False

            # Initialize trading state
            await self._initialize_trading_state()
            
            # Initialize enhanced monitoring
            await self._initialize_enhanced_monitoring()

            self.logger.info("✅ Paper Trader initialization completed successfully")
            return True

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Paper Trader initialization failed: {e}"),
            )
            return False

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def _load_trader_configuration(self) -> None:
        """Load trader configuration."""
        try:
            # Set default trader parameters
            self.trader_config.setdefault("initial_balance", 10000.0)
            self.trader_config.setdefault("max_position_size", 0.1)
            self.trader_config.setdefault("commission_rate", 0.001)
            self.trader_config.setdefault("slippage_rate", 0.0005)
            self.trader_config.setdefault("enable_risk_management", True)
            self.trader_config.setdefault("max_drawdown", 0.2)

            # Update configuration
            self.initial_balance = self.trader_config["initial_balance"]
            self.max_position_size = self.trader_config["max_position_size"]
            self.commission_rate = self.trader_config["commission_rate"]
            self.slippage_rate = self.trader_config["slippage_rate"]

            self.logger.info("Trader configuration loaded successfully")

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error loading trader configuration: {e}"),
            )

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def _validate_configuration(self) -> bool:
        """
        Validate trader configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate initial balance
            if self.initial_balance <= 0:
                self.logger.error(invalid("Invalid initial balance"))
                return False

            # Validate position size
            if self.max_position_size <= 0 or self.max_position_size > 1:
                self.logger.error(invalid("Invalid max position size"))
                return False

            # Validate commission rate
            if self.commission_rate < 0 or self.commission_rate > 0.1:
                self.logger.error(invalid("Invalid commission rate"))
                return False

            # Validate slippage rate
            if self.slippage_rate < 0 or self.slippage_rate > 0.01:
                self.logger.error(invalid("Invalid slippage rate"))
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error validating configuration: {e}"),
            )
            return False

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def _initialize_trading_state(self) -> None:
        """Initialize trading state."""
        try:
            # Set initial balance
            self.balance = self.initial_balance
            self.equity_history = [self.initial_balance]
            self.prices.clear()

            # Clear positions and history
            self.positions.clear()
            self.trade_history.clear()

            self.logger.info(
                f"Trading state initialized with balance: ${self.balance:,.2f}",
            )

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error initializing trading state: {e}"),
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def _initialize_enhanced_monitoring(self) -> None:
        """Initialize enhanced monitoring system."""
        try:
            self.logger.info("🔍 Initializing Enhanced Monitoring for Paper Trader...")
            
            # Initialize enhanced monitoring orchestrator
            self.enhanced_monitoring = EnhancedMonitoringOrchestrator()
            await self.enhanced_monitoring.initialize()
            
            if self.enhanced_monitoring:
                self.logger.info("✅ Enhanced Monitoring initialized for Paper Trader")
                self.logger.info("   📊 Trade decisions will be automatically captured")
                self.logger.info("   🔍 SHAP/LIME explanations will be generated")
                self.logger.info("   📈 Performance metrics will be tracked")
            else:
                self.logger.warning("⚠️ Failed to initialize Enhanced Monitoring for Paper Trader")
                
        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error initializing enhanced monitoring: {e}"),
            )

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def _record_trade_in_monitoring(self, trade_record: dict[str, Any], side: str) -> None:
        """Record trade in enhanced monitoring system."""
        try:
            if not self.enhanced_monitoring:
                return

            # Create comprehensive trade decision context for monitoring
            trade_decision = {
                'timestamp': trade_record['timestamp'],
                'trading_mode': 'PAPER',
                'exchange': 'BINANCE',  # Default exchange
                'symbol': trade_record['symbol'],
                'price': trade_record['price'],
                'action': side,
                'quantity': trade_record['quantity'],
                'confidence': 0.8,  # Default confidence for paper trades
                'position_size': trade_record['quantity'],
                'leverage': 1.0,  # Default leverage for paper trades
                'trade_metadata': {
                    'trade_id': trade_record['trade_id'],
                    'commission': trade_record.get('commission', 0.0),
                    'slippage': trade_record.get('slippage', 0.0),
                    'balance_after': trade_record.get('balance_after', 0.0),
                    'model_weights': trade_record.get('model_weights', {}),
                    'model_confidences': trade_record.get('model_confidences', {}),
                    'regime_analysis': trade_record.get('regime_analysis', {}),
                    'hmm_regime': trade_record.get('hmm_regime', ''),
                    'support_resistance_levels': trade_record.get('support_resistance_levels', {}),
                    'market_conditions': trade_record.get('market_conditions', {}),
                    'risk_metrics': trade_record.get('risk_metrics', {})
                }
            }

            # Record the trade decision
            await self.enhanced_monitoring.record_comprehensive_trade_decision(trade_decision)
            
            self.logger.info(f"📊 {side} trade recorded in enhanced monitoring system")
            
        except Exception as e:
            self.logger.exception(f"Error recording {side} trade in monitoring: {e}")

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_context: Any = None,
    ) -> bool:
        """
        Execute a buy order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            price: Price per unit
            timestamp: Order timestamp
            trade_context: Optional trade context

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_order(symbol, quantity, price):
                return False

            # Calculate costs
            total_cost = quantity * price
            commission = total_cost * self.commission_rate
            slippage = total_cost * self.slippage_rate
            total_with_fees = total_cost + commission + slippage

            # Check if we have enough balance
            if total_with_fees > self.balance:
                self.logger.warning(
                    f"Insufficient balance for buy order: ${total_with_fees:.2f} > ${self.balance:.2f}",
                )
                return False

            # Execute the trade
            self.balance -= total_with_fees

            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "quantity": 0,
                    "avg_price": 0,
                    "total_cost": 0,
                }

            position = self.positions[symbol]
            old_quantity = position["quantity"]
            old_total_cost = position["total_cost"]

            # Update position
            new_quantity = old_quantity + quantity
            new_total_cost = old_total_cost + total_cost
            new_avg_price = new_total_cost / new_quantity if new_quantity > 0 else 0

            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
            position["total_cost"] = new_total_cost

            # Create trade record with comprehensive tracking data
            trade_id = f"BUY_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            trade_record = {
                "trade_id": trade_id,
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": price,
                "total_cost": total_cost,
                "commission": commission,
                "slippage": slippage,
                "balance_after": self.balance,
                "execution_mode": ExecutionMode.PAPER.value,
                "model_weights": trade_context.model_weights if trade_context else {},
                "model_confidences": (
                    trade_context.model_confidences if trade_context else {}
                ),
                "regime_analysis": (
                    trade_context.regime_analysis if trade_context else {}
                ),
                "hmm_regime": trade_context.hmm_regime if trade_context else "",
                "support_resistance_levels": (
                    trade_context.support_resistance_levels if trade_context else {}
                ),
                "market_conditions": (
                    trade_context.market_conditions if trade_context else {}
                ),
                "risk_metrics": trade_context.risk_metrics if trade_context else {},
            }
            self.trade_history.append(trade_record)

            # Record trade decision in enhanced monitoring system
            await self._record_trade_in_monitoring(trade_record, "BUY")

            self.logger.info(
                f"✅ Buy order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error executing buy order: {e}"))
            return False

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.HIGH,
        raise_on_error=True
    )
    async def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
        trade_context: Any = None,
    ) -> bool:
        """
        Execute a sell order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            price: Price per unit
            timestamp: Order timestamp
            trade_context: Optional trade context

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._validate_order(symbol, quantity, price):
                return False

            # Check if we have enough position
            if (
                symbol not in self.positions
                or self.positions[symbol]["quantity"] < quantity
            ):
                self.logger.warning(
                    f"Insufficient position for sell order: {quantity} > {self.positions.get(symbol, {}).get('quantity', 0)}",
                )
                return False

            # Calculate proceeds
            total_proceeds = quantity * price
            commission = total_proceeds * self.commission_rate
            slippage = total_proceeds * self.slippage_rate
            net_proceeds = total_proceeds - commission - slippage

            # Execute the trade
            self.balance += net_proceeds

            # Update position
            position = self.positions[symbol]
            old_quantity = position["quantity"]
            old_total_cost = position["total_cost"]

            # Update position
            new_quantity = old_quantity - quantity
            if new_quantity > 0:
                # Calculate remaining cost proportionally
                remaining_ratio = new_quantity / old_quantity
                new_total_cost = old_total_cost * remaining_ratio
                new_avg_price = new_total_cost / new_quantity
            else:
                # Position closed
                new_total_cost = 0
                new_avg_price = 0

            position["quantity"] = new_quantity
            position["avg_price"] = new_avg_price
            position["total_cost"] = new_total_cost

            # Remove position if quantity is zero
            if new_quantity <= 0:
                del self.positions[symbol]

            # Create trade record with comprehensive tracking data
            trade_id = f"SELL_{symbol}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            trade_record = {
                "trade_id": trade_id,
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "SELL",
                "quantity": quantity,
                "price": price,
                "total_proceeds": total_proceeds,
                "commission": commission,
                "slippage": slippage,
                "net_proceeds": net_proceeds,
                "balance_after": self.balance,
                "execution_mode": ExecutionMode.PAPER.value,
                "model_weights": trade_context.model_weights if trade_context else {},
                "model_confidences": (
                    trade_context.model_confidences if trade_context else {}
                ),
                "regime_analysis": (
                    trade_context.regime_analysis if trade_context else {}
                ),
                "hmm_regime": trade_context.hmm_regime if trade_context else "",
                "support_resistance_levels": (
                    trade_context.support_resistance_levels if trade_context else {}
                ),
                "market_conditions": (
                    trade_context.market_conditions if trade_context else {}
                ),
                "risk_metrics": trade_context.risk_metrics if trade_context else {},
            }
            self.trade_history.append(trade_record)

            # Record trade decision in enhanced monitoring system
            await self._record_trade_in_monitoring(trade_record, "SELL")

            self.logger.info(
                f"✅ Sell order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error executing sell order: {e}"))
            return False

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def _validate_order(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Validate order parameters.

        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Validate symbol
            if not symbol or len(symbol) == 0:
                self.logger.error(invalid("Invalid symbol"))
                return False

            # Validate quantity
            if quantity <= 0:
                self.logger.error(invalid("Invalid quantity"))
                return False

            # Validate price
            if price <= 0:
                self.logger.error(invalid("Invalid price"))
                return False

            # Check position size limits
            total_value = quantity * price
            max_allowed = self.balance * self.max_position_size

            if total_value > max_allowed:
                self.logger.warning(
                    f"Order exceeds max position size: ${total_value:.2f} > ${max_allowed:.2f}",
                )
                return False

            return True

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error validating order: {e}"))
            return False

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """
        Get current position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Optional[Dict[str, Any]]: Position information or None
        """
        try:
            return self.positions.get(symbol, None)

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error getting position for {symbol}: {e}"),
            )
            return None

    def mark_price(self, symbol: str, price: float) -> None:
        """Update latest price for symbol for mark-to-market accounting."""
        try:
            if price <= 0:
                return
            self.prices[symbol] = price
            self._update_equity()
        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error marking price for {symbol}: {e}"),
            )

    def _update_equity(self) -> None:
        """Recompute total equity using current prices and unrealized PnL."""
        try:
            equity = self.balance
            for sym, pos in self.positions.items():
                qty = pos.get("quantity", 0.0)
                avg = pos.get("avg_price", 0.0)
                mark = self.prices.get(sym, avg)
                if qty > 0 and mark > 0 and avg > 0:
                    equity += qty * (mark - avg)
            self.equity_history.append(equity)
        except Exception as e:
            self.logger.exception(ExecutionError(f"Error updating equity: {e}"))

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def get_all_positions(self) -> dict[str, dict[str, Any]]:
        """
        Get all current positions.

        Returns:
            Dict[str, Dict[str, Any]]: All positions
        """
        try:
            return self.positions.copy()

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error getting all positions: {e}"))
            return {}

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def get_balance(self) -> float:
        """
        Get current balance.

        Returns:
            float: Current balance
        """
        try:
            return self.balance

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error getting balance: {e}"))
            return 0.0

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def get_trade_history(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """
        Get trade history.

        Args:
            symbol: Optional symbol filter

        Returns:
            List[Dict[str, Any]]: Trade history
        """
        try:
            if symbol:
                return [
                    trade for trade in self.trade_history if trade["symbol"] == symbol
                ]
            return self.trade_history.copy()

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error getting trade history: {e}"))
            return []

    @trading_error_handler(
        error_types=(ValueError, AttributeError),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    def calculate_performance(self) -> dict[str, Any]:
        """
        Calculate trading performance metrics.

        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            if not self.trade_history:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                }

            # Calculate basic metrics
            total_trades = len(self.trade_history)
            buy_trades = [t for t in self.trade_history if t["side"] == "BUY"]
            sell_trades = [t for t in self.trade_history if t["side"] == "SELL"]

            # Calculate P&L
            total_buy_cost = sum(t["total_cost"] for t in buy_trades)
            total_sell_proceeds = sum(t.get("net_proceeds", 0.0) for t in sell_trades)
            total_pnl = total_sell_proceeds - total_buy_cost

            # Calculate win rate
            profitable_trades = len(
                [t for t in sell_trades if t.get("net_proceeds", 0.0) > 0],
            )
            win_rate = profitable_trades / len(sell_trades) if sell_trades else 0.0

            # Calculate max drawdown using equity history
            if len(self.equity_history) < 2:
                max_drawdown = 0.0
                sharpe_ratio = 0.0
            else:
                equity_series = self.equity_history
                peak = equity_series[0]
                max_drawdown = 0.0
                returns = []
                for i in range(1, len(equity_series)):
                    eq = equity_series[i]
                    prev = equity_series[i - 1]
                    peak = max(peak, eq)
                    dd = (peak - eq) / peak
                    max_drawdown = max(max_drawdown, dd)
                    ret = (eq - prev) / prev if prev > 0 else 0.0
                    returns.append(ret)
                if returns:
                    avg_return = float(np.mean(returns))
                    std_return = float(np.std(returns))
                    sharpe_ratio = avg_return / std_return if std_return > 0 else 0.0
                else:
                    sharpe_ratio = 0.0

            return {
                "total_trades": total_trades,
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "current_balance": self.balance,
                "current_equity": (
                    self.equity_history[-1] if self.equity_history else self.balance
                ),
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "total_return": (
                    (self.equity_history[-1] - self.initial_balance)
                    / self.initial_balance
                    if self.equity_history
                    else 0.0
                ),
            }

        except Exception as e:
            self.logger.exception(
                ExecutionError(f"Error calculating performance: {e}"),
            )
            return {}

    def get_trader_status(self) -> dict[str, Any]:
        """
        Get paper trader status information.

        Returns:
            Dict[str, Any]: Trader status
        """
        return {
            "is_trading": self.is_trading,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "positions_count": len(self.positions),
            "trades_count": len(self.trade_history),
            "max_position_size": self.max_position_size,
            "commission_rate": self.commission_rate,
            "slippage_rate": self.slippage_rate,
        }

    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.MEDIUM,
        raise_on_error=True
    )
    async def stop(self) -> None:
        """Stop the paper trader."""
        self.logger.info("🛑 Stopping Paper Trader...")

        try:
            # Stop enhanced monitoring
            if self.enhanced_monitoring:
                await self.enhanced_monitoring.stop()
                self.logger.info("🔍 Enhanced monitoring stopped")

            # Close all positions
            if self.positions:
                self.logger.info(f"Closing {len(self.positions)} positions...")
                # Note: In a real implementation, you would close positions at current market prices
                self.positions.clear()

            self.is_trading = False
            self.logger.info("✅ Paper Trader stopped successfully")

        except Exception as e:
            self.logger.exception(ExecutionError(f"Error stopping paper trader: {e}"))

# Global paper trader instance
paper_trader: PaperTrader | None = None

@handles_errors(
    exceptions=(Exception,),
    default_return=None,
    context="paper trader setup",
)
async def setup_paper_trader(
    config: dict[str, Any] | None = None,
) -> PaperTrader | None:
    """
    Setup global paper trader.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[PaperTrader]: Global paper trader instance
    """
    try:
        global paper_trader

        if config is None:
            config = {
                "paper_trader": {
                    "initial_balance": 10000.0,
                    "max_position_size": 0.1,
                    "commission_rate": 0.001,
                    "slippage_rate": 0.0005,
                    "enable_risk_management": True,
                    "max_drawdown": 0.2,
                },
            }

        # Create paper trader
        paper_trader = PaperTrader(config)

        # Initialize paper trader
        success = await paper_trader.initialize()
        if success:
            return paper_trader
        return None

    except Exception as e:
        tprint(f"Error setting up paper trader: {e}")
        return None
