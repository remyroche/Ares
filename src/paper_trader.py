# src/paper_trader.py
"""
PaperTrader for training and testnet trading.
Uses Binance testnet via BinanceExchange for all operations.
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.config.constants import (
    DEFAULT_COMMISSION_RATE,
    DEFAULT_INITIAL_BALANCE,
    DEFAULT_MAX_POSITION_SIZE,
    DEFAULT_SLIPPAGE_RATE,
)
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.trading_decorators import (
    ExecutionMode,
    comprehensive_trading_decorator,
    get_trade_tracker,
)
from src.utils.warning_symbols import (
    execution_error,
    initialization_error,
    invalid,
    validation_error,
)


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
            DEFAULT_INITIAL_BALANCE
        )
        self.max_position_size: float = self.trader_config.get(
            "max_position_size",
            DEFAULT_MAX_POSITION_SIZE
        )
        self.commission_rate: float = self.trader_config.get(
            "commission_rate",
            DEFAULT_COMMISSION_RATE
        )
        self.slippage_rate: float = self.trader_config.get(
            "slippage_rate",
            DEFAULT_SLIPPAGE_RATE
        )

        # Trade tracking
        self.trade_tracker = get_trade_tracker()

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid paper trader configuration"),
            AttributeError: (False, "Missing required trader parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False, context="paper trader initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="trader configuration loading",
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
                initialization_error(f"Error loading trader configuration: {e}"),
            )

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False, context="configuration validation",
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
                validation_error(f"Error validating configuration: {e}"),
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="trading state initialization",
    )
    @comprehensive_trading_decorator(
        enable_error_handling=True,
        enable_performance_monitoring=True,
        enable_trade_logging=True,
        enable_risk_management=True,
        enable_regime_awareness=True,
        max_drawdown=0.2,
        max_position_size=0.1,
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
                "model_confidences": trade_context.model_confidences
                if trade_context
                else {},
                "regime_analysis": trade_context.regime_analysis
                if trade_context
                else {},
                "hmm_regime": trade_context.hmm_regime if trade_context else "",
                "support_resistance_levels": trade_context.support_resistance_levels
                if trade_context
                else {},
                "market_conditions": trade_context.market_conditions
                if trade_context
                else {},
                "risk_metrics": trade_context.risk_metrics if trade_context else {},
            }
            self.trade_history.append(trade_record)

            self.logger.info(
                f"✅ Buy order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.exception(execution_error(f"Error executing buy order: {e}"))
            return False

    @comprehensive_trading_decorator(
        enable_error_handling=True,
        enable_performance_monitoring=True,
        enable_trade_logging=True,
        enable_risk_management=True,
        enable_regime_awareness=True,
        max_drawdown=0.2,
        max_position_size=0.1,
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
                "model_confidences": trade_context.model_confidences
                if trade_context
                else {},
                "regime_analysis": trade_context.regime_analysis
                if trade_context
                else {},
                "hmm_regime": trade_context.hmm_regime if trade_context else "",
                "support_resistance_levels": trade_context.support_resistance_levels
                if trade_context
                else {},
                "market_conditions": trade_context.market_conditions
                if trade_context
                else {},
                "risk_metrics": trade_context.risk_metrics if trade_context else {},
            }
            self.trade_history.append(trade_record)

            self.logger.info(
                f"✅ Sell order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.exception(execution_error(f"Error executing sell order: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False, context="order validation",
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
            self.logger.exception(validation_error(f"Error validating order: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="position getting",
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
            self.logger.exception(execution_error(f"Error updating equity: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="all positions getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="balance getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None, context="trade history getting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance calculation",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None, context="paper trader cleanup",
    )

# Global paper trader instance
paper_trader: PaperTrader | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None, context="paper trader setup",
)