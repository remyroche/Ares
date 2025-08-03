# src/paper_trader.py
"""
PaperTrader for training and testnet trading.
Uses Binance testnet via BinanceExchange for all operations.
"""

from datetime import datetime
from typing import Any

import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


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

        # Configuration
        self.trader_config: dict[str, Any] = self.config.get("paper_trader", {})
        self.initial_balance: float = self.trader_config.get("initial_balance", 10000.0)
        self.max_position_size: float = self.trader_config.get("max_position_size", 0.1)
        self.commission_rate: float = self.trader_config.get("commission_rate", 0.001)
        self.slippage_rate: float = self.trader_config.get("slippage_rate", 0.0005)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid paper trader configuration"),
            AttributeError: (False, "Missing required trader parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="paper trader initialization",
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
                self.logger.error("Invalid configuration for paper trader")
                return False

            # Initialize trading state
            await self._initialize_trading_state()

            self.logger.info("✅ Paper Trader initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Paper Trader initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trader configuration loading",
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
            self.logger.error(f"Error loading trader configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
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
                self.logger.error("Invalid initial balance")
                return False

            # Validate position size
            if self.max_position_size <= 0 or self.max_position_size > 1:
                self.logger.error("Invalid max position size")
                return False

            # Validate commission rate
            if self.commission_rate < 0 or self.commission_rate > 0.1:
                self.logger.error("Invalid commission rate")
                return False

            # Validate slippage rate
            if self.slippage_rate < 0 or self.slippage_rate > 0.01:
                self.logger.error("Invalid slippage rate")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trading state initialization",
    )
    async def _initialize_trading_state(self) -> None:
        """Initialize trading state."""
        try:
            # Set initial balance
            self.balance = self.initial_balance

            # Clear positions and history
            self.positions.clear()
            self.trade_history.clear()

            self.logger.info(
                f"Trading state initialized with balance: ${self.balance:,.2f}",
            )

        except Exception as e:
            self.logger.error(f"Error initializing trading state: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade parameters"),
            AttributeError: (False, "Missing trade components"),
            KeyError: (False, "Missing required trade data"),
        },
        default_return=False,
        context="buy order execution",
    )
    async def execute_buy_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> bool:
        """
        Execute a buy order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            price: Price per unit
            timestamp: Order timestamp

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

            # Record trade
            trade_record = {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": "BUY",
                "quantity": quantity,
                "price": price,
                "total_cost": total_cost,
                "commission": commission,
                "slippage": slippage,
                "balance_after": self.balance,
            }
            self.trade_history.append(trade_record)

            self.logger.info(
                f"✅ Buy order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            return False

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid trade parameters"),
            AttributeError: (False, "Missing trade components"),
            KeyError: (False, "Missing required trade data"),
        },
        default_return=False,
        context="sell order execution",
    )
    async def execute_sell_order(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> bool:
        """
        Execute a sell order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            price: Price per unit
            timestamp: Order timestamp

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

            # Record trade
            trade_record = {
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
            }
            self.trade_history.append(trade_record)

            self.logger.info(
                f"✅ Sell order executed: {quantity} {symbol} @ ${price:.4f}",
            )
            return True

        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="order validation",
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
                self.logger.error("Invalid symbol")
                return False

            # Validate quantity
            if quantity <= 0:
                self.logger.error("Invalid quantity")
                return False

            # Validate price
            if price <= 0:
                self.logger.error("Invalid price")
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
            self.logger.error(f"Error validating order: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="position getting",
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
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="all positions getting",
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
            self.logger.error(f"Error getting all positions: {e}")
            return {}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="balance getting",
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
            self.logger.error(f"Error getting balance: {e}")
            return 0.0

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="trade history getting",
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
            self.logger.error(f"Error getting trade history: {e}")
            return []

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="performance calculation",
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
            total_sell_proceeds = sum(t["net_proceeds"] for t in sell_trades)
            total_pnl = total_sell_proceeds - total_buy_cost

            # Calculate win rate
            profitable_trades = len([t for t in sell_trades if t["net_proceeds"] > 0])
            win_rate = profitable_trades / len(sell_trades) if sell_trades else 0.0

            # Calculate max drawdown
            balance_history = [self.initial_balance]
            for trade in self.trade_history:
                if trade["side"] == "BUY":
                    balance_history.append(trade["balance_after"])
                else:
                    balance_history.append(trade["balance_after"])

            max_drawdown = 0.0
            peak = balance_history[0]
            for balance in balance_history:
                peak = max(balance, peak)
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calculate Sharpe ratio (simplified)
            returns = []
            for i in range(1, len(balance_history)):
                ret = (balance_history[i] - balance_history[i - 1]) / balance_history[
                    i - 1
                ]
                returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
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
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "total_return": (self.balance - self.initial_balance)
                / self.initial_balance,
            }

        except Exception as e:
            self.logger.error(f"Error calculating performance: {e}")
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

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="paper trader cleanup",
    )
    async def stop(self) -> None:
        """Stop the paper trader."""
        self.logger.info("🛑 Stopping Paper Trader...")

        try:
            # Close all positions
            if self.positions:
                self.logger.info(f"Closing {len(self.positions)} positions...")
                # Note: In a real implementation, you would close positions at current market prices
                self.positions.clear()

            self.is_trading = False
            self.logger.info("✅ Paper Trader stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping paper trader: {e}")


# Global paper trader instance
paper_trader: PaperTrader | None = None


@handle_errors(
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
        print(f"Error setting up paper trader: {e}")
        return None
