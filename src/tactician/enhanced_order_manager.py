# src/tactician/enhanced_order_manager.py

"""
Enhanced Order Manager for Tactician
Handles sophisticated order management including stop-limit orders and leveraged limit orders
with partial fill management.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class OrderType(Enum):
    """Order types supported by the enhanced order manager."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LIMIT = "stop_limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


class OrderStatus(Enum):
    """Order status enumeration."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderRequest:
    """Order request data structure."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    leverage: float | None = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    reduce_only: bool = False
    close_on_trigger: bool = False
    order_link_id: str | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    trailing_stop: float | None = None
    iceberg_qty: float | None = None
    strategy_id: str | None = None
    strategy_type: str | None = (
        None  # "CHASE_MICRO_BREAKOUT", "LIMIT_ORDER_RETURN", etc.
    )


@dataclass
class OrderFill:
    """Order fill data structure."""

    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    trade_time: datetime
    is_maker: bool = False


@dataclass
class OrderState:
    """Order state tracking."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    original_quantity: float
    executed_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    price: float | None = None
    stop_price: float | None = None
    leverage: float | None = None
    time_in_force: str = "GTC"
    created_time: datetime = field(default_factory=datetime.now)
    updated_time: datetime = field(default_factory=datetime.now)
    fills: list[OrderFill] = field(default_factory=list)
    strategy_id: str | None = None
    strategy_type: str | None = None
    take_profit: float | None = None
    stop_loss: float | None = None
    trailing_stop: float | None = None
    iceberg_qty: float | None = None
    order_link_id: str | None = None

    def __post_init__(self):
        """Initialize remaining quantity."""
        self.remaining_quantity = self.original_quantity

    def add_fill(self, fill: OrderFill) -> None:
        """Add a fill to the order."""
        self.fills.append(fill)
        self.executed_quantity += fill.quantity
        self.remaining_quantity = self.original_quantity - self.executed_quantity

        # Calculate average price
        total_value = sum(f.price * f.quantity for f in self.fills)
        total_quantity = sum(f.quantity for f in self.fills)
        self.average_price = total_value / total_quantity if total_quantity > 0 else 0.0

        # Update status
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
        elif self.executed_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

        self.updated_time = datetime.now()


class EnhancedOrderManager:
    """
    Enhanced order manager for Tactician with sophisticated order management capabilities.
    Handles stop-limit orders for CHASE_MICRO_BREAKOUT and leveraged limit orders for LIMIT_ORDER_RETURN.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild("EnhancedOrderManager")

        # Order management configuration
        self.order_config = config.get("enhanced_order_manager", {})
        self.paper_trading: bool = bool(config.get("paper_trading", True))
        self.max_leverage = self.order_config.get("max_leverage", 10.0)
        self.min_order_size = self.order_config.get("min_order_size", 0.001)
        self.max_order_size = self.order_config.get("max_order_size", 1000.0)
        self.order_timeout_seconds = self.order_config.get("order_timeout_seconds", 300)
        self.partial_fill_threshold = self.order_config.get(
            "partial_fill_threshold",
            0.1,
        )

        # Strategy-specific configurations
        self.chase_micro_breakout_config = self.order_config.get(
            "chase_micro_breakout",
            {},
        )
        self.limit_order_return_config = self.order_config.get("limit_order_return", {})

        # Order tracking
        self.active_orders: dict[str, OrderState] = {}
        self.order_history: list[OrderState] = []
        self.strategy_orders: dict[str, list[str]] = {}  # strategy_id -> order_ids

        # Performance tracking
        self.total_orders_placed = 0
        self.successful_orders = 0
        self.failed_orders = 0
        self.total_volume_traded = 0.0
        self.total_commission_paid = 0.0

        self.is_initialized = False
        self.exchange_client: Any | None = None

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="enhanced order manager initialization",
    )
    async def initialize(self) -> bool:
        """Initialize the enhanced order manager."""
        try:
            self.logger.info("🚀 Initializing Enhanced Order Manager...")

            # Validate configuration
            self._validate_configuration()

            # Initialize order tracking
            self._initialize_order_tracking()

            # Initialize strategy configurations
            self._initialize_strategy_configurations()

            # In live mode, require an injected exchange client to be attached before use
            if not self.paper_trading and not self.exchange_client:
                self.logger.error("Live mode requires an injected exchange client. Call attach_exchange_client first.")
                return False

            self.is_initialized = True
            self.logger.info("✅ Enhanced Order Manager initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error initializing Enhanced Order Manager: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="attach exchange client",
    )
    async def attach_exchange_client(self, client: Any) -> bool:
        """Attach an exchange client that implements create_order/cancel_order/get_order_status as needed."""
        try:
            self.exchange_client = client
            # If client has a connect method and we're live, try to connect
            if not self.paper_trading and hasattr(client, "connect"):
                connected = await client.connect()
                if not connected:
                    self.logger.error("Failed to connect injected exchange client.")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error attaching exchange client: {e}")
            return False

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="update trailing levels",
    )
    async def update_trailing_levels(
        self,
        *,
        order_link_id: str,
        symbol: str,
        side: OrderSide | str,
        trailing_stop_pct: float | None = None,
        trailing_tp_pct: float | None = None,
    ) -> bool:
        """Update trailing stop / take-profit levels for an active position or OCO group.

        In paper/sim mode, this updates internal metadata; in live mode, integrate with exchange OCO/trailing endpoints.
        """
        try:
            # Update active orders with the same order_link_id
            updated = False
            for order in list(self.active_orders.values()):
                if order.order_link_id == order_link_id and order.symbol == symbol:
                    if trailing_stop_pct is not None:
                        order.trailing_stop = trailing_stop_pct
                    if trailing_tp_pct is not None:
                        order.take_profit = trailing_tp_pct
                    order.updated_time = datetime.now()
                    updated = True

            # Live mode: push updates to exchange by cancelling and recreating closing orders
            if updated and self.exchange_client and not self.paper_trading:
                # Normalize side and compute closing side
                side_str = side.value if isinstance(side, OrderSide) else str(side).lower()
                is_long = side_str in ("buy", "long")
                closing_side = "SELL" if is_long else "BUY"

                # Use any one order under the link to derive average price and quantity
                linked_orders = [o for o in self.active_orders.values() if o.order_link_id == order_link_id and o.symbol == symbol]
                if linked_orders:
                    ref = linked_orders[0]
                    avg_price = ref.average_price or (ref.price or 0.0)
                    qty = max(0.0, ref.remaining_quantity or ref.original_quantity)

                    # Best effort: cancel existing linked orders on venue (if we have order ids)
                    for o in linked_orders:
                        try:
                            if hasattr(self.exchange_client, "cancel_order"):
                                await self.exchange_client.cancel_order(symbol=symbol, order_id=o.order_id)
                        except Exception:
                            # Ignore cancellation failures and continue
                            pass

                    # Create new TP/SL closing orders based on trailing percentages
                    # NOTE: For simplicity we submit LIMIT orders at computed prices.
                    # Integrating native STOP_LIMIT/OCO endpoints can be added later.
                    if avg_price > 0 and qty > 0:
                        # Take Profit
                        if trailing_tp_pct is not None and trailing_tp_pct > 0:
                            tp_price = (
                                avg_price * (1 + trailing_tp_pct) if is_long else avg_price * (1 - trailing_tp_pct)
                            )
                            try:
                                if hasattr(self.exchange_client, "create_order"):
                                    await self.exchange_client.create_order(
                                        symbol=symbol,
                                        side=closing_side,
                                        order_type="LIMIT",
                                        quantity=qty,
                                        price=tp_price,
                                    )
                            except Exception as e:
                                self.logger.error(f"Failed to place TP order for {order_link_id}: {e}")

                        # Stop Loss (submit protective LIMIT as placeholder)
                        if trailing_stop_pct is not None and trailing_stop_pct > 0:
                            sl_price = (
                                avg_price * (1 - trailing_stop_pct) if is_long else avg_price * (1 + trailing_stop_pct)
                            )
                            try:
                                if hasattr(self.exchange_client, "create_order"):
                                    await self.exchange_client.create_order(
                                        symbol=symbol,
                                        side=closing_side,
                                        order_type="LIMIT",
                                        quantity=qty,
                                        price=sl_price,
                                    )
                            except Exception as e:
                                self.logger.error(f"Failed to place SL order for {order_link_id}: {e}")

            if updated:
                self.logger.info(
                    f"Updated trailing levels for link {order_link_id}: stop={trailing_stop_pct}, tp={trailing_tp_pct}"
                )
                return True

            self.logger.warning(
                f"No active orders found to update for link {order_link_id} on {symbol}"
            )
            return False
        except Exception as e:
            self.logger.error(f"Error updating trailing levels: {e}")
            return False

    def _validate_configuration(self) -> None:
        """Validate order manager configuration."""
        if self.max_leverage <= 0 or self.max_leverage > 100:
            raise ValueError("Invalid max_leverage configuration")

        if self.min_order_size <= 0:
            raise ValueError("Invalid min_order_size configuration")

        if self.max_order_size <= self.min_order_size:
            raise ValueError("max_order_size must be greater than min_order_size")

    def _initialize_order_tracking(self) -> None:
        """Initialize order tracking systems."""
        self.active_orders.clear()
        self.order_history.clear()
        self.strategy_orders.clear()

        self.logger.info("Order tracking systems initialized")

    def _initialize_strategy_configurations(self) -> None:
        """Initialize strategy-specific configurations."""
        # CHASE_MICRO_BREAKOUT configuration
        if not self.chase_micro_breakout_config:
            self.chase_micro_breakout_config = {
                "stop_limit_buffer": 0.001,  # 0.1% buffer for stop-limit orders
                "max_chase_attempts": 3,  # Maximum number of chase attempts
                "chase_timeout_seconds": 60,  # Timeout for chase orders
                "micro_breakout_threshold": 0.002,  # 0.2% threshold for micro breakouts
                "volume_confirmation": True,  # Require volume confirmation
                "momentum_confirmation": True,  # Require momentum confirmation
            }

        # LIMIT_ORDER_RETURN configuration
        if not self.limit_order_return_config:
            self.limit_order_return_config = {
                "default_leverage": 1.0,  # Default leverage for limit orders
                "max_leverage": 5.0,  # Maximum leverage for limit orders
                "partial_fill_strategy": "aggressive",  # "aggressive", "conservative", "balanced"
                "fill_timeout_seconds": 300,  # Timeout for partial fills
                "price_adjustment_threshold": 0.005,  # 0.5% threshold for price adjustments
                "volume_scaling": True,  # Scale orders based on volume
                "liquidity_consideration": True,  # Consider liquidity for order sizing
            }

        self.logger.info("Strategy configurations initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="chase micro breakout order placement",
    )
    async def place_chase_micro_breakout_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        current_price: float,
        breakout_price: float,
        strategy_id: str | None = None,
        volume_data: dict[str, Any] | None = None,
        momentum_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> OrderState | None:
        """
        Place a stop-limit order for CHASE_MICRO_BREAKOUT strategy.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            current_price: Current market price
            breakout_price: Expected breakout price
            strategy_id: Strategy identifier
            **kwargs: Additional order parameters

        Returns:
            OrderState if successful, None otherwise
        """
        try:
            self.logger.info(f"🎯 Placing CHASE_MICRO_BREAKOUT order for {symbol}")

            # Validate breakout conditions
            if not self._validate_micro_breakout_conditions(
                symbol,
                current_price,
                breakout_price,
                volume_data,
                momentum_data,
            ):
                self.logger.warning(f"Micro breakout conditions not met for {symbol}")
                return None

            # Calculate stop-limit order parameters
            stop_price, limit_price = self._calculate_chase_order_prices(
                side,
                current_price,
                breakout_price,
            )

            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP_LIMIT,
                quantity=quantity,
                price=limit_price,
                stop_price=stop_price,
                strategy_id=strategy_id,
                strategy_type="CHASE_MICRO_BREAKOUT",
                **kwargs,
            )

            # Place the order
            order_state = await self._place_order(order_request)

            if order_state:
                self.logger.info(
                    f"✅ CHASE_MICRO_BREAKOUT order placed: {order_state.order_id}",
                )
                await self._track_chase_order(order_state)

            return order_state

        except Exception as e:
            self.logger.error(f"Error placing CHASE_MICRO_BREAKOUT order: {e}")
            return None

    def _validate_micro_breakout_conditions(
        self,
        symbol: str,
        current_price: float,
        breakout_price: float,
        volume_data: dict[str, Any] | None = None,
        momentum_data: dict[str, Any] | None = None,
    ) -> bool:
        """Validate micro breakout conditions."""
        try:
            # Check price movement threshold
            price_movement = abs(breakout_price - current_price) / current_price
            threshold = self.chase_micro_breakout_config.get(
                "micro_breakout_threshold",
                0.002,
            )

            if price_movement < threshold:
                self.logger.debug(
                    f"Price movement {price_movement:.4f} below threshold {threshold}",
                )
                return False

            # Check volume confirmation if required
            if self.chase_micro_breakout_config.get("volume_confirmation", True):
                if volume_data is not None:
                    avg_volume = volume_data.get("avg_volume", 0)
                    current_volume = volume_data.get("current_volume", 0)
                    if current_volume < avg_volume * 1.2:  # 20% above average
                        self.logger.debug("Volume confirmation not met")
                        return False

            # Check momentum confirmation if required
            if self.chase_micro_breakout_config.get("momentum_confirmation", True):
                if momentum_data is not None:
                    momentum = momentum_data.get("momentum", 0)
                    if abs(momentum) < 0.001:  # Minimum momentum threshold
                        self.logger.debug("Momentum confirmation not met")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating micro breakout conditions: {e}")
            return False

    def _calculate_chase_order_prices(
        self,
        side: OrderSide,
        current_price: float,
        breakout_price: float,
    ) -> tuple[float, float]:
        """Calculate stop and limit prices for chase orders."""
        try:
            buffer = self.chase_micro_breakout_config.get("stop_limit_buffer", 0.001)

            if side == OrderSide.BUY:
                # For buy orders, stop above breakout, limit slightly higher
                stop_price = breakout_price
                limit_price = breakout_price * (1 + buffer)
            else:
                # For sell orders, stop below breakout, limit slightly lower
                stop_price = breakout_price
                limit_price = breakout_price * (1 - buffer)

            return stop_price, limit_price

        except Exception as e:
            self.logger.error(f"Error calculating chase order prices: {e}")
            return current_price, current_price

    async def _track_chase_order(self, order_state: OrderState) -> None:
        """Track and manage chase orders."""
        try:
            # Start monitoring the chase order
            asyncio.create_task(self._monitor_chase_order(order_state))

        except Exception as e:
            self.logger.error(f"Error tracking chase order: {e}")

    async def _monitor_chase_order(self, order_state: OrderState) -> None:
        """Monitor chase order and handle timeouts/retries."""
        try:
            timeout = self.chase_micro_breakout_config.get("chase_timeout_seconds", 60)
            max_attempts = self.chase_micro_breakout_config.get("max_chase_attempts", 3)

            start_time = datetime.now()
            attempts = 0

            while attempts < max_attempts:
                # Check if order is still active
                if order_state.status in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ]:
                    break

                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout:
                    self.logger.info(f"Chase order {order_state.order_id} timed out")
                    await self._cancel_order(order_state.order_id)
                    break

                # Wait before next check
                await asyncio.sleep(5)
                attempts += 1

        except Exception as e:
            self.logger.error(f"Error monitoring chase order: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="limit order return placement",
    )
    async def place_limit_order_return(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        leverage: float | None = None,
        strategy_id: str | None = None,
        liquidity_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> OrderState | None:
        """
        Place a leveraged limit order for LIMIT_ORDER_RETURN strategy.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Order quantity
            price: Limit price
            leverage: Leverage to use (optional)
            strategy_id: Strategy identifier
            **kwargs: Additional order parameters

        Returns:
            OrderState if successful, None otherwise
        """
        try:
            self.logger.info(f"🎯 Placing LIMIT_ORDER_RETURN order for {symbol}")

            # Validate limit order conditions
            if not self._validate_limit_order_conditions(
                symbol,
                price,
                quantity,
                liquidity_data,
            ):
                self.logger.warning(f"Limit order conditions not met for {symbol}")
                return None

            # Set default leverage if not provided
            if leverage is None:
                leverage = self.limit_order_return_config.get("default_leverage", 1.0)

            # Validate leverage
            max_leverage = self.limit_order_return_config.get("max_leverage", 5.0)
            if leverage > max_leverage:
                leverage = max_leverage
                self.logger.warning(f"Leverage reduced to {leverage} (max allowed)")

            # Calculate adjusted quantity based on leverage
            adjusted_quantity = quantity * leverage

            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=adjusted_quantity,
                price=price,
                leverage=leverage,
                strategy_id=strategy_id,
                strategy_type="LIMIT_ORDER_RETURN",
                **kwargs,
            )

            # Place the order
            order_state = await self._place_order(order_request)

            if order_state:
                self.logger.info(
                    f"✅ LIMIT_ORDER_RETURN order placed: {order_state.order_id}",
                )
                await self._track_limit_order(order_state)

            return order_state

        except Exception as e:
            self.logger.error(f"Error placing LIMIT_ORDER_RETURN order: {e}")
            return None

    def _validate_limit_order_conditions(
        self,
        symbol: str,
        price: float,
        quantity: float,
        liquidity_data: dict[str, Any] | None = None,
    ) -> bool:
        """Validate limit order conditions."""
        try:
            # Check minimum order size
            if quantity < self.min_order_size:
                self.logger.debug(
                    f"Quantity {quantity} below minimum {self.min_order_size}",
                )
                return False

            # Check maximum order size
            if quantity > self.max_order_size:
                self.logger.debug(
                    f"Quantity {quantity} above maximum {self.max_order_size}",
                )
                return False

            # Check liquidity if required
            if self.limit_order_return_config.get("liquidity_consideration", True):
                if liquidity_data is not None:
                    available_liquidity = liquidity_data.get("available_liquidity", 0)
                    if (
                        quantity > available_liquidity * 0.1
                    ):  # Max 10% of available liquidity
                        self.logger.debug("Order size exceeds liquidity limits")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating limit order conditions: {e}")
            return False

    async def _track_limit_order(self, order_state: OrderState) -> None:
        """Track and manage limit orders with partial fill handling."""
        try:
            # Start monitoring the limit order
            asyncio.create_task(self._monitor_limit_order(order_state))

        except Exception as e:
            self.logger.error(f"Error tracking limit order: {e}")

    async def _monitor_limit_order(self, order_state: OrderState) -> None:
        """Monitor limit order and handle partial fills."""
        try:
            timeout = self.limit_order_return_config.get("fill_timeout_seconds", 300)
            partial_fill_threshold = self.limit_order_return_config.get(
                "partial_fill_threshold",
                0.1,
            )

            start_time = datetime.now()

            while True:
                # Check if order is completed
                if order_state.status in [
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                ]:
                    break

                # Check timeout
                if (datetime.now() - start_time).total_seconds() > timeout:
                    self.logger.info(f"Limit order {order_state.order_id} timed out")
                    await self._cancel_order(order_state.order_id)
                    break

                # Check for partial fills
                if order_state.status == OrderStatus.PARTIALLY_FILLED:
                    fill_ratio = (
                        order_state.executed_quantity / order_state.original_quantity
                    )

                    if fill_ratio >= partial_fill_threshold:
                        # Consider the order successful enough
                        self.logger.info(
                            f"Limit order {order_state.order_id} partially filled ({fill_ratio:.2%})",
                        )
                        await self._handle_partial_fill_success(order_state)
                        break
                    # Consider adjusting the order
                    await self._handle_partial_fill_adjustment(order_state)

                # Wait before next check
                await asyncio.sleep(10)

        except Exception as e:
            self.logger.error(f"Error monitoring limit order: {e}")

    async def _handle_partial_fill_success(self, order_state: OrderState) -> None:
        """Handle successful partial fill."""
        try:
            self.logger.info(f"Partial fill success for order {order_state.order_id}")

            # Update order status
            order_state.status = OrderStatus.FILLED
            order_state.updated_time = datetime.now()

            # Record success metrics
            self.successful_orders += 1
            self.total_volume_traded += order_state.executed_quantity

        except Exception as e:
            self.logger.error(f"Error handling partial fill success: {e}")

    async def _handle_partial_fill_adjustment(self, order_state: OrderState) -> None:
        """Handle partial fill by adjusting the order."""
        try:
            strategy = self.limit_order_return_config.get(
                "partial_fill_strategy",
                "balanced",
            )

            if strategy == "aggressive":
                # Increase price to get more fills
                new_price = order_state.price * 1.001  # 0.1% increase
                await self._modify_order_price(order_state.order_id, new_price)

            elif strategy == "conservative":
                # Decrease price to get more fills
                new_price = order_state.price * 0.999  # 0.1% decrease
                await self._modify_order_price(order_state.order_id, new_price)

            else:  # balanced
                # Keep current price, just wait longer
                pass

        except Exception as e:
            self.logger.error(f"Error handling partial fill adjustment: {e}")

    async def _place_order(self, order_request: OrderRequest) -> OrderState | None:
        """Place an order and return the order state."""
        try:
            # Generate order ID
            order_id = f"order_{int(time.time() * 1000)}_{order_request.symbol}"

            # Create order state
            order_state = OrderState(
                order_id=order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                order_type=order_request.order_type,
                original_quantity=order_request.quantity,
                price=order_request.price,
                stop_price=order_request.stop_price,
                leverage=order_request.leverage,
                time_in_force=order_request.time_in_force,
                strategy_id=order_request.strategy_id,
                strategy_type=order_request.strategy_type,
                take_profit=order_request.take_profit,
                stop_loss=order_request.stop_loss,
                trailing_stop=order_request.trailing_stop,
                iceberg_qty=order_request.iceberg_qty,
                order_link_id=order_request.order_link_id,
            )

            # Add to active orders
            self.active_orders[order_id] = order_state

            # Track by strategy
            if order_request.strategy_id:
                if order_request.strategy_id not in self.strategy_orders:
                    self.strategy_orders[order_request.strategy_id] = []
                self.strategy_orders[order_request.strategy_id].append(order_id)

            # Update metrics
            self.total_orders_placed += 1

            if self.paper_trading:
                # Simple fill simulation for paper/sim contexts
                simulated_fill_qty = order_state.original_quantity
                simulated_price = order_request.price or 0.0
                if simulated_fill_qty > 0 and simulated_price > 0:
                    fill = OrderFill(
                        order_id=order_id,
                        symbol=order_request.symbol,
                        side=order_request.side,
                        price=simulated_price,
                        quantity=simulated_fill_qty,
                        commission=0.0,
                        commission_asset="USD",
                        trade_time=datetime.now(),
                        is_maker=False,
                    )
                    order_state.add_fill(fill)
                    # Force filled status if fully executed
                    if order_state.remaining_quantity <= 0:
                        order_state.status = OrderStatus.FILLED
            else:
                if not self.exchange_client:
                    self.logger.error("Live mode requires an exchange connection. Order not placed.")
                    order_state.status = OrderStatus.REJECTED
                    return None
                # Live execution via exchange client
                side_map = {OrderSide.BUY: "BUY", OrderSide.SELL: "SELL"}
                type_map = {
                    OrderType.MARKET: "MARKET",
                    OrderType.LIMIT: "LIMIT",
                    OrderType.STOP_LIMIT: "LIMIT",  # map to supported types as needed
                    OrderType.STOP_MARKET: "MARKET",
                    OrderType.TAKE_PROFIT: "LIMIT",
                    OrderType.TAKE_PROFIT_LIMIT: "LIMIT",
                }
                side = side_map.get(order_request.side, "BUY")
                otype = type_map.get(order_request.order_type, "MARKET")
                qty = float(max(order_request.quantity, 0.0))
                price = float(order_request.price) if order_request.price else None

                # Duck-typed order creation to be exchange-agnostic
                if hasattr(self.exchange_client, "create_order"):
                    resp = await self.exchange_client.create_order(
                        symbol=order_request.symbol,
                        side=side,
                        order_type=otype,
                        quantity=qty,
                        price=price,
                    )
                else:
                    self.logger.error("Exchange client missing create_order method")
                    resp = None
                if resp is None:
                    order_state.status = OrderStatus.REJECTED
                else:
                    # Keep pending/open; fills will be tracked by polling get_order_status elsewhere
                    order_state.status = OrderStatus.PENDING

            self.logger.info(
                f"Order placed: {order_id} ({order_request.strategy_type})",
            )
            return order_state

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    async def _cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        try:
            if order_id in self.active_orders:
                order_state = self.active_orders[order_id]
                order_state.status = OrderStatus.CANCELLED
                order_state.updated_time = datetime.now()

                # Move to history
                self.order_history.append(order_state)
                del self.active_orders[order_id]

                self.logger.info(f"Order cancelled: {order_id}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False

    async def _modify_order_price(self, order_id: str, new_price: float) -> bool:
        """Modify the price of an active order."""
        try:
            if order_id in self.active_orders:
                order_state = self.active_orders[order_id]
                order_state.price = new_price
                order_state.updated_time = datetime.now()

                self.logger.info(f"Order price modified: {order_id} -> {new_price}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error modifying order price: {e}")
            return False

    def get_order_status(self, order_id: str) -> OrderState | None:
        """Get the status of an order."""
        return self.active_orders.get(order_id)

    def get_strategy_orders(self, strategy_id: str) -> list[OrderState]:
        """Get all orders for a specific strategy."""
        order_ids = self.strategy_orders.get(strategy_id, [])
        return [
            self.active_orders[oid] for oid in order_ids if oid in self.active_orders
        ]

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics."""
        return {
            "total_orders_placed": self.total_orders_placed,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "success_rate": self.successful_orders / self.total_orders_placed
            if self.total_orders_placed > 0
            else 0.0,
            "total_volume_traded": self.total_volume_traded,
            "total_commission_paid": self.total_commission_paid,
            "active_orders_count": len(self.active_orders),
            "order_history_count": len(self.order_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="enhanced order manager cleanup",
    )
    async def stop(self) -> None:
        """Clean up the enhanced order manager."""
        try:
            self.logger.info("🛑 Stopping Enhanced Order Manager...")

            # Cancel all active orders
            for order_id in list(self.active_orders.keys()):
                await self._cancel_order(order_id)

            self.logger.info("✅ Enhanced Order Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping enhanced order manager: {e}")


# Factory function for creating enhanced order manager
@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="enhanced order manager setup",
)
async def setup_enhanced_order_manager(
    config: dict[str, Any] | None = None,
) -> EnhancedOrderManager | None:
    """
    Setup and initialize enhanced order manager.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized EnhancedOrderManager instance
    """
    try:
        if config is None:
            config = {}

        order_manager = EnhancedOrderManager(config)
        success = await order_manager.initialize()

        if success:
            return order_manager
        return None

    except Exception as e:
        print(f"❌ Error setting up enhanced order manager: {e}")
        return None
