"""
Exchange-Agnostic Trading Receiver

Receives trading orders and routes them to the appropriate exchange.
Enhanced with multi-exchange support and base exchange components.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.utils.tprint import tprint
from .order_router import OrderRouter
from .data_aggregator import DataAggregator
from .exchange_registry import ExchangeRegistry
from .base_exchange import ExchangeMessageHandler, MultiExchangeBase, ExchangeResponseHandler
from .base_exchange.message_handler import MessageBroker, MessageRouter
from .base_exchange.response_handler import ResponseAggregator


class MessageType(Enum):
    """Message type enumeration"""
    ORDER = "order"
    DATA_REQUEST = "data_request"
    ACCOUNT_INFO = "account_info"
    POSITION_INFO = "position_info"
    CANCEL_ORDER = "cancel_order"
    HEARTBEAT = "heartbeat"


@dataclass
class TradingMessage:
    """Trading message structure"""
    id: str
    type: MessageType
    exchange: str
    symbol: str
    timestamp: datetime
    data: Dict[str, Any]
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradingResponse:
    """Trading response structure"""
    id: str
    request_id: str
    success: bool
    timestamp: datetime
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TradingReceiver:
    """Exchange-agnostic trading receiver that routes orders to appropriate exchanges"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize base exchange components
        self.exchange_registry = ExchangeRegistry()
        self.multi_exchange_base = MultiExchangeBase({})
        self.message_handler = ExchangeMessageHandler(self.multi_exchange_base)
        self.response_handler = ExchangeResponseHandler()
        self.response_aggregator = ResponseAggregator()
        self.message_broker = MessageBroker()

        # Initialize legacy components for backward compatibility
        self.order_router = OrderRouter(self.exchange_registry)
        self.data_aggregator = DataAggregator(self.exchange_registry)

        # Multi-exchange order tracking
        self.pending_multi_exchange_orders: Dict[str, Dict[str, Any]] = {}
        
        # Message handling
        self.message_handlers: Dict[MessageType, List[Callable[[TradingMessage], Awaitable[TradingResponse]]]] = {
            MessageType.ORDER: [],
            MessageType.DATA_REQUEST: [],
            MessageType.ACCOUNT_INFO: [],
            MessageType.POSITION_INFO: [],
            MessageType.CANCEL_ORDER: [],
            MessageType.HEARTBEAT: []
        }
        
        # Response tracking
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.response_timeout = 30.0  # seconds
        
        # Statistics
        self.message_stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_errors": 0,
            "by_type": {},
            "by_exchange": {}
        }
        
        # Running state
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

        # Multi-exchange configuration
        self.primary_exchange = self.config.get("primary_exchange", "binance")  # Default to binance for backward compatibility
        self.failover_exchanges = self.config.get("failover_exchanges", ["okx", "gateio"])
        self.broadcast_enabled = self.config.get("broadcast_enabled", False)  # Changed to False
        self.load_balancing_enabled = self.config.get("load_balancing_enabled", False)

        # ML model to exchange and asset mapping
        self.ml_model_exchanges = self.config.get("ml_model_exchanges", {})
        self.ml_model_assets = self.config.get("ml_model_assets", {})
        self.ml_model_exchange_assets = self.config.get("ml_model_exchange_assets", {})
        self.default_ml_exchange = self.config.get("default_ml_exchange", "binance")  # Default to binance for backward compatibility
        self.default_asset = self.config.get("default_asset", "BTCUSDT")
        
    async def start(self) -> None:
        """Start the trading receiver"""
        tprint(f"Starting trading receiver", "INFO")
        if self._running:
            tprint(f"Trading receiver already running", "WARNING")
            return

        self.logger.info("Starting trading receiver...")

        try:
            # Initialize exchange registry with configured exchanges
            await self._initialize_exchanges()

            # Initialize multi-exchange base with registered exchanges
            exchanges = await self.exchange_registry.get_registered_exchanges()
            self.multi_exchange_base = MultiExchangeBase({
                name: await self.exchange_registry.get_exchange(name)
                for name in exchanges
            })

            # Update message handler with new multi-exchange base
            self.message_handler = ExchangeMessageHandler(self.multi_exchange_base)

            # Start base exchange components
            await self.message_handler.start()
            await self.response_handler.start()
            await self.message_broker.start()

            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_responses())

            # Register default handlers
            self._register_default_handlers()

            # Register enhanced message handlers
            self._register_enhanced_handlers()

            self._running = True
            self.logger.info("Trading receiver started successfully")
            tprint(f"Trading receiver started successfully", "SUCCESS")

        except Exception as e:
            tprint(f"Failed to start trading receiver: {e}", "ERROR")
            self.logger.error(f"Failed to start trading receiver: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the trading receiver"""
        tprint(f"Stopping trading receiver", "INFO")
        if not self._running:
            tprint(f"Trading receiver not running", "WARNING")
            return

        self.logger.info("Stopping trading receiver...")

        self._running = False

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop base exchange components
        await self.message_handler.stop()
        await self.response_handler.stop()
        await self.message_broker.stop()

        # Close all exchange connections
        await self.exchange_registry.close_all()
        await self.multi_exchange_base.close_all_exchanges()

        self.logger.info("Trading receiver stopped")
        tprint(f"Trading receiver stopped successfully", "SUCCESS")
    
    def register_handler(self, message_type: MessageType, handler: Callable[[TradingMessage], Awaitable[TradingResponse]]) -> None:
        """Register message handler"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].append(handler)
            self.logger.info(f"Registered handler for {message_type.value}")
    
    async def process_message(self, message: TradingMessage) -> TradingResponse:
        """Process incoming trading message"""
        tprint(f"Processing message: type={message.type.value}, exchange={message.exchange}, symbol={message.symbol}", "INFO")
        try:
            self.message_stats["total_received"] += 1
            self._update_stats_by_type(message.type)
            self._update_stats_by_exchange(message.exchange)
            
            # Get handlers for message type
            handlers = self.message_handlers.get(message.type, [])

            if not handlers:
                tprint(f"No handlers registered for message type: {message.type.value}", "WARNING")
                return TradingResponse(
                    id=str(int(time.time() * 1000)),
                    request_id=message.id,
                    success=False,
                    timestamp=datetime.now(),
                    data={},
                    error=f"No handlers registered for message type: {message.type.value}"
                )
            
            # Process with first available handler
            for handler in handlers:
                try:
                    response = await handler(message)
                    self.message_stats["total_processed"] += 1
                    tprint(f"Message processed successfully: type={message.type.value}", "SUCCESS")
                    return response
                except Exception as e:
                    tprint(f"Handler error for {message.type.value}: {e}", "ERROR")
                    self.logger.error(f"Handler error for {message.type.value}: {e}")
                    continue
            
            # If no handler succeeded
            self.message_stats["total_errors"] += 1
            tprint(f"All handlers failed to process message: type={message.type.value}", "ERROR")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error="All handlers failed to process message"
            )
            
        except Exception as e:
            self.message_stats["total_errors"] += 1
            tprint(f"Error processing message: {e}", "ERROR")
            self.logger.error(f"Error processing message: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def send_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> TradingResponse:
        """Send order to specified exchange"""
        tprint(f"Sending order: exchange={exchange}, symbol={symbol}, side={side}, type={order_type}, quantity={quantity}, price={price}", "INFO")
        try:
            # Validate inputs
            if not exchange:
                tprint(f"Exchange name is required", "ERROR")
                raise ValueError("Exchange name is required")
            if not symbol:
                tprint(f"Symbol is required", "ERROR")
                raise ValueError("Symbol is required")
            if not side:
                tprint(f"Side is required", "ERROR")
                raise ValueError("Side is required")
            if not order_type:
                tprint(f"Order type is required", "ERROR")
                raise ValueError("Order type is required")
            if quantity <= 0:
                tprint(f"Quantity must be positive", "ERROR")
                raise ValueError("Quantity must be positive")

            message = TradingMessage(
                id=str(int(time.time() * 1000)),
                type=MessageType.ORDER,
                exchange=exchange,
                symbol=symbol,
                timestamp=datetime.now(),
                data={
                    "side": side,
                    "order_type": order_type,
                    "quantity": quantity,
                    "price": price,
                    **kwargs
                }
            )

            response = await self.process_message(message)
            if response.success:
                tprint(f"Order sent successfully: exchange={exchange}, symbol={symbol}", "SUCCESS")
            return response

        except Exception as e:
            tprint(f"Error sending order: {e}", "ERROR")
            self.logger.error(f"Error sending order: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )

    async def send_multi_exchange_order(
        self,
        exchanges: List[str],
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> List[TradingResponse]:
        """Send order to multiple exchanges"""
        tprint(f"Sending order to {len(exchanges)} exchanges: {exchanges}", "INFO")
        responses = []

        for exchange in exchanges:
            try:
                response = await self.send_order(
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )
                responses.append(response)

            except Exception as e:
                tprint(f"Error sending order to {exchange}: {e}", "ERROR")
                self.logger.error(f"Error sending order to {exchange}: {e}")
                responses.append(TradingResponse(
                    id=str(int(time.time() * 1000)),
                    request_id=str(int(time.time() * 1000)),
                    success=False,
                    timestamp=datetime.now(),
                    data={},
                    error=str(e)
                ))

        tprint(f"Multi-exchange order completed: {len(responses)} responses", "SUCCESS")
        return responses

    async def send_order_for_ml_model(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        ml_model_id: Optional[str] = None,
        **kwargs
    ) -> TradingResponse:
        """
        Send order to the exchange associated with the ML model and asset.

        Args:
            symbol: Trading symbol/asset
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            ml_model_id: ID of the ML model to determine target exchange
            **kwargs: Additional order parameters

        Returns:
            Response from the appropriate exchange
        """
        tprint(f"Sending order for ML model: model_id={ml_model_id}, symbol={symbol}, side={side}", "INFO")
        if not self._running:
            tprint(f"Trading receiver is not running", "ERROR")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error="Trading receiver is not running"
            )

        # Validate ML model and asset compatibility
        if ml_model_id and not self._validate_ml_model_asset_compatibility(ml_model_id, symbol):
            tprint(f"ML model {ml_model_id} is not compatible with asset {symbol}", "ERROR")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error=f"ML model {ml_model_id} is not compatible with asset {symbol}",
                metadata={"ml_model_id": ml_model_id, "asset": symbol, "validation_failed": "asset_compatibility"}
            )

        # Determine target exchange based on ML model and asset
        target_exchange = self._get_exchange_for_ml_model(ml_model_id, symbol)
        tprint(f"Routing ML model order to exchange: {target_exchange}", "INFO")

        try:
            response = await self.send_order(
                exchange=target_exchange,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **kwargs
            )

            # Add ML model and asset information to response metadata
            if hasattr(response, 'metadata'):
                response.metadata["ml_model_id"] = ml_model_id
                response.metadata["target_exchange"] = target_exchange
                response.metadata["asset"] = symbol
                response.metadata["asset_compatible"] = True

            if response.success:
                tprint(f"ML model order sent successfully: model={ml_model_id}, exchange={target_exchange}", "SUCCESS")
            return response

        except Exception as e:
            tprint(f"Error sending order for ML model {ml_model_id} to {target_exchange}: {e}", "ERROR")
            self.logger.error(f"Error sending order for ML model {ml_model_id} asset {symbol} to {target_exchange}: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e),
                metadata={
                    "ml_model_id": ml_model_id,
                    "target_exchange": target_exchange,
                    "asset": symbol,
                    "asset_compatible": True
                }
            )

    def _get_exchange_for_ml_model(
        self,
        ml_model_id: Optional[str] = None,
        asset: Optional[str] = None
    ) -> str:
        """
        Get the exchange associated with a specific ML model and asset.

        Args:
            ml_model_id: ID of the ML model
            asset: Trading symbol/asset

        Returns:
            Name of the exchange associated with the ML model and asset
        """
        if ml_model_id and asset:
            # First check if there's a specific model-exchange-asset combination
            model_exchange_asset_key = f"{ml_model_id}:{asset}"
            if model_exchange_asset_key in self.ml_model_exchange_assets:
                return self.ml_model_exchange_assets[model_exchange_asset_key]

            # Then check model-specific exchange
            if ml_model_id in self.ml_model_exchanges:
                return self.ml_model_exchanges[ml_model_id]

        # Fallback to default ML exchange
        return self.default_ml_exchange

    def _get_asset_for_ml_model(self, ml_model_id: Optional[str] = None) -> str:
        """
        Get the default asset associated with a specific ML model.

        Args:
            ml_model_id: ID of the ML model

        Returns:
            Default asset for the ML model
        """
        if ml_model_id and ml_model_id in self.ml_model_assets:
            return self.ml_model_assets[ml_model_id]

        # Fallback to default asset
        return self.default_asset

    def _validate_ml_model_asset_compatibility(
        self,
        ml_model_id: str,
        asset: str
    ) -> bool:
        """
        Validate that the ML model is compatible with the given asset.

        Args:
            ml_model_id: ID of the ML model
            asset: Trading symbol/asset

        Returns:
            True if compatible, False otherwise
        """
        # Check if there's a specific model-asset association
        model_asset_key = f"{ml_model_id}:{asset}"
        if model_asset_key in self.ml_model_exchange_assets:
            return True

        # Check if model has specific assets defined
        if ml_model_id in self.ml_model_assets:
            allowed_assets = self.ml_model_assets[ml_model_id]
            if isinstance(allowed_assets, list):
                return asset in allowed_assets
            elif isinstance(allowed_assets, str):
                return asset == allowed_assets

        # If no specific asset restrictions, allow any asset
        return True

    async def send_order_to_all_exchanges(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, TradingResponse]:
        """
        Send order to all configured exchanges (legacy method for backward compatibility).

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            **kwargs: Additional order parameters

        Returns:
            Dictionary mapping exchange names to their responses
        """
        if not self._running:
            return {"error": "Trading receiver is not running"}

        # Get all registered exchanges
        exchanges = await self.exchange_registry.get_registered_exchanges()
        if not exchanges:
            return {"error": "No exchanges registered"}

        responses = {}

        # Send order to each exchange
        for exchange_name in exchanges:
            try:
                response = await self.send_order(
                    exchange=exchange_name,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )
                responses[exchange_name] = response

            except Exception as e:
                self.logger.error(f"Error sending order to {exchange_name}: {e}")
                responses[exchange_name] = TradingResponse(
                    id=str(int(time.time() * 1000)),
                    request_id=str(int(time.time() * 1000)),
                    success=False,
                    timestamp=datetime.now(),
                    data={},
                    error=str(e)
                )

        return responses

    async def send_order_with_routing(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        routing_strategy: str = "ml_model",
        ml_model_id: Optional[str] = None,
        **kwargs
    ) -> Union[TradingResponse, Dict[str, TradingResponse]]:
        """
        Send order using intelligent routing strategy.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            routing_strategy: Routing strategy ("ml_model", "broadcast", "primary", "failover", "best_price")
            ml_model_id: ID of the ML model for routing decisions
            **kwargs: Additional order parameters

        Returns:
            Response from the appropriate exchange(s)
        """
        if not self._running:
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error="Trading receiver is not running"
            )

        if routing_strategy == "ml_model":
            return await self.send_order_for_ml_model(
                symbol, side, order_type, quantity, price, ml_model_id, **kwargs
            )

        elif routing_strategy == "broadcast":
            return await self.send_order_to_all_exchanges(symbol, side, order_type, quantity, price, **kwargs)

        elif routing_strategy == "primary":
            return await self.send_order_to_primary_with_failover(symbol, side, order_type, quantity, price, **kwargs)

        elif routing_strategy == "best_price":
            best_exchange = await self._get_best_execution_exchange(symbol, side, quantity)
            if best_exchange:
                response = await self.send_order(
                    exchange=best_exchange,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )
                return {best_exchange: response}
            else:
                return TradingResponse(
                    id=str(int(time.time() * 1000)),
                    request_id=str(int(time.time() * 1000)),
                    success=False,
                    timestamp=datetime.now(),
                    data={},
                    error="No suitable exchange found for best price execution"
                )

        else:
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=str(int(time.time() * 1000)),
                success=False,
                timestamp=datetime.now(),
                data={},
                error=f"Unknown routing strategy: {routing_strategy}"
            )

    async def send_order_to_primary_with_failover(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, TradingResponse]:
        """
        Send order to primary exchange with automatic failover.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            **kwargs: Additional order parameters

        Returns:
            Dictionary mapping exchange names to their responses
        """
        exchanges_to_try = [self.primary_exchange] + [ex for ex in self.failover_exchanges if ex != self.primary_exchange]

        for exchange_name in exchanges_to_try:
            try:
                response = await self.send_order(
                    exchange=exchange_name,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    **kwargs
                )

                # If successful, return the response
                if response.success:
                    return {exchange_name: response}
                else:
                    self.logger.warning(f"Order failed on {exchange_name}: {response.error}")
                    continue

            except Exception as e:
                self.logger.error(f"Error sending order to {exchange_name}: {e}")
                continue

        # If all exchanges failed
        return {"error": "All exchanges failed to execute order"}

    async def _get_best_execution_exchange(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> Optional[str]:
        """
        Determine the best exchange for order execution.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity

        Returns:
            Name of the best exchange or None
        """
        try:
            # Get best price information from data aggregator
            price_info = await self.data_aggregator.get_aggregated_data(symbol, "ticker")

            if not price_info.get("success"):
                return self.primary_exchange  # Fallback to primary

            exchange_data = price_info.get("exchange_data", {})

            if side == "buy":
                # For buys, prefer lowest ask price
                best_exchange = None
                best_ask = float('inf')

                for exchange, data in exchange_data.items():
                    ask = float(data.get("ask", 0))
                    if ask > 0 and ask < best_ask:
                        best_ask = ask
                        best_exchange = exchange

                return best_exchange or self.primary_exchange

            else:
                # For sells, prefer highest bid price
                best_exchange = None
                best_bid = 0

                for exchange, data in exchange_data.items():
                    bid = float(data.get("bid", 0))
                    if bid > best_bid:
                        best_bid = bid
                        best_exchange = exchange

                return best_exchange or self.primary_exchange

        except Exception as e:
            self.logger.error(f"Error determining best execution exchange: {e}")
            return self.primary_exchange

    async def _handle_multi_exchange_order(self, message: TradingMessage) -> None:
        """
        Handle orders that should be sent to multiple exchanges.

        Args:
            message: Trading message containing order information
        """
        try:
            # Extract order information from message
            symbol = message.symbol
            side = message.data.get("side", "buy")
            order_type = message.data.get("order_type", "market")
            quantity = message.data.get("quantity", 0)
            price = message.data.get("price")

            # Determine target exchanges based on configuration
            target_exchanges = await self._determine_target_exchanges(message)

            # Send order to all target exchanges
            responses = await self.send_order_to_all_exchanges(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                **message.data
            )

            # Store multi-exchange order information
            multi_order_id = f"multi_{symbol}_{int(time.time() * 1000)}"
            self.pending_multi_exchange_orders[multi_order_id] = {
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "target_exchanges": target_exchanges,
                "responses": responses,
                "timestamp": datetime.now(),
                "status": "completed"
            }

            self.logger.info(f"Multi-exchange order {multi_order_id} completed for {symbol}")

        except Exception as e:
            self.logger.error(f"Error handling multi-exchange order: {e}")

    async def _determine_target_exchanges(self, message: TradingMessage) -> List[str]:
        """
        Determine which exchanges should receive the order.

        Args:
            message: Trading message

        Returns:
            List of exchange names
        """
        # Check if specific exchanges are requested
        requested_exchanges = message.data.get("target_exchanges")
        if requested_exchanges:
            return requested_exchanges

        # Use configured primary and failover exchanges
        if self.broadcast_enabled:
            return await self.exchange_registry.get_active_exchanges()
        else:
            return [self.primary_exchange] + self.failover_exchanges

    async def get_multi_exchange_order_status(self, multi_order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a multi-exchange order.

        Args:
            multi_order_id: ID of the multi-exchange order

        Returns:
            Order status information or None if not found
        """
        return self.pending_multi_exchange_orders.get(multi_order_id)

    async def get_all_multi_exchange_orders(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all multi-exchange orders.

        Returns:
            Dictionary of all multi-exchange orders
        """
        return dict(self.pending_multi_exchange_orders)

    def register_ml_model_exchange(
        self,
        ml_model_id: str,
        exchange_name: str,
        assets: Optional[Union[str, List[str]]] = None
    ) -> bool:
        """
        Register which exchange a specific ML model should use for specific assets.

        Args:
            ml_model_id: ID of the ML model
            exchange_name: Name of the exchange
            assets: Specific asset(s) the model should handle (None for all assets)

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            if exchange_name not in [ex for ex in asyncio.run(self.exchange_registry.get_registered_exchanges())]:
                self.logger.warning(f"Exchange {exchange_name} is not registered")
                return False

            self.ml_model_exchanges[ml_model_id] = exchange_name

            # Register specific assets if provided
            if assets:
                if isinstance(assets, str):
                    assets = [assets]
                self.ml_model_assets[ml_model_id] = assets

            self.logger.info(f"Registered ML model {ml_model_id} to exchange {exchange_name} for assets {assets}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering ML model {ml_model_id}: {e}")
            return False

    def register_ml_model_exchange_asset(
        self,
        ml_model_id: str,
        exchange_name: str,
        asset: str
    ) -> bool:
        """
        Register a specific ML model-exchange-asset combination.

        Args:
            ml_model_id: ID of the ML model
            exchange_name: Name of the exchange
            asset: Specific asset for this combination

        Returns:
            True if registered successfully, False otherwise
        """
        try:
            if exchange_name not in [ex for ex in asyncio.run(self.exchange_registry.get_registered_exchanges())]:
                self.logger.warning(f"Exchange {exchange_name} is not registered")
                return False

            # Register the specific combination
            model_exchange_asset_key = f"{ml_model_id}:{asset}"
            self.ml_model_exchange_assets[model_exchange_asset_key] = exchange_name

            self.logger.info(f"Registered ML model {ml_model_id} to exchange {exchange_name} for asset {asset}")
            return True

        except Exception as e:
            self.logger.error(f"Error registering ML model {ml_model_id} asset {asset}: {e}")
            return False

    def unregister_ml_model_exchange(self, ml_model_id: str) -> bool:
        """
        Unregister an ML model's exchange association.

        Args:
            ml_model_id: ID of the ML model

        Returns:
            True if unregistered successfully, False otherwise
        """
        try:
            if ml_model_id in self.ml_model_exchanges:
                del self.ml_model_exchanges[ml_model_id]
                self.logger.info(f"Unregistered ML model {ml_model_id}")
                return True
            return False

        except Exception as e:
            self.logger.error(f"Error unregistering ML model {ml_model_id}: {e}")
            return False

    def get_ml_model_exchange(self, ml_model_id: str) -> Optional[str]:
        """
        Get the exchange associated with a specific ML model.

        Args:
            ml_model_id: ID of the ML model

        Returns:
            Name of the exchange or None if not found
        """
        return self.ml_model_exchanges.get(ml_model_id)

    def get_all_ml_model_exchanges(self) -> Dict[str, str]:
        """
        Get all ML model to exchange mappings.

        Returns:
            Dictionary mapping ML model IDs to exchange names
        """
        return dict(self.ml_model_exchanges)

    def set_default_ml_exchange(self, exchange_name: str) -> bool:
        """
        Set the default exchange for ML models.

        Args:
            exchange_name: Name of the exchange

        Returns:
            True if set successfully, False otherwise
        """
        try:
            if exchange_name not in [ex for ex in asyncio.run(self.exchange_registry.get_registered_exchanges())]:
                self.logger.warning(f"Exchange {exchange_name} is not registered")
                return False

            self.default_ml_exchange = exchange_name
            self.logger.info(f"Set default ML exchange to {exchange_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting default ML exchange: {e}")
            return False

    def _get_ml_models_by_exchange(self) -> Dict[str, List[str]]:
        """
        Get ML models grouped by exchange.

        Returns:
            Dictionary mapping exchange names to lists of ML model IDs
        """
        exchange_models = {}
        for ml_model_id, exchange_name in self.ml_model_exchanges.items():
            if exchange_name not in exchange_models:
                exchange_models[exchange_name] = []
            exchange_models[exchange_name].append(ml_model_id)

        return exchange_models

    def _get_assets_by_ml_model(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get assets associated with each ML model.

        Returns:
            Dictionary mapping ML model IDs to their associated assets
        """
        model_assets = {}

        # From specific model-asset associations
        for model_asset_key, exchange in self.ml_model_exchange_assets.items():
            if ":" in model_asset_key:
                ml_model_id, asset = model_asset_key.split(":", 1)
                if ml_model_id not in model_assets:
                    model_assets[ml_model_id] = []
                if isinstance(model_assets[ml_model_id], list):
                    if asset not in model_assets[ml_model_id]:
                        model_assets[ml_model_id].append(asset)

        # From general model-asset associations
        for ml_model_id, assets in self.ml_model_assets.items():
            if ml_model_id not in model_assets:
                model_assets[ml_model_id] = assets
            else:
                # Merge with existing
                if isinstance(model_assets[ml_model_id], list) and isinstance(assets, list):
                    for asset in assets:
                        if asset not in model_assets[ml_model_id]:
                            model_assets[ml_model_id].append(asset)

        return model_assets

    async def cancel_multi_exchange_order(self, multi_order_id: str) -> Dict[str, Any]:
        """
        Cancel a multi-exchange order on all exchanges.

        Args:
            multi_order_id: ID of the multi-exchange order

        Returns:
            Cancellation results from all exchanges
        """
        if multi_order_id not in self.pending_multi_exchange_orders:
            return {"error": "Multi-exchange order not found"}

        order_info = self.pending_multi_exchange_orders[multi_order_id]
        responses = {}

        # Cancel on each exchange where the order was sent
        for exchange_name in order_info.get("target_exchanges", []):
            try:
                # Extract the original order ID for this exchange from responses
                exchange_response = order_info.get("responses", {}).get(exchange_name, {})
                if exchange_response.get("success"):
                    order_id = exchange_response.get("data", {}).get("order_id")
                    if order_id:
                        cancel_response = await self.cancel_order(exchange_name, order_info["symbol"], order_id)
                        responses[exchange_name] = cancel_response
                    else:
                        responses[exchange_name] = {"error": "No order ID found for cancellation"}
                else:
                    responses[exchange_name] = {"error": "Order was not successfully placed"}

            except Exception as e:
                responses[exchange_name] = {"error": str(e)}

        # Update order status
        order_info["status"] = "cancelled"
        order_info["cancelled_at"] = datetime.now()

        return {
            "multi_order_id": multi_order_id,
            "cancellation_responses": responses,
            "successful_cancellations": [
                ex for ex, resp in responses.items()
                if resp.get("success", False)
            ]
        }

    async def request_data(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> TradingResponse:
        """Request data from specified exchange"""
        message = TradingMessage(
            id=str(int(time.time() * 1000)),
            type=MessageType.DATA_REQUEST,
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.now(),
            data={
                "data_type": data_type,
                **kwargs
            }
        )
        
        return await self.process_message(message)
    
    async def get_account_info(self, exchange: str) -> TradingResponse:
        """Get account information from specified exchange"""
        message = TradingMessage(
            id=str(int(time.time() * 1000)),
            type=MessageType.ACCOUNT_INFO,
            exchange=exchange,
            symbol="",  # Not symbol-specific
            timestamp=datetime.now(),
            data={}
        )
        
        return await self.process_message(message)
    
    async def get_position_info(self, exchange: str, symbol: str) -> TradingResponse:
        """Get position information from specified exchange"""
        message = TradingMessage(
            id=str(int(time.time() * 1000)),
            type=MessageType.POSITION_INFO,
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.now(),
            data={}
        )
        
        return await self.process_message(message)
    
    async def cancel_order(self, exchange: str, symbol: str, order_id: str) -> TradingResponse:
        """Cancel order on specified exchange"""
        message = TradingMessage(
            id=str(int(time.time() * 1000)),
            type=MessageType.CANCEL_ORDER,
            exchange=exchange,
            symbol=symbol,
            timestamp=datetime.now(),
            data={
                "order_id": order_id
            }
        )
        
        return await self.process_message(message)
    
    async def send_heartbeat(self, exchange: str) -> TradingResponse:
        """Send heartbeat to specified exchange"""
        message = TradingMessage(
            id=str(int(time.time() * 1000)),
            type=MessageType.HEARTBEAT,
            exchange=exchange,
            symbol="",
            timestamp=datetime.now(),
            data={}
        )
        
        return await self.process_message(message)
    
    async def _initialize_exchanges(self) -> None:
        """Initialize configured exchanges"""
        tprint(f"Initializing exchanges", "INFO")
        exchanges_config = self.config.get("exchanges", {})

        for exchange_name, exchange_config in exchanges_config.items():
            try:
                # Create exchange instance using factory
                from ..exchange.factory import ExchangeFactory
                exchange = ExchangeFactory.get_exchange(exchange_name)

                # Initialize exchange with configuration
                if hasattr(exchange, '_initialize_exchange'):
                    await exchange._initialize_exchange()

                # Register exchange
                success = await self.exchange_registry.register_exchange(exchange_name, exchange)

                if success:
                    tprint(f"Exchange initialized: {exchange_name}", "SUCCESS")
                    self.logger.info(f"Initialized exchange: {exchange_name}")
                else:
                    tprint(f"Failed to register exchange {exchange_name}", "ERROR")
                    self.logger.error(f"Failed to register exchange {exchange_name}")

            except Exception as e:
                tprint(f"Failed to initialize exchange {exchange_name}: {e}", "ERROR")
                self.logger.error(f"Failed to initialize exchange {exchange_name}: {e}")
                self.message_stats["total_errors"] += 1
    
    def _register_default_handlers(self) -> None:
        """Register default message handlers"""
        self.register_handler(MessageType.ORDER, self._handle_order_message)
        self.register_handler(MessageType.DATA_REQUEST, self._handle_data_request)
        self.register_handler(MessageType.ACCOUNT_INFO, self._handle_account_info)
        self.register_handler(MessageType.POSITION_INFO, self._handle_position_info)
        self.register_handler(MessageType.CANCEL_ORDER, self._handle_cancel_order)
        self.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)

    def _register_enhanced_handlers(self) -> None:
        """Register enhanced multi-exchange message handlers"""
        # Register handlers for the enhanced message handler
        from .base_exchange.message_handler import MessageType as BaseMessageType

        async def handle_multi_exchange_order(message: TradingMessage, exchange_name: str) -> None:
            """Handle orders sent to multiple exchanges"""
            await self._handle_multi_exchange_order(message)

        # Note: This would register with the base exchange message handler
        # For now, we'll use the existing structure but add multi-exchange capability
    
    async def _handle_order_message(self, message: TradingMessage) -> TradingResponse:
        """Handle order message"""
        tprint(f"Handling order message: exchange={message.exchange}, symbol={message.symbol}", "INFO")
        try:
            # Validate message data
            required_fields = ["side", "order_type", "quantity"]
            for field in required_fields:
                if field not in message.data:
                    tprint(f"Missing required field: {field}", "ERROR")
                    raise ValueError(f"Missing required field: {field}")

            # Normalize order type and side
            side = message.data["side"].lower()
            order_type = message.data["order_type"].upper()

            # Validate side
            valid_sides = ["buy", "sell"]
            if side not in valid_sides:
                tprint(f"Invalid side: {side}", "ERROR")
                raise ValueError(f"Invalid side: {side}. Must be one of {valid_sides}")

            # Validate order type
            valid_order_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
            if order_type not in valid_order_types:
                tprint(f"Invalid order type: {order_type}", "ERROR")
                raise ValueError(f"Invalid order type: {order_type}. Must be one of {valid_order_types}")

            # Route order to appropriate exchange
            tprint(f"Routing order via order_router: {message.exchange}/{message.symbol}", "INFO")
            result = await self.order_router.route_order(
                exchange=message.exchange,
                symbol=message.symbol,
                side=side,
                order_type=order_type,
                quantity=float(message.data["quantity"]),
                price=message.data.get("price"),
                **{k: v for k, v in message.data.items()
                   if k not in ["side", "order_type", "quantity", "price"]}
            )

            if result.get("success"):
                tprint(f"Order routed successfully via order_router", "SUCCESS")

            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=result.get("success", False),
                timestamp=datetime.now(),
                data=result,
                error=result.get("error")
            )

        except Exception as e:
            tprint(f"Error handling order message: {e}", "ERROR")
            self.logger.error(f"Error handling order message: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _handle_data_request(self, message: TradingMessage) -> TradingResponse:
        """Handle data request message"""
        tprint(f"Handling data request: exchange={message.exchange}, symbol={message.symbol}", "INFO")
        try:
            # Validate message data
            if "data_type" not in message.data:
                tprint(f"Missing required field: data_type", "ERROR")
                raise ValueError("Missing required field: data_type")

            # Validate data type
            valid_data_types = ["ticker", "klines", "trades", "orderbook", "account_info", "position_info", "open_orders"]
            data_type = message.data["data_type"].lower()
            if data_type not in valid_data_types:
                tprint(f"Invalid data type: {data_type}", "ERROR")
                raise ValueError(f"Invalid data type: {data_type}. Must be one of {valid_data_types}")

            # Route data request to appropriate exchange
            result = await self.data_aggregator.get_data(
                exchange=message.exchange,
                symbol=message.symbol,
                data_type=data_type,
                **{k: v for k, v in message.data.items() if k != "data_type"}
            )

            if result.get("success"):
                tprint(f"Data request completed: type={data_type}", "SUCCESS")

            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=result.get("success", False),
                timestamp=datetime.now(),
                data=result,
                error=result.get("error")
            )

        except Exception as e:
            tprint(f"Error handling data request: {e}", "ERROR")
            self.logger.error(f"Error handling data request: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _handle_account_info(self, message: TradingMessage) -> TradingResponse:
        """Handle account info request"""
        try:
            exchange = await self.exchange_registry.get_exchange(message.exchange)
            if not exchange:
                raise ValueError(f"Exchange {message.exchange} not found")
            
            account_info = await exchange.get_account_info()
            
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=True,
                timestamp=datetime.now(),
                data=account_info
            )
            
        except Exception as e:
            self.logger.error(f"Error handling account info request: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _handle_position_info(self, message: TradingMessage) -> TradingResponse:
        """Handle position info request"""
        try:
            exchange = await self.exchange_registry.get_exchange(message.exchange)
            if not exchange:
                raise ValueError(f"Exchange {message.exchange} not found")
            
            position_info = await exchange.get_position_risk(message.symbol)
            
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=True,
                timestamp=datetime.now(),
                data=position_info
            )
            
        except Exception as e:
            self.logger.error(f"Error handling position info request: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _handle_cancel_order(self, message: TradingMessage) -> TradingResponse:
        """Handle cancel order request"""
        tprint(f"Handling cancel order: exchange={message.exchange}, symbol={message.symbol}", "INFO")
        try:
            exchange = await self.exchange_registry.get_exchange(message.exchange)
            if not exchange:
                tprint(f"Exchange {message.exchange} not found", "ERROR")
                raise ValueError(f"Exchange {message.exchange} not found")
            
            result = await exchange.cancel_order(
                message.symbol,
                message.data["order_id"]
            )

            tprint(f"Order cancelled: order_id={message.data['order_id']}", "SUCCESS")

            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=True,
                timestamp=datetime.now(),
                data=result
            )
            
        except Exception as e:
            tprint(f"Error handling cancel order request: {e}", "ERROR")
            self.logger.error(f"Error handling cancel order request: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _handle_heartbeat(self, message: TradingMessage) -> TradingResponse:
        """Handle heartbeat message"""
        try:
            exchange = await self.exchange_registry.get_exchange(message.exchange)
            if not exchange:
                raise ValueError(f"Exchange {message.exchange} not found")
            
            # Simple health check - try to get server time or similar
            if hasattr(exchange, 'get_ticker'):
                await exchange.get_ticker("BTCUSDT")  # Simple health check
            
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=True,
                timestamp=datetime.now(),
                data={"status": "healthy"}
            )
            
        except Exception as e:
            self.logger.error(f"Error handling heartbeat: {e}")
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=False,
                timestamp=datetime.now(),
                data={},
                error=str(e)
            )
    
    async def _cleanup_expired_responses(self) -> None:
        """Cleanup expired response futures"""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, future in self.pending_responses.items():
                    if current_time - float(key.split('_')[0]) > self.response_timeout:
                        expired_keys.append(key)
                        if not future.done():
                            future.cancel()
                
                for key in expired_keys:
                    del self.pending_responses[key]
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(10)
    
    def _update_stats_by_type(self, message_type: MessageType) -> None:
        """Update statistics by message type"""
        type_str = message_type.value
        if type_str not in self.message_stats["by_type"]:
            self.message_stats["by_type"][type_str] = 0
        self.message_stats["by_type"][type_str] += 1
    
    def _update_stats_by_exchange(self, exchange: str) -> None:
        """Update statistics by exchange"""
        if exchange not in self.message_stats["by_exchange"]:
            self.message_stats["by_exchange"][exchange] = 0
        self.message_stats["by_exchange"][exchange] += 1
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get receiver statistics"""
        try:
            # Get basic statistics
            basic_stats = {
                "running": self._running,
                "statistics": self.message_stats,
                "registered_exchanges": await self.exchange_registry.get_registered_exchanges(),
                "active_connections": len(self.pending_responses),
                "timestamp": datetime.now().isoformat()
            }

            # Get multi-exchange statistics
            multi_exchange_stats = {
                "multi_exchange_orders": len(self.pending_multi_exchange_orders),
                "primary_exchange": self.primary_exchange,
                "failover_exchanges": self.failover_exchanges,
                "broadcast_enabled": self.broadcast_enabled,
                "load_balancing_enabled": self.load_balancing_enabled,
                "total_exchanges_configured": len(await self.exchange_registry.get_registered_exchanges())
            }

            # Get ML model statistics
            ml_model_stats = {
                "registered_ml_models": len(self.ml_model_exchanges),
                "default_ml_exchange": self.default_ml_exchange,
                "default_asset": self.default_asset,
                "ml_model_exchanges": self.ml_model_exchanges,
                "ml_model_assets": self.ml_model_assets,
                "ml_model_exchange_assets": self.ml_model_exchange_assets,
                "ml_models_by_exchange": self._get_ml_models_by_exchange(),
                "assets_by_ml_model": self._get_assets_by_ml_model()
            }

            # Get message handler statistics
            message_handler_stats = {}
            if hasattr(self.message_handler, 'get_queue_status'):
                message_handler_stats = await self.message_handler.get_queue_status()

            # Get response handler statistics
            response_handler_stats = {}
            if hasattr(self.response_handler, 'get_response_statistics'):
                response_handler_stats = await self.response_handler.get_response_statistics()

            return {
                **basic_stats,
                "multi_exchange": multi_exchange_stats,
                "ml_model": ml_model_stats,
                "message_handler": message_handler_stats,
                "response_handler": response_handler_stats,
                "config": {
                    "primary_exchange": self.primary_exchange,
                    "failover_exchanges": self.failover_exchanges,
                    "broadcast_enabled": self.broadcast_enabled,
                    "load_balancing_enabled": self.load_balancing_enabled,
                    "default_ml_exchange": self.default_ml_exchange,
                    "default_asset": self.default_asset
                }
            }

        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
            return {
                "error": str(e),
                "running": self._running,
                "timestamp": datetime.now().isoformat()
            }