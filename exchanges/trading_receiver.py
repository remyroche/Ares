"""
Exchange-Agnostic Trading Receiver

Receives trading orders and routes them to the appropriate exchange.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .order_router import OrderRouter
from .data_aggregator import DataAggregator
from .exchange_registry import ExchangeRegistry


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
        
        # Initialize components
        self.exchange_registry = ExchangeRegistry()
        self.order_router = OrderRouter(self.exchange_registry)
        self.data_aggregator = DataAggregator(self.exchange_registry)
        
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
        
    async def start(self) -> None:
        """Start the trading receiver"""
        if self._running:
            return
            
        self.logger.info("Starting trading receiver...")
        
        try:
            # Initialize exchange registry with configured exchanges
            await self._initialize_exchanges()
            
            # Start cleanup task
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_responses())
            
            # Register default handlers
            self._register_default_handlers()
            
            self._running = True
            self.logger.info("Trading receiver started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start trading receiver: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the trading receiver"""
        if not self._running:
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
        
        # Close all exchange connections
        await self.exchange_registry.close_all()
        
        self.logger.info("Trading receiver stopped")
    
    def register_handler(self, message_type: MessageType, handler: Callable[[TradingMessage], Awaitable[TradingResponse]]) -> None:
        """Register message handler"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type].append(handler)
            self.logger.info(f"Registered handler for {message_type.value}")
    
    async def process_message(self, message: TradingMessage) -> TradingResponse:
        """Process incoming trading message"""
        try:
            self.message_stats["total_received"] += 1
            self._update_stats_by_type(message.type)
            self._update_stats_by_exchange(message.exchange)
            
            # Get handlers for message type
            handlers = self.message_handlers.get(message.type, [])
            
            if not handlers:
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
                    return response
                except Exception as e:
                    self.logger.error(f"Handler error for {message.type.value}: {e}")
                    continue
            
            # If no handler succeeded
            self.message_stats["total_errors"] += 1
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
        try:
            # Validate inputs
            if not exchange:
                raise ValueError("Exchange name is required")
            if not symbol:
                raise ValueError("Symbol is required")
            if not side:
                raise ValueError("Side is required")
            if not order_type:
                raise ValueError("Order type is required")
            if quantity <= 0:
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

            return await self.process_message(message)

        except Exception as e:
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
                self.logger.error(f"Error sending order to {exchange}: {e}")
                responses.append(TradingResponse(
                    id=str(int(time.time() * 1000)),
                    request_id=str(int(time.time() * 1000)),
                    success=False,
                    timestamp=datetime.now(),
                    data={},
                    error=str(e)
                ))

        return responses

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
                    self.logger.info(f"Initialized exchange: {exchange_name}")
                else:
                    self.logger.error(f"Failed to register exchange {exchange_name}")

            except Exception as e:
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
    
    async def _handle_order_message(self, message: TradingMessage) -> TradingResponse:
        """Handle order message"""
        try:
            # Validate message data
            required_fields = ["side", "order_type", "quantity"]
            for field in required_fields:
                if field not in message.data:
                    raise ValueError(f"Missing required field: {field}")

            # Normalize order type and side
            side = message.data["side"].lower()
            order_type = message.data["order_type"].upper()

            # Validate side
            valid_sides = ["buy", "sell"]
            if side not in valid_sides:
                raise ValueError(f"Invalid side: {side}. Must be one of {valid_sides}")

            # Validate order type
            valid_order_types = ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"]
            if order_type not in valid_order_types:
                raise ValueError(f"Invalid order type: {order_type}. Must be one of {valid_order_types}")

            # Route order to appropriate exchange
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

            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=result.get("success", False),
                timestamp=datetime.now(),
                data=result,
                error=result.get("error")
            )

        except Exception as e:
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
        try:
            # Validate message data
            if "data_type" not in message.data:
                raise ValueError("Missing required field: data_type")

            # Validate data type
            valid_data_types = ["ticker", "klines", "trades", "orderbook", "account_info", "position_info", "open_orders"]
            data_type = message.data["data_type"].lower()
            if data_type not in valid_data_types:
                raise ValueError(f"Invalid data type: {data_type}. Must be one of {valid_data_types}")

            # Route data request to appropriate exchange
            result = await self.data_aggregator.get_data(
                exchange=message.exchange,
                symbol=message.symbol,
                data_type=data_type,
                **{k: v for k, v in message.data.items() if k != "data_type"}
            )

            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=result.get("success", False),
                timestamp=datetime.now(),
                data=result,
                error=result.get("error")
            )

        except Exception as e:
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
        try:
            exchange = await self.exchange_registry.get_exchange(message.exchange)
            if not exchange:
                raise ValueError(f"Exchange {message.exchange} not found")
            
            result = await exchange.cancel_order(
                message.symbol,
                message.data["order_id"]
            )
            
            return TradingResponse(
                id=str(int(time.time() * 1000)),
                request_id=message.id,
                success=True,
                timestamp=datetime.now(),
                data=result
            )
            
        except Exception as e:
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
        return {
            "running": self._running,
            "statistics": self.message_stats,
            "registered_exchanges": await self.exchange_registry.get_registered_exchanges(),
            "active_connections": len(self.pending_responses),
            "timestamp": datetime.now().isoformat()
        }