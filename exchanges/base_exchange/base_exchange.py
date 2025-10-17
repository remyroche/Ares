from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime

from src.interfaces.base_interfaces import IExchangeClient, MarketData

from typing import Any, Dict, List, Optional, Set
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


@dataclass
class OrderMapping:
    """Order mapping between internal and exchange-specific IDs"""
    internal_id: str
    exchange_id: str
    exchange_name: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    exchange_response: Optional[Dict[str, Any]] = None


class OrderIdMapper:
    """Manages order ID mapping between internal and exchange-specific IDs"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.OrderIdMapper")
        self._order_mappings: Dict[str, OrderMapping] = {}  # internal_id -> OrderMapping
        self._exchange_mappings: Dict[str, Dict[str, str]] = {}  # exchange_name -> {exchange_id -> internal_id}
        self._lock = asyncio.Lock()
    
    async def create_mapping(
        self,
        exchange_name: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new order mapping"""
        async with self._lock:
            internal_id = str(uuid.uuid4())
            
            mapping = OrderMapping(
                internal_id=internal_id,
                exchange_id="",  # Will be set when exchange responds
                exchange_name=exchange_name,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                metadata=metadata or {}
            )
            
            self._order_mappings[internal_id] = mapping
            
            # Initialize exchange mapping if needed
            if exchange_name not in self._exchange_mappings:
                self._exchange_mappings[exchange_name] = {}
            
            self.logger.debug(f"Created order mapping: {internal_id} for {exchange_name}")
            return internal_id
    
    async def update_exchange_id(self, internal_id: str, exchange_id: str) -> bool:
        """Update the exchange-specific ID for an order"""
        async with self._lock:
            if internal_id not in self._order_mappings:
                self.logger.warning(f"Order mapping not found: {internal_id}")
                return False
            
            mapping = self._order_mappings[internal_id]
            old_exchange_id = mapping.exchange_id
            
            # Update mapping
            mapping.exchange_id = exchange_id
            mapping.updated_at = datetime.now()
            
            # Update exchange mapping
            if old_exchange_id:
                self._exchange_mappings[mapping.exchange_name].pop(old_exchange_id, None)
            
            self._exchange_mappings[mapping.exchange_name][exchange_id] = internal_id
            
            self.logger.debug(f"Updated exchange ID: {internal_id} -> {exchange_id}")
            return True
    
    async def update_order_status(
        self,
        internal_id: str,
        status: OrderStatus,
        exchange_response: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update order status"""
        async with self._lock:
            if internal_id not in self._order_mappings:
                self.logger.warning(f"Order mapping not found: {internal_id}")
                return False
            
            mapping = self._order_mappings[internal_id]
            mapping.status = status
            mapping.updated_at = datetime.now()
            
            if exchange_response:
                mapping.exchange_response = exchange_response
            
            self.logger.debug(f"Updated order status: {internal_id} -> {status.value}")
            return True
    
    async def get_mapping_by_internal_id(self, internal_id: str) -> Optional[OrderMapping]:
        """Get order mapping by internal ID"""
        return self._order_mappings.get(internal_id)
    
    async def get_mapping_by_exchange_id(self, exchange_name: str, exchange_id: str) -> Optional[OrderMapping]:
        """Get order mapping by exchange ID"""
        if exchange_name not in self._exchange_mappings:
            return None
        
        internal_id = self._exchange_mappings[exchange_name].get(exchange_id)
        if not internal_id:
            return None
        
        return self._order_mappings.get(internal_id)
    
    async def get_orders_by_status(self, status: OrderStatus) -> List[OrderMapping]:
        """Get all orders with a specific status"""
        return [mapping for mapping in self._order_mappings.values() if mapping.status == status]
    
    async def get_orders_by_exchange(self, exchange_name: str) -> List[OrderMapping]:
        """Get all orders for a specific exchange"""
        return [
            mapping for mapping in self._order_mappings.values()
            if mapping.exchange_name == exchange_name
        ]
    
    async def get_orders_by_symbol(self, symbol: str) -> List[OrderMapping]:
        """Get all orders for a specific symbol"""
        return [mapping for mapping in self._order_mappings.values() if mapping.symbol == symbol]
    
    async def remove_mapping(self, internal_id: str) -> bool:
        """Remove an order mapping"""
        async with self._lock:
            if internal_id not in self._order_mappings:
                return False
            
            mapping = self._order_mappings[internal_id]
            
            # Remove from exchange mapping
            if mapping.exchange_id and mapping.exchange_name in self._exchange_mappings:
                self._exchange_mappings[mapping.exchange_name].pop(mapping.exchange_id, None)
            
            # Remove from main mapping
            del self._order_mappings[internal_id]
            
            self.logger.debug(f"Removed order mapping: {internal_id}")
            return True
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get order mapping statistics"""
        total_orders = len(self._order_mappings)
        status_counts = {}
        
        for status in OrderStatus:
            status_counts[status.value] = len([
                mapping for mapping in self._order_mappings.values()
                if mapping.status == status
            ])
        
        exchange_counts = {}
        for mapping in self._order_mappings.values():
            exchange_name = mapping.exchange_name
            exchange_counts[exchange_name] = exchange_counts.get(exchange_name, 0) + 1
        
        return {
            "total_orders": total_orders,
            "status_counts": status_counts,
            "exchange_counts": exchange_counts,
            "active_exchanges": list(exchange_counts.keys())
        }


class ExchangeRegistry:
    """Manages exchange registration and discovery"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ExchangeRegistry")
        self._exchanges: Dict[str, BaseExchange] = {}
        self._exchange_configs: Dict[str, Dict[str, Any]] = {}
        self._active_exchanges: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def register_exchange(
        self,
        name: str,
        exchange: BaseExchange,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an exchange"""
        async with self._lock:
            self._exchanges[name] = exchange
            self._exchange_configs[name] = config or {}
            self._active_exchanges.add(name)
            
            self.logger.info(f"Registered exchange: {name}")
    
    async def unregister_exchange(self, name: str) -> bool:
        """Unregister an exchange"""
        async with self._lock:
            if name not in self._exchanges:
                return False
            
            # Close exchange connection if possible
            exchange = self._exchanges[name]
            if hasattr(exchange, 'close'):
                try:
                    await exchange.close()
                except Exception as e:
                    self.logger.error(f"Error closing exchange {name}: {e}")
            
            del self._exchanges[name]
            self._exchange_configs.pop(name, None)
            self._active_exchanges.discard(name)
            
            self.logger.info(f"Unregistered exchange: {name}")
            return True
    
    async def get_exchange(self, name: str) -> Optional[BaseExchange]:
        """Get an exchange by name"""
        return self._exchanges.get(name)
    
    async def get_active_exchanges(self) -> List[str]:
        """Get list of active exchange names"""
        return list(self._active_exchanges)
    
    async def get_exchange_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get exchange configuration"""
        return self._exchange_configs.get(name)
    
    async def set_exchange_active(self, name: str, active: bool) -> bool:
        """Set exchange active status"""
        if name not in self._exchanges:
            return False
        
        if active:
            self._active_exchanges.add(name)
        else:
            self._active_exchanges.discard(name)
        
        self.logger.info(f"Set exchange {name} active: {active}")
        return True
    
    async def get_exchange_capabilities(self, name: str) -> Dict[str, Any]:
        """Get exchange capabilities"""
        exchange = self._exchanges.get(name)
        if not exchange:
            return {}
        
        capabilities = {
            "supports_market_orders": hasattr(exchange, 'create_order'),
            "supports_limit_orders": hasattr(exchange, 'create_order'),
            "supports_cancellation": hasattr(exchange, 'cancel_order'),
            "supports_position_info": hasattr(exchange, 'get_position_risk'),
            "supports_account_info": hasattr(exchange, 'get_account_info'),
            "supports_historical_data": hasattr(exchange, 'get_historical_klines'),
            "supports_streaming": any([
                hasattr(exchange, 'subscribe_trades'),
                hasattr(exchange, 'subscribe_ticker'),
                hasattr(exchange, 'subscribe_order_book')
            ])
        }
        
        return capabilities
    
    async def get_best_exchange_for_symbol(self, symbol: str) -> Optional[str]:
        """Determine the best exchange for a given symbol"""
        # Simple implementation - return first active exchange
        # In a real implementation, this would consider factors like:
        # - Exchange-specific symbol support
        # - Liquidity
        # - Fees
        # - Latency
        # - Reliability
        
        active_exchanges = await self.get_active_exchanges()
        if not active_exchanges:
            return None
        
        # For now, prefer binance, then okx
        preferred_order = ["binance", "okx", "gateio", "mexc"]
        
        for preferred in preferred_order:
            if preferred in active_exchanges:
                return preferred
        
        return active_exchanges[0]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get exchange registry statistics"""
        return {
            "total_exchanges": len(self._exchanges),
            "active_exchanges": len(self._active_exchanges),
            "exchange_names": list(self._exchanges.keys()),
            "active_exchange_names": list(self._active_exchanges)
        }


class BaseExchange(IExchangeClient, ABC):
    """
    Base class for all exchange implementations.
    Provides standardized method signatures and common functionality.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        trade_symbol: str,
        password: str | None = None,
        order_mapper: Optional[OrderIdMapper] = None,
        exchange_registry: Optional[ExchangeRegistry] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.trade_symbol = trade_symbol.upper()
        self.password = password
        self.exchange: Any | None = None  # Will be set by subclasses
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        
        # Order mapping and registry
        self.order_mapper = order_mapper or OrderIdMapper()
        self.exchange_registry = exchange_registry or ExchangeRegistry()
        self.exchange_name = self.__class__.__name__.lower().replace('exchange', '')

    @abstractmethod
    async def _initialize_exchange(self) -> None:
        """Initialize the exchange client. Must be implemented by subclasses."""

    async def _convert_to_market_data(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Convert raw exchange data to standardized MarketData format using shared standardizer."""
        try:
            from exchanges.shared import OHLCVDataStandardizer, DataSource
            
            # Determine exchange source
            exchange_name = self.__class__.__name__.lower()
            if 'binance' in exchange_name:
                source = DataSource.BINANCE
            elif 'bingx' in exchange_name:
                source = DataSource.BINGX
            elif 'okx' in exchange_name:
                source = DataSource.OKX
            elif 'mexc' in exchange_name:
                source = DataSource.MEXC
            elif 'gateio' in exchange_name:
                source = DataSource.GATEIO
            elif 'phemex' in exchange_name:
                source = DataSource.PHEMEX
            else:
                # Fallback to generic conversion
                return await self._convert_to_market_data_fallback(raw_data, symbol, interval)
            
            # Use standardized converter
            standardizer = OHLCVDataStandardizer()
            standardized_data = standardizer.standardize_data(raw_data, source, symbol, interval)
            
            # Convert to MarketData format for backward compatibility
            market_data = []
            for item in standardized_data:
                market_data.append(MarketData(
                    symbol=item.symbol,
                    timestamp=item.timestamp,
                    open=item.open,
                    high=item.high,
                    low=item.low,
                    close=item.close,
                    volume=item.volume,
                    interval=item.interval
                ))
            
            return market_data
            
        except ImportError:
            # Fallback to generic conversion if shared module not available
            return await self._convert_to_market_data_fallback(raw_data, symbol, interval)
        except Exception as e:
            self.logger.error(f"Failed to standardize data: {e}")
            return await self._convert_to_market_data_fallback(raw_data, symbol, interval)

    async def _convert_to_market_data_fallback(
        self,
        raw_data: list[dict[str, Any]],
        symbol: str,
        interval: str,
    ) -> list[MarketData]:
        """Fallback conversion method for backward compatibility."""
        market_data = []
        for item in raw_data:
            try:
                # Handle different timestamp field names
                timestamp = None
                for ts_field in ['timestamp', 'open_time', 'ts']:
                    if ts_field in item:
                        ts_value = item[ts_field]
                        if isinstance(ts_value, (int, float)):
                            # Convert milliseconds to datetime
                            timestamp = datetime.fromtimestamp(ts_value / 1000)
                        elif isinstance(ts_value, str):
                            timestamp = datetime.fromisoformat(ts_value.replace('Z', '+00:00'))
                        break
                
                if timestamp is None:
                    timestamp = datetime.utcnow()
                
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open=float(item.get('open', 0.0)),
                    high=float(item.get('high', 0.0)),
                    low=float(item.get('low', 0.0)),
                    close=float(item.get('close', 0.0)),
                    volume=float(item.get('volume', 0.0)),
                    interval=interval
                ))
            except Exception as e:
                self.logger.warning(f"Failed to convert data point: {e}")
                continue
        
        return market_data

    @abstractmethod
    async def _get_market_id(self, symbol: str) -> str:
        """Get the market ID for a given symbol."""

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[MarketData]:
        raw_data = await self._get_klines_raw(symbol, interval, limit)
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @abstractmethod
    async def _get_klines_raw(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw kline data from exchange."""

    async def get_account_info(self) -> dict[str, Any]:
        return await self._get_account_info_raw()

    @abstractmethod
    async def _get_account_info_raw(self) -> dict[str, Any]:
        """Get raw account information from exchange."""

    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Create an order with order ID mapping"""
        try:
            # Create internal order mapping
            internal_id = await self.order_mapper.create_mapping(
                exchange_name=self.exchange_name,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                metadata=metadata
            )
            
            # Create the order on the exchange
            exchange_response = await self._create_order_raw(symbol, side, order_type, quantity, price, None)
            
            # Extract exchange order ID from response
            exchange_order_id = self._extract_order_id_from_response(exchange_response)
            
            if exchange_order_id:
                # Update mapping with exchange order ID
                await self.order_mapper.update_exchange_id(internal_id, exchange_order_id)
                await self.order_mapper.update_order_status(
                    internal_id, 
                    OrderStatus.SUBMITTED, 
                    exchange_response
                )
            
            # Add internal ID to response
            response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
            response['internal_order_id'] = internal_id
            response['exchange_order_id'] = exchange_order_id
            response['exchange_name'] = self.exchange_name
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            # Update order status to rejected if mapping exists
            if 'internal_id' in locals():
                await self.order_mapper.update_order_status(
                    internal_id, 
                    OrderStatus.REJECTED, 
                    {"error": str(e)}
                )
            raise
    
    def _extract_order_id_from_response(self, response: Any) -> Optional[str]:
        """Extract order ID from exchange response"""
        if not isinstance(response, dict):
            return None
        
        # Try common order ID field names
        order_id_fields = ['id', 'orderId', 'order_id', 'clientOrderId', 'client_order_id']
        
        for field in order_id_fields:
            if field in response:
                return str(response[field])
        
        return None

    @abstractmethod
    async def _create_order_raw(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Create raw order on exchange."""

    async def get_position_risk(self, symbol: str) -> dict[str, Any]:
        return await self._get_position_risk_raw(symbol)

    @abstractmethod
    async def _get_position_risk_raw(self, symbol: str) -> dict[str, Any]:
        """Get raw position risk information from exchange."""

    # Additional standardized helpers
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_time: int | datetime,
        end_time: int | datetime,
        limit: int = 1000,
        **kwargs  # Accept additional parameters for backwards compatibility
    ) -> list[MarketData]:
        # Handle both datetime and milliseconds parameters for backwards compatibility
        if isinstance(start_time, datetime):
            start_time_ms = int(start_time.timestamp() * 1000)
        else:
            start_time_ms = start_time

        if isinstance(end_time, datetime):
            end_time_ms = int(end_time.timestamp() * 1000)
        else:
            end_time_ms = end_time

        raw_data = await self._get_historical_klines_raw(
            symbol,
            interval,
            start_time_ms,
            end_time_ms,
            limit,
        )
        return await self._convert_to_market_data(raw_data, symbol, interval)

    @abstractmethod
    async def _get_historical_klines_raw(
        self,
        symbol: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical kline data from exchange."""

    async def get_historical_agg_trades(
        self,
        symbol: str,
        start_time: int | datetime,
        end_time: int | datetime,
        limit: int = 1000,
        **kwargs  # Accept additional parameters for backwards compatibility
    ) -> list[dict[str, Any]]:
        # Handle both datetime and milliseconds parameters for backwards compatibility
        if isinstance(start_time, datetime):
            start_time_ms = int(start_time.timestamp() * 1000)
        else:
            start_time_ms = start_time

        if isinstance(end_time, datetime):
            end_time_ms = int(end_time.timestamp() * 1000)
        else:
            end_time_ms = end_time

        return await self._get_historical_agg_trades_raw(
            symbol,
            start_time_ms,
            end_time_ms,
            limit,
        )

    @abstractmethod
    async def _get_historical_agg_trades_raw(
        self,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Get raw historical aggregated trades from exchange."""

    async def get_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return await self._get_open_orders_raw(symbol)

    @abstractmethod
    async def _get_open_orders_raw(self, symbol: str | None) -> list[dict[str, Any]]:
        """Get raw open orders from exchange."""

    async def cancel_order(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel order with mapping support"""
        try:
            # Check if this is an internal order ID
            mapping = await self.order_mapper.get_mapping_by_internal_id(str(order_id))
            
            if mapping:
                # Use exchange order ID for the actual cancellation
                exchange_order_id = mapping.exchange_id
                if not exchange_order_id:
                    return {
                        "status": "error",
                        "internal_order_id": order_id,
                        "error": "Exchange order ID not available"
                    }
                
                # Cancel order on exchange
                exchange_response = await self._cancel_order_raw(symbol, exchange_order_id)
                
                # Update mapping status
                await self.order_mapper.update_order_status(
                    str(order_id),
                    OrderStatus.CANCELLED,
                    exchange_response
                )
                
                # Add mapping info to response
                response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                response['internal_order_id'] = order_id
                response['exchange_order_id'] = exchange_order_id
                response['exchange_name'] = self.exchange_name
                response['mapped_status'] = OrderStatus.CANCELLED.value
                
                return response
            else:
                # Assume it's an exchange order ID
                exchange_response = await self._cancel_order_raw(symbol, order_id)
                
                # Try to find mapping by exchange ID
                mapping = await self.order_mapper.get_mapping_by_exchange_id(self.exchange_name, str(order_id))
                if mapping:
                    await self.order_mapper.update_order_status(
                        mapping.internal_id,
                        OrderStatus.CANCELLED,
                        exchange_response
                    )
                    
                    response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                    response['internal_order_id'] = mapping.internal_id
                    response['exchange_order_id'] = order_id
                    response['exchange_name'] = self.exchange_name
                    response['mapped_status'] = OrderStatus.CANCELLED.value
                    
                    return response
                else:
                    # No mapping found, return raw response
                    response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                    response['exchange_order_id'] = order_id
                    response['exchange_name'] = self.exchange_name
                    response['mapped_status'] = 'unknown'
                    
                    return response
                    
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return {
                "status": "error",
                "error": str(e),
                "order_id": order_id
            }

    @abstractmethod
    async def _cancel_order_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Cancel raw order on exchange."""

    async def get_order_status(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get order status with mapping support"""
        try:
            # Check if this is an internal order ID
            mapping = await self.order_mapper.get_mapping_by_internal_id(str(order_id))
            
            if mapping:
                # Use exchange order ID for the actual query
                exchange_order_id = mapping.exchange_id
                if not exchange_order_id:
                    return {
                        "status": "unknown",
                        "internal_order_id": order_id,
                        "error": "Exchange order ID not available"
                    }
                
                # Get status from exchange
                exchange_response = await self._get_order_status_raw(symbol, exchange_order_id)
                
                # Update mapping with new status
                status = self._map_exchange_status_to_internal(exchange_response)
                await self.order_mapper.update_order_status(
                    str(order_id),
                    status,
                    exchange_response
                )
                
                # Add mapping info to response
                response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                response['internal_order_id'] = order_id
                response['exchange_order_id'] = exchange_order_id
                response['exchange_name'] = self.exchange_name
                response['mapped_status'] = status.value
                
                return response
            else:
                # Assume it's an exchange order ID
                exchange_response = await self._get_order_status_raw(symbol, order_id)
                
                # Try to find mapping by exchange ID
                mapping = await self.order_mapper.get_mapping_by_exchange_id(self.exchange_name, str(order_id))
                if mapping:
                    status = self._map_exchange_status_to_internal(exchange_response)
                    await self.order_mapper.update_order_status(
                        mapping.internal_id,
                        status,
                        exchange_response
                    )
                    
                    response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                    response['internal_order_id'] = mapping.internal_id
                    response['exchange_order_id'] = order_id
                    response['exchange_name'] = self.exchange_name
                    response['mapped_status'] = status.value
                    
                    return response
                else:
                    # No mapping found, return raw response
                    response = exchange_response.copy() if isinstance(exchange_response, dict) else {}
                    response['exchange_order_id'] = order_id
                    response['exchange_name'] = self.exchange_name
                    response['mapped_status'] = 'unknown'
                    
                    return response
                    
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "order_id": order_id
            }
    
    def _map_exchange_status_to_internal(self, exchange_response: Any) -> OrderStatus:
        """Map exchange order status to internal status"""
        if not isinstance(exchange_response, dict):
            return OrderStatus.UNKNOWN
        
        status = exchange_response.get('status', '').lower()
        
        status_mapping = {
            'new': OrderStatus.SUBMITTED,
            'pending': OrderStatus.PENDING,
            'partially_filled': OrderStatus.PARTIALLY_FILLED,
            'filled': OrderStatus.FILLED,
            'cancelled': OrderStatus.CANCELLED,
            'canceled': OrderStatus.CANCELLED,
            'rejected': OrderStatus.REJECTED,
            'expired': OrderStatus.EXPIRED,
            'done': OrderStatus.FILLED,
            'closed': OrderStatus.FILLED,
        }
        
        return status_mapping.get(status, OrderStatus.UNKNOWN)

    @abstractmethod
    async def _get_order_status_raw(self, symbol: str, order_id: Any) -> dict[str, Any]:
        """Get raw order status from exchange."""

    async def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Best-effort leverage setter using underlying client if supported."""
        try:
            market_id = await self._get_market_id(symbol)
        except Exception:
            market_id = symbol

        if not self.exchange:
            return False

        attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
            ("set_leverage", (leverage, market_id), {}),
            ("set_leverage", (), {"leverage": leverage, "symbol": market_id}),
            ("setLeverage", (leverage, market_id), {}),
        ]

        for method, args, kwargs in attempts:
            if hasattr(self.exchange, method):
                try:
                    await getattr(self.exchange, method)(*args, **kwargs)
                    return True
                except Exception:
                    continue
        return False

    async def set_margin_mode(self, symbol: str, mode: str) -> bool:
        """Best-effort margin mode setter using underlying client if supported."""
        try:
            market_id = await self._get_market_id(symbol)
        except Exception:
            market_id = symbol

        if not self.exchange:
            return False

        attempts: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
            ("set_margin_mode", (mode, market_id), {}),
            ("set_margin_mode", (), {"marginMode": mode, "symbol": market_id}),
            ("setMarginMode", (mode, market_id), {}),
        ]

        for method, args, kwargs in attempts:
            if hasattr(self.exchange, method):
                try:
                    await getattr(self.exchange, method)(*args, **kwargs)
                    return True
                except Exception:
                    continue
        return False

    async def close(self) -> None:
        """Close the exchange connection if supported by underlying client."""
        if self.exchange and hasattr(self.exchange, "close"):
            await self.exchange.close()
    
    # Order mapping methods
    async def get_order_mapping(self, internal_id: str) -> Optional[OrderMapping]:
        """Get order mapping by internal ID"""
        return await self.order_mapper.get_mapping_by_internal_id(internal_id)
    
    async def get_orders_by_status(self, status: OrderStatus) -> List[OrderMapping]:
        """Get orders by status for this exchange"""
        all_orders = await self.order_mapper.get_orders_by_status(status)
        return [order for order in all_orders if order.exchange_name == self.exchange_name]
    
    async def get_orders_by_symbol(self, symbol: str) -> List[OrderMapping]:
        """Get orders by symbol for this exchange"""
        all_orders = await self.order_mapper.get_orders_by_symbol(symbol)
        return [order for order in all_orders if order.exchange_name == self.exchange_name]
    
    async def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics for this exchange"""
        stats = await self.order_mapper.get_statistics()
        exchange_orders = await self.order_mapper.get_orders_by_exchange(self.exchange_name)
        
        exchange_stats = {
            "exchange_name": self.exchange_name,
            "total_orders": len(exchange_orders),
            "status_counts": {}
        }
        
        for status in OrderStatus:
            count = len([order for order in exchange_orders if order.status == status])
            exchange_stats["status_counts"][status.value] = count
        
        return exchange_stats

    def _convert_timestamp(self, timestamp: Any) -> datetime:
        """Convert exchange timestamp to datetime."""
        if isinstance(timestamp, int | float):
            # Assume milliseconds if timestamp is large
            if timestamp > 1e10:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        if isinstance(timestamp, str):
            # Try to parse as ISO format, fall back to common formats
            try:
                return datetime.fromisoformat(timestamp)
            except ValueError:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except ValueError:
                        continue
                msg = f"Unable to parse timestamp: {timestamp}"
                raise ValueError(msg)
        msg = f"Unsupported timestamp type: {type(timestamp)}"
        raise ValueError(msg)

    # --- Optional streaming hooks (to be implemented by subclasses as needed) ---
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    async def subscribe_order_book(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        raise NotImplementedError

    # --- Convenience polling helpers ---
    async def fetch_price(self, symbol: str) -> float | None:
        """Fetch current price using ticker, falling back to order book mid."""
        try:
            # Prefer a direct ticker if subclass implements get_ticker
            if hasattr(self, "get_ticker"):
                ticker = await self.get_ticker(symbol)  # type: ignore[attr-defined]
                if ticker:
                    last = ticker.get("last") or ticker.get("mark") or ticker.get("close")
                    if last is not None:
                        return float(last)
                    bid = ticker.get("bid")
                    ask = ticker.get("ask")
                    if bid is not None and ask is not None:
                        return (float(bid) + float(ask)) / 2.0
            # Fallback to order book mid
            if hasattr(self, "get_order_book"):
                book = await self.get_order_book(symbol, 5)  # type: ignore[attr-defined]
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                best_bid = float(bids[0][0]) if bids else None
                best_ask = float(asks[0][0]) if asks else None
                if best_bid is not None and best_ask is not None:
                    return (best_bid + best_ask) / 2.0
                if best_bid is not None:
                    return best_bid
                if best_ask is not None:
                    return best_ask
        except Exception:
            return None
        return None

    async def get_liquidation_price(self, symbol: str) -> float | None:
        """Best-effort liquidation price for current position on symbol."""
        try:
            risk = await self.get_position_risk(symbol)
            # Try common ccxt fields
            if isinstance(risk, list) and risk:
                # Find matching symbol
                for position in risk:
                    inst = position.get("symbol") or position.get("info", {}).get("symbol")
                    if inst and inst.replace("-", "").replace("_", "").upper().startswith(
                        symbol.upper().replace("USDT", ""),
                    ):
                        liq = (
                            position.get("liquidationPrice")
                            or position.get("liqPrice")
                            or position.get("liquidation_price")
                        )
                        if liq:
                            return float(liq)
                # Otherwise take first
                pos0 = risk[0]
                liq = pos0.get("liquidationPrice") or pos0.get("liqPrice") or pos0.get("liquidation_price")
                if liq:
                    return float(liq)
        except Exception:
            return None
        return None

    # --- Default CCXT-based helpers (can be overridden by subclasses) ---
    async def get_ticker(self, symbol: str | None = None) -> dict[str, Any]:
        """Default ticker fetch using ccxt if underlying client is set."""
        try:
            if not self.exchange:
                return {}
            market_id = await self._get_market_id(symbol) if symbol else None  # type: ignore[arg-type]
            if market_id:
                return await self.exchange.fetch_ticker(market_id)  # type: ignore[union-attr]
            # All tickers fallback
            tickers = await self.exchange.fetch_tickers()  # type: ignore[union-attr]
            return tickers or {}
        except Exception:
            return {}

    async def get_order_book(self, symbol: str, limit: int = 10) -> dict[str, Any]:
        """Default order book fetch using ccxt if underlying client is set."""
        try:
            if not self.exchange:
                return {}
            market_id = await self._get_market_id(symbol)
            return await self.exchange.fetch_order_book(market_id, limit)  # type: ignore[union-attr]
        except Exception:
            return {}


class MultiExchangeBase:
    """
    Base class for multi-exchange operations.
    Provides functionality to route requests to multiple exchanges.
    """

    def __init__(self, exchanges: Dict[str, BaseExchange]):
        self.exchanges = exchanges
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")

    async def broadcast_to_all_exchanges(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Broadcast an operation to all configured exchanges.

        Args:
            operation: The method name to call on each exchange
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Dict mapping exchange names to their responses
        """
        results = {}

        for exchange_name, exchange in self.exchanges.items():
            try:
                if not exchange:
                    self.logger.warning(f"Exchange {exchange_name} is not initialized")
                    results[exchange_name] = {"error": "Exchange not initialized"}
                    continue

                # Get the method from the exchange
                method = getattr(exchange, operation, None)
                if not method:
                    self.logger.warning(f"Exchange {exchange_name} does not support operation {operation}")
                    results[exchange_name] = {"error": f"Operation {operation} not supported"}
                    continue

                # Call the operation
                if asyncio.iscoroutinefunction(method):
                    response = await method(*args, **kwargs)
                else:
                    response = method(*args, **kwargs)

                results[exchange_name] = {"success": True, "data": response}

            except Exception as e:
                self.logger.error(f"Error calling {operation} on {exchange_name}: {e}")
                results[exchange_name] = {"error": str(e)}

        return results

    async def route_to_primary_exchange(
        self,
        primary_exchange: str,
        operation: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route an operation to a specific primary exchange.

        Args:
            primary_exchange: Name of the primary exchange
            operation: The method name to call
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Response from the primary exchange
        """
        if primary_exchange not in self.exchanges:
            return {"error": f"Primary exchange {primary_exchange} not found"}

        exchange = self.exchanges[primary_exchange]
        if not exchange:
            return {"error": f"Primary exchange {primary_exchange} is not initialized"}

        try:
            method = getattr(exchange, operation, None)
            if not method:
                return {"error": f"Operation {operation} not supported by {primary_exchange}"}

            if asyncio.iscoroutinefunction(method):
                response = await method(*args, **kwargs)
            else:
                response = method(*args, **kwargs)

            return {"success": True, "exchange": primary_exchange, "data": response}

        except Exception as e:
            self.logger.error(f"Error calling {operation} on {primary_exchange}: {e}")
            return {"error": str(e)}

    async def route_with_failover(
        self,
        primary_exchange: str,
        failover_exchanges: List[str],
        operation: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Route an operation with failover support.

        Args:
            primary_exchange: Name of the primary exchange
            failover_exchanges: List of failover exchanges
            operation: The method name to call
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Response from the first successful exchange
        """
        all_exchanges = [primary_exchange] + [ex for ex in failover_exchanges if ex != primary_exchange]

        for exchange_name in all_exchanges:
            if exchange_name not in self.exchanges:
                continue

            exchange = self.exchanges[exchange_name]
            if not exchange:
                continue

            try:
                method = getattr(exchange, operation, None)
                if not method:
                    continue

                if asyncio.iscoroutinefunction(method):
                    response = await method(*args, **kwargs)
                else:
                    response = method(*args, **kwargs)

                return {
                    "success": True,
                    "exchange": exchange_name,
                    "data": response,
                    "failover_used": exchange_name != primary_exchange
                }

            except Exception as e:
                self.logger.warning(f"Exchange {exchange_name} failed for {operation}: {e}")
                continue

        return {"error": "All exchanges failed"}

    async def aggregate_responses(
        self,
        operation: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Aggregate responses from all exchanges.

        Args:
            operation: The method name to call
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Aggregated response from all exchanges
        """
        results = await self.broadcast_to_all_exchanges(operation, *args, **kwargs)

        # Filter successful responses
        successful_responses = {
            exchange: response["data"]
            for exchange, response in results.items()
            if response.get("success") and "data" in response
        }

        failed_responses = {
            exchange: response.get("error", "Unknown error")
            for exchange, response in results.items()
            if not response.get("success")
        }

        return {
            "aggregated_data": successful_responses,
            "successful_exchanges": list(successful_responses.keys()),
            "failed_exchanges": failed_responses,
            "total_exchanges": len(self.exchanges),
            "successful_count": len(successful_responses)
        }

    async def get_best_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get the best price across all exchanges.

        Args:
            symbol: Trading symbol

        Returns:
            Best price information
        """
        results = await self.broadcast_to_all_exchanges("get_ticker", symbol)

        best_bid = 0
        best_ask = float('inf')
        best_exchange_bid = None
        best_exchange_ask = None
        all_prices = {}

        for exchange_name, response in results.items():
            if response.get("success") and "data" in response:
                ticker = response["data"]
                if ticker:
                    bid = float(ticker.get("bid", 0))
                    ask = float(ticker.get("ask", 0))

                    all_prices[exchange_name] = {"bid": bid, "ask": ask}

                    if bid > best_bid:
                        best_bid = bid
                        best_exchange_bid = exchange_name

                    if ask < best_ask:
                        best_ask = ask
                        best_exchange_ask = exchange_name

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "best_bid_exchange": best_exchange_bid,
            "best_ask_exchange": best_exchange_ask,
            "spread": best_ask - best_bid if best_ask != float('inf') and best_bid > 0 else 0,
            "all_prices": all_prices
        }

    async def close_all_exchanges(self) -> None:
        """Close all exchange connections."""
        for exchange_name, exchange in self.exchanges.items():
            try:
                if exchange:
                    await exchange.close()
                    self.logger.info(f"Closed connection to {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error closing connection to {exchange_name}: {e}")


class ExchangeMessageHandler:
    """
    Handles messages between the trading system and exchanges.
    """

    def __init__(self, multi_exchange_base: MultiExchangeBase):
        self.multi_exchange = multi_exchange_base
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.pending_orders: Dict[str, Dict[str, Any]] = {}
        self.order_responses: Dict[str, Any] = {}

    async def send_order_to_all_exchanges(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send order to all exchanges.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            order_type: Order type ("market", "limit", etc.)
            quantity: Order quantity
            price: Order price (optional)
            **kwargs: Additional order parameters

        Returns:
            Responses from all exchanges
        """
        operation = "create_order"
        args = (symbol, side, quantity, price, order_type)
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "price"}

        # Add to pending orders
        order_id = f"{symbol}_{side}_{int(time.time() * 1000)}"
        self.pending_orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "kwargs": kwargs_clean
        }

        try:
            # Broadcast to all exchanges
            responses = await self.multi_exchange.broadcast_to_all_exchanges(
                operation, *args, **kwargs_clean
            )

            # Store responses
            self.order_responses[order_id] = responses

            return {
                "order_id": order_id,
                "responses": responses,
                "successful_exchanges": [
                    ex for ex, resp in responses.items()
                    if resp.get("success")
                ]
            }

        except Exception as e:
            self.logger.error(f"Error sending order to all exchanges: {e}")
            return {"error": str(e)}

    async def send_order_to_primary(
        self,
        primary_exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send order to primary exchange with failover.

        Args:
            primary_exchange: Primary exchange name
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price (optional)
            **kwargs: Additional parameters

        Returns:
            Response from exchange
        """
        operation = "create_order"
        args = (symbol, side, quantity, price, order_type)
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "price"}

        try:
            response = await self.multi_exchange.route_with_failover(
                primary_exchange, [], operation, *args, **kwargs_clean
            )

            return response

        except Exception as e:
            self.logger.error(f"Error sending order to primary exchange: {e}")
            return {"error": str(e)}

    async def get_aggregated_data(
        self,
        data_type: str,
        symbol: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get aggregated data from all exchanges.

        Args:
            data_type: Type of data to fetch
            symbol: Trading symbol
            **kwargs: Additional parameters

        Returns:
            Aggregated data from all exchanges
        """
        try:
            response = await self.multi_exchange.aggregate_responses(
                f"get_{data_type}", symbol, **kwargs
            )

            return response

        except Exception as e:
            self.logger.error(f"Error getting aggregated data: {e}")
            return {"error": str(e)}

    async def get_best_execution_venue(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> str:
        """
        Determine the best execution venue based on various factors.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity

        Returns:
            Name of the best exchange for execution
        """
        try:
            # Get best price information
            price_info = await self.multi_exchange.get_best_price(symbol)

            if side == "buy":
                # For buys, prefer lowest ask price
                best_exchange = price_info.get("best_ask_exchange")
            else:
                # For sells, prefer highest bid price
                best_exchange = price_info.get("best_bid_exchange")

            if best_exchange:
                return best_exchange

            # Fallback to primary exchange
            if hasattr(self.multi_exchange, 'exchanges'):
                # Return first available exchange as fallback
                for exchange_name in self.multi_exchange.exchanges.keys():
                    return exchange_name

            return "unknown"

        except Exception as e:
            self.logger.error(f"Error determining best execution venue: {e}")
            return "unknown"

    async def get_order_status_all_exchanges(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status from all exchanges.

        Args:
            order_id: Order ID to check

        Returns:
            Status from all exchanges
        """
        if order_id not in self.pending_orders:
            return {"error": "Order not found in pending orders"}

        order_info = self.pending_orders[order_id]
        symbol = order_info["symbol"]

        # This would need to be implemented based on how order IDs are mapped
        # For now, return a placeholder
        return {
            "order_id": order_id,
            "status": "checking",
            "exchanges": list(self.multi_exchange.exchanges.keys())
        }

    async def cancel_order_all_exchanges(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order on all exchanges.

        Args:
            order_id: Order ID to cancel

        Returns:
            Cancellation results from all exchanges
        """
        if order_id not in self.pending_orders:
            return {"error": "Order not found in pending orders"}

        # This would need to be implemented with proper order ID mapping
        # For now, return a placeholder
        return {
            "order_id": order_id,
            "cancelled_exchanges": [],
            "failed_exchanges": list(self.multi_exchange.exchanges.keys())
        }


class ExchangeResponseHandler:
    """
    Handles responses from exchanges and routes them back to the appropriate recipients.
    """

    def __init__(self, multi_exchange_base: MultiExchangeBase):
        self.multi_exchange = multi_exchange_base
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.response_callbacks: Dict[str, List[Callable]] = {}

    def register_response_callback(self, operation_id: str, callback: Callable) -> None:
        """
        Register a callback for a specific operation.

        Args:
            operation_id: Unique identifier for the operation
            callback: Callback function to handle the response
        """
        if operation_id not in self.response_callbacks:
            self.response_callbacks[operation_id] = []
        self.response_callbacks[operation_id].append(callback)

    async def handle_response(self, operation_id: str, response: Dict[str, Any]) -> None:
        """
        Handle a response from an exchange operation.

        Args:
            operation_id: Unique identifier for the operation
            response: Response data from the exchange
        """
        try:
            # Call registered callbacks
            if operation_id in self.response_callbacks:
                for callback in self.response_callbacks[operation_id]:
                    try:
                        await callback(response)
                    except Exception as e:
                        self.logger.error(f"Error in response callback for {operation_id}: {e}")

        except Exception as e:
            self.logger.error(f"Error handling response for {operation_id}: {e}")

    async def handle_order_response(
        self,
        order_id: str,
        exchange_name: str,
        response: Dict[str, Any]
    ) -> None:
        """
        Handle order response from a specific exchange.

        Args:
            order_id: Order identifier
            exchange_name: Name of the exchange
            response: Order response data
        """
        try:
            # Create a structured response
            structured_response = {
                "order_id": order_id,
                "exchange": exchange_name,
                "success": "error" not in response,
                "data": response.get("data", response),
                "error": response.get("error"),
                "timestamp": datetime.now().isoformat()
            }

            # Route back to the appropriate handler
            await self.handle_response(f"order_{order_id}", structured_response)

        except Exception as e:
            self.logger.error(f"Error handling order response: {e}")

    async def handle_data_response(
        self,
        data_type: str,
        symbol: str,
        exchange_name: str,
        response: Dict[str, Any]
    ) -> None:
        """
        Handle data response from a specific exchange.

        Args:
            data_type: Type of data
            symbol: Trading symbol
            exchange_name: Name of the exchange
            response: Data response
        """
        try:
            structured_response = {
                "data_type": data_type,
                "symbol": symbol,
                "exchange": exchange_name,
                "success": "error" not in response,
                "data": response.get("data", response),
                "error": response.get("error"),
                "timestamp": datetime.now().isoformat()
            }

            await self.handle_response(f"data_{data_type}_{symbol}", structured_response)

        except Exception as e:
            self.logger.error(f"Error handling data response: {e}")

    async def aggregate_and_route_responses(
        self,
        operation_type: str,
        responses: Dict[str, Any]
    ) -> None:
        """
        Aggregate responses and route them appropriately.

        Args:
            operation_type: Type of operation
            responses: Responses from all exchanges
        """
        try:
            # Aggregate successful responses
            successful_responses = {
                exchange: response["data"]
                for exchange, response in responses.items()
                if response.get("success")
            }

            # Create aggregated response
            aggregated_response = {
                "operation_type": operation_type,
                "successful_responses": successful_responses,
                "all_responses": responses,
                "success_count": len(successful_responses),
                "total_count": len(responses),
                "timestamp": datetime.now().isoformat()
            }

            # Route the aggregated response
            await self.handle_response(f"aggregated_{operation_type}", aggregated_response)

        except Exception as e:
            self.logger.error(f"Error aggregating responses: {e}")