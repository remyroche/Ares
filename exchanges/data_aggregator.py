"""
Data Aggregator

Aggregates data from multiple exchanges and provides unified data access.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .exchange_registry import ExchangeRegistry


class DataType(Enum):
    """Data type enumeration"""
    TICKER = "ticker"
    KLINES = "klines"
    TRADES = "trades"
    ORDERBOOK = "orderbook"
    ACCOUNT_INFO = "account_info"
    POSITION_INFO = "position_info"
    OPEN_ORDERS = "open_orders"


@dataclass
class AggregatedData:
    """Aggregated data structure"""
    symbol: str
    data_type: DataType
    timestamp: datetime
    exchange_data: Dict[str, Any]  # exchange_name -> data
    aggregated_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataAggregator:
    """Aggregates data from multiple exchanges"""
    
    def __init__(self, exchange_registry: ExchangeRegistry):
        self.exchange_registry = exchange_registry
        self.logger = logging.getLogger(__name__)
        
        # Data caching
        self.data_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 5.0  # seconds
        
        # Data aggregation rules
        self.aggregation_rules = {
            DataType.TICKER: self._aggregate_ticker_data,
            DataType.KLINES: self._aggregate_klines_data,
            DataType.TRADES: self._aggregate_trades_data,
            DataType.ORDERBOOK: self._aggregate_orderbook_data,
            DataType.ACCOUNT_INFO: self._aggregate_account_data,
            DataType.POSITION_INFO: self._aggregate_position_data,
            DataType.OPEN_ORDERS: self._aggregate_orders_data
        }
        
        # Statistics
        self.data_stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "by_type": {},
            "by_exchange": {},
            "errors": 0
        }
        
        # Cache cleanup
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> None:
        """Start data aggregator"""
        if self._running:
            return
            
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_cache())
        self.logger.info("Data aggregator started")
    
    async def stop(self) -> None:
        """Stop data aggregator"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Data aggregator stopped")
    
    async def get_data(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Get data from specified exchange"""
        try:
            self.data_stats["total_requests"] += 1
            self._update_stats_by_type(data_type)
            self._update_stats_by_exchange(exchange)
            
            # Check cache first
            cache_key = f"{exchange}_{symbol}_{data_type}_{hash(str(kwargs))}"
            cached_data = self._get_from_cache(cache_key)
            
            if cached_data:
                self.data_stats["cache_hits"] += 1
                return {
                    "success": True,
                    "data": cached_data,
                    "cached": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            self.data_stats["cache_misses"] += 1
            
            # Get exchange instance
            exchange_instance = await self.exchange_registry.get_exchange(exchange)
            if not exchange_instance:
                raise ValueError(f"Exchange {exchange} not found or not available")
            
            # Get data from exchange
            data = await self._fetch_data_from_exchange(
                exchange_instance, symbol, data_type, **kwargs
            )
            
            # Cache the data
            self._cache_data(cache_key, data)
            
            return {
                "success": True,
                "data": data,
                "cached": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.data_stats["errors"] += 1
            self.logger.error(f"Error getting data from {exchange}: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_aggregated_data(
        self,
        symbol: str,
        data_type: str,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Get aggregated data from multiple exchanges"""
        try:
            # Get available exchanges
            if exchanges is None:
                exchanges = await self.exchange_registry.get_registered_exchanges()
            
            # Fetch data from all exchanges
            exchange_data = {}
            for exchange in exchanges:
                try:
                    result = await self.get_data(exchange, symbol, data_type, **kwargs)
                    if result["success"]:
                        exchange_data[exchange] = result["data"]
                except Exception as e:
                    self.logger.warning(f"Failed to get data from {exchange}: {e}")
                    continue
            
            if not exchange_data:
                return {
                    "success": False,
                    "error": "No data available from any exchange",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Aggregate data
            data_type_enum = DataType(data_type)
            aggregator = self.aggregation_rules.get(data_type_enum)
            
            if aggregator:
                aggregated_data = await aggregator(exchange_data)
            else:
                aggregated_data = exchange_data
            
            return {
                "success": True,
                "data": aggregated_data,
                "exchange_data": exchange_data,
                "exchanges": list(exchange_data.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting aggregated data: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_data_from_exchange(
        self,
        exchange_instance: Any,
        symbol: str,
        data_type: str,
        **kwargs
    ) -> Any:
        """Fetch data from exchange instance"""
        data_type_enum = DataType(data_type)
        
        if data_type_enum == DataType.TICKER:
            return await exchange_instance.get_ticker(symbol)
        
        elif data_type_enum == DataType.KLINES:
            interval = kwargs.get("interval", "1m")
            limit = kwargs.get("limit", 100)
            return await exchange_instance.get_klines(symbol, interval, limit)
        
        elif data_type_enum == DataType.TRADES:
            limit = kwargs.get("limit", 100)
            # Note: get_recent_trades might not be implemented in all exchanges
            if hasattr(exchange_instance, 'get_recent_trades'):
                return await exchange_instance.get_recent_trades(symbol, limit)
            else:
                return []
        
        elif data_type_enum == DataType.ORDERBOOK:
            limit = kwargs.get("limit", 20)
            return await exchange_instance.get_order_book(symbol, limit)
        
        elif data_type_enum == DataType.ACCOUNT_INFO:
            return await exchange_instance.get_account_info()
        
        elif data_type_enum == DataType.POSITION_INFO:
            return await exchange_instance.get_position_risk(symbol)
        
        elif data_type_enum == DataType.OPEN_ORDERS:
            return await exchange_instance.get_open_orders(symbol)
        
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
    
    async def _aggregate_ticker_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate ticker data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Calculate volume-weighted average price
        total_volume = 0
        weighted_price = 0
        
        for exchange, ticker in exchange_data.items():
            volume = float(ticker.get("volume", 0))
            last_price = float(ticker.get("last", ticker.get("close", 0)))
            
            total_volume += volume
            weighted_price += last_price * volume
        
        avg_price = weighted_price / total_volume if total_volume > 0 else 0
        
        # Calculate best bid/ask across exchanges
        best_bid = 0
        best_ask = float('inf')
        
        for exchange, ticker in exchange_data.items():
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            
            if bid > best_bid:
                best_bid = bid
            if ask < best_ask:
                best_ask = ask
        
        # Calculate spread
        spread = best_ask - best_bid if best_ask != float('inf') and best_bid > 0 else 0
        
        return {
            "symbol": list(exchange_data.values())[0].get("symbol", ""),
            "last_price": avg_price,
            "best_bid": best_bid,
            "best_ask": best_ask if best_ask != float('inf') else 0,
            "spread": spread,
            "total_volume": total_volume,
            "exchange_count": len(exchange_data),
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_klines_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate klines data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # For klines, we'll return the data from the first exchange
        # In a more sophisticated implementation, we might average OHLCV data
        first_exchange = list(exchange_data.keys())[0]
        return {
            "data": exchange_data[first_exchange],
            "source_exchange": first_exchange,
            "available_exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_trades_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate trades data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Combine trades from all exchanges
        all_trades = []
        for exchange, trades in exchange_data.items():
            if isinstance(trades, list):
                for trade in trades:
                    trade["source_exchange"] = exchange
                    all_trades.append(trade)
        
        # Sort by timestamp
        all_trades.sort(key=lambda x: x.get("timestamp", x.get("time", 0)))
        
        return {
            "trades": all_trades,
            "total_trades": len(all_trades),
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_orderbook_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate orderbook data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Combine bids and asks from all exchanges
        all_bids = []
        all_asks = []
        
        for exchange, orderbook in exchange_data.items():
            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])
            
            for bid in bids:
                all_bids.append({
                    "price": float(bid[0]),
                    "quantity": float(bid[1]),
                    "exchange": exchange
                })
            
            for ask in asks:
                all_asks.append({
                    "price": float(ask[0]),
                    "quantity": float(ask[1]),
                    "exchange": exchange
                })
        
        # Sort bids descending, asks ascending
        all_bids.sort(key=lambda x: x["price"], reverse=True)
        all_asks.sort(key=lambda x: x["price"])
        
        return {
            "bids": all_bids[:20],  # Top 20 bids
            "asks": all_asks[:20],  # Top 20 asks
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_account_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate account data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Combine account balances
        total_balance = 0
        total_available = 0
        all_balances = {}
        
        for exchange, account in exchange_data.items():
            balance = float(account.get("totalBalance", 0))
            available = float(account.get("availableBalance", 0))
            
            total_balance += balance
            total_available += available
            
            all_balances[exchange] = {
                "total_balance": balance,
                "available_balance": available
            }
        
        return {
            "total_balance": total_balance,
            "total_available": total_available,
            "exchange_balances": all_balances,
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_position_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate position data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Combine positions
        all_positions = {}
        total_exposure = 0
        
        for exchange, positions in exchange_data.items():
            if isinstance(positions, list):
                for position in positions:
                    symbol = position.get("symbol", "")
                    size = float(position.get("size", 0))
                    mark_price = float(position.get("markPrice", 0))
                    
                    if symbol not in all_positions:
                        all_positions[symbol] = {
                            "total_size": 0,
                            "total_value": 0,
                            "exchanges": []
                        }
                    
                    all_positions[symbol]["total_size"] += size
                    all_positions[symbol]["total_value"] += size * mark_price
                    all_positions[symbol]["exchanges"].append(exchange)
                    
                    total_exposure += abs(size * mark_price)
            else:
                # Single position
                symbol = positions.get("symbol", "")
                size = float(positions.get("size", 0))
                mark_price = float(positions.get("markPrice", 0))
                
                if symbol not in all_positions:
                    all_positions[symbol] = {
                        "total_size": 0,
                        "total_value": 0,
                        "exchanges": []
                    }
                
                all_positions[symbol]["total_size"] += size
                all_positions[symbol]["total_value"] += size * mark_price
                all_positions[symbol]["exchanges"].append(exchange)
                
                total_exposure += abs(size * mark_price)
        
        return {
            "positions": all_positions,
            "total_exposure": total_exposure,
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _aggregate_orders_data(self, exchange_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate orders data from multiple exchanges"""
        if not exchange_data:
            return {}
        
        # Combine orders from all exchanges
        all_orders = []
        
        for exchange, orders in exchange_data.items():
            if isinstance(orders, list):
                for order in orders:
                    order["source_exchange"] = exchange
                    all_orders.append(order)
        
        return {
            "orders": all_orders,
            "total_orders": len(all_orders),
            "exchanges": list(exchange_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired"""
        if cache_key not in self.data_cache:
            return None
        
        cached_entry = self.data_cache[cache_key]
        if time.time() - cached_entry["timestamp"] > self.cache_ttl:
            del self.data_cache[cache_key]
            return None
        
        return cached_entry["data"]
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self.data_cache[cache_key] = {
            "data": data,
            "timestamp": time.time()
        }
    
    async def _cleanup_expired_cache(self) -> None:
        """Cleanup expired cache entries"""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = []
                
                for key, entry in self.data_cache.items():
                    if current_time - entry["timestamp"] > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.data_cache[key]
                
                await asyncio.sleep(30)  # Cleanup every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")
                await asyncio.sleep(30)
    
    def _update_stats_by_type(self, data_type: str) -> None:
        """Update statistics by data type"""
        if data_type not in self.data_stats["by_type"]:
            self.data_stats["by_type"][data_type] = 0
        self.data_stats["by_type"][data_type] += 1
    
    def _update_stats_by_exchange(self, exchange: str) -> None:
        """Update statistics by exchange"""
        if exchange not in self.data_stats["by_exchange"]:
            self.data_stats["by_exchange"][exchange] = 0
        self.data_stats["by_exchange"][exchange] += 1
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get data aggregator statistics"""
        return {
            "running": self._running,
            "statistics": self.data_stats,
            "cache_size": len(self.data_cache),
            "cache_ttl": self.cache_ttl,
            "timestamp": datetime.now().isoformat()
        }