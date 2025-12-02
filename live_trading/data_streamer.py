"""
Data Streaming System

Handles real-time market data streaming for live trading.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
import logging

from .config import TradingConfig
from src.interfaces.base_interfaces import MarketData


@dataclass
class StreamData:
    """Streaming data structure"""
    symbol: str
    data_type: str  # 'ticker', 'trade', 'orderbook', 'kline'
    timestamp: datetime
    data: Dict[str, Any]
    raw_data: Dict[str, Any] = field(default_factory=dict)


class DataStreamer:
    """Manages real-time data streaming from exchanges"""
    
    def __init__(self, config: TradingConfig, exchange_client: Any):
        self.config = config
        self.exchange_client = exchange_client
        self.logger = logging.getLogger(__name__)
        
        # Data handlers
        self.data_handlers: Dict[str, List[Callable[[StreamData], Awaitable[None]]]] = {
            "ticker": [],
            "trade": [],
            "orderbook": [],
            "kline": []
        }
        
        # Streaming state
        self._streaming_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._last_data: Dict[str, Dict[str, Any]] = {}
        
        # Connection management
        self._reconnect_attempts: Dict[str, int] = {}
        self._last_reconnect: Dict[str, datetime] = {}
        
    async def start(self) -> None:
        """Start data streaming"""
        if self._running:
            return
            
        self._running = True
        
        # Start streaming for each symbol
        for symbol in self.config.symbols:
            await self._start_symbol_streaming(symbol)
        
        self.logger.info(f"Data streamer started for symbols: {self.config.symbols}")
    
    async def stop(self) -> None:
        """Stop data streaming"""
        self._running = False
        
        # Cancel all streaming tasks
        for task in self._streaming_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self._streaming_tasks:
            await asyncio.gather(*self._streaming_tasks.values(), return_exceptions=True)
        
        self._streaming_tasks.clear()
        self.logger.info("Data streamer stopped")
    
    def register_handler(self, data_type: str, handler: Callable[[StreamData], Awaitable[None]]) -> None:
        """Register data handler"""
        if data_type in self.data_handlers:
            self.data_handlers[data_type].append(handler)
            self.logger.info(f"Registered {data_type} handler")
    
    async def get_latest_data(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Get latest data for symbol and type"""
        key = f"{symbol}_{data_type}"
        return self._last_data.get(key)
    
    async def _start_symbol_streaming(self, symbol: str) -> None:
        """Start streaming for a specific symbol"""
        # Start ticker streaming
        if "ticker" in [h for handlers in self.data_handlers.values() for h in handlers]:
            task = asyncio.create_task(self._stream_ticker(symbol))
            self._streaming_tasks[f"{symbol}_ticker"] = task
        
        # Start trade streaming
        if "trade" in [h for handlers in self.data_handlers.values() for h in handlers]:
            task = asyncio.create_task(self._stream_trades(symbol))
            self._streaming_tasks[f"{symbol}_trades"] = task
        
        # Start orderbook streaming
        if "orderbook" in [h for handlers in self.data_handlers.values() for h in handlers]:
            task = asyncio.create_task(self._stream_orderbook(symbol))
            self._streaming_tasks[f"{symbol}_orderbook"] = task
        
        # Start kline streaming
        if "kline" in [h for handlers in self.data_handlers.values() for h in handlers]:
            task = asyncio.create_task(self._stream_klines(symbol))
            self._streaming_tasks[f"{symbol}_klines"] = task
    
    async def _stream_ticker(self, symbol: str) -> None:
        """Stream ticker data for symbol"""
        while self._running:
            try:
                # Get ticker data from exchange
                ticker_data = await self.exchange_client.get_ticker(symbol)
                
                if ticker_data:
                    stream_data = StreamData(
                        symbol=symbol,
                        data_type="ticker",
                        timestamp=datetime.now(),
                        data=self._normalize_ticker_data(ticker_data),
                        raw_data=ticker_data
                    )
                    
                    # Store latest data
                    self._last_data[f"{symbol}_ticker"] = stream_data.data
                    
                    # Notify handlers
                    await self._notify_handlers("ticker", stream_data)
                
                # Wait before next update
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error streaming ticker for {symbol}: {e}")
                self.logger.warning(f"⚠️ Ticker streaming failed for {symbol} - continuing with other streams")
                await self._handle_streaming_error(symbol, "ticker")
    
    async def _stream_trades(self, symbol: str) -> None:
        """Stream trade data for symbol"""
        while self._running:
            try:
                # Get recent trades from exchange
                trades_data = await self.exchange_client.get_recent_trades(symbol, limit=100)
                
                if trades_data:
                    for trade in trades_data:
                        stream_data = StreamData(
                            symbol=symbol,
                            data_type="trade",
                            timestamp=self._parse_timestamp(trade.get("timestamp", trade.get("time", 0))),
                            data=self._normalize_trade_data(trade),
                            raw_data=trade
                        )
                        
                        # Notify handlers
                        await self._notify_handlers("trade", stream_data)
                
                # Wait before next update
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error streaming trades for {symbol}: {e}")
                self.logger.warning(f"⚠️ Trade streaming failed for {symbol} - continuing with other streams")
                await self._handle_streaming_error(symbol, "trades")
    
    async def _stream_orderbook(self, symbol: str) -> None:
        """Stream orderbook data for symbol"""
        while self._running:
            try:
                # Get orderbook from exchange
                orderbook_data = await self.exchange_client.get_order_book(symbol, limit=20)
                
                if orderbook_data:
                    stream_data = StreamData(
                        symbol=symbol,
                        data_type="orderbook",
                        timestamp=datetime.now(),
                        data=self._normalize_orderbook_data(orderbook_data),
                        raw_data=orderbook_data
                    )
                    
                    # Store latest data
                    self._last_data[f"{symbol}_orderbook"] = stream_data.data
                    
                    # Notify handlers
                    await self._notify_handlers("orderbook", stream_data)
                
                # Wait before next update
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error streaming orderbook for {symbol}: {e}")
                self.logger.warning(f"⚠️ Orderbook streaming failed for {symbol} - continuing with other streams")
                await self._handle_streaming_error(symbol, "orderbook")
    
    async def _stream_klines(self, symbol: str) -> None:
        """Stream kline data for symbol"""
        last_kline_time = None
        
        while self._running:
            try:
                # Get recent klines
                klines = await self.exchange_client.get_klines(symbol, "1m", limit=5)
                
                if klines:
                    for kline in klines:
                        # Only process new klines
                        if last_kline_time is None or kline.timestamp > last_kline_time:
                            stream_data = StreamData(
                                symbol=symbol,
                                data_type="kline",
                                timestamp=kline.timestamp,
                                data={
                                    "open": kline.open,
                                    "high": kline.high,
                                    "low": kline.low,
                                    "close": kline.close,
                                    "volume": kline.volume,
                                    "interval": kline.interval
                                },
                                raw_data={
                                    "open": kline.open,
                                    "high": kline.high,
                                    "low": kline.low,
                                    "close": kline.close,
                                    "volume": kline.volume,
                                    "timestamp": kline.timestamp.isoformat(),
                                    "interval": kline.interval
                                }
                            )
                            
                            # Notify handlers
                            await self._notify_handlers("kline", stream_data)
                            
                            # Update last kline time
                            if last_kline_time is None or kline.timestamp > last_kline_time:
                                last_kline_time = kline.timestamp
                
                # Wait before next update
                await asyncio.sleep(self.config.data_update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Error streaming klines for {symbol}: {e}")
                self.logger.warning(f"⚠️ Kline streaming failed for {symbol} - continuing with other streams")
                await self._handle_streaming_error(symbol, "klines")
    
    async def _handle_streaming_error(self, symbol: str, stream_type: str) -> None:
        """Handle streaming errors with reconnection logic"""
        key = f"{symbol}_{stream_type}"
        
        # Increment reconnect attempts
        if key not in self._reconnect_attempts:
            self._reconnect_attempts[key] = 0
        self._reconnect_attempts[key] += 1
        
        # Check if we should attempt reconnection
        if self._reconnect_attempts[key] <= self.config.reconnect_attempts:
            self.logger.warning(f"Attempting to reconnect {key} (attempt {self._reconnect_attempts[key]})")
            await asyncio.sleep(self.config.reconnect_delay)
            
            # Restart streaming
            if key in self._streaming_tasks:
                self._streaming_tasks[key].cancel()
                del self._streaming_tasks[key]
            
            # Restart the specific stream
            if stream_type == "ticker":
                task = asyncio.create_task(self._stream_ticker(symbol))
                self._streaming_tasks[key] = task
            elif stream_type == "trades":
                task = asyncio.create_task(self._stream_trades(symbol))
                self._streaming_tasks[key] = task
            elif stream_type == "orderbook":
                task = asyncio.create_task(self._stream_orderbook(symbol))
                self._streaming_tasks[key] = task
            elif stream_type == "klines":
                task = asyncio.create_task(self._stream_klines(symbol))
                self._streaming_tasks[key] = task
        else:
            self.logger.error(f"Max reconnection attempts reached for {key}")
            # Cancel the task permanently
            if key in self._streaming_tasks:
                self._streaming_tasks[key].cancel()
                del self._streaming_tasks[key]
    
    def _normalize_ticker_data(self, ticker_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize ticker data to standard format"""
        return {
            "symbol": ticker_data.get("symbol", ""),
            "last_price": float(ticker_data.get("last", ticker_data.get("close", 0))),
            "bid_price": float(ticker_data.get("bid", 0)),
            "ask_price": float(ticker_data.get("ask", 0)),
            "volume": float(ticker_data.get("volume", 0)),
            "quote_volume": float(ticker_data.get("quoteVolume", ticker_data.get("quote_volume", 0))),
            "price_change": float(ticker_data.get("change", 0)),
            "price_change_percent": float(ticker_data.get("percentage", 0)),
            "high_24h": float(ticker_data.get("high", 0)),
            "low_24h": float(ticker_data.get("low", 0)),
            "open_24h": float(ticker_data.get("open", 0))
        }
    
    def _normalize_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trade data to standard format"""
        return {
            "price": float(trade_data.get("price", 0)),
            "quantity": float(trade_data.get("quantity", trade_data.get("qty", 0))),
            "side": trade_data.get("side", "unknown"),
            "timestamp": self._parse_timestamp(trade_data.get("timestamp", trade_data.get("time", 0))).isoformat(),
            "trade_id": trade_data.get("id", trade_data.get("trade_id", ""))
        }
    
    def _normalize_orderbook_data(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize orderbook data to standard format"""
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])
        
        return {
            "bids": [{"price": float(bid[0]), "quantity": float(bid[1])} for bid in bids[:10]],
            "asks": [{"price": float(ask[0]), "quantity": float(ask[1])} for ask in asks[:10]],
            "timestamp": datetime.now().isoformat()
        }
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime"""
        if isinstance(timestamp, (int, float)):
            # Assume milliseconds if timestamp is large
            if timestamp > 1e10:
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                return datetime.now()
        else:
            return datetime.now()
    
    async def _notify_handlers(self, data_type: str, stream_data: StreamData) -> None:
        """Notify registered handlers of new data"""
        if data_type in self.data_handlers:
            for handler in self.data_handlers[data_type]:
                try:
                    await handler(stream_data)
                except Exception as e:
                    self.logger.error(f"❌ Error in data handler: {e}")
                    self.logger.warning("⚠️ Data handler failed - continuing with other handlers")
    
    async def get_streaming_status(self) -> Dict[str, Any]:
        """Get current streaming status"""
        return {
            "running": self._running,
            "active_streams": len(self._streaming_tasks),
            "symbols": self.config.symbols,
            "stream_types": list(self.data_handlers.keys()),
            "reconnect_attempts": dict(self._reconnect_attempts)
        }