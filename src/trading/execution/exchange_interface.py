"""
Exchange Interface

Abstract interface for different cryptocurrency exchanges.
Provides unified API for market data access and order execution.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, traced, log_execution_time
from src.utils.tprint import (
    tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_structured, LogLevel
)
from ...exchanges.exchange_dispatcher import ExchangeDispatcher, ExchangeConfig, ExchangeType
from ..utils.error_handling import (
    ExecutionError, TradingErrorSeverity, trading_error_handler,
    critical_operation, require_no_fallback
)
from ..utils.validation import validate_trading_config

logger = system_logger.getChild('ExchangeInterface')

class ExchangeType(Enum):
    """Exchange types."""
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    SIMULATED = "simulated"

class MarketDataType(Enum):
    """Market data types."""
    TICKER = "ticker"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    KLINES = "klines"
    AGGREGATE_TRADES = "agg_trades"

class ConnectionStatus(Enum):
    """Connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

@dataclass
class MarketData:
    """Market data container."""
    symbol: str
    exchange: str
    data_type: MarketDataType
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TickerData:
    """Ticker data."""
    symbol: str
    price: float
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    volume_24h: float
    price_change_24h: float
    price_change_percent_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime

@dataclass
class KlineData:
    """Kline (candlestick) data."""
    symbol: str
    interval: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    close_time: datetime
    quote_asset_volume: float
    number_of_trades: int
    taker_buy_base_asset_volume: float
    taker_buy_quote_asset_volume: float

class ExchangeInterface:
    """
    Exchange interface that uses the exchange dispatcher.

    Provides unified API for:
    - Market data access (ticker, order book, klines, trades)
    - Order execution and management
    - Account information and balances
    - Rate limiting and error handling
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize exchange interface.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exchange_type = config.get('exchange_type', 'simulated')
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.testnet = config.get('testnet', True)
        self.rate_limits = config.get('rate_limits', {})

        # Exchange dispatcher
        self.dispatcher: Optional[ExchangeDispatcher] = None
        
        # Connection state
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.last_connection_attempt = None
        self.connection_errors = []

        # Data streams
        self.ticker_streams: Dict[str, Any] = {}
        self.order_book_streams: Dict[str, Any] = {}
        self.kline_streams: Dict[str, Dict[str, Any]] = {}

        # Rate limiting
        self.request_counts: Dict[str, int] = {}
        self.last_requests: Dict[str, datetime] = {}

        # Performance tracking
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0

        # Simulated exchange data
        self.price_feeds: Dict[str, Dict[str, float]] = {}
        self.simulated_orders: Dict[str, Dict[str, Any]] = {}
        
        # Initialize simulated data
        self._initialize_simulated_data()

        self.logger = logger.getChild(f'{self.exchange_type}')
    
    def _initialize_simulated_data(self) -> None:
        """Initialize simulated exchange data."""
        self.price_feeds['ETHUSDT'] = {
            'price': 3000.0,
            'bid_price': 2999.5,
            'ask_price': 3000.5,
            'volume_24h': 1000000.0,
            'price_change_24h': 50.0,
            'high_24h': 3100.0,
            'low_24h': 2900.0
        }

        self.price_feeds['BTCUSDT'] = {
            'price': 50000.0,
            'bid_price': 49995.0,
            'ask_price': 50005.0,
            'volume_24h': 500000.0,
            'price_change_24h': 1000.0,
            'high_24h': 51000.0,
            'low_24h': 48000.0
        }

    async def connect(self) -> bool:
        """Connect to exchange."""
        try:
            if self.exchange_type == 'simulated':
                self.connection_status = ConnectionStatus.CONNECTED
                tprint_success(f"✅ Connected to {self.exchange_type} (simulated)")
                return True
            
            # Create exchange dispatcher
            exchange_type = ExchangeType.OKX if self.exchange_type == 'okx' else ExchangeType.BINANCE
            config = ExchangeConfig(
                exchange_type=exchange_type,
                api_key=self.api_key,
                api_secret=self.api_secret,
                use_testnet=self.testnet,
                trade_symbol=self.config.get('trade_symbol', 'BTCUSDT')
            )
            
            self.dispatcher = ExchangeDispatcher(config)
            success = await self.dispatcher.initialize()
            
            if success:
                self.connection_status = ConnectionStatus.CONNECTED
                tprint_success(f"✅ Connected to {self.exchange_type}")
                return True
            else:
                self.connection_status = ConnectionStatus.ERROR
                return False
                
        except Exception as e:
            self.connection_status = ConnectionStatus.ERROR
            await self._handle_error(e, "connect")
            return False

    async def disconnect(self) -> None:
        """Disconnect from exchange."""
        if self.dispatcher:
            await self.dispatcher.close()
            self.dispatcher = None
        
        self.connection_status = ConnectionStatus.DISCONNECTED
        tprint_info(f"📴 Disconnected from {self.exchange_type}")

    async def is_connected(self) -> bool:
        """Check if connected to exchange."""
        if self.exchange_type == 'simulated':
            return self.connection_status == ConnectionStatus.CONNECTED
        
        if self.dispatcher:
            return await self.dispatcher.is_connected()
        
        return False

    async def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get ticker data for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_ticker(symbol)
        
        if self.dispatcher:
            ticker_data = await self.dispatcher.get_ticker(symbol)
            if ticker_data:
                return TickerData(
                    symbol=symbol,
                    price=ticker_data.get('price', 0),
                    bid_price=ticker_data.get('bid', 0),
                    ask_price=ticker_data.get('ask', 0),
                    bid_quantity=ticker_data.get('bidQty', 0),
                    ask_quantity=ticker_data.get('askQty', 0),
                    volume_24h=ticker_data.get('volume', 0),
                    price_change_24h=ticker_data.get('change', 0),
                    price_change_percent_24h=ticker_data.get('changePercent', 0),
                    high_24h=ticker_data.get('high', 0),
                    low_24h=ticker_data.get('low', 0),
                    timestamp=datetime.now()
                )
        
        return None

    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Get order book for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_order_book(symbol, limit)
        
        if self.dispatcher:
            return await self.dispatcher.get_order_book(symbol, limit)
        
        return None

    async def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> List[KlineData]:
        """Get kline data for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_klines(symbol, interval, start_time, end_time, limit)
        
        if self.dispatcher:
            ohlcv_data = await self.dispatcher.get_ohlcv(symbol, interval, limit)
            klines = []
            for candle in ohlcv_data:
                klines.append(KlineData(
                    symbol=candle.symbol,
                    interval=interval,
                    timestamp=candle.timestamp,
                    open_price=candle.open,
                    high_price=candle.high,
                    low_price=candle.low,
                    close_price=candle.close,
                    volume=candle.volume,
                    close_time=candle.timestamp,
                    quote_asset_volume=candle.volume * candle.close,
                    number_of_trades=0,
                    taker_buy_base_asset_volume=candle.volume * 0.5,
                    taker_buy_quote_asset_volume=candle.volume * candle.close * 0.5
                ))
            return klines
        
        return []

    async def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """Get recent trades for symbol."""
        if self.exchange_type == 'simulated':
            return await self._get_simulated_recent_trades(symbol, limit)
        
        # For real exchanges, this would be implemented in the dispatcher
        return []

    async def get_account_balance(self, asset: Optional[str] = None) -> Dict[str, float]:
        """Get account balance."""
        if self.exchange_type == 'simulated':
            return self._get_simulated_balance(asset)
        
        if self.dispatcher:
            if asset:
                balance = await self.dispatcher.get_balance(asset)
                return {asset: balance}
            else:
                # Get all balances - this would need to be implemented in dispatcher
                return {}
        
        return {}

    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create order."""
        if self.exchange_type == 'simulated':
            return await self._create_simulated_order(symbol, side, order_type, quantity, price)
        
        if self.dispatcher:
            result = await self.dispatcher.create_order(symbol, side, order_type, quantity, price)
            return result or {}
        
        return {}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order."""
        if self.exchange_type == 'simulated':
            return await self._cancel_simulated_order(symbol, order_id)
        
        if self.dispatcher:
            return await self.dispatcher.cancel_order(symbol, order_id)
        
        return False

    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        if self.exchange_type == 'simulated':
            return self._get_simulated_order_status(symbol, order_id)
        
        if self.dispatcher:
            result = await self.dispatcher.get_order_status(symbol, order_id)
            return result or {}
        
        return {}

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        if self.exchange_type == 'simulated':
            return self._get_simulated_open_orders(symbol)
        
        if self.dispatcher:
            return await self.dispatcher.get_open_orders(symbol)
        
        return []

    # Simulated exchange methods
    async def _get_simulated_ticker(self, symbol: str) -> Optional[TickerData]:
        """Get simulated ticker data."""
        # Use the existing simulated exchange logic
        if symbol not in self.price_feeds:
            return None

        data = self.price_feeds[symbol]
        price_variation = np.random.normal(0, data['price'] * 0.001)
        current_price = data['price'] + price_variation

        return TickerData(
            symbol=symbol,
            price=current_price,
            bid_price=current_price - 0.5,
            ask_price=current_price + 0.5,
            bid_quantity=np.random.uniform(1, 10),
            ask_quantity=np.random.uniform(1, 10),
            volume_24h=data['volume_24h'],
            price_change_24h=data['price_change_24h'],
            price_change_percent_24h=(data['price_change_24h'] / data['price']) * 100,
            high_24h=data['high_24h'],
            low_24h=data['low_24h'],
            timestamp=datetime.now()
        )

    async def _get_simulated_order_book(self, symbol: str, limit: int) -> Optional[Dict[str, Any]]:
        """Get simulated order book."""
        if symbol not in self.price_feeds:
            return None

        data = self.price_feeds[symbol]
        base_price = data['price']

        bids = []
        asks = []

        for i in range(limit):
            bid_price = base_price - 0.5 - (i * 0.1)
            ask_price = base_price + 0.5 + (i * 0.1)

            bid_quantity = np.random.uniform(0.1, 5.0)
            ask_quantity = np.random.uniform(0.1, 5.0)

            bids.append([bid_price, bid_quantity])
            asks.append([ask_price, ask_quantity])

        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }

    async def _get_simulated_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        limit: int
    ) -> List[KlineData]:
        """Get simulated kline data."""
        if symbol not in self.price_feeds:
            return []

        data = self.price_feeds[symbol]
        base_price = data['price']
        klines = []
        current_time = datetime.now()

        for i in range(min(limit, 500)):
            timestamp = current_time - timedelta(minutes=i)
            open_price = base_price + np.random.normal(0, base_price * 0.02)
            high_price = open_price + abs(np.random.normal(0, base_price * 0.01))
            low_price = open_price - abs(np.random.normal(0, base_price * 0.01))
            close_price = low_price + np.random.uniform(0, high_price - low_price)
            volume = np.random.uniform(100, 1000)

            klines.append(KlineData(
                symbol=symbol,
                interval=interval,
                timestamp=timestamp,
                open_price=open_price,
                high_price=high_price,
                low_price=low_price,
                close_price=close_price,
                volume=volume,
                close_time=timestamp + timedelta(minutes=1),
                quote_asset_volume=close_price * volume,
                number_of_trades=int(np.random.uniform(10, 100)),
                taker_buy_base_asset_volume=volume * np.random.uniform(0.3, 0.7),
                taker_buy_quote_asset_volume=close_price * volume * np.random.uniform(0.3, 0.7)
            ))

        return klines

    async def _get_simulated_recent_trades(self, symbol: str, limit: int) -> List[Dict[str, Any]]:
        """Get simulated recent trades."""
        if symbol not in self.price_feeds:
            return []

        data = self.price_feeds[symbol]
        base_price = data['price']
        trades = []

        for i in range(min(limit, 500)):
            timestamp = datetime.now() - timedelta(seconds=i)
            price = base_price + np.random.normal(0, base_price * 0.001)
            quantity = np.random.uniform(0.01, 1.0)

            trades.append({
                'id': f'sim_trade_{i}',
                'price': price,
                'qty': quantity,
                'quoteQty': price * quantity,
                'time': timestamp.isoformat(),
                'isBuyerMaker': np.random.choice([True, False]),
                'isBestMatch': True
            })

        return trades

    def _get_simulated_balance(self, asset: Optional[str]) -> Dict[str, float]:
        """Get simulated account balance."""
        if asset:
            return {asset: 1000.0 if asset == 'USDT' else 10.0}

        return {
            'USDT': 10000.0,
            'ETH': 10.0,
            'BTC': 1.0
        }

    async def _create_simulated_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Create simulated order."""
        order_id = f'sim_order_{len(self.simulated_orders)}'

        order_data = {
            'symbol': symbol,
            'orderId': order_id,
            'orderListId': -1,
            'clientOrderId': f'client_{order_id}',
            'price': price,
            'origQty': quantity,
            'executedQty': quantity,
            'cummulativeQuoteQty': price * quantity if price else 3000.0 * quantity,
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': order_type,
            'side': side,
            'workingTime': datetime.now().isoformat(),
            'selfTradePreventionMode': 'NONE'
        }

        self.simulated_orders[order_id] = order_data
        return order_data

    async def _cancel_simulated_order(self, symbol: str, order_id: str) -> bool:
        """Cancel simulated order."""
        if order_id in self.simulated_orders:
            self.simulated_orders[order_id]['status'] = 'CANCELLED'
            return True
        return False

    def _get_simulated_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Get simulated order status."""
        return self.simulated_orders.get(order_id, {})

    def _get_simulated_open_orders(self, symbol: Optional[str]) -> List[Dict[str, Any]]:
        """Get simulated open orders."""
        open_orders = []

        for order_id, order_data in self.simulated_orders.items():
            if order_data['status'] in ['NEW', 'PARTIALLY_FILLED']:
                open_orders.append(order_data)

        return open_orders

    def _check_rate_limit(self, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.now()

        # Simple rate limiting implementation
        if endpoint in self.request_counts:
            if self.request_counts[endpoint] >= self.rate_limits.get(endpoint, 100):
                return False

        return True

    def _update_rate_limit(self, endpoint: str) -> None:
        """Update rate limit counters."""
        now = datetime.now()

        if endpoint not in self.request_counts:
            self.request_counts[endpoint] = 0
            self.last_requests[endpoint] = now

        self.request_counts[endpoint] += 1
        self.total_requests += 1

    async def _handle_error(self, error: Exception, operation: str) -> None:
        """Handle exchange errors."""
        self.connection_errors.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'error': str(error)
        })

        self.failed_requests += 1

        if len(self.connection_errors) > 10:
            self.connection_status = ConnectionStatus.ERROR

        tprint_error(f"❌ Exchange error in {operation}: {str(error)}")


# Factory function for creating exchange interfaces
def create_exchange_interface(config: Dict[str, Any]) -> ExchangeInterface:
    """
    Create exchange interface.

    Args:
        config: Configuration dictionary

    Returns:
        Exchange interface instance
    """
    return ExchangeInterface(config)

def get_exchange_interface(exchange_type: str) -> Optional[ExchangeInterface]:
    """Get exchange interface by type."""
    # Placeholder for getting cached interface
    return None