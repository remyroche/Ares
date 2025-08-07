# src/types/data_types.py

"""
Data structure type definitions for market data and trading information.
"""

from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict

from .base_types import (
    OrderId,
    Price,
    PositionId,
    Symbol,
    Timestamp,
    TradeId,
    Volume,
)


class OHLCVData(TypedDict):
    """Type-safe OHLCV market data."""
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume


class TickerData(TypedDict):
    """Type-safe ticker data."""
    symbol: Symbol
    price: Price
    change_24h: float
    volume_24h: Volume
    high_24h: Price
    low_24h: Price
    timestamp: Timestamp


class OrderBookLevel(TypedDict):
    """Type-safe order book level."""
    price: Price
    quantity: Volume


class OrderBookData(TypedDict):
    """Type-safe order book data."""
    symbol: Symbol
    timestamp: Timestamp
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]


class TradeData(TypedDict):
    """Type-safe individual trade data."""
    trade_id: TradeId
    symbol: Symbol
    price: Price
    quantity: Volume
    side: Literal["buy", "sell"]
    timestamp: Timestamp


class AccountInfo(TypedDict):
    """Type-safe account information."""
    account_id: str
    total_balance: float
    available_balance: float
    margin_balance: Optional[float]
    unrealized_pnl: Optional[float]
    margin_ratio: Optional[float]
    positions: List[Dict[str, float]]  # Will be typed more specifically
    open_orders: List[Dict[str, str]]  # Will be typed more specifically


class PositionInfo(TypedDict):
    """Type-safe position information."""
    position_id: PositionId
    symbol: Symbol
    side: Literal["long", "short"]
    size: Volume
    entry_price: Price
    mark_price: Price
    unrealized_pnl: float
    leverage: float
    margin: float
    timestamp: Timestamp


class OrderInfo(TypedDict):
    """Type-safe order information."""
    order_id: OrderId
    symbol: Symbol
    side: Literal["buy", "sell"]
    type: Literal["market", "limit", "stop", "stop_limit"]
    quantity: Volume
    price: Optional[Price]
    stop_price: Optional[Price]
    status: Literal["pending", "open", "filled", "cancelled", "rejected"]
    filled_quantity: Volume
    timestamp: Timestamp


# Aggregate types for convenience
MarketDataDict = Dict[Symbol, List[OHLCVData]]
TickerDict = Dict[Symbol, TickerData]
OrderBookDict = Dict[Symbol, OrderBookData]