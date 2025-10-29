"""
Status and Type Enum Mappings

Maps exchange-specific status and type values to unified enums.
"""

from typing import Dict
from exchanges.exchange_types import ExchangeType
from exchanges.base_exchange.exchange_interface import OrderSide, OrderType, OrderStatus


# Order status mappings
ORDER_STATUS_MAPPINGS = {
    ExchangeType.BINANCE: {
        'NEW': OrderStatus.SUBMITTED,
        'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
        'FILLED': OrderStatus.FILLED,
        'CANCELED': OrderStatus.CANCELLED,
        'PENDING_CANCEL': OrderStatus.PENDING,
        'REJECTED': OrderStatus.REJECTED,
        'EXPIRED': OrderStatus.EXPIRED,
    },
    ExchangeType.OKX: {
        'live': OrderStatus.PENDING,
        'partially_filled': OrderStatus.PARTIALLY_FILLED,
        'filled': OrderStatus.FILLED,
        'canceled': OrderStatus.CANCELLED,
        'cancelled': OrderStatus.CANCELLED,
        'rejected': OrderStatus.REJECTED,
    },
    ExchangeType.BINGX: {
        'NEW': OrderStatus.SUBMITTED,
        'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
        'FILLED': OrderStatus.FILLED,
        'CANCELED': OrderStatus.CANCELLED,
        'REJECTED': OrderStatus.REJECTED,
    },
    ExchangeType.MEXC: {
        'NEW': OrderStatus.SUBMITTED,
        'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
        'FILLED': OrderStatus.FILLED,
        'CANCELED': OrderStatus.CANCELLED,
        'REJECTED': OrderStatus.REJECTED,
    },
    ExchangeType.GATEIO: {
        'open': OrderStatus.PENDING,
        'closed': OrderStatus.FILLED,
        'cancelled': OrderStatus.CANCELLED,
        'filled': OrderStatus.FILLED,
    },
    ExchangeType.PHEMEX: {
        'New': OrderStatus.SUBMITTED,
        'PartiallyFilled': OrderStatus.PARTIALLY_FILLED,
        'Filled': OrderStatus.FILLED,
        'Canceled': OrderStatus.CANCELLED,
        'Rejected': OrderStatus.REJECTED,
    },
}

# Order type mappings
ORDER_TYPE_MAPPINGS = {
    ExchangeType.BINANCE: {
        'MARKET': OrderType.MARKET,
        'LIMIT': OrderType.LIMIT,
        'STOP_MARKET': OrderType.STOP,
        'STOP_LOSS_MARKET': OrderType.STOP,
        'TAKE_PROFIT_MARKET': OrderType.STOP,
        'STOP_LOSS_LIMIT': OrderType.STOP_LIMIT,
        'TAKE_PROFIT_LIMIT': OrderType.STOP_LIMIT,
    },
    ExchangeType.OKX: {
        'market': OrderType.MARKET,
        'limit': OrderType.LIMIT,
        'conditional': OrderType.STOP,
        'post_only': OrderType.LIMIT,
        'fok': OrderType.MARKET,
        'ioc': OrderType.MARKET,
    },
    ExchangeType.BINGX: {
        'MARKET': OrderType.MARKET,
        'LIMIT': OrderType.LIMIT,
        'STOP_MARKET': OrderType.STOP,
        'STOP_LIMIT': OrderType.STOP_LIMIT,
    },
    ExchangeType.MEXC: {
        'MARKET': OrderType.MARKET,
        'LIMIT': OrderType.LIMIT,
        'STOP_LIMIT': OrderType.STOP_LIMIT,
    },
    ExchangeType.GATEIO: {
        'limit': OrderType.LIMIT,
        'market': OrderType.MARKET,
    },
    ExchangeType.PHEMEX: {
        'Market': OrderType.MARKET,
        'Limit': OrderType.LIMIT,
        'StopLimit': OrderType.STOP_LIMIT,
    },
}

# Order side mappings (most exchanges use BUY/SELL directly)
ORDER_SIDE_MAPPINGS = {
    ExchangeType.BINANCE: {
        'BUY': OrderSide.BUY,
        'SELL': OrderSide.SELL,
    },
    ExchangeType.OKX: {
        'buy': OrderSide.BUY,
        'sell': OrderSide.SELL,
    },
    ExchangeType.BINGX: {
        'BUY': OrderSide.BUY,
        'SELL': OrderSide.SELL,
    },
    ExchangeType.MEXC: {
        'BUY': OrderSide.BUY,
        'SELL': OrderSide.SELL,
    },
    ExchangeType.GATEIO: {
        'buy': OrderSide.BUY,
        'sell': OrderSide.SELL,
    },
    ExchangeType.PHEMEX: {
        'Buy': OrderSide.BUY,
        'Sell': OrderSide.SELL,
    },
}

# Position side mappings (to "long"/"short"/"neutral")
POSITION_SIDE_MAPPINGS = {
    ExchangeType.BINANCE: {
        'LONG': 'long',
        'SHORT': 'short',
        'BOTH': 'neutral',
        'long': 'long',
        'short': 'short',
    },
    ExchangeType.OKX: {
        'long': 'long',
        'short': 'short',
        'net': 'neutral',
    },
    ExchangeType.BINGX: {
        'LONG': 'long',
        'SHORT': 'short',
    },
    ExchangeType.MEXC: {
        'LONG': 'long',
        'SHORT': 'short',
    },
    ExchangeType.GATEIO: {
        'long': 'long',
        'short': 'short',
    },
    ExchangeType.PHEMEX: {
        'Long': 'long',
        'Short': 'short',
    },
}


def normalize_order_status(raw_status: str, exchange: ExchangeType) -> OrderStatus:
    """Normalize exchange-specific order status to unified OrderStatus enum."""
    mapping = ORDER_STATUS_MAPPINGS.get(exchange, {})
    raw_upper = raw_status.upper()
    raw_lower = raw_status.lower()
    
    # Try different variations
    if raw_upper in mapping:
        return mapping[raw_upper]
    elif raw_lower in mapping:
        return mapping[raw_lower]
    elif raw_status in mapping:
        return mapping[raw_status]
    else:
        # Default fallback
        if 'FILLED' in raw_upper or 'filled' in raw_lower:
            return OrderStatus.FILLED
        elif 'PARTIALLY' in raw_upper or 'partial' in raw_lower:
            return OrderStatus.PARTIALLY_FILLED
        elif 'CANCEL' in raw_upper or 'cancel' in raw_lower:
            return OrderStatus.CANCELLED
        elif 'REJECT' in raw_upper or 'reject' in raw_lower:
            return OrderStatus.REJECTED
        else:
            return OrderStatus.PENDING


def normalize_order_type(raw_type: str, exchange: ExchangeType) -> OrderType:
    """Normalize exchange-specific order type to unified OrderType enum."""
    mapping = ORDER_TYPE_MAPPINGS.get(exchange, {})
    raw_upper = raw_type.upper()
    raw_lower = raw_type.lower()
    
    if raw_upper in mapping:
        return mapping[raw_upper]
    elif raw_lower in mapping:
        return mapping[raw_lower]
    elif raw_type in mapping:
        return mapping[raw_type]
    else:
        # Default fallback
        if 'MARKET' in raw_upper or 'market' in raw_lower:
            return OrderType.MARKET
        elif 'LIMIT' in raw_upper or 'limit' in raw_lower:
            return OrderType.LIMIT
        elif 'STOP' in raw_upper or 'stop' in raw_lower:
            return OrderType.STOP
        else:
            return OrderType.MARKET  # Safe default


def normalize_order_side(raw_side: str, exchange: ExchangeType) -> OrderSide:
    """Normalize exchange-specific order side to unified OrderSide enum."""
    mapping = ORDER_SIDE_MAPPINGS.get(exchange, {})
    raw_upper = raw_side.upper()
    raw_lower = raw_side.lower()
    
    if raw_upper in mapping:
        return mapping[raw_upper]
    elif raw_lower in mapping:
        return mapping[raw_lower]
    elif raw_side in mapping:
        return mapping[raw_side]
    else:
        # Default fallback
        if 'BUY' in raw_upper or 'buy' in raw_lower:
            return OrderSide.BUY
        elif 'SELL' in raw_upper or 'sell' in raw_lower:
            return OrderSide.SELL
        else:
            return OrderSide.BUY  # Safe default


def normalize_position_side(raw_side: str, exchange: ExchangeType) -> str:
    """Normalize exchange-specific position side to unified format."""
    mapping = POSITION_SIDE_MAPPINGS.get(exchange, {})
    raw_upper = raw_side.upper()
    raw_lower = raw_side.lower()
    
    if raw_upper in mapping:
        return mapping[raw_upper]
    elif raw_lower in mapping:
        return mapping[raw_lower]
    elif raw_side in mapping:
        return mapping[raw_side]
    else:
        # Default fallback
        if 'LONG' in raw_upper or 'long' in raw_lower:
            return 'long'
        elif 'SHORT' in raw_upper or 'short' in raw_lower:
            return 'short'
        else:
            return 'neutral'  # Safe default