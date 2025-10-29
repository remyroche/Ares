"""
Exchange-specific validation utilities.
"""

from typing import Dict, Any, Optional
from .error_handling import ValidationError, TradingErrorSeverity, InvalidSymbolError
from .constants import VALID_EXCHANGES, DEFAULT_PRICE_PRECISION, DEFAULT_QUANTITY_PRECISION

# Exchange-specific configurations
EXCHANGE_CONFIGS = {
    'binance': {
        'symbol_format': 'UPPERCASE',  # BTCUSDT
        'price_precision': 8,
        'quantity_precision': 8,
        'min_order_size': 10.0,  # USD
        'supported_order_types': ['market', 'limit', 'stop', 'stop_limit'],
        'rate_limit_requests': 1200,
        'rate_limit_window': 60,
        'leverage_max': 125
    },
    'binance_testnet': {
        'symbol_format': 'UPPERCASE',
        'price_precision': 8,
        'quantity_precision': 8,
        'min_order_size': 10.0,
        'supported_order_types': ['market', 'limit', 'stop', 'stop_limit'],
        'rate_limit_requests': 1200,
        'rate_limit_window': 60,
        'leverage_max': 125
    },
    'simulated': {
        'symbol_format': 'UPPERCASE',
        'price_precision': 8,
        'quantity_precision': 8,
        'min_order_size': 1.0,
        'supported_order_types': ['market', 'limit', 'stop'],
        'rate_limit_requests': 10000,
        'rate_limit_window': 60,
        'leverage_max': 100
    }
}

def validate_symbol_format(
    symbol: str,
    exchange: str = 'binance'
) -> bool:
    """
    Validate symbol format for exchange.

    Args:
        symbol: Trading symbol
        exchange: Exchange name

    Returns:
        bool: True if valid

    Raises:
        InvalidSymbolError: If symbol format is invalid
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    config = EXCHANGE_CONFIGS[exchange]
    format_type = config['symbol_format']

    if format_type == 'UPPERCASE':
        if not symbol.isupper():
            raise InvalidSymbolError(
                f"Symbol {symbol} must be uppercase for {exchange}",
                context={'symbol': symbol, 'exchange': exchange}
            )
        if not symbol.replace('/', '').isalnum():
            raise InvalidSymbolError(
                f"Symbol {symbol} contains invalid characters for {exchange}",
                context={'symbol': symbol, 'exchange': exchange}
            )

    # Basic validation
    if len(symbol) < 3:
        raise InvalidSymbolError(
            f"Symbol {symbol} is too short",
            context={'symbol': symbol, 'exchange': exchange}
        )

    return True

def validate_exchange_order_type(
    order_type: str,
    exchange: str = 'binance'
) -> bool:
    """
    Validate order type is supported by exchange.

    Args:
        order_type: Order type
        exchange: Exchange name

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If order type not supported
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    config = EXCHANGE_CONFIGS[exchange]
    supported_types = config['supported_order_types']

    if order_type.lower() not in [t.lower() for t in supported_types]:
        raise ValidationError(
            f"Order type {order_type} not supported by {exchange}. Supported: {supported_types}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'order_type': order_type,
                'exchange': exchange,
                'supported_types': supported_types
            }
        )

    return True

def validate_exchange_precision(
    price: Optional[float] = None,
    quantity: Optional[float] = None,
    exchange: str = 'binance'
) -> bool:
    """
    Validate price and quantity precision for exchange.

    Args:
        price: Order price
        quantity: Order quantity
        exchange: Exchange name

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If precision invalid
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    config = EXCHANGE_CONFIGS[exchange]
    price_precision = config.get('price_precision', DEFAULT_PRICE_PRECISION)
    quantity_precision = config.get('quantity_precision', DEFAULT_QUANTITY_PRECISION)

    errors = []

    if price is not None:
        price_str = f"{price:.{price_precision}f}"
        if float(price_str) != price:
            errors.append(
                f"Price {price} exceeds precision {price_precision} for {exchange}"
            )

    if quantity is not None:
        quantity_str = f"{quantity:.{quantity_precision}f}"
        if float(quantity_str) != quantity:
            errors.append(
                f"Quantity {quantity} exceeds precision {quantity_precision} for {exchange}"
            )

    if errors:
        raise ValidationError(
            f"Exchange precision validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'exchange': exchange,
                'price_precision': price_precision,
                'quantity_precision': quantity_precision,
                'errors': errors
            }
        )

    return True

def validate_exchange_min_order_size(
    order_value: float,
    exchange: str = 'binance'
) -> bool:
    """
    Validate order meets minimum size requirement for exchange.

    Args:
        order_value: Order value in base currency
        exchange: Exchange name

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If order size too small
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    config = EXCHANGE_CONFIGS[exchange]
    min_order_size = config.get('min_order_size', 10.0)

    if order_value < min_order_size:
        raise ValidationError(
            f"Order value {order_value} is below minimum {min_order_size} for {exchange}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'order_value': order_value,
                'min_order_size': min_order_size,
                'exchange': exchange
            }
        )

    return True

def validate_exchange_leverage(
    leverage: float,
    exchange: str = 'binance'
) -> bool:
    """
    Validate leverage is within exchange limits.

    Args:
        leverage: Leverage value
        exchange: Exchange name

    Returns:
        bool: True if valid

    Raises:
        ValidationError: If leverage exceeds limits
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    config = EXCHANGE_CONFIGS[exchange]
    max_leverage = config.get('leverage_max', 100.0)

    if leverage > max_leverage:
        raise ValidationError(
            f"Leverage {leverage} exceeds maximum {max_leverage} for {exchange}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'leverage': leverage,
                'max_leverage': max_leverage,
                'exchange': exchange
            }
        )

    return True

def get_exchange_config(exchange: str) -> Dict[str, Any]:
    """
    Get exchange configuration.

    Args:
        exchange: Exchange name

    Returns:
        Exchange configuration dictionary
    """
    if exchange not in EXCHANGE_CONFIGS:
        raise ValidationError(
            f"Unknown exchange: {exchange}",
            severity=TradingErrorSeverity.HIGH
        )

    return EXCHANGE_CONFIGS[exchange].copy()
