"""
Trading Validation

Comprehensive validation utilities for trading operations
ensuring data integrity and system safety.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.utils.tprint import tprint_error, tprint_warning, tprint_info, tprint_success
from .error_handling import ValidationError, TradingErrorSeverity
from ..execution.order_manager import OrderSide, OrderType

def validate_trading_config(config: Dict[str, Any], strict: bool = True) -> bool:
    """
    Validate trading configuration parameters.

    Args:
        config: Trading configuration dictionary
        strict: Whether to use strict validation (no warnings allowed)

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Required fields
    required_fields = ['symbol', 'exchange', 'trading_mode']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    # Symbol validation
    if 'symbol' in config:
        symbol = config['symbol']
        if not isinstance(symbol, str) or len(symbol) < 3:
            errors.append(f"Invalid symbol: {symbol}")
        if not symbol.isalnum():
            warnings.append(f"Symbol contains non-alphanumeric characters: {symbol}")

    # Exchange validation
    if 'exchange' in config:
        valid_exchanges = ['binance', 'binance_testnet', 'simulated']
        if config['exchange'] not in valid_exchanges:
            errors.append(f"Invalid exchange: {config['exchange']}. Must be one of {valid_exchanges}")

    # Trading mode validation
    if 'trading_mode' in config:
        valid_modes = ['paper', 'live', 'backtest', 'simulation']
        if config['trading_mode'] not in valid_modes:
            errors.append(f"Invalid trading mode: {config['trading_mode']}. Must be one of {valid_modes}")

    # Risk parameters validation
    risk_params = {
        'max_portfolio_risk': (0.0, 1.0),
        'max_drawdown': (0.0, 1.0),
        'max_leverage': (1.0, 100.0),
        'base_position_size': (0.0, 1.0),
        'max_position_size': (0.0, 1.0),
        'min_position_size': (0.0, 1.0)
    }

    for param, (min_val, max_val) in risk_params.items():
        if param in config:
            value = config[param]
            if not isinstance(value, (int, float)):
                errors.append(f"{param} must be a number, got {type(value)}")
            elif not (min_val <= value <= max_val):
                errors.append(f"{param} must be between {min_val} and {max_val}, got {value}")

    # Position size consistency
    if all(param in config for param in ['min_position_size', 'base_position_size', 'max_position_size']):
        min_size = config['min_position_size']
        base_size = config['base_position_size']
        max_size = config['max_position_size']

        if not (min_size <= base_size <= max_size):
            errors.append(f"Position sizes must satisfy: min <= base <= max, got min={min_size}, base={base_size}, max={max_size}")

    # Confidence thresholds
    confidence_params = ['confidence_threshold', 'regime_confidence_threshold']
    for param in confidence_params:
        if param in config:
            value = config[param]
            if not isinstance(value, (int, float)):
                errors.append(f"{param} must be a number, got {type(value)}")
            elif not (0.0 <= value <= 1.0):
                errors.append(f"{param} must be between 0.0 and 1.0, got {value}")

    # Log results
    if errors:
        for error in errors:
            tprint_error(f"❌ Configuration Error: {error}")
        raise ValidationError(
            f"Trading configuration validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.CRITICAL,
            context={'config': config, 'errors': errors, 'warnings': warnings}
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Configuration Warning: {warning}")
        if strict:
            raise ValidationError(
                f"Trading configuration has warnings in strict mode: {'; '.join(warnings)}",
                severity=TradingErrorSeverity.HIGH,
                context={'config': config, 'warnings': warnings}
            )

    if not warnings and not errors:
        tprint_success("✅ Trading configuration validation passed")

    return True

def validate_market_data(
    data: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10,
    check_completeness: bool = True,
    check_quality: bool = True
) -> bool:
    """
    Validate market data DataFrame.

    Args:
        data: Market data DataFrame
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        check_completeness: Whether to check for missing data
        check_quality: Whether to check data quality

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Basic structure validation
    if not isinstance(data, pd.DataFrame):
        errors.append(f"Market data must be a DataFrame, got {type(data)}")
        raise ValidationError(
            "Invalid market data type",
            severity=TradingErrorSeverity.CRITICAL,
            context={'data_type': type(data)}
        )

    # Size validation
    if len(data) < min_rows:
        errors.append(f"Market data has insufficient rows: {len(data)} < {min_rows}")

    if data.empty:
        errors.append("Market data is empty")

    # Column validation
    default_required_columns = ['open', 'high', 'low', 'close', 'volume']
    columns_to_check = required_columns or default_required_columns

    missing_columns = [col for col in columns_to_check if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Data completeness check
    if check_completeness and not data.empty:
        for col in columns_to_check:
            if col in data.columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    null_percentage = (null_count / len(data)) * 100
                    if null_percentage > 10:  # More than 10% missing
                        errors.append(f"Column {col} has {null_percentage:.1f}% missing values")
                    elif null_percentage > 5:  # 5-10% missing
                        warnings.append(f"Column {col} has {null_percentage:.1f}% missing values")

    # Data quality checks
    if check_quality and not data.empty:
        # Price validation
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                # Check for negative prices
                negative_count = (data[col] <= 0).sum()
                if negative_count > 0:
                    errors.append(f"Column {col} has {negative_count} non-positive values")

                # Check for extreme values
                if len(data) > 1:
                    price_changes = data[col].pct_change().abs()
                    extreme_changes = (price_changes > 0.5).sum()  # 50% change
                    if extreme_changes > 0:
                        warnings.append(f"Column {col} has {extreme_changes} extreme price changes (>50%)")

        # OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= Open, Close, Low
            high_violations = ((data['high'] < data['open']) |
                             (data['high'] < data['close']) |
                             (data['high'] < data['low'])).sum()
            if high_violations > 0:
                errors.append(f"OHLC inconsistency: High is not highest in {high_violations} rows")

            # Low should be <= Open, Close, High
            low_violations = ((data['low'] > data['open']) |
                            (data['low'] > data['close']) |
                            (data['low'] > data['high'])).sum()
            if low_violations > 0:
                errors.append(f"OHLC inconsistency: Low is not lowest in {low_violations} rows")

        # Volume validation
        if 'volume' in data.columns:
            negative_volume = (data['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Volume has {negative_volume} negative values")

            zero_volume = (data['volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # More than 10% zero volume
                warnings.append(f"Volume has {zero_volume} zero values ({zero_volume/len(data)*100:.1f}%)")

    # Timestamp validation (if present)
    if 'timestamp' in data.columns:
        # Check for duplicates
        duplicate_timestamps = data['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            errors.append(f"Found {duplicate_timestamps} duplicate timestamps")

        # Check for proper ordering
        if len(data) > 1:
            unsorted_count = (data['timestamp'].diff().dt.total_seconds() < 0).sum()
            if unsorted_count > 0:
                warnings.append(f"Found {unsorted_count} out-of-order timestamps")

    # Log results
    if errors:
        for error in errors:
            tprint_error(f"❌ Market Data Error: {error}")
        raise ValidationError(
            f"Market data validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'data_shape': data.shape,
                'data_columns': list(data.columns),
                'errors': errors,
                'warnings': warnings
            }
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Market Data Warning: {warning}")

    if not warnings and not errors:
        tprint_success(f"✅ Market data validation passed ({len(data)} rows, {len(data.columns)} columns)")

    return True

def validate_signal_data(
    signal: Dict[str, Any],
    required_fields: Optional[List[str]] = None,
    check_confidence: bool = True
) -> bool:
    """
    Validate trading signal data.

    Args:
        signal: Signal dictionary
        required_fields: List of required field names
        check_confidence: Whether to validate confidence scores

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Basic structure validation
    if not isinstance(signal, dict):
        errors.append(f"Signal must be a dictionary, got {type(signal)}")

    # Required fields validation
    default_required_fields = ['timestamp', 'symbol', 'action', 'confidence']
    fields_to_check = required_fields or default_required_fields

    missing_fields = [field for field in fields_to_check if field not in signal]
    if missing_fields:
        errors.append(f"Missing required signal fields: {missing_fields}")

    # Action validation
    if 'action' in signal:
        valid_actions = ['buy', 'sell', 'hold', 'close']
        if signal['action'] not in valid_actions:
            errors.append(f"Invalid action: {signal['action']}. Must be one of {valid_actions}")

    # Confidence validation
    if check_confidence and 'confidence' in signal:
        confidence = signal['confidence']
        if not isinstance(confidence, (int, float)):
            errors.append(f"Confidence must be a number, got {type(confidence)}")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"Confidence must be between 0.0 and 1.0, got {confidence}")
        elif confidence < 0.5:
            warnings.append(f"Low confidence signal: {confidence}")

    # Timestamp validation
    if 'timestamp' in signal:
        timestamp = signal['timestamp']
        if isinstance(timestamp, str):
            try:
                datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"Invalid timestamp format: {timestamp}")
        elif not isinstance(timestamp, datetime):
            errors.append(f"Timestamp must be datetime or ISO string, got {type(timestamp)}")

    # Symbol validation
    if 'symbol' in signal:
        symbol = signal['symbol']
        if not isinstance(symbol, str) or len(symbol) < 3:
            errors.append(f"Invalid symbol: {symbol}")

    # Price validation (if present)
    price_fields = ['price', 'price_target', 'stop_loss']
    for field in price_fields:
        if field in signal:
            price = signal[field]
            if price is not None:
                if not isinstance(price, (int, float)):
                    errors.append(f"{field} must be a number, got {type(price)}")
                elif price <= 0:
                    errors.append(f"{field} must be positive, got {price}")

    # Log results
    if errors:
        for error in errors:
            tprint_error(f"❌ Signal Validation Error: {error}")
        raise ValidationError(
            f"Signal validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={'signal': signal, 'errors': errors, 'warnings': warnings}
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Signal Validation Warning: {warning}")

    if not warnings and not errors:
        tprint_success("✅ Signal validation passed")

    return True

def validate_position_size(
    size: float,
    account_balance: float,
    max_position_size: float = 0.25,
    min_position_size: float = 0.01
) -> bool:
    """
    Validate position size parameters.

    Args:
        size: Position size (as fraction of account balance)
        account_balance: Current account balance
        max_position_size: Maximum allowed position size
        min_position_size: Minimum allowed position size

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Basic validation
    if not isinstance(size, (int, float)):
        errors.append(f"Position size must be a number, got {type(size)}")

    if not isinstance(account_balance, (int, float)):
        errors.append(f"Account balance must be a number, got {type(account_balance)}")

    if account_balance <= 0:
        errors.append(f"Account balance must be positive, got {account_balance}")

    # Size validation
    if isinstance(size, (int, float)):
        if size < 0:
            errors.append(f"Position size cannot be negative, got {size}")
        elif size < min_position_size:
            errors.append(f"Position size {size} is below minimum {min_position_size}")
        elif size > max_position_size:
            errors.append(f"Position size {size} exceeds maximum {max_position_size}")
        elif size > 0.5:  # Warning for large positions
            warnings.append(f"Large position size: {size} (>50% of account)")

    # Dollar amount validation
    if isinstance(size, (int, float)) and isinstance(account_balance, (int, float)):
        dollar_amount = size * account_balance
        if dollar_amount < 10:  # Minimum trade size
            warnings.append(f"Small trade size: ${dollar_amount:.2f}")

    # Log results
    if errors:
        for error in errors:
            tprint_error(f"❌ Position Size Error: {error}")
        raise ValidationError(
            f"Position size validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'size': size,
                'account_balance': account_balance,
                'max_position_size': max_position_size,
                'min_position_size': min_position_size,
                'errors': errors,
                'warnings': warnings
            }
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Position Size Warning: {warning}")

    if not warnings and not errors:
        tprint_success(f"✅ Position size validation passed (${size * account_balance:.2f})")

    return True

def validate_regime_data(
    regime_data: Dict[str, Any],
    check_probabilities: bool = True
) -> bool:
    """
    Validate regime detection data.

    Args:
        regime_data: Regime detection result
        check_probabilities: Whether to validate probability distributions

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Basic structure validation
    if not isinstance(regime_data, dict):
        errors.append(f"Regime data must be a dictionary, got {type(regime_data)}")

    # Required fields
    required_fields = ['primary_regime', 'confidence', 'regime_probabilities']
    missing_fields = [field for field in required_fields if field not in regime_data]
    if missing_fields:
        errors.append(f"Missing required regime fields: {missing_fields}")

    # Confidence validation
    if 'confidence' in regime_data:
        confidence = regime_data['confidence']
        if not isinstance(confidence, (int, float)):
            errors.append(f"Regime confidence must be a number, got {type(confidence)}")
        elif not (0.0 <= confidence <= 1.0):
            errors.append(f"Regime confidence must be between 0.0 and 1.0, got {confidence}")
        elif confidence < 0.3:
            warnings.append(f"Low regime confidence: {confidence}")

    # Probability validation
    if check_probabilities and 'regime_probabilities' in regime_data:
        probs = regime_data['regime_probabilities']
        if not isinstance(probs, dict):
            errors.append(f"Regime probabilities must be a dictionary, got {type(probs)}")
        else:
            # Check probability values
            for regime, prob in probs.items():
                if not isinstance(prob, (int, float)):
                    errors.append(f"Probability for {regime} must be a number, got {type(prob)}")
                elif not (0.0 <= prob <= 1.0):
                    errors.append(f"Probability for {regime} must be between 0.0 and 1.0, got {prob}")

            # Check if probabilities sum to approximately 1.0
            total_prob = sum(prob for prob in probs.values() if isinstance(prob, (int, float)))
            if abs(total_prob - 1.0) > 0.1:  # Allow 10% tolerance
                warnings.append(f"Regime probabilities don't sum to 1.0: {total_prob}")

    # Log results
    if errors:
        for error in errors:
            tprint_error(f"❌ Regime Data Error: {error}")
        raise ValidationError(
            f"Regime data validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={'regime_data': regime_data, 'errors': errors, 'warnings': warnings}
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Regime Data Warning: {warning}")

    if not warnings and not errors:
        tprint_success("✅ Regime data validation passed")

    return True

def validate_order_params(
    symbol: str,
    side: Any,
    order_type: Any,
    quantity: float,
    price: Optional[float] = None,
    stop_price: Optional[float] = None
) -> None:
    """
    Validate order parameters for trading.

    Args:
        symbol: Trading symbol
        side: Order side ('buy' or 'sell')
        order_type: Order type
        quantity: Order quantity
        price: Order price (optional)
        stop_price: Stop price (optional)

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    # Validate symbol
    if not isinstance(symbol, str) or not symbol:
        errors.append("Symbol must be a non-empty string")

    # Validate side
    if side not in ['buy', 'sell', OrderSide.BUY, OrderSide.SELL]:
        errors.append(f"Invalid order side: {side}")

    # Validate quantity
    if not isinstance(quantity, (int, float)) or quantity <= 0:
        errors.append(f"Quantity must be a positive number, got {quantity}")

    # Validate price for limit orders
    if order_type in ['limit', OrderType.LIMIT] and price is not None:
        if not isinstance(price, (int, float)) or price <= 0:
            errors.append(f"Price must be a positive number, got {price}")

    # Validate stop price for stop orders
    if order_type in ['stop', OrderType.STOP] and stop_price is not None:
        if not isinstance(stop_price, (int, float)) or stop_price <= 0:
            errors.append(f"Stop price must be a positive number, got {stop_price}")

    if errors:
        raise ValidationError(
            f"Order parameter validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price
            }
        )

def validate_order_precision(
    price: float,
    quantity: float,
    price_precision: int = 8,
    quantity_precision: int = 8
) -> bool:
    """
    Validate order price and quantity precision.

    Args:
        price: Order price
        quantity: Order quantity
        price_precision: Required decimal places for price
        quantity_precision: Required decimal places for quantity

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    # Check price precision
    price_str = f"{price:.{price_precision}f}"
    if float(price_str) != price:
        errors.append(f"Price {price} exceeds precision {price_precision}")

    # Check quantity precision
    quantity_str = f"{quantity:.{quantity_precision}f}"
    if float(quantity_str) != quantity:
        errors.append(f"Quantity {quantity} exceeds precision {quantity_precision}")

    if errors:
        raise ValidationError(
            f"Order precision validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'price': price,
                'quantity': quantity,
                'price_precision': price_precision,
                'quantity_precision': quantity_precision
            }
        )

    return True

def validate_leverage(
    leverage: float,
    max_leverage: float = 100.0,
    min_leverage: float = 1.0
) -> bool:
    """
    Validate leverage value.

    Args:
        leverage: Leverage value
        max_leverage: Maximum allowed leverage
        min_leverage: Minimum allowed leverage

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    if not isinstance(leverage, (int, float)):
        errors.append(f"Leverage must be a number, got {type(leverage)}")
    elif leverage < min_leverage:
        errors.append(f"Leverage {leverage} is below minimum {min_leverage}")
    elif leverage > max_leverage:
        errors.append(f"Leverage {leverage} exceeds maximum {max_leverage}")

    if errors:
        raise ValidationError(
            f"Leverage validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'leverage': leverage,
                'max_leverage': max_leverage,
                'min_leverage': min_leverage
            }
        )

    return True

def validate_order_type_compatibility(
    order_type: Any,
    price: Optional[float] = None,
    stop_price: Optional[float] = None
) -> bool:
    """
    Validate order type compatibility with price parameters.

    Args:
        order_type: Order type
        price: Order price
        stop_price: Stop price

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    # Market orders don't need price
    if order_type in ['market', 'MARKET']:
        if price is not None:
            warnings = []
            warnings.append("Market orders don't require price parameter")

    # Limit orders require price
    if order_type in ['limit', 'LIMIT']:
        if price is None:
            errors.append("Limit orders require price parameter")

    # Stop orders require stop_price
    if order_type in ['stop', 'STOP']:
        if stop_price is None:
            errors.append("Stop orders require stop_price parameter")

    if errors:
        raise ValidationError(
            f"Order type compatibility validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'order_type': order_type,
                'price': price,
                'stop_price': stop_price
            }
        )

    return True

def validate_position(
    position: Dict[str, Any],
    account_balance: float,
    max_total_positions: Optional[int] = None
) -> bool:
    """
    Validate position data structure and values.

    Args:
        position: Position dictionary
        account_balance: Account balance
        max_total_positions: Maximum total positions allowed

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []
    warnings = []

    # Required fields
    required_fields = ['symbol', 'quantity', 'entry_price', 'side']
    missing_fields = [field for field in required_fields if field not in position]
    if missing_fields:
        errors.append(f"Missing required position fields: {missing_fields}")

    # Validate quantity
    if 'quantity' in position:
        quantity = position['quantity']
        if not isinstance(quantity, (int, float)) or quantity <= 0:
            errors.append(f"Position quantity must be positive, got {quantity}")

    # Validate entry price
    if 'entry_price' in position:
        entry_price = position['entry_price']
        if not isinstance(entry_price, (int, float)) or entry_price <= 0:
            errors.append(f"Entry price must be positive, got {entry_price}")

    # Validate side
    if 'side' in position:
        side = position['side']
        if side not in ['long', 'short', 'buy', 'sell']:
            errors.append(f"Invalid position side: {side}")

    # Validate position value against account
    if 'quantity' in position and 'entry_price' in position:
        position_value = position['quantity'] * position['entry_price']
        if position_value > account_balance * 1.1:  # 10% tolerance for leverage
            warnings.append(f"Position value {position_value} exceeds account balance {account_balance}")

    if errors:
        raise ValidationError(
            f"Position validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'position': position,
                'account_balance': account_balance,
                'errors': errors,
                'warnings': warnings
            }
        )

    if warnings:
        for warning in warnings:
            tprint_warning(f"⚠️ Position Warning: {warning}")

    return True

def validate_account_balance(
    balance: float,
    available_balance: Optional[float] = None,
    min_balance: float = 0.0
) -> bool:
    """
    Validate account balance values.

    Args:
        balance: Total account balance
        available_balance: Available balance (if provided)
        min_balance: Minimum required balance

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    if not isinstance(balance, (int, float)):
        errors.append(f"Balance must be a number, got {type(balance)}")
    elif balance < min_balance:
        errors.append(f"Balance {balance} is below minimum {min_balance}")

    if available_balance is not None:
        if not isinstance(available_balance, (int, float)):
            errors.append(f"Available balance must be a number, got {type(available_balance)}")
        elif available_balance < 0:
            errors.append(f"Available balance cannot be negative, got {available_balance}")
        elif available_balance > balance:
            errors.append(f"Available balance {available_balance} exceeds total balance {balance}")

    if errors:
        raise ValidationError(
            f"Account balance validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'balance': balance,
                'available_balance': available_balance,
                'min_balance': min_balance
            }
        )

    return True

def validate_market_hours(
    timestamp: datetime,
    exchange: str = 'binance',
    market_open_hour: int = 0,
    market_close_hour: int = 23
) -> bool:
    """
    Validate if market is open at given timestamp.

    Args:
        timestamp: Timestamp to check
        exchange: Exchange name
        market_open_hour: Market open hour (0-23)
        market_close_hour: Market close hour (0-23)

    Returns:
        bool: True if market is open

    Raises:
        MarketClosedError: If market is closed
    """
    from .error_handling import MarketClosedError

    hour = timestamp.hour

    # Crypto markets are typically 24/7, but we check configured hours
    if market_open_hour < market_close_hour:
        # Normal hours (e.g., 9 AM - 5 PM)
        is_open = market_open_hour <= hour < market_close_hour
    else:
        # Overnight hours (e.g., 9 PM - 5 AM)
        is_open = hour >= market_open_hour or hour < market_close_hour

    if not is_open:
        raise MarketClosedError(
            f"Market is closed at {timestamp}. Open hours: {market_open_hour}:00 - {market_close_hour}:00",
            context={
                'timestamp': timestamp.isoformat(),
                'exchange': exchange,
                'market_open_hour': market_open_hour,
                'market_close_hour': market_close_hour
            }
        )

    return True

def validate_batch_orders(
    orders: List[Dict[str, Any]],
    max_batch_size: int = 100
) -> bool:
    """
    Validate a batch of orders.

    Args:
        orders: List of order dictionaries
        max_batch_size: Maximum orders per batch

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    if not isinstance(orders, list):
        errors.append(f"Orders must be a list, got {type(orders)}")
    elif len(orders) > max_batch_size:
        errors.append(f"Batch size {len(orders)} exceeds maximum {max_batch_size}")

    if isinstance(orders, list):
        for i, order in enumerate(orders):
            try:
                validate_order_params(
                    symbol=order.get('symbol', ''),
                    side=order.get('side', ''),
                    order_type=order.get('order_type', ''),
                    quantity=order.get('quantity', 0),
                    price=order.get('price'),
                    stop_price=order.get('stop_price')
                )
            except ValidationError as e:
                errors.append(f"Order {i}: {str(e)}")

    if errors:
        raise ValidationError(
            f"Batch order validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'batch_size': len(orders) if isinstance(orders, list) else 0,
                'max_batch_size': max_batch_size,
                'errors': errors
            }
        )

    return True

def validate_batch_signals(
    signals: List[Dict[str, Any]],
    max_batch_size: int = 100
) -> bool:
    """
    Validate a batch of trading signals.

    Args:
        signals: List of signal dictionaries
        max_batch_size: Maximum signals per batch

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    errors = []

    if not isinstance(signals, list):
        errors.append(f"Signals must be a list, got {type(signals)}")
    elif len(signals) > max_batch_size:
        errors.append(f"Batch size {len(signals)} exceeds maximum {max_batch_size}")

    if isinstance(signals, list):
        for i, signal in enumerate(signals):
            try:
                validate_signal_data(signal)
            except ValidationError as e:
                errors.append(f"Signal {i}: {str(e)}")

    if errors:
        raise ValidationError(
            f"Batch signal validation failed: {'; '.join(errors)}",
            severity=TradingErrorSeverity.HIGH,
            context={
                'batch_size': len(signals) if isinstance(signals, list) else 0,
                'max_batch_size': max_batch_size,
                'errors': errors
            }
        )

    return True
