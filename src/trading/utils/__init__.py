"""
Trading Utilities

Utilities for trading operations including error handling,
validation, and helper functions.
"""

from .error_handling import (
    TradingError,
    RegimeDetectionError,
    SignalGenerationError,
    PositionSizingError,
    ExecutionError,
    DataCollectionError,
    ValidationError,
    TradingErrorSeverity,
    trading_error_handler,
    handle_trading_errors,
    log_trading_error,
    create_trading_error_context
)
from .validation import (
    validate_trading_config,
    validate_market_data,
    validate_trade_params,
    validate_position_size,
    validate_signal_data,
    validate_regime_data,
    validate_execution_params
)
from .helpers import (
    calculate_returns,
    calculate_volatility,
    normalize_price_data,
    format_trading_metrics,
    calculate_atr14,
    calculate_realized_volatility,
    calculate_three_bar_momentum,
    calculate_three_bar_rsi,
    calculate_volatility_slope,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    format_currency,
    get_trading_hours,
    TradingHelper,
    TradingMetrics
)
from .ohlcv import ensure_ohlcv_dataframe

__all__ = [
    'TradingError', 'RegimeDetectionError', 'SignalGenerationError',
    'PositionSizingError', 'ExecutionError', 'DataCollectionError',
    'trading_error_handler', 'validate_trading_config', 'validate_market_data',
    'calculate_returns', 'normalize_price_data', 'format_trading_metrics',
    'calculate_atr14', 'calculate_realized_volatility', 'calculate_three_bar_momentum',
    'calculate_three_bar_rsi', 'calculate_volatility_slope', 'ensure_ohlcv_dataframe'
]
