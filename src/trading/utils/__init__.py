"""
Trading Utilities

Utilities for trading operations including error handling,
validation, and helper functions.
"""

from .error_handling import (
    TradingError, RegimeDetectionError, SignalGenerationError,
    PositionSizingError, ExecutionError, DataCollectionError,
    ConfigurationError, ValidationError, NetworkError, RateLimitError,
    InsufficientFundsError, InvalidSymbolError, MarketClosedError,
    TradingErrorSeverity, trading_error_handler, require_no_fallback,
    critical_operation, warn_on_failure, extract_symbol_context,
    extract_market_data_context
)
from .validation import (
    validate_trading_config, validate_market_data, validate_signal_data,
    validate_position_size, validate_regime_data, validate_order_params,
    validate_order_precision, validate_leverage, validate_order_type_compatibility,
    validate_position, validate_account_balance, validate_market_hours,
    validate_batch_orders, validate_batch_signals
)
from .helpers import (
    calculate_returns, normalize_price_data, format_trading_metrics,
    calculate_atr14, calculate_realized_volatility, calculate_three_bar_momentum,
    calculate_three_bar_rsi, calculate_volatility_slope, calculate_volatility,
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_calmar_ratio,
    calculate_max_drawdown, calculate_maximum_adverse_excursion,
    calculate_maximum_favorable_excursion, calculate_omega_ratio,
    calculate_position_metrics, create_trading_summary, log_trading_summary,
    save_trading_data, prepare_trailing_feature_bundle, TrailingFeatureBundle
)
from .ohlcv import ensure_ohlcv_dataframe
from .retry import (
    retry_on_error, retry_on_rate_limit, retry_on_network_error
)
from .circuit_breaker import (
    CircuitBreaker, CircuitState, circuit_breaker
)
from .ohlcv_validation import (
    detect_timestamp_gaps, detect_price_jumps, detect_volume_spikes,
    validate_ohlcv_enhanced, validate_multi_timeframe_consistency
)
from .timeseries import (
    align_time_series, fill_time_series_gaps, resample_time_series,
    validate_time_series_continuity, merge_time_series,
    detect_time_series_anomalies, aggregate_time_series_features
)
from .data_quality import (
    calculate_completeness_score, calculate_consistency_score,
    calculate_freshness_score, calculate_data_quality_score,
    score_data_quality, DataQualityScore
)

__all__ = [
    # Error handling
    'TradingError', 'RegimeDetectionError', 'SignalGenerationError',
    'PositionSizingError', 'ExecutionError', 'DataCollectionError',
    'ConfigurationError', 'ValidationError', 'NetworkError', 'RateLimitError',
    'InsufficientFundsError', 'InvalidSymbolError', 'MarketClosedError',
    'TradingErrorSeverity', 'trading_error_handler', 'require_no_fallback',
    'critical_operation', 'warn_on_failure', 'extract_symbol_context',
    'extract_market_data_context',
    # Validation
    'validate_trading_config', 'validate_market_data', 'validate_signal_data',
    'validate_position_size', 'validate_regime_data', 'validate_order_params',
    'validate_order_precision', 'validate_leverage', 'validate_order_type_compatibility',
    'validate_position', 'validate_account_balance', 'validate_market_hours',
    'validate_batch_orders', 'validate_batch_signals',
    # Helpers
    'calculate_returns', 'normalize_price_data', 'format_trading_metrics',
    'calculate_atr14', 'calculate_realized_volatility', 'calculate_three_bar_momentum',
    'calculate_three_bar_rsi', 'calculate_volatility_slope', 'calculate_volatility',
    'calculate_sharpe_ratio', 'calculate_sortino_ratio', 'calculate_calmar_ratio',
    'calculate_max_drawdown', 'calculate_maximum_adverse_excursion',
    'calculate_maximum_favorable_excursion', 'calculate_omega_ratio',
    'calculate_position_metrics', 'create_trading_summary', 'log_trading_summary',
    'save_trading_data', 'prepare_trailing_feature_bundle', 'TrailingFeatureBundle',
    # OHLCV
    'ensure_ohlcv_dataframe',
    # Retry
    'retry_on_error', 'retry_on_rate_limit', 'retry_on_network_error',
    # Circuit breaker
    'CircuitBreaker', 'CircuitState', 'circuit_breaker',
    # OHLCV validation
    'detect_timestamp_gaps', 'detect_price_jumps', 'detect_volume_spikes',
    'validate_ohlcv_enhanced', 'validate_multi_timeframe_consistency',
    # Time series
    'align_time_series', 'fill_time_series_gaps', 'resample_time_series',
    'validate_time_series_continuity', 'merge_time_series',
    'detect_time_series_anomalies', 'aggregate_time_series_features',
    # Data quality
    'calculate_completeness_score', 'calculate_consistency_score',
    'calculate_freshness_score', 'calculate_data_quality_score',
    'score_data_quality', 'DataQualityScore'
]
