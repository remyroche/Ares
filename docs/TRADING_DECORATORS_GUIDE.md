# Trading Decorators Guide

## Overview

This guide provides comprehensive documentation for the trading decorators system designed to enhance your trading and backtesting pipelines with error handling, trade tracking, monitoring, and operational management capabilities.

## Table of Contents

1. [Introduction](#introduction)
2. [Error Handling Decorators](#error-handling-decorators)
3. [Trade Tracking Decorators](#trade-tracking-decorators)
4. [Performance Monitoring Decorators](#performance-monitoring-decorators)
5. [Operational Decorators](#operational-decorators)
6. [Composite Decorators](#composite-decorators)
7. [Integration Examples](#integration-examples)
8. [Best Practices](#best-practices)
9. [Configuration Options](#configuration-options)

## Introduction

The trading decorators system provides a comprehensive set of decorators that can be applied to your existing trading and backtesting functions to add:

- **Error Handling**: Robust error handling with retries, circuit breakers, and fallback strategies
- **Trade Tracking**: Comprehensive tracking of all trade data including model weights, confidences, regime analysis, and market conditions
- **Performance Monitoring**: Real-time performance monitoring with alerts for slow operations
- **Operational Management**: Rate limiting, validation, and operational safety measures

### Key Benefits

1. **Non-intrusive**: Decorators can be added to existing functions without changing their core logic
2. **Comprehensive**: Captures all relevant trading data for analysis and monitoring
3. **Configurable**: Each decorator can be customized for specific use cases
4. **Type Safe**: Full type hints and validation
5. **Async Support**: Works with both synchronous and asynchronous functions

## Error Handling Decorators

### `trading_error_handler`

Enhanced error handling specifically designed for trading operations.

```python
@trading_error_handler(
    retry_attempts=3,
    retry_delay=1.0,
    circuit_breaker_threshold=5,
    fallback_strategy=my_fallback_function
)
async def execute_trade(symbol: str, quantity: float, price: float):
    # Your trading logic here
    pass
```

**Parameters:**
- `retry_attempts`: Number of retry attempts (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)
- `circuit_breaker_threshold`: Number of failures before circuit breaker opens (default: 5)
- `fallback_strategy`: Fallback function to call if all retries fail

**Features:**
- Exponential backoff with jitter
- Circuit breaker pattern
- Fallback strategy support
- Comprehensive error logging

### `market_data_error_handler`

Specialized error handling for market data operations with data validation and caching.

```python
@market_data_error_handler(
    data_validation=True,
    fallback_to_cached=True,
    max_age_seconds=300
)
async def fetch_market_data(symbol: str, interval: str):
    # Your market data fetching logic here
    pass
```

**Parameters:**
- `data_validation`: Whether to validate data quality (default: True)
- `fallback_to_cached`: Whether to fallback to cached data (default: True)
- `max_age_seconds`: Maximum age of cached data to use (default: 300)

**Features:**
- Data quality validation
- Automatic fallback to cached data
- Stale data detection
- Integration with your existing database systems

## Trade Tracking Decorators

### `track_trade`

Comprehensive trade tracking that captures all relevant trading data.

```python
@track_trade(
    capture_model_data=True,
    capture_regime_data=True,
    capture_market_conditions=True,
    capture_risk_metrics=True
)
async def execute_trade(symbol: str, side: str, quantity: float, price: float):
    # Your trade execution logic here
    pass
```

**Parameters:**
- `capture_model_data`: Whether to capture model weights and confidences (default: True)
- `capture_regime_data`: Whether to capture regime analysis (default: True)
- `capture_market_conditions`: Whether to capture market conditions (default: True)
- `capture_risk_metrics`: Whether to capture risk metrics (default: True)

**Captured Data:**
- Trade execution details (symbol, side, quantity, price, timestamp)
- Model ensemble data (weights, confidences, predictions)
- Regime analysis (HMM regime, regime confidence, regime features)
- Support/resistance levels
- Market conditions (trend, volume, volatility)
- Risk metrics (VaR, max drawdown, Sharpe ratio)
- Performance metrics (execution time, success status)

### `track_model_performance`

Tracks model performance metrics including predictions, feature importance, and confidence scores.

```python
@track_model_performance(
    model_name="XGBoost_Ensemble",
    capture_predictions=True,
    capture_feature_importance=True,
    capture_confidence=True
)
async def predict_with_ensemble(features: pd.DataFrame):
    # Your model prediction logic here
    pass
```

**Parameters:**
- `model_name`: Name of the model to track
- `capture_predictions`: Whether to capture predictions (default: True)
- `capture_feature_importance`: Whether to capture feature importance (default: True)
- `capture_confidence`: Whether to capture confidence scores (default: True)

**Captured Data:**
- Model predictions and probabilities
- Feature importance rankings
- Confidence scores
- Execution time and performance metrics
- Error tracking for failed predictions

## Performance Monitoring Decorators

### `monitor_performance`

Monitors function performance and alerts on slow operations.

```python
@monitor_performance(
    alert_threshold_ms=1000.0,
    log_slow_operations=True,
    capture_memory_usage=False
)
async def process_market_data(data: pd.DataFrame):
    # Your data processing logic here
    pass
```

**Parameters:**
- `alert_threshold_ms`: Threshold in milliseconds to trigger alerts (default: 1000.0)
- `log_slow_operations`: Whether to log slow operations (default: True)
- `capture_memory_usage`: Whether to capture memory usage (default: False)

**Features:**
- Performance threshold monitoring
- Automatic alerting for slow operations
- Memory usage tracking (optional)
- Integration with monitoring systems

### `validate_trade_parameters`

Validates trade parameters before execution.

```python
@validate_trade_parameters(
    validate_price=True,
    validate_quantity=True,
    validate_symbol=True,
    min_price=0.0,
    min_quantity=0.0
)
async def place_order(symbol: str, quantity: float, price: float):
    # Your order placement logic here
    pass
```

**Parameters:**
- `validate_price`: Whether to validate price (default: True)
- `validate_quantity`: Whether to validate quantity (default: True)
- `validate_symbol`: Whether to validate symbol (default: True)
- `min_price`: Minimum valid price (default: 0.0)
- `min_quantity`: Minimum valid quantity (default: 0.0)

**Features:**
- Parameter validation before execution
- Configurable validation rules
- Early error detection
- Prevents invalid trades

## Operational Decorators

### `rate_limit`

Rate limiting for API calls and trading operations.

```python
@rate_limit(
    max_calls=100,
    time_window=60.0
)
async def call_exchange_api(symbol: str):
    # Your API call logic here
    pass
```

**Parameters:**
- `max_calls`: Maximum number of calls allowed (default: 100)
- `time_window`: Time window in seconds (default: 60.0)

**Features:**
- Sliding window rate limiting
- Automatic waiting when limit reached
- Configurable limits per function
- Prevents API rate limit violations

### `circuit_breaker`

Circuit breaker pattern for trading operations.

```python
@circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    monitor_interval=10.0
)
async def risky_operation():
    # Your risky operation logic here
    pass
```

**Parameters:**
- `failure_threshold`: Number of failures before opening circuit (default: 5)
- `recovery_timeout`: Time to wait before attempting recovery (default: 60.0)
- `monitor_interval`: Interval to check circuit state (default: 10.0)

**Features:**
- Automatic circuit breaker management
- Three states: CLOSED, OPEN, HALF_OPEN
- Automatic recovery after timeout
- Prevents cascading failures

### `retry_with_backoff`

Retry decorator with exponential backoff and jitter.

```python
@retry_with_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    backoff_factor=2.0,
    jitter=True
)
async def unreliable_operation():
    # Your unreliable operation logic here
    pass
```

**Parameters:**
- `max_retries`: Maximum number of retry attempts (default: 3)
- `base_delay`: Base delay between retries (default: 1.0)
- `max_delay`: Maximum delay between retries (default: 60.0)
- `backoff_factor`: Factor to multiply delay by each retry (default: 2.0)
- `jitter`: Whether to add random jitter to delays (default: True)

**Features:**
- Exponential backoff
- Random jitter to prevent thundering herd
- Configurable retry limits
- Automatic failure handling

## Composite Decorators

### `comprehensive_trade_decorator`

Convenience decorator that combines multiple trading decorators.

```python
@comprehensive_trade_decorator(
    enable_error_handling=True,
    enable_tracking=True,
    enable_performance_monitoring=True,
    enable_validation=True,
    enable_rate_limiting=True,
    enable_circuit_breaker=True,
    retry_attempts=3,
    alert_threshold_ms=2000.0,
    max_calls=50,
    time_window=60.0
)
async def execute_live_trade(symbol: str, side: str, quantity: float, price: float):
    # Your live trading logic here
    pass
```

**Features:**
- Combines all major trading decorators
- Configurable enable/disable for each feature
- Optimized for live trading scenarios
- Comprehensive safety and monitoring

### `comprehensive_model_decorator`

Convenience decorator for model operations.

```python
@comprehensive_model_decorator(
    enable_error_handling=True,
    enable_tracking=True,
    enable_performance_monitoring=True,
    enable_retry=True,
    model_name="XGBoost_Ensemble",
    capture_predictions=True,
    capture_feature_importance=True,
    capture_confidence=True,
    retry_attempts=3,
    alert_threshold_ms=3000.0
)
async def predict_with_ensemble(features: pd.DataFrame):
    # Your ensemble prediction logic here
    pass
```

**Features:**
- Combines model-specific decorators
- Optimized for ML model operations
- Comprehensive model tracking
- Performance monitoring for ML workloads

## Integration Examples

### Backtesting Pipeline Integration

```python
class EnhancedBacktester:
    def __init__(self, config):
        self.config = config
        self.trade_tracker = get_trade_tracker()
    
    @comprehensive_trade_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_validation=True,
        enable_rate_limiting=False,  # Not needed for backtesting
        enable_circuit_breaker=True,
        retry_attempts=3,
        alert_threshold_ms=5000.0
    )
    async def execute_backtest_trade(
        self,
        symbol: str,
        signal: int,
        price: float,
        timestamp: datetime,
        trade_metadata: Dict[str, Any] = None
    ):
        # Your backtesting logic here
        # All trade data will be automatically tracked
        pass
```

### Live Trading Pipeline Integration

```python
class LiveTradingPipeline:
    def __init__(self, config):
        self.config = config
        self.trade_tracker = get_trade_tracker()
    
    @comprehensive_trade_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_validation=True,
        enable_rate_limiting=True,
        enable_circuit_breaker=True,
        max_calls=50,
        time_window=60.0,
        alert_threshold_ms=2000.0
    )
    async def execute_live_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = 'market',
        model_weights: Dict[str, float] = None,
        model_confidences: Dict[str, float] = None,
        regime_analysis: Dict[str, Any] = None,
        hmm_regime: str = '',
        support_resistance_levels: Dict[str, float] = None,
        market_conditions: Dict[str, Any] = None,
        risk_metrics: Dict[str, float] = None
    ):
        # Your live trading logic here
        # All safety measures and tracking are automatic
        pass
```

### Model Operations Integration

```python
class ModelOperations:
    @comprehensive_model_decorator(
        enable_error_handling=True,
        enable_tracking=True,
        enable_performance_monitoring=True,
        enable_retry=True,
        model_name="XGBoost_Ensemble",
        capture_predictions=True,
        capture_feature_importance=True,
        capture_confidence=True,
        retry_attempts=3,
        alert_threshold_ms=3000.0
    )
    async def predict_with_ensemble(
        self,
        features: pd.DataFrame,
        model_weights: Dict[str, float],
        regime_analysis: Dict[str, Any] = None
    ):
        # Your ensemble prediction logic here
        # All model performance will be tracked
        pass
```

## Best Practices

### 1. Decorator Order

Use the comprehensive decorators for most cases, as they include all necessary functionality:

```python
# Recommended: Use comprehensive decorator (includes tracking, error handling, monitoring)
@comprehensive_trade_decorator(
    enable_error_handling=True,
    enable_tracking=True,
    enable_performance_monitoring=True,
    enable_validation=True
)
async def your_function():
    pass

# Only use individual decorators when you need specific functionality
@track_trade(capture_model_data=True)
@monitor_performance(alert_threshold_ms=1000.0)
async def specific_function():
    pass
```

**Note**: Avoid stacking `@comprehensive_trade_decorator` with `@track_trade` as this will cause double logging.

### 2. Configuration Management

Use configuration files to manage decorator parameters:

```python
# config/trading_decorators.yaml
trading_decorators:
  live_trading:
    enable_error_handling: true
    enable_tracking: true
    enable_performance_monitoring: true
    enable_validation: true
    enable_rate_limiting: true
    enable_circuit_breaker: true
    max_calls: 50
    time_window: 60.0
    alert_threshold_ms: 2000.0
  
  backtesting:
    enable_error_handling: true
    enable_tracking: true
    enable_performance_monitoring: true
    enable_validation: true
    enable_rate_limiting: false
    enable_circuit_breaker: true
    retry_attempts: 3
    alert_threshold_ms: 5000.0
```

### 3. Using TradeContext for Cleaner Code

For cleaner function signatures, use the `TradeContext` dataclass:

```python
from src.utils.trading_decorators import TradeContext, TradeSide, ExecutionMode

# Create trade context
trade_context = TradeContext(
    symbol='BTCUSDT',
    side=TradeSide.BUY,
    quantity=0.001,
    price=150.0,
    timestamp=datetime.now(),
    execution_mode=ExecutionMode.LIVE,
    model_weights={'xgboost': 0.4, 'lstm': 0.3, 'random_forest': 0.3},
    model_confidences={'xgboost': 0.85, 'lstm': 0.78, 'random_forest': 0.82},
    regime_analysis={'regime_type': 'trending', 'regime_confidence': 0.75},
    hmm_regime='bull_market',
    support_resistance_levels={'support': 148.0, 'resistance': 155.0},
    market_conditions={'trend': 'upward', 'volume': 'high'},
    risk_metrics={'var_95': 0.02, 'max_drawdown': 0.05}
)

# Use in function calls
await trader.execute_buy_order(
    symbol='BTCUSDT',
    quantity=0.001,
    price=150.0,
    timestamp=datetime.now(),
    trade_context=trade_context
)
```

### 4. Trade Data Structure

Structure your trade metadata consistently:

```python
trade_metadata = {
    'model_weights': {
        'xgboost': 0.4,
        'lstm': 0.3,
        'random_forest': 0.3
    },
    'model_confidences': {
        'xgboost': 0.85,
        'lstm': 0.78,
        'random_forest': 0.82
    },
    'regime_analysis': {
        'regime_type': 'trending',
        'regime_confidence': 0.75,
        'volatility': 0.15
    },
    'hmm_regime': 'bull_market',
    'support_resistance_levels': {
        'support': 148.0,
        'resistance': 155.0
    },
    'market_conditions': {
        'trend': 'upward',
        'volume': 'high',
        'volatility': 'medium'
    },
    'risk_metrics': {
        'var_95': 0.02,
        'max_drawdown': 0.05,
        'sharpe_ratio': 1.2
    }
}
```

### 5. Error Handling Strategy

Implement a layered error handling strategy:

```python
# Layer 1: Function-specific error handling
@trading_error_handler(retry_attempts=3, fallback_strategy=fallback_function)
async def risky_operation():
    pass

# Layer 2: Circuit breaker for repeated failures
@circuit_breaker(failure_threshold=5, recovery_timeout=60.0)
async def api_call():
    pass

# Layer 3: Rate limiting for API protection
@rate_limit(max_calls=100, time_window=60.0)
async def exchange_api_call():
    pass
```

### 6. Monitoring and Alerting

Set up proper monitoring and alerting:

```python
# Configure performance monitoring
@monitor_performance(
    alert_threshold_ms=1000.0,
    log_slow_operations=True,
    capture_memory_usage=True
)
async def critical_operation():
    pass

# Set up trade tracking
@track_trade(
    capture_model_data=True,
    capture_regime_data=True,
    capture_market_conditions=True,
    capture_risk_metrics=True
)
async def execute_trade():
    pass
```

## Configuration Options

### Environment-Specific Configurations

#### Live Trading Configuration

```python
LIVE_TRADING_CONFIG = {
    'enable_error_handling': True,
    'enable_tracking': True,
    'enable_performance_monitoring': True,
    'enable_validation': True,
    'enable_rate_limiting': True,
    'enable_circuit_breaker': True,
    'max_calls': 50,
    'time_window': 60.0,
    'alert_threshold_ms': 2000.0,
    'retry_attempts': 3,
    'failure_threshold': 5,
    'recovery_timeout': 60.0
}
```

#### Backtesting Configuration

```python
BACKTESTING_CONFIG = {
    'enable_error_handling': True,
    'enable_tracking': True,
    'enable_performance_monitoring': True,
    'enable_validation': True,
    'enable_rate_limiting': False,
    'enable_circuit_breaker': True,
    'alert_threshold_ms': 5000.0,
    'retry_attempts': 3,
    'failure_threshold': 10,
    'recovery_timeout': 30.0
}
```

#### Model Operations Configuration

```python
MODEL_CONFIG = {
    'enable_error_handling': True,
    'enable_tracking': True,
    'enable_performance_monitoring': True,
    'enable_retry': True,
    'capture_predictions': True,
    'capture_feature_importance': True,
    'capture_confidence': True,
    'retry_attempts': 3,
    'alert_threshold_ms': 3000.0,
    'base_delay': 1.0,
    'max_delay': 30.0,
    'backoff_factor': 2.0,
    'jitter': True
}
```

### Trade Tracker Configuration

```python
TRADE_TRACKER_CONFIG = {
    'enable_feature_importance_tracking': True,
    'enable_decision_path_tracking': True,
    'enable_model_behavior_tracking': True,
    'max_trade_history': 10000,
    'enable_detailed_logging': True,
    'enable_performance_metrics': True,
    'enable_regime_analysis': True,
    'enable_risk_metrics': True
}
```

## Conclusion

The trading decorators system provides a comprehensive solution for enhancing your trading and backtesting pipelines with minimal code changes. By applying these decorators strategically, you can achieve:

1. **Robust Error Handling**: Automatic retries, circuit breakers, and fallback strategies
2. **Comprehensive Tracking**: Complete trade data capture for analysis and monitoring
3. **Performance Monitoring**: Real-time performance tracking with alerts
4. **Operational Safety**: Rate limiting, validation, and safety measures
5. **Easy Integration**: Non-intrusive decorators that work with existing code

Start by applying the composite decorators to your main trading functions, then customize individual decorators as needed for specific use cases. The system is designed to be flexible and scalable, growing with your trading system's needs.