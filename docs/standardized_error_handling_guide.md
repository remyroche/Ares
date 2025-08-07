# Standardized Error Handling Guide

## Overview

The Ares trading bot now features a comprehensive, categorized error handling system that provides:
- **Intelligent error categorization** (Network, Data, Model, Business Logic, etc.)
- **Automatic recovery strategies** (Retry, Fallback, Circuit Breaker, Graceful Degradation)
- **Severity-based logging** (Critical, High, Medium, Low)
- **Type-safe decorators** for consistent error handling across modules

## Error Categories

### 1. Network Errors (`ErrorCategory.NETWORK`)
- **Covers**: Connection timeouts, rate limits, API failures, exchange connectivity
- **Default Severity**: HIGH
- **Recovery**: Retry with exponential backoff, rate limit handling
- **Example**: Exchange API calls, WebSocket connections

### 2. Data Errors (`ErrorCategory.DATA`)
- **Covers**: Missing data, corrupt files, invalid format, insufficient data
- **Default Severity**: MEDIUM
- **Recovery**: Fallback data sources, data cleaning, retry with limited attempts
- **Example**: Market data collection, feature calculation

### 3. Model Errors (`ErrorCategory.MODEL`)
- **Covers**: ML prediction failures, training issues, model loading errors
- **Default Severity**: HIGH
- **Recovery**: Safe defaults, graceful degradation
- **Example**: Regime classification, price predictions

### 4. Validation Errors (`ErrorCategory.VALIDATION`)
- **Covers**: Input validation, type checking, parameter validation
- **Default Severity**: MEDIUM
- **Recovery**: Graceful degradation, no retry
- **Example**: Configuration validation, input sanitization

### 5. Business Logic Errors (`ErrorCategory.BUSINESS_LOGIC`)
- **Covers**: Trading rule violations, strategy logic errors
- **Default Severity**: HIGH
- **Recovery**: No retry, logging only
- **Example**: Position sizing rules, risk management violations

### 6. System Errors (`ErrorCategory.SYSTEM`)
- **Covers**: File I/O, memory issues, CPU problems, disk space
- **Default Severity**: CRITICAL
- **Recovery**: Limited retry, circuit breaker
- **Example**: File operations, resource exhaustion

### 7. Configuration Errors (`ErrorCategory.CONFIGURATION`)
- **Covers**: Missing config, invalid settings, environment issues
- **Default Severity**: HIGH
- **Recovery**: Safe defaults, fallback configuration
- **Example**: Missing API keys, invalid parameters

### 8. Security Errors (`ErrorCategory.SECURITY`)
- **Covers**: Authentication, authorization, credential issues
- **Default Severity**: CRITICAL
- **Recovery**: No retry, immediate escalation
- **Example**: API key validation, permission errors

### 9. Performance Errors (`ErrorCategory.PERFORMANCE`)
- **Covers**: Slow operations, resource exhaustion, memory leaks
- **Default Severity**: MEDIUM
- **Recovery**: Graceful degradation, simplified operations
- **Example**: Timeout operations, memory warnings

## Usage Examples

### Basic Categorized Error Handling

```python
from src.utils.error_handler import (
    handle_network_errors,
    handle_data_errors,
    handle_model_errors,
    handle_validation_errors,
    handle_business_logic_errors,
    handle_critical_operations,
)

# Network operations (API calls, WebSocket)
@handle_network_errors(context="binance_api", max_retries=5, base_delay=2.0)
async def fetch_market_data(symbol: str):
    # Will automatically retry on network issues with exponential backoff
    # Rate limits get special handling with longer delays
    pass

# Data processing with fallback
@handle_data_errors(context="feature_calc", use_fallback=True)
def calculate_technical_indicators(df: pd.DataFrame):
    # Will return fallback data structure if calculation fails
    # Automatically handles missing/corrupt data
    pass

# ML model predictions with safe defaults
@handle_model_errors(context="regime_prediction")
def predict_market_regime(features: np.ndarray):
    # Returns safe default: {"prediction": 0.0, "confidence": 0.0, "strategy": "hold"}
    # if prediction fails
    pass

# Input validation (no retry)
@handle_validation_errors(context="config_validation")
def validate_trading_config(config: dict):
    # Logs validation errors and returns default/None
    # No retry attempts for validation failures
    pass

# Critical operations with circuit breaker
@handle_critical_operations(context="order_execution", failure_threshold=3)
async def execute_trade_order(order: dict):
    # Circuit breaker opens after 3 failures
    # Comprehensive recovery strategies applied
    pass
```

### Custom Error Creation

```python
from src.utils.error_handler import (
    NetworkError,
    DataError,
    ModelError,
    ValidationError,
    BusinessLogicError,
    SystemError,
    ConfigurationError,
    SecurityError,
    PerformanceError,
    ErrorSeverity,
)

# Create categorized errors with appropriate severity
def validate_api_key(key: str):
    if not key:
        raise SecurityError("API key is required", ErrorSeverity.CRITICAL)
    
def process_market_data(data: pd.DataFrame):
    if data.empty:
        raise DataError("No market data available", ErrorSeverity.MEDIUM)
    
def execute_trading_strategy(signal: float):
    if abs(signal) > 1.0:
        raise BusinessLogicError("Invalid trading signal", ErrorSeverity.HIGH)
```

### Advanced Recovery Configuration

```python
from src.utils.error_handler import (
    ErrorHandler,
    NetworkRecoveryStrategy,
    DataRecoveryStrategy,
    RetryStrategy,
    GracefulDegradationStrategy,
)

# Custom error handler with specific strategies
handler = ErrorHandler(context="custom_trading_logic")

# Add custom recovery strategies
handler.recovery_manager.add_strategy(
    RetryStrategy(
        categories=[ErrorCategory.NETWORK],
        max_retries=3,
        base_delay=1.0
    )
)

handler.recovery_manager.add_strategy(
    DataRecoveryStrategy(use_fallback_data=True)
)

# Use custom handler
@handler.handle_generic_errors(
    exceptions=(Exception,),
    default_return=None
)
async def custom_operation():
    pass
```

## Applied Modules

The standardized error handling has been applied to these critical modules:

### 1. Exchange Operations (`exchange/binance.py`)
- **Network errors**: API calls, WebSocket connections
- **Rate limiting**: Specialized handling for 429 responses
- **Connection issues**: Automatic retry with circuit breaker

### 2. Data Collection (`src/training/steps/step1_data_collection.py`)
- **Data errors**: Missing files, corrupt data, insufficient data
- **Fallback strategies**: Alternative data sources
- **Legacy compatibility**: Error class aliases maintained

### 3. Market Analysis (`src/analyst/analyst.py`)
- **Model errors**: ML prediction failures with safe defaults
- **Data errors**: Technical analysis with fallback values
- **Mixed strategies**: Different handling for different analysis types

### 4. Regime Classification (`src/analyst/unified_regime_classifier.py`)
- **Model errors**: Prediction failures return safe defaults
- **Data errors**: Feature calculation with fallback
- **Training errors**: Model training with proper error categorization
- **Validation**: Input data validation with appropriate errors

### 5. Training Validation (`src/training/steps/step2_*`)
- **Validation errors**: Training parameter validation
- **Model errors**: Training process error handling
- **Data consistency**: Validation of training data integrity

## Error Logging Enhancement

All errors are now logged with:
- **Category prefix**: `[NETWORK]`, `[DATA]`, `[MODEL]`, etc.
- **Severity-based levels**: Critical → critical, High → error, Medium → warning, Low → info
- **Rich context**: Function name, error type, timestamp, category, severity
- **Structured data**: Extra fields for monitoring and analysis

Example log output:
```
2024-01-15 10:30:45 ERROR [NETWORK] Error in binance_api.fetch_orderbook: Rate limit exceeded for fetch_orderbook: 429 Too Many Requests
2024-01-15 10:30:46 WARNING [DATA] Error in feature_calculation.calculate_rsi: Missing price data, using fallback calculation
2024-01-15 10:30:47 ERROR [MODEL] Error in regime_prediction.predict_regime: Model prediction failed, returning safe default
```

## Recovery Strategies

### 1. Network Recovery Strategy
- **Rate limit detection**: Automatic backoff for 429 errors
- **Connection timeout handling**: Reduced timeouts for retries
- **Progressive backoff**: 1s, 2s, 4s, 8s, 16s delays

### 2. Data Recovery Strategy
- **Fallback data**: Returns structured fallback when data missing
- **Data cleaning**: Attempts to clean corrupt data
- **Alternative sources**: Tries cached or historical data

### 3. Model Recovery Strategy
- **Safe defaults**: Returns neutral predictions
- **Confidence indicators**: Low confidence for fallback predictions
- **Strategy safety**: Always returns "hold" for failed predictions

### 4. Graceful Degradation
- **Performance fallback**: Simplified calculations for slow operations
- **Memory management**: Reduced data processing for memory issues
- **Feature toggles**: Disables non-critical features under stress

## Migration Guide

### For Existing Code

1. **Replace generic decorators**:
```python
# Old
@handle_errors(exceptions=(Exception,), default_return=None)

# New
@handle_network_errors(context="api_call")  # for network operations
@handle_data_errors(context="processing")   # for data operations
@handle_model_errors(context="prediction")  # for ML operations
```

2. **Update exception raising**:
```python
# Old
raise Exception("API call failed")

# New
raise NetworkError("API call failed")
```

3. **Add context to decorators**:
```python
# Old
@handle_errors()

# New
@handle_network_errors(context="exchange_api")
```

### Best Practices

1. **Choose appropriate categories**: Match error types to categories
2. **Set meaningful contexts**: Use descriptive context strings
3. **Test recovery paths**: Ensure fallback behaviors work correctly
4. **Monitor error patterns**: Use structured logging for analysis
5. **Update gradually**: Migrate module by module to avoid disruption

## Performance Impact

- **Minimal overhead**: Decorators add ~1-2ms per function call
- **Smart retry**: Prevents unnecessary retries for non-recoverable errors
- **Circuit breaker**: Prevents cascade failures
- **Efficient logging**: Structured logging with appropriate levels

## Monitoring and Alerts

The categorized error system enables:
- **Category-based alerts**: Different thresholds for different error types
- **Trend analysis**: Track error patterns over time
- **Recovery effectiveness**: Monitor success rates of recovery strategies
- **Performance impact**: Measure error handling overhead

## Future Enhancements

Planned improvements include:
- **Dynamic thresholds**: Adaptive retry limits based on success rates
- **Error prediction**: ML-based error forecasting
- **Recovery optimization**: Self-learning recovery strategies
- **Integration monitoring**: Cross-service error correlation