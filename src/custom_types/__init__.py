"""
Type definitions and type safety utilities for the Ares trading system.
This module provides comprehensive type coverage to eliminate Any types
and improve type safety throughout the codebase.
"""

    connection_error,
    critical,
    error,
    execution_error,
    failed,
    initialization_error,
    invalid,
    missing,
    problem,
    timeout,
    validation_error,
    warning,
)

# Import specific types from each module
from .base_types import (
    Interval,
    Percentage,
    Price,
    Score,
    Symbol,
    Timestamp,
    Volume,
)
from .config_types import (
    ConfigDict,
    DatabaseConfig,
    ExchangeConfig,
    MLConfig,
    MonitoringConfig,
    TradingConfig,
)
from .data_types import (
    AccountInfo,
    MarketDataDict,
    OHLCVData,
    OrderBookData,
    PositionInfo,
    TickerData,
    TradeData,
)
from .ml_types import (
    FeatureDict,
    ModelInput,
    ModelMetrics,
    ModelOutput,
    PredictionResult,
    TrainingData,
    ValidationData,
)
from .protocol_types import (
    DataProvider,
    EventHandler,
    ModelPredictor,
    OrderExecutor,
    RiskManager,
    StateManager,
    TradingComponent,
)
from .trading_types import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskLevel,
    TradeAction,
    PerformanceMetrics,
)
from .validation import (
    RuntimeTypeError,
    TypeValidator,
    validate_config,
    validate_market_data,
    validate_model_input,
    validate_type,
)

__all__ = [
    # Base types
    "Timestamp",
    "Symbol",
    "Price",
    "Volume",
    "Percentage",
    "Score",
    "Interval",
    # Config types
    "ConfigDict",
    "DatabaseConfig",
    "ExchangeConfig",
    "TradingConfig",
    "MLConfig",
    "MonitoringConfig",
    # Data types
    "MarketDataDict",
    "OHLCVData",
    "TickerData",
    "OrderBookData",
    "TradeData",
    "AccountInfo",
    "PositionInfo",
    # ML types
    "ModelInput",
    "ModelOutput",
    "PredictionResult",
    "FeatureDict",
    "ModelMetrics",
    "TrainingData",
    "ValidationData",
    # Trading types
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "PositionSide",
    "TradeAction",
    "RiskLevel",
    "PerformanceMetrics",
    # Protocols
    "DataProvider",
    "ModelPredictor",
    "RiskManager",
    "OrderExecutor",
    "StateManager",
    "EventHandler",
    "TradingComponent",
    # Validation
    "TypeValidator",
    "validate_type",
    "validate_config",
    "validate_market_data",
    "validate_model_input",
    "RuntimeTypeError",
]
