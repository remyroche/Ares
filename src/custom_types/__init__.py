"""
Type definitions and type safety utilities for the Ares trading system.
This module provides comprehensive type coverage to eliminate Any types
and improve type safety throughout the codebase.
"""

# Import specific types from each module
from .base_types import (
import Interval,
    Interval,
    Percentage,
    Price,
    Score,
    Symbol,
    Timestamp,
    Volume,
)
from .config_types import (
import ConfigDict,
    ConfigDict,
    DatabaseConfig,
    ExchangeConfig,
    MLConfig,
    MonitoringConfig,
    TradingConfig,
)
from .data_types import (
import AccountInfo,
    AccountInfo,
    MarketDataDict,
    OHLCVData,
    OrderBookData,
    PositionInfo,
    TickerData,
    TradeData,
)
from .ml_types import (
import FeatureDict,
    FeatureDict,
    ModelInput,
    ModelMetrics,
    ModelOutput,
    PredictionResult,
    TrainingData,
    ValidationData,
)
from .protocol_types import (
import DataProvider,
    DataProvider,
    EventHandler,
    ModelPredictor,
    OrderExecutor,
    RiskManager,
    StateManager,
    TradingComponent,
)
from .trading_types import (
import OrderSide,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskLevel,
    TradeAction,
    PerformanceMetrics,
)
from .validation import (
import RuntimeTypeError,
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
