# src/types/__init__.py

"""
Type definitions and type safety utilities for the Ares trading system.
This module provides comprehensive type coverage to eliminate Any types
and improve type safety throughout the codebase.
"""

from .base_types import *
from .config_types import *
from .data_types import *
from .ml_types import *
from .protocol_types import *
from .trading_types import *
from .validation import *

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
    
    # Protocols
    "DataProvider",
    "ModelPredictor",
    "RiskManager",
    "OrderExecutor",
    "StateManager",
    "EventHandler",
    
    # Validation
    "TypeValidator",
    "validate_type",
    "validate_config",
    "validate_market_data",
    "validate_model_input",
    "RuntimeTypeError",
]