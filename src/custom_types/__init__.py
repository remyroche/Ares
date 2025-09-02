"""
Type definitions and type safety utilities for the Ares trading system.

This module provides comprehensive type coverage to eliminate Any types
and improve type safety throughout the codebase. It includes:

- Base type definitions (Timestamp, Symbol, Price, etc.)
- Configuration type definitions (Database, Exchange, ML, etc.)
- Trading type definitions (Orders, Positions, Signals, etc.)
- Validation utilities and error handling

Version: 1.0.0
Last Updated: 2025-09-02
"""

from __future__ import annotations

# Import base types
from .base_types import (
    ConfidenceLevel, Interval, LeverageMultiplier, ModelId, OrderId, 
    Percentage, PositionId, Price, RiskScore, Score, SessionId, 
    Symbol, Timestamp, TradeId, UserId, Volume
)

# Import specific types from each module
from .config_types import (
    ConfigDict, DatabaseConfig, ExchangeConfig,
    MLConfig, MonitoringConfig, TradingConfig, SystemConfig, TrainingConfig
)

from .trading_types import (
    OrderSide, OrderStatus, OrderType, PositionSide, RiskLevel, TradeAction,
    OrderRequest, PositionInfo, TradeDecision, TradingSignal, PositionRisk, RiskParameters
)

from .validation import (
    RuntimeTypeError, TypeValidator, validate_market_data,
    validate_model_input, handle_errors
)

# Module version
__version__ = "1.0.0"

__all__ = [
    # Base types
    "Timestamp", "Symbol", "Price", "Volume", "Percentage", "Score", "Interval",
    "OrderId", "TradeId", "PositionId", "ModelId", "UserId", "SessionId",
    "LeverageMultiplier", "RiskScore", "ConfidenceLevel",
    
    # Config types
    "ConfigDict", "DatabaseConfig", "ExchangeConfig", "TradingConfig", 
    "MLConfig", "MonitoringConfig", "SystemConfig", "TrainingConfig",
    
    # Trading types
    "OrderType", "OrderSide", "OrderStatus", "PositionSide", "TradeAction",
    "OrderRequest", "PositionInfo", "TradeDecision", "TradingSignal",
    "PositionRisk", "RiskParameters", "RiskLevel",
    
    # Validation
    "TypeValidator", "validate_market_data", "validate_model_input", 
    "RuntimeTypeError", "handle_errors",
    
    # Module metadata
    "__version__",
]
