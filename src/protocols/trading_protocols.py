# src/protocols/trading_protocols.py

"""
Enhanced trading system protocols with comprehensive type safety (minimal scaffold).
"""


from abc import abstractmethod
from typing import Protocol, runtime_checkable

from src.custom_types.base_types import Symbol, Timestamp
from src.custom_types.ml_types import ModelInput, PredictionResult
from src.custom_types.trading_types import (
    OrderRequest,
    PerformanceMetrics,
    PositionInfo,
    RegimeClassification,
    RiskParameters,
    TradeDecision,
    TradingSignal,
)


@runtime_checkable
class TradingDataProvider(Protocol):
    """Protocol for trading data providers."""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class TradingMLPredictor(Protocol):
    """Protocol for ML trading predictors."""

    @abstractmethod
    async def predict_market_direction(self, input_data: ModelInput) -> PredictionResult:
        ...

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

@runtime_checkable
class TradingRiskManager(Protocol):
    """Protocol for trading risk management."""

    @abstractmethod
    async def validate_trade(self, trade_decision: TradeDecision) -> bool:
        ...

    @abstractmethod
    @abstractmethod
    @abstractmethod