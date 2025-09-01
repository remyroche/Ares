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
    pass  # TODO: Add implementation
class TradingDataProvider(Protocol):
    pass  # TODO: Add implementation
class TradingDataProvider(Protocol):
    """Protocol for trading data providers."""

@abstractmethod
async def get_market_data(self, symbol: Symbol, start_time: Timestamp, end_time: Timestamp) -> dict:
        ...

@abstractmethod
async def get_live_data(self, symbol: Symbol) -> dict:
        ...

@abstractmethod
async def get_account_info(self) -> dict:
        ...

@abstractmethod
async def get_positions(self) -> list[PositionInfo]:
        ...

@abstractmethod
def is_connected(self) -> bool:
        ...


@runtime_checkable
class TradingMLPredictor(Protocol):
    pass  # TODO: Add implementation
class TradingMLPredictor(Protocol):
    pass  # TODO: Add implementation
class TradingMLPredictor(Protocol):
    """Protocol for ML trading predictors."""

@abstractmethod
async def predict_market_direction(self, input_data: ModelInput) -> PredictionResult:
        ...

@abstractmethod
async def classify_regime(self, input_data: ModelInput) -> RegimeClassification:
        ...

@abstractmethod
async def generate_signals(self, input_data: ModelInput) -> list[TradingSignal]:
        ...

@abstractmethod
def get_model_confidence(self) -> float:
        ...

@abstractmethod
def is_model_ready(self) -> bool:
        ...


@runtime_checkable
class TradingRiskManager(Protocol):
    pass  # TODO: Add implementation
class TradingRiskManager(Protocol):
    pass  # TODO: Add implementation
class TradingRiskManager(Protocol):
    """Protocol for trading risk management."""

@abstractmethod
async def validate_trade(self, trade_decision: TradeDecision) -> bool:
        ...

@abstractmethod
async def calculate_position_size(
self, symbol: Symbol, account_info: dict, risk_parameters: RiskParameters
) -> float:
        ...

@abstractmethod
async def assess_portfolio_risk(self, positions: list[PositionInfo]) -> dict[str, float]:
        ...

@abstractmethod
async def get_stop_loss_price(self, symbol: Symbol, entry_price: float, position_side: str) -> float:
        ...
