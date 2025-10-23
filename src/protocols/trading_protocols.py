# src/protocols/trading_protocols.py

"""
Enhanced trading system protocols with comprehensive type safety (minimal scaffold).
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable, NamedTuple
from datetime import datetime
from typing import NewType

# Define basic types for runtime use
Symbol = NewType('Symbol', str)
Timestamp = NewType('Timestamp', datetime)

# Define basic trading types for runtime use
class PositionInfo(NamedTuple):
    symbol: str
    size: float
    side: str
    entry_price: float
    current_price: float
    unrealized_pnl: float
    margin_used: float

class PredictionResult(NamedTuple):
    prediction: float
    confidence: float
    probability: float
    features_used: list
    model_version: str
    timestamp: datetime

class ModelInput(NamedTuple):
    features: list
    symbol: str
    timestamp: datetime
    market_data: dict

class RegimeClassification(NamedTuple):
    regime: str
    confidence: float
    probability_distribution: dict
    features_used: list
    timestamp: datetime

class RiskParameters(NamedTuple):
    max_position_size: float
    stop_loss_pct: float
    take_profit_pct: float
    max_drawdown: float
    risk_score: float

class TradeDecision(NamedTuple):
    symbol: str
    action: str
    quantity: float
    price: float
    leverage: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_score: float
    timestamp: datetime

class TradingSignal(NamedTuple):
    signal_type: str
    strength: float
    direction: str
    confidence: float
    features: dict
    timestamp: datetime

if TYPE_CHECKING:
    from src.custom_types.ml_types import PredictionResult, ModelInput
    import logging

    from src.custom_types.trading_types import (
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
    async def get_market_data(
        self, symbol: Symbol, start_time: Timestamp, end_time: Timestamp
    ) -> dict: ...

    @abstractmethod
    async def get_live_data(self, symbol: Symbol) -> dict: ...

    @abstractmethod
    async def get_account_info(self) -> dict: ...

    @abstractmethod
    async def get_positions(self) -> list[PositionInfo]: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

@runtime_checkable
class TradingMLPredictor(Protocol):
    """Protocol for ML trading predictors."""

    @abstractmethod
    async def predict_market_direction(
        self, input_data: ModelInput
    ) -> PredictionResult: ...

    @abstractmethod
    async def classify_regime(self, input_data: ModelInput) -> RegimeClassification: ...

    @abstractmethod
    async def generate_signals(self, input_data: ModelInput) -> list[TradingSignal]: ...

    @abstractmethod
    def get_model_confidence(self) -> float: ...

    @abstractmethod
    def is_model_ready(self) -> bool: ...

@runtime_checkable
class TradingRiskManager(Protocol):
    """Protocol for trading risk management."""

    @abstractmethod
    async def validate_trade(self, trade_decision: TradeDecision) -> bool: ...

    @abstractmethod
    async def calculate_position_size(
        self,
        symbol: Symbol,
        account_info: dict,
        risk_parameters: RiskParameters,
    ) -> float: ...

    @abstractmethod
    async def assess_portfolio_risk(
        self, positions: list[PositionInfo]
    ) -> dict[str, float]: ...

    @abstractmethod
    async def get_stop_loss_price(
        self, symbol: Symbol, entry_price: float, position_side: str
    ) -> float: ...
