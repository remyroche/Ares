# src/interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class MarketData:
    """Standardized market data structure"""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    interval: str


@dataclass
class AnalysisResult:
    """Standardized analysis result structure"""

    timestamp: datetime
    symbol: str
    confidence: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    features: dict[str , float]
    technical_indicators: dict[str , float]
    market_regime: str
    support_resistance: dict[str , float]
    risk_metrics: dict[str , float]


@dataclass
class StrategyResult:
    """Standardized strategy result structure"""

    timestamp: datetime
    symbol: str
    position_bias: str  # 'LONG', 'SHORT', 'NEUTRAL'
    leverage_cap: float
    max_notional_size: float
    risk_parameters: dict[str , float]
    market_conditions: dict[str , Any]


@dataclass
class TradeDecision:
    """Standardized trade decision structure"""

    timestamp: datetime
    symbol: str
    action: str  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE_LONG', 'CLOSE_SHORT'
    quantity: float
    price: float
    leverage: float
    stop_loss: float
    take_profit: float
    confidence: float
    risk_score: float


class IExchangeClient(ABC):
    """Interface for exchange client implementations"""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

class IStateManager(ABC):
    """Interface for state management"""

    @abstractmethod
    @abstractmethod
    @abstractmethod

class IPerformanceReporter(ABC):
    """Interface for performance reporting"""

    @abstractmethod
    @abstractmethod
    @abstractmethod

class IEventBus(ABC):
    """Interface for event bus"""

    @abstractmethod
    @abstractmethod
    @abstractmethod

class IAnalyst(ABC):
    """Interface for market analysis components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the analyst"""

    @abstractmethod
    @abstractmethod
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult:
        """Analyze market data and return analysis result"""

    @abstractmethod
    @abstractmethod
    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train analysis models"""

    @abstractmethod

class IStrategist(ABC):
    """Interface for strategy formulation components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the strategist"""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

class ITactician(ABC):
    """Interface for trade execution components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the tactician"""

    @abstractmethod
    @abstractmethod
    async def execute_trade_decision(
        self, strategy_result: StrategyResult,
        analysis_result: AnalysisResult) -> TradeDecision | None:
        """Execute trade decision based on strategy and analysis"""

    @abstractmethod
    @abstractmethod

class ISupervisor(ABC):
    """Interface for supervision and coordination components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the supervisor"""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod

class IModelManager(ABC):
    """Interface for model management"""

    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod
    @abstractmethod