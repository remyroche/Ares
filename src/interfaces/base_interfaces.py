# src/interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

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
    features: dict[str, float]
    technical_indicators: dict[str, float]
    market_regime: str
    support_resistance: dict[str, float]
    risk_metrics: dict[str, float]


@dataclass
class StrategyResult:
    """Standardized strategy result structure"""

    timestamp: datetime
    symbol: str
    position_bias: str  # 'LONG', 'SHORT', 'NEUTRAL'
    leverage_cap: float
    max_notional_size: float
    risk_parameters: dict[str, float]
    market_conditions: dict[str, Any]


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
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
    ) -> list[MarketData]:
        """Get historical kline data"""

    @abstractmethod
    async def get_account_info(self) -> dict[str, Any]:
        """Get account information"""

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "MARKET",
    ) -> dict[str, Any]:
        """Create a trading order"""

    @abstractmethod
    async def get_position_risk(self, symbol: str) -> dict[str, Any]:
        """Get position risk information"""


class IStateManager(ABC):
    """Interface for state management"""

    @abstractmethod
    def get_state(self, key: str) -> Any:
        """Get state value"""

    @abstractmethod
    def set_state(self, key: str, value: Any) -> None:
        """Set state value"""

    @abstractmethod
    def get_state_if_not_exists(self, key: str, default_value: Any) -> Any:
        # default_value parameter used in the method implementation
        """Get state value or set default if not exists"""


class IPerformanceReporter(ABC):
    """Interface for performance reporting"""

    @abstractmethod
    async def log_trade(self, trade_data: dict[str, Any]) -> None:
        """Log a trade"""

    @abstractmethod
    async def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary"""

    @abstractmethod
    async def generate_report(self) -> str:
        """Generate performance report"""


class IEventBus(ABC):
    """Interface for event bus"""

    @abstractmethod
    async def publish(self, event_type: str, data: Any) -> None:
        """Publish an event"""

    @abstractmethod
    async def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to an event type"""

    @abstractmethod
    async def unsubscribe(self, event_type: str, callback) -> None:
        """Unsubscribe from an event type"""


class IAnalyst(ABC):
    """Interface for market analysis components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the analyst"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the analyst"""

    @abstractmethod
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult:
        """Analyze market data and return analysis result"""

    @abstractmethod
    async def get_historical_analysis(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[AnalysisResult]:
        """Get historical analysis results"""

    @abstractmethod
    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train analysis models"""

    @abstractmethod
    async def load_models(self, model_path: str) -> bool:
        """Load trained models"""


class IStrategist(ABC):
    """Interface for strategy formulation components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the strategist"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the strategist"""

    @abstractmethod
    async def formulate_strategy(
        self,
        analysis_result: AnalysisResult,
    ) -> StrategyResult:
        """Formulate trading strategy based on analysis"""

    @abstractmethod
    async def update_strategy_parameters(self, parameters: dict[str, Any]) -> None:
        """Update strategy parameters"""

    @abstractmethod
    async def get_strategy_performance(self) -> dict[str, Any]:
        """Get strategy performance metrics"""


class ITactician(ABC):
    """Interface for trade execution components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the tactician"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the tactician"""

    @abstractmethod
    async def execute_trade_decision(
        self,
        strategy_result: StrategyResult,
        analysis_result: AnalysisResult,
    ) -> TradeDecision | None:
        """Execute trade decision based on strategy and analysis"""

    @abstractmethod
    async def calculate_position_size(
        self,
        strategy_result: StrategyResult,
        account_balance: float,
    ) -> float:
        """Calculate position size"""

    @abstractmethod
    async def calculate_risk_parameters(
        self,
        strategy_result: StrategyResult,
        market_data: MarketData,
    ) -> dict[str, float]:
        """Calculate risk parameters"""


class ISupervisor(ABC):
    """Interface for supervision and coordination components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the supervisor"""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the supervisor"""

    @abstractmethod
    async def monitor_performance(self) -> dict[str, Any]:
        """Monitor system performance"""

    @abstractmethod
    async def manage_risk(self) -> dict[str, Any]:
        """Manage risk across all components"""

    @abstractmethod
    async def coordinate_components(self) -> None:
        """Coordinate all trading components"""


class IModelManager(ABC):
    """Interface for model management"""

    @abstractmethod
    def get_analyst(self) -> IAnalyst:
        """Get analyst instance"""

    @abstractmethod
    def get_strategist(self) -> IStrategist:
        """Get strategist instance"""

    @abstractmethod
    def get_tactician(self) -> ITactician:
        """Get tactician instance"""

    @abstractmethod
    async def load_models(self, model_version: str) -> bool:
        """Load specific model version"""

    @abstractmethod
    async def promote_challenger_to_champion(self) -> bool:
        """Promote challenger model to champion"""
