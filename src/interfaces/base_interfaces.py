# src/interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
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
    features: Dict[str, float]
    technical_indicators: Dict[str, float]
    market_regime: str
    support_resistance: Dict[str, float]
    risk_metrics: Dict[str, float]


@dataclass
class StrategyResult:
    """Standardized strategy result structure"""

    timestamp: datetime
    symbol: str
    position_bias: str  # 'LONG', 'SHORT', 'NEUTRAL'
    leverage_cap: float
    max_notional_size: float
    risk_parameters: Dict[str, float]
    market_conditions: Dict[str, Any]


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
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[MarketData]:
        """Get historical kline data"""
        pass

    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        order_type: str = "MARKET",
    ) -> Dict[str, Any]:
        """Create a trading order"""
        pass

    @abstractmethod
    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Get position risk information"""
        pass


class IStateManager(ABC):
    """Interface for state management"""

    @abstractmethod
    def get_state(self, key: str) -> Any:
        """Get state value"""
        pass

    @abstractmethod
    def set_state(self, key: str, value: Any) -> None:
        """Set state value"""
        pass

    @abstractmethod
    def get_state_if_not_exists(self, key: str, default_value: Any) -> Any:
        """Get state value or set default if not exists"""
        pass


class IPerformanceReporter(ABC):
    """Interface for performance reporting"""

    @abstractmethod
    async def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """Log a trade"""
        pass

    @abstractmethod
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        pass

    @abstractmethod
    async def generate_report(self) -> str:
        """Generate performance report"""
        pass


class IEventBus(ABC):
    """Interface for event bus"""

    @abstractmethod
    async def publish(self, event_type: str, data: Any) -> None:
        """Publish an event"""
        pass

    @abstractmethod
    async def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to an event type"""
        pass

    @abstractmethod
    async def unsubscribe(self, event_type: str, callback) -> None:
        """Unsubscribe from an event type"""
        pass


class IAnalyst(ABC):
    """Interface for market analysis components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the analyst"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the analyst"""
        pass

    @abstractmethod
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult:
        """Analyze market data and return analysis result"""
        pass

    @abstractmethod
    async def get_historical_analysis(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> List[AnalysisResult]:
        """Get historical analysis results"""
        pass

    @abstractmethod
    async def train_models(self, training_data: pd.DataFrame) -> bool:
        """Train analysis models"""
        pass

    @abstractmethod
    async def load_models(self, model_path: str) -> bool:
        """Load trained models"""
        pass


class IStrategist(ABC):
    """Interface for strategy formulation components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the strategist"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the strategist"""
        pass

    @abstractmethod
    async def formulate_strategy(
        self, analysis_result: AnalysisResult
    ) -> StrategyResult:
        """Formulate trading strategy based on analysis"""
        pass

    @abstractmethod
    async def update_strategy_parameters(self, parameters: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        pass

    @abstractmethod
    async def get_strategy_performance(self) -> Dict[str, Any]:
        """Get strategy performance metrics"""
        pass


class ITactician(ABC):
    """Interface for trade execution components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the tactician"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the tactician"""
        pass

    @abstractmethod
    async def execute_trade_decision(
        self, strategy_result: StrategyResult, analysis_result: AnalysisResult
    ) -> Optional[TradeDecision]:
        """Execute trade decision based on strategy and analysis"""
        pass

    @abstractmethod
    async def calculate_position_size(
        self, strategy_result: StrategyResult, account_balance: float
    ) -> float:
        """Calculate position size"""
        pass

    @abstractmethod
    async def calculate_risk_parameters(
        self, strategy_result: StrategyResult, market_data: MarketData
    ) -> Dict[str, float]:
        """Calculate risk parameters"""
        pass


class ISupervisor(ABC):
    """Interface for supervision and coordination components"""

    @abstractmethod
    async def start(self) -> None:
        """Start the supervisor"""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the supervisor"""
        pass

    @abstractmethod
    async def monitor_performance(self) -> Dict[str, Any]:
        """Monitor system performance"""
        pass

    @abstractmethod
    async def manage_risk(self) -> Dict[str, Any]:
        """Manage risk across all components"""
        pass

    @abstractmethod
    async def coordinate_components(self) -> None:
        """Coordinate all trading components"""
        pass


class IModelManager(ABC):
    """Interface for model management"""

    @abstractmethod
    def get_analyst(self) -> IAnalyst:
        """Get analyst instance"""
        pass

    @abstractmethod
    def get_strategist(self) -> IStrategist:
        """Get strategist instance"""
        pass

    @abstractmethod
    def get_tactician(self) -> ITactician:
        """Get tactician instance"""
        pass

    @abstractmethod
    async def load_models(self, model_version: str) -> bool:
        """Load specific model version"""
        pass

    @abstractmethod
    async def promote_challenger_to_champion(self) -> bool:
        """Promote challenger model to champion"""
        pass
