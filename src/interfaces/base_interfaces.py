# src/interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


class IEventBus(ABC):
    """Interface for event bus system."""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Any) -> bool:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, callback: callable) -> bool:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, callback: callable) -> bool:
        """Unsubscribe from an event type."""
        pass


class IAnalyst(ABC):
    """Interface for market analysis."""
    
    @abstractmethod
    async def analyze(self, data: Any) -> Dict[str, Any]:
        """Analyze market data."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the analyst."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the analyst."""
        pass


class IStrategist(ABC):
    """Interface for trading strategy."""
    
    @abstractmethod
    async def generate_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading strategy based on analysis."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the strategist."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the strategist."""
        pass


class ITactician(ABC):
    """Interface for trade execution tactics."""
    
    @abstractmethod
    async def execute_trade(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on strategy."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the tactician."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the tactician."""
        pass


class ISupervisor(ABC):
    """Interface for system supervision."""
    
    @abstractmethod
    async def monitor_system(self) -> Dict[str, Any]:
        """Monitor system health and performance."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the supervisor."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the supervisor."""
        pass


class IExchangeClient(ABC):
    """Interface for exchange communication."""
    
    @abstractmethod
    async def get_market_data(self, symbol: str, timeframe: str) -> Any:
        """Get market data from exchange."""
        pass
    
    @abstractmethod
    async def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order on the exchange."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the exchange client."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the exchange client."""
        pass


class IStateManager(ABC):
    """Interface for state management."""
    
    @abstractmethod
    async def save_state(self, key: str, data: Any) -> bool:
        """Save state data."""
        pass
    
    @abstractmethod
    async def load_state(self, key: str) -> Any:
        """Load state data."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the state manager."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the state manager."""
        pass


class IPerformanceReporter(ABC):
    """Interface for performance reporting."""
    
    @abstractmethod
    async def report_performance(self, metrics: Dict[str, Any]) -> bool:
        """Report performance metrics."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the performance reporter."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the performance reporter."""
        pass


class IModelManager(ABC):
    """Interface for model management."""
    
    @abstractmethod
    async def get_analyst(self) -> IAnalyst:
        """Get the analyst component."""
        pass
    
    @abstractmethod
    async def get_strategist(self) -> IStrategist:
        """Get the strategist component."""
        pass
    
    @abstractmethod
    async def get_tactician(self) -> ITactician:
        """Get the tactician component."""
        pass
    
    @abstractmethod
    async def load_models(self) -> bool:
        """Load models."""
        pass
    
    @abstractmethod
    async def promote_challenger_to_champion(self) -> bool:
        """Promote challenger model to champion."""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model manager."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the model manager."""
        pass


# Data classes for common structures
@dataclass
class MarketData:
    """Market data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        """Post-initialization setup."""
        pass


@dataclass
class AnalysisResult:
    """Analysis result structure."""
    timestamp: datetime
    symbol: str
    indicators: Dict[str, float]
    signals: Dict[str, str]
    
    def __post_init__(self):
        """Post-initialization setup."""
        pass


@dataclass
class StrategyResult:
    """Strategy result structure."""
    timestamp: datetime
    symbol: str
    action: str
    confidence: float
    parameters: Dict[str, Any]
    
    def __post_init__(self):
        """Post-initialization setup."""
        pass


@dataclass
class TradeDecision:
    """Trade decision structure."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        pass