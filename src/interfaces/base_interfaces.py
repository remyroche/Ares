# src/interfaces/base_interfaces.py

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from src.utils.logger import system_logger


@dataclass
class MarketData:
    """Market data structure for trading operations."""
    
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    
    def __post_init__(self):
        """Validate market data after initialization."""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.timestamp > datetime.now():
            raise ValueError("Timestamp cannot be in the future")


@dataclass
class AnalysisResult:
    """Result of market analysis operations."""
    
    symbol: str
    analysis_type: str
    result: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate analysis result after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.timestamp > datetime.now():
            raise ValueError("Timestamp cannot be in the future")


@dataclass
class StrategyResult:
    """Result of strategy execution."""
    
    strategy_name: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    timestamp: datetime
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate strategy result after initialization."""
        if self.action not in ['buy', 'sell', 'hold']:
            raise ValueError("Action must be 'buy', 'sell', or 'hold'")
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")


@dataclass
class TradeDecision:
    """Trading decision structure."""
    
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    quantity: float
    price: float
    timestamp: datetime
    strategy: str
    confidence: float
    risk_level: str = "medium"
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate trade decision after initialization."""
        if self.action not in ['buy', 'sell', 'hold']:
            raise ValueError("Action must be 'buy', 'sell', or 'hold'")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.risk_level not in ['low', 'medium', 'high']:
            raise ValueError("Risk level must be 'low', 'medium', or 'high'")


class IAnalyst(ABC):
    """Interface for market analysis components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IAnalyst."""
        self.config = config or {}
        self.logger = system_logger.getChild("IAnalyst")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the analyst component."""
        pass
    
    @abstractmethod
    async def analyze_market_data(self, market_data: MarketData) -> AnalysisResult:
        """Analyze market data and return analysis result."""
        pass
    
    @abstractmethod
    async def get_analysis_history(self, symbol: str, limit: int = 100) -> List[AnalysisResult]:
        """Get analysis history for a specific symbol."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IEventBus(ABC):
    """Interface for event bus components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IEventBus."""
        self.config = config or {}
        self.logger = system_logger.getChild("IEventBus")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the event bus."""
        pass
    
    @abstractmethod
    async def publish_event(self, event_type: str, data: Any, source: str) -> bool:
        """Publish an event to the bus."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, callback: callable) -> bool:
        """Subscribe to events of a specific type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, callback: callable) -> bool:
        """Unsubscribe from events of a specific type."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IExchangeClient(ABC):
    """Interface for exchange client components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IExchangeClient."""
        self.config = config or {}
        self.logger = system_logger.getChild("IExchangeClient")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the exchange client."""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol."""
        pass
    
    @abstractmethod
    async def place_order(self, order: TradeDecision) -> bool:
        """Place a trading order."""
        pass
    
    @abstractmethod
    async def get_balance(self, asset: str) -> float:
        """Get current balance for an asset."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IModelManager(ABC):
    """Interface for model management components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IModelManager."""
        self.config = config or {}
        self.logger = system_logger.getChild("IModelManager")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the model manager."""
        pass
    
    @abstractmethod
    async def load_model(self, model_name: str, model_path: str) -> bool:
        """Load a model from the specified path."""
        pass
    
    @abstractmethod
    async def predict(self, model_name: str, input_data: Any) -> Any:
        """Make a prediction using the specified model."""
        pass
    
    @abstractmethod
    async def update_model(self, model_name: str, new_model_path: str) -> bool:
        """Update an existing model."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IPerformanceReporter(ABC):
    """Interface for performance reporting components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IPerformanceReporter."""
        self.config = config or {}
        self.logger = system_logger.getChild("IPerformanceReporter")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the performance reporter."""
        pass
    
    @abstractmethod
    async def record_trade(self, trade: TradeDecision, result: Dict[str, Any]) -> bool:
        """Record a completed trade."""
        pass
    
    @abstractmethod
    async def generate_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate a performance report for the specified period."""
        pass
    
    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IStateManager(ABC):
    """Interface for state management components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IStateManager."""
        self.config = config or {}
        self.logger = system_logger.getChild("IStateManager")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the state manager."""
        pass
    
    @abstractmethod
    async def save_state(self, key: str, value: Any) -> bool:
        """Save a state value."""
        pass
    
    @abstractmethod
    async def load_state(self, key: str) -> Any:
        """Load a state value."""
        pass
    
    @abstractmethod
    async def delete_state(self, key: str) -> bool:
        """Delete a state value."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class IStrategist(ABC):
    """Interface for strategy components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize IStrategist."""
        self.config = config or {}
        self.logger = system_logger.getChild("IStrategist")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the strategist."""
        pass
    
    @abstractmethod
    async def formulate_strategy(self, market_data: MarketData, analysis: AnalysisResult) -> StrategyResult:
        """Formulate a trading strategy based on market data and analysis."""
        pass
    
    @abstractmethod
    async def evaluate_strategy(self, strategy: StrategyResult) -> float:
        """Evaluate the effectiveness of a strategy."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class ISupervisor(ABC):
    """Interface for supervision components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ISupervisor."""
        self.config = config or {}
        self.logger = system_logger.getChild("ISupervisor")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the supervisor."""
        pass
    
    @abstractmethod
    async def start_component(self, component_name: str) -> bool:
        """Start a component."""
        pass
    
    @abstractmethod
    async def stop_component(self, component_name: str) -> bool:
        """Stop a component."""
        pass
    
    @abstractmethod
    async def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """Get the status of a component."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass


class ITactician(ABC):
    """Interface for tactical execution components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ITactician."""
        self.config = config or {}
        self.logger = system_logger.getChild("ITactician")
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the tactician."""
        pass
    
    @abstractmethod
    async def execute_strategy(self, strategy: StrategyResult) -> TradeDecision:
        """Execute a trading strategy."""
        pass
    
    @abstractmethod
    async def adjust_execution(self, decision: TradeDecision, market_conditions: Dict[str, Any]) -> TradeDecision:
        """Adjust execution based on market conditions."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources."""
        pass