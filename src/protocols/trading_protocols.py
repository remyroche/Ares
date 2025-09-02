# src/protocols/trading_protocols.py

"""
Enhanced trading system protocols with comprehensive type safety.
"""

from abc import abstractmethod
from typing import Protocol, runtime_checkable, Any, Dict, List, Union, Optional
from datetime import datetime
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum


# Type aliases using basic Python types
Symbol = str
Timestamp = datetime
Price = Union[float, Decimal]
Volume = Union[float, Decimal]


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class PositionSide(str, Enum):
    """Position side enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"


class SignalType(str, Enum):
    """Trading signal type enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class RegimeType(str, Enum):
    """Market regime type enumeration."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SIDEWAYS = "SIDEWAYS"
    VOLATILE = "VOLATILE"
    LOW_VOLATILITY = "LOW_VOLATILITY"


# Enhanced type definitions for the protocols
@dataclass
class ModelInput:
    """Model input data structure."""
    features: Dict[str, float]
    timestamp: Timestamp
    symbol: Symbol
    feature_version: str = "1.0"
    
    def validate(self) -> bool:
        """Validate input data."""
        return (
            isinstance(self.features, dict) and
            isinstance(self.timestamp, datetime) and
            isinstance(self.symbol, str) and
            len(self.features) > 0
        )


@dataclass
class PredictionResult:
    """Model prediction result."""
    prediction: float  # -1.0 to 1.0 (sell to buy)
    confidence: float  # 0.0 to 1.0
    timestamp: Timestamp
    model_id: str
    features_used: List[str]
    
    def is_buy_signal(self, threshold: float = 0.1) -> bool:
        """Check if prediction indicates buy signal."""
        return self.prediction > threshold and self.confidence > 0.7
    
    def is_sell_signal(self, threshold: float = -0.1) -> bool:
        """Check if prediction indicates sell signal."""
        return self.prediction < threshold and self.confidence > 0.7


@dataclass
class OrderRequest:
    """Trading order request."""
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Volume
    price: Optional[Price] = None
    stop_price: Optional[Price] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    
    def validate(self) -> bool:
        """Validate order request."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            return False
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            return False
        return self.quantity > 0


@dataclass
class PerformanceMetrics:
    """Trading performance metrics."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    @property
    def average_win(self) -> float:
        """Calculate average win amount."""
        return self.total_return / self.winning_trades if self.winning_trades > 0 else 0.0
    
    @property
    def average_loss(self) -> float:
        """Calculate average loss amount."""
        return self.total_return / self.losing_trades if self.losing_trades > 0 else 0.0


@dataclass
class PositionInfo:
    """Current position information."""
    symbol: Symbol
    side: PositionSide
    quantity: Volume
    entry_price: Price
    current_price: Price
    unrealized_pnl: float
    realized_pnl: float
    timestamp: Timestamp
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L."""
        return self.unrealized_pnl + self.realized_pnl
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100


@dataclass
class RegimeClassification:
    """Market regime classification."""
    regime_type: RegimeType
    confidence: float  # 0.0 to 1.0
    timestamp: Timestamp
    features: Dict[str, float]
    description: str
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if classification has high confidence."""
        return self.confidence > threshold


@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_size: float  # Maximum position size as % of portfolio
    max_drawdown: float  # Maximum allowed drawdown
    stop_loss_pct: float  # Stop loss percentage
    take_profit_pct: float  # Take profit percentage
    max_leverage: float  # Maximum leverage allowed
    correlation_threshold: float  # Maximum correlation between positions
    
    def validate(self) -> bool:
        """Validate risk parameters."""
        return (
            0 < self.max_position_size <= 1.0 and
            0 < self.max_drawdown <= 1.0 and
            0 < self.stop_loss_pct <= 1.0 and
            0 < self.take_profit_pct <= 1.0 and
            self.max_leverage > 0 and
            0 <= self.correlation_threshold <= 1.0
        )


@dataclass
class TradeDecision:
    """Trading decision with rationale."""
    symbol: Symbol
    action: SignalType
    quantity: Volume
    price: Price
    confidence: float
    rationale: str
    risk_score: float
    timestamp: Timestamp
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if decision has high confidence."""
        return self.confidence > threshold
    
    def is_high_risk(self, threshold: float = 0.7) -> bool:
        """Check if decision has high risk."""
        return self.risk_score > threshold


@dataclass
class TradingSignal:
    """Trading signal with metadata."""
    signal_type: SignalType
    symbol: Symbol
    strength: float  # 0.0 to 1.0
    timestamp: Timestamp
    price_target: Optional[Price] = None
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def is_strong_signal(self, threshold: float = 0.7) -> bool:
        """Check if signal is strong."""
        return self.strength > threshold
    
    def has_risk_management(self) -> bool:
        """Check if signal includes risk management."""
        return self.stop_loss is not None or self.take_profit is not None


@runtime_checkable
class TradingDataProvider(Protocol):
    """Protocol for trading data providers."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the trading data provider."""
        ...
    
    @abstractmethod
    async def get_market_data(self, symbol: Symbol, start_time: Timestamp, end_time: Timestamp) -> Dict[str, Any]:
        """Get historical market data for a symbol."""
        ...
    
    @abstractmethod
    async def get_live_data(self, symbol: Symbol) -> Dict[str, Any]:
        """Get live/real-time market data for a symbol."""
        ...
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        ...
    
    @abstractmethod
    async def get_positions(self) -> List[PositionInfo]:
        """Get current open positions."""
        ...
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the data provider is connected."""
        ...


@runtime_checkable
class TradingMLPredictor(Protocol):
    """Protocol for ML-based trading predictors."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the ML predictor."""
        ...
    
    @abstractmethod
    async def predict_market_direction(self, input_data: ModelInput) -> PredictionResult:
        """Predict market direction based on input data."""
        ...
    
    @abstractmethod
    async def classify_regime(self, input_data: ModelInput) -> RegimeClassification:
        """Classify current market regime."""
        ...
    
    @abstractmethod
    async def generate_signals(self, input_data: ModelInput) -> List[TradingSignal]:
        """Generate trading signals based on input data."""
        ...
    
    @abstractmethod
    def get_model_confidence(self) -> float:
        """Get current model confidence score."""
        ...
    
    @abstractmethod
    def is_model_ready(self) -> bool:
        """Check if the model is ready for predictions."""
        ...


@runtime_checkable
class TradingRiskManager(Protocol):
    """Protocol for trading risk management."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the risk manager."""
        ...
    
    @abstractmethod
    async def validate_trade(self, trade_decision: TradeDecision) -> bool:
        """Validate if a trade decision meets risk requirements."""
        ...
    
    @abstractmethod
    async def calculate_position_size(
        self, 
        symbol: Symbol, 
        account_info: Dict[str, Any], 
        risk_parameters: RiskParameters
    ) -> float:
        """Calculate appropriate position size based on risk parameters."""
        ...
    
    @abstractmethod
    async def assess_portfolio_risk(self, positions: List[PositionInfo]) -> Dict[str, float]:
        """Assess overall portfolio risk."""
        ...
    
    @abstractmethod
    async def get_stop_loss_price(
        self, 
        symbol: Symbol, 
        entry_price: float, 
        position_side: str
    ) -> float:
        """Calculate stop loss price for a position."""
        ...
