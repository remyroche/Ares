# src/types/trading_types.py

"""Trading-specific type definitions for orders, positions, and trade decisions."""

from typing import Literal, TypedDict, Any, Dict, List, Optional
from datetime import datetime

# Trading action enums
OrderType = Literal["market", "limit", "stop", "stop_limit", "trailing_stop"]
OrderSide = Literal["buy", "sell"]
OrderStatus = Literal[
    "pending",
    "open",
    "filled",
    "partially_filled",
    "cancelled",
    "rejected",
    "expired",
]
PositionSide = Literal["long", "short", "neutral"]
TradeAction = Literal["open_long", "open_short", "close_long", "close_short", "hold"]
RiskLevel = Literal["very_low", "low", "medium", "high", "very_high"]


class OrderRequest(TypedDict):
    """Order request data structure."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float]  # Required for limit orders
    stop_price: Optional[float]  # Required for stop orders
    time_in_force: Optional[str]
    client_order_id: Optional[str]


class PositionInfo(TypedDict):
    """Position information data structure."""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    leverage: float
    margin_used: float
    timestamp: datetime


class TradeDecision(TypedDict):
    """Trade decision data structure."""
    action: TradeAction
    symbol: str
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_score: float
    confidence: float
    reasoning: List[str]
    timestamp: datetime


class TradingSignal(TypedDict):
    """Trading signal data structure."""
    symbol: str
    signal: TradeAction
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    price_target: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reasoning: List[str]
    timestamp: datetime
    model_source: str
    model_version: str


class PositionRisk(TypedDict):
    """Position risk metrics."""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation: float


class RiskParameters(TypedDict):
    """Risk management parameters."""
    max_position_size: float
    max_leverage: float
    max_drawdown: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    position_sizing_method: str
    risk_per_trade: float


class MarketData(TypedDict):
    """Market data structure."""
    symbol: str
    price: float
    volume: float
    timestamp: datetime
    bid: Optional[float]
    ask: Optional[float]
    spread: Optional[float]
    high_24h: Optional[float]
    low_24h: Optional[float]
    change_24h: Optional[float]
    change_pct_24h: Optional[float]


class OrderExecution(TypedDict):
    """Order execution details."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    executed_quantity: float
    price: float
    commission: float
    timestamp: datetime
    status: OrderStatus
    fills: List[Dict[str, Any]]


class PortfolioSummary(TypedDict):
    """Portfolio summary information."""
    total_value: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    margin_used: float
    margin_available: float
    leverage: float
    risk_metrics: PositionRisk
    timestamp: datetime


class TradeHistory(TypedDict):
    """Trade history entry."""
    trade_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    order_id: str
    execution_quality: Optional[str]
    slippage: Optional[float]
    market_impact: Optional[float]


class RiskMetrics(TypedDict):
    """Comprehensive risk metrics."""
    portfolio_var: float
    position_var: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    beta_exposure: Dict[str, float]
    sector_exposure: Dict[str, float]
    currency_exposure: Dict[str, float]
    liquidity_metrics: Dict[str, float]
    stress_test_results: Dict[str, float]
    scenario_analysis: Dict[str, float]
