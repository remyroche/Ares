# src/types/trading_types.py

"""
Trading-specific type definitions for orders, positions, and trade decisions.
"""

from typing import Literal, TypedDict

from .base_types import (
    ConfidenceLevel,
    LeverageMultiplier,
    Percentage,
    PositionId,
    Price,
    RiskScore,
    Score,
    Symbol,
    Timestamp,
    Volume,
)

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
    """Type-safe order request."""

    symbol: Symbol
    side: OrderSide
    type: OrderType
    quantity: Volume
    price: Price | None
    stop_price: Price | None
    time_in_force: Literal["GTC", "IOC", "FOK"] | None
    reduce_only: bool | None
    leverage: LeverageMultiplier | None


class TradeDecision(TypedDict):
    """Type-safe trade decision."""

    timestamp: Timestamp
    symbol: Symbol
    action: TradeAction
    quantity: Volume
    price: Price | None
    leverage: LeverageMultiplier | None
    stop_loss: Price | None
    take_profit: Price | None
    confidence: ConfidenceLevel
    risk_score: RiskScore
    reasoning: str


class PositionRisk(TypedDict):
    """Type-safe position risk assessment."""

    position_id: PositionId
    symbol: Symbol
    current_risk: RiskScore
    max_loss_usd: float
    liquidation_price: Price | None
    margin_ratio: float
    unrealized_pnl_percentage: Percentage
    days_held: int
    risk_level: RiskLevel


class RiskParameters(TypedDict):
    """Type-safe risk management parameters."""

    max_position_size: Volume
    max_leverage: LeverageMultiplier
    stop_loss_percentage: Percentage
    take_profit_percentage: Percentage
    max_drawdown: Percentage
    max_daily_loss: float
    position_correlation_limit: float
    var_limit: float  # Value at Risk


class TradingSignal(TypedDict):
    """Type-safe trading signal."""

    timestamp: Timestamp
    symbol: Symbol
    signal_type: Literal["entry", "exit", "hold"]
    direction: PositionSide | None
    strength: Score  # 0.0 to 1.0
    confidence: ConfidenceLevel
    time_horizon: Literal["scalp", "short_term", "medium_term", "long_term"]
    source: str  # e.g., "analyst", "ml_model", "technical_indicator"


class PerformanceMetrics(TypedDict):
    """Type-safe performance metrics."""

    total_return: Percentage
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: Percentage
    win_rate: Percentage
    profit_factor: float
    average_win: float
    average_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int


class PortfolioState(TypedDict):
    """Type-safe portfolio state."""

    timestamp: Timestamp
    total_value: float
    available_balance: float
    unrealized_pnl: float
    margin_used: float
    positions: list[PositionRisk]
    open_orders: list[OrderRequest]
    daily_pnl: float
    performance: PerformanceMetrics


class BacktestResult(TypedDict):
    """Type-safe backtest result."""

    start_date: Timestamp
    end_date: Timestamp
    initial_capital: float
    final_value: float
    performance: PerformanceMetrics
    trades: list[TradeDecision]
    daily_returns: list[float]
    drawdown_periods: list[dict[str, Timestamp]]
    config_used: dict[str, str]
