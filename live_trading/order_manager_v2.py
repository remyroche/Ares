"""Quarantined legacy order manager.

The previous implementation generated STOP_LOSS/OCO prices from ATR/defaults and
legacy trailing/giveback fields. Runtime stop-loss decisions are now allowed only
through ``extreme_price_movements.inference.simple_policy_stop`` using validated
``simple_policy_optimiser`` artifacts. This module remains importable only so
older imports fail with a clear migration error instead of silently placing
legacy orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


MIGRATION_ERROR = (
    "OrderManagerV2 is quarantined: legacy OCO/SL/TP logic is disabled. "
    "Use extreme_price_movements.inference.simple_policy_stop.py with "
    "TradeExecutor; executors may only place exchange orders from validated "
    "SimplePolicyStopDecision objects produced from simple_policy_optimiser "
    "artifacts."
)


class OrderStatus(Enum):
    """Order status enumeration retained for import compatibility."""

    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(Enum):
    """Position status enumeration retained for import compatibility."""

    PENDING_ENTRY = "pending_entry"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    EMERGENCY_CLOSE = "emergency_close"


@dataclass
class Order:
    """Passive order record retained for import compatibility only."""

    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_price: Optional[float] = None
    exchange_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Passive position record retained for import compatibility only."""

    position_id: str
    symbol: str
    side: str
    strategy_id: str
    entry_price: float
    size: float
    quantity: float
    status: PositionStatus = PositionStatus.PENDING_ENTRY
    entry_order: Optional[Order] = None
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrderManagerV2:
    """Disabled legacy order manager that fails before any order side effect."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(MIGRATION_ERROR)

    async def start(self) -> None:
        raise RuntimeError(MIGRATION_ERROR)

    async def stop(self) -> None:
        raise RuntimeError(MIGRATION_ERROR)

    async def place_oco_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError(MIGRATION_ERROR)

    async def emergency_close_all(self) -> None:
        raise RuntimeError(MIGRATION_ERROR)

    def get_position(self, symbol: str) -> Optional[Position]:
        raise RuntimeError(MIGRATION_ERROR)

    def get_all_positions(self) -> Dict[str, Position]:
        raise RuntimeError(MIGRATION_ERROR)

    def get_portfolio_state(self) -> Dict[str, Any]:
        raise RuntimeError(MIGRATION_ERROR)

    def register_position_open_callback(self, callback: Callable[[Position], None]) -> None:
        raise RuntimeError(MIGRATION_ERROR)

    def register_position_close_callback(self, callback: Callable[[Position], None]) -> None:
        raise RuntimeError(MIGRATION_ERROR)
