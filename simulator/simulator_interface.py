"""
Simulator Interface

Defines the interface for paper trading simulation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SimulatorMode(Enum):
    """Simulator operation modes"""
    PAPER = "paper"
    TRADE = "trade"


@dataclass
class SimulatedOrder:
    """Simulated order data structure"""
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    commission: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SimulatedPosition:
    """Simulated position data structure"""
    symbol: str
    side: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime
    updated_at: datetime


class ISimulator(ABC):
    """Interface for trading simulators"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the simulator"""

    @abstractmethod
    async def close(self) -> None:
        """Close the simulator"""

    @abstractmethod
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a simulated order"""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order"""

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order status"""

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open simulated orders"""

    @abstractmethod
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get simulated positions"""

    @abstractmethod
    async def get_portfolio_value(self) -> Dict[str, Any]:
        """Get simulated portfolio value"""

    @abstractmethod
    async def update_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """Update market data for simulation"""

    @abstractmethod
    async def process_order_book(self, symbol: str, order_book: Dict[str, Any]) -> None:
        """Process order book data for accurate simulation"""