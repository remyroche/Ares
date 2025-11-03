"""
Paper Trading Simulator Package

Provides simulated trading with real order book data, configurable fees,
and comprehensive position management for testing trading strategies.
"""

from typing import Optional
from .config import SimulatorConfig
from .fee_calculator import FeeCalculator
from .slippage_calculator import SlippageCalculator
from .order_validator import OrderValidator
from .position_manager import PositionManager
from .persistence import SimulatorPersistence
from .paper_trading_simulator import PaperTradingSimulator

# Global simulator registry for balance tracking
_simulator_registry = {}


def register_simulator(simulator_id: str, simulator: 'PaperTradingSimulator') -> None:
    """Register a simulator instance for global access."""
    _simulator_registry[simulator_id] = simulator


def get_simulator(simulator_id: Optional[str] = None) -> Optional['PaperTradingSimulator']:
    """Get a registered simulator instance. If no ID provided, returns the most recent one."""
    if simulator_id is None:
        # Return the most recently registered simulator
        if _simulator_registry:
            return list(_simulator_registry.values())[-1]
        return None
    return _simulator_registry.get(simulator_id)


def get_simulator_balance(simulator_id: Optional[str] = None) -> float:
    """Get the current balance from a registered simulator."""
    simulator = get_simulator(simulator_id)
    if simulator:
        return simulator.current_balance
    return 0.0


__all__ = [
    "SimulatorConfig",
    "FeeCalculator",
    "SlippageCalculator",
    "OrderValidator",
    "PositionManager",
    "SimulatorPersistence",
    "PaperTradingSimulator",
    "register_simulator",
    "get_simulator",
    "get_simulator_balance",
]
