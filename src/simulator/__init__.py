"""
Paper Trading Simulator Package

Provides simulated trading with real order book data, configurable fees,
and comprehensive position management for testing trading strategies.
"""

from typing import Optional
from src.utils.tprint import tprint
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
    tprint(f"[SIMULATOR] register_simulator: simulator_id={simulator_id}")
    _simulator_registry[simulator_id] = simulator
    tprint(f"[SIMULATOR] register_simulator -> registered ({len(_simulator_registry)} total simulators)")


def get_simulator(simulator_id: Optional[str] = None) -> Optional['PaperTradingSimulator']:
    """Get a registered simulator instance. If no ID provided, returns the most recent one."""
    tprint(f"[SIMULATOR] get_simulator: simulator_id={simulator_id}, registry_size={len(_simulator_registry)}")
    if simulator_id is None:
        # Return the most recently registered simulator
        if _simulator_registry:
            sim = list(_simulator_registry.values())[-1]
            tprint(f"[SIMULATOR] get_simulator -> returning most recent simulator")
            return sim
        tprint(f"[SIMULATOR] get_simulator -> no simulators registered")
        return None
    sim = _simulator_registry.get(simulator_id)
    tprint(f"[SIMULATOR] get_simulator -> {'found' if sim else 'not found'} simulator {simulator_id}")
    return sim


def get_simulator_balance(simulator_id: Optional[str] = None) -> float:
    """Get the current balance from a registered simulator."""
    tprint(f"[SIMULATOR] get_simulator_balance: simulator_id={simulator_id}")
    simulator = get_simulator(simulator_id)
    if simulator:
        balance = simulator.current_balance
        tprint(f"[SIMULATOR] get_simulator_balance -> balance={balance:.2f}")
        return balance
    tprint(f"[SIMULATOR] get_simulator_balance -> no simulator found, returning 0.0")
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
