"""
Paper Trading Simulator Package

Provides simulated trading with real order book data, configurable fees,
and comprehensive position management for testing trading strategies.
"""

from .config import SimulatorConfig
from .fee_calculator import FeeCalculator
from .slippage_calculator import SlippageCalculator
from .order_validator import OrderValidator
from .position_manager import PositionManager
from .persistence import SimulatorPersistence
from .paper_trading_simulator import PaperTradingSimulator

__all__ = [
    "SimulatorConfig",
    "FeeCalculator",
    "SlippageCalculator",
    "OrderValidator",
    "PositionManager",
    "SimulatorPersistence",
    "PaperTradingSimulator",
]
