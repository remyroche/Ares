"""
Simulator Package

This package provides paper trading simulation capabilities for the trading system.
When in PAPER mode, orders are routed to the simulator instead of real exchanges.
"""

from .simulator_interface import SimulatorInterface
from .order_simulator import OrderSimulator
from .portfolio_simulator import PortfolioSimulator

__all__ = [
    'SimulatorInterface',
    'OrderSimulator', 
    'PortfolioSimulator'
]