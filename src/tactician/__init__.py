# src/tactician/__init__.py

# Import the main components for easier access
from .tactician import Tactician, setup_tactician
from .position_sizer import PositionSizer
from .leverage_sizer import LeverageSizer

__all__ = [
    "Tactician",
    "setup_tactician", 
    "PositionSizer",
    "LeverageSizer",
]
