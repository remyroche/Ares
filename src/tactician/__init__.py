from __future__ import annotations
# src/tactician/__init__.py

# Import the main components for easier access
from .leverage_sizer import LeverageSizer
from .ml_tactics_manager import MLTacticsManager
from .position_sizer import PositionSizer
from .sr_breakout_predictor import SRBreakoutPredictor
from .tactician import Tactician, setup_tactician

__all__ = [
    "Tactician",
    "setup_tactician",
    "PositionSizer",
    "LeverageSizer",
    "SRBreakoutPredictor",
    "MLTacticsManager",
]
