"""
Trading Module

This module provides a comprehensive trading system with the following components:

- regime/: ML-based detection of 15-25 market regimes with percentage weights
- execution/: Order management and exchange interfaces
- monitoring/: Real-time trading monitoring and performance tracking
- data/: Live data collection and market data providers
- utils/: Trading utilities and validation
- sizing/: Leverage and position management
- signal_generation/: Analyst and tactician signal integration
- backtesting/: Backtesting engine and performance analysis
- config/: Trading configuration management

The trading system integrates with the existing ML pipeline to provide
regime-aware trading decisions with proper risk management.
"""

from .config import *
from .regime import *
from .execution import *
from .monitoring import *
from .data import *
from .utils import *
from .sizing import *
from .signal_generation import *
from .backtesting import *

__version__ = "1.0.0"
__author__ = "Ares Trading System"