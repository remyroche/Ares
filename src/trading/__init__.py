"""
Trading Module

This module provides a comprehensive trading system with the following components:

- regime/: ML-based detection of 15-25 market regimes with percentage weights
- execution/: Order management, exchange interfaces, and live trading coordination
- monitoring/: Real-time trading monitoring and performance tracking
- data/: Live data collection and market data providers with ML integration
- utils/: Trading utilities and validation
- sizing/: Leverage and position management
- signal_generation/: Analyst and tactician signal integration
- backtesting/: Backtesting engine and performance analysis
- config/: Trading configuration management

The trading system integrates with the existing ML pipeline to provide
regime-aware trading decisions with proper risk management.

New Features:
- Live Trading Scheduler: Coordinates HMM, Analyst, and Tactician execution
- Enhanced Live Data Collector: Multi-timeframe data collection with ML integration
- Signal Generators: Analyst and Tactician signal generation components
- Trading Orchestrator: Unified coordination of all trading components
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

# Import new live trading components
from .execution.live_trading_scheduler import (
    LiveTradingScheduler, ModelType, ExecutionStatus, 
    create_live_trading_scheduler, start_live_trading_scheduler
)

from .execution.trading_orchestrator import (
    TradingOrchestrator, TradingMode, OrchestratorStatus,
    create_trading_orchestrator, start_trading_orchestrator
)

from .signal_generation.analyst_signals import (
    AnalystSignalGenerator, AnalystSignal, SignalType, SignalStrength,
    create_analyst_signal_generator, generate_analyst_signal
)

from .signal_generation.tactician_signals import (
    TacticianSignalGenerator, TacticianSignal, TimingSignal, TimingConfidence,
    create_tactician_signal_generator, generate_tactician_signal
)

# Enhanced data collector
from .data.live_data_collector import (
    LiveDataCollector, LiveDataConfig, LiveDataPoint,
    CollectionMode, DataQuality, CollectionInterval,
    create_live_data_collector, start_live_collection
)

__version__ = "1.1.0"
__author__ = "Ares Trading System"