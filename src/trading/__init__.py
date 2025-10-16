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
from .reporting import *
from .integration import *

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

# Comprehensive monitoring and reporting
from .monitoring.comprehensive_trade_monitor import (
    ComprehensiveTradeMonitor, DetailedTradeMetrics, TradingSessionMetrics,
    comprehensive_trade_monitor, initialize_comprehensive_monitoring,
    record_detailed_trade, update_trade_outcome
)
from .monitoring.unified_trailing_manager import (
    UnifiedTrailingManager, TrailingAction, TrailingDecision
)

from .reporting.performance_reporter import (
    PerformanceReporter, performance_reporter, generate_trading_report
)

from .reporting.dashboard_generator import (
    DashboardGenerator, dashboard_generator, create_trading_dashboard
)

from .reporting.trade_analyzer import (
    TradeAnalyzer, trade_analyzer, analyze_trade_performance
)

# Integration utilities
from .integration.model_integration import (
    TrainingModelLoader, training_model_loader, load_trained_models, validate_model_compatibility
)

from .integration.training_integration import (
    TrainingDataProvider, training_data_provider, get_training_features, sync_with_training_pipeline
)

__version__ = "1.2.0"
__author__ = "Ares Trading System"
