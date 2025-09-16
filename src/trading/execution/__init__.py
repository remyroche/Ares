"""
Execution Module

Order management and exchange interfaces for trading execution.
Handles both paper and live trading with proper risk management.

New Features:
- Live Trading Scheduler: Coordinates HMM, Analyst, and Tactician execution
- Trading Orchestrator: Unified coordination of all trading components
"""

from .order_manager import OrderManager
from .exchange_interface import ExchangeInterface
from .paper_trader import PaperTrader
from .live_trader import LiveTrader

# Import new live trading components
from .live_trading_scheduler import (
    LiveTradingScheduler, ModelType, ExecutionStatus, 
    create_live_trading_scheduler, start_live_trading_scheduler
)

from .trading_orchestrator import (
    TradingOrchestrator, TradingMode, OrchestratorStatus,
    create_trading_orchestrator, start_trading_orchestrator
)

__all__ = [
    "OrderManager",
    "ExchangeInterface",
    "PaperTrader", 
    "LiveTrader",
    "LiveTradingScheduler",
    "ModelType",
    "ExecutionStatus",
    "create_live_trading_scheduler",
    "start_live_trading_scheduler",
    "TradingOrchestrator",
    "TradingMode",
    "OrchestratorStatus",
    "create_trading_orchestrator",
    "start_trading_orchestrator"
]