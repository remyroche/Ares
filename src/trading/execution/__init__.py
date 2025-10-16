"""
Execution Module

Order management and exchange interfaces for trading execution.
Handles both paper and live trading with proper risk management.

New Features:
- Live Trading Scheduler: Coordinates HMM, Analyst, and Tactician execution
- Trading Orchestrator: Unified coordination of all trading components
"""

from .order_manager import (
    OrderManager, Order, OrderType, OrderSide, OrderStatus,
    TimeInForce, OrderBook, create_order_manager, get_order_manager
)
from .exchange_interface import (
    ExchangeInterface, ExchangeType, MarketDataType, ConnectionStatus,
    MarketData, TickerData, KlineData, SimulatedExchange,
    create_exchange_interface, get_exchange_interface
)
from .paper_trader import PaperTrader
from .live_trader import (
    LiveTrader, LiveTraderStatus, Position, TradingSession,
    create_live_trader, get_live_trader
)

# Import live trading components
from .live_trading_scheduler import (
    LiveTradingScheduler, ModelType, ExecutionStatus,
    create_live_trading_scheduler, start_live_trading_scheduler
)

from .trading_orchestrator import (
    TradingOrchestrator, TradingMode, OrchestratorStatus,
    create_trading_orchestrator, start_trading_orchestrator
)

__all__ = [
    # Order Management
    "OrderManager",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "OrderBook",
    "create_order_manager",
    "get_order_manager",

    # Exchange Interface
    "ExchangeInterface",
    "ExchangeType",
    "MarketDataType",
    "ConnectionStatus",
    "MarketData",
    "TickerData",
    "KlineData",
    "SimulatedExchange",
    "create_exchange_interface",
    "get_exchange_interface",

    # Trading Implementations
    "PaperTrader",
    "LiveTrader",
    "LiveTraderStatus",
    "Position",
    "TradingSession",
    "create_live_trader",
    "get_live_trader",

    # Live Trading Components
    "LiveTradingScheduler",
    "ModelType",
    "ExecutionStatus",
    "create_live_trading_scheduler",
    "start_live_trading_scheduler",

    # Orchestration
    "TradingOrchestrator",
    "TradingMode",
    "OrchestratorStatus",
    "create_trading_orchestrator",
    "start_trading_orchestrator"
]
