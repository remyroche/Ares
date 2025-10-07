"""
Monitoring Module

Real-time trading monitoring and performance tracking.
Provides trade monitoring, performance analysis, and alert management.
"""

from .trade_monitor import (
    TradeMonitor, Trade, TradeStatus, Alert, AlertLevel, PerformanceMetrics,
    create_trade_monitor, start_trade_monitoring
)
# Comprehensive trade monitor imports removed to avoid circular imports
# from .comprehensive_trade_monitor import (
#     ComprehensiveTradeMonitor, DetailedTradeMetrics, TradingSessionMetrics,
#     comprehensive_trade_monitor, initialize_comprehensive_monitoring,
#     record_detailed_trade, update_trade_outcome
# )
from .performance_tracker import (
    PerformanceTracker, TradeRecord, DailyPerformance, PerformanceSnapshot,
    MetricType, create_performance_tracker, get_performance_tracker
)
from .regime_monitor import (
    RegimeMonitor, RegimeState, RegimeTransition, RegimeAlert,
    RegimeStability, AlertSeverity, create_regime_monitor, get_regime_monitor
)
from .alert_manager import (
    AlertManager, AlertRule, Alert, NotificationChannel, AlertType,
    AlertPriority, NotificationResult, create_alert_manager, get_alert_manager
)
from .unified_trailing_manager import (
    UnifiedTrailingManager, TrailingAction, TrailingDecision
)

__all__ = [
    # Trade Monitor
    "TradeMonitor",
    "Trade",
    "TradeStatus",
    "Alert",
    "AlertLevel",
    "PerformanceMetrics",
    "create_trade_monitor",
    "start_trade_monitoring",

    # Comprehensive Monitor exports removed to avoid circular imports

    # Performance Tracker
    "PerformanceTracker",
    "TradeRecord",
    "DailyPerformance",
    "PerformanceSnapshot",
    "MetricType",
    "create_performance_tracker",
    "get_performance_tracker",

    # Regime Monitor
    "RegimeMonitor",
    "RegimeState",
    "RegimeTransition",
    "RegimeAlert",
    "RegimeStability",
    "AlertSeverity",
    "create_regime_monitor",
    "get_regime_monitor",

    # Alert Manager
    "AlertManager",
    "AlertRule",
    "Alert",
    "NotificationChannel",
    "AlertType",
    "AlertPriority",
    "NotificationResult",
    "create_alert_manager",
    "get_alert_manager",

    # Unified trailing manager
    "UnifiedTrailingManager",
    "TrailingAction",
    "TrailingDecision",
]