"""
Trading Reporting

Comprehensive reporting system for trading operations including
detailed trade analysis, performance metrics, and ML explanations.
"""

from .performance_reporter import (
    PerformanceReporter,
    generate_trading_report,
    calculate_performance_metrics,
    PerformanceMetrics,
    TradingPerformanceError
)
from .trade_analyzer import (
    TradeAnalyzer,
    analyze_trade_performance,
    TradeAnalysisResult,
    TradePatternAnalysis,
    TradingAnalysisError
)
from .dashboard_generator import (
    DashboardGenerator,
    create_trading_dashboard,
    update_dashboard_data,
    DashboardConfig,
    DashboardError
)
from .daily_recorder import (
    DailyRecorder,
    DailyTradingRecord,
    record_daily_trading_summary,
    get_daily_trading_summary,
    get_trading_history,
    DailyRecordingError
)

__all__ = [
    'PerformanceReporter', 'TradeAnalyzer', 'DashboardGenerator', 'DailyRecorder',
    'generate_trading_report', 'analyze_trade_performance', 'create_trading_dashboard',
    'record_daily_trading_summary', 'get_daily_trading_summary', 'get_trading_history'
]
