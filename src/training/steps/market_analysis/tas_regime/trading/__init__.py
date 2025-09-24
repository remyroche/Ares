"""
Trading Integration for TAS

Production-ready trading system integration for tree architecture search including:
- Trading signal generation
- Position management
- Risk management
- Performance monitoring
- Order execution simulation
"""

from .trading_engine import TradingEngine, TradingConfig, TradingResult
from .signal_generator import TradingSignalGenerator, SignalConfig
from .position_manager import PositionManager, PositionConfig
from .risk_manager import RiskManager, RiskConfig
from .performance_monitor import TradingPerformanceMonitor, PerformanceConfig

__all__ = [
    'TradingEngine', 'TradingConfig', 'TradingResult',
    'TradingSignalGenerator', 'SignalConfig',
    'PositionManager', 'PositionConfig',
    'RiskManager', 'RiskConfig',
    'TradingPerformanceMonitor', 'PerformanceConfig'
]