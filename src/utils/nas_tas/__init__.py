"""
NAS-TAS Common Utilities

This module provides common utilities and unified backtesting framework that
consolidates all functionality from TAS, NAS, and hybrid systems.

Components:
- BacktestingEngine: Core backtesting functionality
- MonteCarloEngine: Monte Carlo simulation
- PerformanceAttribution: Performance analysis
- WalkForwardAnalyzer: Time series validation
- DataManager: Data management utilities
- RiskAnalyzer: Risk assessment tools
"""

from .backtesting_engine import (
    BacktestingEngine,
    BacktestingConfig,
    BacktestingResult,
    BacktestingMode
)

from .monte_carlo_engine import (
    MonteCarloEngine,
    MonteCarloConfig,
    MonteCarloResult
)

from .performance_attribution import (
    PerformanceAttribution,
    PerformanceAttributionConfig,
    PerformanceMetrics
)

from .walk_forward_analyzer import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardResult
)

from .data_manager import (
    BacktestingDataManager,
    DataManagerConfig
)

from .risk_analyzer import (
    RiskAnalyzer,
    RiskAnalysisConfig,
    RiskMetrics
)

from .unified_orchestrator import (
    UnifiedBacktestingOrchestrator,
    OrchestratorConfig
)

__all__ = [
    # Core backtesting
    "BacktestingEngine",
    "BacktestingConfig", 
    "BacktestingResult",
    "BacktestingMode",
    
    # Monte Carlo
    "MonteCarloEngine",
    "MonteCarloConfig",
    "MonteCarloResult",
    
    # Performance attribution
    "PerformanceAttribution",
    "PerformanceAttributionConfig",
    "PerformanceMetrics",
    
    # Walk forward analysis
    "WalkForwardAnalyzer",
    "WalkForwardConfig",
    "WalkForwardResult",
    
    # Data management
    "BacktestingDataManager",
    "DataManagerConfig",
    
    # Risk analysis
    "RiskAnalyzer",
    "RiskAnalysisConfig",
    "RiskMetrics",
    
    # Unified orchestrator
    "UnifiedBacktestingOrchestrator",
    "OrchestratorConfig"
]