"""
Integration Components for Statsmodels Clustering

This module provides integration components for statsmodels regime switching models,
including hardware optimization and VectorBT backtesting.

Key Components:
- StatsmodelsHardwareOptimizer: Hardware optimization integration
- VectorBTIntegration: Backtesting and portfolio analysis
"""

from .hardware_optimizer import (
    StatsmodelsHardwareOptimizer,
    HardwareOptimizationConfig,
    HardwareOptimizationResult,
    create_hardware_optimizer,
    optimize_for_regime_switching
)

try:
    from .vectorbt_integration import (
        VectorBTIntegration,
        VectorBTConfig,
        VectorBTResult,
        create_vectorbt_integration,
        backtest_regime_strategy
    )
    VECTORBT_AVAILABLE = True
except ImportError:
    VectorBTIntegration = None
    VectorBTConfig = None
    VectorBTResult = None
    create_vectorbt_integration = None
    backtest_regime_strategy = None
    VECTORBT_AVAILABLE = False

__all__ = [
    # Hardware optimization
    'StatsmodelsHardwareOptimizer',
    'HardwareOptimizationConfig',
    'HardwareOptimizationResult',
    'create_hardware_optimizer',
    'optimize_for_regime_switching',
    
    # VectorBT integration (conditional)
    'VectorBTIntegration',
    'VectorBTConfig',
    'VectorBTResult',
    'create_vectorbt_integration',
    'backtest_regime_strategy'
]

# Only export VectorBT components if available
if VECTORBT_AVAILABLE:
    __all__.extend([
        'VectorBTIntegration',
        'VectorBTConfig',
        'VectorBTResult',
        'create_vectorbt_integration',
        'backtest_regime_strategy'
    ])