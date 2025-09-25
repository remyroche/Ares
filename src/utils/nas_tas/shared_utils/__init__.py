"""
Unified NAS-TAS System

This package provides a unified interface for both Neural Architecture Search (NAS)
and Tree Architecture Search (TAS) systems. It consolidates all search strategies,
optimization algorithms, evaluation methods, and utilities into a single, cohesive framework.

Key Features:
- Unified search engine supporting both neural and tree architectures
- Comprehensive multi-objective optimization with multiple algorithms
- Advanced economic evaluation and trading viability assessment
- Flexible regime detection with multiple methods
- Hardware optimization and parallel processing support

Main Components:
- SearchStrategyManager: Unified interface for all search strategies
- UnifiedMultiObjectiveOptimizer: Multi-objective optimization with NSGA-II, Bayesian, etc.
- UnifiedEconomicEvaluator: Economic significance and trading viability evaluation
- UnifiedClusteringAlgorithm: Regime detection with clustering algorithms
- UnifiedHardwareManager: Hardware optimization and parallel processing
- PositionAwareTradingAnalyzer: Position-aware trading analysis
- MetricsReporter: Comprehensive metrics reporting

Usage Example:
    from src.utils.nas_tas.shared_utils import (
        SearchStrategyManager, UnifiedMultiObjectiveOptimizer, UnifiedEconomicEvaluator
    )
    
    # Create search strategy manager
    search_manager = SearchStrategyManager()
    
    # Create optimizer
    optimizer = UnifiedMultiObjectiveOptimizer()
    
    # Create evaluator
    evaluator = UnifiedEconomicEvaluator()
"""

# Import main unified components
from .search_strategies import (
    SearchStrategyManager,
    SearchStrategyConfig
)

from .unified_multi_objective_optimizer import (
    UnifiedMultiObjectiveOptimizer,
    UnifiedMultiObjectiveConfig,
    OptimizationAlgorithm
)

from .unified_economic_evaluator import (
    UnifiedEconomicEvaluator,
    EconomicEvaluationConfig
)

from .unified_clustering_algorithms import (
    UnifiedClusteringAlgorithm,
    ClusteringConfig
)

from .unified_hardware_manager import (
    UnifiedHardwareManager,
    HardwareType,
    WorkloadType,
    HardwareMetrics
)

from .unified_architecture_config import (
    ArchitectureType,
    ArchitectureConfig
)

from .analysis_components import (
    SharedClusteringUtilities,
    AnalysisComponentConfig
)

from .position_aware_trading import (
    PositionAwareTradingAnalyzer,
    PositionAwareConfig
)

from .metrics_reporting import (
    MetricsReporter,
    MetricsConfig
)

from .unified_trading_viability_evaluator import (
    UnifiedTradingViabilityEvaluator,
    TradingViabilityConfig
)

# Version information
__version__ = "1.0.0"
__author__ = "Unified NAS-TAS System Team"
__email__ = "unified-system@example.com"

# Main exports
__all__ = [
    # Core unified components
    'SearchStrategyManager',
    'UnifiedMultiObjectiveOptimizer', 
    'UnifiedEconomicEvaluator',
    'UnifiedClusteringAlgorithm',
    'UnifiedHardwareManager',
    'UnifiedTradingViabilityEvaluator',
    
    # Configuration classes
    'SearchStrategyConfig',
    'UnifiedMultiObjectiveConfig',
    'EconomicEvaluationConfig',
    'ClusteringConfig',
    'ArchitectureConfig',
    'AnalysisComponentConfig',
    'PositionAwareConfig',
    'MetricsConfig',
    'TradingViabilityConfig',
    
    # Enums
    'ArchitectureType',
    'OptimizationAlgorithm',
    'HardwareType',
    'WorkloadType',
    
    # Utility classes
    'SharedClusteringUtilities',
    'PositionAwareTradingAnalyzer',
    'MetricsReporter',
    'HardwareMetrics',
    
    # Version info
    '__version__',
    '__author__',
    '__email__'
]