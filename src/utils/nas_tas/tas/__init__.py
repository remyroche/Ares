"""
Tree-Based Architecture Search (TAS) module.

This module contains TAS-specific implementations for automated tree-based model architecture discovery.
"""

from .tree_based_architecture_search import (
    TreeBasedArchitectureSearch,
    TreeArchitectureConfig,
    TreeArchitectureCandidate,
    TreeArchitectureSearchSpace,
    search_tree_architecture
)

from .pure_tree_nas import (
    PureTreeNAS,
    PureTreeNASConfig,
    TreeArchitectureCandidate as PureTreeArchitectureCandidate,
    NODEModel,
    ObliviousTree,
    ObliviousTreeModel,
    RotationForestModel,
    HistogramGradientBoostingModel
)

from .unsupervised_tree_nas import (
    UnsupervisedTreeNAS,
    UnsupervisedTreeNASConfig,
    RegimeCandidate,
    UnsupervisedArchitectureCandidate
)

from .regime_trading_tree_nas import (
    RegimeTradingTreeNAS,
    RegimeTradingTreeNASConfig,
    RegimeDetectionTree,
    TradingSignalTree,
    RiskManagementTree,
    PositionSizingTree
)

from .trading_tree_architecture_search import (
    TradingTreeArchitectureSearch,
    TradingTASConfig,
    TradingRegime,
    TradingTASResult,
    TradingObjective,
    MarketRegime,
    optimize_trading_regimes,
    select_trading_model
)

__all__ = [
    # Core TAS
    'TreeBasedArchitectureSearch',
    'TreeArchitectureConfig',
    'TreeArchitectureCandidate', 
    'TreeArchitectureSearchSpace',
    'search_tree_architecture',
    
    # Pure Tree NAS
    'PureTreeNAS',
    'PureTreeNASConfig',
    'PureTreeArchitectureCandidate',
    'NODEModel',
    'ObliviousTree',
    'ObliviousTreeModel',
    'RotationForestModel',
    'HistogramGradientBoostingModel',
    
    # Unsupervised Tree NAS
    'UnsupervisedTreeNAS',
    'UnsupervisedTreeNASConfig',
    'RegimeCandidate',
    'UnsupervisedArchitectureCandidate',
    
    # Regime Trading Tree NAS
    'RegimeTradingTreeNAS',
    'RegimeTradingTreeNASConfig',
    'RegimeDetectionTree',
    'TradingSignalTree',
    'RiskManagementTree',
    'PositionSizingTree',
    
    # Trading Tree Architecture Search
    'TradingTreeArchitectureSearch',
    'TradingTASConfig',
    'TradingRegime',
    'TradingTASResult',
    'TradingObjective',
    'MarketRegime',
    'optimize_trading_regimes',
    'select_trading_model'
]