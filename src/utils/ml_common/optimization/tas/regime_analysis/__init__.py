"""
Regime Analysis for TAS

Advanced regime analysis capabilities for tree architecture search including:
- Regime detection and classification
- Regime-aware architecture optimization
- Regime transition analysis
- Regime-specific performance evaluation
- Regime visualization and reporting
"""

from .tree_regime_analyzer import TreeRegimeAnalyzer, TreeRegimeDetector, TreeRegimeClassifier
from .regime_optimization import TreeRegimeOptimizer, TreeRegimeSelector, TreeRegimeAdapter
from .regime_reporting import TreeRegimeReporter, TreeRegimeVisualizer, TreeRegimeDashboard

__all__ = [
    'TreeRegimeAnalyzer', 'TreeRegimeDetector', 'TreeRegimeClassifier',
    'TreeRegimeOptimizer', 'TreeRegimeSelector', 'TreeRegimeAdapter',
    'TreeRegimeReporter', 'TreeRegimeVisualizer', 'TreeRegimeDashboard'
]