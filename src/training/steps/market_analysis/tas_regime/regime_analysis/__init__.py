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
from src.utils.ml_common.optimization.tas.regime_analysis.clustering_regime_detection import (
    TreeBasedClusteringRegimeDetector,
    ClusteringRegimeConfig,
    quick_clustering_detection
)
from .unsupervised_regime_detection import UnsupervisedRegimeDetector, RegimeDetectionConfig

__all__ = [
    'TreeRegimeAnalyzer', 'TreeRegimeDetector', 'TreeRegimeClassifier',
    'TreeRegimeOptimizer', 'TreeRegimeSelector', 'TreeRegimeAdapter',
    'TreeRegimeReporter', 'TreeRegimeVisualizer', 'TreeRegimeDashboard',
    'TreeBasedClusteringRegimeDetector', 'ClusteringRegimeConfig', 'quick_clustering_detection',
    'UnsupervisedRegimeDetector', 'RegimeDetectionConfig'
]