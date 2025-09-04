#!/usr/bin/env python3
"""HMM Clustering Package for Enhanced Regime Discovery.

This package contains all the enhanced components for HMM regime discovery:
- Optimized Bayesian parameter optimization
- Enhanced regime discovery features
- Economic significance validation
- Ensemble clustering (HMM + K-means + DBSCAN)
- Enhanced ML transition detection (Random Forest + LGBM)
- Memory optimization and streaming processing
- Hierarchical multi-scale detection
- Dynamic regime count optimization
- Regime persistence and forecasting
- Microservices architecture
- Real-time streaming pipeline
"""

from .step03_optimized_bayesian_optimization import OptimizedBayesianParameterOptimization
from .step03_regime_discovery_features import RegimeDiscoveryFeatureEngineer
from .step03_economic_significance_validator import EconomicSignificanceValidator
from .step03_ensemble_clustering import EnsembleClusteringRegimeDetector
from .step03_ml_transition_detector import MLRegimeTransitionDetector
from .step03_enhanced_ml_transition_detector import EnhancedMLRegimeTransitionDetector
from .step03_streaming_regime_discovery import StreamingRegimeDiscovery, MemoryOptimizedRegimeDiscovery
from .step03_hierarchical_regime_detection import HierarchicalRegimeDetector
from .step03_dynamic_regime_optimization import DynamicRegimeCountOptimizer
from .step03_regime_persistence_forecasting import RegimePersistenceForecaster
from .step03_microservices_regime_discovery import MicroservicesRegimeDiscovery, ServiceOrchestrator
from .step03_realtime_streaming_pipeline import RealTimeStreamingPipeline, StreamingRegimeDiscovery as RealtimeStreaming
from .step03_enhanced_hmm_regime_discovery import EnhancedHMMRegimeDiscoveryStep, run_enhanced_step

__all__ = [
    'OptimizedBayesianParameterOptimization',
    'RegimeDiscoveryFeatureEngineer', 
    'EconomicSignificanceValidator',
    'EnsembleClusteringRegimeDetector',
    'MLRegimeTransitionDetector',
    'EnhancedMLRegimeTransitionDetector',
    'StreamingRegimeDiscovery',
    'MemoryOptimizedRegimeDiscovery',
    'HierarchicalRegimeDetector',
    'DynamicRegimeCountOptimizer',
    'RegimePersistenceForecaster',
    'MicroservicesRegimeDiscovery',
    'ServiceOrchestrator',
    'RealTimeStreamingPipeline',
    'RealtimeStreaming',
    'EnhancedHMMRegimeDiscoveryStep',
    'run_enhanced_step'
]