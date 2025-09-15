"""
Market Analysis Pipeline Components.

This package contains individual components for the market analysis pipeline.
Each component is responsible for a specific part of the analysis process.
"""

from .base_component import (
    BaseMarketAnalysisComponent,
    ComponentConfig,
    ComponentResult
)
from .component_factory import ComponentFactory
from .artifact_manager import ArtifactManager
from .sr_parameter_optimization import SRParameterOptimizationComponent
from .sr_detection import SRDetectionComponent
from .sr_clustering import SRClusteringComponent
from .hmm_regime_discovery import HMMRegimeDiscoveryComponent
from .hmm_clustering import HMMClusteringComponent
from .hmm_models_training import HMMModelsTrainingComponent
from .hmm_ensemble_training import HMMEnsembleTrainingComponent
from .regime_data_splitting import RegimeDataSplittingComponent
from .triple_barrier_labeling import TripleBarrierLabelingComponent
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent

__all__ = [
    'BaseMarketAnalysisComponent',
    'ComponentConfig', 
    'ComponentResult',
    'ComponentFactory',
    'ArtifactManager',
    'SRParameterOptimizationComponent',
    'SRDetectionComponent',
    'SRClusteringComponent',
    'HMMRegimeDiscoveryComponent',
    'HMMClusteringComponent',
    'HMMModelsTrainingComponent',
    'HMMEnsembleTrainingComponent',
    'RegimeDataSplittingComponent',
    'TripleBarrierLabelingComponent',
    'FeatureLookbackOptimizationComponent',
    'CrossTimeframeAnalysisComponent'
]