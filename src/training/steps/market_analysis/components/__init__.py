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
from .nas_regime_discovery import NASRegimeDiscoveryComponent
from .tas_regime_discovery import TASRegimeDiscoveryComponent
from .nas_tas_regime_discovery import NASTASRegimeDiscoveryComponent
from .nas_tas_clustering import NASTASClusteringComponent
# from .hmm_regime_discovery import HMMRegimeDiscoveryComponent  # DEPRECATED
# HMM training components moved to hmm_models_training module
# from .hmm_models_training import HMMModelsTrainingComponent
# from .hmm_ensemble_training_component import HMMEnsembleTrainingComponent  # DEPRECATED
# RegimeDataSplittingComponent imported lazily to avoid circular imports
# TripleBarrierLabelingComponent moved to triple_barrier_labeling package
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent
from .nas_tas_models_training import NASTASModelsTrainingComponent
from .nas_tas_ensemble_training import NASTASEnsembleTrainingComponent
from .regime_models_training import RegimeModelsTrainingComponent
from .regime_ensemble_training import RegimeEnsembleTrainingComponent

__all__ = [
    'BaseMarketAnalysisComponent',
    'ComponentConfig',
    'ComponentResult',
    'ComponentFactory',
    'ArtifactManager',
    'SRParameterOptimizationComponent',
    'SRDetectionComponent',
    'SRClusteringComponent',
    'NASRegimeDiscoveryComponent',
    'TASRegimeDiscoveryComponent',
    'NASTASRegimeDiscoveryComponent',
    'NASTASClusteringComponent',
    'NASTASModelsTrainingComponent',
    'NASTASEnsembleTrainingComponent',
    'HMMRegimeDiscoveryComponent',
    # 'HMMModelsTrainingComponent',  # Moved to hmm_models_training module
    'HMMEnsembleTrainingComponent'
    # 'RegimeDataSplittingComponent',  # Imported lazily to avoid circular imports
    # 'TripleBarrierLabelingComponent',  # Moved to triple_barrier_labeling package
    'FeatureLookbackOptimizationComponent',
    'CrossTimeframeAnalysisComponent',
    'RegimeModelsTrainingComponent',
    'RegimeEnsembleTrainingComponent'
]