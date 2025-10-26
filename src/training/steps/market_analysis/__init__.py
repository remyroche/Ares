"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .components.sr_clustering import SRClusteringComponent
from .components.sr_detection import SRDetectionComponent

# Import HDBSCAN regime discovery step
from .hdbscan_clustering import HDBSCANRegimeDiscoveryStep

# Import SR parameter optimization step (BaseStep version)
from .components.sr_parameter_optimization import SRParameterOptimizationStep

# Import regime steps (BaseStep versions)
from .regime_clustering_step import RegimeClusteringStep
from .regime_models_training_step import RegimeModelsTrainingStep
from .regime_ensemble_training_step import RegimeEnsembleTrainingStep
from .regime_data_splitting_step import RegimeDataSplittingStep

# Import economic regime feature selector
from .economic_regime_feature_selector import EconomicRegimeFeatureSelector

# Register market analysis steps
step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
step_registry.register("regime_clustering", RegimeClusteringStep)
step_registry.register("regime_models_training", RegimeModelsTrainingStep)
step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
step_registry.register("regime_data_splitting", RegimeDataSplittingStep)
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)
step_registry.register("regime_feature_selection", EconomicRegimeFeatureSelector)