"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry

# Import existing components
from .components.sr_clustering import SRClusteringComponent
from .components.sr_detection import SRDetectionComponent

# Import HDBSCAN regime discovery step
from .hdbscan_clustering import HDBSCANRegimeDiscoveryStep

# Import migrated steps
from .sr_detection import SRDetectionStep
from .model_persistence_components.model_persistence_step import ModelPersistenceStep
from .regime_data_splitting.regime_data_splitting_step import RegimeDataSplittingStep
from .regime_models_training_step import RegimeModelsTrainingStep
from .regime_ensemble_training_step import RegimeEnsembleTrainingStep
from .sr_parameter_optimization_step import SRParameterOptimizationStep

# Register existing components
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)

# Register migrated steps
step_registry.register("sr_detection_step", SRDetectionStep)
step_registry.register("model_persistence", ModelPersistenceStep)
step_registry.register("regime_data_splitting", RegimeDataSplittingStep)
step_registry.register("regime_models_training", RegimeModelsTrainingStep)
step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)