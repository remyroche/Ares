"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry

# Import existing components
from .components.sr_clustering import SRClusteringComponent
from .components.sr_detection import SRDetectionComponent
from .components.regime_clustering import RegimeClusteringComponent

# Import HDBSCAN regime discovery step
from .hdbscan_clustering import HDBSCANRegimeDiscoveryStep

# Import migrated steps
from .sr_detection import SRDetectionStep
from .model_persistence_components.model_persistence_step import ModelPersistenceStep
from .regime_data_splitting.regime_data_splitting_main import RegimeDataSplittingStep
from .components.regime_models_training import RegimeModelsTrainingStep
from .components.regime_ensemble_training import RegimeEnsembleTrainingStep
from .components.sr_parameter_optimization import SRParameterOptimizationStep
from .regime_clustering_step import RegimeClusteringStep

# Import market analysis orchestrator step
from .market_analysis_step import MarketAnalysisStep

# Register existing components
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
step_registry.register("hdbscan_clustering", HDBSCANRegimeDiscoveryStep)

# Register migrated steps
step_registry.register("sr_detection_step", SRDetectionStep)
step_registry.register("model_persistence", ModelPersistenceStep)
step_registry.register("regime_data_splitting", RegimeDataSplittingStep)
step_registry.register("regime_models_training", RegimeModelsTrainingStep)
step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
step_registry.register("regime_clustering", RegimeClusteringStep)

# Register market analysis orchestrator step
step_registry.register("market_analysis", MarketAnalysisStep)