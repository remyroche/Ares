"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .components.sr_clustering import SRClusteringComponent
from .components.sr_detection import SRDetectionComponent

# Import HDBSCAN regime discovery step
# from .hdbscan_clustering import HDBSCANRegimeDiscoveryStep  # Module not available

# Import GMM regime discovery step
# from .gmm_clustering import GMMRegimeDiscoveryStep  # Module not available

# Import SR parameter optimization step (BaseStep version)
from .components.sr_parameter_optimization import SRParameterOptimizationStep

# Import regime steps (BaseStep versions)
from .regime_clustering_step import RegimeClusteringStep
from .regime_models_training_step import RegimeModelsTrainingStep
from .regime_ensemble_training_step import RegimeEnsembleTrainingStep

# Import regime feature selector (EnhancedRegimeFeatureSelector with unsupervised mode)
from .regime_feature_selector import EnhancedRegimeFeatureSelector

# Import statsmodel clustering pipeline step
# from .statsmodel_clustering_pipeline_step import StatsmodelClusteringPipelineStep  # Module has missing dependencies

# Import Sticky Finite HMM regime discovery step
from .sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import StickyFiniteHMMRegimeDiscoveryStep

# Import Rolling HMM regime discovery step
from .rolling_hmm_clustering.rolling_hmm_regime_discovery_step import RollingHMMRegimeDiscoveryStep

# Import HMM ML alpha step (derives alpha labels and regimes from 1h HMM outputs)
from .hmm_ml_alpha_step import HMMMLAlphaStep

# Import ML Risk Regime step (risk-based regime classification)
from .ml_risk_regime_step import MLRiskRegimeStep

# Import ML Liquidity Regime step (liquidity-based regime classification)
from .ml_liquidity_regime_step import MLLiquidityRegimeStep

# Import Feature Generation Meta-Labeling Step (now in labeling module)
from src.training.steps.labeling import FeatureGenerationMetaLabelingStep

# Import Meta-Labeling HPO Experiment Step (offline, optional) (now in labeling module)
from src.training.steps.labeling import MetaLabelingHPOExperimentStep

# Note: EconomicRegimeFeatureSelector removed - it had circular dependency issues
# and lacked unsupervised mode. Use EnhancedRegimeFeatureSelector instead.

# Register market analysis steps
step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
step_registry.register("regime_clustering", RegimeClusteringStep)
step_registry.register("regime_models_training", RegimeModelsTrainingStep)
step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
# step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)  # Module not available
# step_registry.register("gmm_regime_discovery", GMMRegimeDiscoveryStep)  # Module not available
# Use EnhancedRegimeFeatureSelector which has proper unsupervised mode for pre-clustering selection
step_registry.register("regime_feature_selection", EnhancedRegimeFeatureSelector)
# step_registry.register("statsmodel_clustering_pipeline", StatsmodelClusteringPipelineStep)  # Module has missing dependencies
step_registry.register("sticky_finite_hmm_regime_discovery", StickyFiniteHMMRegimeDiscoveryStep)
step_registry.register("rolling_hmm_regime_discovery", RollingHMMRegimeDiscoveryStep)
step_registry.register("hmm_ml_alpha_step", HMMMLAlphaStep)
step_registry.register("ml_risk_regime_step", MLRiskRegimeStep)
step_registry.register("ml_liquidity_regime_step", MLLiquidityRegimeStep)
step_registry.register("feature_generation_meta_labeling_step", FeatureGenerationMetaLabelingStep)
step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)