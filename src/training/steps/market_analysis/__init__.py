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
from .regime_ensemble_training_step import RegimeEnsembleTrainingStep

# Import regime feature selector (EnhancedRegimeFeatureSelector with unsupervised mode)
from .regime_feature_selector import EnhancedRegimeFeatureSelector

# Import statsmodel clustering pipeline step
# from .statsmodel_clustering_pipeline_step import StatsmodelClusteringPipelineStep  # Module has missing dependencies

# Import Sticky Finite HMM regime discovery step (optional; depends on torch)
try:
    from .sticky_finite_hmm_clustering.sticky_finite_hmm_regime_discovery_step import StickyFiniteHMMRegimeDiscoveryStep
except Exception:  # pragma: no cover - optional dependency (e.g. torch)
    StickyFiniteHMMRegimeDiscoveryStep = None

# Import Rolling HMM regime discovery step (optional; depends on hmmlearn)
# Disabled here to avoid hard dependency on hmmlearn when importing MARKET_ANALYSIS.
RollingHMMRegimeDiscoveryStep = None

try:
    # Import HMM ML alpha and related XGB regime steps (these may depend on heavy
    # optional libraries like numba or custom HMM modules)
    from .xgb_meso_regime_step import XGBMesoTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    XGBMesoTrendStep = None

try:
    from .hmm_ml_alpha_step import HMMMLMesoTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    HMMMLMesoTrendStep = None

try:
    from .xgb_macro_regime_step import XGBMacroTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    XGBMacroTrendStep = None

try:
    from .xgb_mr_trend_step import XGBMrTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    XGBMrTrendStep = None

# Import ML Risk Regime HMM step (risk-based regime classification)
try:
    from .ml_risk_regime_step import MLRiskRegimeStepHMM
except Exception:  # pragma: no cover - optional heavy dependency
    MLRiskRegimeStepHMM = None

try:
    from .ml_smc_regime_step import MLSMCRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLSMCRegimeStep = None

# Import ML Path Regime step (efficiency/entropy-based path analysis)
try:
    from .ml_path_regime_step import MLPathRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLPathRegimeStep = None

try:
    from .ml_reversion_regime_step import MLMeanReversionRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLMeanReversionRegimeStep = None

# Import ML Breakout/Bounce Regime step (Relative-State breakout/bounce classifier)
try:
    from .ml_breakout_bounce_regime_step import MLBreakoutBounceRegimeStep
except Exception:  # pragma: no cover - defensive import guard
    MLBreakoutBounceRegimeStep = None

# Import ML Liquidity Regime step (liquidity-based regime classification)
try:
    from .ml_liquidity_regime_step import MLLiquidityRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLLiquidityRegimeStep = None

try:
    from .ml_map_regime_step import MLMapRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLMapRegimeStep = None

# Import ML Volume Force Step (Volume Force/Impulse prediction)
try:
    from .ml_volume_force_step import MLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLVolumeForceStep = None

# Import SR labeling XGB specialist step (S/R-based meta signal)
try:
    from .sr_labeling_xgb_step import SRLabelingXGBStep
except Exception:  # pragma: no cover - optional heavy dependency
    SRLabelingXGBStep = None

# Import Feature Generation Meta-Labeling Step (now in labeling module)
from src.training.steps.labeling import FeatureGenerationMetaLabelingStep

# Import Meta-Labeling HPO Experiment Step (offline, optional) (now in labeling
# module). This remains the dedicated HPO step registered under
# 'meta_labeling_hpo_experiment'.
from src.training.steps.labeling import MetaLabelingHPOExperimentStep

# Note: EconomicRegimeFeatureSelector removed - it had circular dependency issues
# and lacked unsupervised mode. Use EnhancedRegimeFeatureSelector instead.

# Register market analysis steps
step_registry.register("sr_parameter_optimization", SRParameterOptimizationStep)
step_registry.register("regime_clustering", RegimeClusteringStep)
step_registry.register("regime_ensemble_training", RegimeEnsembleTrainingStep)
step_registry.register("sr_clustering", SRClusteringComponent)
step_registry.register("sr_detection", SRDetectionComponent)
# step_registry.register("hdbscan_regime_discovery", HDBSCANRegimeDiscoveryStep)  # Module not available
# step_registry.register("gmm_regime_discovery", GMMRegimeDiscoveryStep)  # Module not available
# Use EnhancedRegimeFeatureSelector which has proper unsupervised mode for pre-clustering selection
step_registry.register("regime_feature_selection", EnhancedRegimeFeatureSelector)
# step_registry.register("statsmodel_clustering_pipeline", StatsmodelClusteringPipelineStep)  # Module has missing dependencies
if StickyFiniteHMMRegimeDiscoveryStep is not None:
    step_registry.register("sticky_finite_hmm_regime_discovery", StickyFiniteHMMRegimeDiscoveryStep)
if RollingHMMRegimeDiscoveryStep is not None:
    step_registry.register("rolling_hmm_regime_discovery", RollingHMMRegimeDiscoveryStep)
if XGBMesoTrendStep is not None:
    step_registry.register("xgb_meso_regime", XGBMesoTrendStep)
if XGBMrTrendStep is not None:
    step_registry.register("xgb_mr_trend_step", XGBMrTrendStep)
if HMMMLMesoTrendStep is not None:
    step_registry.register("hmm_ml_alpha_step", HMMMLMesoTrendStep)
if XGBMacroTrendStep is not None:
    step_registry.register("xgb_macro_regime", XGBMacroTrendStep)
if MLRiskRegimeStepHMM is not None:
    step_registry.register("ml_risk_regime_step", MLRiskRegimeStepHMM)
if MLSMCRegimeStep is not None:
    step_registry.register("ml_smc_regime_step", MLSMCRegimeStep)
if MLPathRegimeStep is not None:
    step_registry.register("ml_path_regime_step", MLPathRegimeStep)
if MLMeanReversionRegimeStep is not None:
    step_registry.register("ml_mean_reversion_step", MLMeanReversionRegimeStep)
if MLBreakoutBounceRegimeStep is not None:
    step_registry.register("ml_breakout_bounce_regime_step", MLBreakoutBounceRegimeStep)
if MLMapRegimeStep is not None:
    step_registry.register("ml_map_regime_step", MLMapRegimeStep)
if MLLiquidityRegimeStep is not None:
    step_registry.register("ml_liquidity_regime_step", MLLiquidityRegimeStep)
if MLVolumeForceStep is not None:
    step_registry.register("ml_volume_force_step", MLVolumeForceStep)
step_registry.register("feature_generation_meta_labeling_step", FeatureGenerationMetaLabelingStep)
if SRLabelingXGBStep is not None:
    # S/R-based specialist step producing sr_labeling_xgb_predictions_{timeframe}
    step_registry.register("sr_labeling_xgb", SRLabelingXGBStep)
if MetaLabelingHPOExperimentStep is not None:
    # Keep a dedicated alias for the meta-labeling HPO experiment.
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)
