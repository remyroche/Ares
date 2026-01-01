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
    # Import XGB regime steps (these may depend on heavy optional libraries like numba)
    from .xgb_meso_regime_step import XGBMesoTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    XGBMesoTrendStep = None

try:
    from .xgb_macro_regime_step import XGBMacroTrendStep
except Exception:  # pragma: no cover - optional heavy dependency
    XGBMacroTrendStep = None

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

# Import ML Volume Force Step (Volume Force/Impulse prediction)
try:
    from .ml_volume_force_step import MLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLVolumeForceStep = None

# Import ML Momentum Persistence Step (1.5-3% Range Specialist)
try:
    from .ml_momentum_persistence_step import MLMomentumPersistenceStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLMomentumPersistenceStep = None

# Import ML Volatility Burst Step (1.5-3% Range Specialist)
try:
    from .ml_volatility_burst_step import MLVolatilityBurstStep
except Exception:  # pragma: no cover - optional heavy dependency
    MLVolatilityBurstStep = None

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
if MLLiquidityRegimeStep is not None:
    step_registry.register("ml_liquidity_regime_step", MLLiquidityRegimeStep)
if MLVolumeForceStep is not None:
    step_registry.register("ml_volume_force_step", MLVolumeForceStep)
if MLMomentumPersistenceStep is not None:
    step_registry.register("ml_momentum_persistence_step", MLMomentumPersistenceStep)
if MLVolatilityBurstStep is not None:
    step_registry.register("ml_volatility_burst_step", MLVolatilityBurstStep)
step_registry.register("feature_generation_meta_labeling_step", FeatureGenerationMetaLabelingStep)
if SRLabelingXGBStep is not None:
    # S/R-based specialist step producing sr_labeling_xgb_predictions_{timeframe}
    step_registry.register("sr_labeling_xgb", SRLabelingXGBStep)
if MetaLabelingHPOExperimentStep is not None:
    # Keep a dedicated alias for the meta-labeling HPO experiment.
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)

# Import ML Momentum Persistence Step (1.5-3% Range Specialist)
try:
    from .ml_momentum_persistence_step import MLMomentumPersistenceStep
except Exception:
    MLMomentumPersistenceStep = None

# Import ML Volatility Burst Step (1.5-3% Range Specialist)
try:
    from .ml_volatility_burst_step import MLVolatilityBurstStep
except Exception:
    MLVolatilityBurstStep = None

# Register specialist steps
if MLMomentumPersistenceStep is not None:
    step_registry.register("ml_momentum_persistence_step", MLMomentumPersistenceStep)
if MLVolatilityBurstStep is not None:
    step_registry.register("ml_volatility_burst_step", MLVolatilityBurstStep)


# Import Enhanced Specialists
try:
    from .ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLMomentumPersistenceStep = None

try:
    from .ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLSMCRegimeStep = None

try:
    from .ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolatilityBurstStep = None

try:
    from .ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolumeForceStep = None

try:
    from .ml_breakout_bounce_regime_step_enhanced import EnhancedMLBreakoutBounceRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLBreakoutBounceRegimeStep = None

try:
    from .ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLReversionRegimeStep = None

try:
    from .xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMacroRegimeStep = None

try:
    from .ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLLiquidityRegimeStep = None

try:
    from .ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLPathRegimeStep = None

try:
    from .ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLRiskRegimeStep = None

try:
    from .xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMesoRegimeStep = None

try:
    from .ml_microstructure_step_enhanced import EnhancedMLMicrostructureStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLMicrostructureStep = None

try:
    from .ml_candlestick_step_enhanced import EnhancedMLCandlestickStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLCandlestickStep = None

try:
    from .ml_spectral_step_enhanced import EnhancedMLSpectralStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLSpectralStep = None

# Import Enhanced Specialists
try:
    from .ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLMomentumPersistenceStep = None

try:
    from .ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLSMCRegimeStep = None

try:
    from .ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolatilityBurstStep = None

try:
    from .ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolumeForceStep = None

try:
    from .ml_breakout_bounce_regime_step_enhanced import EnhancedMLBreakoutBounceRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLBreakoutBounceRegimeStep = None

try:
    from .ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLReversionRegimeStep = None

try:
    from .xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMacroRegimeStep = None

try:
    from .ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLLiquidityRegimeStep = None

try:
    from .ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLPathRegimeStep = None

try:
    from .ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLRiskRegimeStep = None

try:
    from .xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMesoRegimeStep = None

# Import Enhanced Specialists
try:
    from .ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLMomentumPersistenceStep = None

try:
    from .ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLSMCRegimeStep = None

try:
    from .ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolatilityBurstStep = None

try:
    from .ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolumeForceStep = None

try:
    from .ml_breakout_bounce_regime_step_enhanced import EnhancedMLBreakoutBounceRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLBreakoutBounceRegimeStep = None

try:
    from .ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLReversionRegimeStep = None

try:
    from .xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMacroRegimeStep = None

try:
    from .ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLLiquidityRegimeStep = None

try:
    from .ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLPathRegimeStep = None

try:
    from .ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLRiskRegimeStep = None

try:
    from .xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMesoRegimeStep = None

# Import Enhanced Specialists
try:
    from .ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLMomentumPersistenceStep = None

try:
    from .ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLSMCRegimeStep = None

try:
    from .ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolatilityBurstStep = None

try:
    from .ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLVolumeForceStep = None

try:
    from .ml_breakout_bounce_regime_step_enhanced import EnhancedMLBreakoutBounceRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLBreakoutBounceRegimeStep = None

try:
    from .ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLReversionRegimeStep = None

try:
    from .xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMacroRegimeStep = None

try:
    from .ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLLiquidityRegimeStep = None

try:
    from .ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLPathRegimeStep = None

try:
    from .ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedMLRiskRegimeStep = None

try:
    from .xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
except Exception:  # pragma: no cover - optional heavy dependency
    EnhancedXGBMesoRegimeStep = None
# Note: All specialists now have independent diagnostics via SpecialistDiagnosticsMixin
# Register Enhanced Specialists
if EnhancedMLMomentumPersistenceStep is not None:
    step_registry.register("enhanced_ml_momentum_persistence_step", EnhancedMLMomentumPersistenceStep)
if EnhancedMLSMCRegimeStep is not None:
    step_registry.register("enhanced_ml_smc_regime_step", EnhancedMLSMCRegimeStep)
if EnhancedMLVolatilityBurstStep is not None:
    step_registry.register("enhanced_ml_volatility_burst_step", EnhancedMLVolatilityBurstStep)
if EnhancedMLVolumeForceStep is not None:
    step_registry.register("enhanced_ml_volume_force_step", EnhancedMLVolumeForceStep)
if EnhancedMLBreakoutBounceRegimeStep is not None:
    step_registry.register("enhanced_ml_breakout_bounce_regime_step", EnhancedMLBreakoutBounceRegimeStep)
if EnhancedMLReversionRegimeStep is not None:
    step_registry.register("enhanced_ml_reversion_regime_step", EnhancedMLReversionRegimeStep)
if EnhancedXGBMacroRegimeStep is not None:
    step_registry.register("enhanced_xgb_macro_regime_step", EnhancedXGBMacroRegimeStep)
if EnhancedMLLiquidityRegimeStep is not None:
    step_registry.register("enhanced_ml_liquidity_regime_step", EnhancedMLLiquidityRegimeStep)
if EnhancedMLPathRegimeStep is not None:
    step_registry.register("enhanced_ml_path_regime_step", EnhancedMLPathRegimeStep)
if EnhancedMLRiskRegimeStep is not None:
    step_registry.register("enhanced_ml_risk_regime_step", EnhancedMLRiskRegimeStep)
if EnhancedXGBMesoRegimeStep is not None:
    step_registry.register("enhanced_xgb_meso_regime_step", EnhancedXGBMesoRegimeStep)

# Register Enhanced Specialists
if EnhancedMLMomentumPersistenceStep is not None:
    step_registry.register("enhanced_ml_momentum_persistence_step", EnhancedMLMomentumPersistenceStep)
if EnhancedMLSMCRegimeStep is not None:
    step_registry.register("enhanced_ml_smc_regime_step", EnhancedMLSMCRegimeStep)
if EnhancedMLVolatilityBurstStep is not None:
    step_registry.register("enhanced_ml_volatility_burst_step", EnhancedMLVolatilityBurstStep)
if EnhancedMLVolumeForceStep is not None:
    step_registry.register("enhanced_ml_volume_force_step", EnhancedMLVolumeForceStep)
if EnhancedMLBreakoutBounceRegimeStep is not None:
    step_registry.register("enhanced_ml_breakout_bounce_regime_step", EnhancedMLBreakoutBounceRegimeStep)
if EnhancedMLReversionRegimeStep is not None:
    step_registry.register("enhanced_ml_reversion_regime_step", EnhancedMLReversionRegimeStep)
if EnhancedXGBMacroRegimeStep is not None:
    step_registry.register("enhanced_xgb_macro_regime_step", EnhancedXGBMacroRegimeStep)
if EnhancedMLLiquidityRegimeStep is not None:
    step_registry.register("enhanced_ml_liquidity_regime_step", EnhancedMLLiquidityRegimeStep)
if EnhancedMLPathRegimeStep is not None:
    step_registry.register("enhanced_ml_path_regime_step", EnhancedMLPathRegimeStep)
if EnhancedMLRiskRegimeStep is not None:
    step_registry.register("enhanced_ml_risk_regime_step", EnhancedMLRiskRegimeStep)
if EnhancedXGBMesoRegimeStep is not None:
    step_registry.register("enhanced_xgb_meso_regime_step", EnhancedXGBMesoRegimeStep)

# Register Enhanced Specialists
if EnhancedMLMomentumPersistenceStep is not None:
    step_registry.register("enhanced_ml_momentum_persistence_step", EnhancedMLMomentumPersistenceStep)
if EnhancedMLSMCRegimeStep is not None:
    step_registry.register("enhanced_ml_smc_regime_step", EnhancedMLSMCRegimeStep)
if EnhancedMLVolatilityBurstStep is not None:
    step_registry.register("enhanced_ml_volatility_burst_step", EnhancedMLVolatilityBurstStep)
if EnhancedMLVolumeForceStep is not None:
    step_registry.register("enhanced_ml_volume_force_step", EnhancedMLVolumeForceStep)
if EnhancedMLBreakoutBounceRegimeStep is not None:
    step_registry.register("enhanced_ml_breakout_bounce_regime_step", EnhancedMLBreakoutBounceRegimeStep)
if EnhancedMLReversionRegimeStep is not None:
    step_registry.register("enhanced_ml_reversion_regime_step", EnhancedMLReversionRegimeStep)
if EnhancedXGBMacroRegimeStep is not None:
    step_registry.register("enhanced_xgb_macro_regime_step", EnhancedXGBMacroRegimeStep)
if EnhancedMLLiquidityRegimeStep is not None:
    step_registry.register("enhanced_ml_liquidity_regime_step", EnhancedMLLiquidityRegimeStep)
if EnhancedMLPathRegimeStep is not None:
    step_registry.register("enhanced_ml_path_regime_step", EnhancedMLPathRegimeStep)
if EnhancedMLRiskRegimeStep is not None:
    step_registry.register("enhanced_ml_risk_regime_step", EnhancedMLRiskRegimeStep)
if EnhancedXGBMesoRegimeStep is not None:
    step_registry.register("enhanced_xgb_meso_regime_step", EnhancedXGBMesoRegimeStep)

# Register Enhanced Specialists
if EnhancedMLMomentumPersistenceStep is not None:
    step_registry.register("enhanced_ml_momentum_persistence_step", EnhancedMLMomentumPersistenceStep)
if EnhancedMLSMCRegimeStep is not None:
    step_registry.register("enhanced_ml_smc_regime_step", EnhancedMLSMCRegimeStep)
if EnhancedMLVolatilityBurstStep is not None:
    step_registry.register("enhanced_ml_volatility_burst_step", EnhancedMLVolatilityBurstStep)
if EnhancedMLVolumeForceStep is not None:
    step_registry.register("enhanced_ml_volume_force_step", EnhancedMLVolumeForceStep)
if EnhancedMLBreakoutBounceRegimeStep is not None:
    step_registry.register("enhanced_ml_breakout_bounce_regime_step", EnhancedMLBreakoutBounceRegimeStep)
if EnhancedMLReversionRegimeStep is not None:
    step_registry.register("enhanced_ml_reversion_regime_step", EnhancedMLReversionRegimeStep)
if EnhancedXGBMacroRegimeStep is not None:
    step_registry.register("enhanced_xgb_macro_regime_step", EnhancedXGBMacroRegimeStep)
if EnhancedMLLiquidityRegimeStep is not None:
    step_registry.register("enhanced_ml_liquidity_regime_step", EnhancedMLLiquidityRegimeStep)
if EnhancedMLPathRegimeStep is not None:
    step_registry.register("enhanced_ml_path_regime_step", EnhancedMLPathRegimeStep)
if EnhancedMLRiskRegimeStep is not None:
    step_registry.register("enhanced_ml_risk_regime_step", EnhancedMLRiskRegimeStep)
if EnhancedXGBMesoRegimeStep is not None:
    step_registry.register("enhanced_xgb_meso_regime_step", EnhancedXGBMesoRegimeStep)


# Register New Orthogonal Specialists
if EnhancedMLMicrostructureStep is not None:
    step_registry.register("enhanced_ml_microstructure_step", EnhancedMLMicrostructureStep)
if EnhancedMLCandlestickStep is not None:
    step_registry.register("enhanced_ml_candlestick_step", EnhancedMLCandlestickStep)
if EnhancedMLSpectralStep is not None:
    step_registry.register("enhanced_ml_spectral_step", EnhancedMLSpectralStep)
