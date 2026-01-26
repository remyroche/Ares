"""
Market Analysis Steps Module.

This module registers all market analysis steps for autonomous execution.
"""

from src.training.steps.base_step import step_registry
from .regime_ensemble_training_step import RegimeEnsembleTrainingStep
from .regime_feature_selector import EnhancedRegimeFeatureSelector


# Enhanced Specialists
try:
    from .ml_momentum_persistence_step_enhanced import EnhancedMLMomentumPersistenceStep
except Exception:
    EnhancedMLMomentumPersistenceStep = None

try:
    from .ml_smc_regime_step_enhanced import EnhancedMLSMCRegimeStep
except Exception:
    EnhancedMLSMCRegimeStep = None

try:
    from .ml_volatility_burst_step_enhanced import EnhancedMLVolatilityBurstStep
except Exception:
    EnhancedMLVolatilityBurstStep = None

try:
    from .ml_volume_force_step_enhanced import EnhancedMLVolumeForceStep
except Exception:
    EnhancedMLVolumeForceStep = None

try:
    from .ml_reversion_regime_step_enhanced import EnhancedMLReversionRegimeStep
except Exception:
    EnhancedMLReversionRegimeStep = None

try:
    from .xgb_macro_regime_step_enhanced import EnhancedXGBMacroRegimeStep
except Exception:
    EnhancedXGBMacroRegimeStep = None

try:
    from .ml_liquidity_regime_step_enhanced import EnhancedMLLiquidityRegimeStep
except Exception:
    EnhancedMLLiquidityRegimeStep = None

try:
    from .ml_path_regime_step_enhanced import EnhancedMLPathRegimeStep
except Exception:
    EnhancedMLPathRegimeStep = None

try:
    from .ml_risk_regime_step_enhanced import EnhancedMLRiskRegimeStep
except Exception:
    EnhancedMLRiskRegimeStep = None

try:
    from .xgb_meso_regime_step_enhanced import EnhancedXGBMesoRegimeStep
except Exception:
    EnhancedXGBMesoRegimeStep = None

try:
    from .ml_microstructure_step_enhanced import EnhancedMLMicrostructureStep
except Exception:
    EnhancedMLMicrostructureStep = None

try:
    from .ml_candlestick_step_enhanced import EnhancedMLCandlestickStep
except Exception:
    EnhancedMLCandlestickStep = None

try:
    from .ml_spectral_step_enhanced import EnhancedMLSpectralStep
except Exception:
    EnhancedMLSpectralStep = None

try:
    from .gmm_based_features import GMMFeaturePipeline
except Exception:
    GMMFeaturePipeline = None

# Registry helper
def register_if_exists(name, step_class):
    if step_class is not None:
        step_registry.register(name, step_class)


# Enhanced Specialist registration
register_if_exists("enhanced_ml_momentum_persistence_step", EnhancedMLMomentumPersistenceStep)
register_if_exists("enhanced_ml_smc_regime_step", EnhancedMLSMCRegimeStep)
register_if_exists("enhanced_ml_volatility_burst_step", EnhancedMLVolatilityBurstStep)
register_if_exists("enhanced_ml_volume_force_step", EnhancedMLVolumeForceStep)
register_if_exists("enhanced_ml_reversion_regime_step", EnhancedMLReversionRegimeStep)
register_if_exists("enhanced_xgb_macro_regime_step", EnhancedXGBMacroRegimeStep)
register_if_exists("enhanced_ml_liquidity_regime_step", EnhancedMLLiquidityRegimeStep)
register_if_exists("enhanced_ml_path_regime_step", EnhancedMLPathRegimeStep)
register_if_exists("enhanced_ml_risk_regime_step", EnhancedMLRiskRegimeStep)
register_if_exists("enhanced_xgb_meso_regime_step", EnhancedXGBMesoRegimeStep)
register_if_exists("enhanced_ml_microstructure_step", EnhancedMLMicrostructureStep)
register_if_exists("enhanced_ml_candlestick_step", EnhancedMLCandlestickStep)
register_if_exists("enhanced_ml_spectral_step", EnhancedMLSpectralStep)
register_if_exists("gmm_based_features", GMMFeaturePipeline)

# Import and register labeling steps (ensuring they are available)
try:
    from src.training.steps.labeling import FeatureGenerationMetaLabelingStep, MetaLabelingHPOExperimentStep
    register_if_exists("feature_generation_meta_labeling_step", FeatureGenerationMetaLabelingStep)
    register_if_exists("meta_labeling_hpo_experiment", MetaLabelingHPOExperimentStep)
except Exception:
    pass
