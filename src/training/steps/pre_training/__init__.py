"""
Feature Generation Steps

This module contains all the individual steps for the unified data-driven pipeline
with the feature_generation_ prefix for better organization and ares_launcher integration.
"""

from .feature_generation_data_validation_step import (
    FeatureGenerationDataValidationStep,
    DataValidationResult,
    handle_feature_generation_data_validation_step
)

from .feature_generation_feature_generation_step import (
    FeatureGenerationStep,
    FeatureGenerationResult,
    handle_feature_generation_step
)

from .feature_generation_feature_selection_step import (
    FeatureGenerationFeatureSelectionStep,
    FeatureSelectionResult,
    handle_feature_generation_feature_selection_step
)

from .feature_generation_period_optimization_step import (
    FeatureGenerationPeriodOptimizationStep,
    PeriodOptimizationResult,
    handle_feature_generation_period_optimization_step
)

from .feature_generation_lookback_optimization_step import (
    FeatureGenerationLookbackOptimizationStep,
    LookbackOptimizationResult,
    handle_feature_generation_lookback_optimization_step
)

from .feature_generation_interaction_generation_step_analyst import (
    FeatureGenerationInteractionGenerationStepAnalyst,
    InteractionGenerationResult,
    handle_feature_generation_interaction_generation_step_analyst
)

from .feature_generation_interaction_generation_step_tactician import (
    FeatureGenerationInteractionGenerationStepTactician,
    handle_feature_generation_interaction_generation_step_tactician
)


from .feature_generation_labeling_integration_step import (
    FeatureGenerationLabelingIntegrationStep,
    LabelingIntegrationResult,
    handle_feature_generation_labeling_integration_step
)

from .feature_generation_final_feature_selection_step import (
    FeatureGenerationFinalFeatureSelectionStep,
    FinalFeatureSelectionResult,
    handle_feature_generation_final_feature_selection_step
)

from .feature_generation_final_validation_step import (
    FeatureGenerationFinalValidationStep,
    FinalValidationResult,
    handle_feature_generation_final_validation_step
)

from .feature_generation_period_lookback_optimization_step import (
    FeatureGenerationPeriodLookbackOptimizationStep
)

# Import step registry for registration
from src.training.steps.base_step import step_registry

# Register all pre_training steps with the step registry
step_registry.register("feature_generation_data_validation_step", FeatureGenerationDataValidationStep)
step_registry.register("feature_generation_feature_generation_step", FeatureGenerationStep)
step_registry.register("feature_generation_feature_selection_step", FeatureGenerationFeatureSelectionStep)
step_registry.register("feature_generation_period_optimization_step", FeatureGenerationPeriodOptimizationStep)
step_registry.register("feature_generation_lookback_optimization_step", FeatureGenerationLookbackOptimizationStep)
step_registry.register("feature_generation_period_lookback_optimization_step", FeatureGenerationPeriodLookbackOptimizationStep)
step_registry.register("feature_generation_interaction_generation_step_analyst", FeatureGenerationInteractionGenerationStepAnalyst)
step_registry.register("feature_generation_interaction_generation_step_tactician", FeatureGenerationInteractionGenerationStepTactician)
step_registry.register("feature_generation_labeling_integration_step", FeatureGenerationLabelingIntegrationStep)
step_registry.register("feature_generation_final_feature_selection_step", FeatureGenerationFinalFeatureSelectionStep)
step_registry.register("feature_generation_final_validation_step", FeatureGenerationFinalValidationStep)

# Export all step classes and handlers
__all__ = [
    # Step classes
    "FeatureGenerationDataValidationStep",
    "FeatureGenerationStep",
    "FeatureGenerationFeatureSelectionStep",
    "FeatureGenerationPeriodOptimizationStep",
    "FeatureGenerationLookbackOptimizationStep",
    "FeatureGenerationPeriodLookbackOptimizationStep",
    "FeatureGenerationInteractionGenerationStepAnalyst",
    "FeatureGenerationInteractionGenerationStepTactician",
    "FeatureGenerationLabelingIntegrationStep",
    "FeatureGenerationFinalFeatureSelectionStep",
    "FeatureGenerationFinalValidationStep",

    # Result classes
    "DataValidationResult",
    "FeatureGenerationResult",
    "FeatureSelectionResult",
    "PeriodOptimizationResult",
    "LookbackOptimizationResult",
    "InteractionGenerationResult",
    "LabelingIntegrationResult",
    "FinalFeatureSelectionResult",
    "FinalValidationResult",

    # Command handlers
    "handle_feature_generation_data_validation_step",
    "handle_feature_generation_step",
    "handle_feature_generation_feature_selection_step",
    "handle_feature_generation_period_optimization_step",
    "handle_feature_generation_lookback_optimization_step",
    "handle_feature_generation_interaction_generation_step_analyst",
    "handle_feature_generation_interaction_generation_step_tactician",
    "handle_feature_generation_labeling_integration_step",
    "handle_feature_generation_final_feature_selection_step",
    "handle_feature_generation_final_validation_step"
]

# Step execution order
STEP_EXECUTION_ORDER = [
    "feature_generation_data_validation_step",
    "feature_generation_labeling_integration_step",  # Moved to step 2
    "feature_generation_feature_generation_step",
    "feature_generation_feature_selection_step",
    "feature_generation_period_lookback_optimization_step",  # Merged period + lookback optimization
    "feature_generation_interaction_generation_step_analyst",
    "feature_generation_interaction_generation_step_tactician",
    "feature_generation_final_feature_selection_step",
    "feature_generation_final_validation_step"
]

# Step descriptions
STEP_DESCRIPTIONS = {
    "feature_generation_data_validation_step": "Data validation and quality assessment",
    "feature_generation_labeling_integration_step": "Independent labeling integration (moved to step 2)",
    "feature_generation_feature_generation_step": "Feature Bank generation (16 categories: momentum, volatility, trend, volume, support/resistance, returns, oscillator, candlestick, entropy, order_flow, acceleration, cross_timeframe, interaction, microstructure, advanced_statistical, spectral_wavelet)",
    "feature_generation_feature_selection_step": "Intelligent feature selection (4-45 features, -10% early pruning)",
    "feature_generation_period_lookback_optimization_step": "Concurrent period + lookback optimization (min 2 periods per feature, no recency bias)",
    "feature_generation_interaction_generation_step_analyst": "Analyst mode: Three-phase LGBM+SHAP interaction generation (variant generation, refinement, deep discovery)",
    "feature_generation_interaction_generation_step_tactician": "Tactician mode: Original interaction generation with CMI complementarity filtering",
    "feature_generation_final_feature_selection_step": "Target-aware feature selection with downstream pipeline: PCA → MI → mRMR → LASSO+Stability → LGBM+RFE+SHAP",
    "feature_generation_final_validation_step": "Final validation and quality check"
}
