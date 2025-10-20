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
    FeatureGenerationFeatureGenerationStep,
    FeatureGenerationResult,
    handle_feature_generation_feature_generation_step
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

from .feature_generation_interaction_generation_step import (
    FeatureGenerationInteractionGenerationStep,
    InteractionGenerationResult,
    handle_feature_generation_interaction_generation_step
)

from .feature_generation_vectorization_step import (
    FeatureGenerationVectorizationStep,
    VectorizationResult,
    handle_feature_generation_vectorization_step
)

from .feature_generation_labeling_integration_step import (
    FeatureGenerationLabelingIntegrationStep,
    LabelingIntegrationResult,
    handle_feature_generation_labeling_integration_step
)

from .feature_generation_period_lookback_optimization_step import (
    PeriodLookbackOptimizationStep,
)
from .feature_generation_final_validation_step import (
    FeatureGenerationFinalValidationStep,
    FinalValidationResult,
    handle_feature_generation_final_validation_step
)

# Export all step classes and handlers
__all__ = [
    # Step classes
    "FeatureGenerationDataValidationStep",
    "FeatureGenerationFeatureGenerationStep", 
    "FeatureGenerationFeatureSelectionStep",
    "FeatureGenerationPeriodOptimizationStep",
    "FeatureGenerationLookbackOptimizationStep",
    "FeatureGenerationInteractionGenerationStep",
    "FeatureGenerationVectorizationStep",
    "FeatureGenerationLabelingIntegrationStep",
    "FeatureGenerationFinalValidationStep",
    "PeriodLookbackOptimizationStep",
    
    # Result classes
    "DataValidationResult",
    "FeatureGenerationResult",
    "FeatureSelectionResult",
    "PeriodOptimizationResult",
    "LookbackOptimizationResult",
    "InteractionGenerationResult",
    "VectorizationResult",
    "LabelingIntegrationResult",
    "FinalValidationResult",
    
    # Command handlers
    "handle_feature_generation_data_validation_step",
    "handle_feature_generation_feature_generation_step",
    "handle_feature_generation_feature_selection_step",
    "handle_feature_generation_period_optimization_step",
    "handle_feature_generation_lookback_optimization_step",
    "handle_feature_generation_interaction_generation_step",
    "handle_feature_generation_vectorization_step",
    "handle_feature_generation_labeling_integration_step",
    "handle_feature_generation_final_validation_step"
]

# Step execution order
STEP_EXECUTION_ORDER = [
    "feature_generation_data_validation_step",
    "feature_generation_labeling_integration_step",  # Moved to step 2
    "feature_generation_feature_generation_step",
    "feature_generation_feature_selection_step",
    "feature_generation_period_lookback_optimization_step",  # Merged period + lookback optimization
    "feature_generation_interaction_generation_step",
    "feature_generation_vectorization_step",
    "feature_generation_final_validation_step"
]

# Step descriptions
STEP_DESCRIPTIONS = {
    "feature_generation_data_validation_step": "Data validation and quality assessment",
    "feature_generation_labeling_integration_step": "Analyst/Tactician labeling integration (moved to step 2)",
    "feature_generation_feature_generation_step": "Feature Bank only feature generation",
    "feature_generation_feature_selection_step": "Intelligent feature selection (5-50 features)",
    "feature_generation_period_lookback_optimization_step": "Concurrent period + lookback optimization (min 2 periods per feature, no recency bias)",
    "feature_generation_interaction_generation_step": "Feature interaction generation (X² max, log relationships, ML generators, max 100 features)",
    "feature_generation_vectorization_step": "Feature vectorization optimization",
    "feature_generation_final_validation_step": "Final validation and quality check"
}