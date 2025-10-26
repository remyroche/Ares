"""
Pre-Training Steps Module

This module registers all pre-training feature generation steps for autonomous execution.
The pre-training pipeline includes:
- Gate feature protection
- Final feature selection
- Final validation

Note: Some earlier pipeline steps are referenced but not yet implemented as BaseStep classes.
"""

from src.training.steps.base_step import step_registry

# Import and register the implemented steps
try:
    from .feature_generation_gate_feature_step import FeatureGenerationGateFeatureStep
    GATE_FEATURE_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FeatureGenerationGateFeatureStep: {e}")
    GATE_FEATURE_STEP_AVAILABLE = False

try:
    from .feature_generation_final_feature_selection_step import FeatureGenerationFinalFeatureSelectionStep
    FINAL_FEATURE_SELECTION_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FeatureGenerationFinalFeatureSelectionStep: {e}")
    FINAL_FEATURE_SELECTION_STEP_AVAILABLE = False

try:
    from .feature_generation_final_validation_step import FeatureGenerationFinalValidationStep
    FINAL_VALIDATION_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import FeatureGenerationFinalValidationStep: {e}")
    FINAL_VALIDATION_STEP_AVAILABLE = False

# Import missing feature generation steps
from .feature_generation_data_validation_step import FeatureGenerationDataValidationStep
from .feature_generation_labeling_integration_step import FeatureGenerationLabelingIntegrationStep
from .feature_generation_feature_generation_step import FeatureGenerationFeatureGenerationStep
from .feature_generation_period_lookback_optimization_step import FeatureGenerationPeriodLookbackOptimizationStep
from .feature_generation_feature_selection_step import FeatureGenerationFeatureSelectionStep
from .feature_generation_interaction_generation_step import FeatureGenerationInteractionGenerationStep

# Register available steps with the global registry
step_registry.register('feature_generation_data_validation_step', FeatureGenerationDataValidationStep)
step_registry.register('feature_generation_labeling_integration_step', FeatureGenerationLabelingIntegrationStep)
step_registry.register('feature_generation_feature_generation_step', FeatureGenerationFeatureGenerationStep)
step_registry.register('feature_generation_period_lookback_optimization_step', FeatureGenerationPeriodLookbackOptimizationStep)
step_registry.register('feature_generation_feature_selection_step', FeatureGenerationFeatureSelectionStep)
step_registry.register('feature_generation_interaction_generation_step', FeatureGenerationInteractionGenerationStep)

if GATE_FEATURE_STEP_AVAILABLE:
    step_registry.register('feature_generation_gate_feature_step', FeatureGenerationGateFeatureStep)

if FINAL_FEATURE_SELECTION_STEP_AVAILABLE:
    step_registry.register('feature_generation_final_feature_selection_step', FeatureGenerationFinalFeatureSelectionStep)

if FINAL_VALIDATION_STEP_AVAILABLE:
    step_registry.register('feature_generation_final_validation_step', FeatureGenerationFinalValidationStep)

# Register component-based steps (legacy compatibility)
# Note: These components exist but are not BaseStep implementations
try:
    from .components.component_registry import ComponentRegistry
    # The component registry handles component registration separately
    print("Pre-training components registry available")
except ImportError as e:
    print(f"Warning: Could not import component registry: {e}")

__all__ = [
    'FeatureGenerationGateFeatureStep' if GATE_FEATURE_STEP_AVAILABLE else None,
    'FeatureGenerationFinalFeatureSelectionStep' if FINAL_FEATURE_SELECTION_STEP_AVAILABLE else None,
    'FeatureGenerationFinalValidationStep' if FINAL_VALIDATION_STEP_AVAILABLE else None
]
