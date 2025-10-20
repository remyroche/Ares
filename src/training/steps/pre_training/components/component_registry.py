"""
Component Registry for Pre-Training Pipeline

This module automatically registers all pre-training components with the ComponentFactory.
"""

from typing import Dict, Type, Any, Optional
from .component_factory import BaseComponent, ComponentConfig

# Import all the step components
try:
    from src.training.steps.pre_training.analyst_profit_labeler import AnalystProfitLabelerComponent
    ANALYST_PROFIT_LABELER_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import AnalystProfitLabelerComponent: {e}")
    ANALYST_PROFIT_LABELER_AVAILABLE = False
    AnalystProfitLabelerComponent = None

try:
    from src.training.steps.pre_training.tactician_entry_labeler import TacticianEntryLabelerComponent
    TACTICIAN_ENTRY_LABELER_AVAILABLE = True
except ImportError:
    TACTICIAN_ENTRY_LABELER_AVAILABLE = False
    TacticianEntryLabelerComponent = None

# Feature Generation Steps
try:
    from src.training.steps.pre_training.feature_generation_period_lookback_optimization_step import FeatureGenerationPeriodLookbackOptimizationStep
    FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FeatureGenerationPeriodLookbackOptimizationStep: {e}")
    FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE = False
    FeatureGenerationPeriodLookbackOptimizationStep = None

try:
    from src.training.steps.pre_training.feature_generation_feature_selection_step import FeatureGenerationFeatureSelectionStep
    FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FeatureGenerationFeatureSelectionStep: {e}")
    FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE = False
    FeatureGenerationFeatureSelectionStep = None

try:
    from src.training.steps.pre_training.feature_generation_interaction_generation_step_analyst import FeatureGenerationInteractionGenerationStepAnalyst
    FEATURE_GENERATION_INTERACTION_GENERATION_STEP_ANALYST_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FeatureGenerationInteractionGenerationStepAnalyst: {e}")
    FEATURE_GENERATION_INTERACTION_GENERATION_STEP_ANALYST_AVAILABLE = False
    FeatureGenerationInteractionGenerationStepAnalyst = None

try:
    from src.training.steps.pre_training.feature_generation_interaction_generation_step_tactician import FeatureGenerationInteractionGenerationStepTactician
    FEATURE_GENERATION_INTERACTION_GENERATION_STEP_TACTICIAN_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FeatureGenerationInteractionGenerationStepTactician: {e}")
    FEATURE_GENERATION_INTERACTION_GENERATION_STEP_TACTICIAN_AVAILABLE = False
    FeatureGenerationInteractionGenerationStepTactician = None

try:
    from src.training.steps.pre_training.feature_generation_final_feature_selection_step import FeatureGenerationFinalFeatureSelectionStep
    FEATURE_GENERATION_FINAL_FEATURE_SELECTION_STEP_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import FeatureGenerationFinalFeatureSelectionStep: {e}")
    FEATURE_GENERATION_FINAL_FEATURE_SELECTION_STEP_AVAILABLE = False
    FeatureGenerationFinalFeatureSelectionStep = None

# Legacy - keeping for backward compatibility
FEATURE_GENERATION_DATA_VALIDATION_STEP_AVAILABLE = False
FeatureGenerationDataValidationStep = None
FEATURE_GENERATION_FEATURE_GENERATION_STEP_AVAILABLE = False
FeatureGenerationStep = None
FEATURE_GENERATION_FINAL_VALIDATION_STEP_AVAILABLE = False
FeatureGenerationFinalValidationStep = None
FEATURE_GENERATION_LABELING_INTEGRATION_STEP_AVAILABLE = False
FeatureGenerationLabelingIntegrationStep = None



class ComponentRegistry:
    """Registry for pre-training pipeline components."""
    
    @staticmethod
    def register_all_components():
        """Register all available components with the ComponentFactory."""
        # Import ComponentFactory here to avoid circular imports
        from .component_factory import ComponentFactory
        
        # Register analyst profit labeler
        if ANALYST_PROFIT_LABELER_AVAILABLE and AnalystProfitLabelerComponent is not None:
            ComponentFactory.register_component('analyst_profit_labeler', AnalystProfitLabelerComponent)
        
        # Register tactician entry labeler
        if TACTICIAN_ENTRY_LABELER_AVAILABLE and TacticianEntryLabelerComponent is not None:
            ComponentFactory.register_component('tactician_entry_labeler', TacticianEntryLabelerComponent)
        
        # Register feature generation period lookback optimization step
        if FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE and FeatureGenerationPeriodLookbackOptimizationStep is not None:
            ComponentFactory.register_component('feature_generation_period_lookback_optimization_step', FeatureGenerationPeriodLookbackOptimizationStep)
        
        # Register feature generation feature selection step
        if FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE and FeatureGenerationFeatureSelectionStep is not None:
            ComponentFactory.register_component('feature_generation_feature_selection_step', FeatureGenerationFeatureSelectionStep)
        
        # Register feature generation interaction generation step - analyst
        if FEATURE_GENERATION_INTERACTION_GENERATION_STEP_ANALYST_AVAILABLE and FeatureGenerationInteractionGenerationStepAnalyst is not None:
            ComponentFactory.register_component('feature_generation_interaction_generation_step_analyst', FeatureGenerationInteractionGenerationStepAnalyst)
        
        # Register feature generation interaction generation step - tactician
        if FEATURE_GENERATION_INTERACTION_GENERATION_STEP_TACTICIAN_AVAILABLE and FeatureGenerationInteractionGenerationStepTactician is not None:
            ComponentFactory.register_component('feature_generation_interaction_generation_step_tactician', FeatureGenerationInteractionGenerationStepTactician)
        
        # Register feature generation final feature selection step
        if FEATURE_GENERATION_FINAL_FEATURE_SELECTION_STEP_AVAILABLE and FeatureGenerationFinalFeatureSelectionStep is not None:
            ComponentFactory.register_component('feature_generation_final_feature_selection_step', FeatureGenerationFinalFeatureSelectionStep)
        
    
    @staticmethod
    def get_registration_status() -> Dict[str, bool]:
        """Get the registration status of all components."""
        return {
            'analyst_profit_labeler': ANALYST_PROFIT_LABELER_AVAILABLE,
            'tactician_entry_labeler': TACTICIAN_ENTRY_LABELER_AVAILABLE,
            'feature_generation_period_lookback_optimization_step': FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE,
            'feature_generation_feature_selection_step': FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE,
            'feature_generation_interaction_generation_step_analyst': FEATURE_GENERATION_INTERACTION_GENERATION_STEP_ANALYST_AVAILABLE,
            'feature_generation_interaction_generation_step_tactician': FEATURE_GENERATION_INTERACTION_GENERATION_STEP_TACTICIAN_AVAILABLE,
            'feature_generation_final_feature_selection_step': FEATURE_GENERATION_FINAL_FEATURE_SELECTION_STEP_AVAILABLE,
            # Legacy components (kept for backward compatibility)
            'feature_generation_data_validation_step': FEATURE_GENERATION_DATA_VALIDATION_STEP_AVAILABLE,
            'feature_generation_feature_generation_step': FEATURE_GENERATION_FEATURE_GENERATION_STEP_AVAILABLE,
            'feature_generation_final_validation_step': FEATURE_GENERATION_FINAL_VALIDATION_STEP_AVAILABLE,
            'feature_generation_labeling_integration_step': FEATURE_GENERATION_LABELING_INTEGRATION_STEP_AVAILABLE,
        }


# Components will be registered manually when needed
