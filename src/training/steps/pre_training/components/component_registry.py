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

# REMOVED - unified_data_driven_pipeline deleted
FEATURE_GENERATION_DATA_VALIDATION_STEP_AVAILABLE = False
FeatureGenerationDataValidationStep = None
FEATURE_GENERATION_FEATURE_GENERATION_STEP_AVAILABLE = False
FeatureGenerationStep = None
FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE = False
FeatureGenerationFeatureSelectionStep = None
FEATURE_GENERATION_FINAL_VALIDATION_STEP_AVAILABLE = False
FeatureGenerationFinalValidationStep = None

# REMOVED - unified_data_driven_pipeline deleted
FEATURE_GENERATION_INTERACTION_GENERATION_STEP_AVAILABLE = False
FeatureGenerationInteractionGenerationStep = None
FEATURE_GENERATION_LABELING_INTEGRATION_STEP_AVAILABLE = False
FeatureGenerationLabelingIntegrationStep = None
FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE = False
FeatureGenerationPeriodLookbackOptimizationStep = None



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
        
        # Register feature generation data validation step
        if FEATURE_GENERATION_DATA_VALIDATION_STEP_AVAILABLE and FeatureGenerationDataValidationStep is not None:
            ComponentFactory.register_component('feature_generation_data_validation_step', FeatureGenerationDataValidationStep)
        
        # Register feature generation feature generation step
        if FEATURE_GENERATION_FEATURE_GENERATION_STEP_AVAILABLE and FeatureGenerationStep is not None:
            ComponentFactory.register_component('feature_generation_feature_generation_step', FeatureGenerationStep)
        
        # Register feature generation feature selection step
        if FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE and FeatureGenerationFeatureSelectionStep is not None:
            ComponentFactory.register_component('feature_generation_feature_selection_step', FeatureGenerationFeatureSelectionStep)
        
        # Register feature generation final validation step
        if FEATURE_GENERATION_FINAL_VALIDATION_STEP_AVAILABLE and FeatureGenerationFinalValidationStep is not None:
            ComponentFactory.register_component('feature_generation_final_validation_step', FeatureGenerationFinalValidationStep)
        
        # Register feature generation interaction generation step
        if FEATURE_GENERATION_INTERACTION_GENERATION_STEP_AVAILABLE and FeatureGenerationInteractionGenerationStep is not None:
            ComponentFactory.register_component('feature_generation_interaction_generation_step', FeatureGenerationInteractionGenerationStep)
        
        # Register feature generation labeling integration step
        if FEATURE_GENERATION_LABELING_INTEGRATION_STEP_AVAILABLE and FeatureGenerationLabelingIntegrationStep is not None:
            ComponentFactory.register_component('feature_generation_labeling_integration_step', FeatureGenerationLabelingIntegrationStep)
        
        # Register feature generation period lookback optimization step
        if FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE and FeatureGenerationPeriodLookbackOptimizationStep is not None:
            ComponentFactory.register_component('feature_generation_period_lookback_optimization', FeatureGenerationPeriodLookbackOptimizationStep)
        
    
    @staticmethod
    def get_registration_status() -> Dict[str, bool]:
        """Get the registration status of all components."""
        return {
            'analyst_profit_labeler': ANALYST_PROFIT_LABELER_AVAILABLE,
            'tactician_entry_labeler': TACTICIAN_ENTRY_LABELER_AVAILABLE,
            'feature_generation_data_validation_step': FEATURE_GENERATION_DATA_VALIDATION_STEP_AVAILABLE,
            'feature_generation_feature_generation_step': FEATURE_GENERATION_FEATURE_GENERATION_STEP_AVAILABLE,
            'feature_generation_feature_selection_step': FEATURE_GENERATION_FEATURE_SELECTION_STEP_AVAILABLE,
            'feature_generation_final_validation_step': FEATURE_GENERATION_FINAL_VALIDATION_STEP_AVAILABLE,
            'feature_generation_interaction_generation_step': FEATURE_GENERATION_INTERACTION_GENERATION_STEP_AVAILABLE,
            'feature_generation_labeling_integration_step': FEATURE_GENERATION_LABELING_INTEGRATION_STEP_AVAILABLE,
            'feature_generation_period_lookback_optimization': FEATURE_GENERATION_PERIOD_LOOKBACK_OPTIMIZATION_STEP_AVAILABLE,
        }


# Components will be registered manually when needed
