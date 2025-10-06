"""
Component Factory for Pre-Training Pipeline Components.

This factory manages the creation and registration of all pre-training pipeline components.
"""

import numpy as np
import pandas as pd
from typing import Dict, Type, Any, Optional, List
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)

# Import base component classes
from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult

# Import pre-training components
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .final_feature_selection import FinalFeatureSelectionComponent

# Import PID-based feature generation component
try:
    from ..pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
    PID_COMPONENT_AVAILABLE = True
except ImportError:
    PID_COMPONENT_AVAILABLE = False

# Import multi-horizon profit labeler
try:
    from ..multi_horizon_profit_labeler import MultiHorizonProfitLabelerComponent, MultiHorizonConfig
    MULTI_HORIZON_AVAILABLE = True
except ImportError:
    MULTI_HORIZON_AVAILABLE = False


class MultiHorizonComponentWrapper(BasePreTrainingComponent):
    """Wrapper for Multi-Horizon Profit Labeler to work as a component."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.component = None

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']

    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute multi-horizon labeling as a component."""
        try:
            # Create component instance if not exists
            if self.component is None:
                if not MULTI_HORIZON_AVAILABLE:
                    raise ImportError("Multi-horizon profit labeler not available")

                self.component = MultiHorizonProfitLabelerComponent(self.config)

            # Execute component
            return await self.component.execute(data, pipeline_state)

        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'multi_horizon_profit_labeler'}
            )


class ComponentFactory:
    """
    Factory for creating pre-training pipeline components.
    
    Provides centralized component creation and management.
    """
    
    _components: Dict[str, Type[BasePreTrainingComponent]] = {
        'multi_horizon_profit_labeler': MultiHorizonComponentWrapper if MULTI_HORIZON_AVAILABLE else None,
        'feature_lookback_optimization': FeatureLookbackOptimizationComponent,
        'pid_based_feature_generation': PIDBasedFeatureGenerationComponent if PID_COMPONENT_AVAILABLE else None,
        'final_feature_selection': FinalFeatureSelectionComponent
    }
    
    @classmethod
    def create_component(
        self, 
        component_name: str, 
        config: Optional[ComponentConfig] = None
    ) -> BasePreTrainingComponent:
        """
        Create a component instance.
        
        Args:
            component_name: Name of the component to create
            config: Component configuration
            
        Returns:
            Component instance
            
        Raises:
            ValueError: If component name is not registered
        """
        tprint(f"🏭 [PRE_TRAINING_FACTORY] Creating component: {component_name}", color="cyan")
        
        if component_name not in self._components:
            available_components = list(self._components.keys())
            tprint(f"❌ [PRE_TRAINING_FACTORY] Unknown component: {component_name}", color="red")
            tprint(f"📊 [PRE_TRAINING_FACTORY] Available components: {available_components}", color="cyan")
            raise ValueError(
                f"Unknown component: {component_name}. "
                f"Available components: {available_components}"
            )

        tprint(f"🔧 [PRE_TRAINING_FACTORY] Creating {component_name} from registered components", color="yellow")
        component_class = self._components[component_name]

        # Handle None component classes
        if component_class is None:
            tprint(f"❌ [PRE_TRAINING_FACTORY] Component {component_name} is not available", color="red")
            raise ValueError(f"Component {component_name} is not available. Required dependencies may be missing.")

        component = component_class(config)
        tprint(f"✅ [PRE_TRAINING_FACTORY] Successfully created {component_name}", color="green")
        return component
    
    @classmethod
    def register_component(
        self, 
        name: str, 
        component_class: Type[BasePreTrainingComponent]
    ) -> None:
        """
        Register a new component.
        
        Args:
            name: Component name
            component_class: Component class
        """
        if not issubclass(component_class, BasePreTrainingComponent):
            raise ValueError(
                f"Component class must inherit from BasePreTrainingComponent"
            )
        
        self._components[name] = component_class
    
    @classmethod
    def get_available_components(self) -> list[str]:
        """
        Get list of available component names.
        
        Returns:
            List of component names
        """
        return list(self._components.keys())
    
    @classmethod
    def is_component_available(self, component_name: str) -> bool:
        """
        Check if a component is available.
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if component is available
        """
        return component_name in self._components and self._components[component_name] is not None