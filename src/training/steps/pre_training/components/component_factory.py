"""
Component Factory for Pre-Training Pipeline Components.

This factory manages the creation and registration of all pre-training pipeline components.
"""

from typing import Dict, Type, Any, Optional
from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
)

# Import base component classes
from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult

# Import pre-training components
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .final_feature_selection import FinalFeatureSelectionComponent

# Import PID-based feature generation component
try:
    from ..pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
    PID_COMPONENT_AVAILABLE = True
    tprint_debug("✅ PID-based feature generation component loaded successfully")
except ImportError as e:
    PID_COMPONENT_AVAILABLE = False
    tprint_warning(f"⚠️ PID-based feature generation component not available: {e}")
    tprint_info("ℹ️ Some advanced feature generation capabilities will be disabled")

# Import optimized lookback component
try:
    from ..interaction_feature_generator.feature_interaction_generation.optimized_lookback_component import OptimizedLookbackComponent
    OPTIMIZED_LOOKBACK_AVAILABLE = True
    tprint_debug("✅ Optimized lookback component loaded successfully")
except ImportError as e:
    OPTIMIZED_LOOKBACK_AVAILABLE = False
    tprint_warning(f"⚠️ Optimized lookback component not available: {e}")

# Import multi-horizon profit labeler
try:
    from ..multi_horizon_profit_labeler import MultiHorizonProfitLabelerComponent
    MULTI_HORIZON_AVAILABLE = True
    tprint_debug("✅ Multi-horizon profit labeler component loaded successfully")
except ImportError as e:
    MULTI_HORIZON_AVAILABLE = False
    tprint_error(f"❌ Multi-horizon profit labeler component not available: {e}")
    tprint_error("❌ This is a CRITICAL component - pipeline may not function correctly")


class MultiHorizonComponentWrapper(BasePreTrainingComponent):
    """Wrapper for Multi-Horizon Profit Labeler to work as a component."""

    def __init__(self, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.component = None
        tprint(
            "🧩 [MULTI_HORIZON_WRAPPER] Initialized wrapper for multi-horizon component",
            color="blue",
        )

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        tprint(
            "📦 [MULTI_HORIZON_WRAPPER] Retrieving required artifacts",
            color="magenta",
        )
        return ['multi_horizon_labeling_result', 'labeling_report']

    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute multi-horizon labeling as a component."""
        try:
            tprint(
                "🚀 [MULTI_HORIZON_WRAPPER] Executing multi-horizon profit labeler",
                color="cyan",
            )
            # Create component instance if not exists
            if self.component is None:
                if not MULTI_HORIZON_AVAILABLE:
                    tprint(
                        "❌ [MULTI_HORIZON_WRAPPER] Multi-horizon profit labeler not available",
                        color="red",
                    )
                    raise ImportError("Multi-horizon profit labeler not available")

                self.component = MultiHorizonProfitLabelerComponent(self.config)
                tprint(
                    "🛠️ [MULTI_HORIZON_WRAPPER] Instantiated multi-horizon component",
                    color="yellow",
                )

            # Execute component
            result = await self.component.execute(data, pipeline_state)
            tprint(
                "✅ [MULTI_HORIZON_WRAPPER] Execution completed successfully",
                color="green",
            )
            return result

        except Exception as e:
            tprint(
                f"⚠️ [MULTI_HORIZON_WRAPPER] Execution failed: {e}",
                color="red",
            )
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
        'optimized_lookback_generation': OptimizedLookbackComponent if OPTIMIZED_LOOKBACK_AVAILABLE else None,
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
            raise ValueError("Component class must inherit from BasePreTrainingComponent")

        self._components[name] = component_class
        tprint(
            f"🧾 [PRE_TRAINING_FACTORY] Registered component: {name}",
            color="blue",
        )

    @classmethod
    def get_available_components(self) -> list[str]:
        """
        Get list of available component names.

        Returns:
            List of component names
        """
        available = list(self._components.keys())
        tprint(
            f"📋 [PRE_TRAINING_FACTORY] Available components: {available}",
            color="magenta",
        )
        return available

    @classmethod
    def is_component_available(self, component_name: str) -> bool:
        """
        Check if a component is available.
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if component is available
        """
        available = component_name in self._components and self._components[component_name] is not None
        tprint(
            f"🔍 [PRE_TRAINING_FACTORY] Component '{component_name}' available: {available}",
            color="yellow",
        )
        return available
