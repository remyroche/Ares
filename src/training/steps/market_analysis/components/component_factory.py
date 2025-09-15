"""
Component Factory for Market Analysis Pipeline Components.

This factory manages the creation and registration of all pipeline components.
"""

from typing import Dict, Type, Any, Optional
from .base_component import BaseMarketAnalysisComponent, ComponentConfig
from .sr_parameter_optimization import SRParameterOptimizationComponent
from .sr_detection import SRDetectionComponent
from .sr_clustering import SRClusteringComponent
from .hmm_regime_discovery import HMMRegimeDiscoveryComponent
from .hmm_clustering import HMMClusteringComponent
# HMM training components moved to hmm_models_training module
# from .hmm_models_training import HMMModelsTrainingComponent
# from .hmm_ensemble_training import HMMEnsembleTrainingComponent
# RegimeDataSplittingComponent imported lazily to avoid circular imports
# TripleBarrierLabelingComponent moved to triple_barrier_labeling package
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent  # Now uses PID-based feature generation
# Import the actual PID-based component for direct use
try:
    from ..pid_based_feature_generation.pid_based_feature_generation_component import PIDBasedFeatureGenerationComponent
    PID_COMPONENT_AVAILABLE = True
except ImportError:
    PID_COMPONENT_AVAILABLE = False


class ComponentFactory:
    """
    Factory for creating market analysis pipeline components.
    
    Provides centralized component creation and management.
    """
    
    _components: Dict[str, Type[BaseMarketAnalysisComponent]] = {
        'sr_parameter_optimization': SRParameterOptimizationComponent,
        'sr_detection': SRDetectionComponent,
        'sr_clustering': SRClusteringComponent,
        'hmm_regime_discovery': HMMRegimeDiscoveryComponent,
        'hmm_clustering': HMMClusteringComponent,
        # 'hmm_models_training': HMMModelsTrainingComponent,  # Moved to hmm_models_training module
        # 'hmm_ensemble_training': HMMEnsembleTrainingComponent,  # Removed
        # 'regime_data_splitting': RegimeDataSplittingComponent,  # Imported lazily to avoid circular imports
        # 'triple_barrier_labeling': TripleBarrierLabelingComponent,  # Moved to triple_barrier_labeling package
        'feature_lookback_optimization': FeatureLookbackOptimizationComponent,
        'cross_timeframe_analysis': CrossTimeframeAnalysisComponent,  # Now uses PID-based feature generation
        'pid_based_feature_generation': PIDBasedFeatureGenerationComponent if PID_COMPONENT_AVAILABLE else CrossTimeframeAnalysisComponent  # Direct PID component or fallback
    }
    
    @classmethod
    def create_component(
        self, 
        component_name: str, 
        config: Optional[ComponentConfig] = None
    ) -> BaseMarketAnalysisComponent:
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
        # Handle lazy imports for components that might cause circular imports
        if component_name == 'regime_data_splitting':
            try:
                from .regime_data_splitting import RegimeDataSplittingComponent
                return RegimeDataSplittingComponent(config)
            except ImportError as e:
                raise ValueError(f"Failed to import RegimeDataSplittingComponent: {e}")
        
        if component_name not in self._components:
            available_components = list(self._components.keys()) + ['regime_data_splitting']
            raise ValueError(
                f"Unknown component: {component_name}. "
                f"Available components: {available_components}"
            )
        
        component_class = self._components[component_name]
        return component_class(config)
    
    @classmethod
    def register_component(
        self, 
        name: str, 
        component_class: Type[BaseMarketAnalysisComponent]
    ) -> None:
        """
        Register a new component.
        
        Args:
            name: Component name
            component_class: Component class
        """
        if not issubclass(component_class, BaseMarketAnalysisComponent):
            raise ValueError(
                f"Component class must inherit from BaseMarketAnalysisComponent"
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
        return component_name in self._components