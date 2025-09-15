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
from .hmm_models_training import HMMModelsTrainingComponent
from .hmm_ensemble_training import HMMEnsembleTrainingComponent
from .regime_data_splitting import RegimeDataSplittingComponent
# TripleBarrierLabelingComponent moved to triple_barrier_labeling package
from .feature_lookback_optimization import FeatureLookbackOptimizationComponent
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent


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
        'hmm_models_training': HMMModelsTrainingComponent,
        'hmm_ensemble_training': HMMEnsembleTrainingComponent,
        'regime_data_splitting': RegimeDataSplittingComponent,
        # 'triple_barrier_labeling': TripleBarrierLabelingComponent,  # Moved to triple_barrier_labeling package
        'feature_lookback_optimization': FeatureLookbackOptimizationComponent,
        'cross_timeframe_analysis': CrossTimeframeAnalysisComponent
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
        if component_name not in self._components:
            available_components = list(self._components.keys())
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