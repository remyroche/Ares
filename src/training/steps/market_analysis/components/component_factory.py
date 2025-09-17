"""
Component Factory for Market Analysis Pipeline Components.

This factory manages the creation and registration of all pipeline components.
"""

from typing import Dict, Type, Any, Optional
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
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


class HMMModelsTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for HMM Models Training Enhanced to work as a component."""
    
    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.training_class = training_class
        self.training_instance = None
    
    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_models_training_result']
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM models training as a component."""
        try:
            # Create training instance if not exists
            if self.training_instance is None:
                self.training_instance = self.training_class()
            
            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            
            # If cluster_assignments is missing, try to get from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    print(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")
            
            # If we don't have features/targets, try to extract from dataframe
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:
                    import pandas as pd
                    import numpy as np
                    
                    # Create basic features and targets from OHLCV data
                    if 'close' in dataframe.columns:
                        # Simple features: returns, volatility, etc.
                        returns = dataframe['close'].pct_change().fillna(0)
                        volatility = returns.rolling(20).std().fillna(0)
                        volume_ratio = (dataframe['volume'] / dataframe['volume'].rolling(20).mean()).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)
                        
                        X = np.column_stack([returns.values, volatility.values, volume_ratio.values])
                        feature_names = ['returns', 'volatility', 'volume_ratio']
                        
                        # Create targets (future returns)
                        y = returns.shift(-1).fillna(0).values
                        
                        # Remove last row where target is NaN
                        X = X[:-1]
                        y = y[:-1]
                        
                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]
            
            if X is None or y is None or cluster_assignments is None:
              # Detailed error reporting for missing data
              missing_data = []
              if X is None:
                  missing_data.append("features")
              if y is None:
                  missing_data.append("targets")
              if regime_labels is None:
                  missing_data.append("regime_labels")

              if missing_data:
                  available_keys = list(pipeline_state.keys())
                  error_msg = (
                      f"Missing required data: {', '.join(missing_data)}. "
                      f"Available pipeline state keys: {available_keys}"
                  )
                  raise ValueError(error_msg)
            
            # Execute training
            results = self.training_instance.execute(X, y, cluster_assignments, feature_names)
            
            # Create comprehensive artifact
            artifact = {
                'hmm_models_training_result': {
                    'hmm_models': results.get('model_results', {}),
                    'hmm_training_metrics': results.get('comprehensive_report', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', 0),
                    'success': 'error' not in results
                }
            }
            
            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'hmm_models_training', 'execution_time': results.get('training_time', 0)}
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'hmm_models_training'}
            )


class HMMEnsembleTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for HMM Ensemble Training Component to work as a component."""
    
    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.training_class = training_class
        self.training_instance = None
    
    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM ensemble training as a component."""
        try:
            # Create training instance if not exists
            if self.training_instance is None:
                self.training_instance = self.training_class()
            
            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            hmm_states = pipeline_state.get('hmm_states')
            base_hmm_models = pipeline_state.get('hmm_models', {}).get('hmm_models', {})
            hmm_training_metrics = pipeline_state.get('hmm_models', {}).get('hmm_training_metrics', {})
            
            # If cluster_assignments is missing, try to get from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    print(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")
            
            # If we don't have features/targets, try to extract from dataframe
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:
                    import pandas as pd
                    import numpy as np
                    
                    # Create basic features and targets from OHLCV data
                    if 'close' in dataframe.columns:
                        # Simple features: returns, volatility, etc.
                        returns = dataframe['close'].pct_change().fillna(0)
                        volatility = returns.rolling(20).std().fillna(0)
                        volume_ratio = (dataframe['volume'] / dataframe['volume'].rolling(20).mean()).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)
                        
                        X = np.column_stack([returns.values, volatility.values, volume_ratio.values])
                        feature_names = ['returns', 'volatility', 'volume_ratio']
                        
                        # Create targets (future returns)
                        y = returns.shift(-1).fillna(0).values
                        
                        # Remove last row where target is NaN
                        X = X[:-1]
                        y = y[:-1]
                        
                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]
            
            if X is None or y is None or cluster_assignments is None:
                missing_items = []
                if X is None: missing_items.append("features")
                if y is None: missing_items.append("targets")
                if cluster_assignments is None: missing_items.append("cluster_assignments")
                raise ValueError(f"Missing required data: {', '.join(missing_items)}")
            
            # Execute training
            results = self.training_instance.execute(
                X, y, cluster_assignments, feature_names, hmm_states, 
                base_hmm_models, hmm_training_metrics
            )
            
            # Create comprehensive artifact
            artifact = {
                'hmm_ensemble_training_result': {
                    'hmm_ensemble': results.get('models', {}),
                    'hmm_ensemble_metrics': results.get('comprehensive_report', {}),
                    'ensemble_metrics': results.get('ensemble_metrics', {}),
                    'performance_summary': results.get('performance_summary', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', 0),
                    'success': 'error' not in results
                }
            }
            
            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'hmm_ensemble_training', 'execution_time': results.get('training_time', 0)}
            )
            
        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'hmm_ensemble_training'}
            )


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
        
        # Handle HMM training components (moved to hmm_models_training module)
        if component_name == 'hmm_models_training':
            try:
                from ..hmm_models_training.hmm_models_training_enhanced import HMMModelsTrainingEnhanced
                return HMMModelsTrainingComponentWrapper(HMMModelsTrainingEnhanced, config)
            except ImportError as e:
                raise ValueError(f"Failed to import HMMModelsTrainingEnhanced: {e}")
        
        if component_name == 'hmm_ensemble_training':
            try:
                from ..hmm_models_training import HMMEnsembleTrainingComponent
                return HMMEnsembleTrainingComponentWrapper(HMMEnsembleTrainingComponent, config)
            except ImportError as e:
                raise ValueError(f"Failed to import HMMEnsembleTrainingComponent: {e}")
        
        if component_name not in self._components:
            available_components = list(self._components.keys()) + ['regime_data_splitting', 'hmm_models_training', 'hmm_ensemble_training']
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
        # Include both registered components and lazy-loaded components
        lazy_components = ['regime_data_splitting', 'hmm_models_training', 'hmm_ensemble_training']
        return list(self._components.keys()) + lazy_components
    
    @classmethod
    def is_component_available(self, component_name: str) -> bool:
        """
        Check if a component is available.
        
        Args:
            component_name: Name of the component
            
        Returns:
            True if component is available
        """
        # Check both registered components and lazy-loaded components
        lazy_components = ['regime_data_splitting', 'hmm_models_training', 'hmm_ensemble_training']
        return component_name in self._components or component_name in lazy_components