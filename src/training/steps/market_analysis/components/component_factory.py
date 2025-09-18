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


class MultiHorizonComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for Multi-Horizon Profit Labeler to work as a component."""
    
    def __init__(self, adapter_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.adapter_class = adapter_class
        self.adapter_instance = None
    
    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result']
    
    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute multi-horizon labeling as a component."""
        try:
            # Create adapter instance if not exists
            if self.adapter_instance is None:
                self.adapter_instance = self.adapter_class()
            
            # Extract configuration from component config
            labeling_config = {}
            if self.config and hasattr(self.config, 'custom_params'):
                labeling_config = self.config.custom_params.get('multi_horizon_labeling', {})
            
            # Execute multi-horizon labeling with proper execution mode detection
            execution_mode = 'full'  # Default
            
            # Try multiple sources for execution mode
            if pipeline_state.get('execution_mode'):
                execution_mode = pipeline_state.get('execution_mode')
            elif self.config and hasattr(self.config, 'mode'):
                execution_mode = self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode)
            elif pipeline_state.get('mode'):
                execution_mode = pipeline_state.get('mode')
            
            # Force data filtering before calling the adapter
            original_data_size = len(data)
            if execution_mode.lower() == 'light' and original_data_size > 20000:
                data = data.tail(14400).copy()  # 10 days for 1m data
                print(f"🔥 COMPONENT FACTORY LIGHT FILTERING: {original_data_size:,} → {len(data):,} rows")
            elif execution_mode.lower() == 'blank' and original_data_size > 300000:
                data = data.tail(259200).copy()  # 180 days for 1m data  
                print(f"🔥 COMPONENT FACTORY BLANK FILTERING: {original_data_size:,} → {len(data):,} rows")
            
            result = self.adapter_instance.execute_multi_horizon_labeling_step(
                data=data,
                regime_labels=pipeline_state.get('regime_labels'),
                config=labeling_config,
                symbol=pipeline_state.get('symbol', 'UNKNOWN'),
                exchange=pipeline_state.get('exchange', 'UNKNOWN'),
                timeframe=pipeline_state.get('timeframe', 'UNKNOWN'),
                mode=execution_mode
            )
            
            # Convert to ComponentResult
            from .base_component import ComponentResult
            
            # Handle case where result is None
            if result is None:
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={},
                    error_message="Multi-horizon labeling returned None result"
                )
            
            if result.get('status') == 'completed':
                return ComponentResult(
                    success=True,
                    artifacts=result.get('artifacts', {}),
                    metadata=result.get('metadata', {}),
                    error_message=None
                )
            else:
                return ComponentResult(
                    success=False,
                    artifacts=result.get('artifacts', {}),
                    metadata=result.get('metadata', {}),
                    error_message=result.get('error', 'Unknown error in multi-horizon labeling')
                )
                
        except Exception as e:
            from .base_component import ComponentResult
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=f"Multi-horizon labeling component failed: {str(e)}"
            )

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
            
            # If still missing, try to load from artifacts (previous outcome files)
            if cluster_assignments is None:
                artifacts = pipeline_state.get('artifacts', {})
                
                # Check hmm_clustering artifacts
                hmm_clustering_result = artifacts.get('hmm_clustering_result', {})
                if hmm_clustering_result:
                    cluster_assignments = hmm_clustering_result.get('cluster_assignments')
                    if cluster_assignments is not None:
                        print(f"✅ Found cluster_assignments in hmm_clustering artifacts: {len(cluster_assignments)} samples")
                
                # Check hmm_regime_discovery artifacts if still missing
                if cluster_assignments is None:
                    hmm_regime_result = artifacts.get('hmm_regime_discovery_result', {})
                    if hmm_regime_result:
                        # Try to get regime assignments as cluster assignments
                        regime_assignments = hmm_regime_result.get('regime_assignments')
                        if regime_assignments is not None:
                            cluster_assignments = regime_assignments
                            print(f"✅ Found regime_assignments as cluster_assignments: {len(cluster_assignments)} samples")
                        
                        # Also try direct cluster_assignments from regime discovery
                        if cluster_assignments is None:
                            cluster_assignments = hmm_regime_result.get('cluster_assignments')
                            if cluster_assignments is not None:
                                print(f"✅ Found cluster_assignments in hmm_regime_discovery artifacts: {len(cluster_assignments)} samples")
            
            # If still missing, try to load from the most recent outcome files
            if cluster_assignments is None:
                print("🔍 Attempting to load cluster_assignments from recent outcome files...")
                try:
                    from pathlib import Path
                    import json
                    
                    outcome_dir = Path("outcomes")
                    if outcome_dir.exists():
                        # Look for the most recent hmm_clustering outcome
                        clustering_files = list(outcome_dir.glob("market_analysis_hmm_clustering_outcome_*.json"))
                        if clustering_files:
                            latest_clustering = max(clustering_files, key=lambda f: f.stat().st_mtime)
                            print(f"📂 Loading from: {latest_clustering}")
                            
                            with open(latest_clustering, 'r') as f:
                                clustering_data = json.load(f)
                            
                            clustering_artifacts = clustering_data.get('artifacts', {})
                            hmm_clustering_result = clustering_artifacts.get('hmm_clustering_result', {})
                            cluster_assignments = hmm_clustering_result.get('cluster_assignments')
                            
                            if cluster_assignments is not None:
                                print(f"✅ Loaded cluster_assignments from outcome file: {len(cluster_assignments)} samples")
                        
                        # If still missing, try hmm_regime_discovery outcomes
                        if cluster_assignments is None:
                            regime_files = list(outcome_dir.glob("market_analysis_hmm_regime_discovery_outcome_*.json"))
                            if regime_files:
                                latest_regime = max(regime_files, key=lambda f: f.stat().st_mtime)
                                print(f"📂 Loading from: {latest_regime}")
                                
                                with open(latest_regime, 'r') as f:
                                    regime_data = json.load(f)
                                
                                regime_artifacts = regime_data.get('artifacts', {})
                                hmm_regime_result = regime_artifacts.get('hmm_regime_discovery_result', {})
                                
                                # Try regime assignments first
                                regime_assignments = hmm_regime_result.get('regime_assignments')
                                if regime_assignments is not None:
                                    cluster_assignments = regime_assignments
                                    print(f"✅ Loaded regime_assignments as cluster_assignments: {len(cluster_assignments)} samples")
                                else:
                                    # Try direct cluster assignments
                                    cluster_assignments = hmm_regime_result.get('cluster_assignments')
                                    if cluster_assignments is not None:
                                        print(f"✅ Loaded cluster_assignments from regime discovery: {len(cluster_assignments)} samples")
                                
                except Exception as e:
                    print(f"⚠️ Failed to load from outcome files: {e}")
            
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
                        # 30-day volume average (30 days * 96 15-min periods = 2880 periods)
                        # Use available data for extrapolation when insufficient historical data
                        min_periods_30d = min(len(dataframe), 96)  # At least 1 day of data
                        volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
                        # Handle division by zero and missing values robustly
                        volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)
                        
                        X = np.column_stack([returns.values, volatility.values, volume_ratio_30d.values])
                        feature_names = ['returns', 'volatility', 'volume_ratio_30d']
                        
                        # Create targets (current returns) - convert to discrete classes for on-the-spot classification
                        current_returns = returns  # Use current returns, not future ones
                        
                        # Convert continuous returns to discrete classes for on-the-spot market condition
                        # Class 0: Strong Down (< -2%), Class 1: Down (-2% to -0.5%), 
                        # Class 2: Sideways (-0.5% to 0.5%), Class 3: Up (0.5% to 2%), Class 4: Strong Up (> 2%)
                        y_continuous = current_returns.values
                        y = np.zeros_like(y_continuous, dtype=int)
                        y[y_continuous < -0.02] = 0  # Strong Down
                        y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
                        y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
                        y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
                        y[y_continuous > 0.02] = 4  # Strong Up
                        
                        # Remove first row where returns is NaN (due to pct_change)
                        X = X[1:]
                        y = y[1:]
                        
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
              if cluster_assignments is None:
                  missing_data.append("cluster_assignments")

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
    
    def _convert_to_numpy_array(self, data):
        """Convert list data to numpy array if needed."""
        if data is not None:
            import numpy as np
            if isinstance(data, list):
                return np.array(data)
        return data
    
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
            
            # If still missing, try to load from artifacts (previous outcome files)
            if cluster_assignments is None:
                artifacts = pipeline_state.get('artifacts', {})
                
                # Check hmm_clustering artifacts
                hmm_clustering_result = artifacts.get('hmm_clustering_result', {})
                if hmm_clustering_result:
                    cluster_assignments = hmm_clustering_result.get('cluster_assignments')
                    if cluster_assignments is not None:
                        print(f"✅ Found cluster_assignments in hmm_clustering artifacts: {len(cluster_assignments)} samples")
                
                # Check hmm_regime_discovery artifacts if still missing
                if cluster_assignments is None:
                    hmm_regime_result = artifacts.get('hmm_regime_discovery_result', {})
                    if hmm_regime_result:
                        # Try to get regime assignments as cluster assignments
                        regime_assignments = hmm_regime_result.get('regime_assignments')
                        if regime_assignments is not None:
                            cluster_assignments = regime_assignments
                            print(f"✅ Found regime_assignments as cluster_assignments: {len(cluster_assignments)} samples")
                        
                        # Also try direct cluster_assignments from regime discovery
                        if cluster_assignments is None:
                            cluster_assignments = hmm_regime_result.get('cluster_assignments')
                            if cluster_assignments is not None:
                                print(f"✅ Found cluster_assignments in hmm_regime_discovery artifacts: {len(cluster_assignments)} samples")
            
            # If still missing, try to load from the most recent outcome files
            if cluster_assignments is None:
                print("🔍 Attempting to load cluster_assignments from recent outcome files...")
                try:
                    from pathlib import Path
                    import json
                    
                    outcome_dir = Path("outcomes")
                    if outcome_dir.exists():
                        # Look for the most recent hmm_clustering outcome
                        clustering_files = list(outcome_dir.glob("market_analysis_hmm_clustering_outcome_*.json"))
                        if clustering_files:
                            latest_clustering = max(clustering_files, key=lambda f: f.stat().st_mtime)
                            print(f"📂 Loading from: {latest_clustering}")
                            
                            with open(latest_clustering, 'r') as f:
                                clustering_data = json.load(f)
                            
                            clustering_artifacts = clustering_data.get('artifacts', {})
                            hmm_clustering_result = clustering_artifacts.get('hmm_clustering_result', {})
                            cluster_assignments = hmm_clustering_result.get('cluster_assignments')
                            
                            if cluster_assignments is not None:
                                print(f"✅ Loaded cluster_assignments from outcome file: {len(cluster_assignments)} samples")
                        
                        # If still missing, try hmm_regime_discovery outcomes
                        if cluster_assignments is None:
                            regime_files = list(outcome_dir.glob("market_analysis_hmm_regime_discovery_outcome_*.json"))
                            if regime_files:
                                latest_regime = max(regime_files, key=lambda f: f.stat().st_mtime)
                                print(f"📂 Loading from: {latest_regime}")
                                
                                with open(latest_regime, 'r') as f:
                                    regime_data = json.load(f)
                                
                                regime_artifacts = regime_data.get('artifacts', {})
                                hmm_regime_result = regime_artifacts.get('hmm_regime_discovery_result', {})
                                
                                # Try regime assignments first
                                regime_assignments = hmm_regime_result.get('regime_assignments')
                                if regime_assignments is not None:
                                    cluster_assignments = regime_assignments
                                    print(f"✅ Loaded regime_assignments as cluster_assignments: {len(cluster_assignments)} samples")
                                else:
                                    # Try direct cluster assignments
                                    cluster_assignments = hmm_regime_result.get('cluster_assignments')
                                    if cluster_assignments is not None:
                                        print(f"✅ Loaded cluster_assignments from regime discovery: {len(cluster_assignments)} samples")
                                
                except Exception as e:
                    print(f"⚠️ Failed to load from outcome files: {e}")
            
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
                        # 30-day volume average (30 days * 96 15-min periods = 2880 periods)
                        # Use available data for extrapolation when insufficient historical data
                        min_periods_30d = min(len(dataframe), 96)  # At least 1 day of data
                        volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
                        # Handle division by zero and missing values robustly
                        volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)
                        
                        X = np.column_stack([returns.values, volatility.values, volume_ratio_30d.values])
                        feature_names = ['returns', 'volatility', 'volume_ratio_30d']
                        
                        # Create targets (current returns) - convert to discrete classes for on-the-spot classification
                        current_returns = returns  # Use current returns, not future ones
                        
                        # Convert continuous returns to discrete classes for on-the-spot market condition
                        # Class 0: Strong Down (< -2%), Class 1: Down (-2% to -0.5%), 
                        # Class 2: Sideways (-0.5% to 0.5%), Class 3: Up (0.5% to 2%), Class 4: Strong Up (> 2%)
                        y_continuous = current_returns.values
                        y = np.zeros_like(y_continuous, dtype=int)
                        y[y_continuous < -0.02] = 0  # Strong Down
                        y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
                        y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
                        y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
                        y[y_continuous > 0.02] = 4  # Strong Up
                        
                        # Remove first row where returns is NaN (due to pct_change)
                        X = X[1:]
                        y = y[1:]
                        
                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]
            
            if X is None or y is None or cluster_assignments is None:
                missing_items = []
                if X is None: missing_items.append("features")
                if y is None: missing_items.append("targets")
                if cluster_assignments is None: missing_items.append("cluster_assignments")
                raise ValueError(f"Missing required data: {', '.join(missing_items)}")
            
            # Ensure all data is in proper numpy format before training
            cluster_assignments = self._convert_to_numpy_array(cluster_assignments)
            
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
        
        # Handle multi-horizon profit labeler
        if component_name == 'multi_horizon_profit_labeler':
            try:
                from ..multi_horizon_sub_pipeline_adapter import MultiHorizonSubPipelineAdapter
                return MultiHorizonComponentWrapper(MultiHorizonSubPipelineAdapter, config)
            except ImportError as e:
                raise ValueError(f"Failed to import MultiHorizonSubPipelineAdapter: {e}")
        
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
        lazy_components = ['regime_data_splitting', 'multi_horizon_profit_labeler', 'hmm_models_training', 'hmm_ensemble_training']
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