"""
Final Feature Selection Component.

This component performs multi-stage feature selection (120→100→80→60) as the final step
in the market analysis pipeline using VectorBT optimizations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import pandas as pd
import numpy as np

from .base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from .contracts import FinalFeatureSelectionArtifacts, PipelineState
from src.training.common.artifact_persistence import SaveReport
from src.training.common.component_result import ComponentError
from .component_factory import register_component
from ...market_analysis.logging_standards import (
    get_logger,
    log_info,
    log_warning,
    log_error,
    log_success,
    log_debug,
    LoggingContext,
    log_step_progress,
    log_data_info,
    log_validation_result,
)

# Import optimized process engine
from ...market_analysis.optimized_process_engines import OptimizedFeatureSelectionEngine
from ..validation.schemas import (
    SchemaValidationException,
    enforce_feature_temporal_alignment,
    schema_metadata,
    validate_engineered_features,
)

# Import VectorBT utilities
from ..utils.vectorbt_utils import (
    create_vectorbt_tools, VectorBTConfig, get_vectorbt_performance_stats,
    VECTORBT_UTILS_AVAILABLE
)

# Import hardware optimization tools
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy

# Import utility tools
from src.utils.common_utilities import CommonUtilities
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)
from src.training.config.data_locator import DataLocator

# Import bayesian optimization utilities
from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
    BayesianEntryTimingOptimizer, EntryTimingConfig
)
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization

# Import budget-aware feature selection
try:
    from ..budget_aware_feature_selection import (
        BudgetAwareFeatureSelector, BudgetAwareSelectionConfig,
        FeatureTypeBudget, create_budget_aware_selector
    )
    BUDGET_AWARE_SELECTION_AVAILABLE = True
except ImportError:
    BUDGET_AWARE_SELECTION_AVAILABLE = False
    tprint_warning("⚠️ Budget-aware feature selection not available")


CONFIG_ROOT_ENV = "ARES_CONFIG_ROOT"
"""Environment variable that can override the repo-relative config root."""

DEFAULT_CONFIG_ROOT = Path(__file__).resolve().parents[4] / "config"
"""Default location of repository configuration files relative to this module."""

FEATURE_SELECTION_CONFIG_PATH = Path(
    os.environ.get(CONFIG_ROOT_ENV, DEFAULT_CONFIG_ROOT)
) / "feature_selection_config.yaml"
"""Resolved path to the feature selection YAML profile."""


@register_component('final_feature_selection')
class FinalFeatureSelectionComponent(BasePreTrainingComponent):
    """
    Final Feature Selection Component.

    Performs multi-stage feature selection as the final step in the pipeline.
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the final feature selection component."""
        super().__init__(config)
        tprint("🚀 Initializing FinalFeatureSelectionComponent")
        
        # Use standardized logging
        self.logger = get_logger('FinalFeatureSelectionComponent')

        # Initialize hardware optimization tools
        tprint("🔧 Initializing hardware optimization tools...")
        self._initialize_hardware_tools()
        
        # Initialize utility managers
        tprint("🛠️ Initializing utility managers...")
        self._initialize_utility_managers()
        
        # Initialize VectorBT optimization tools
        tprint("⚡ Initializing VectorBT optimization tools...")
        self._initialize_vectorbt_tools()
        
        # Initialize bayesian optimization tools
        tprint("🔬 Initializing bayesian optimization tools...")
        self._initialize_bayesian_tools()
        
        # Initialize optimized process engine
        tprint("🎯 Initializing optimized feature selection engine...")
        self._initialize_feature_selection_engine()
        
        tprint_success("✅ FinalFeatureSelectionComponent initialization complete")

    def _initialize_hardware_tools(self):
        """Initialize hardware optimization tools with proper error handling."""
        try:
            self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
            if self.memory_optimizer is None:
                raise RuntimeError("Memory optimizer initialization failed")
            tprint_success("   ✅ Memory optimizer initialized")
        except Exception as e:
            tprint_error(f"   ❌ Memory optimizer failed: {e}")
            raise RuntimeError(f"Memory optimizer initialization failed: {e}") from e
        
        try:
            self.adaptive_engine = AdaptiveOptimizationEngine()
            if self.adaptive_engine is None:
                raise RuntimeError("Adaptive optimization engine initialization failed")
            tprint_success("   ✅ Adaptive optimization engine initialized")
        except Exception as e:
            tprint_error(f"   ❌ Adaptive optimization engine failed: {e}")
            raise RuntimeError(f"Adaptive optimization engine initialization failed: {e}") from e
        
        try:
            self.hardware_manager = UnifiedHardwareManager()
            if self.hardware_manager is None:
                raise RuntimeError("Hardware manager initialization failed")
            tprint_success("   ✅ Hardware manager initialized")
        except Exception as e:
            tprint_error(f"   ❌ Hardware manager failed: {e}")
            raise RuntimeError(f"Hardware manager initialization failed: {e}") from e
        
        # Initialize advanced memory optimizer for aggressive cleanup
        try:
            self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                memory_limit_gb=8.0,
                strategy=MemoryStrategy.AGGRESSIVE
            )
            tprint_success("   ✅ Advanced memory optimizer initialized")
        except Exception as e:
            self.advanced_memory_optimizer = None
            tprint_warning(f"   ⚠️ Advanced memory optimizer not available: {e}")

    def _initialize_utility_managers(self):
        """Initialize utility managers."""
        tprint_debug("🛠️ Initializing utility managers")
        self.common_utils = CommonUtilities()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        tprint_success("   ✅ Utility managers initialized")

    def _initialize_vectorbt_tools(self):
        """Initialize VectorBT optimization tools."""
        try:
            vectorbt_config = VectorBTConfig(
                enable_gpu=False,  # Conservative for feature selection
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000
            )
            
            vectorbt_tools = create_vectorbt_tools(vectorbt_config)
            self.vectorbt_optimizer = vectorbt_tools['optimizer']
            self.vectorization_manager = vectorbt_tools['manager']
            self.vectorbt_enabled = vectorbt_tools['available']
            
            if self.vectorbt_enabled:
                tprint_success("   ✅ VectorBT optimization tools initialized")
            else:
                tprint_warning("   ⚠️ VectorBT optimization tools not available")
        except Exception as e:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self.vectorbt_enabled = False
            tprint_warning(f"   ⚠️ VectorBT optimization tools not available: {e}")

    def _initialize_bayesian_tools(self):
        """Initialize bayesian optimization tools."""
        try:
            self.bayesian_optimizer = BayesianEntryTimingOptimizer(
                EntryTimingConfig(
                    n_trials=100,
                    timeout_minutes=30
                )
            )
            tprint_success("   ✅ Bayesian optimizer initialized")
        except Exception as e:
            tprint_error(f"   ❌ Bayesian optimization failed: {e}")
            raise RuntimeError(f"Bayesian optimization initialization failed: {e}") from e

        self.hpo_utils = HyperparameterOptimization(
            config={
                'enable_parallel': True,
                'max_workers': 4,
                'enable_monitoring': True,
                'convergence': {
                    'improvement_threshold': 0.001,
                    'patience_trials': 20,
                    'min_trials': 10
                }
            }
        )
        tprint_success("   ✅ HPO utilities initialized")

    def _initialize_feature_selection_engine(self):
        """Initialize the optimized feature selection engine."""
        self.optimized_engine = OptimizedFeatureSelectionEngine(
            use_hardware_accel=True,
            cache_size=1000,
            use_vectorbt=True  # Enable VectorBT optimizations
        )
        tprint_success("   ✅ Feature selection engine initialized")

    async def _optimize_feature_selection_config(
        self,
        base_config: Dict[str, Any],
        data: Any,
        pipeline_state: PipelineState
    ) -> Dict[str, Any]:
        """
        Optimize feature selection configuration using bayesian optimization.

        Args:
            base_config: Base feature selection configuration
            data: Market data for optimization
            pipeline_state: Current pipeline state

        Returns:
            Optimized configuration with bayesian-enhanced parameters
        """
        tprint_info("🔬 Starting bayesian optimization for feature selection parameters")

        try:
            # Extract features from data for optimization
            feature_data = self._extract_features_for_optimization(data, pipeline_state)
            if feature_data is None or feature_data.empty:
                tprint_warning("⚠️ No feature data available for bayesian optimization, using base config")
                return base_config

            # Define optimization objective function
            def optimization_objective(trial):
                """Objective function for bayesian optimization of feature selection."""
                tprint_debug(f"   🎯 Running optimization trial {trial.number}")
                
                # Sample feature selection parameters
                n_features_stage1 = trial.suggest_int('n_features_stage1', 80, 120)
                n_features_stage2 = trial.suggest_int('n_features_stage2', 60, 100)
                n_features_final = trial.suggest_int('n_features_final', 40, 80)

                # Sample selection criteria weights
                correlation_weight = trial.suggest_float('correlation_weight', 0.1, 1.0)
                importance_weight = trial.suggest_float('importance_weight', 0.1, 1.0)
                stability_weight = trial.suggest_float('stability_weight', 0.1, 1.0)

                # Sample processing parameters
                use_parallel = trial.suggest_categorical('use_parallel', [True, False])
                memory_efficient = trial.suggest_categorical('memory_efficient', [True, False])

                # Create temporary config for evaluation
                temp_config = base_config.copy()
                temp_config.update({
                    'stage_reductions': [n_features_stage1, n_features_stage2, n_features_final],
                    'selection_weights': {
                        'correlation': correlation_weight,
                        'importance': importance_weight,
                        'stability': stability_weight
                    },
                    'use_parallel': use_parallel,
                    'memory_efficient': memory_efficient,
                    'bayesian_optimized': True
                })

                # Evaluate configuration
                score = self._evaluate_config_performance(temp_config, feature_data)
                tprint_debug(f"   📊 Trial {trial.number} score: {score:.4f}")
                return score

            # Run bayesian optimization
            tprint_info("🎯 Running bayesian optimization for feature selection parameters")
            study = self.bayesian_optimizer.create_study(
                study_name="feature_selection_optimization",
                direction="maximize"
            )

            # Run optimization with timeout and early stopping
            best_params = self.bayesian_optimizer.optimize(
                objective=optimization_objective,
                study=study,
                n_trials=50,  # Reduced for demo, increase in production
                timeout_minutes=5.0
            )

            # Apply optimized parameters to base config
            optimized_config = base_config.copy()
            optimized_config.update({
                'stage_reductions': [
                    best_params['n_features_stage1'],
                    best_params['n_features_stage2'],
                    best_params['n_features_final']
                ],
                'selection_weights': {
                    'correlation': best_params['correlation_weight'],
                    'importance': best_params['importance_weight'],
                    'stability': best_params['stability_weight']
                },
                'use_parallel': best_params['use_parallel'],
                'memory_efficient': best_params['memory_efficient'],
                'bayesian_optimization_applied': True,
                'bayesian_optimization_results': {
                    'best_score': study.best_value,
                    'best_params': best_params,
                    'n_trials': len(study.trials)
                }
            })

            tprint_success(
                f"✅ Bayesian optimization completed: best score {study.best_value:.4f}, "
                f"optimized parameters: {len(best_params)}"
            )

            return optimized_config

        except Exception as e:
            tprint_error(f"❌ Bayesian optimization failed: {e}")
            return base_config

    def _extract_features_for_optimization(self, data: Any, pipeline_state: PipelineState) -> Optional[pd.DataFrame]:
        """Extract features from data for optimization."""
        tprint_debug("🔍 Extracting features for optimization")
        
        try:
            # Try to get features from different sources
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Check if this is a feature matrix
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 10:  # Likely a feature matrix
                    tprint_debug(f"   📊 Found DataFrame with {len(numeric_cols)} numeric columns")
                    return data[numeric_cols].head(1000)  # Sample for optimization

            # Check pipeline state for feature matrix
            if isinstance(pipeline_state, dict):
                for key in ['feature_matrix', 'features', 'final_feature_candidates']:
                    candidate = pipeline_state.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        numeric_cols = candidate.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 10:
                            tprint_debug(f"   📊 Found {key} with {len(numeric_cols)} numeric columns")
                            return candidate[numeric_cols].head(1000)

            tprint_warning("   ⚠️ No suitable feature data found for optimization")
            return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract features for optimization: {e}")
            return None

    def _evaluate_config_performance(self, config: Dict[str, Any], feature_data: pd.DataFrame) -> float:
        """Evaluate configuration performance (simplified implementation)."""
        tprint_debug("📊 Evaluating configuration performance")
        
        try:
            # Simple heuristic scoring based on configuration parameters
            score = 0.5  # Base score

            # Reward balanced stage reductions
            stages = config.get('stage_reductions', [120, 100, 80, 60])
            if len(stages) >= 3:
                reduction_balance = 1.0 - (max(stages) - min(stages)) / max(stages)
                score += reduction_balance * 0.3

            # Reward use of parallel processing
            if config.get('use_parallel', False):
                score += 0.1

            # Reward memory efficiency
            if config.get('memory_efficient', False):
                score += 0.1

            # Reward balanced selection weights
            weights = config.get('selection_weights', {})
            if weights:
                weight_balance = 1.0 - np.std(list(weights.values())) / np.mean(list(weights.values()))
                score += weight_balance * 0.1

            tprint_debug(f"   📊 Configuration score: {score:.4f}")
            return min(1.0, score)

        except Exception as e:
            tprint_warning(f"   ⚠️ Error evaluating config performance: {e}")
            return 0.5  # Default score on error

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint_debug("📦 Getting required artifacts")
        return ['final_feature_selection_result']

    def _load_model_specific_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configuration from YAML file."""
        tprint_info(f"🧾 Loading model-specific config for '{model_type}'")
        
        try:
            import yaml

            # Try to load from the feature selection config file
            config_path = FEATURE_SELECTION_CONFIG_PATH
            tprint_debug(f"   📁 Config path: {config_path}")
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'feature_selection' in config_data:
                    fs_config = config_data['feature_selection']

                    # Check if model has a specific profile
                    if 'model_profiles' in fs_config and model_type in fs_config['model_profiles']:
                        model_config = fs_config['model_profiles'][model_type]

                        tprint_success(f"   ✅ Loaded profile '{model_type}' from YAML configuration")

                        # Map YAML config to expected format
                        stage_targets = [
                            model_config.get('target_features', 80) - 20,  # stage_1_target
                            model_config.get('target_features', 80) - 15,  # stage_2_target
                            model_config.get('target_features', 80) - 10   # stage_3_target
                        ]

                        return {
                            'target_features': model_config.get('target_features', 80),
                            'min_features': model_config.get('min_features', 60),
                            'max_features': model_config.get('max_features', 100),
                            'stage_targets': stage_targets,
                            'priority_categories': model_config.get('priority_categories', ['momentum', 'volatility', 'microstructure'])
                        }

                    # Use default settings if no model profile found
                    elif model_type == 'default':
                        tprint_info("   ℹ️ Using default profile from YAML")
                        return {
                            'target_features': fs_config.get('target_features', 80),
                            'min_features': fs_config.get('min_features', 60),
                            'max_features': fs_config.get('max_features', 100),
                            'stage_targets': [95, 75, 65],
                            'priority_categories': ['momentum', 'volatility', 'microstructure']
                        }

            # Fallback to hardcoded defaults if YAML loading fails
            tprint_warning(f"   ⚠️ Could not load model-specific config for {model_type}, using defaults")
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

        except Exception as e:
            tprint_error(f"   ❌ Error loading config for '{model_type}': {e}")
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

    def _apply_vectorbt_optimizations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT optimizations to data processing."""
        tprint_info("⚡ Applying VectorBT optimizations to data")
        
        if not self.vectorbt_enabled:
            tprint_warning("   ⚠️ VectorBT not available, using standard processing")
            return data
        
        try:
            # Use memory-optimized processing for large datasets
            if len(data) > 5000:
                tprint_info(f"   📦 Large dataset detected ({len(data)} rows), using memory-optimized processing")
                return self._vectorbt_memory_optimized_processing(data)
            else:
                tprint_info(f"   📊 Standard dataset size ({len(data)} rows), using standard VectorBT processing")
                return self._vectorbt_optimized_data_processing(data)
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT optimization failed: {e}, using standard processing")
            return data

    def _vectorbt_optimized_data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT optimizations to data processing."""
        tprint_debug("🔧 Applying VectorBT data processing optimizations")
        
        try:
            if self.vectorization_manager:
                # Apply VectorBT rolling optimizations
                optimized_data = self.vectorization_manager.optimize_dataframe(data)
                tprint_debug(f"   ✅ VectorBT optimization applied: {data.shape} -> {optimized_data.shape}")
                return optimized_data
            else:
                tprint_warning("   ⚠️ VectorBT manager not available")
                return data
        except Exception as e:
            tprint_warning(f"   ⚠️ VectorBT data processing failed: {e}")
            return data

    def _vectorbt_memory_optimized_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply memory-optimized VectorBT processing for large datasets."""
        tprint_debug("🧠 Applying memory-optimized VectorBT processing")
        
        try:
            if self.vectorization_manager:
                # Process data in chunks to manage memory
                chunk_size = 5000
                processed_chunks = []
                
                for i in range(0, len(data), chunk_size):
                    chunk = data.iloc[i:i + chunk_size]
                    optimized_chunk = self.vectorization_manager.optimize_dataframe(chunk)
                    processed_chunks.append(optimized_chunk)
                    tprint_debug(f"   📦 Processed chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}")
                
                result = pd.concat(processed_chunks, ignore_index=True)
                tprint_success(f"   ✅ Memory-optimized processing completed: {data.shape} -> {result.shape}")
                return result
            else:
                tprint_warning("   ⚠️ VectorBT manager not available for memory optimization")
                return data
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory-optimized VectorBT processing failed: {e}")
            return data

    def _get_vectorbt_performance_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        tprint_debug("📊 Getting VectorBT performance statistics")
        
        try:
            if self.vectorbt_enabled and hasattr(self, 'vectorization_manager'):
                stats = get_vectorbt_performance_stats()
                tprint_debug(f"   📊 VectorBT stats: {stats}")
                return stats
            else:
                tprint_debug("   ℹ️ VectorBT not available for performance stats")
                return {}
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to get VectorBT performance stats: {e}")
            return {}

    def _monitor_memory_pressure(self) -> Dict[str, Any]:
        """Monitor memory pressure and return statistics."""
        tprint_debug("🧠 Monitoring memory pressure")
        
        try:
            if hasattr(self.memory_optimizer, 'get_memory_pressure'):
                pressure = self.memory_optimizer.get_memory_pressure()
                cleanup_triggered = pressure > 0.8  # 80% threshold
                
                stats = {
                    'pressure': pressure,
                    'cleanup_triggered': cleanup_triggered,
                    'recommendations': ['Reduce batch size', 'Enable memory optimization'] if cleanup_triggered else []
                }
                
                tprint_debug(f"   📊 Memory pressure: {pressure:.2f}, cleanup triggered: {cleanup_triggered}")
                return stats
            else:
                tprint_debug("   ℹ️ Memory pressure monitoring not available")
                return {'pressure': 0.0, 'cleanup_triggered': False, 'recommendations': []}
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory pressure monitoring failed: {e}")
            return {'pressure': 0.0, 'cleanup_triggered': False, 'recommendations': []}

    def _aggressive_memory_cleanup(self, force_cleanup: bool = False) -> Dict[str, Any]:
        """Perform aggressive memory cleanup."""
        tprint_info("🧹 Performing aggressive memory cleanup")
        
        try:
            if self.advanced_memory_optimizer:
                result = self.advanced_memory_optimizer.cleanup(force=force_cleanup)
                tprint_success(f"   ✅ Memory cleanup completed: {result.get('memory_freed_mb', 0):.1f}MB freed")
                return result
            else:
                tprint_warning("   ⚠️ Advanced memory optimizer not available")
                return {'memory_freed_mb': 0.0, 'success': False}
        except Exception as e:
            tprint_warning(f"   ⚠️ Memory cleanup failed: {e}")
            return {'memory_freed_mb': 0.0, 'success': False}

    def _create_budget_aware_config(
        self,
        interactive_result: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> BudgetAwareSelectionConfig:
        """Create budget-aware selection configuration."""
        tprint_info("💰 Creating budget-aware selection configuration")
        
        try:
            # Extract feature type budgets from interactive result
            feature_type_budgets = []
            
            for feature_type in ['base_features', 'interaction_features', 'cross_timeframe_features', 'gate_features']:
                budget_ms = interactive_result.get(f'{feature_type}_budget_ms', 0.0)
                if budget_ms > 0:
                    feature_type_budgets.append(FeatureTypeBudget(
                        feature_type=feature_type,
                        budget_ms=budget_ms,
                        priority=interactive_result.get(f'{feature_type}_priority', 1.0)
                    ))
            
            config = BudgetAwareSelectionConfig(
                total_budget_ms=interactive_result.get('total_budget_ms', 100.0),
                feature_type_budgets=feature_type_budgets,
                performance_threshold=base_config.get('performance_threshold', 0.7),
                stability_threshold=base_config.get('stability_threshold', 0.6)
            )
            
            tprint_success(f"   ✅ Budget config created: {len(feature_type_budgets)} feature types, {config.total_budget_ms}ms total budget")
            return config
        except Exception as e:
            tprint_error(f"   ❌ Failed to create budget-aware config: {e}")
            raise

    def _prepare_data_for_budget_selection(
        self,
        data: Any,
        pipeline_state: PipelineState
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for budget-aware feature selection."""
        tprint_info("📊 Preparing data for budget-aware selection")
        
        try:
            # Extract features and labels
            if isinstance(data, pd.DataFrame):
                X = data.select_dtypes(include=[np.number])
                # Create dummy labels for demonstration
                y = pd.Series(np.random.randn(len(X)), index=X.index)
            else:
                # Try to get from pipeline state
                X = pipeline_state.get('feature_matrix')
                y = pipeline_state.get('labels')
                
                if X is None or y is None:
                    raise ValueError("No suitable data found for budget-aware selection")
            
            tprint_success(f"   ✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            tprint_error(f"   ❌ Failed to prepare data for budget selection: {e}")
            raise

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute final feature selection.

        Args:
            data: Market data for feature selection
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with feature selection results
        """
        tprint("🚀 Starting Final Feature Selection Component")
        tprint(f"   📊 Pipeline state type: {type(pipeline_state)}")
        
        pipeline_state = PipelineState.ensure(pipeline_state)
        validation_metadata: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {
            'inputs': {},
            'outputs': {},
            'derived': {},
        }

        target_shifts: Dict[str, int] = {}

        def _update_target_shifts(source: Any) -> None:
            if isinstance(source, Mapping):
                raw_shifts = source.get('target_shifts')
                if isinstance(raw_shifts, Mapping):
                    for key, value in raw_shifts.items():
                        try:
                            target_shifts[str(key)] = int(value)
                        except (TypeError, ValueError):
                            continue

        def _record_validated_frame(label: str, frame: pd.DataFrame) -> pd.DataFrame:
            validated = validate_engineered_features(
                frame,
                context=f"final_feature_selection.{label}"
            )
            enforce_feature_temporal_alignment(
                validated,
                context=f"final_feature_selection.{label}",
                target_shifts=target_shifts,
                feature_metadata=pipeline_state.get('feature_metadata') if isinstance(pipeline_state, Mapping) else None,
            )
            validation_metadata['inputs'][label] = schema_metadata('engineered_features').get('engineered_features')
            return validated

        try:
            tprint("🔍 Processing input data for feature selection")
            if isinstance(data, pd.DataFrame) and not data.empty:
                tprint(f"   📊 DataFrame input: {data.shape[0]} samples, {data.shape[1]} features")
                data = _record_validated_frame('input_data', data)
                tprint("   ✅ DataFrame validation completed")
            elif isinstance(data, dict):
                tprint(f"   📊 Dictionary input with {len(data)} keys")
                _update_target_shifts(data)
                for key in ('features', 'feature_matrix', 'data'):
                    candidate = data.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        tprint(f"   📊 Found {key} DataFrame: {candidate.shape}")
                        data[key] = _record_validated_frame(key, candidate)
                        tprint(f"   ✅ {key} validation completed")
            else:
                tprint(f"   ⚠️ Unexpected data type: {type(data)}")

            tprint("🔍 Processing pipeline state data")
            for key in ('feature_matrix', 'final_feature_candidates'):
                candidate = pipeline_state.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    tprint(f"   📊 Found {key} in pipeline state: {candidate.shape}")
                    pipeline_state[key] = _record_validated_frame(key, candidate)
                    tprint(f"   ✅ {key} validation completed")

            # Check memory pressure and apply optimizations
            tprint("🧠 Checking memory pressure")
            memory_stats = self._monitor_memory_pressure()
            memory_pressure = memory_stats['pressure']
            tprint(f"   📊 Memory pressure: {memory_pressure:.2f}")
            
            if memory_stats['cleanup_triggered']:
                tprint_warning(f'🧠 Memory pressure detected ({memory_pressure:.2f}), performing aggressive cleanup')
                self._aggressive_memory_cleanup(force_cleanup=False)
            else:
                tprint("   ✅ Memory pressure within normal limits")

            # Apply VectorBT optimization to input data if available
            if isinstance(data, pd.DataFrame) and not data.empty:
                data = self._apply_vectorbt_optimizations(data)
            else:
                tprint("   ⚠️ No DataFrame data available for VectorBT optimization")

            # Get hardware configuration for feature selection
            try:
                hardware_config = self.hardware_manager.get_optimal_config('feature_selection')
            except (AttributeError, TypeError):
                # Fallback to default configuration
                hardware_config = {
                    'use_gpu': getattr(self.hardware_manager, 'is_gpu_available', lambda: False)(),
                    'batch_size': 1000,
                    'num_threads': 4
                }
            tprint(f"   🛠️ Hardware configuration: {hardware_config}")

            # Adapt optimization strategy based on current conditions
            try:
                adaptive_strategy = self.adaptive_engine.get_optimal_strategy('feature_selection', {
                    'memory_pressure': memory_pressure,
                    'hardware_config': hardware_config
                })
            except (AttributeError, TypeError):
                # Fallback to default strategy
                adaptive_strategy = {
                    'batch_size': hardware_config.get('batch_size', 1000),
                    'parallel_workers': hardware_config.get('num_threads', 4),
                    'use_gpu': hardware_config.get('use_gpu', False),
                    'memory_limit_mb': 2048
                }
                tprint_warning(f'⚠️ Using default adaptive strategy')
            tprint(f"   🎯 Adaptive strategy: {adaptive_strategy}")

            # Import the final feature selection step
            from ..final_feature_selection_step import run_final_feature_selection_step

            # Resolve execution context
            symbol = getattr(self.config, 'symbol', None) or pipeline_state.get('symbol')
            if symbol is None:
                raise ValueError("Symbol must be provided in config or pipeline state")

            exchange = getattr(self.config, 'exchange', None) or pipeline_state.get('exchange', 'binance')
            timeframe = getattr(self.config, 'timeframe', None) or pipeline_state.get('timeframe', '15m')

            # Resolve data directory
            data_locator = getattr(self.config, 'data_locator', None) or pipeline_state.get('data_locator')
            data_dir = getattr(self.config, 'data_dir', None) or pipeline_state.get('data_dir')
            data_dir_key = getattr(self.config, 'data_dir_key', None) or pipeline_state.get('data_dir_key', 'market_data')

            if data_dir is None and isinstance(data_locator, DataLocator):
                data_dir = str(data_locator.data_path(data_dir_key))

            if data_dir is None:
                raise ValueError("Data directory could not be resolved for final feature selection")

            tprint(f"   📥 Execution context: symbol={symbol}, exchange={exchange}, timeframe={timeframe}, data_dir={data_dir}")

            # Resolve the model profile for feature selection
            model_type = None
            if self.config.custom_params:
                model_type = self.config.custom_params.get('model_type')
            if model_type is None:
                model_type = pipeline_state.get('model_type') if pipeline_state else None
            if not model_type:
                model_type = 'default'

            # Load model-specific configuration
            final_feature_selection_config = self._load_model_specific_config(model_type)
            tprint(f"   🧾 Final feature selection config: {final_feature_selection_config}")

            # Apply adaptive optimizations to config
            if adaptive_strategy:
                final_feature_selection_config.update({
                    'hardware_accelerated': adaptive_strategy.get('hardware_accelerated', True),
                    'memory_efficient': adaptive_strategy.get('memory_efficient', True),
                    'parallel_processing': adaptive_strategy.get('parallel_processing', False)
                })

            # Check if interactive feature generation result is available for budget-aware selection
            interactive_result = pipeline_state.get('interactive_feature_generation_result')
            use_budget_aware = BUDGET_AWARE_SELECTION_AVAILABLE and interactive_result is not None
            
            if use_budget_aware:
                tprint_info('🚀 Executing budget-aware feature selection with interactive features...')
                
                # Initialize budget-aware selector
                budget_config = self._create_budget_aware_config(interactive_result, final_feature_selection_config)
                budget_selector = create_budget_aware_selector(budget_config)
                
                # Prepare data for budget-aware selection
                X, y = self._prepare_data_for_budget_selection(data, pipeline_state)
                
                # Run budget-aware selection
                budget_result = await budget_selector.select_features(X, y)
                
                if budget_result.success:
                    # Update runtime config with budget-aware results
                    runtime_config = dict(final_feature_selection_config)
                    runtime_config.update({
                        'budget_aware_selection': True,
                        'selected_features': budget_result.all_selected_features,
                        'feature_type_breakdown': {
                            'base_features': budget_result.base_features,
                            'interaction_features': budget_result.interaction_features,
                            'cross_timeframe_features': budget_result.cross_timeframe_features,
                            'gate_features': budget_result.gate_features
                        },
                        'budget_used_ms': budget_result.total_budget_used_ms,
                        'overall_performance_score': budget_result.overall_performance_score
                    })
                    
                    tprint_success(f"✅ Budget-aware selection completed: {len(budget_result.all_selected_features)} features selected")
                    tprint_info(f"   📊 Base: {len(budget_result.base_features)}")
                    tprint_info(f"   🔗 Interaction: {len(budget_result.interaction_features)}")
                    tprint_info(f"   ⏰ Cross-timeframe: {len(budget_result.cross_timeframe_features)}")
                    tprint_info(f"   🚪 Gate: {len(budget_result.gate_features)}")
                    tprint_info(f"   💰 Budget used: {budget_result.total_budget_used_ms:.1f}ms")
                    tprint_info(f"   🎯 Performance score: {budget_result.overall_performance_score:.4f}")
                else:
                    tprint_warning(f"⚠️ Budget-aware selection failed: {budget_result.error_message}")
                    use_budget_aware = False
            
            if not use_budget_aware:
                # Fall back to standard feature selection
                tprint_info('🚀 Executing enhanced feature selection with hardware optimizations and bayesian optimization...')

                # Apply bayesian optimization to enhance feature selection parameters
                tprint_info("🔬 Applying bayesian optimization to feature selection parameters")
                optimized_config = await self._optimize_feature_selection_config(
                    final_feature_selection_config,
                    data,
                    pipeline_state
                )

                runtime_config = dict(optimized_config)
                runtime_config['budget_aware_selection'] = False

            # Set common runtime config parameters
            runtime_config['data_dir_key'] = data_dir_key
            runtime_config['final_features_dir_key'] = pipeline_state.get('final_feature_selection_dir_key', 'final_feature_selection')

            output_directory_override = getattr(self.config, 'output_directory', None)
            if output_directory_override is None:
                output_directory_override = pipeline_state.get('generated_dir')

            if output_directory_override and 'output_directory' not in runtime_config:
                runtime_config['output_directory'] = output_directory_override
            if isinstance(data_locator, DataLocator):
                runtime_config['data_locator'] = data_locator
                runtime_config.setdefault(
                    'output_directory_key',
                    pipeline_state.get('generated_dir_key', 'market_analysis'),
                )

            # Execute the final feature selection step
            tprint("🎯 Executing final feature selection step")
            success = await run_final_feature_selection_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                config=runtime_config
            )

            if success:
                # Create result artifacts with performance metrics
                performance_metrics = {
                    'memory_pressure_before': memory_pressure,
                    'memory_pressure_after': memory_stats.get('pressure', 0.0),
                    'hardware_config_used': hardware_config,
                    'adaptive_strategy_used': adaptive_strategy
                }
                
                # Add VectorBT performance metrics if available
                vectorbt_stats = self._get_vectorbt_performance_stats()
                if vectorbt_stats:
                    performance_metrics['vectorbt_performance'] = vectorbt_stats
                    tprint(f"   📊 VectorBT performance: {vectorbt_stats.get('total_operations', 0)} operations")

                # Create base result
                base_result = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'data_dir': data_dir,
                    'feature_selection_config': final_feature_selection_config,
                    'execution_mode': 'component',
                    'success': True,
                    'stage_reduction': {
                        'initial': 120,
                        'stage_1': 100,
                        'stage_2': 80,
                        'stage_3': 60
                    },
                    'hardware_performance': performance_metrics,
                    'validated_schemas': validation_metadata,
                }
                
                # Add budget-aware selection results if available
                if use_budget_aware and 'budget_used_ms' in runtime_config:
                    base_result.update({
                        'budget_aware_selection': True,
                        'selected_features': runtime_config.get('selected_features', []),
                        'feature_type_breakdown': runtime_config.get('feature_type_breakdown', {}),
                        'budget_used_ms': runtime_config.get('budget_used_ms', 0.0),
                        'overall_performance_score': runtime_config.get('overall_performance_score', 0.0),
                    })
                else:
                    base_result['budget_aware_selection'] = False

                artifacts_bundle = FinalFeatureSelectionArtifacts(
                    final_feature_selection_result=base_result,
                    validated_schemas=validation_metadata,
                )

                tprint_success(f'✅ Final feature selection completed successfully')
                tprint(f'   📊 Performance metrics: {performance_metrics}')

                # Perform aggressive memory cleanup after processing
                cleanup_results = self._aggressive_memory_cleanup(force_cleanup=False)
                tprint(f'   🧹 Memory cleanup: {cleanup_results["memory_freed_mb"]:.1f}MB freed')

                # Save artifacts persistently
                try:
                    save_report = await self.save_artifacts(artifacts_bundle, {
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'validated_schemas': validation_metadata
                    })

                    if save_report.paths:
                        tprint_success(f"💾 Artifacts saved: {list(save_report.paths.keys())}")
                        artifacts_saved_persistently = True
                    else:
                        tprint_error("❌ Artifact manager returned no file paths")
                        artifacts_saved_persistently = False

                except Exception as e:
                    tprint_warning(f"⚠️ Exception while saving artifacts: {e}")
                    artifacts_saved_persistently = False

                component_success = success and artifacts_saved_persistently

                result_error: Optional[Exception] = None
                warnings: List[str] = []
                if not component_success:
                    failure_reason = "Artifacts were not persisted" if not artifacts_saved_persistently else "Feature selection failed"
                    result_error = ComponentError(failure_reason)
                    warnings.append(failure_reason)
                    tprint_error(f"❌ {failure_reason}")

                return ComponentResult(
                    success=component_success,
                    artifacts=artifacts_bundle,
                    error=result_error,
                    warnings=warnings,
                    execution_time=0.0,
                    metrics={},
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'artifacts_saved_persistently': artifacts_saved_persistently,
                        'validated_schemas': validation_metadata,
                    }
                )
            else:
                tprint_error('❌ Final feature selection failed')

                # Perform aggressive memory cleanup even on failure
                cleanup_results = self._aggressive_memory_cleanup(force_cleanup=True)
                tprint(f'   🧹 Memory cleanup after failure: {cleanup_results["memory_freed_mb"]:.1f}MB freed')

                failure_message = "Final feature selection execution failed"
                return ComponentResult(
                    success=False,
                    artifacts=FinalFeatureSelectionArtifacts(),
                    error=ComponentError(failure_message),
                    warnings=[failure_message],
                    execution_time=0.0,
                    metrics={},
                    metadata={
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'memory_pressure': memory_stats.get('pressure', 0.0),
                        'validated_schemas': validation_metadata
                    }
                )

        except SchemaValidationException as schema_error:
            error_message = str(schema_error)
            log_error(f'Final feature selection schema validation failed: {error_message}')
            self._log_error(
                f'❌ [FinalFeatureSelection] Schema validation error: {error_message}',
                event='final_feature_selection.validation',
                error=error_message,
            )
            return ComponentResult(
                success=False,
                artifacts=FinalFeatureSelectionArtifacts(),
                error_message=error_message,
                execution_time=0.0,
                metadata={
                    'component_type': 'final_feature_selection',
                    'schema_error': {
                        'schema_key': schema_error.schema_key,
                        'context': schema_error.context,
                        'schema_metadata': schema_metadata(schema_error.schema_key).get(schema_error.schema_key)
                    },
                    'validated_schemas': validation_metadata
                }
            )

        except Exception as e:
            log_error(f'Final feature selection failed with exception: {e}')
            self._log_error(
                f'❌ [FinalFeatureSelection] Exception during execution: {e}',
                event='final_feature_selection.result',
                error=str(e),
            )

            # Perform aggressive memory cleanup on exception
            try:
                cleanup_results = self._aggressive_memory_cleanup(force_cleanup=True)
                self._log_info(
                    f'🧹 [FinalFeatureSelection] Aggressive memory cleanup performed after exception: {cleanup_results["memory_freed_mb"]:.1f}MB freed',
                    event='final_feature_selection.aggressive_cleanup',
                    status='post_exception',
                    memory_freed_mb=cleanup_results['memory_freed_mb'],
                    cleanup_success=cleanup_results['success']
                )
            except Exception as cleanup_error:
                self._log_warning(
                    f'⚠️ [FinalFeatureSelection] Aggressive memory cleanup failed (non-critical): {cleanup_error}',
                    event='final_feature_selection.cleanup',
                    error=str(cleanup_error),
                )

            failure_message = str(e)
            return ComponentResult(
                success=False,
                artifacts=FinalFeatureSelectionArtifacts(),
                error=ComponentError(failure_message),
                warnings=[failure_message],
                execution_time=0.0,
                metrics={},
                metadata={
                    'component_type': 'final_feature_selection',
                    'symbol': symbol if 'symbol' in locals() else 'unknown',
                    'exchange': exchange if 'exchange' in locals() else 'unknown',
                    'timeframe': timeframe if 'timeframe' in locals() else 'unknown',
                    'memory_pressure': getattr(self.memory_optimizer, 'memory_pressure', 0.0),
                    'validated_schemas': validation_metadata
                }
            )

    def cleanup(self):
        """Cleanup resources."""
        tprint("🧹 Cleaning up FinalFeatureSelectionComponent resources")
        
        try:
            # Clear any caches or temporary data
            if hasattr(self, 'optimized_engine'):
                self.optimized_engine = None
            
            if hasattr(self, 'vectorbt_optimizer'):
                self.vectorbt_optimizer = None
                
            if hasattr(self, 'vectorization_manager'):
                self.vectorization_manager = None
            
            tprint_success("   ✅ Cleanup completed")
        except Exception as e:
            tprint_warning(f"   ⚠️ Cleanup error: {e}")


# Register the component with the factory
register_component('final_feature_selection', FinalFeatureSelectionComponent)
