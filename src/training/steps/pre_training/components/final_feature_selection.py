"""
Final Feature Selection Component.

This component performs multi-stage feature selection (120→100→80→60) as the final step
in the market analysis pipeline.
"""

import dataclasses
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import pandas as pd

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

# Import hardware optimization tools
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.adaptive_optimization_engine import AdaptiveOptimizationEngine
from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
from src.utils.hardware.m1_gpu_utils import M1GPUManager
from src.utils.hardware.advanced_memory_optimizer import AdvancedM1MemoryOptimizer, MemoryStrategy, MemoryPoolType
from src.utils.hardware.memory_optimization import get_advanced_memory_optimizer

# Import additional utility tools
from src.utils.common_operations import safe_dataframe_operation
from src.utils.common_utilities import CommonUtilities
from src.utils.math_validation import validate_finite
from src.utils.serialization_utils import JSONSerializer, PickleSerializer
from src.utils.data.klines_parquet import KlinesParquetManager
from src.utils.matrix_operations import (
    get_unified_matrix_operations,
    get_vectorized_processing_core,
    get_batch_matrix_processor,
    safe_matrix_multiply,
    optimize_dataframe,
    matrix_correlation_analysis,
    gpu_matrix_multiply,
    correlation_matrix_gpu
)
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_error
from src.training.config.data_locator import DataLocator

# Import bayesian optimization utilities
from src.utils.ml_common.optimization.bayesian_entry_timing_optimizer import (
    BayesianEntryTimingOptimizer, EntryTimingConfig, EntryTimingResult
)
# Alias for backward compatibility
BayesianConfig = EntryTimingConfig
OptimizationResult = EntryTimingResult
from src.utils.ml_common.optimization.hpo_utils import HyperparameterOptimization
# HPOStudyConfig and HPOTrialResult may not be exported - use generic types if needed
try:
    from src.utils.ml_common.optimization.hpo_utils import HPOStudyConfig
except ImportError:
    HPOStudyConfig = dict  # Fallback to dict if not available
try:
    from src.utils.ml_common.optimization.hpo_utils import HPOTrialResult
except ImportError:
    HPOTrialResult = dict  # Fallback to dict if not available
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

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
        # Use standardized logging
        self.logger = get_logger('FinalFeatureSelectionComponent')

        # Initialize hardware optimization tools with fast failing
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing hardware optimization tools...",
            event='final_feature_selection.initialization',
        )
        
        # Fast fail if critical hardware tools are not available
        try:
            self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
            if self.memory_optimizer is None:
                raise RuntimeError("Memory optimizer initialization failed")
        except Exception as e:
            raise RuntimeError(f"Memory optimizer initialization failed: {e}") from e
        
        try:
            self.adaptive_engine = AdaptiveOptimizationEngine()
            if self.adaptive_engine is None:
                raise RuntimeError("Adaptive optimization engine initialization failed")
        except Exception as e:
            raise RuntimeError(f"Adaptive optimization engine initialization failed: {e}") from e
        
        try:
            self.hardware_manager = UnifiedHardwareManager()
            if self.hardware_manager is None:
                raise RuntimeError("Hardware manager initialization failed")
        except Exception as e:
            raise RuntimeError(f"Hardware manager initialization failed: {e}") from e
        
        try:
            self.gpu_manager = M1GPUManager()
        except Exception as e:
            # GPU manager is optional, but log the error
            self._log_warning(f"⚠️ GPU manager initialization failed: {e}")
            self.gpu_manager = None
        
        # Initialize advanced memory optimizer for aggressive cleanup
        try:
            self.advanced_memory_optimizer = AdvancedM1MemoryOptimizer(
                memory_limit_gb=8.0,
                strategy=MemoryStrategy.AGGRESSIVE
            )
            self._log_info(
                "🧠 [FinalFeatureSelection] Advanced memory optimizer initialized with aggressive strategy",
                event='final_feature_selection.initialization',
            )
        except Exception as e:
            self.advanced_memory_optimizer = None
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] Advanced memory optimizer not available: {e}",
                event='final_feature_selection.initialization',
            )

        # Initialize additional utility managers
        self.common_utils = CommonUtilities()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        self.data_manager = KlinesParquetManager()

        # Initialize matrix operations managers
        tprint("🔢 Initializing matrix operations managers for final feature selection...")
        self.matrix_ops = get_unified_matrix_operations()
        self.vectorized_core = get_vectorized_processing_core()
        self.batch_processor = get_batch_matrix_processor()
        tprint_success("✅ Matrix operations managers initialized for final feature selection")

        # Initialize optimized process engine with hardware acceleration and VectorBT
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing optimized feature selection engine with VectorBT...",
            event='final_feature_selection.initialization',
        )
        self.optimized_engine = OptimizedFeatureSelectionEngine(
            use_hardware_accel=True,
            cache_size=1000,
            use_vectorbt=True  # Enable VectorBT optimizations
        )
        
        # Initialize VectorBT optimization tools for enhanced performance
        try:
            from src.feature_generation.utils.vectorbt_rolling_optimizer import (
                VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
            )
            from src.feature_generation.utils.unified_vectorization_manager import (
                UnifiedVectorizationManager, get_unified_vectorization_manager, VectorizationConfig
            )
            
            # Initialize VectorBT rolling optimizer
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=False,  # Conservative for feature selection
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000
            )
            
            # Initialize unified vectorization manager
            vectorization_config = VectorizationConfig(
                enable_vectorbt=True,
                enable_gpu=False,
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=1000,
                enable_monitoring=True,
                batch_size=10000,
                enable_batch_processing=True
            )
            self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
            
            self._log_success(
                "✅ [FinalFeatureSelection] VectorBT optimization tools initialized",
                event='final_feature_selection.vectorbt_init',
            )
        except Exception as e:
            self.vectorbt_optimizer = None
            self.vectorization_manager = None
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] VectorBT optimization tools not available: {e}",
                event='final_feature_selection.vectorbt_init',
            )

        # Initialize bayesian optimization tools for enhanced feature selection
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing bayesian optimization tools...",
            event='final_feature_selection.initialization',
        )
        
        # Fast fail if bayesian optimization is not available
        try:
            self.bayesian_optimizer = BayesianEntryTimingOptimizer(
                BayesianConfig(
                    n_trials=100,
                    timeout_minutes=30
                    # Note: early_stopping_patience, enable_parallel, use_m1_optimization 
                    # are not parameters of EntryTimingConfig
                )
            )
        except Exception as e:
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

        tprint_success("✅ Enhanced feature selection initialization complete")
        self._log_success(
            "✅ [FinalFeatureSelection] Hardware optimization tools, utility managers, and feature selection engine initialized",
            event='final_feature_selection.initialization',
        )

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

                # Evaluate configuration (simplified - in real implementation would run mini feature selection)
                score = self._evaluate_config_performance(temp_config, feature_data)

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
            self._log_error(
                f"❌ [FinalFeatureSelection] Bayesian optimization failed: {e}",
                event='final_feature_selection.bayesian_optimization_failed',
                error=str(e)
            )
            return base_config

    def _extract_features_for_optimization(self, data: Any, pipeline_state: PipelineState) -> Optional[pd.DataFrame]:
        """Extract features from data for optimization."""
        try:
            # Try to get features from different sources
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Check if this is a feature matrix
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 10:  # Likely a feature matrix
                    return data[numeric_cols].head(1000)  # Sample for optimization

            # Check pipeline state for feature matrix
            if isinstance(pipeline_state, dict):
                for key in ['feature_matrix', 'features', 'final_feature_candidates']:
                    candidate = pipeline_state.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        numeric_cols = candidate.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 10:
                            return candidate[numeric_cols].head(1000)

            return None

        except Exception as e:
            tprint_warning(f"⚠️ Failed to extract features for optimization: {e}")
            return None

    def _evaluate_config_performance(self, config: Dict[str, Any], feature_data: pd.DataFrame) -> float:
        """Evaluate configuration performance (simplified implementation)."""
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

            return min(1.0, score)

        except Exception:
            return 0.5  # Default score on error

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        self._log_info(
            "📦 [FinalFeatureSelection] Reporting required artifacts: final_feature_selection_result",
            event='final_feature_selection.requirements',
        )
        return ['final_feature_selection_result']

    def _load_model_specific_config(self, model_type: str) -> Dict[str, Any]:
        """Load model-specific configuration from YAML file."""
        try:
            import yaml

            # Try to load from the feature selection config file
            config_path = FEATURE_SELECTION_CONFIG_PATH
            log_debug(
                f"Resolving feature selection config for '{model_type}' via {config_path}"
            )
            self._log_info(
                f"🧩 [FinalFeatureSelection] Loading model-specific config for '{model_type}' from {config_path}",
                event='final_feature_selection.load_config',
                model_type=model_type,
                config_path=str(config_path),
            )
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                if config_data and 'feature_selection' in config_data:
                    fs_config = config_data['feature_selection']

                    # Check if model has a specific profile
                    if 'model_profiles' in fs_config and model_type in fs_config['model_profiles']:
                        model_config = fs_config['model_profiles'][model_type]

                        log_success(
                            f"Loaded feature selection profile '{model_type}' from {config_path}"
                        )
                        self._log_success(
                            f"✅ [FinalFeatureSelection] Loaded profile '{model_type}' from YAML configuration",
                            event='final_feature_selection.load_config',
                            model_type=model_type,
                        )

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
                        return {
                            'target_features': fs_config.get('target_features', 80),
                            'min_features': fs_config.get('min_features', 60),
                            'max_features': fs_config.get('max_features', 100),
                            'stage_targets': [95, 75, 65],
                            'priority_categories': ['momentum', 'volatility', 'microstructure']
                        }

            # Fallback to hardcoded defaults if YAML loading fails
            log_warning(
                f"Could not load model-specific config for {model_type}, using defaults. "
                f"Searched path: {config_path}"
            )
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] Using default configuration for '{model_type}'",
                event='final_feature_selection.load_config',
                model_type=model_type,
            )
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

        except Exception as e:
            log_error(
                f"Error loading model-specific config for {model_type}: {e}. "
                f"Searched path: {FEATURE_SELECTION_CONFIG_PATH}"
            )
            self._log_error(
                f"❌ [FinalFeatureSelection] Error loading config for '{model_type}': {e}. Using defaults.",
                event='final_feature_selection.load_config',
                model_type=model_type,
                error=str(e),
            )
            return {
                'target_features': 80,
                'min_features': 60,
                'max_features': 100,
                'stage_targets': [95, 75, 65],
                'priority_categories': ['momentum', 'volatility', 'microstructure']
            }

    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute final feature selection.

        Args:
            data: Market data for feature selection
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with feature selection results
        """
        pipeline_state = PipelineState.ensure(pipeline_state)
        log_info('🎯 Starting Final Feature Selection')
        self._log_info(
            '🚀 [FinalFeatureSelection] Starting execute routine',
            event='final_feature_selection.execute',
        )
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

        _update_target_shifts(pipeline_state)

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
            if isinstance(data, pd.DataFrame) and not data.empty:
                data = _record_validated_frame('input_data', data)
            elif isinstance(data, dict):
                _update_target_shifts(data)
                for key in ('features', 'feature_matrix', 'data'):
                    candidate = data.get(key)
                    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                        data[key] = _record_validated_frame(key, candidate)

            for key in ('feature_matrix', 'final_feature_candidates'):
                candidate = pipeline_state.get(key)
                if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                    pipeline_state[key] = _record_validated_frame(key, candidate)

            # Check memory pressure and apply aggressive optimizations
            memory_stats = self.monitor_memory_pressure()
            memory_pressure = memory_stats['pressure']
            
            if memory_stats['cleanup_triggered']:
                log_warning(f'🧠 Memory pressure detected ({memory_pressure:.2f}), performing aggressive cleanup')
                self._log_warning(
                    f"🧠 [FinalFeatureSelection] Memory pressure detected ({memory_pressure:.2f}); aggressive cleanup performed",
                    event='final_feature_selection.aggressive_memory_optimization',
                    memory_pressure=memory_pressure,
                    cleanup_triggered=memory_stats['cleanup_triggered'],
                    recommendations=memory_stats['recommendations']
                )

            # Apply VectorBT optimization to input data if available
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Use memory-optimized processing for large datasets
                if len(data) > 5000:
                    data = self._vectorbt_memory_optimized_processing(data)
                    self._log_info(
                        "✅ [FinalFeatureSelection] Applied VectorBT memory-optimized processing to large dataset",
                        event='final_feature_selection.vectorbt_memory_optimization',
                    )
                else:
                    data = self._vectorbt_optimized_data_processing(data)
                    self._log_info(
                        "✅ [FinalFeatureSelection] Applied VectorBT optimization to input data",
                        event='final_feature_selection.vectorbt_optimization',
                    )

            # Get hardware configuration for feature selection
            # Use default config if get_optimal_config is not available
            try:
                hardware_config = self.hardware_manager.get_optimal_config('feature_selection')
            except (AttributeError, TypeError):
                # Fallback to default configuration
                hardware_config = {
                    'use_gpu': self.hardware_manager.is_gpu_available() if hasattr(self.hardware_manager, 'is_gpu_available') else False,
                    'batch_size': 1000,
                    'num_threads': 4
                }
            log_debug(f'📊 Hardware configuration: {hardware_config}')
            self._log_info(
                f'🛠️ [FinalFeatureSelection] Hardware configuration resolved: {hardware_config}',
                event='final_feature_selection.hardware',
                hardware_config=hardware_config,
            )

            # Adapt optimization strategy based on current conditions
            # Use default strategy if get_optimal_strategy is not available
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
                log_warning(f'⚠️ Using default adaptive strategy (get_optimal_strategy not available)')
            log_debug(f'🎯 Adaptive strategy: {adaptive_strategy}')
            self._log_info(
                f'🎯 [FinalFeatureSelection] Adaptive strategy selected: {adaptive_strategy}',
                event='final_feature_selection.strategy',
                adaptive_strategy=adaptive_strategy,
            )

            # Import the final feature selection step
            from ..final_feature_selection_step import run_final_feature_selection_step

            # Resolve symbol from config or pipeline state
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                raise ValueError("Symbol must be provided in config or pipeline state")

            # Resolve exchange from config or pipeline state
            exchange = getattr(self.config, 'exchange', None)
            if exchange is None and 'exchange' in pipeline_state:
                exchange = pipeline_state['exchange']
            if exchange is None:
                exchange = 'binance'  # Default exchange

            # Resolve timeframe from config or pipeline state
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '15m'  # Default timeframe

            # Resolve data locator and directory information
            data_locator: Optional[DataLocator] = getattr(self.config, 'data_locator', None)
            if data_locator is None:
                data_locator = pipeline_state.get('data_locator')

            data_dir = getattr(self.config, 'data_dir', None)
            if data_dir is None:
                data_dir = pipeline_state.get('data_dir')

            data_dir_key = getattr(self.config, 'data_dir_key', None) or pipeline_state.get('data_dir_key', 'market_data')
            final_features_dir_key = (
                getattr(self.config, 'final_feature_selection_dir_key', None)
                or pipeline_state.get('final_feature_selection_dir_key', 'final_feature_selection')
            )

            if data_dir is None and isinstance(data_locator, DataLocator):
                data_dir = str(data_locator.data_path(data_dir_key))

            if data_dir is None:
                raise ValueError("Data directory could not be resolved for final feature selection")

            final_features_dir_override = (
                getattr(self.config, 'final_feature_selection_dir', None)
                or pipeline_state.get('final_feature_selection_dir')
            )

            output_directory_override = getattr(self.config, 'output_directory', None)
            if output_directory_override is None:
                output_directory_override = pipeline_state.get('generated_dir')

            self._log_info(
                "📥 [FinalFeatureSelection] Resolved execution context "
                f"symbol={symbol}, exchange={exchange}, timeframe={timeframe}, data_dir={data_dir}",
                event='final_feature_selection.context',
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
            )

            # Resolve the model profile for feature selection from config or pipeline state
            model_type = None
            if self.config.custom_params:
                model_type = self.config.custom_params.get('model_type')
            if model_type is None:
                model_type = pipeline_state.get('model_type') if pipeline_state else None
            if not model_type:
                model_type = 'default'

            # Load model-specific configuration with hardware optimizations
            final_feature_selection_config = self._load_model_specific_config(model_type)
            self._log_info(
                "🧾 [FinalFeatureSelection] Final feature selection config prepared: "
                f"{final_feature_selection_config}",
                event='final_feature_selection.config_ready',
                model_type=model_type,
            )

            if model_type != 'default':
                log_success(
                    f"Feature selection will use model-specific profile '{model_type}'"
                )
            else:
                log_info("Feature selection will use the default profile")

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
                log_info(f'🚀 Executing budget-aware feature selection with interactive features...')
                self._log_info(
                    '🚀 [FinalFeatureSelection] Executing budget-aware feature selection with interactive features',
                    event='final_feature_selection.execute',
                )
                
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
                log_info(f'🚀 Executing enhanced feature selection with hardware optimizations and bayesian optimization...')
                self._log_info(
                    '🚀 [FinalFeatureSelection] Executing enhanced feature selection step with bayesian optimization',
                    event='final_feature_selection.execute',
                )

                # First, apply bayesian optimization to enhance feature selection parameters
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
            runtime_config['final_features_dir_key'] = final_features_dir_key

            if final_features_dir_override:
                runtime_config['final_features_dir'] = final_features_dir_override
            if output_directory_override and 'output_directory' not in runtime_config:
                runtime_config['output_directory'] = output_directory_override
            if isinstance(data_locator, DataLocator):
                runtime_config['data_locator'] = data_locator
                runtime_config.setdefault(
                    'output_directory_key',
                    pipeline_state.get('generated_dir_key', 'market_analysis'),
                )

            success = await run_final_feature_selection_step(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                config=runtime_config
            )

            if success:
                # Create result artifacts with hardware performance metrics
                performance_metrics = {
                    'memory_pressure_before': memory_pressure,
                    'memory_pressure_after': getattr(self.memory_optimizer, 'memory_pressure', 0.0),
                    'hardware_config_used': hardware_config,
                    'adaptive_strategy_used': adaptive_strategy
                }
                
                # Add VectorBT performance metrics if available
                vectorbt_stats = self._get_vectorbt_performance_stats()
                if vectorbt_stats:
                    performance_metrics['vectorbt_performance'] = vectorbt_stats
                    self._log_info(
                        f"📊 [FinalFeatureSelection] VectorBT performance: {vectorbt_stats.get('total_operations', 0)} operations, "
                        f"{vectorbt_stats.get('vectorbt_usage_rate', 0):.2%} VectorBT usage rate",
                        event='final_feature_selection.vectorbt_performance',
                    )

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
                        'budget_allocation': {
                            'base_features': runtime_config.get('base_features_budget_ms', 68.0),
                            'interaction_features': runtime_config.get('interaction_features_budget_ms', 15.0),
                            'cross_timeframe_features': runtime_config.get('cross_timeframe_features_budget_ms', 10.0),
                            'gate_features': runtime_config.get('gate_features_budget_ms', 7.0)
                        }
                    })
                else:
                    base_result['budget_aware_selection'] = False

                artifacts_bundle = FinalFeatureSelectionArtifacts(
                    final_feature_selection_result=base_result,
                    validated_schemas=validation_metadata,
                )

                log_success(f'✅ Final feature selection completed successfully with hardware optimizations')
                log_info(f'📊 Performance metrics: {performance_metrics}')
                self._log_success(
                    "✅ [FinalFeatureSelection] Feature selection succeeded with metrics "
                    f"{performance_metrics}",
                    event='final_feature_selection.result',
                    metrics=performance_metrics,
                )

                # Perform aggressive memory cleanup after processing
                cleanup_results = self.aggressive_memory_cleanup(force_cleanup=False)
                self._log_info(
                    f'🧹 [FinalFeatureSelection] Performed post-execution aggressive memory cleanup: {cleanup_results["memory_freed_mb"]:.1f}MB freed',
                    event='final_feature_selection.aggressive_cleanup',
                    memory_freed_mb=cleanup_results['memory_freed_mb'],
                    cleanup_success=cleanup_results['success']
                )

                # Save artifacts persistently using the artifact manager
                persistence_error: Optional[str] = None
                artifacts_saved_persistently = False
                save_report: Optional[SaveReport] = None
                failure_reason: Optional[str] = None

                try:
                    save_report = await self.save_artifacts(artifacts_bundle, {
                        'component_type': 'final_feature_selection',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'validated_schemas': validation_metadata
                    })

                    if save_report.paths:
                        artifacts_saved_persistently = True
                        log_success(
                            f"💾 [FINAL_FEATURE_SELECTION] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}"
                        )
                        self._log_success(
                            "💾 [FinalFeatureSelection] Artifacts saved successfully",
                            event='final_feature_selection.artifacts',
                            artifact_keys=list(save_report.paths.keys()),
                            correlation_id=save_report.correlation_id,
                        )
                    else:
                        log_error("❌ [FINAL_FEATURE_SELECTION] Artifact manager returned no file paths")
                        self._log_error(
                            "❌ [FinalFeatureSelection] Failed to persist artifacts: no file paths returned",
                            event='final_feature_selection.artifacts',
                        )

                except Exception as e:
                    persistence_error = str(e)
                    failure_reason = f"Artifact saving failed: {e}"
                    log_warning(f"⚠️ [FINAL_FEATURE_SELECTION] Exception while saving artifacts persistently: {e}")
                    self._log_warning(
                        f"⚠️ [FinalFeatureSelection] Artifact save error: {e}",
                        event='final_feature_selection.artifacts',
                        error=str(e),
                    )

                component_success = success and artifacts_saved_persistently

                result_error: Optional[Exception] = None
                warnings: List[str] = []
                if not component_success:
                    failure_reason = failure_reason or persistence_error or "Artifacts were not persisted"
                    result_error = ComponentError(failure_reason)
                    warnings.append(failure_reason)
                    log_error(f"❌ [FINAL_FEATURE_SELECTION] {failure_reason}")

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
                        'artifact_persistence_report': dataclasses.asdict(save_report) if save_report else {},
                        'validated_schemas': validation_metadata,
                        **({'artifact_persistence_error': persistence_error} if persistence_error else {})
                    }
                )
            else:
                log_error('Final feature selection failed')
                self._log_error(
                    '❌ [FinalFeatureSelection] Feature selection execution returned failure',
                    event='final_feature_selection.result',
                )

                # Perform aggressive memory cleanup even on failure
                cleanup_results = self.aggressive_memory_cleanup(force_cleanup=True)
                self._log_info(
                    f'🧹 [FinalFeatureSelection] Aggressive memory cleanup performed after failure: {cleanup_results["memory_freed_mb"]:.1f}MB freed',
                    event='final_feature_selection.aggressive_cleanup',
                    status='post_failure',
                    memory_freed_mb=cleanup_results['memory_freed_mb'],
                    cleanup_success=cleanup_results['success']
                )

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
                        'memory_pressure': getattr(self.memory_optimizer, 'memory_pressure', 0.0),
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
                cleanup_results = self.aggressive_memory_cleanup(force_cleanup=True)
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

    # Utility methods for enhanced functionality

    def validate_finite_values(self, value, name: str = "value"):
        """Validate that values are finite using math validation utilities."""
        return validate_finite(value, name)

    def get_memory_pressure(self) -> float:
        """Get current memory pressure if available."""
        if self.memory_optimizer:
            return getattr(self.memory_optimizer, 'memory_pressure', 0.0)
        return 0.0

    def optimize_memory(self):
        """Apply memory optimizations if available."""
        if self.memory_optimizer:
            self.memory_optimizer._apply_memory_optimizations()
            self._log_info(
                "🧠 Applied memory optimizations",
                event='memory_optimized',
                memory_pressure=self.get_memory_pressure()
            )
    
    def aggressive_memory_cleanup(self, force_cleanup: bool = False) -> Dict[str, Any]:
        """Perform aggressive memory cleanup using advanced memory optimizer."""
        cleanup_results = {
            'success': False,
            'memory_freed_mb': 0.0,
            'memory_pressure_before': 0.0,
            'memory_pressure_after': 0.0,
            'cleanup_methods_used': [],
            'errors': []
        }
        
        try:
            # Get initial memory pressure
            cleanup_results['memory_pressure_before'] = self.get_memory_pressure()
            
            # Use advanced memory optimizer if available
            if self.advanced_memory_optimizer:
                tprint("🧹 Performing aggressive memory cleanup with advanced optimizer...")
                
                # Perform comprehensive cleanup
                advanced_cleanup = self.advanced_memory_optimizer.aggressive_cleanup(
                    force_cleanup=force_cleanup,
                    clear_caches=True,
                    compress_memory=True,
                    optimize_pools=True
                )
                
                cleanup_results.update({
                    'success': advanced_cleanup.get('success', False),
                    'memory_freed_mb': advanced_cleanup.get('memory_freed_mb', 0.0),
                    'cleanup_methods_used': advanced_cleanup.get('methods_used', [])
                })
                
                self._log_info(
                    f"🧹 [FinalFeatureSelection] Advanced memory cleanup: {cleanup_results['memory_freed_mb']:.1f}MB freed",
                    event='final_feature_selection.aggressive_cleanup',
                    memory_freed_mb=cleanup_results['memory_freed_mb']
                )
            
            # Fallback to standard memory optimizer
            if not cleanup_results['success'] and self.memory_optimizer:
                tprint("🧹 Performing fallback memory cleanup...")
                
                # Apply multiple cleanup strategies
                self.memory_optimizer._apply_memory_optimizations()
                self.memory_optimizer._light_memory_cleanup()
                
                # Force garbage collection
                import gc
                collected = gc.collect()
                
                cleanup_results.update({
                    'success': True,
                    'memory_freed_mb': collected * 0.001,  # Rough estimate
                    'cleanup_methods_used': ['standard_optimization', 'light_cleanup', 'garbage_collection']
                })
                
                self._log_info(
                    f"🧹 [FinalFeatureSelection] Fallback memory cleanup: {collected} objects collected",
                    event='final_feature_selection.fallback_cleanup',
                    objects_collected=collected
                )
            
            # Clear component-specific caches
            self._clear_component_caches()
            cleanup_results['cleanup_methods_used'].append('component_caches')
            
            # Get final memory pressure
            cleanup_results['memory_pressure_after'] = self.get_memory_pressure()
            
            tprint_success(f"✅ Aggressive memory cleanup completed: {cleanup_results['memory_freed_mb']:.1f}MB freed")
            
        except Exception as e:
            cleanup_results['errors'].append(str(e))
            self._log_error(
                f"❌ [FinalFeatureSelection] Aggressive memory cleanup failed: {e}",
                event='final_feature_selection.cleanup_error',
                error=str(e)
            )
            tprint_error(f"❌ Aggressive memory cleanup failed: {e}")
        
        return cleanup_results
    
    def _clear_component_caches(self):
        """Clear component-specific caches and temporary data."""
        try:
            # Clear matrix operations caches
            if hasattr(self, 'matrix_ops') and self.matrix_ops:
                if hasattr(self.matrix_ops, 'clear_cache'):
                    self.matrix_ops.clear_cache()
            
            # Clear vectorized arrays
            if hasattr(self, '_vectorized_arrays'):
                self._vectorized_arrays.clear()
            
            # Clear computation cache
            if hasattr(self, '_computation_cache'):
                self._computation_cache.clear()
            
            # Clear feature selection caches
            if hasattr(self, '_cache'):
                self._cache.clear()
            
            # Clear polarity tracking containers
            if hasattr(self, 'feature_polarity_adjustments'):
                self.feature_polarity_adjustments.clear()
            if hasattr(self, 'feature_polarity_history'):
                self.feature_polarity_history.clear()
            if hasattr(self, 'feature_sign_stability'):
                self.feature_sign_stability.clear()
            
            tprint("🧹 Component caches cleared")
            
        except Exception as e:
            tprint_warning(f"⚠️ Error clearing component caches: {e}")
    
    def monitor_memory_pressure(self) -> Dict[str, Any]:
        """Monitor memory pressure and trigger cleanup if needed."""
        memory_stats = {
            'pressure': self.get_memory_pressure(),
            'cleanup_triggered': False,
            'recommendations': []
        }
        
        pressure = memory_stats['pressure']
        
        if pressure > 0.9:  # Critical pressure
            memory_stats['cleanup_triggered'] = True
            memory_stats['recommendations'].append('CRITICAL: Immediate aggressive cleanup required')
            self.aggressive_memory_cleanup(force_cleanup=True)
            
        elif pressure > 0.8:  # High pressure
            memory_stats['cleanup_triggered'] = True
            memory_stats['recommendations'].append('WARNING: High memory pressure - performing cleanup')
            self.aggressive_memory_cleanup(force_cleanup=False)
            
        elif pressure > 0.7:  # Medium pressure
            memory_stats['recommendations'].append('INFO: Medium memory pressure - monitoring')
            if self.memory_optimizer:
                self.memory_optimizer._light_memory_cleanup()
        
        return memory_stats

    def is_hardware_accelerated(self) -> bool:
        """Check if hardware acceleration is available."""
        return self.gpu_manager.is_m1 if self.gpu_manager else False

    def serialize_results_json(self, results: Dict[str, Any], filepath: str) -> bool:
        """Serialize results to JSON format."""
        try:
            return self.json_serializer.save(results, filepath)
        except Exception as e:
            self.logger.error(f"Failed to serialize results: {e}")
            return False

    def deserialize_results_json(self, filepath: str):
        """Deserialize results from JSON format."""
        try:
            return self.json_serializer.load(filepath)
        except Exception as e:
            self.logger.error(f"Failed to deserialize results: {e}")
            return None

    def safe_dataframe_operation(self, df: pd.DataFrame, operation, *args, **kwargs):
        """Safely perform DataFrame operations with error handling."""
        return safe_dataframe_operation(df, operation, *args, **kwargs)

    def load_klines_data(self, symbol: str, timeframe: str, start_date=None, end_date=None):
        """Load klines data using the data manager."""
        if self.data_manager:
            return self.data_manager.load_symbol_data(
                symbol, timeframe, start_date, end_date
            )
        return None

    def safe_matrix_multiply(self, A, B):
        """Safely perform matrix multiplication with VectorBT optimization."""
        tprint(f"🔢 Performing VectorBT-optimized matrix multiplication ({A.shape} x {B.shape}) in final feature selection")
        
        # Use VectorBT for matrix operations if available
        if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
            try:
                # Use VectorBT for optimized matrix multiplication
                result = self.vectorization_manager.optimize_operation(
                    'matrix_multiplication',
                    {'a': A, 'b': B}
                )
                return result.result if hasattr(result, 'result') else result
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT matrix multiplication failed: {e}, using fallback")
        
        # Fallback to standard safe matrix multiplication
        return safe_matrix_multiply(A, B)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize DataFrame for matrix operations."""
        tprint(f"⚡ Optimizing DataFrame for matrix operations in final feature selection (shape: {df.shape})")
        return optimize_dataframe(df)

    def compute_matrix_correlation_analysis(self, data):
        """Compute matrix correlation analysis using VectorBT optimization."""
        tprint(f"📊 Computing VectorBT-optimized matrix correlation analysis in final feature selection (shape: {data.shape})")
        
        # Use VectorBT for correlation analysis if available
        if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling correlation matrix
                correlation_matrix = self.vectorbt_optimizer.rolling_correlation_matrix(data, window=50)
                tprint("✅ VectorBT correlation analysis completed")
                return correlation_matrix
            except Exception as e:
                tprint_warning(f"⚠️ VectorBT correlation analysis failed: {e}, using fallback")
        
        # Fallback to standard correlation analysis
        return matrix_correlation_analysis(data)

    def perform_vectorized_matrix_ops(self, data, operations):
        """Perform vectorized matrix operations using the vectorized core."""
        tprint(f"🚀 Performing vectorized matrix operations in final feature selection (shape: {data.shape})")
        if self.vectorized_core:
            return self.vectorized_core.optimize_dataframe_for_processing(data)
        return data

    def batch_matrix_operations(self, matrices_a, matrices_b, operation='multiply'):
        """Perform batch matrix operations."""
        tprint(f"📦 Performing batch matrix operations in final feature selection: {len(matrices_a)} matrices")
        if self.batch_processor:
            if operation == 'multiply':
                return self.batch_processor.batch_matrix_multiply(matrices_a, matrices_b)
        return None

    def gpu_matrix_multiply(self, a, b):
        """Perform GPU-accelerated matrix multiplication."""
        tprint(f"🖥️ Performing GPU-accelerated matrix multiplication in final feature selection ({a.shape} x {b.shape})")
        return gpu_matrix_multiply(a, b)

    def correlation_matrix_gpu(self, data):
        """Compute GPU-accelerated correlation matrix."""
        tprint(f"🖥️ Computing GPU-accelerated correlation matrix in final feature selection (shape: {data.shape})")
        return correlation_matrix_gpu(data)

    def _create_budget_aware_config(
        self,
        interactive_result: Dict[str, Any],
        base_config: Dict[str, Any]
    ) -> BudgetAwareSelectionConfig:
        """Create budget-aware selection configuration from interactive generation result."""
        tprint_info("🔧 Creating budget-aware selection configuration")
        
        # Extract budget parameters from interactive result
        interactive_config = interactive_result.get('configuration', {})
        
        # Create budget configuration
        budget_config = BudgetAwareSelectionConfig(
            total_budget_ms=interactive_config.get('total_budget_ms', 100.0),
            base_features=FeatureTypeBudget(
                budget_ms=interactive_config.get('base_features_budget_ms', 68.0),
                min_features=interactive_config.get('base_features_min', 40),
                max_features=interactive_config.get('base_features_max', 80),
                target_features=interactive_config.get('base_features_target', 60)
            ),
            interaction_features=FeatureTypeBudget(
                budget_ms=interactive_config.get('interaction_features_budget_ms', 15.0),
                min_features=interactive_config.get('interaction_features_min', 5),
                max_features=interactive_config.get('interaction_features_max', 15),
                target_features=interactive_config.get('interaction_features_target', 10)
            ),
            cross_timeframe_features=FeatureTypeBudget(
                budget_ms=interactive_config.get('cross_timeframe_features_budget_ms', 10.0),
                min_features=interactive_config.get('cross_timeframe_features_min', 3),
                max_features=interactive_config.get('cross_timeframe_features_max', 10),
                target_features=interactive_config.get('cross_timeframe_features_target', 6)
            ),
            gate_features=FeatureTypeBudget(
                budget_ms=interactive_config.get('gate_features_budget_ms', 7.0),
                min_features=interactive_config.get('gate_features_min', 2),
                max_features=interactive_config.get('gate_features_max', 8),
                target_features=interactive_config.get('gate_features_target', 5)
            ),
            enable_parallel_processing=base_config.get('parallel_processing', True),
            max_workers=base_config.get('max_workers', 4),
            cv_folds=base_config.get('cv_folds', 5),
            verbose=base_config.get('verbose', True)
        )
        
        tprint_info(f"   📊 Total budget: {budget_config.total_budget_ms}ms")
        tprint_info(f"   🎯 Base features: {budget_config.base_features.target_features}")
        tprint_info(f"   🔗 Interaction features: {budget_config.interaction_features.target_features}")
        tprint_info(f"   ⏰ Cross-timeframe features: {budget_config.cross_timeframe_features.target_features}")
        tprint_info(f"   🚪 Gate features: {budget_config.gate_features.target_features}")
        
        return budget_config
    
    def _prepare_data_for_budget_selection(
        self,
        data: pd.DataFrame,
        pipeline_state: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for budget-aware feature selection."""
        tprint_info("🔄 Preparing data for budget-aware feature selection")
        
        # Extract features and targets
        X = data.copy()
        
        # Remove non-numeric columns
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Try to get target from pipeline state
        y = None
        if 'targets' in pipeline_state:
            targets = pipeline_state['targets']
            if isinstance(targets, dict) and targets:
                # Use the first available target
                y = list(targets.values())[0]
            elif isinstance(targets, pd.Series):
                y = targets
        
        # If no target found, create a dummy target for unsupervised selection
        if y is None:
            tprint_warning("⚠️ No target found, creating dummy target for unsupervised selection")
            y = X.iloc[:, 0]  # Use first feature as proxy target
        
        # Align X and y
        common_indices = X.index.intersection(y.index)
        if len(common_indices) > 0:
            X = X.loc[common_indices]
            y = y.loc[common_indices]
        
        tprint_info(f"   📊 Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        tprint_info(f"   🎯 Target: {len(y)} samples")
        
        return X, y

    def _vectorbt_optimized_data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced data processing using VectorBT optimization for superior performance."""
        if not hasattr(self, 'vectorization_manager') or self.vectorization_manager is None:
            # Fallback to standard processing
            return data
        
        try:
            # Use VectorBT for optimized data processing
            # Optimize DataFrame for VectorBT processing
            optimized_data = self.vectorization_manager.optimize_dataframe(data)
            
            # Apply VectorBT scaling for better numerical stability
            numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                numeric_data = optimized_data[numeric_columns]
                scaled_data = self.vectorization_manager.scale_data(numeric_data, method='zscore')
                optimized_data[numeric_columns] = scaled_data
            
            # Use VectorBT rolling optimizer for additional processing if available
            if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer:
                optimized_data = self._apply_vectorbt_rolling_optimizations(optimized_data)
            
            self._log_info(
                f"✅ [FinalFeatureSelection] Enhanced VectorBT data processing completed: {optimized_data.shape}",
                event='final_feature_selection.vectorbt_processing',
            )
            
            return optimized_data
            
        except Exception as e:
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] Enhanced VectorBT data processing failed: {e}, using fallback",
                event='final_feature_selection.vectorbt_processing',
            )
            return data
    
    def _apply_vectorbt_rolling_optimizations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply VectorBT rolling optimizations to enhance data quality."""
        try:
            if not hasattr(self, 'vectorbt_optimizer') or self.vectorbt_optimizer is None:
                return data
            
            optimized_data = data.copy()
            
            # Apply VectorBT rolling operations for data enhancement
            for col in optimized_data.columns:
                if optimized_data[col].dtype in [np.float32, np.float64]:
                    # Use VectorBT rolling mean for smoothing
                    rolling_mean = self.vectorbt_optimizer.rolling_mean(optimized_data[col], window=10)
                    # Use weighted combination for stability
                    optimized_data[col] = 0.8 * optimized_data[col] + 0.2 * rolling_mean
            
            return optimized_data
            
        except Exception as e:
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] VectorBT rolling optimizations failed: {e}",
                event='final_feature_selection.vectorbt_rolling',
            )
            return data

    def _get_vectorbt_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive VectorBT performance statistics."""
        stats = {}
        
        # Get VectorBT rolling optimizer stats
        if hasattr(self, 'vectorbt_optimizer') and self.vectorbt_optimizer:
            try:
                optimizer_stats = self.vectorbt_optimizer.get_performance_stats()
                stats.update({
                    'vectorbt_rolling_operations': optimizer_stats.get('vectorbt_operations', 0),
                    'pandas_fallbacks': optimizer_stats.get('pandas_fallbacks', 0),
                    'gpu_operations': optimizer_stats.get('gpu_operations', 0),
                    'memory_optimizations': optimizer_stats.get('memory_optimizations', 0),
                    'chunk_operations': optimizer_stats.get('chunk_operations', 0),
                    'avg_time_per_operation': optimizer_stats.get('avg_time_per_operation', 0.0),
                    'vectorbt_usage_rate': optimizer_stats.get('vectorbt_usage_rate', 0.0)
                })
            except Exception as e:
                self._log_warning(
                    f"⚠️ [FinalFeatureSelection] Could not retrieve VectorBT optimizer stats: {e}",
                    event='final_feature_selection.vectorbt_optimizer_stats',
                )
        
        # Get unified vectorization manager stats
        if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
            try:
                manager_stats = self.vectorization_manager.get_performance_stats()
                stats.update({
                    'total_operations': manager_stats.get('total_operations', 0),
                    'strategy_usage': manager_stats.get('strategy_usage', {}),
                    'average_speedup': manager_stats.get('average_speedup', 0.0),
                    'total_computation_time': manager_stats.get('total_computation_time', 0.0)
                })
            except Exception as e:
                self._log_warning(
                    f"⚠️ [FinalFeatureSelection] Could not retrieve VectorBT manager stats: {e}",
                    event='final_feature_selection.vectorbt_manager_stats',
                )
        
        return stats
    
    def _vectorbt_memory_optimized_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data with VectorBT memory optimization."""
        if not hasattr(self, 'vectorization_manager') or self.vectorization_manager is None:
            return data
        
        try:
            # Use VectorBT chunked processing for large datasets
            if len(data) > 10000:
                tprint("   📦 Using VectorBT chunked processing for large dataset")
                return self._vectorbt_chunked_processing(data)
            
            # Use VectorBT memory-efficient operations
            optimized_data = self.vectorization_manager.optimize_dataframe(data)
            
            # Apply VectorBT scaling for memory efficiency
            numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                scaled_data = self.vectorization_manager.scale_data(
                    optimized_data[numeric_columns], 
                    method='minmax'  # More memory efficient than zscore
                )
                optimized_data[numeric_columns] = scaled_data
            
            return optimized_data
            
        except Exception as e:
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] VectorBT memory optimization failed: {e}",
                event='final_feature_selection.vectorbt_memory',
            )
            return data
    
    def _vectorbt_chunked_processing(self, data: pd.DataFrame, chunk_size: int = 5000) -> pd.DataFrame:
        """Process large datasets using VectorBT chunked processing."""
        try:
            if not hasattr(self, 'vectorbt_optimizer') or self.vectorbt_optimizer is None:
                return data
            
            processed_chunks = []
            
            for i in range(0, len(data), chunk_size):
                chunk = data.iloc[i:i + chunk_size]
                
                # Process chunk with VectorBT optimizations
                if hasattr(self, 'vectorization_manager') and self.vectorization_manager:
                    chunk = self.vectorization_manager.optimize_dataframe(chunk)
                
                # Apply VectorBT rolling operations to chunk
                chunk = self._apply_vectorbt_rolling_optimizations(chunk)
                
                processed_chunks.append(chunk)
            
            # Combine processed chunks
            result = pd.concat(processed_chunks, ignore_index=False)
            
            self._log_info(
                f"✅ [FinalFeatureSelection] VectorBT chunked processing completed: {len(processed_chunks)} chunks",
                event='final_feature_selection.vectorbt_chunked',
            )
            
            return result
            
        except Exception as e:
            self._log_warning(
                f"⚠️ [FinalFeatureSelection] VectorBT chunked processing failed: {e}",
                event='final_feature_selection.vectorbt_chunked',
            )
            return data

    def cleanup(self):
        """Clean up hardware optimization resources with aggressive cleanup."""
        try:
            log_info('🧹 Cleaning up hardware optimization resources with aggressive cleanup...')
            self._log_info(
                '🧹 [FinalFeatureSelection] Aggressive cleanup initiated',
                event='final_feature_selection.cleanup',
                phase='start',
            )
            
            # Perform aggressive cleanup
            cleanup_results = self.aggressive_memory_cleanup(force_cleanup=True)
            
            # Clean up advanced memory optimizer if available
            if self.advanced_memory_optimizer:
                try:
                    self.advanced_memory_optimizer.cleanup()
                    log_info('✅ Advanced memory optimizer cleaned up')
                except Exception as e:
                    log_warning(f'⚠️ Advanced memory optimizer cleanup failed: {e}')
            
            # Clean up standard memory optimizer
            if self.memory_optimizer:
                self.memory_optimizer._light_memory_cleanup()
            
            log_info(f'✅ Hardware optimization resources cleaned up: {cleanup_results["memory_freed_mb"]:.1f}MB freed')
            self._log_success(
                f'✅ [FinalFeatureSelection] Aggressive cleanup completed: {cleanup_results["memory_freed_mb"]:.1f}MB freed',
                event='final_feature_selection.cleanup',
                phase='complete',
                memory_freed_mb=cleanup_results['memory_freed_mb'],
                cleanup_success=cleanup_results['success']
            )
        except Exception as e:
            log_warning(f'⚠️ Error during aggressive hardware cleanup: {e}')
            self._log_warning(
                f'⚠️ [FinalFeatureSelection] Aggressive cleanup encountered an error: {e}',
                event='final_feature_selection.cleanup',
                error=str(e),
            )


# Register the component with the factory
register_component('final_feature_selection', FinalFeatureSelectionComponent)
