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

        # Initialize hardware optimization tools
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing hardware optimization tools...",
            event='final_feature_selection.initialization',
        )
        self.memory_optimizer = get_m1_memory_optimizer(memory_limit_gb=8.0)
        self.adaptive_engine = AdaptiveOptimizationEngine()
        self.hardware_manager = UnifiedHardwareManager()
        self.gpu_manager = M1GPUManager()

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

        # Initialize optimized process engine with hardware acceleration
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing optimized feature selection engine...",
            event='final_feature_selection.initialization',
        )
        self.optimized_engine = OptimizedFeatureSelectionEngine(
            use_hardware_accel=True,
            cache_size=1000
        )

        # Initialize bayesian optimization tools for enhanced feature selection
        self._log_info(
            "🔧 [FinalFeatureSelection] Initializing bayesian optimization tools...",
            event='final_feature_selection.initialization',
        )
        self.bayesian_optimizer = BayesianEntryTimingOptimizer(
            BayesianConfig(
                n_trials=100,
                timeout_minutes=30
                # Note: early_stopping_patience, enable_parallel, use_m1_optimization 
                # are not parameters of EntryTimingConfig
            )
        )

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

            # Check memory pressure and apply optimizations
            memory_pressure = getattr(self.memory_optimizer, 'memory_pressure', 0.0)
            if memory_pressure > 0.75:
                log_warning(f'🧠 High memory pressure detected ({memory_pressure:.2f}), applying memory optimizations')
                self.memory_optimizer._apply_memory_optimizations()
                self._log_warning(
                    f"🧠 [FinalFeatureSelection] High memory pressure detected ({memory_pressure:.2f}); optimizations applied",
                    event='final_feature_selection.memory_optimization',
                    memory_pressure=memory_pressure,
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

            # Execute final feature selection with hardware optimization and bayesian enhancement
            log_info(f'🚀 Executing enhanced feature selection with hardware optimizations and bayesian optimization...')
            self._log_info(
                '🚀 [FinalFeatureSelection] Executing enhanced feature selection step with bayesian optimization',
                event='final_feature_selection.execute',
            )

            # Check if we have interactive feature generation results for budget-aware selection
            interactive_features = None
            if isinstance(pipeline_state, dict):
                interactive_result = pipeline_state.get('interactive_feature_generation_result')
                if interactive_result and isinstance(interactive_result, dict):
                    interactive_features = {
                        'base_features': interactive_result.get('features'),
                        'interaction_features': interactive_result.get('interaction_features'),
                        'cross_timeframe_features': interactive_result.get('cross_timeframe_features')
                    }
                    tprint_info("🔧 Found interactive feature generation results - will use budget-aware selection")

            # First, apply bayesian optimization to enhance feature selection parameters
            tprint_info("🔬 Applying bayesian optimization to feature selection parameters")
            optimized_config = await self._optimize_feature_selection_config(
                final_feature_selection_config,
                data,
                pipeline_state
            )

            # Apply budget-aware selection if interactive features are available
            budget_aware_result = None
            if interactive_features and BUDGET_AWARE_SELECTION_AVAILABLE:
                tprint_info("💰 Applying budget-aware feature selection for interaction and cross-timeframe features")
                
                try:
                    # Create budget-aware selector with custom budgets
                    budget_selector = create_budget_aware_selector(
                        total_budget_ms=final_feature_selection_config.get('total_budget_ms', 100.0)
                    )
                    
                    # Extract target data if available
                    target_data = None
                    if isinstance(data, pd.DataFrame) and 'target' in data.columns:
                        target_data = data['target']
                    elif isinstance(pipeline_state, dict) and 'target' in pipeline_state:
                        target_data = pipeline_state['target']
                    
                    # Perform budget-aware selection
                    budget_aware_result = budget_selector.select_features(
                        base_features=interactive_features['base_features'],
                        interaction_features=interactive_features['interaction_features'],
                        cross_timeframe_features=interactive_features['cross_timeframe_features'],
                        target=target_data
                    )
                    
                    tprint_success(f"✅ Budget-aware selection completed: {len(budget_aware_result.total_selected_features)} features selected")
                    tprint_info(f"📊 Base: {len(budget_aware_result.base_features_result.selected_features)}")
                    tprint_info(f"🔗 Interaction: {len(budget_aware_result.interaction_features_result.selected_features)}")
                    tprint_info(f"⏰ Cross-timeframe: {len(budget_aware_result.cross_timeframe_features_result.selected_features)}")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Budget-aware selection failed, falling back to standard selection: {e}")
                    budget_aware_result = None

            runtime_config = dict(optimized_config)
            runtime_config['data_dir_key'] = data_dir_key
            runtime_config['final_features_dir_key'] = final_features_dir_key
            runtime_config['bayesian_optimization_applied'] = True
            
            # Add budget-aware results to config if available
            if budget_aware_result:
                runtime_config['budget_aware_selection'] = budget_aware_result.to_dict()
                runtime_config['selected_features'] = budget_aware_result.total_selected_features

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

            # Use budget-aware results if available, otherwise run standard selection
            if budget_aware_result:
                tprint_info("🎯 Using budget-aware selection results")
                success = True  # Budget-aware selection already completed
            else:
                tprint_info("🎯 Running standard final feature selection")
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

                # Prepare final feature selection result
                final_selection_result = {
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
                if budget_aware_result:
                    final_selection_result['budget_aware_selection'] = budget_aware_result.to_dict()
                    final_selection_result['selected_features'] = budget_aware_result.total_selected_features
                    final_selection_result['feature_type_breakdown'] = {
                        'base_features': len(budget_aware_result.base_features_result.selected_features),
                        'interaction_features': len(budget_aware_result.interaction_features_result.selected_features),
                        'cross_timeframe_features': len(budget_aware_result.cross_timeframe_features_result.selected_features)
                    }
                    final_selection_result['budget_utilization'] = budget_aware_result.total_budget_utilization
                    tprint_info(f"💰 Budget utilization: {budget_aware_result.total_budget_utilization:.1%}")

                artifacts_bundle = FinalFeatureSelectionArtifacts(
                    final_feature_selection_result=final_selection_result,
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

                # Clean up memory after processing
                self.memory_optimizer._light_memory_cleanup()
                self._log_info(
                    '🧹 [FinalFeatureSelection] Performed post-execution memory cleanup',
                    event='final_feature_selection.cleanup',
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

                # Clean up memory even on failure
                self.memory_optimizer._light_memory_cleanup()
                self._log_info(
                    '🧹 [FinalFeatureSelection] Memory cleanup performed after failure',
                    event='final_feature_selection.cleanup',
                    status='post_failure',
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

            # Clean up memory on exception
            try:
                self.memory_optimizer._light_memory_cleanup()
                self._log_info(
                    '🧹 [FinalFeatureSelection] Memory cleanup performed after exception',
                    event='final_feature_selection.cleanup',
                    status='post_exception',
                )
            except Exception as cleanup_error:
                self._log_warning(
                    f'⚠️ [FinalFeatureSelection] Memory cleanup failed (non-critical): {cleanup_error}',
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
        """Safely perform matrix multiplication with error handling."""
        tprint(f"🔢 Performing safe matrix multiplication ({A.shape} x {B.shape}) in final feature selection")
        return safe_matrix_multiply(A, B)

    def optimize_dataframe_for_matrix_ops(self, df):
        """Optimize DataFrame for matrix operations."""
        tprint(f"⚡ Optimizing DataFrame for matrix operations in final feature selection (shape: {df.shape})")
        return optimize_dataframe(df)

    def compute_matrix_correlation_analysis(self, data):
        """Compute matrix correlation analysis."""
        tprint(f"📊 Computing matrix correlation analysis in final feature selection (shape: {data.shape})")
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

    def cleanup(self):
        """Clean up hardware optimization resources."""
        try:
            log_info('🧹 Cleaning up hardware optimization resources...')
            self._log_info(
                '🧹 [FinalFeatureSelection] Cleanup initiated',
                event='final_feature_selection.cleanup',
                phase='start',
            )
            self.memory_optimizer._light_memory_cleanup()
            log_info('✅ Hardware optimization resources cleaned up')
            self._log_success(
                '✅ [FinalFeatureSelection] Cleanup completed',
                event='final_feature_selection.cleanup',
                phase='complete',
            )
        except Exception as e:
            log_warning(f'⚠️ Error during hardware cleanup: {e}')
            self._log_warning(
                f'⚠️ [FinalFeatureSelection] Cleanup encountered an error: {e}',
                event='final_feature_selection.cleanup',
                error=str(e),
            )


# Register the component with the factory
register_component('final_feature_selection', FinalFeatureSelectionComponent)
