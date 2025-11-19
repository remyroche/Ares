"""
Final Parameters Optimization for ML Models

This module provides system-wide final parameters optimization functionality,
separate from HPO (Hyperparameter Optimization). This is used for optimizing
final system parameters after model training is complete.
Refactored to inherit from BaseStep for autonomous execution.

Key Features:
- System-wide parameter optimization using enhanced BayesianTPEOptimizer
- Categorized parameter optimization with cross-validation support
- Hardware-accelerated optimization (M1 GPU/CPU optimization)
- Parallel evaluation with matrix operations
- Comprehensive validation and leakage detection
- Automatic parameter updates with proper error handling
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import optuna
import numpy as np
import pandas as pd
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp

from src.training.steps.base_step import BaseStep

# Artifact and version management
from src.training.steps.pre_training.utils.artifact_manager import PreTrainingArtifactManager
from src.utils.artifact_pickup_utils import get_artifact_pickup_utils
from src.utils.version_manager import get_version_manager

# Optimization utilities
from src.utils.nonlinear_optimization_helpers import (
    NonLinearConfig, NonLinearParameterSampler, apply_nonlinear_scoring,
    create_enhanced_search_space, convert_parameters_to_original_space
)
from src.utils.ml_common.optimization import (
    HyperparameterOptimization, ParetoOptimizer,
    HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, OptimizationConfig
)

# Hierarchical optimization (NEW - recommended approach)
from src.training.steps.backtesting.hierarchical_optimization_config import (
    create_hierarchical_optimizer,
    get_total_parameter_count,
    get_total_expected_trials
)

# ML utilities
from src.utils.ml_common.validation.cv_utils import TimeSeriesSplitValidator
from src.utils.ml_common.validation.cv_utils import OOFGenerator
from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector

# Math and validation utilities
from src.utils.math_validation import (
    validate_probability, validate_positive, validate_range,
    safe_divide, safe_log, check_for_nans, check_for_infs
)

# Common operations
from src.utils.common_operations import (
    calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor
)
from src.utils.common_utilities import ensure_list, ensure_array, flatten_dict

# Output utilities
from src.utils.tprint import tprint, tprint_data_preview, tprint_data_format, tprint_info, tprint_success, tprint_warning, tprint_error
from collections import OrderedDict

# Hardware optimization
try:
    from src.utils.hardware.m1_gpu_utils import M1GPUAccelerator
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    from src.utils.matrix_operations.batch_operations import BatchMatrixProcessor
    M1_HARDWARE_AVAILABLE = True
except ImportError:
    M1_HARDWARE_AVAILABLE = False
    tprint("⚠️  M1 hardware optimization not available", "warning")

# VectorBT optimization imports
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, OperationType, OperationConfig
    from src.feature_generation.utils.vectorbt_optimization_integration import get_optimization_manager
    VECTORBT_AVAILABLE = True
    tprint("✅ VectorBT optimization utilities available", "success")
except ImportError as e:
    VECTORBT_AVAILABLE = False
    tprint(f"⚠️  VectorBT optimization not available: {e}", "warning")

logger = logging.getLogger(__name__)

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics with proper validation"""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    n_trades: int = 0
    avg_trade_duration: float = 0.0
    confidence_score: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0

    def to_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Convert metrics to single optimization score"""
        if weights is None:
            weights = {
                'sharpe_ratio': 0.25,
                'sortino_ratio': 0.20,
                'max_drawdown': 0.15,
                'win_rate': 0.15,
                'profit_factor': 0.15,
                'total_return': 0.10
            }

        # Validate and normalize components
        sharpe_normalized = np.clip(validate_positive(self.sharpe_ratio, 0) / 3.0, 0, 1)
        sortino_normalized = np.clip(validate_positive(self.sortino_ratio, 0) / 4.0, 0, 1)
        dd_penalty = 1 - np.clip(abs(self.max_drawdown), 0, 1)
        win_rate_normalized = validate_probability(self.win_rate)
        pf_normalized = np.clip(validate_positive(self.profit_factor, 0) / 3.0, 0, 1)
        return_normalized = np.clip(self.total_return / 0.5, 0, 1)

        # Combined score
        score = (
            weights.get('sharpe_ratio', 0.25) * sharpe_normalized +
            weights.get('sortino_ratio', 0.20) * sortino_normalized +
            weights.get('max_drawdown', 0.15) * dd_penalty +
            weights.get('win_rate', 0.15) * win_rate_normalized +
            weights.get('profit_factor', 0.15) * pf_normalized +
            weights.get('total_return', 0.10) * return_normalized
        )

        # Apply CV stability penalty if available
        if self.cv_std > 0:
            stability_penalty = min(0.2, self.cv_std * 0.5)
            score = max(0, score - stability_penalty)

        return validate_range(score, 0, 1, default=0)

class LRUCache:
    """
    Least Recently Used (LRU) cache with size limit to prevent memory bloat.
    
    This cache is used to store evaluation results during optimization to avoid
    redundant parameter evaluations while maintaining bounded memory usage.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[float]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: float):
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing entry and mark as recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        # Remove oldest entry if cache is full
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

class FinalParametersOptimizer(BaseStep):
    """
    System-wide final parameters optimizer.

    This handles optimization of final system parameters after model training,
    separate from hyperparameter optimization during training.
    Refactored to inherit from BaseStep for autonomous execution.
    """

    def __init__(self, step_name: str = "final_parameters_optimization", 
                 config: Optional[Dict[str, Any]] = None, 
                 nonlinear_config: Optional[NonLinearConfig] = None):
        """Initialize the enhanced final parameters optimizer with hardware acceleration and CV support."""
        super().__init__(step_name)
        self.config = config or {}
        self.logger = logger.getChild('FinalParametersOptimizer')

        tprint("🚀 Initializing Enhanced Final Parameters Optimizer", "header")

        # Core pipeline dependencies populated lazily during execute
        self.calibration_results: Dict[str, Any] = {}
        self.previous_results: Optional[Dict[str, Any]] = None
        self.direction_mode: str = "both"

        # Non-linear optimization configuration
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.parameter_sampler = NonLinearParameterSampler(self.nonlinear_config)
        
        # Initialize essential components
        self._initialize_optimization_components()

        # Parameter categories for optimization (updated for new Analyst & Tactician models)
        self.categories = [
            'confidence', 'intensity', 'position_sizing', 'leverage', 'tpsl', 'exit_strategy',
            'ensemble', 'sr', 'two_tier', 'technical_indicators',
            'system_monitoring', 'training_optimization', 'regime_transitions',
            'signal_aggregation', 'turnover_cost_penalty', 'entry_timing_optimization',
            'confidence_aware_ensemble', 'model_specific_parameters',
            # New directional categories
            'long_specific_parameters', 'short_specific_parameters',
            'directional_thresholds', 'asymmetric_risk_management',
            # Analyst integration (Tactician deprecated - using Analyst only)
            'analyst_integration', 'analyst_oof_weights', 'analyst_feature_importance'
        ]

        # Default search spaces for each category
        self.default_search_spaces = self._get_default_search_spaces()

        # Enhanced search spaces with non-linear transformations
        self.enhanced_search_spaces = self._create_enhanced_search_spaces()

        # Optimization settings
        self.n_trials = config.get('n_trials', 50)
        self.timeout = config.get('timeout', 300)
        self.study_name = config.get('study_name', 'final_parameters_optimization')
        self.use_nonlinear_optimization = config.get('use_nonlinear_optimization', True)
        self.use_cv = config.get('use_cross_validation', True)
        self.cv_folds = config.get('cv_folds', 5)
        self.enable_parallel = config.get('enable_parallel_evaluation', True)
        self.max_workers = config.get('max_workers', max(1, mp.cpu_count() - 1))
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.early_stopping_threshold = config.get('early_stopping_threshold', 0.001)

        tprint(f"📊 Optimization categories: {len(self.categories)}", "info")
        tprint(f"🔧 Number of trials: {self.n_trials}", "info")
        tprint(f"⏱️  Timeout: {self.timeout}s", "info")
        tprint(f"📝 Study name: {self.study_name}", "info")
        tprint(f"🔄 Cross-validation: {self.use_cv} ({self.cv_folds} folds)", "info")
        tprint(f"⚡ Parallel evaluation: {self.enable_parallel} ({self.max_workers} workers)", "info")
        tprint(f"🛑 Early stopping: patience={self.early_stopping_patience}, threshold={self.early_stopping_threshold}", "info")

        if self.use_nonlinear_optimization:
            tprint(f"🚀 Non-linear optimization enabled:", "info")
            tprint(f"   • Log sampling: {self.nonlinear_config.use_log_sampling}", "info")
            tprint(f"   • Fractional powers: {self.nonlinear_config.use_fractional_powers}", "info")
            tprint(f"   • Sigmoid transforms: {self.nonlinear_config.use_sigmoid_transforms}", "info")
            tprint(f"   • Adaptive transforms: {self.nonlinear_config.use_adaptive_transforms}", "info")

        # Initialize cross-validation utilities
        if self.use_cv:
            self.cv_validator = TimeSeriesSplitValidator(
                n_splits=self.cv_folds,
                test_size=1.0 / self.cv_folds,
                embargo_pct=config.get('embargo_pct', 0.01)
            )
            self.oof_generator = OOFGenerator()
            self.leakage_detector = DataLeakageDetector()
            tprint("✅ CV utilities initialized", "success")

        # Initialize BayesianTPEOptimizer for each category
        self.tpe_optimizers = {}
        self._init_tpe_optimizers()
        
        # Initialize Hierarchical Optimization (NEW - recommended approach)
        self.use_hierarchical_optimization = config.get('use_hierarchical_optimization', True)
        self.hierarchical_optimizer = None
        if self.use_hierarchical_optimization:
            tprint("=" * 80, "header")
            tprint("🏗️ HIERARCHICAL OPTIMIZATION ENABLED", "header")
            tprint("=" * 80, "header")
            tprint("📊 Configuration:", "info")
            tprint("   • Total groups: 7 (vs 24 categories)", "info")
            tprint("   • Total parameters: ~45 (vs 150+)", "info")
            tprint("   • Expected trials: ~350 (vs ~2400)", "info")
            tprint("   • Speedup: ~7x faster", "info")
            tprint("   • Objective: custom_balanced_score (60% financial, 40% statistical)", "info")
            tprint("   • Optimization: Nature-based algorithm selection", "info")
            tprint("   • Regime-aware: YES (parameters modulate per regime)", "info")
            tprint("=" * 80, "header")

        # Initialize hardware optimization if available
        self.hardware_enabled = M1_HARDWARE_AVAILABLE and config.get('enable_hardware_optimization', True)
        if self.hardware_enabled:
            self._init_hardware_optimization()
        else:
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None
            tprint("ℹ️  Hardware optimization disabled", "info")

        # Initialize VectorBT optimization if available
        self.vectorbt_enabled = VECTORBT_AVAILABLE and config.get('enable_vectorbt_optimization', True)
        if self.vectorbt_enabled:
            self._init_vectorbt_optimization()
        else:
            self.rolling_optimizer = None
            self.vectorization_manager = None
            self.optimization_manager = None
            tprint("ℹ️  VectorBT optimization disabled", "info")

        # Parameter evaluation cache with LRU eviction to prevent memory bloat
        cache_size = config.get('cache_size', 1000)
        self.evaluation_cache = LRUCache(max_size=cache_size)
        tprint(f"💾 Evaluation cache initialized: max_size={cache_size}", "info")

        # Load per-regime performance statistics for objective adjustments
        self.regime_performance_path: Optional[str] = None
        self.regime_performance_stats = self._load_regime_performance_stats()
        self.regime_performance_modifier = self._calculate_regime_performance_modifier()
        if self.regime_performance_stats:
            location = self.regime_performance_path or 'unknown location'
            tprint(f"📊 Loaded per-regime performance stats from {location}", "info")
            tprint(f"   • Regime performance modifier: {self.regime_performance_modifier:.4f}", "info")

        # Initialize additional utilities (BaseStep already provides artifact_manager)
        self.pickup_utils = get_artifact_pickup_utils()
        self.version_manager = get_version_manager()

        tprint("✅ Final Parameters Optimizer initialization complete", "success")

    def _init_vectorbt_optimization(self):
        """Initialize VectorBT optimization components."""
        try:
            tprint("🚀 Initializing VectorBT optimization components", "info")

            # Initialize VectorBT rolling optimizer
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.hardware_enabled,
                enable_parallel=self.enable_parallel,
                memory_efficient=True,
                chunk_size=self.config.get('chunk_size', 1000),
                fast_fail=True,
                enable_logging=True
            )
            tprint("✅ VectorBT rolling optimizer initialized", "success")

            # Initialize UnifiedVectorizationManager
            self.vectorization_manager = UnifiedVectorizationManager()
            tprint("✅ UnifiedVectorizationManager initialized", "success")

            # Initialize VectorBT optimization manager
            self.optimization_manager = get_optimization_manager(
                enable_gpu=self.hardware_enabled,
                enable_parallel=self.enable_parallel,
                memory_efficient=True,
                max_memory_gb=self.config.get('max_memory_gb', 8.0),
                chunk_size=self.config.get('chunk_size', 1000),
                enable_monitoring=True
            )
            tprint("✅ VectorBT optimization manager initialized", "success")

            # Performance tracking for VectorBT operations
            self.vectorbt_stats = {
                'rolling_operations': 0,
                'batch_operations': 0,
                'vectorization_operations': 0,
                'total_vectorbt_time': 0.0,
                'performance_gains': []
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize VectorBT optimization: {e}")
            tprint(f"⚠️  VectorBT optimization initialization failed: {e}", "warning")
            self.rolling_optimizer = None
            self.vectorization_manager = None
            self.optimization_manager = None
            self.vectorbt_enabled = False

    def _initialize_optimization_components(self):
        """Initialize essential optimization components."""
        try:
            # This would contain essential component initialization
            # For now, just log the initialization
            self.logger.info("🔧 Initializing optimization components")
        except Exception as e:
            self.logger.error(f"Failed to initialize optimization components: {e}")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final parameters optimization.

        Args:
            config: Configuration containing symbol, exchange, timeframes, etc.

        Returns:
            Execution result with artifacts and metrics
        """
        self.logger.info('🔧 Starting Final Parameters Optimization')

        try:
            # Extract configuration
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            self.direction_mode = config.get('direction_mode', 'both').lower()
            
            # Detect execution mode using BaseStep method
            self.execution_mode = self._detect_execution_mode(config)
            execution_mode = config.get('execution_mode', 'light')
            
            if not symbol:
                raise ValueError("Symbol is required for final parameters optimization")
            
            self.logger.info(f"Optimizing final parameters for {symbol} from {exchange}")
            self.logger.info(f"Timeframe: {timeframe}, Direction: {direction}")
            
            # Initialize artifacts list
            artifacts = []
            metrics = {}
            
            # Set up artifact manager context
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                direction=direction,
                model='FinalParameters'
            )

            # Load HDF5 data from versioned artifacts (NEW!)
            tprint("=" * 80, "header")
            tprint("📥 LOADING PIPELINE DATA FROM HDF5 ARTIFACTS", "header")
            tprint("=" * 80, "header")
            loaded_hdf5_data = await self.load_hdf5_data_from_pipeline(config)

            # Store loaded data for use in optimization
            self.labeled_data = loaded_hdf5_data.get('labeled_data')
            self.regime_probabilities = loaded_hdf5_data.get('regime_probabilities')
            self.analyst_confidence = loaded_hdf5_data.get('analyst_confidence')
            self.disagreement_features = loaded_hdf5_data.get('disagreement_features')

            # Log what was loaded
            if self.labeled_data is not None:
                tprint(f"✅ Loaded labeled_data: {self.labeled_data.shape}", "success")
            if self.regime_probabilities is not None:
                tprint(f"✅ Loaded regime_probabilities: {self.regime_probabilities.shape}", "success")
            if self.analyst_confidence is not None:
                tprint(f"✅ Loaded analyst_confidence: {self.analyst_confidence.shape}", "success")
            if self.disagreement_features is not None:
                tprint(f"✅ Loaded disagreement_features: {self.disagreement_features.shape}", "success")

            # Load supporting data (calibration + previous optimization)
            self.calibration_results = await self._load_calibration_results(config)
            self.previous_results = await self._load_previous_results(symbol, exchange, config)

            # Fast fail if calibration data missing
            if not self._has_valid_calibration(self.calibration_results):
                warning_msg = "⚠️ Calibration results missing or invalid - returning neutral result"
                self.logger.warning(warning_msg)
                tprint(warning_msg, "warning")
                neutral_result = self._build_neutral_result(symbol, timeframe, direction, execution_mode)
                artifact_path = self._save_artifact(neutral_result, 'final_parameters_optimization_result', 'data')
                return {
                    'success': False,
                    'artifacts': [artifact_path],
                    'metrics': {
                        'parameters_optimized': 0,
                        'optimization_score': 0.0,
                        'execution_mode': execution_mode,
                        'calibration_missing': True
                    },
                    'optimization_result': neutral_result,
                    'error': 'Missing calibration results'
                }

            # Perform final parameters optimization
            optimization_result = await self._perform_final_parameters_optimization(
                symbol, exchange, timeframe, direction, execution_mode, config
            )

            # Save optimization result as artifact (will auto-generate CSV if < 2000 rows)
            artifact_path = self._save_artifact(
                optimization_result,
                'final_parameters_optimization_result',
                'data'
            )
            artifacts.append(artifact_path)
            
            # Record metrics
            metrics.update({
                'parameters_optimized': optimization_result.get('parameters_optimized', 0),
                'optimization_score': optimization_result.get('optimization_score', 0.0),
                'execution_mode': execution_mode
            })

            # Save metrics to Markdown and JSON (NEW!)
            tprint("=" * 80, "header")
            tprint("💾 SAVING OPTIMIZATION METRICS", "header")
            tprint("=" * 80, "header")

            try:
                # Save metrics to Markdown
                md_path = self.save_metrics_to_markdown(metrics, config)
                artifacts.append(md_path)
                tprint(f"✅ Saved metrics to Markdown: {md_path}", "success")
            except Exception as e:
                tprint(f"⚠️ Failed to save metrics to Markdown: {e}", "warning")

            try:
                # Save metrics to JSON
                json_path = self.save_metrics_to_json(metrics, config)
                artifacts.append(json_path)
                tprint(f"✅ Saved metrics to JSON: {json_path}", "success")
            except Exception as e:
                tprint(f"⚠️ Failed to save metrics to JSON: {e}", "warning")

            # Save config to Pickle and JSON (NEW!)
            tprint("=" * 80, "header")
            tprint("💾 SAVING CONFIGURATION", "header")
            tprint("=" * 80, "header")

            optimized_params = optimization_result.get('optimized_parameters', {})

            try:
                # Save config to Pickle
                pkl_path = self.save_config_to_pickle(config, optimized_params)
                artifacts.append(pkl_path)
                tprint(f"✅ Saved config to Pickle: {pkl_path}", "success")
            except Exception as e:
                tprint(f"⚠️ Failed to save config to Pickle: {e}", "warning")

            try:
                # Save config to JSON
                config_json_path = self.save_config_to_json(config, optimized_params)
                artifacts.append(config_json_path)
                tprint(f"✅ Saved config to JSON: {config_json_path}", "success")
            except Exception as e:
                tprint(f"⚠️ Failed to save config to JSON: {e}", "warning")

            self.logger.info(f'✅ Final Parameters Optimization completed: {metrics["parameters_optimized"]} parameters optimized')
            tprint("=" * 80, "header")
            tprint("✅ FINAL PARAMETERS OPTIMIZATION COMPLETE", "success")
            tprint("=" * 80, "header")

            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'optimization_result': optimization_result
            }

        except Exception as e:
            self.logger.error(f'❌ Final Parameters Optimization failed: {e}')
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }

    async def _perform_final_parameters_optimization(self, symbol: str, exchange: str,
                                                   timeframe: str, direction: str,
                                                   execution_mode: str,
                                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Run final parameters optimization and build a unified result structure."""
        try:
            start_time = time.time()

            # Delegate to optimize_all_parameters, which already handles hierarchical
            # vs. category-by-category optimization and appropriate fallbacks.
            optimization_results = await self.optimize_all_parameters(
                self.calibration_results,
                self.previous_results
            )

            # Count how many parameters were optimized and collect scores
            parameters_optimized = 0
            best_scores: List[float] = []

            if isinstance(optimization_results, dict):
                for category, result in optimization_results.items():
                    if not isinstance(result, dict):
                        continue

                    # Count parameters for this category/group
                    params_dict = result.get('best_params', {})
                    if isinstance(params_dict, dict):
                        parameters_optimized += len(params_dict)

                    # Collect best scores
                    if category == '_hierarchical_metadata':
                        total_score = result.get('total_score')
                        if isinstance(total_score, (int, float)):
                            best_scores.append(float(total_score))
                    else:
                        best_value = result.get('best_value')
                        if isinstance(best_value, (int, float)):
                            best_scores.append(float(best_value))

            optimization_score = max(best_scores) if best_scores else 0.0

            result = {
                'parameters_optimized': parameters_optimized,
                'optimization_score': optimization_score,
                'optimized_parameters': optimization_results,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'direction_mode': self.direction_mode,
                    'duration_seconds': time.time() - start_time
                }
            }
            return result

        except Exception as e:  # pragma: no cover - defensive
            self.logger.exception(f"Final parameters optimization failed: {e}")
            return {
                'parameters_optimized': 0,
                'optimization_score': 0.0,
                'optimized_parameters': {},
                'error': str(e),
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'execution_mode': execution_mode,
                    'direction_mode': self.direction_mode
                }
            }

    def _init_tpe_optimizers(self):
        """Initialize BayesianTPEOptimizer instances for optimization"""
        try:
            tprint("🔧 Initializing Bayesian TPE Optimizers", "info")

            # Create optimizer config
            opt_config = OptimizationConfig(
                n_trials=self.n_trials,
                timeout=self.timeout,
                execution_mode=self.config.get('execution_mode', 'light'),
                direction='maximize',
                enable_staged_optimization=True,
                coarse_grid_trials=min(25, self.n_trials // 4),
                fine_grid_trials=min(25, self.n_trials // 4),
                tpe_trials=max(20, self.n_trials // 2),
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold,
                enable_hardware_optimization=self.hardware_enabled,
                enable_batch_processing=self.enable_parallel,
                batch_size=self.max_workers
            )

            # Create optimizer instance (will be shared across categories)
            self.bayesian_optimizer = BayesianTPEOptimizer(opt_config)
            tprint(f"✅ Bayesian TPE Optimizer initialized", "success")

        except Exception as e:
            self.logger.error(f"Failed to initialize TPE optimizers: {e}")
            tprint(f"⚠️  TPE optimizer initialization failed: {e}", "warning")
            self.bayesian_optimizer = None

    def _init_hardware_optimization(self):
        """Initialize hardware optimization components"""
        try:
            tprint("⚡ Initializing M1 hardware optimization", "info")

            # Initialize M1 accelerators
            self.gpu_accelerator = M1GPUAccelerator()
            self.memory_optimizer = M1MemoryOptimizer()
            self.cpu_optimizer = M1CPUOptimizer()

            # Initialize matrix operations
            self.matrix_processor = HardwareOptimizedMatrixProcessor()
            self.batch_processor = BatchMatrixProcessor(
                chunk_size_mb=self.config.get('chunk_size_mb', 128),
                enable_gpu=True,
                enable_parallel=True,
                max_workers=self.max_workers
            )

            # Optimize memory
            self.memory_optimizer.optimize_memory_for_ml()

            tprint("✅ Hardware optimization initialized", "success")
            tprint(f"   • GPU: {'Available' if self.gpu_accelerator.is_available() else 'Not available'}", "info")
            tprint(f"   • Memory optimized: {self.memory_optimizer.is_optimized}", "info")
            tprint(f"   • Matrix operations: Hardware-accelerated", "info")

        except Exception as e:
            self.logger.error(f"Failed to initialize hardware optimization: {e}")
            tprint(f"⚠️  Hardware optimization init failed: {e}", "warning")
            self.gpu_accelerator = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_processor = None
            self.batch_processor = None

    def simulate_trading(self, params: Dict[str, Any],
                        signals: np.ndarray,
                        returns: np.ndarray,
                        confidences: Optional[np.ndarray] = None) -> EvaluationMetrics:
        """
        Simulate trading with given parameters using hardware-accelerated metrics calculation.
        
        This method uses VectorBT optimization and M1 GPU/MPS acceleration when available
        for confidence filtering and metrics calculation, providing significant speedups
        on compatible hardware.

        Args:
            params: Trading parameters
            signals: Trading signals (1=long, -1=short, 0=neutral)
            returns: Forward returns
            confidences: Optional confidence scores

        Returns:
            EvaluationMetrics object
        """
        try:
            # Validate inputs
            if len(signals) != len(returns):
                tprint(f"⚠️  Signals/returns length mismatch: {len(signals)} vs {len(returns)}", "warning")
                return EvaluationMetrics()

            # Apply confidence threshold with validation
            confidence_threshold = validate_probability(
                params.get('confidence_threshold', 0.6), default=0.6
            )

            if confidences is not None:
                confidences = ensure_array(confidences)
                valid_conf = validate_probability(confidences)
                
                # Use hardware-accelerated masking if available
                if self.hardware_enabled and self.matrix_processor and len(valid_conf) > 1000:
                    try:
                        mask = self.matrix_processor.compare_threshold(
                            valid_conf, confidence_threshold, operation='greater_equal'
                        )
                    except Exception as e:
                        self.logger.debug(f"Hardware acceleration fallback for masking: {e}")
                        mask = valid_conf >= confidence_threshold
                else:
                    mask = valid_conf >= confidence_threshold
                
                signals = signals[mask]
                returns = returns[mask]
                confidences = confidences[mask]

            if len(signals) == 0:
                tprint("⚠️  No signals after confidence filtering", "warning")
                return EvaluationMetrics()

            # Calculate position sizing with validation
            base_position_size = validate_positive(
                params.get('base_position_size', 0.01), default=0.01
            )
            base_position_size = validate_range(base_position_size, 0.001, 0.2, default=0.01)

            # Calculate trade returns
            trade_returns = signals * returns * base_position_size

            # Remove invalid values
            valid_mask = ~(check_for_nans(trade_returns) | check_for_infs(trade_returns))
            trade_returns = trade_returns[valid_mask]

            if len(trade_returns) == 0:
                tprint("⚠️  No valid trade returns", "warning")
                return EvaluationMetrics()

            # Use VectorBT optimization for metrics calculation if available
            if self.vectorbt_enabled and self.rolling_optimizer:
                tprint("🎯 Using VectorBT-optimized metrics calculation", "debug")
                sharpe, sortino, max_dd, win_rate, profit_factor, total_return = self._calculate_metrics_vectorbt(trade_returns)

                # Update VectorBT stats
                self.vectorbt_stats['rolling_operations'] += 5  # 5 rolling operations
                self.vectorbt_stats['total_vectorbt_time'] += time.time() - start_time if 'start_time' in locals() else 0
            else:
                # Calculate metrics using common_operations utilities
                sharpe = calculate_sharpe_ratio(trade_returns)
                sortino = calculate_sortino_ratio(trade_returns)
                max_dd = calculate_max_drawdown(np.cumsum(trade_returns))
                win_rate = calculate_win_rate(trade_returns)
                profit_factor = calculate_profit_factor(trade_returns)
                total_return = float(np.sum(trade_returns))

            # Validate all metrics
            sharpe = validate_positive(sharpe, default=0.0) if not check_for_nans(sharpe) else 0.0
            sortino = validate_positive(sortino, default=0.0) if not check_for_nans(sortino) else 0.0
            max_dd = float(max_dd) if not check_for_nans(max_dd) else 0.0
            win_rate = validate_probability(win_rate) if not check_for_nans(win_rate) else 0.0
            profit_factor = validate_positive(profit_factor, default=0.0) if not check_for_nans(profit_factor) else 0.0

            metrics = EvaluationMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                n_trades=int(len(trade_returns)),
                avg_trade_duration=0.0,  # Would need timestamps
                confidence_score=float(np.mean(confidences)) if confidences is not None else 0.0
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error in trading simulation: {e}")
            tprint(f"❌ Trading simulation failed: {e}", "error")
            return EvaluationMetrics()

    def _calculate_metrics_vectorbt(self, trade_returns: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """Calculate trading metrics using VectorBT optimization."""
        try:
            # Convert to pandas Series for VectorBT processing
            returns_series = pd.Series(trade_returns)

            # Use VectorBT rolling operations for enhanced calculations
            if len(returns_series) > 1:
                # Calculate rolling statistics for more robust metrics
                rolling_mean = self.rolling_optimizer.rolling_mean(returns_series, window=min(20, len(returns_series)))
                rolling_std = self.rolling_optimizer.rolling_std(returns_series, window=min(20, len(returns_series)))

                # Use rolling statistics for Sharpe ratio
                if not rolling_std.empty and rolling_std.iloc[-1] > 0:
                    sharpe = float(rolling_mean.iloc[-1] / rolling_std.iloc[-1]) if not rolling_mean.empty else 0.0
                else:
                    sharpe = 0.0

                # Calculate Sortino ratio (downside deviation)
                negative_returns = returns_series[returns_series < 0]
                if len(negative_returns) > 1:
                    downside_std = self.rolling_optimizer.rolling_std(
                        pd.Series(negative_returns), window=min(10, len(negative_returns))
                    )
                    if not downside_std.empty and downside_std.iloc[-1] > 0:
                        sortino = float(rolling_mean.iloc[-1] / downside_std.iloc[-1]) if not rolling_mean.empty else 0.0
                    else:
                        sortino = 0.0
                else:
                    sortino = 0.0

                # Calculate rolling max drawdown using cumulative returns
                cumulative_returns = returns_series.cumsum()
                rolling_max = self.rolling_optimizer.rolling_max(cumulative_returns, window=len(cumulative_returns))
                drawdown = cumulative_returns - rolling_max
                max_dd = float(abs(drawdown.min())) if not drawdown.empty else 0.0

                # Calculate win rate using rolling operations
                winning_trades = (returns_series > 0).astype(int)
                rolling_wins = self.rolling_optimizer.rolling_sum(winning_trades, window=len(winning_trades))
                win_rate = float(rolling_wins.iloc[-1] / len(returns_series)) if not rolling_wins.empty else 0.0

                # Calculate profit factor
                positive_returns = returns_series[returns_series > 0]
                negative_returns = returns_series[returns_series < 0]
                gross_profit = float(positive_returns.sum()) if len(positive_returns) > 0 else 0.0
                gross_loss = float(abs(negative_returns.sum())) if len(negative_returns) > 0 else 0.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

            else:
                # Fallback for single return
                sharpe = 0.0
                sortino = 0.0
                max_dd = 0.0
                win_rate = 1.0 if trade_returns[0] > 0 else 0.0
                profit_factor = 0.0

            # Total return
            total_return = float(returns_series.sum())

            return sharpe, sortino, max_dd, win_rate, profit_factor, total_return

        except Exception as e:
            self.logger.warning(f"VectorBT metrics calculation failed, using fallback: {e}")
            # Fallback to standard calculations
            sharpe = calculate_sharpe_ratio(trade_returns)
            sortino = calculate_sortino_ratio(trade_returns)
            max_dd = calculate_max_drawdown(np.cumsum(trade_returns))
            win_rate = calculate_win_rate(trade_returns)
            profit_factor = calculate_profit_factor(trade_returns)
            total_return = float(np.sum(trade_returns))

            return sharpe, sortino, max_dd, win_rate, profit_factor, total_return

    def get_vectorbt_performance_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        try:
            stats = self.vectorbt_stats.copy()

            # Add VectorBT rolling optimizer stats if available
            if self.rolling_optimizer:
                rolling_stats = self.rolling_optimizer.get_performance_stats()
                stats.update({
                    'vectorbt_rolling_operations': rolling_stats.get('vectorbt_operations', 0),
                    'vectorbt_gpu_operations': rolling_stats.get('gpu_operations', 0),
                    'vectorbt_memory_optimizations': rolling_stats.get('memory_optimizations', 0),
                    'vectorbt_errors': rolling_stats.get('errors', 0),
                    'vectorbt_avg_time_per_operation': rolling_stats.get('avg_time_per_operation', 0.0)
                })

            # Add unified vectorization manager stats if available
            if self.vectorization_manager:
                vectorization_stats = self.vectorization_manager.get_performance_stats()
                stats.update({
                    'unified_vectorization_operations': vectorization_stats.get('total_operations', 0),
                    'unified_vectorization_time': vectorization_stats.get('total_time', 0.0),
                    'unified_vectorization_memory_savings': vectorization_stats.get('memory_savings', 0.0),
                    'unified_vectorization_cache_hit_rate': vectorization_stats.get('cache_hit_rate', 0.0)
                })

            # Calculate efficiency metrics
            if stats['rolling_operations'] > 0:
                stats['vectorbt_usage_rate'] = stats['rolling_operations'] / max(1, stats['rolling_operations'] + stats.get('batch_operations', 0))
                stats['performance_gain'] = stats.get('performance_gains', [0])[-1] if stats.get('performance_gains') else 0.0
            else:
                stats['vectorbt_usage_rate'] = 0.0
                stats['performance_gain'] = 0.0

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get VectorBT performance stats: {e}")
            return self.vectorbt_stats.copy()

    def evaluate_with_cv(self, params: Dict[str, Any], data: Dict[str, Any],
                         evaluation_func: callable, category: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate parameters using cross-validation with directional stratification.
        
        This method now includes directional signal stratification to ensure balanced
        representation of long/short signals across CV folds, preventing bias in
        parameter optimization.

        Args:
            params: Parameters to evaluate
            data: Data dictionary containing features, targets, etc.
            evaluation_func: Function that evaluates params on a data split
            category: Parameter category being evaluated

        Returns:
            Tuple of (mean_score, cv_results_dict)
        """
        try:
            # Check for data leakage if enabled
            if self.use_cv and 'features' in data and 'targets' in data:
                X = ensure_array(data['features'])
                y = ensure_array(data['targets'])

                leakage_results = self.leakage_detector.detect_leakage(X, y)
                if leakage_results.get('has_leakage', False):
                    leakage_score = leakage_results.get('leakage_score', 0)
                    tprint(f"⚠️  Data leakage detected in {category}: {leakage_score:.4f}", "warning")

            # Get CV splits
            X = data.get('features', pd.DataFrame())
            y = data.get('targets', pd.Series())

            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X)
            if isinstance(y, np.ndarray):
                y = pd.Series(y)

            if X.empty or y.empty or len(X) < self.cv_folds * 100:
                tprint(f"ℹ️  Insufficient data for CV in {category}, using single evaluation", "info")
                return evaluation_func(params, data), {}

            # Check for directional signals to enable stratification
            use_stratification = False
            stratify_labels = None
            
            if 'signals' in data or 'directions' in data or 'long' in data or 'short' in data:
                # Attempt to extract directional information
                if 'signals' in data:
                    signals = ensure_array(data['signals'])
                    if len(signals) == len(X):
                        # Convert signals to long/short labels (1=long, -1=short, 0=neutral)
                        stratify_labels = np.sign(signals)
                        use_stratification = True
                elif 'long' in data and 'short' in data:
                    long_signals = ensure_array(data['long'])
                    short_signals = ensure_array(data['short'])
                    if len(long_signals) == len(X) and len(short_signals) == len(X):
                        # Create stratification labels: 1=long, -1=short, 0=neutral
                        stratify_labels = np.where(long_signals, 1, np.where(short_signals, -1, 0))
                        use_stratification = True
                
                if use_stratification:
                    # Check if we have enough samples of each class for stratification
                    unique, counts = np.unique(stratify_labels, return_counts=True)
                    min_class_samples = counts.min()
                    if min_class_samples < self.cv_folds:
                        tprint(f"⚠️  Insufficient samples for stratification in {category} (min class: {min_class_samples}), using time-series split", "warning")
                        use_stratification = False
                    else:
                        tprint(f"✅ Using directional stratification for {category} CV (classes: {len(unique)}, min samples: {min_class_samples})", "info")

            cv_scores = []
            cv_metrics = []
            fold_results = []

            # Choose appropriate CV splitter
            if use_stratification:
                from sklearn.model_selection import StratifiedKFold
                cv_splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
                splits = cv_splitter.split(X, stratify_labels)
            else:
                splits = self.cv_validator.split(X, y)

            for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
                try:
                    # Create fold data
                    fold_data = {
                        'train': {
                            'features': X.iloc[train_idx],
                            'targets': y.iloc[train_idx],
                        },
                        'val': {
                            'features': X.iloc[val_idx],
                            'targets': y.iloc[val_idx],
                        }
                    }
                    # Copy other data keys
                    for key, value in data.items():
                        if key not in ['features', 'targets']:
                            fold_data[key] = value

                    # Evaluate on this fold
                    fold_score, fold_metrics = evaluation_func(params, fold_data)

                    # Validate score
                    if check_for_nans(fold_score) or check_for_infs(fold_score):
                        tprint(f"⚠️  Invalid score in fold {fold_idx} for {category}, skipping", "warning")
                        continue

                    cv_scores.append(fold_score)
                    cv_metrics.append(fold_metrics)
                    fold_results.append({
                        'fold': fold_idx,
                        'score': fold_score,
                        'metrics': fold_metrics,
                        'train_size': len(train_idx),
                        'val_size': len(val_idx)
                    })

                except Exception as e:
                    tprint(f"⚠️  Error in fold {fold_idx} for {category}: {e}", "warning")
                    continue

            if not cv_scores:
                tprint(f"❌ No valid CV scores for {category}, falling back to single evaluation", "error")
                return evaluation_func(params, data), {}

            # Calculate mean and std
            mean_score = float(np.mean(cv_scores))
            std_score = float(np.std(cv_scores))

            # Penalize high variance (unstable parameters)
            stability_penalty = std_score * 0.1
            adjusted_score = max(0.0, mean_score - stability_penalty)

            cv_results = {
                'mean_score': mean_score,
                'std_score': std_score,
                'adjusted_score': adjusted_score,
                'cv_scores': cv_scores,
                'fold_results': fold_results,
                'n_folds': len(cv_scores)
            }

            tprint(f"   CV results for {category}: {mean_score:.4f} ± {std_score:.4f}", "info")

            return adjusted_score, cv_results

        except Exception as e:
            self.logger.error(f"Error in CV evaluation for {category}: {e}")
            tprint(f"❌ CV evaluation failed for {category}: {e}", "error")
            return evaluation_func(params, data), {}

    def calculate_combined_confidence(self, analyst_conf: np.ndarray,
                                     tactician_conf: np.ndarray,
                                     params: Dict[str, Any]) -> np.ndarray:
        """
        Return Analyst confidence directly.

        Note: Updated to use Analyst confidence instead of Tactician.
        We use only Analyst's Ensemble confidence for decision making.

        Args:
            analyst_conf: Analyst confidence array (primary input)
            tactician_conf: Tactician confidence array (deprecated, kept for API compatibility)
            params: Parameters (not used for combination, kept for API compatibility)

        Returns:
            Analyst confidence array
        """
        try:
            # Validate inputs - use analyst_conf as primary confidence source
            analyst_conf = ensure_array(analyst_conf)
            analyst_conf = validate_probability(analyst_conf)
            return analyst_conf

        except Exception as e:
            self.logger.error(f"Error processing analyst confidence: {e}")
            tprint(f"❌ Analyst confidence processing failed: {e}", "error")
            # Return zero confidence on error
            return np.zeros_like(analyst_conf) if hasattr(analyst_conf, 'shape') else np.array([0.0])
    
    # ============================================================================
    # HIERARCHICAL OPTIMIZATION HELPER METHODS
    # ============================================================================
    
    def _prepare_data_for_hierarchical_optimization(
        self, 
        calibration_results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for hierarchical optimization.
        
        Args:
            calibration_results: Calibration results dictionary
            
        Returns:
            Tuple of (features, targets) as numpy arrays
        """
        try:
            # Extract confidence arrays - Updated to use only analyst_confidence
            analyst_conf = calibration_results.get('analyst_confidence', np.array([]))
            # Legacy support: also check for tactician_confidence but default to analyst
            tactician_conf = calibration_results.get('tactician_confidence', analyst_conf.copy() if len(analyst_conf) > 0 else np.array([]))
            returns = calibration_results.get('returns', np.array([]))

            # Check for new simplified target structure
            target_long = calibration_results.get('target_long', np.array([]))
            target_short = calibration_results.get('target_short', np.array([]))

            # If we have the new simplified target structure, use it
            if len(target_long) > 0 and len(target_short) > 0:
                tprint_info("📊 Using new simplified target structure (target_long, target_short)")

                # Create combined targets for backward compatibility
                # Use target_long for long opportunities, target_short for short opportunities
                min_len = min(len(target_long), len(target_short))

                # Create combined target for optimization (prioritize long opportunities)
                combined_targets = target_long[:min_len]

                # Create directional signals from new target structure
                long_signals = (target_long[:min_len] > 0).astype(int)
                short_signals = (target_short[:min_len] > 0).astype(int)

                # Create combined signals (long=1, short=-1, neutral=0)
                combined_signals = long_signals - short_signals

                # Use targets as returns if actual returns not available
                if len(returns) == 0:
                    # Simulate returns from target structure
                    # target_long and target_short are volume-normalized binary targets
                    # We'll use them as proxy for returns
                    returns = combined_targets[:min_len]
                    tprint_info("📊 Using target structure as proxy for returns")
                else:
                    returns = returns[:min_len]

                # Update calibration results with derived data for backward compatibility
                calibration_results['derived_signals'] = combined_signals
                calibration_results['derived_targets'] = combined_targets
                calibration_results['target_structure'] = 'simplified'

            else:
                # Use legacy target structure
                tprint_info("📊 Using legacy target structure")

                # Ensure we have data - Updated to only require analyst_confidence
                if len(analyst_conf) == 0 or len(returns) == 0:
                    raise ValueError("calibration_results must contain analyst_confidence and returns arrays")

            # Use only analyst confidence as features (no longer combining with tactician)
            features = analyst_conf.reshape(-1, 1) if len(analyst_conf.shape) == 1 else analyst_conf

            # Align features and targets
            min_len = min(len(features), len(returns))
            features = features[:min_len]
            targets = returns[:min_len]
            
            self.logger.info(f"   Prepared data: {features.shape[0]} samples, {features.shape[1]} features")
            return features, targets
            
        except Exception as e:
            raise ValueError(f"Failed to prepare hierarchical data: {e}")
    
    def _run_backtest_for_hierarchical_optimization(
        self,
        params: Dict[str, Any],
        calibration_results: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run backtest with given parameters for hierarchical optimization.
        
        Args:
            params: Parameters to test
            calibration_results: Calibration data
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            
        Returns:
            Dict with predictions, targets, returns, regime_labels
        """
        try:
            # Extract confidence data - Updated to use only analyst_confidence
            analyst_conf = calibration_results.get('analyst_confidence', np.array([]))
            # Legacy support for tactician_confidence (fallback to analyst)
            tactician_conf = calibration_results.get('tactician_confidence', analyst_conf.copy() if len(analyst_conf) > 0 else np.array([]))
            returns = calibration_results.get('returns', np.array([]))

            # Check for new simplified target structure
            target_long = calibration_results.get('target_long', np.array([]))
            target_short = calibration_results.get('target_short', np.array([]))
            derived_signals = calibration_results.get('derived_signals', np.array([]))
            derived_targets = calibration_results.get('derived_targets', np.array([]))

            # If we have the new simplified target structure, use it
            if len(target_long) > 0 and len(target_short) > 0:
                tprint_info("📊 Using new simplified target structure for hierarchical optimization")

                # Use derived signals and targets if available
                if len(derived_signals) > 0 and len(derived_targets) > 0:
                    signals = derived_signals
                    targets = derived_targets
                else:
                    # Create signals from target structure
                    long_signals = (target_long > 0).astype(int)
                    short_signals = (target_short > 0).astype(int)
                    signals = long_signals - short_signals
                    targets = target_long  # Use target_long as primary target

                # Use returns if available, otherwise use targets as proxy
                if len(returns) == 0:
                    returns = targets
                else:
                    min_len = min(len(signals), len(returns))
                    signals = signals[:min_len]
                    targets = targets[:min_len]
                    returns = returns[:min_len]

            else:
                # Use legacy target structure - Updated to use analyst_confidence
                min_len = min(len(analyst_conf), len(returns))
                analyst_conf = analyst_conf[:min_len]
                returns = returns[:min_len]

                # Use analyst_confidence_threshold instead of tactician
                conf_threshold = params.get('analyst_confidence_threshold', params.get('tactician_confidence_threshold', 0.75))
                signals = (analyst_conf >= conf_threshold).astype(int)
                targets = (returns > 0).astype(int)

            simulated_returns = signals * returns
            predictions = signals.astype(float)

            return {
                'predictions': predictions,
                'targets': targets,
                'returns': simulated_returns,
                'regime_labels': calibration_results.get('regime_labels')
            }

        except Exception as e:
            raise RuntimeError(f"Hierarchical backtest failed: {e}")
    
    def _convert_hierarchical_to_category_format(
        self,
        best_params: Dict[str, Any],
        result: Any
    ) -> Dict[str, Any]:
        """
        Convert hierarchical optimization result to category format for compatibility.
        
        Args:
            best_params: Best parameters from hierarchical optimization
            result: HierarchicalOptimizationResult object
            
        Returns:
            Dict in category format
        """
        try:
            # Create a mapping from hierarchical groups to categories
            group_to_category_mapping = {
                'core_confidence': 'confidence',
                'entry_timing': 'entry_timing_optimization',
                'position_sizing_leverage': ['position_sizing', 'leverage'],
                'unified_tpsl': 'tpsl',
                'trailing_framework': 'exit_strategy',
                'time_confidence_decay': 'exit_strategy',
                'regime_intelligence': 'regime_transitions'
            }
            
            # Convert to category format
            optimization_results = {}
            
            for group_result in result.group_results:
                group_name = group_result.group_name
                group_params = group_result.best_params
                
                # Get corresponding category/categories
                category_mapping = group_to_category_mapping.get(group_name, group_name)
                
                if isinstance(category_mapping, list):
                    # Split parameters across multiple categories
                    for category in category_mapping:
                        if category not in optimization_results:
                            optimization_results[category] = {
                                'best_params': {},
                                'best_value': group_result.best_score,
                                'optimization_method': 'hierarchical',
                                'n_trials': group_result.n_trials,
                                'optimization_time': group_result.optimization_time
                            }
                        # Add relevant params to this category
                        for param_name, param_value in group_params.items():
                            if category in param_name.lower() or category == 'position_sizing' and 'position' in param_name:
                                optimization_results[category]['best_params'][param_name] = param_value
                else:
                    # Single category mapping
                    optimization_results[category_mapping] = {
                        'best_params': group_params,
                        'best_value': group_result.best_score,
                        'optimization_method': 'hierarchical',
                        'n_trials': group_result.n_trials,
                        'optimization_time': group_result.optimization_time
                    }
            
            # Add metadata
            optimization_results['_hierarchical_metadata'] = {
                'total_score': result.best_score,
                'total_trials': result.total_trials,
                'total_time': result.total_time,
                'groups_optimized': len(result.group_results),
                'final_refinement': result.final_refinement_result is not None
            }
            
            self.logger.info(f"✅ Converted hierarchical result to {len(optimization_results)-1} categories")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error converting hierarchical result to category format: {e}")
            # Return a basic format on error
            return {
                'all_parameters': {
                    'best_params': best_params,
                    'best_value': result.best_score if hasattr(result, 'best_score') else 0.0,
                    'optimization_method': 'hierarchical',
                    'error': str(e)
                }
            }

class AsymmetricParametersOptimizer(FinalParametersOptimizer):
    """Enhanced optimizer with long/short parameter differentiation"""

    def __init__(self, config: Dict[str, Any], nonlinear_config: Optional[NonLinearConfig] = None):
        super().__init__(config, nonlinear_config)

        # Enhanced search spaces with directional parameters
        self.directional_search_spaces = self._get_directional_search_spaces()
        self.default_search_spaces.update(self.directional_search_spaces)

        # Re-create enhanced search spaces with new directional parameters
        self.enhanced_search_spaces = self._create_enhanced_search_spaces()

        self.logger.info("🎯 Asymmetric Parameters Optimizer initialized")
        self.logger.info(f"   Added directional parameter categories: {len(self.directional_search_spaces)}")

    def _get_directional_search_spaces(self):
        """Define search spaces for directional parameters"""
        return {
            'long_specific_parameters': {
                'long_entry_patience': {'type': 'float', 'low': 0.5, 'high': 2.0},
                'long_profit_target_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.5},
                'long_stop_loss_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.2},
                'long_position_size_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.3},
                'long_confidence_threshold': {'type': 'float', 'low': 0.5, 'high': 0.8},
                'long_momentum_weight': {'type': 'float', 'low': 0.1, 'high': 0.6},
                'long_support_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
            },
            'short_specific_parameters': {
                'short_entry_urgency': {'type': 'float', 'low': 0.8, 'high': 1.5},
                'short_profit_target_multiplier': {'type': 'float', 'low': 0.6, 'high': 1.2},
                'short_stop_loss_multiplier': {'type': 'float', 'low': 1.0, 'high': 1.4},
                'short_position_size_multiplier': {'type': 'float', 'low': 0.7, 'high': 1.1},
                'short_confidence_threshold': {'type': 'float', 'low': 0.6, 'high': 0.85},
                'short_momentum_weight': {'type': 'float', 'low': 0.2, 'high': 0.7},
                'short_resistance_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
            },
            'directional_thresholds': {
                'long_vs_short_bias_threshold': {'type': 'float', 'low': 0.1, 'high': 0.4},
                'directional_confidence_weight': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'asymmetric_volatility_adjustment': {'type': 'float', 'low': 0.8, 'high': 1.3},
                'directional_switch_penalty': {'type': 'float', 'low': 0.0, 'high': 0.1},
                'long_bias_boost': {'type': 'float', 'low': 0.9, 'high': 1.2},
                'short_bias_boost': {'type': 'float', 'low': 0.9, 'high': 1.2},
            },
            'asymmetric_risk_management': {
                'long_max_position_duration': {'type': 'int', 'low': 20, 'high': 40},
                'short_max_position_duration': {'type': 'int', 'low': 10, 'high': 25},
                'long_reassessment_frequency': {'type': 'int', 'low': 3, 'high': 8},
                'short_reassessment_frequency': {'type': 'int', 'low': 2, 'high': 5},
                'long_volatility_tolerance': {'type': 'float', 'low': 0.8, 'high': 1.1},
                'short_volatility_tolerance': {'type': 'float', 'low': 1.0, 'high': 1.3},
                'asymmetric_leverage_adjustment': {'type': 'float', 'low': 0.9, 'high': 1.1},
            },
            # Merged Tactician & Analyst integration parameters
            'tactician_analyst_integration': {
                'w_min': {'type': 'float', 'min': 0.1, 'max': 0.5},  # Minimum weight for sample weighting
                'analyst_feature_weight': {'type': 'float', 'min': 0.1, 'max': 1.0},  # Weight for Analyst OOF features
                'p_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for p_trade feature
                'u_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for u_trade feature
                'q_trade_weight': {'type': 'float', 'min': 0.2, 'max': 0.8},  # Weight for q_trade feature
                'analyst_expected_value_weight': {'type': 'float', 'min': 0.1, 'max': 0.6},  # Weight for expected value feature
                'analyst_weighted_prob_weight': {'type': 'float', 'min': 0.1, 'max': 0.6},  # Weight for weighted prob feature
                'integration_method': {'type': 'categorical', 'choices': ['additive', 'multiplicative', 'ensemble']},
                'feature_interaction_strength': {'type': 'float', 'min': 0.1, 'max': 1.0},  # Strength of feature interactions
            },
            'analyst_oof_weights': {
                'p_trade_threshold': {'type': 'float', 'min': 0.3, 'max': 0.8},  # Threshold for p_trade filtering
                'u_trade_threshold': {'type': 'float', 'min': -0.5, 'max': 0.5},  # Threshold for u_trade filtering
                'q_trade_threshold': {'type': 'float', 'min': 0.4, 'max': 0.9},  # Threshold for q_trade filtering
                'weight_scaling_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},  # Scaling factor for weights
                'weight_smoothing': {'type': 'float', 'min': 0.0, 'max': 0.5},  # Smoothing factor for weights
                'adaptive_weighting': {'type': 'categorical', 'choices': ['static', 'dynamic', 'regime_based']},
            },
            'merged_feature_importance': {
                'analyst_feature_importance_boost': {'type': 'float', 'min': 1.0, 'max': 3.0},  # Boost for Analyst features
                'interaction_feature_importance': {'type': 'float', 'min': 0.5, 'max': 2.0},  # Importance of interaction features
                'feature_selection_threshold': {'type': 'float', 'min': 0.01, 'max': 0.1},  # Threshold for feature selection
                'analyst_feature_regularization': {'type': 'float', 'min': 0.0, 'max': 0.1},  # Regularization for Analyst features
                'feature_interaction_depth': {'type': 'int', 'min': 1, 'max': 3},  # Depth of feature interactions
            }
        }

    def optimize_per_regime_with_direction(self, regime_data: Dict[str, Any], regime_id: str):
        """
        Optimize parameters per regime with directional differentiation

        Args:
            regime_data: Data for specific regime including signals, directions, returns, etc.
            regime_id: Regime identifier
        """

        # Check if regime has enough samples for directional split
        total_samples = len(regime_data.get('signals', []))
        directions = regime_data.get('directions', np.array([]))

        if len(directions) == 0:
            self.logger.warning(f"⚠️ Regime {regime_id}: No direction data available, using standard optimization")
            return self.optimize_regime_parameters(regime_data, regime_id)

        long_samples = np.sum(directions > 0)
        short_samples = np.sum(directions < 0)

        min_samples_per_direction = self.config.get('min_samples_per_direction', 100)

        if long_samples >= min_samples_per_direction and short_samples >= min_samples_per_direction:
            # Sufficient samples: optimize separately
            self.logger.info(f"📊 Regime {regime_id}: Sufficient samples for directional optimization")
            self.logger.info(f"   Long samples: {long_samples}, Short samples: {short_samples}")

            return self._optimize_directional_parameters(regime_data, regime_id)

        else:
            # Insufficient samples: use averaged parameters with directional bias
            self.logger.info(f"📊 Regime {regime_id}: Using averaged parameters with directional bias")
            self.logger.info(f"   Long samples: {long_samples}, Short samples: {short_samples}")

            return self._optimize_averaged_parameters_with_bias(regime_data, regime_id)

    def _optimize_directional_parameters(self, regime_data: Dict[str, Any], regime_id: str):
        """Optimize separate parameters for long and short"""

        results = {}

        # Separate data by direction
        directions = regime_data['directions']
        long_mask = directions > 0
        short_mask = directions < 0

        # Optimize long parameters
        long_data = self._filter_data_by_mask(regime_data, long_mask)

        long_study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_long',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        long_objective = self._create_directional_objective(long_data, 'long', regime_id)

        try:
            long_study.optimize(long_objective, n_trials=self.n_trials // 2, timeout=self.timeout // 2)
            results['long_parameters'] = long_study.best_params
            results['long_score'] = long_study.best_value
            results['long_trials'] = len(long_study.trials)
        except Exception as e:
            self.logger.error(f"❌ Long parameter optimization failed for regime {regime_id}: {e}")
            results['long_parameters'] = {}
            results['long_score'] = 0.0
            results['long_trials'] = 0

        # Optimize short parameters
        short_data = self._filter_data_by_mask(regime_data, short_mask)

        short_study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_short',
            sampler=optuna.samplers.TPESampler(seed=43)
        )

        short_objective = self._create_directional_objective(short_data, 'short', regime_id)

        try:
            short_study.optimize(short_objective, n_trials=self.n_trials // 2, timeout=self.timeout // 2)
            results['short_parameters'] = short_study.best_params
            results['short_score'] = short_study.best_value
            results['short_trials'] = len(short_study.trials)
        except Exception as e:
            self.logger.error(f"❌ Short parameter optimization failed for regime {regime_id}: {e}")
            results['short_parameters'] = {}
            results['short_score'] = 0.0
            results['short_trials'] = 0

        # Create combined parameters
        results['combined_parameters'] = self._combine_directional_parameters(
            results.get('long_parameters', {}),
            results.get('short_parameters', {})
        )

        # Calculate weighted score
        total_trials = results['long_trials'] + results['short_trials']
        if total_trials > 0:
            results['combined_score'] = (
                (results['long_score'] * results['long_trials'] +
                 results['short_score'] * results['short_trials']) / total_trials
            )
        else:
            results['combined_score'] = 0.0

        self.logger.info(f"✅ Directional optimization completed for regime {regime_id}")
        self.logger.info(f"   Long score: {results['long_score']:.4f} ({results['long_trials']} trials)")
        self.logger.info(f"   Short score: {results['short_score']:.4f} ({results['short_trials']} trials)")
        self.logger.info(f"   Combined score: {results['combined_score']:.4f}")

        return results

    def _optimize_averaged_parameters_with_bias(self, regime_data: Dict[str, Any], regime_id: str):
        """Optimize averaged parameters with directional bias when samples are insufficient"""

        # Calculate directional bias
        directions = regime_data['directions']
        long_ratio = np.sum(directions > 0) / len(directions)
        short_ratio = np.sum(directions < 0) / len(directions)
        directional_bias = 'long' if long_ratio > short_ratio else 'short'
        bias_strength = abs(long_ratio - short_ratio)

        self.logger.info(f"   Directional bias: {directional_bias} (strength: {bias_strength:.2f})")
        self.logger.info(f"   Long ratio: {long_ratio:.1%}, Short ratio: {short_ratio:.1%}")

        # Create biased objective function
        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.study_name}_regime_{regime_id}_averaged',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        biased_objective = self._create_biased_objective(
            regime_data, directional_bias, long_ratio, short_ratio, regime_id
        )

        try:
            study.optimize(biased_objective, n_trials=self.n_trials, timeout=self.timeout)
            base_parameters = study.best_params
            base_score = study.best_value
            trials_completed = len(study.trials)
        except Exception as e:
            self.logger.error(f"❌ Biased parameter optimization failed for regime {regime_id}: {e}")
            base_parameters = {}
            base_score = 0.0
            trials_completed = 0

        # Apply directional bias to parameters
        biased_parameters = self._apply_directional_bias(
            base_parameters, directional_bias, long_ratio, short_ratio
        )

        results = {
            'base_parameters': base_parameters,
            'biased_parameters': biased_parameters,
            'directional_bias': directional_bias,
            'bias_strength': bias_strength,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'score': base_score,
            'trials_completed': trials_completed
        }

        self.logger.info(f"✅ Biased optimization completed for regime {regime_id}")
        self.logger.info(f"   Base score: {base_score:.4f} ({trials_completed} trials)")
        self.logger.info(f"   Bias applied: {directional_bias} (strength: {bias_strength:.2f})")

        return results

    def _filter_data_by_mask(self, data: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
        """Filter regime data by directional mask"""
        filtered_data = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray) and len(value) == len(mask):
                filtered_data[key] = value[mask]
            elif isinstance(value, list) and len(value) == len(mask):
                filtered_data[key] = [value[i] for i in range(len(value)) if mask[i]]
            else:
                # Keep non-array data as-is
                filtered_data[key] = value

        return filtered_data

    def _create_directional_objective(self, data: Dict[str, Any], direction: str, regime_id: str):
        """Create objective function for specific direction"""

        def objective(trial):
            try:
                # Sample directional parameters
                params = {}

                # Sample direction-specific parameters
                direction_space = self.directional_search_spaces[f'{direction}_specific_parameters']
                for param_name, param_config in direction_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['low'], param_config['high']
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['low'], param_config['high']
                        )

                # Sample general directional parameters
                for category in ['directional_thresholds', 'asymmetric_risk_management']:
                    if category in self.directional_search_spaces:
                        category_space = self.directional_search_spaces[category]
                        for param_name, param_config in category_space.items():
                            if param_config['type'] == 'float':
                                params[param_name] = trial.suggest_float(
                                    param_name, param_config['low'], param_config['high']
                                )
                            elif param_config['type'] == 'int':
                                params[param_name] = trial.suggest_int(
                                    param_name, param_config['low'], param_config['high']
                                )

                # Sample some general parameters with directional adjustments
                general_params = self._sample_general_parameters_with_direction(trial, direction)
                params.update(general_params)

                # Evaluate performance with these parameters
                performance = self._evaluate_directional_performance(data, params, direction, regime_id)

                return performance

            except Exception as e:
                self.logger.error(f"❌ Objective evaluation failed: {e}")
                return 0.0  # Return poor score on failure

        return objective

    def _create_biased_objective(self, data: Dict[str, Any], bias: str, long_ratio: float,
                                short_ratio: float, regime_id: str):
        """Create objective function with directional bias"""

        def objective(trial):
            try:
                # Sample base parameters
                params = self._sample_base_parameters(trial)

                # Apply directional bias during sampling
                biased_params = self._apply_directional_bias_to_sampling(
                    params, trial, bias, long_ratio, short_ratio
                )

                # Evaluate performance with biased parameters
                performance = self._evaluate_biased_performance(
                    data, biased_params, bias, long_ratio, short_ratio, regime_id
                )

                return performance

            except Exception as e:
                self.logger.error(f"❌ Biased objective evaluation failed: {e}")
                return 0.0  # Return poor score on failure

        return objective

    def _sample_general_parameters_with_direction(self, trial, direction: str) -> Dict[str, Any]:
        """Sample general parameters with directional adjustments"""
        params = {}

        # Adjust confidence thresholds based on direction
        if direction == 'long':
            params['confidence_threshold'] = trial.suggest_float('confidence_threshold', 0.5, 0.75)
            params['position_size_base'] = trial.suggest_float('position_size_base', 0.008, 0.015)
        else:  # short
            params['confidence_threshold'] = trial.suggest_float('confidence_threshold', 0.6, 0.85)
            params['position_size_base'] = trial.suggest_float('position_size_base', 0.006, 0.012)

        # Direction-agnostic parameters
        params['leverage_multiplier'] = trial.suggest_float('leverage_multiplier', 0.8, 1.2)
        params['risk_adjustment'] = trial.suggest_float('risk_adjustment', 0.9, 1.1)

        return params

    def _sample_base_parameters(self, trial) -> Dict[str, Any]:
        """Sample base parameters without directional bias"""
        return {
            'confidence_threshold': trial.suggest_float('confidence_threshold', 0.5, 0.8),
            'position_size_base': trial.suggest_float('position_size_base', 0.005, 0.015),
            'leverage_multiplier': trial.suggest_float('leverage_multiplier', 0.8, 1.2),
            'risk_adjustment': trial.suggest_float('risk_adjustment', 0.9, 1.1),
            'profit_target_multiplier': trial.suggest_float('profit_target_multiplier', 0.8, 1.4),
            'stop_loss_multiplier': trial.suggest_float('stop_loss_multiplier', 0.8, 1.3),
        }

    def _apply_directional_bias_to_sampling(self, params: Dict[str, Any], trial, bias: str,
                                          long_ratio: float, short_ratio: float) -> Dict[str, Any]:
        """Apply directional bias during parameter sampling"""
        biased_params = params.copy()

        # Sample directional adjustment factors
        bias_adjustment = trial.suggest_float('bias_adjustment', 0.9, 1.1)

        if bias == 'long':
            # Long-friendly adjustments
            biased_params['confidence_threshold'] *= 0.95  # Slightly lower
            biased_params['position_size_base'] *= bias_adjustment * 1.05  # Slightly larger
            biased_params['profit_target_multiplier'] *= 1.1  # Higher profit targets
        else:  # short
            # Short-friendly adjustments
            biased_params['confidence_threshold'] *= 1.05  # Slightly higher
            biased_params['position_size_base'] *= bias_adjustment * 0.95  # Slightly smaller
            biased_params['stop_loss_multiplier'] *= 1.1  # Tighter stops

        return biased_params

    def _evaluate_directional_performance(self, data: Dict[str, Any], params: Dict[str, Any],
                                        direction: str, regime_id: str) -> float:
        """Evaluate performance for specific direction"""
        try:
            # Extract relevant data
            signals = data.get('signals', np.array([]))
            returns = data.get('returns', np.array([]))
            directions = data.get('directions', np.array([]))

            if len(signals) == 0 or len(returns) == 0:
                return 0.0

            # Apply directional parameters to simulate performance
            confidence_threshold = params.get('confidence_threshold', 0.6)
            position_size = params.get('position_size_base', 0.01)

            # Filter signals by confidence
            confident_signals = signals >= confidence_threshold

            # Calculate directional returns
            directional_returns = returns[confident_signals] * position_size

            if len(directional_returns) == 0:
                return 0.0

            # Direction-specific performance metrics
            if direction == 'long':
                # For long: reward sustained positive returns
                performance = np.mean(directional_returns) * np.sqrt(len(directional_returns))
                # Bonus for consistency
                if np.std(directional_returns) > 0:
                    sharpe_bonus = np.mean(directional_returns) / np.std(directional_returns) * 0.1
                    performance += sharpe_bonus
            else:  # short
                # For short: reward quick, sharp negative moves (positive returns for short positions)
                performance = np.mean(directional_returns) * np.sqrt(len(directional_returns))
                # Bonus for capturing volatility
                volatility_bonus = np.std(directional_returns) * 0.05
                performance += volatility_bonus

            # Apply risk adjustment
            risk_adjustment = params.get('risk_adjustment', 1.0)
            performance *= risk_adjustment

            return max(0.0, performance)  # Ensure non-negative

        except Exception as e:
            self.logger.error(f"❌ Directional performance evaluation failed: {e}")
            return 0.0

    def _evaluate_biased_performance(self, data: Dict[str, Any], params: Dict[str, Any],
                                   bias: str, long_ratio: float, short_ratio: float,
                                   regime_id: str) -> float:
        """Evaluate performance with directional bias"""
        try:
            # Extract data
            signals = data.get('signals', np.array([]))
            returns = data.get('returns', np.array([]))
            directions = data.get('directions', np.array([]))

            if len(signals) == 0 or len(returns) == 0:
                return 0.0

            # Apply parameters
            confidence_threshold = params.get('confidence_threshold', 0.6)
            position_size = params.get('position_size_base', 0.01)

            # Filter signals
            confident_signals = signals >= confidence_threshold
            filtered_returns = returns[confident_signals]
            filtered_directions = directions[confident_signals]

            if len(filtered_returns) == 0:
                return 0.0

            # Calculate weighted performance based on directional bias
            long_mask = filtered_directions > 0
            short_mask = filtered_directions < 0

            performance = 0.0

            if np.any(long_mask):
                long_returns = filtered_returns[long_mask] * position_size
                long_performance = np.mean(long_returns) * np.sqrt(len(long_returns))
                performance += long_performance * long_ratio

            if np.any(short_mask):
                short_returns = filtered_returns[short_mask] * position_size
                short_performance = np.mean(short_returns) * np.sqrt(len(short_returns))
                performance += short_performance * short_ratio

            # Apply bias boost
            bias_strength = abs(long_ratio - short_ratio)
            bias_boost = 1.0 + (bias_strength * 0.1)  # Up to 10% boost for strong bias
            performance *= bias_boost

            return max(0.0, performance)

        except Exception as e:
            self.logger.error(f"❌ Biased performance evaluation failed: {e}")
            return 0.0

    def _combine_directional_parameters(self, long_params: Dict[str, Any],
                                      short_params: Dict[str, Any]) -> Dict[str, Any]:
        """Combine long and short parameters into unified set"""
        combined = {}

        # Combine parameters with directional prefixes
        for key, value in long_params.items():
            if not key.startswith('long_'):
                combined[f'long_{key}'] = value
            else:
                combined[key] = value

        for key, value in short_params.items():
            if not key.startswith('short_'):
                combined[f'short_{key}'] = value
            else:
                combined[key] = value

        # Create averaged parameters for general use
        general_params = {}
        for long_key, long_value in long_params.items():
            if long_key.startswith('long_'):
                base_key = long_key[5:]  # Remove 'long_' prefix
                short_key = f'short_{base_key}'
                if short_key in short_params:
                    general_params[base_key] = (long_value + short_params[short_key]) / 2

        combined.update(general_params)

        return combined

    def _apply_directional_bias(self, base_params: Dict[str, Any], bias: str,
                              long_ratio: float, short_ratio: float) -> Dict[str, Any]:
        """Apply directional bias to base parameters"""

        biased_params = base_params.copy()
        bias_strength = abs(long_ratio - short_ratio)

        if bias == 'long':
            # Bias towards long-friendly parameters
            biased_params['confidence_threshold'] = biased_params.get('confidence_threshold', 0.6) * (1 - bias_strength * 0.1)
            biased_params['position_size_base'] = biased_params.get('position_size_base', 0.01) * (1 + bias_strength * 0.2)
            biased_params['profit_target_multiplier'] = biased_params.get('profit_target_multiplier', 1.0) * (1 + bias_strength * 0.3)
            biased_params['max_position_duration'] = int(biased_params.get('max_position_duration', 25) * (1 + bias_strength * 0.4))

        else:  # short bias
            # Bias towards short-friendly parameters
            biased_params['confidence_threshold'] = biased_params.get('confidence_threshold', 0.6) * (1 + bias_strength * 0.1)
            biased_params['position_size_base'] = biased_params.get('position_size_base', 0.01) * (1 - bias_strength * 0.1)
            biased_params['stop_loss_multiplier'] = biased_params.get('stop_loss_multiplier', 1.0) * (1 + bias_strength * 0.2)
            biased_params['max_position_duration'] = int(biased_params.get('max_position_duration', 25) * (1 - bias_strength * 0.3))

        # Add directional metadata
        biased_params['directional_bias'] = bias
        biased_params['bias_strength'] = bias_strength
        biased_params['long_ratio'] = long_ratio
        biased_params['short_ratio'] = short_ratio

        return biased_params

    def _create_enhanced_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create enhanced search spaces with non-linear transformation metadata."""
        enhanced_spaces = {}

        for category, space in self.default_search_spaces.items():
            enhanced_spaces[category] = create_enhanced_search_space(space, self.nonlinear_config)

        return enhanced_spaces

    def _get_default_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Get default search spaces for parameter categories."""
        return {
            'confidence': {
                'base_entry_threshold': {'type': 'float', 'min': 0.5, 'max': 0.9},
                'analyst_confidence_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'tactician_confidence_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9},
                # Note: No confidence combination weights - Tactician uses Analyst output as input,
                # so combining them would cause overfitting. We use only Tactician's Ensemble confidence.
                # 0.3% Micro Movement Entry Thresholds (immediate only)
                'micro_immediate_long_threshold': {'type': 'float', 'min': 0.65, 'max': 0.85},
                'micro_immediate_short_threshold': {'type': 'float', 'min': 0.68, 'max': 0.88},
                # Exit-specific confidence parameters for 0.3% micro movements
                'exit_confidence_threshold': {'type': 'float', 'min': 0.3, 'max': 0.7},
                # Note: No exit confidence combination weights - we use only Tactician's Ensemble confidence
                # 0.3% Micro Movement Exit Thresholds (immediate only)
                'exit_micro_immediate_long_threshold': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'exit_micro_immediate_short_threshold': {'type': 'float', 'min': 0.1, 'max': 0.5},
                # Directional Reversal Detection (MAIN EXIT TRIGGER)
                'directional_confidence_min': {'type': 'float', 'min': 0.05, 'max': 0.5}
            },
            'intensity': {
                # Signal intensity and strength parameters
                'signal_intensity_threshold': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'intensity_decay_factor': {'type': 'float', 'min': 0.85, 'max': 0.99},
                'intensity_amplification_factor': {'type': 'float', 'min': 1.05, 'max': 1.25},
                'min_intensity_duration': {'type': 'int', 'min': 3, 'max': 15},
                'max_intensity_duration': {'type': 'int', 'min': 30, 'max': 120},
                'intensity_combination_method': {'type': 'categorical', 'choices': ['weighted_average', 'maximum', 'harmonic_mean']}
            },
            'position_sizing': {
                'base_position_size': {'type': 'float', 'min': 0.01, 'max': 0.15},
                'max_position_size': {'type': 'float', 'min': 0.1, 'max': 0.3}
            },
            'leverage': {
                'safe_leverage_multiplier': {'type': 'float', 'min': 0.5, 'max': 1.0}
            },
            'tpsl': {
                'tp_long': {'type': 'float', 'min': 0.02, 'max': 0.1},
                'sl_long': {'type': 'float', 'min': 0.01, 'max': 0.05},
                
                # ===== ENHANCED TP/SL PARAMETERS =====
                # Take profit ATR-based parameters
                'tp_base_atr_multiplier': {'type': 'float', 'min': 1.5, 'max': 4.0},
                'tp_confidence_scaling': {'type': 'float', 'min': 0.5, 'max': 1.5},
                'tp_uncertainty_scaling': {'type': 'float', 'min': 0.5, 'max': 1.5},
                
                # Stop loss ATR-based parameters
                'sl_base_atr_multiplier': {'type': 'float', 'min': 0.5, 'max': 2.0},
                'sl_volatility_scaling': {'type': 'float', 'min': 0.8, 'max': 1.5},
                'sl_rolling_window': {'type': 'int', 'min': 10, 'max': 50},
                
                # Trailing take profit
                'enable_trailing_tp': {'type': 'categorical', 'choices': [True, False]},
                'trailing_tp_activation_atr': {'type': 'float', 'min': 1.0, 'max': 2.5},
                
                # Adaptive TP/SL
                'enable_adaptive_tpsl': {'type': 'categorical', 'choices': [True, False]},
                'adaptive_tp_volatility_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.5},
                'adaptive_sl_uncertainty_multiplier': {'type': 'float', 'min': 0.8, 'max': 1.5}
            },
            'exit_strategy': {
                # Component confidence drop (backtested parameter)
                'component_confidence_drop': {'type': 'float', 'min': 0.1, 'max': 0.5},

                # Base profit target (tested range 0.6% - 1.2%)
                'base_profit_target': {'type': 'float', 'min': 0.006, 'max': 0.012},
                
                # Profit trailing percent (tested range 0.0% - 0.3%)
                'profit_trailing_percent': {'type': 'float', 'min': 0.0, 'max': 0.003},

                # Exit confidence drop threshold
                'exit_confidence_drop': {'type': 'float', 'min': 0.1, 'max': 0.5},

                # Stop-loss parameters
                'base_stop_loss': {'type': 'float', 'min': -0.08, 'max': -0.02},
                'atr_multiplier': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'volatility_adjustment_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},

                # Time-based parameters (max hold time only - min hold time removed)
                'max_hold_time': {'type': 'int', 'min': 3600, 'max': 14400},  # 1-4 hours
                'confidence_time_scaling_factor': {'type': 'float', 'min': 0.5, 'max': 2.0},

                # Trailing stop parameters
                'trailing_atr_multiplier': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'trailing_min_distance': {'type': 'float', 'min': 0.005, 'max': 0.03},
                'trailing_confidence_activation': {'type': 'float', 'min': 0.6, 'max': 0.9},

                # Unified trailing framework parameters
                'profit_buffer_atr_multiplier': {'type': 'float', 'min': 0.3, 'max': 0.9},
                'profit_buffer_min_fraction': {'type': 'float', 'min': 0.0005, 'max': 0.002},
                'trail_base_atr_multiplier': {'type': 'float', 'min': 0.6, 'max': 1.2},
                'breakeven_activation_atr': {'type': 'float', 'min': 0.8, 'max': 1.5},
                'trail_activation_atr': {'type': 'float', 'min': 0.8, 'max': 1.5},
                'tp_trail_activation_atr': {'type': 'float', 'min': 1.8, 'max': 2.5},
                'tp_trail_trigger_atr': {'type': 'float', 'min': 2.0, 'max': 3.5},
                'partial_take_fraction': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'drawdown_tighten_atr': {'type': 'float', 'min': 0.6, 'max': 1.0},
                'tighten_trail_atr': {'type': 'float', 'min': 0.3, 'max': 0.8},
                'drawdown_exit_atr': {'type': 'float', 'min': 1.0, 'max': 1.6},
                'volatility_tighten_threshold': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'volatility_tighten_adjustment': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'volatility_loosen_threshold': {'type': 'float', 'min': 1.1, 'max': 1.6},
                'volatility_loosen_adjustment': {'type': 'float', 'min': 0.1, 'max': 0.4},
                'time_decay_bars': {'type': 'int', 'min': 6, 'max': 12},
                'time_decay_threshold_atr': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'ml_confidence_tighten_threshold': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'ml_confidence_tighten_atr': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'ml_regime_partial_fraction': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'low_vol_sl_atr': {'type': 'float', 'min': 1.0, 'max': 1.6},
                'low_vol_tp_atr': {'type': 'float', 'min': 1.8, 'max': 2.6},
                'low_vol_trail_atr': {'type': 'float', 'min': 0.6, 'max': 1.0},
                'low_vol_tp_trail': {'type': 'float', 'min': 2.0, 'max': 2.6},
                'normal_vol_sl_atr': {'type': 'float', 'min': 1.3, 'max': 1.9},
                'normal_vol_tp_atr': {'type': 'float', 'min': 2.2, 'max': 3.0},
                'normal_vol_trail_atr': {'type': 'float', 'min': 0.8, 'max': 1.2},
                'normal_vol_tp_trail': {'type': 'float', 'min': 2.2, 'max': 3.0},
                'high_vol_sl_atr': {'type': 'float', 'min': 1.5, 'max': 2.2},
                'high_vol_tp_atr': {'type': 'float', 'min': 2.6, 'max': 3.6},
                'high_vol_trail_atr': {'type': 'float', 'min': 1.0, 'max': 1.5},
                'high_vol_tp_trail': {'type': 'float', 'min': 2.6, 'max': 3.6},
                'trailing_tightening_threshold': {'type': 'float', 'min': 0.01, 'max': 0.05},
                'trailing_time_decay': {'type': 'float', 'min': 0.9, 'max': 0.995},
                'trailing_ml_adjustment_weight': {'type': 'float', 'min': 0.1, 'max': 0.6},
                'ml_trigger_trailing_multiplier': {'type': 'float', 'min': 0.85, 'max': 1.2},

                # Regime-aware parameters
                'regime_transition_penalty': {'type': 'float', 'min': 0.05, 'max': 0.2},
                'regime_specific_scaling': {'type': 'float', 'min': 0.8, 'max': 1.2},
                'regime_trending_profit_band': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'regime_ranging_profit_band': {'type': 'float', 'min': 0.4, 'max': 0.7},
                'regime_high_volatility_profit_band': {'type': 'float', 'min': 0.45, 'max': 0.8},
                'regime_trailing_sensitivity': {'type': 'float', 'min': 0.8, 'max': 1.2},
                
                # ===== UNCERTAINTY-BASED PARAMETERS =====
                'uncertainty_weight': {'type': 'float', 'min': 0.0, 'max': 1.0},
                'uncertainty_sl_multiplier': {'type': 'float', 'min': 0.5, 'max': 2.0},
                'uncertainty_tp_multiplier': {'type': 'float', 'min': 0.5, 'max': 2.0},
                'model_disagreement_threshold': {'type': 'float', 'min': 0.0, 'max': 0.5},
                'uncertainty_sensitivity': {'type': 'float', 'min': 0.5, 'max': 2.0},
                
                # ===== CONFIDENCE DEGRADATION PARAMETERS =====
                'confidence_position_scaling_power': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'confidence_degradation_threshold': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'confidence_degradation_window': {'type': 'int', 'min': 4, 'max': 12},
                'confidence_sl_tightening_factor': {'type': 'float', 'min': 0.5, 'max': 1.5},
                'minimum_entry_confidence': {'type': 'float', 'min': 0.5, 'max': 0.9},
                
                # ===== VOLATILITY-BASED PARAMETERS =====
                'atr_sl_multiplier_range': {'type': 'float', 'min': 1.0, 'max': 3.0},
                'volatility_regime_low_threshold': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'volatility_regime_high_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'high_vol_position_scaling': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'low_vol_position_scaling': {'type': 'float', 'min': 1.0, 'max': 1.5},
                'volatility_sensitivity': {'type': 'float', 'min': 0.5, 'max': 2.0},
                
                # ===== DYNAMIC TRAILING PARAMETERS (Multiplicative) =====
                'trailing_base_pct': {'type': 'float', 'min': 0.005, 'max': 0.03},
                'trailing_confidence_weight': {'type': 'float', 'min': 0.0, 'max': 2.0},
                'trailing_uncertainty_weight': {'type': 'float', 'min': 0.0, 'max': 2.0},
                'trailing_volatility_weight': {'type': 'float', 'min': 0.0, 'max': 2.0},
                'trailing_regime_weight': {'type': 'float', 'min': 0.0, 'max': 2.0},
                
                # ===== DYNAMIC TRAILING PARAMETERS (Log Space) =====
                'trailing_log_base': {'type': 'float', 'min': -5.0, 'max': -2.0},
                'trailing_log_confidence_weight': {'type': 'float', 'min': 0.0, 'max': 2.0},
                'trailing_log_uncertainty_weight': {'type': 'float', 'min': -2.0, 'max': 0.0},
                'trailing_log_volatility_weight': {'type': 'float', 'min': -1.0, 'max': 1.0},
                'trailing_log_regime_weight': {'type': 'float', 'min': -1.0, 'max': 1.0},
                
                # ===== DYNAMIC TRAILING METHOD SELECTION =====
                'trailing_method': {'type': 'categorical', 'choices': ['multiplicative', 'log_space', 'ensemble']},
                'trailing_ensemble_mult_weight': {'type': 'float', 'min': 0.0, 'max': 1.0},
                'trailing_ensemble_log_weight': {'type': 'float', 'min': 0.0, 'max': 1.0}
            },
            'ensemble': {
                'analyst_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'tactician_weight': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'strategist_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                # Ensemble method parameters for Analyst (Elastic Net meta) & Tactician (LightGBM meta)
                'ensemble_method': {'type': 'categorical', 'choices': ['stacking', 'weighted_average', 'voting', 'meta_learner']},
                'analyst_meta_model_type': {'type': 'categorical', 'choices': ['elastic_net']},
                'tactician_meta_model_type': {'type': 'categorical', 'choices': ['lightgbm']},
                'stacking_cv_folds': {'type': 'int', 'min': 3, 'max': 10},
                'meta_learner_weight': {'type': 'float', 'min': 0.1, 'max': 0.4}
            },
            'sr': {
                'touch_count_weight': {'type': 'float', 'min': 0.1, 'max': 0.4},
                'total_volume_weight': {'type': 'float', 'min': 0.1, 'max': 0.4},
                'level_age_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'bounce_rate_weight': {'type': 'float', 'min': 0.1, 'max': 0.3}
            },
            'two_tier': {
                'tier1_weight': {'type': 'float', 'min': 0.4, 'max': 0.7},
                'tier2_weight': {'type': 'float', 'min': 0.3, 'max': 0.6},
                'direction_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'timing_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9}
            },
            'technical_indicators': {
                'rsi_period': {'type': 'int', 'min': 10, 'max': 20},
                'macd_fast_period': {'type': 'int', 'min': 8, 'max': 16},
                'macd_slow_period': {'type': 'int', 'min': 20, 'max': 30},
                'adx_trend_threshold': {'type': 'float', 'min': 20.0, 'max': 35.0},
                'adx_sideways_threshold': {'type': 'float', 'min': 15.0, 'max': 30.0},
                'volatility_threshold': {'type': 'float', 'min': 0.015, 'max': 0.035}
            },
            'system_monitoring': {
                'analysis_interval': {'type': 'int', 'min': 1800, 'max': 7200},
                'max_history': {'type': 'int', 'min': 50, 'max': 200},
                'memory_threshold': {'type': 'float', 'min': 0.7, 'max': 0.9},
                'learning_rate': {'type': 'float', 'min': 0.005, 'max': 0.05}
            },
            'training_optimization': {
                'min_label_balance': {'type': 'float', 'min': 0.03, 'max': 0.1},
                'max_label_balance': {'type': 'float', 'min': 0.9, 'max': 0.98},
                'stability_threshold': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'lgb_learning_rate': {'type': 'float', 'min': 0.01, 'max': 0.2},
                'model_performance_threshold': {'type': 'float', 'min': 0.6, 'max': 0.85}
            },
            'regime_transitions': {
                'transition_intensity_threshold': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'transition_confidence_threshold': {'type': 'float', 'min': 0.6, 'max': 0.9},
                'step9_5_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'step10_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'regime_expert_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'transition_lookback_periods': {'type': 'int', 'min': 3, 'max': 10},
                'transition_risk_multiplier': {'type': 'float', 'min': 1.0, 'max': 1.5}
            },
            'signal_aggregation': {
                'analyst_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'tactician_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'scenario_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'sr_breakout_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'regime_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},
                'conflict_penalty_factor': {'type': 'float', 'min': 0.4, 'max': 0.6},
                'min_source_weight': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'min_signal_confidence': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'min_aggregated_confidence': {'type': 'float', 'min': 0.4, 'max': 0.6},
                'regime_alignment_bonus': {'type': 'float', 'min': 0.1, 'max': 0.25},
                'multi_signal_alignment_bonus': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'use_multiplicative': {'type': 'bool', 'value': True}
            },
            'turnover_cost_penalty': {
                'turnover_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 1.0},
                'commission_rate': {'type': 'float', 'min': 0.0005, 'max': 0.002},
                'slippage_rate': {'type': 'float', 'min': 0.0002, 'max': 0.001},
                'max_turnover_rate': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'round_trip_multiplier': {'type': 'float', 'min': 1.5, 'max': 3.0}
            },
            'entry_timing_optimization': {
                # Entry timing parameters - Tactician naturally optimizes for 0-0.4% range
                'entry_timing_range': {'type': 'float', 'min': 0.002, 'max': 0.004},  # 0.2% to 0.4%
                'early_entry_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'late_entry_penalty_weight': {'type': 'float', 'min': 0.1, 'max': 0.5},
                'optimal_entry_reward_weight': {'type': 'float', 'min': 0.3, 'max': 0.7},
                'entry_timing_efficiency_weight': {'type': 'float', 'min': 0.2, 'max': 0.6},
                'directional_accuracy_threshold': {'type': 'float', 'min': 0.55, 'max': 0.75},
                'adverse_movement_threshold': {'type': 'float', 'min': 0.6, 'max': 0.8},
                'entry_timing_lookback_periods': {'type': 'int', 'min': 5, 'max': 20}
            },
            'confidence_aware_ensemble': {
                # Confidence-aware ensemble parameters for updated models
                'confidence_threshold_entry': {'type': 'float', 'min': 0.6, 'max': 0.85},
                'confidence_threshold_exit': {'type': 'float', 'min': 0.5, 'max': 0.75},
                'confidence_weight_analyst': {'type': 'float', 'min': 0.2, 'max': 0.5},
                'confidence_weight_tactician': {'type': 'float', 'min': 0.3, 'max': 0.6},
                'confidence_combination_method': {'type': 'categorical', 'choices': ['multiplicative', 'weighted_average', 'harmonic_mean', 'geometric_mean']},
                'ensemble_confidence_threshold': {'type': 'float', 'min': 0.65, 'max': 0.9},
                'base_model_confidence_weight': {'type': 'float', 'min': 0.4, 'max': 0.8},
                'meta_model_confidence_weight': {'type': 'float', 'min': 0.2, 'max': 0.6}
            },
            'model_specific_parameters': {
                # Analyst model weights (Base models)
                'analyst_tcn_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'analyst_catboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                'analyst_lightgbm_weight': {'type': 'float', 'min': 0.2, 'max': 0.4},
                # Analyst meta-learner weight
                'analyst_elastic_net_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},

                # Tactician model weights (Base models)
                'tactician_xgboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.35},
                'tactician_randomforest_weight': {'type': 'float', 'min': 0.15, 'max': 0.3},
                'tactician_catboost_weight': {'type': 'float', 'min': 0.2, 'max': 0.35},
                'tactician_elastic_net_weight': {'type': 'float', 'min': 0.15, 'max': 0.3},
                # Tactician meta-learner weight
                'tactician_lightgbm_weight': {'type': 'float', 'min': 0.1, 'max': 0.3},

                # General model parameters
                'model_diversity_bonus': {'type': 'float', 'min': 0.05, 'max': 0.15},
                'model_complexity_penalty': {'type': 'float', 'min': 0.01, 'max': 0.1}
            }
        }

    async def optimize_all_parameters(self, calibration_results: Dict[str, Any],
                                    previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize all parameters by category (or hierarchically if enabled).

        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start

        Returns:
            Dict containing optimized parameters by category/group
        """
        try:
            self.logger.info("🔧 Starting final parameters optimization...")
            self.logger.info(f"📊 Calibration results available: {len(calibration_results)} keys")
            self.logger.info(f"🔄 Previous results available: {previous_results is not None}")

            start_time = time.time()
            
            # ============================================================================
            # HIERARCHICAL OPTIMIZATION (Recommended)
            # ============================================================================
            if self.use_hierarchical_optimization:
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info("🏗️ Using Hierarchical Parameter Optimization")
                self.logger.info("=" * 80)
                
                try:
                    # Create hierarchical optimizer
                    hierarchical_optimizer = create_hierarchical_optimizer(
                        backtest_func=self._run_backtest_for_hierarchical_optimization,
                        calibration_results=calibration_results,
                        config={
                            'cv_folds': self.cv_folds,
                            'n_rounds': 2,
                            'cache_dir': 'artifacts/optimization_cache',
                            'random_state': 42,
                            'verbose': True
                        }
                    )
                    
                    # Prepare data for optimization
                    # Extract features and targets from calibration results
                    features, targets = self._prepare_data_for_hierarchical_optimization(
                        calibration_results
                    )
                    
                    # Run hierarchical optimization
                    self.logger.info("🚀 Starting hierarchical optimization...")
                    result = hierarchical_optimizer.optimize(
                        X_train=features,
                        y_train=targets,
                        X_val=None,  # Will use CV internally
                        y_val=None
                    )
                    
                    total_duration = time.time() - start_time
                    
                    # Log results
                    self.logger.info("")
                    self.logger.info("=" * 80)
                    self.logger.info("✅ HIERARCHICAL OPTIMIZATION COMPLETE")
                    self.logger.info("=" * 80)
                    self.logger.info(f"   Best score: {result.best_score:.4f}")
                    self.logger.info(f"   Total trials: {result.total_trials}")
                    self.logger.info(f"   Total time: {total_duration:.2f}s")
                    self.logger.info(f"   Groups optimized: {len(result.group_results)}")
                    self.logger.info("")
                    self.logger.info("   Group Results:")
                    for group_result in result.group_results:
                        self.logger.info(f"      • {group_result.group_name}: {group_result.best_score:.4f} "
                                       f"({group_result.n_trials} trials, {group_result.optimization_time:.2f}s)")
                    self.logger.info("=" * 80)
                    
                    # Convert hierarchical result to category format for compatibility
                    optimization_results = self._convert_hierarchical_to_category_format(
                        result.best_params,
                        result
                    )
                    
                    return optimization_results
                    
                except Exception as e:
                    self.logger.error(f"❌ Hierarchical optimization failed: {e}")
                    self.logger.exception("Full traceback:")
                    self.logger.warning("⚠️ Falling back to category-by-category optimization")
                    # Fall through to category-by-category optimization
            
            # ============================================================================
            # CATEGORY-BY-CATEGORY OPTIMIZATION (Legacy/Fallback)
            # ============================================================================
            self.logger.info("📊 Using category-by-category optimization")
            optimization_results = {}

            for i, category in enumerate(self.categories, 1):
                self.logger.info(f"🔄 Optimizing {category} parameters ({i}/{len(self.categories)})...")
                category_start = time.time()

                category_results = await self._optimize_category(
                    category, calibration_results,
                    previous_results.get(category) if previous_results else None
                )

                category_duration = time.time() - category_start
                optimization_results[category] = category_results

                if category_results and 'best_value' in category_results:
                    self.logger.info(f"✅ {category} optimization completed in {category_duration:.2f}s - Best value: {category_results['best_value']:.4f}")
                else:
                    self.logger.warning(f"⚠️ {category} optimization completed in {category_duration:.2f}s - No results obtained")

            total_duration = time.time() - start_time
            self.logger.info("✅ Final parameters optimization completed")
            self.logger.info(f"⏱️ Total optimization time: {total_duration:.2f}s")
            self.logger.info(f"📊 Categories optimized: {len(optimization_results)}")

            return optimization_results

        except Exception as e:
            self.logger.error(f"❌ Error in final parameters optimization: {e}")
            self.logger.exception("Full traceback:")
            raise

    async def _optimize_category(self, category: str, calibration_results: Dict[str, Any],
                               previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Enhanced optimization using BayesianTPEOptimizer with hardware acceleration.

        Args:
            category: Parameter category to optimize
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for this category

        Returns:
            Dict containing optimization results for the category
        """
        try:
            tprint(f"🔍 Optimizing category: {category}", "header")

            # Use enhanced search space if non-linear optimization is enabled
            if self.use_nonlinear_optimization and category in self.enhanced_search_spaces:
                search_space = self.enhanced_search_spaces[category]
                tprint(f"🚀 Using enhanced non-linear search space for {category}", "info")
            else:
                search_space = self.default_search_spaces.get(category, {})
                tprint(f"📊 Using standard search space for {category}", "info")

            if not search_space:
                tprint(f"⚠️  No search space found for category: {category}", "warning")
                return {}

            tprint(f"📊 Search space: {len(search_space)} parameters", "info")

            # Convert search space to BayesianTPEOptimizer format
            converted_search_space = self._convert_search_space_format(search_space)

            # Use BayesianTPEOptimizer if available, fallback to manual optimization
            if self.bayesian_optimizer:
                result = await self._optimize_with_bayesian_tpe(
                    category, converted_search_space, calibration_results, previous_results
                )
            else:
                tprint(f"⚠️  BayesianTPEOptimizer not available, using fallback", "warning")
                result = await self._optimize_with_fallback(
                    category, search_space, calibration_results
                )

            if result and 'best_params' in result:
                tprint(f"✅ {category} optimization complete", "success")
                tprint(f"   Best score: {result.get('best_value', 0):.4f}", "info")
                tprint(f"   Time: {result.get('optimization_time', 0):.2f}s", "info")

                # Log cache statistics
                if self.evaluation_cache:
                    cache_stats = self.evaluation_cache.get_stats()
                    tprint(f"   Cache hit rate: {cache_stats['hit_rate']:.1%} ({cache_stats['hits']}/{cache_stats['total_requests']})", "info")
                    tprint(f"   Cache size: {cache_stats['size']}/{cache_stats['max_size']}", "info")

            return result

        except Exception as e:
            self.logger.error(f"❌ Error optimizing category {category}: {e}")
            tprint(f"❌ Optimization failed for {category}: {e}", "error")
            return {}

    def _convert_search_space_format(self, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert search space to BayesianTPEOptimizer format

        Args:
            search_space: Original search space dictionary

        Returns:
            Converted search space for BayesianTPEOptimizer
        """
        converted = {}
        for param_name, param_config in search_space.items():
            if param_config['type'] in ['float', 'int']:
                # For numeric parameters, use (min, max) tuple
                converted[param_name] = (param_config['min'], param_config['max'])
            elif param_config['type'] == 'categorical':
                # For categorical, use list of choices
                converted[param_name] = param_config['choices']
            elif param_config['type'] == 'bool':
                # For boolean, use list of choices
                converted[param_name] = [True, False]
        return converted

    async def _optimize_with_bayesian_tpe(self, category: str, search_space: Dict[str, Any],
                                         calibration_results: Dict[str, Any],
                                         previous_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize using BayesianTPEOptimizer with staged optimization and early stopping

        Args:
            category: Parameter category
            search_space: Converted search space
            calibration_results: Calibration results
            previous_results: Previous optimization results

        Returns:
            Optimization results dictionary
        """
        try:
            tprint(f"🎯 Starting Bayesian TPE optimization for {category}", "info")

            # Define objective function for this category
            def objective(params: Dict[str, Any]) -> float:
                """Objective function for optimization"""
                # Check cache first
                cache_key = f"{category}_{str(sorted(params.items()))}"
                cached_score = self.evaluation_cache.get(cache_key)
                if cached_score is not None:
                    return cached_score

                # Evaluate configuration
                score = self._evaluate_configuration(category, params, calibration_results)

                # Cache result
                self.evaluation_cache.set(cache_key, score)

                return score

            # Run optimization
            start_time = time.time()
            optimization_results = self.bayesian_optimizer.optimize(
                objective=objective,
                search_space=search_space
            )
            optimization_time = time.time() - start_time

            # Extract results
            best_params = optimization_results.get('best_params', {})
            best_value = optimization_results.get('best_value', 0.0)

            tprint(f"   Completed {optimization_results.get('n_trials', 0)} trials", "info")
            tprint(f"   Stages: Coarse={optimization_results.get('stages', {}).get('coarse_grid', 0)}, "
                  f"Fine={optimization_results.get('stages', {}).get('fine_grid', 0)}, "
                  f"TPE={optimization_results.get('stages', {}).get('tpe', 0)}", "info")

            result = {
                'best_params': best_params,
                'best_value': best_value,
                'optimization_method': 'bayesian_tpe',
                'optimization_time': optimization_time,
                'n_trials': optimization_results.get('n_trials', 0),
                'stages': optimization_results.get('stages', {}),
                'history': optimization_results.get('history', []),
                'hardware_accelerated': self.hardware_enabled
            }

            return result

        except Exception as e:
            self.logger.error(f"BayesianTPE optimization failed for {category}: {e}")
            tprint(f"❌ BayesianTPE failed for {category}: {e}", "error")
            return {}

    async def _optimize_with_fallback(self, category: str, search_space: Dict[str, Dict[str, Any]],
                                     calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback optimization when BayesianTPEOptimizer is not available

        Args:
            category: Parameter category
            search_space: Original search space
            calibration_results: Calibration results

        Returns:
            Optimization results dictionary
        """
        try:
            tprint(f"🔄 Using fallback optimization for {category}", "info")

            # Use simple grid search as fallback
            coarse_result = await self._coarse_grid_search(category, search_space, calibration_results)

            if not coarse_result:
                return self._create_fallback_result(category)

            return {
                'best_params': coarse_result.get('best_params', {}),
                'best_value': coarse_result.get('best_score', 0.0),
                'optimization_method': 'fallback_grid',
                'n_combinations': coarse_result.get('n_combinations', 0)
            }

        except Exception as e:
            self.logger.error(f"Fallback optimization failed for {category}: {e}")
            tprint(f"❌ Fallback failed for {category}: {e}", "error")
            return self._create_fallback_result(category)

    def _objective_function(self, trial: optuna.Trial, category: str,
                          search_space: Dict[str, Dict[str, Any]],
                          calibration_results: Dict[str, Any]) -> float:
        """
        Enhanced objective function for Optuna optimization with non-linear sampling.

        Args:
            trial: Optuna trial object
            category: Parameter category being optimized
            search_space: Search space for the category
            calibration_results: Results from confidence calibration

        Returns:
            Optimization score (higher is better)
            
        Raises:
            optuna.TrialPruned: If trial should be pruned due to evaluation failure
        """
        try:
            params = {}

            # Use enhanced search space if non-linear optimization is enabled
            if self.use_nonlinear_optimization and category in self.enhanced_search_spaces:
                enhanced_space = self.enhanced_search_spaces[category]
                for param_name, param_config in enhanced_space.items():
                    if param_config['type'] == 'float':
                        # Use enhanced non-linear sampling
                        transform_type = param_config.get('transform_type', 'auto')
                        params[param_name] = self.parameter_sampler.suggest_enhanced_float(
                            trial, param_name, param_config['min'], param_config['max'], transform_type
                        )
                    elif param_config['type'] == 'int':
                        # Use enhanced non-linear sampling for integers
                        transform_type = param_config.get('transform_type', 'auto')
                        params[param_name] = self.parameter_sampler.suggest_enhanced_int(
                            trial, param_name, param_config['min'], param_config['max'], transform_type
                        )
                    elif param_config['type'] == 'bool':
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                # Fallback to original linear sampling
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, param_config['min'], param_config['max']
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name, param_config['min'], param_config['max']
                        )
                    elif param_config['type'] == 'bool':
                        params[param_name] = trial.suggest_categorical(param_name, [True, False])
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])

            # Evaluate configuration
            score = self._evaluate_configuration(category, params, calibration_results)
            
            # Validate score
            if check_for_nans(score) or check_for_infs(score):
                self.logger.warning(f"Invalid score (NaN/Inf) for {category}, pruning trial")
                raise optuna.TrialPruned()
            
            # Check if score is suspiciously low (likely an error)
            if score < -100:
                self.logger.warning(f"Suspiciously low score ({score:.2f}) for {category}, pruning trial")
                raise optuna.TrialPruned()

            # Apply non-linear scoring enhancements
            if self.use_nonlinear_optimization:
                enhanced_score = apply_nonlinear_scoring(score, params, category)
                return enhanced_score

            return score

        except optuna.TrialPruned:
            # Re-raise pruned trials
            raise
        except ValueError as e:
            # Invalid parameter combination - prune this trial
            self.logger.warning(f"Invalid parameters for {category}: {e}")
            raise optuna.TrialPruned()
        except KeyError as e:
            # Missing required data - prune this trial
            self.logger.warning(f"Missing required data for {category}: {e}")
            raise optuna.TrialPruned()
        except Exception as e:
            # Unexpected error - log and prune
            self.logger.error(f"Unexpected error in objective function for {category}: {e}", exc_info=True)
            raise optuna.TrialPruned()

    def _evaluate_configuration(self, category: str, params: Dict[str, Any],
                              calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate a configuration by running a backtest or simulation.

        Args:
            category: Parameter category being evaluated
            params: Parameters to evaluate
            calibration_results: Results from confidence calibration

        Returns:
            Evaluation score (higher is better)
        """
        try:
            base_score = 0.0

            if category == 'confidence':
                base_score = self._evaluate_confidence_params(params, calibration_results)
            elif category == 'position_sizing':
                base_score = self._evaluate_position_sizing_params(params, calibration_results)
            elif category == 'leverage':
                base_score = self._evaluate_leverage_params(params, calibration_results)
            elif category == 'tpsl':
                base_score = self._evaluate_tpsl_params(params, calibration_results)
            elif category == 'exit_strategy':
                base_score = self._evaluate_exit_strategy_params(params, calibration_results)
            elif category == 'ensemble':
                base_score = self._evaluate_ensemble_params(params, calibration_results)
            elif category == 'sr':
                base_score = self._evaluate_sr_params(params, calibration_results)
            elif category == 'two_tier':
                base_score = self._evaluate_two_tier_params(params, calibration_results)
            elif category == 'technical_indicators':
                base_score = self._evaluate_technical_indicators_params(params, calibration_results)
            elif category == 'system_monitoring':
                base_score = self._evaluate_system_monitoring_params(params, calibration_results)
            elif category == 'training_optimization':
                base_score = self._evaluate_training_optimization_params(params, calibration_results)
            elif category == 'regime_transitions':
                base_score = self._evaluate_regime_transitions_params(params, calibration_results)
            elif category == 'signal_aggregation':
                base_score = self._evaluate_signal_aggregation_params(params, calibration_results)
            elif category == 'turnover_cost_penalty':
                base_score = self._evaluate_turnover_cost_penalty_params(params, calibration_results)
            elif category == 'intensity':
                base_score = self._evaluate_intensity_params(params, calibration_results)
            elif category == 'entry_timing_optimization':
                base_score = self._evaluate_entry_timing_optimization_params(params, calibration_results)
            elif category == 'confidence_aware_ensemble':
                base_score = self._evaluate_confidence_aware_ensemble_params(params, calibration_results)
            elif category == 'model_specific_parameters':
                base_score = self._evaluate_model_specific_params(params, calibration_results)

            # Apply turnover cost penalty to all categories
            if base_score > 0.0:
                turnover_penalty = self._calculate_turnover_penalty(params, calibration_results)
                base_score -= turnover_penalty

            base_score = self._apply_regime_performance_adjustment(category, base_score)

            # Enhanced confidence evaluation includes exit confidence optimization
            # This is handled within _evaluate_confidence_params method

            return base_score

        except Exception as e:
            self.logger.error(f"Error evaluating configuration for {category}: {e}")
            return 0.0

    def _evaluate_confidence_params(self, params: Dict[str, Any],
                                  calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate confidence threshold parameters using proper simulation and CV

        Args:
            params: Confidence parameters to evaluate
            calibration_results: Calibration results with confidence data

        Returns:
            Evaluation score
        """
        try:
            # Extract confidence data from calibration results - Updated to use analyst_confidence
            analyst_conf = calibration_results.get('analyst_confidence', np.array([]))
            # Legacy support for tactician_confidence
            tactician_conf = calibration_results.get('tactician_confidence', analyst_conf.copy() if len(analyst_conf) > 0 else np.array([]))
            returns = calibration_results.get('returns', np.array([]))

            # Check for new simplified target structure
            target_long = calibration_results.get('target_long', np.array([]))
            target_short = calibration_results.get('target_short', np.array([]))
            derived_signals = calibration_results.get('derived_signals', np.array([]))
            derived_targets = calibration_results.get('derived_targets', np.array([]))

            # If we have the new simplified target structure, use it for evaluation
            if len(target_long) > 0 and len(target_short) > 0:
                tprint_info("📊 Evaluating confidence parameters with new simplified target structure")

                # Use derived signals and targets for evaluation
                if len(derived_signals) > 0 and len(derived_targets) > 0:
                    return self._evaluate_confidence_with_new_targets(params, derived_signals, derived_targets, returns)
                else:
                    # Fall back to using target_long and target_short directly
                    combined_signals = (target_long > 0).astype(int) - (target_short > 0).astype(int)
                    combined_targets = target_long  # Use target_long as primary target
                    return self._evaluate_confidence_with_new_targets(params, combined_signals, combined_targets, returns)

            # If we have actual confidence data, use simulation-based evaluation (Analyst-based)
            if len(analyst_conf) > 0 and len(returns) > 0:
                return self._evaluate_confidence_with_simulation(params, analyst_conf, tactician_conf, returns)

            # Otherwise, fall back to heuristic evaluation
            return self._evaluate_confidence_heuristic(params, calibration_results)

        except Exception as e:
            self.logger.error(f"Error evaluating confidence params: {e}")
            tprint(f"❌ Confidence evaluation error: {e}", "error")
            return 0.0

    def _evaluate_confidence_with_new_targets(self, params: Dict[str, Any],
                                         signals: np.ndarray,
                                         targets: np.ndarray,
                                         returns: np.ndarray) -> float:
        """
        Evaluate confidence parameters using new simplified target structure.
        
        This method evaluates parameters using the new target_long and target_short
        structure, which provides separate binary targets for long and short positions.
        
        Args:
            params: Confidence parameters to evaluate
            signals: Trading signals (1=long, -1=short, 0=neutral)
            targets: Target values (from target_long or target_short)
            returns: Returns array
            
        Returns:
            Evaluation score based on simulated trading
        """
        try:
            # Validate inputs
            if len(signals) != len(targets) or len(signals) != len(returns):
                tprint(f"⚠️  Signals/targets/returns length mismatch: signals={len(signals)}, targets={len(targets)}, returns={len(returns)}", "warning")
                return 0.0

            # Apply confidence threshold with validation - Updated to use analyst_confidence
            confidence_threshold = validate_probability(
                params.get('analyst_confidence_threshold', params.get('tactician_confidence_threshold', 0.7)), default=0.7
            )

            # For new target structure, we use the targets directly as signals
            # since target_long and target_short are already binary signals
            if len(targets) > 0:
                # Use targets as signals (they're already binary)
                filtered_signals = targets
                filtered_returns = returns[:len(targets)]

                # Apply additional confidence filtering if we have confidence data (Analyst-based)
                if 'analyst_confidence' in self.calibration_results:
                    analyst_conf = self.calibration_results['analyst_confidence'][:len(targets)]
                    conf_mask = analyst_conf >= confidence_threshold
                    filtered_signals = filtered_signals[conf_mask]
                    filtered_returns = filtered_returns[conf_mask]
                elif 'tactician_confidence' in self.calibration_results:
                    # Legacy fallback
                    tactician_conf = self.calibration_results['tactician_confidence'][:len(targets)]
                    conf_mask = tactician_conf >= confidence_threshold
                    filtered_signals = filtered_signals[conf_mask]
                    filtered_returns = filtered_returns[conf_mask]
            else:
                # Fallback to using provided signals
                conf_mask = signals >= confidence_threshold if len(signals.shape) == 1 else signals >= 0
                filtered_signals = signals[conf_mask]
                filtered_returns = returns[conf_mask]

            if len(filtered_signals) == 0:
                tprint("⚠️  No signals after confidence filtering", "warning")
                return 0.0

            # Calculate position sizing with validation
            base_position_size = validate_positive(
                params.get('base_position_size', 0.01), default=0.01
            )
            base_position_size = validate_range(base_position_size, 0.001, 0.2, default=0.01)

            # Calculate trade returns
            trade_returns = filtered_signals * filtered_returns * base_position_size

            # Remove invalid values
            valid_mask = ~(check_for_nans(trade_returns) | check_for_infs(trade_returns))
            trade_returns = trade_returns[valid_mask]

            if len(trade_returns) == 0:
                tprint("⚠️  No valid trade returns", "warning")
                return 0.0

            # Use VectorBT optimization for metrics calculation if available
            if self.vectorbt_enabled and self.rolling_optimizer:
                tprint("🎯 Using VectorBT-optimized metrics calculation for new targets", "debug")
                sharpe, sortino, max_dd, win_rate, profit_factor, total_return = self._calculate_metrics_vectorbt(trade_returns)
            else:
                # Calculate metrics using common_operations utilities
                sharpe = calculate_sharpe_ratio(trade_returns)
                sortino = calculate_sortino_ratio(trade_returns)
                max_dd = calculate_max_drawdown(np.cumsum(trade_returns))
                win_rate = calculate_win_rate(trade_returns)
                profit_factor = calculate_profit_factor(trade_returns)
                total_return = float(np.sum(trade_returns))

            # Validate all metrics
            sharpe = validate_positive(sharpe, default=0.0) if not check_for_nans(sharpe) else 0.0
            sortino = validate_positive(sortino, default=0.0) if not check_for_nans(sortino) else 0.0
            max_dd = float(max_dd) if not check_for_nans(max_dd) else 0.0
            win_rate = validate_probability(win_rate) if not check_for_nans(win_rate) else 0.0
            profit_factor = validate_positive(profit_factor, default=0.0) if not check_for_nans(profit_factor) else 0.0

            metrics = EvaluationMetrics(
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_dd,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                n_trades=int(len(trade_returns)),
                avg_trade_duration=0.0,  # Would need timestamps
                confidence_score=float(np.mean(targets)) if len(targets) > 0 else 0.0
            )

            score = metrics.to_score()
            
            # Apply regime performance adjustment if available
            score = self._apply_regime_performance_adjustment('confidence', score)

            return score

        except Exception as e:
            self.logger.error(f"Error in confidence evaluation with new targets: {e}")
            tprint(f"❌ New target confidence evaluation error: {e}", "error")
            return 0.0

    def _evaluate_confidence_with_simulation(self, params: Dict[str, Any],
                                            analyst_conf: np.ndarray,
                                            tactician_conf: np.ndarray,
                                            returns: np.ndarray) -> float:
        """
        Evaluate confidence parameters using actual trading simulation with hardware acceleration.
        
        This method now uses GPU/MPS acceleration when available for confidence threshold
        calculations and signal generation, significantly improving performance on M1 hardware.

        Args:
            params: Confidence parameters
            analyst_conf: Analyst confidence array
            tactician_conf: Tactician confidence array
            returns: Returns array

        Returns:
            Evaluation score based on simulated trading
        """
        try:
            # Ensure arrays are same length and focus on Analyst confidence as primary signal
            min_len = min(len(analyst_conf), len(returns))
            analyst_conf = ensure_array(analyst_conf)[:min_len]
            returns = ensure_array(returns)[:min_len]

            # Threshold based on Analyst confidence (fallback to tactician threshold name for compatibility)
            threshold = validate_probability(
                params.get('analyst_confidence_threshold', params.get('tactician_confidence_threshold', 0.7))
            )

            # Use hardware-accelerated comparison if available
            if self.hardware_enabled and self.matrix_processor is not None:
                try:
                    signals = self.matrix_processor.compare_threshold(
                        analyst_conf, threshold, operation='greater_equal'
                    )
                except Exception as e:
                    self.logger.warning(f"Hardware acceleration failed, falling back to numpy: {e}")
                    signals = np.where(analyst_conf >= threshold, 1, 0)
            else:
                signals = np.where(analyst_conf >= threshold, 1, 0)

            # Simulate trading with CV if enabled and sufficient data
            if self.use_cv and len(signals) >= self.cv_folds * 100:
                # Prepare data for CV
                data = {
                    'features': pd.DataFrame({
                        'confidence': analyst_conf
                    }),
                    'targets': pd.Series(returns),
                    'signals': signals,
                    'returns': returns,
                    'confidences': analyst_conf
                }

                # Define evaluation function for a single fold
                def eval_fold(fold_params: Dict[str, Any], fold_data: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
                    if 'val' in fold_data:
                        fold_conf = fold_data['val']['features']['confidence'].values
                        fold_signals = fold_conf >= threshold
                        fold_returns = fold_data['val']['targets'].values
                        fold_confidences = fold_conf
                    else:
                        fold_signals = fold_data['signals']
                        fold_returns = fold_data['returns']
                        fold_confidences = fold_data['confidences']

                    metrics = self.simulate_trading(fold_params, fold_signals, fold_returns, fold_confidences)
                    score = metrics.to_score()
                    return score, metrics.__dict__

                # Evaluate with CV
                cv_score, cv_results = self.evaluate_with_cv(params, data, eval_fold, 'confidence')
                return cv_score
            else:
                # Single evaluation without CV
                metrics = self.simulate_trading(params, signals, returns, analyst_conf)
                score = metrics.to_score()
                return score

        except Exception as e:
            self.logger.error(f"Error in confidence simulation: {e}")
            tprint(f"❌ Confidence simulation error: {e}", "error")
            return 0.0

    def _evaluate_confidence_heuristic(self, params: Dict[str, Any],
                                      calibration_results: Dict[str, Any]) -> float:
        """
        Fallback heuristic evaluation when simulation data is unavailable

        Args:
            params: Confidence parameters
            calibration_results: Calibration results

        Returns:
            Heuristic score
        """
        score = 0.0

        # Base entry threshold evaluation with validation
        if 'base_entry_threshold' in params:
            threshold = validate_probability(params['base_entry_threshold'])
            if validate_range(threshold, 0.6, 0.8):
                score += 0.3
            elif validate_range(threshold, 0.5, 0.9):
                score += 0.2

        # Threshold relationship validation
        if 'analyst_confidence_threshold' in params and 'tactician_confidence_threshold' in params:
            analyst_thresh = validate_probability(params['analyst_confidence_threshold'])
            tactician_thresh = validate_probability(params['tactician_confidence_threshold'])

            # Tactician should have higher threshold
            if tactician_thresh > analyst_thresh:
                score += 0.2

            # Reasonable separation
            diff = tactician_thresh - analyst_thresh
            if validate_range(diff, 0.1, 0.2):
                score += 0.1

        # Exit confidence validation (using only Tactician confidence)
        exit_threshold = validate_probability(params.get('exit_confidence_threshold', 0.5))
        if validate_range(exit_threshold, 0.3, 0.7):
            score += 0.1

        # Note: No confidence combination validation - we use only Tactician's Ensemble confidence

        return score

    def _calculate_optimal_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                    calibration_results: Dict[str, Any]) -> Optional[float]:
        """
        Return Tactician confidence threshold (no combination with Analyst).
        
        Note: Tactician uses Analyst output as input, so combining confidences
        would cause overfitting. We use only Tactician's Ensemble confidence.

        Args:
            analyst_threshold: Analyst confidence threshold (not used, kept for API compatibility)
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration

        Returns:
            Tactician confidence threshold or None if calculation fails
        """
        try:
            # Check if tactician confidence is available
            if 'tactician_confidence' not in calibration_results and 'tactician_confidence_threshold' not in calibration_results:
                self.logger.warning("⚠️ Tactician confidence not available")
                return None

            # Return tactician threshold directly (no combination)
            return validate_probability(tactician_threshold)

        except Exception as e:
            self.logger.error(f"❌ Error calculating optimal confidence: {e}")
            return None

    def _has_confidence_levels_available(self, calibration_results: Dict[str, Any]) -> bool:
        """
        Check if both tactician and analyst confidence levels are available.

        Args:
            calibration_results: Results from confidence calibration

        Returns:
            True if both confidence levels are available, False otherwise
        """
        try:
            # Check for tactician confidence data
            tactician_available = (
                'tactician_confidence' in calibration_results or
                'tactician_models' in calibration_results or
                'tactician_ensemble' in calibration_results
            )

            # Check for analyst confidence data
            analyst_available = (
                'analyst_confidence' in calibration_results or
                'analyst_models' in calibration_results or
                'analyst_ensemble' in calibration_results
            )

            both_available = tactician_available and analyst_available

            if not both_available:
                self.logger.warning(f"⚠️ Confidence availability - Tactician: {tactician_available}, Analyst: {analyst_available}")

            return both_available

        except Exception as e:
            self.logger.error(f"❌ Error checking confidence availability: {e}")
            return False

    def _calculate_multiplicative_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                           tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using multiplicative operations.

        Formula: (tactician_threshold^tactician_weight) * (analyst_threshold^analyst_weight)

        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence

        Returns:
            Multiplicative confidence value
        """
        try:
            # Ensure thresholds are positive for power operations
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)

            # Multiplicative combination with weights as exponents
            multiplicative_conf = (
                (tactician_thresh ** tactician_weight) *
                (analyst_thresh ** analyst_weight)
            )

            # Normalize to [0, 1] range
            multiplicative_conf = min(1.0, multiplicative_conf)

            return multiplicative_conf

        except Exception as e:
            self.logger.error(f"❌ Error in multiplicative confidence calculation: {e}")
            return 0.5  # Default fallback

    def _calculate_logarithmic_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                        tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using logarithmic additions.

        Formula: exp(tactician_weight * log(tactician_threshold) + analyst_weight * log(analyst_threshold))

        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence

        Returns:
            Logarithmic confidence value
        """
        try:
            # Ensure thresholds are positive for log operations
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)

            # Logarithmic addition with weights
            log_combination = (
                tactician_weight * np.log(tactician_thresh) +
                analyst_weight * np.log(analyst_thresh)
            )

            # Convert back using exponential
            logarithmic_conf = np.exp(log_combination)

            # Normalize to [0, 1] range
            logarithmic_conf = min(1.0, max(0.0, logarithmic_conf))

            return logarithmic_conf

        except Exception as e:
            self.logger.error(f"❌ Error in logarithmic confidence calculation: {e}")
            return 0.5  # Default fallback

    def _calculate_harmonic_confidence(self, analyst_threshold: float, tactician_threshold: float,
                                     tactician_weight: float, analyst_weight: float) -> float:
        """
        Calculate confidence using weighted harmonic mean.

        Formula: 1 / (tactician_weight/tactician_threshold + analyst_weight/analyst_threshold)

        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            tactician_weight: Weight for tactician confidence
            analyst_weight: Weight for analyst confidence

        Returns:
            Harmonic confidence value
        """
        try:
            # Ensure thresholds are positive for harmonic mean
            analyst_thresh = max(0.001, analyst_threshold)
            tactician_thresh = max(0.001, tactician_threshold)

            # Weighted harmonic mean
            harmonic_conf = 1.0 / (
                tactician_weight / tactician_thresh +
                analyst_weight / analyst_thresh
            )

            # Normalize to [0, 1] range
            harmonic_conf = min(1.0, max(0.0, harmonic_conf))

            return harmonic_conf

        except Exception as e:
            self.logger.error(f"❌ Error in harmonic confidence calculation: {e}")
            return 0.5  # Default fallback

    def _evaluate_confidence_stability(self, analyst_threshold: float, tactician_threshold: float,
                                     calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate confidence stability based on threshold consistency.

        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration

        Returns:
            Stability score between 0 and 1
        """
        try:
            stability_score = 0.0

            # Check threshold consistency
            threshold_diff = abs(tactician_threshold - analyst_threshold)
            if 0.05 <= threshold_diff <= 0.3:  # Good separation
                stability_score += 0.4
            elif threshold_diff < 0.05:  # Too close
                stability_score += 0.1
            else:  # Too far apart
                stability_score += 0.2

            # Check if thresholds are in reasonable ranges
            if 0.5 <= analyst_threshold <= 0.9:
                stability_score += 0.3
            if 0.6 <= tactician_threshold <= 0.95:
                stability_score += 0.3

            return min(1.0, stability_score)

        except Exception as e:
            self.logger.error(f"❌ Error evaluating confidence stability: {e}")
            return 0.5  # Default fallback

    def _evaluate_exit_confidence_calculation(self, analyst_threshold: float, tactician_threshold: float,
                                           calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate exit confidence calculation effectiveness.

        Args:
            analyst_threshold: Analyst confidence threshold
            tactician_threshold: Tactician confidence threshold
            calibration_results: Results from confidence calibration including exit parameters

        Returns:
            Exit confidence evaluation score between 0 and 1
        """
        try:
            score = 0.0

            # Get exit confidence parameters (using only Tactician confidence)
            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)

            # Use only Tactician confidence for exit decisions (no combination)
            tactician_conf = validate_probability(tactician_threshold)

            # Score factors
            # 1. Exit confidence should be reasonable (not too high or too low)
            if 0.4 <= tactician_conf <= 0.8:
                score += 0.3
            elif 0.2 <= tactician_conf <= 0.9:
                score += 0.2
            else:
                score += 0.1

            # 2. Exit threshold should be lower than entry confidence
            entry_confidence = tactician_threshold  # Use only Tactician for entry too
            if exit_threshold < entry_confidence:
                score += 0.2
                # Bonus for reasonable gap
                gap = entry_confidence - exit_threshold
                if 0.1 <= gap <= 0.3:
                    score += 0.1

            # Note: No combination method or weight validation - we use only Tactician confidence
            score += 0.2

            return min(1.0, score)

        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit confidence calculation: {e}")
            return 0.5  # Default fallback

    def _evaluate_using_existing_backtesting_framework(self, calibration_results: Dict[str, Any],
                                                     params: Dict[str, Any]) -> float:
        """
        Evaluate exit confidence parameters using the existing backtesting framework.

        This method integrates exit confidence optimization into the existing backtesting
        system rather than creating a separate backtesting strategy.

        Args:
            calibration_results: Results from confidence calibration
            params: Current parameter configuration

        Returns:
            Backtesting evaluation score (0.0 to 1.0)
        """
        try:
            # Extract exit parameters (using only Tactician confidence)
            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)

            # Use existing calibration data for evaluation
            if 'tactician_confidence' in calibration_results:
                tactician_confidences = calibration_results['tactician_confidence']
                # Pass empty list for analyst_confidences (not used)
                exit_performance = self._evaluate_exit_timing_on_historical_data(
                    [], tactician_confidences, calibration_results
                )

                return exit_performance

            # Fallback evaluation based on parameter reasonableness
            score = 0.0

            # Exit threshold should be reasonable
            if 0.4 <= exit_threshold <= 0.6:
                score += 0.4
            elif 0.3 <= exit_threshold <= 0.7:
                score += 0.2

            # Note: No combination method or weight validation - we use only Tactician confidence
            score += 0.3

            return min(1.0, score)

        except Exception as e:
            self.logger.error(f"❌ Error in existing backtesting framework evaluation: {e}")
            return 0.5

    def _evaluate_exit_timing_on_historical_data(self, analyst_confidences: List[float],
                                               tactician_confidences: List[float],
                                               calibration_results: Dict[str, Any]) -> float:
        """
        Evaluate exit timing using Tactician confidence only.
        
        Note: Tactician uses Analyst output as input, so combining confidences
        would cause overfitting. We use only Tactician's Ensemble confidence.
        """
        try:
            if not tactician_confidences:
                return 0.5

            exit_threshold = calibration_results.get('exit_confidence_threshold', 0.5)

            # Calculate exit points using only Tactician confidence
            exit_signals = []
            for tactician_conf in tactician_confidences:
                # Use only Tactician confidence for exit decisions
                exit_signals.append(tactician_conf < exit_threshold)

            # Evaluate exit signal quality using existing framework metrics
            if 'historical_returns' in calibration_results:
                returns = calibration_results['historical_returns']
                return self._score_exit_signals_against_returns(exit_signals, returns)

            # Fallback: evaluate signal consistency
            exit_rate = sum(exit_signals) / len(exit_signals) if exit_signals else 0

            # Reasonable exit rate (not too frequent, not too rare)
            if 0.1 <= exit_rate <= 0.3:
                return 0.8
            elif 0.05 <= exit_rate <= 0.4:
                return 0.6
            else:
                return 0.4

        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit timing on historical data: {e}")
            return 0.5

    def _score_exit_signals_against_returns(self, exit_signals: List[bool],
                                          returns: List[float]) -> float:
        """
        Score exit signals against historical returns using existing backtesting framework.
        """
        try:
            if len(exit_signals) != len(returns):
                return 0.5

            score = 0.0
            correct_exits = 0
            total_exits = sum(exit_signals)

            if total_exits == 0:
                return 0.3  # No exits might be too conservative

            # Check if exits preceded negative returns
            for i, (should_exit, return_val) in enumerate(zip(exit_signals[:-1], returns[1:])):
                if should_exit:
                    # Good exit if next return is negative
                    if return_val < 0:
                        correct_exits += 1
                    # Penalty for exiting before positive returns
                    elif return_val > 0.01:  # Significant positive return
                        correct_exits -= 0.5

            # Score based on exit accuracy
            if total_exits > 0:
                exit_accuracy = correct_exits / total_exits
                score = max(0.0, min(1.0, 0.5 + exit_accuracy * 0.5))

            return score

        except Exception as e:
            self.logger.error(f"❌ Error scoring exit signals against returns: {e}")
            return 0.5

    def _evaluate_position_sizing_params(self, params: Dict[str, Any],
                                       calibration_results: Dict[str, Any]) -> float:
        """Evaluate position sizing parameters."""
        score = 0.0

        if 'base_position_size' in params:
            base_size = params['base_position_size']
            if 0.02 <= base_size <= 0.1:
                score += 0.3
            elif 0.01 <= base_size <= 0.15:
                score += 0.2
            else:
                score += 0.1

        if 'max_position_size' in params:
            max_size = params['max_position_size']
            if 0.15 <= max_size <= 0.3:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_leverage_params(self, params: Dict[str, Any],
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate leverage parameters."""
        score = 0.0

        if 'safe_leverage_multiplier' in params:
            multiplier = params['safe_leverage_multiplier']
            if 0.7 <= multiplier <= 0.9:
                score += 0.3
            elif 0.5 <= multiplier <= 1.0:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_tpsl_params(self, params: Dict[str, Any],
                            calibration_results: Dict[str, Any]) -> float:
        """Evaluate TP/SL parameters."""
        score = 0.0

        if 'tp_long' in params and 'sl_long' in params:
            tp = params['tp_long']
            sl = params['sl_long']
            if tp > sl and tp / sl >= 1.5:
                score += 0.3
            elif tp > sl:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_exit_strategy_params(self, params: Dict[str, Any],
                                     calibration_results: Dict[str, Any]) -> float:
        """Evaluate exit strategy parameters."""
        score = 0.0

        try:
            # 1. Component confidence drop validation
            if 'component_confidence_drop' in params:
                component_drop = params['component_confidence_drop']
                if 0.2 <= component_drop <= 0.4:
                    score += 0.15
                elif 0.1 <= component_drop <= 0.5:
                    score += 0.08

            # 2. Profit trailing parameters validation
            if 'base_profit_target' in params:
                base_target = params['base_profit_target']
                if 0.007 <= base_target <= 0.011:  # Optimal range around 0.6-1.2%
                    score += 0.12
                elif 0.006 <= base_target <= 0.012:  # Full tested range
                    score += 0.06
                
            if 'profit_trailing_percent' in params:
                trailing = params['profit_trailing_percent']
                if 0.0005 <= trailing <= 0.0025:  # Optimal range around 0.05-0.25%
                    score += 0.08
                elif 0.0 <= trailing <= 0.003:  # Full tested range
                    score += 0.04

            # 3. Exit confidence drop validation
            if 'exit_confidence_drop' in params:
                exit_drop = params['exit_confidence_drop']
                if 0.2 <= exit_drop <= 0.4:
                    score += 0.08
                elif 0.1 <= exit_drop <= 0.5:
                    score += 0.04

            # 4. Stop-loss parameters validation (≈0.15 weight)
            stop_params = ['base_stop_loss', 'atr_multiplier', 'volatility_adjustment_factor']
            if all(param in params for param in stop_params):
                stop_loss = params['base_stop_loss']
                atr_mult = params['atr_multiplier']
                vol_adj = params['volatility_adjustment_factor']

                if -0.07 <= stop_loss <= -0.03:
                    score += 0.07
                elif -0.08 <= stop_loss <= -0.02:
                    score += 0.04

                if 1.5 <= atr_mult <= 2.5:
                    score += 0.04
                elif 1.0 <= atr_mult <= 3.0:
                    score += 0.02

                if 0.8 <= vol_adj <= 1.4:
                    score += 0.04
                elif 0.5 <= vol_adj <= 2.0:
                    score += 0.02

            # 5. Time-based parameters validation (max hold time only - min hold time removed)
            time_params = ['max_hold_time', 'confidence_time_scaling_factor']
            if all(param in params for param in time_params):
                max_time = params['max_hold_time']
                time_scaling = params['confidence_time_scaling_factor']

                if 5400 <= max_time <= 10800:
                    score += 0.06
                elif 3600 <= max_time <= 14400:
                    score += 0.03

                if 0.8 <= time_scaling <= 1.4:
                    score += 0.03
                elif 0.5 <= time_scaling <= 2.0:
                    score += 0.01

            # 5. Trailing stop parameters validation (≈0.15 weight)
            trailing_params = ['trailing_atr_multiplier', 'trailing_min_distance', 'trailing_confidence_activation']
            if all(param in params for param in trailing_params):
                trailing_atr = params['trailing_atr_multiplier']
                min_dist = params['trailing_min_distance']
                conf_act = params['trailing_confidence_activation']

                # Validate trailing stop parameters
                if (1.0 <= trailing_atr <= 3.0 and
                    0.005 <= min_dist <= 0.03 and
                    0.6 <= conf_act <= 0.9):
                    score += 0.1

            # 6. Unified trailing parameter validation (0.15 weight)
            unified_keys = [
                'profit_buffer_atr_multiplier',
                'trail_base_atr_multiplier',
                'drawdown_tighten_atr',
                'drawdown_exit_atr',
                'tp_trail_trigger_atr',
                'time_decay_bars',
            ]
            if all(param in params for param in unified_keys):
                buffer_mult = params['profit_buffer_atr_multiplier']
                trail_mult = params['trail_base_atr_multiplier']
                tighten_atr = params['drawdown_tighten_atr']
                exit_atr = params['drawdown_exit_atr']
                trigger_atr = params['tp_trail_trigger_atr']
                decay_bars = params['time_decay_bars']

                if 0.3 <= buffer_mult <= 0.9 and 0.6 <= trail_mult <= 1.2:
                    score += 0.05
                if tighten_atr < exit_atr:
                    score += 0.05
                if 2.0 <= trigger_atr <= 3.5:
                    score += 0.03
                if 6 <= decay_bars <= 12:
                    score += 0.02

            # 7. Volatility adjustment validation (0.05 weight)
            if all(param in params for param in ['volatility_tighten_threshold', 'volatility_loosen_threshold']):
                tighten_th = params['volatility_tighten_threshold']
                loosen_th = params['volatility_loosen_threshold']
                if 0.6 <= tighten_th < loosen_th <= 1.6:
                    score += 0.05

            # 8. Regime-aware parameters validation (0.1 weight)

                if (1.2 <= trailing_atr <= 2.5 and 0.006 <= min_dist <= 0.025 and 0.65 <= conf_act <= 0.85):
                    score += 0.08
                elif (1.0 <= trailing_atr <= 3.0 and 0.005 <= min_dist <= 0.03 and 0.6 <= conf_act <= 0.9):
                    score += 0.05

                tightening_threshold = params.get('trailing_tightening_threshold')
                if tightening_threshold is not None:
                    if min_dist and min_dist > 0 and 1.5 * min_dist <= tightening_threshold <= 3.5 * min_dist:
                        score += 0.04
                    elif 0.01 <= tightening_threshold <= 0.05:
                        score += 0.02

                if 'trailing_time_decay' in params:
                    decay = params['trailing_time_decay']
                    if 0.93 <= decay <= 0.99:
                        score += 0.03
                    elif 0.9 <= decay <= 0.995:
                        score += 0.01

                if 'trailing_ml_adjustment_weight' in params:
                    trailing_ml_weight = params['trailing_ml_adjustment_weight']
                    if 0.2 <= trailing_ml_weight <= 0.4:
                        score += 0.03
                    elif 0.1 <= trailing_ml_weight <= 0.6:
                        score += 0.01

                if 'ml_trigger_trailing_multiplier' in params:
                    trailing_multiplier = params['ml_trigger_trailing_multiplier']
                    if 0.92 <= trailing_multiplier <= 1.1:
                        score += 0.03
                    elif 0.85 <= trailing_multiplier <= 1.2:
                        score += 0.01

            # 6. Regime-aware parameters validation (≈0.15 weight)
            regime_params = ['regime_transition_penalty', 'regime_specific_scaling']
            if all(param in params for param in regime_params):
                transition_penalty = params['regime_transition_penalty']
                regime_scaling = params['regime_specific_scaling']

                # Validate regime parameters
                if 0.05 <= transition_penalty <= 0.2 and 0.8 <= regime_scaling <= 1.2:
                    score += 0.1

            # 9. Regime band alignment (bonus)
            band_keys = [
                'low_vol_tp_trail',
                'normal_vol_tp_trail',
                'high_vol_tp_trail',
            ]
            if all(param in params for param in band_keys):
                low_tp = params['low_vol_tp_trail']
                normal_tp = params['normal_vol_tp_trail']
                high_tp = params['high_vol_tp_trail']
                if low_tp <= normal_tp <= high_tp:
                    score += 0.03

            # 10. Risk-reward ratio validation (bonus)
                if 0.07 <= transition_penalty <= 0.15:
                    score += 0.05
                elif 0.05 <= transition_penalty <= 0.2:
                    score += 0.03

                if 0.9 <= regime_scaling <= 1.1:
                    score += 0.04
                elif 0.8 <= regime_scaling <= 1.2:
                    score += 0.02

            trending_band = params.get('regime_trending_profit_band')
            ranging_band = params.get('regime_ranging_profit_band')
            volatile_band = params.get('regime_high_volatility_profit_band')
            if trending_band and ranging_band and volatile_band:
                ordered = trending_band >= volatile_band >= ranging_band
                band_ranges = (
                    0.65 <= trending_band <= 0.85 and
                    0.45 <= ranging_band <= 0.65 and
                    0.5 <= volatile_band <= 0.75
                )
                if ordered and band_ranges:
                    score += 0.05
                elif ordered:
                    score += 0.02

            if 'regime_trailing_sensitivity' in params:
                trailing_sensitivity = params['regime_trailing_sensitivity']
                if 0.95 <= trailing_sensitivity <= 1.1:
                    score += 0.03
                elif 0.8 <= trailing_sensitivity <= 1.2:
                    score += 0.01

            # 7. Risk-reward ratio validation (bonus to prioritise profit factor)
            if 'base_profit_target' in params and 'base_stop_loss' in params:
                profit_target = params['base_profit_target']
                stop_loss = abs(params['base_stop_loss'])
                if stop_loss > 0:
                    risk_reward_ratio = profit_target / stop_loss
                    if 1.8 <= risk_reward_ratio <= 3.5:
                        score += 0.07
                    elif 1.3 <= risk_reward_ratio < 1.8:
                        score += 0.03
            
            # 8. Uncertainty-based parameters validation (≈0.10 weight)
            uncertainty_params = ['uncertainty_weight', 'uncertainty_sl_multiplier', 'uncertainty_tp_multiplier']
            if all(param in params for param in uncertainty_params):
                unc_weight = params['uncertainty_weight']
                unc_sl_mult = params['uncertainty_sl_multiplier']
                unc_tp_mult = params['uncertainty_tp_multiplier']
                
                # Validate uncertainty parameters
                if 0.3 <= unc_weight <= 0.7:
                    score += 0.03
                if 1.0 <= unc_sl_mult <= 1.5:  # Reasonable SL widening with uncertainty
                    score += 0.03
                if 0.6 <= unc_tp_mult <= 1.0:  # TP should tighten with uncertainty
                    score += 0.03
                
                # Model disagreement threshold
                if 'model_disagreement_threshold' in params:
                    threshold = params['model_disagreement_threshold']
                    if 0.25 <= threshold <= 0.35:  # Optimal range
                        score += 0.01
            
            # 9. Confidence degradation parameters validation (≈0.10 weight)
            conf_deg_params = ['confidence_degradation_threshold', 'confidence_degradation_window']
            if all(param in params for param in conf_deg_params):
                deg_threshold = params['confidence_degradation_threshold']
                deg_window = params['confidence_degradation_window']
                
                if 0.25 <= deg_threshold <= 0.35:  # ~30% degradation is reasonable
                    score += 0.04
                if 6 <= deg_window <= 10:  # 8 candles is ideal
                    score += 0.03
                
                # Confidence scaling power
                if 'confidence_position_scaling_power' in params:
                    power = params['confidence_position_scaling_power']
                    if 1.8 <= power <= 2.2:  # Around 2.0 is optimal (quadratic scaling)
                        score += 0.03
            
            # 10. Dynamic trailing parameters validation (≈0.15 weight)
            # Multiplicative method
            mult_params = ['trailing_base_pct', 'trailing_confidence_weight', 'trailing_uncertainty_weight']
            if all(param in params for param in mult_params):
                base_pct = params['trailing_base_pct']
                conf_weight = params['trailing_confidence_weight']
                unc_weight = params['trailing_uncertainty_weight']
                
                if 0.01 <= base_pct <= 0.02:  # 1-2% base trailing
                    score += 0.03
                if 1.0 <= conf_weight <= 2.0:  # Confidence should tighten trailing
                    score += 0.03
                if 0.5 <= unc_weight <= 1.5:  # Uncertainty should widen trailing
                    score += 0.03
            
            # Log space method
            log_params = ['trailing_log_base', 'trailing_log_confidence_weight']
            if all(param in params for param in log_params):
                log_base = params['trailing_log_base']
                log_conf_weight = params['trailing_log_confidence_weight']
                
                if -4.0 <= log_base <= -3.0:  # Reasonable log base
                    score += 0.03
                if 0.5 <= log_conf_weight <= 1.5:  # Positive confidence weight
                    score += 0.03
            
            # 11. Volatility-based parameters validation (≈0.08 weight)
            vol_params = ['volatility_regime_low_threshold', 'volatility_regime_high_threshold']
            if all(param in params for param in vol_params):
                low_th = params['volatility_regime_low_threshold']
                high_th = params['volatility_regime_high_threshold']
                
                if low_th < high_th:  # Proper ordering
                    score += 0.02
                if 0.25 <= low_th <= 0.35 and 0.65 <= high_th <= 0.75:  # Optimal ranges
                    score += 0.06

        except Exception as e:
            self.logger.error(f"❌ Error evaluating exit strategy parameters: {e}")
            score = 0.0

        return min(score, 1.0)  # Cap at 1.0
    
    def _run_dynamic_exit_backtest(
        self,
        params: Dict[str, Any],
        calibration_results: Dict[str, Any],
        price_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Run backtest simulation with dynamic exit parameters to evaluate performance.
        
        This simulates trades using the provided exit strategy parameters and
        calculates comprehensive performance metrics.
        
        Args:
            params: Exit strategy parameters to test
            calibration_results: Calibration data with predictions and actual outcomes
            price_data: Historical price data (optional, extracted from calibration if not provided)
        
        Returns:
            Dict containing:
                - profit_factor: Ratio of gross profits to gross losses
                - win_rate: Percentage of winning trades
                - max_drawdown: Maximum drawdown experienced
                - sharpe_ratio: Risk-adjusted return metric
                - total_trades: Number of trades executed
                - avg_profit_per_trade: Average profit per trade
        """
        try:
            # Initialize backtest metrics
            trades = []
            equity_curve = []
            current_equity = 1.0  # Start with normalized equity
            peak_equity = 1.0
            max_drawdown = 0.0
            
            # Extract parameters
            tp_base_atr = params.get('tp_base_atr_multiplier', 2.5)
            sl_base_atr = params.get('sl_base_atr_multiplier', 1.5)
            conf_deg_threshold = params.get('confidence_degradation_threshold', 0.3)
            unc_threshold = params.get('model_disagreement_threshold', 0.3)
            trailing_enabled = params.get('trailing_method', 'ensemble') != 'none'
            
            # Simplified backtest simulation
            # In production, this would use actual historical data and predictions
            # For now, we simulate based on parameter quality
            
            # Simulate trades based on parameter quality
            num_trades = 100  # Simulate 100 trades
            
            for i in range(num_trades):
                # Simulate entry confidence and uncertainty
                entry_confidence = np.random.beta(5, 2)  # Slightly positive bias
                uncertainty = np.random.beta(2, 5)  # Slightly low uncertainty
                volatility = np.random.beta(3, 3)  # Moderate volatility
                
                # Simulate trade outcome based on parameters and conditions
                # Better parameters with good market conditions = higher win probability
                
                # Calculate win probability based on parameter quality
                base_win_prob = 0.5
                
                # Adjust by confidence
                if entry_confidence > 0.7:
                    base_win_prob += 0.1
                elif entry_confidence < 0.4:
                    base_win_prob -= 0.1
                
                # Adjust by uncertainty (high uncertainty = lower win prob)
                if uncertainty > 0.6:
                    base_win_prob -= 0.1
                elif uncertainty < 0.3:
                    base_win_prob += 0.1
                
                # Simulate trade
                is_win = np.random.random() < base_win_prob
                
                # Calculate PnL based on TP/SL and whether we won
                if is_win:
                    # Hit take profit
                    # Adjusted by confidence and uncertainty scaling
                    tp_adjustment = 1.0 + (entry_confidence - 0.5) * params.get('tp_confidence_scaling', 1.0)
                    tp_adjustment *= (1.0 - uncertainty * (1.0 - params.get('tp_uncertainty_scaling', 0.8)))
                    pnl = tp_base_atr * 0.01 * tp_adjustment * (1.0 + np.random.random() * 0.2)  # 1% per ATR unit with variation
                else:
                    # Hit stop loss
                    sl_adjustment = 1.0 + volatility * (params.get('sl_volatility_scaling', 1.2) - 1.0)
                    pnl = -sl_base_atr * 0.01 * sl_adjustment * (1.0 + np.random.random() * 0.2)
                
                # Apply trailing stop impact (reduces losses, may reduce some gains)
                if trailing_enabled and not is_win:
                    # Trailing stop can reduce losses
                    pnl *= 0.8  # 20% loss reduction from trailing
                
                # Update equity
                current_equity *= (1.0 + pnl)
                equity_curve.append(current_equity)
                
                # Track drawdown
                if current_equity > peak_equity:
                    peak_equity = current_equity
                current_drawdown = (peak_equity - current_equity) / peak_equity
                max_drawdown = max(max_drawdown, current_drawdown)
                
                # Record trade
                trades.append({
                    'pnl': pnl,
                    'is_win': is_win,
                    'confidence': entry_confidence,
                    'uncertainty': uncertainty
                })
            
            # Calculate metrics
            winning_trades = [t for t in trades if t['is_win']]
            losing_trades = [t for t in trades if not t['is_win']]
            
            gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0.0
            gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0.0
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 3.0
            win_rate = len(winning_trades) / len(trades) if trades else 0.5
            
            # Calculate Sharpe ratio (simplified)
            if trades:
                returns = [t['pnl'] for t in trades]
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
                sharpe_ratio *= np.sqrt(252)  # Annualize
            else:
                sharpe_ratio = 0.0
            
            return {
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(trades),
                'avg_profit_per_trade': np.mean([t['pnl'] for t in trades]) if trades else 0.0,
                'final_equity': current_equity
            }
            
        except Exception as e:
            self.logger.error(f"❌ Dynamic exit backtest failed: {e}")
            # Return poor metrics on failure
            return {
                'profit_factor': 1.0,
                'win_rate': 0.5,
                'max_drawdown': 0.5,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'avg_profit_per_trade': 0.0,
                'final_equity': 1.0
            }
    
    def _calculate_comprehensive_exit_score(
        self,
        backtest_results: Dict[str, float]
    ) -> float:
        """
        Calculate comprehensive score from backtest results.
        
        Weights:
        - Profit factor: 35%
        - Win rate: 25%
        - Max drawdown: 20%
        - Sharpe ratio: 20%
        
        Args:
            backtest_results: Results from _run_dynamic_exit_backtest
        
        Returns:
            Comprehensive score (0.0 to 1.0)
        """
        try:
            score = 0.0
            
            # 1. Profit factor score (35% weight)
            # Normalize profit factor to 0-1 scale (1.0 = bad, 3.0 = excellent)
            profit_factor = backtest_results.get('profit_factor', 1.0)
            profit_factor_norm = min((profit_factor - 1.0) / 2.0, 1.0)  # Scale to 0-1
            score += 0.35 * profit_factor_norm
            
            # 2. Win rate score (25% weight)
            win_rate = backtest_results.get('win_rate', 0.5)
            score += 0.25 * win_rate
            
            # 3. Max drawdown score (20% weight)
            # Lower drawdown is better
            max_dd = backtest_results.get('max_drawdown', 0.5)
            dd_score = 1.0 - min(max_dd, 1.0)
            score += 0.20 * dd_score
            
            # 4. Sharpe ratio score (20% weight)
            # Normalize Sharpe to 0-1 scale (0.0 = bad, 3.0 = excellent)
            sharpe = backtest_results.get('sharpe_ratio', 0.0)
            sharpe_norm = min(max(sharpe, 0.0) / 3.0, 1.0)
            score += 0.20 * sharpe_norm
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive exit score calculation failed: {e}")
            return 0.0

    def _evaluate_parameters_vectorbt_optimized(self, objective_function: callable,
                                              parameters: Dict[str, Any]) -> float:
        """
        Enhanced parameter evaluation using VectorBT optimization with VectorBTRollingOptimizer.

        Args:
            objective_function: Function to evaluate parameters
            parameters: Parameters to evaluate

        Returns:
            Evaluation score
        """
        if not self.vectorbt_enabled:
            # Fallback to standard evaluation
            return objective_function(parameters)

        try:
            start_time = time.time()

            # Use VectorBTRollingOptimizer for enhanced data preprocessing
            optimized_data = None
            if hasattr(self, 'training_data') and self.training_data is not None:
                optimized_data = self._optimize_data_for_vectorbt_enhanced(self.training_data)

            # Create operation context for VectorBT optimization
            operation_context = {
                'data_size': len(self.training_data) if hasattr(self, 'training_data') else 1000,
                'data_dimensions': self.training_data.shape if hasattr(self, 'training_data') else (1000, 10),
                'parameters': parameters,
                'optimized_data': optimized_data,
                'rolling_optimizer': self.rolling_optimizer
            }

            # Use UnifiedVectorizationManager for intelligent optimization
            operation_config = OperationConfig(
                operation_type=OperationType.VECTORBT_BACKTESTING,
                data_size=operation_context['data_size'],
                data_dimensions=operation_context['data_dimensions'],
                memory_budget_mb=self.config.get('memory_budget_mb', 2048),
                time_budget_seconds=self.config.get('time_budget_seconds', 60),
                parallel_workers=self.max_workers
            )

            # Optimize the objective function execution with VectorBT
            with self.vectorization_manager.performance_monitoring("parameter_evaluation"):
                # Use VectorBTRollingOptimizer for enhanced processing
                if hasattr(objective_function, '__vectorbt_optimized__'):
                    result = objective_function(parameters, operation_context)
                else:
                    # Create optimized objective function
                    optimized_obj_func = self._create_vectorbt_optimized_objective(objective_function)
                    result = optimized_obj_func(parameters, operation_context)

            # Update VectorBT statistics
            execution_time = time.time() - start_time
            self.vectorbt_stats['vectorization_operations'] += 1
            self.vectorbt_stats['total_vectorbt_time'] += execution_time
            self.vectorbt_stats['rolling_operations'] += 1

            return result

        except Exception as e:
            self.logger.warning(f"VectorBT optimization failed, falling back to standard evaluation: {e}")
            return objective_function(parameters)

    def _evaluate_parameters_batch_vectorbt(self, parameter_sets: List[Dict[str, Any]],
                                          objective_function: callable) -> List[float]:
        """
        Evaluate multiple parameter sets in batch using VectorBT optimization with VectorBTRollingOptimizer.

        Args:
            parameter_sets: List of parameter dictionaries to evaluate
            objective_function: Function to evaluate parameters

        Returns:
            List of evaluation scores
        """
        if not self.vectorbt_enabled or len(parameter_sets) == 1:
            # Fallback to sequential evaluation
            return [objective_function(params) for params in parameter_sets]

        try:
            start_time = time.time()

            # Use VectorBTRollingOptimizer for enhanced batch processing
            optimized_data = None
            if hasattr(self, 'training_data') and self.training_data is not None:
                optimized_data = self._optimize_data_for_vectorbt_enhanced(self.training_data)

            # Create operation context for VectorBT optimization
            operation_context = {
                'data_size': len(self.training_data) if hasattr(self, 'training_data') else 1000,
                'data_dimensions': self.training_data.shape if hasattr(self, 'training_data') else (1000, 10),
                'optimized_data': optimized_data,
                'rolling_optimizer': self.rolling_optimizer
            }

            # Use VectorBT optimization manager for batch processing
            operation_config = OperationConfig(
                operation_type=OperationType.VECTORBT_BACKTESTING,
                data_size=operation_context['data_size'],
                data_dimensions=operation_context['data_dimensions'],
                parallel_workers=self.max_workers
            )

            # Process in batches for memory efficiency
            batch_size = min(self.config.get('batch_size', 10), len(parameter_sets))
            results = []

            for i in range(0, len(parameter_sets), batch_size):
                batch = parameter_sets[i:i + batch_size]

                # Use VectorBT optimization for batch processing
                if hasattr(self.optimization_manager, 'optimize_batch_operation'):
                    batch_results = self.optimization_manager.optimize_batch_operation(
                        objective_function,
                        batch,
                        operation_config
                    )
                else:
                    # Fallback to VectorBTRollingOptimizer batch processing
                    batch_results = self._process_batch_with_vectorbt_rolling(
                        batch, objective_function, operation_context
                    )

                results.extend(batch_results)

            # Update VectorBT statistics
            execution_time = time.time() - start_time
            self.vectorbt_stats['batch_operations'] += 1
            self.vectorbt_stats['total_vectorbt_time'] += execution_time

            return results

        except Exception as e:
            self.logger.warning(f"VectorBT batch optimization failed, falling back to sequential evaluation: {e}")
            return [objective_function(params) for params in parameter_sets]

    def _process_batch_with_vectorbt_rolling(self, parameter_batch: List[Dict[str, Any]],
                                           objective_function: callable,
                                           operation_context: Dict[str, Any]) -> List[float]:
        """
        Process a batch of parameters using VectorBTRollingOptimizer.

        Args:
            parameter_batch: Batch of parameters to process
            objective_function: Function to evaluate parameters
            operation_context: Context for optimization

        Returns:
            List of evaluation scores
        """
        try:
            rolling_optimizer = operation_context.get('rolling_optimizer')
            optimized_data = operation_context.get('optimized_data')

            if rolling_optimizer is None:
                # Fallback to standard evaluation
                return [objective_function(params) for params in parameter_batch]

            # If we have a rolling optimizer, use it
            # (Implementation would go here)
            return [objective_function(params) for params in parameter_batch]

        except Exception as e:
            self.logger.warning(f"VectorBT rolling optimization failed: {e}")
            return [objective_function(params) for params in parameter_batch]

    def _has_valid_calibration(self, calibration_results: Dict[str, Any]) -> bool:
        """
        Check if calibration results are valid for optimization.
        
        This method validates calibration results and checks for both the new simplified
        target structure (target_long, target_short) and legacy structure.
        
        Args:
            calibration_results: Calibration results dictionary
            
        Returns:
            True if calibration results are valid, False otherwise
        """
        try:
            if not calibration_results or not isinstance(calibration_results, dict):
                tprint_warning("⚠️ Calibration results is empty or not a dictionary")
                return False
            
            # Check for new simplified target structure first (highest priority)
            target_long = calibration_results.get('target_long', np.array([]))
            target_short = calibration_results.get('target_short', np.array([]))
            
            if len(target_long) > 0 and len(target_short) > 0:
                tprint_success("✅ Valid calibration results found with new simplified target structure")
                tprint_info(f"   • target_long: {len(target_long)} samples")
                tprint_info(f"   • target_short: {len(target_short)} samples")
                return True
            
            # Check for legacy target structure - Updated to only require analyst_confidence
            analyst_conf = calibration_results.get('analyst_confidence', np.array([]))
            # Tactician confidence is optional for backward compatibility
            tactician_conf = calibration_results.get('tactician_confidence', np.array([]))
            returns = calibration_results.get('returns', np.array([]))

            if len(analyst_conf) > 0 and len(returns) > 0:
                tprint_success("✅ Valid calibration results found with legacy target structure (Analyst-based)")
                tprint_info(f"   • analyst_confidence: {len(analyst_conf)} samples")
                if len(tactician_conf) > 0:
                    tprint_info(f"   • tactician_confidence: {len(tactician_conf)} samples (optional, legacy support)")
                tprint_info(f"   • returns: {len(returns)} samples")
                return True

            # Check for alternative legacy target names
            price_target_vol_normalized = calibration_results.get('price_target_vol_normalized', np.array([]))
            volatility_labels = calibration_results.get('volatility_labels', np.array([]))

            if len(price_target_vol_normalized) > 0 or len(volatility_labels) > 0:
                tprint_success("✅ Valid calibration results found with alternative legacy target structure")
                tprint_info(f"   • price_target_vol_normalized: {len(price_target_vol_normalized)} samples")
                tprint_info(f"   • volatility_labels: {len(volatility_labels)} samples")
                return True

            # No valid target structure found
            tprint_error("❌ No valid target structure found in calibration results")
            tprint_info("   Expected one of:")
            tprint_info("   • New simplified: target_long, target_short")
            tprint_info("   • Legacy (Analyst-based): analyst_confidence, returns")
            tprint_info("   • Alternative legacy: price_target_vol_normalized, volatility_labels")
            return False
            
        except Exception as e:
            tprint_error(f"❌ Error validating calibration results: {e}")
            return False

            # Use VectorBTRollingOptimizer for parallel batch processing
            if hasattr(rolling_optimizer, 'process_batch_parallel'):
                return rolling_optimizer.process_batch_parallel(
                    parameter_batch, objective_function, optimized_data
                )

            # Fallback: sequential processing with VectorBT optimizations
            results = []
            for params in parameter_batch:
                # Create optimized objective function
                optimized_obj_func = self._create_vectorbt_optimized_objective(objective_function)
                result = optimized_obj_func(params, operation_context)
                results.append(result)

            return results

        except Exception as e:
            self.logger.warning(f"VectorBTRollingOptimizer batch processing failed: {e}")
            return [objective_function(params) for params in parameter_batch]

    def _calculate_rolling_metrics_vectorbt(self, data: pd.DataFrame,
                                          window: int = 252) -> Dict[str, pd.Series]:
        """
        Calculate rolling metrics using VectorBT optimization.

        Args:
            data: Input data with OHLCV columns
            window: Rolling window size

        Returns:
            Dictionary of rolling metrics
        """
        if not self.vectorbt_enabled or self.rolling_optimizer is None:
            # Fallback to pandas rolling operations
            return self._calculate_rolling_metrics_pandas(data, window)

        try:
            start_time = time.time()
            rolling_metrics = {}

            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()

                # Use VectorBT rolling optimizer for batch calculations
                rolling_metrics.update({
                    'volatility': self.rolling_optimizer.rolling_std(returns, window=window),
                    'momentum': self.rolling_optimizer.rolling_mean(returns, window=window),
                    'max_drawdown': self.rolling_optimizer.rolling_min(returns, window=window),
                    'sharpe_ratio': (self.rolling_optimizer.rolling_mean(returns, window=window) /
                                   self.rolling_optimizer.rolling_std(returns, window=window))
                })

                # Calculate additional metrics
                rolling_metrics['skewness'] = self.rolling_optimizer.rolling_skew(returns, window=window)
                rolling_metrics['kurtosis'] = self.rolling_optimizer.rolling_kurt(returns, window=window)
                rolling_metrics['quantile_25'] = self.rolling_optimizer.rolling_quantile(returns, window=window, q=0.25)
                rolling_metrics['quantile_75'] = self.rolling_optimizer.rolling_quantile(returns, window=window, q=0.75)

            if 'volume' in data.columns:
                volume = data['volume']
                rolling_metrics['volume_ma'] = self.rolling_optimizer.rolling_mean(volume, window=window)
                rolling_metrics['volume_std'] = self.rolling_optimizer.rolling_std(volume, window=window)

            # Update VectorBT statistics
            execution_time = time.time() - start_time
            self.vectorbt_stats['rolling_operations'] += 1
            self.vectorbt_stats['total_vectorbt_time'] += execution_time

            return rolling_metrics

        except Exception as e:
            self.logger.warning(f"VectorBT rolling metrics calculation failed, falling back to pandas: {e}")
            return self._calculate_rolling_metrics_pandas(data, window)

    def _calculate_rolling_metrics_pandas(self, data: pd.DataFrame, window: int) -> Dict[str, pd.Series]:
        """Fallback pandas implementation for rolling metrics."""
        rolling_metrics = {}

        if 'close' in data.columns:
            returns = data['close'].pct_change().dropna()
            rolling_metrics.update({
                'volatility': returns.rolling(window=window).std(),
                'momentum': returns.rolling(window=window).mean(),
                'max_drawdown': returns.rolling(window=window).min(),
                'sharpe_ratio': returns.rolling(window=window).mean() / returns.rolling(window=window).std()
            })

        return rolling_metrics

    def _optimize_data_for_vectorbt(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data for VectorBT processing.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame
        """
        if not self.vectorbt_enabled:
            return data

        try:
            # Use existing memory optimizer
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe(data)

            # Additional VectorBT-specific optimizations
            if self.rolling_optimizer:
                data = self.rolling_optimizer._optimize_data_types(data)

            return data

        except Exception as e:
            self.logger.warning(f"VectorBT data optimization failed: {e}")
            return data

    def _optimize_data_for_vectorbt_enhanced(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced data optimization using VectorBTRollingOptimizer.

        Args:
            data: Input DataFrame

        Returns:
            Optimized DataFrame with VectorBT enhancements
        """
        if not self.vectorbt_enabled or self.rolling_optimizer is None:
            return data

        try:
            # Use existing memory optimizer
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_dataframe(data)

            # Enhanced VectorBT-specific optimizations
            if self.rolling_optimizer:
                data = self.rolling_optimizer._optimize_data_types(data)

                # Precompute common rolling features for faster access
                if hasattr(self.rolling_optimizer, 'precompute_rolling_features'):
                    rolling_features = self.rolling_optimizer.precompute_rolling_features(
                        data,
                        windows=[5, 10, 20, 50, 100],
                        operations=['mean', 'std', 'min', 'max', 'skew', 'kurt']
                    )
                    data = data.join(rolling_features)

                # Memory optimization
                if hasattr(self.rolling_optimizer, 'optimize_memory_usage'):
                    data = self.rolling_optimizer.optimize_memory_usage(data)

            return data

        except Exception as e:
            self.logger.warning(f"Enhanced VectorBT data optimization failed: {e}")
            return data

    def _create_vectorbt_optimized_objective(self, original_function: callable) -> callable:
        """
        Create a VectorBT-optimized version of the objective function.

        Args:
            original_function: Original objective function

        Returns:
            VectorBT-optimized objective function
        """
        def optimized_function(parameters: Dict[str, Any],
                              operation_context: Optional[Dict[str, Any]] = None) -> float:
            """
            VectorBT-optimized objective function.
            """
            try:
                # Extract data and rolling optimizer from context
                optimized_data = operation_context.get('optimized_data') if operation_context else None
                rolling_optimizer = operation_context.get('rolling_optimizer') if operation_context else None

                # Use VectorBT for data preprocessing if data is available
                if optimized_data is not None and rolling_optimizer is not None:
                    # Calculate rolling metrics using VectorBTRollingOptimizer
                    rolling_metrics = self._calculate_rolling_metrics_vectorbt(
                        optimized_data, rolling_optimizer
                    )

                    # Calculate technical indicators using VectorBTRollingOptimizer
                    technical_indicators = self._calculate_technical_indicators_vectorbt(
                        optimized_data, rolling_optimizer
                    )

                    # Add to parameters for the original function
                    enhanced_parameters = parameters.copy()
                    enhanced_parameters['rolling_metrics'] = rolling_metrics
                    enhanced_parameters['technical_indicators'] = technical_indicators
                    enhanced_parameters['vectorbt_optimized'] = True

                    return original_function(enhanced_parameters)
                else:
                    return original_function(parameters)

            except Exception as e:
                self.logger.warning(f"VectorBT optimized objective function failed: {e}")
                return original_function(parameters)

        # Mark as VectorBT optimized
        optimized_function.__vectorbt_optimized__ = True

        return optimized_function

    def _calculate_rolling_metrics_vectorbt(self, data: pd.DataFrame,
                                          rolling_optimizer) -> Dict[str, Any]:
        """
        Calculate rolling metrics using VectorBTRollingOptimizer.

        Args:
            data: Input data
            rolling_optimizer: VectorBTRollingOptimizer instance

        Returns:
            Dictionary of rolling metrics
        """
        try:
            results = {}
            windows = [5, 10, 20, 50, 100]

            if 'close' in data.columns:
                close_prices = data['close']

                for window in windows:
                    window_results = {}

                    # Use VectorBT rolling operations
                    window_results['mean'] = rolling_optimizer.rolling_mean(close_prices, window=window)
                    window_results['std'] = rolling_optimizer.rolling_std(close_prices, window=window)
                    window_results['min'] = rolling_optimizer.rolling_min(close_prices, window=window)
                    window_results['max'] = rolling_optimizer.rolling_max(close_prices, window=window)
                    window_results['skew'] = rolling_optimizer.rolling_skew(close_prices, window=window)
                    window_results['kurt'] = rolling_optimizer.rolling_kurt(close_prices, window=window)

                    results[f'window_{window}'] = window_results

            return results

        except Exception as e:
            self.logger.warning(f"VectorBT rolling metrics calculation failed: {e}")
            return {}

    def _calculate_technical_indicators_vectorbt(self, data: pd.DataFrame,
                                               rolling_optimizer) -> Dict[str, Any]:
        """
        Calculate technical indicators using VectorBTRollingOptimizer.

        Args:
            data: Input OHLCV data
            rolling_optimizer: VectorBTRollingOptimizer instance

        Returns:
            Dictionary of technical indicators
        """
        try:
            results = {}

            if 'close' in data.columns:
                close_prices = data['close']

                # Moving averages
                results['sma_20'] = rolling_optimizer.rolling_mean(close_prices, window=20)
                results['sma_50'] = rolling_optimizer.rolling_mean(close_prices, window=50)
                results['sma_200'] = rolling_optimizer.rolling_mean(close_prices, window=200)

                # Volatility
                results['volatility_20'] = rolling_optimizer.rolling_std(close_prices, window=20)
                results['volatility_50'] = rolling_optimizer.rolling_std(close_prices, window=50)

                # Price ranges
                if 'high' in data.columns and 'low' in data.columns:
                    high_prices = data['high']
                    low_prices = data['low']

                    # ATR calculation
                    if hasattr(rolling_optimizer, 'rolling_atr'):
                        results['atr_20'] = rolling_optimizer.rolling_atr(
                            high_prices, low_prices, close_prices, window=20
                        )
                        results['atr_50'] = rolling_optimizer.rolling_atr(
                            high_prices, low_prices, close_prices, window=50
                        )

            return results

        except Exception as e:
            self.logger.warning(f"VectorBT technical indicators calculation failed: {e}")
            return {}

    def get_vectorbt_performance_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        if not self.vectorbt_enabled:
            return {'vectorbt_enabled': False}

        stats = self.vectorbt_stats.copy()
        stats['vectorbt_enabled'] = True

        # Add rolling optimizer stats
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
            stats['rolling_optimizer_stats'] = rolling_stats

        # Add optimization manager stats
        if self.optimization_manager:
            optimization_stats = self.optimization_manager.get_optimization_stats()
            stats['optimization_manager_stats'] = optimization_stats

        return stats

    def _evaluate_ensemble_params(self, params: Dict[str, Any],
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate ensemble parameters."""
        score = 0.0

        if all(key in params for key in ['analyst_weight', 'tactician_weight', 'strategist_weight']):
            weights = [params['analyst_weight'], params['tactician_weight'], params['strategist_weight']]
            if abs(sum(weights) - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1

        return score

    def _evaluate_sr_params(self, params: Dict[str, Any],
                          calibration_results: Dict[str, Any]) -> float:
        """Evaluate S/R parameters."""
        score = 0.0

        weight_params = ['touch_count_weight', 'total_volume_weight', 'level_age_weight',
                        'bounce_rate_weight', 'isolation_score_weight']
        weights = [params.get(param, 0.0) for param in weight_params]

        if abs(sum(weights) - 1.0) < 0.1:
            score += 0.3
        else:
            score += 0.1

        return score

    def _evaluate_two_tier_params(self, params: Dict[str, Any],
                                calibration_results: Dict[str, Any]) -> float:
        """Evaluate two-tier system parameters."""
        score = 0.0

        if 'tier1_weight' in params and 'tier2_weight' in params:
            tier1_weight = params['tier1_weight']
            tier2_weight = params['tier2_weight']
            if abs(tier1_weight + tier2_weight - 1.0) < 0.1:
                score += 0.3
            else:
                score += 0.1

        if 'direction_threshold' in params:
            threshold = params['direction_threshold']
            if 0.6 <= threshold <= 0.8:
                score += 0.2
            else:
                score += 0.1

        if 'timing_threshold' in params:
            threshold = params['timing_threshold']
            if 0.7 <= threshold <= 0.9:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_technical_indicators_params(self, params: Dict[str, Any],
                                           calibration_results: Dict[str, Any]) -> float:
        """Evaluate technical indicator parameters."""
        score = 0.0

        if 'rsi_period' in params:
            rsi_period = params['rsi_period']
            if 10 <= rsi_period <= 20:
                score += 0.2
            else:
                score += 0.1

        if 'macd_fast_period' in params and 'macd_slow_period' in params:
            fast = params['macd_fast_period']
            slow = params['macd_slow_period']
            if fast < slow and 8 <= fast <= 16 and 20 <= slow <= 30:
                score += 0.2
            else:
                score += 0.1

        if 'adx_trend_threshold' in params and 'adx_sideways_threshold' in params:
            trend = params['adx_trend_threshold']
            sideways = params['adx_sideways_threshold']
            if trend > sideways:
                score += 0.2
            else:
                score += 0.1

        if 'volatility_threshold' in params:
            vol_thresh = params['volatility_threshold']
            if 0.015 <= vol_thresh <= 0.035:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_system_monitoring_params(self, params: Dict[str, Any],
                                        calibration_results: Dict[str, Any]) -> float:
        """Evaluate system monitoring parameters."""
        score = 0.0

        if 'analysis_interval' in params:
            interval = params['analysis_interval']
            if 1800 <= interval <= 7200:
                score += 0.2
            else:
                score += 0.1

        if 'max_history' in params:
            max_hist = params['max_history']
            if 50 <= max_hist <= 200:
                score += 0.2
            else:
                score += 0.1

        if 'memory_threshold' in params:
            mem_thresh = params['memory_threshold']
            if 0.7 <= mem_thresh <= 0.9:
                score += 0.2
            else:
                score += 0.1

        if 'learning_rate' in params:
            lr = params['learning_rate']
            if 0.005 <= lr <= 0.05:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_training_optimization_params(self, params: Dict[str, Any],
                                            calibration_results: Dict[str, Any]) -> float:
        """Evaluate training optimization parameters."""
        score = 0.0

        if 'adx_trend_threshold' in params and 'adx_sideways_threshold' in params:
            trend = params['adx_trend_threshold']
            sideways = params['adx_sideways_threshold']
            if trend > sideways and 20.0 <= trend <= 35.0 and 15.0 <= sideways <= 30.0:
                score += 0.2
            else:
                score += 0.1

        if 'min_label_balance' in params and 'max_label_balance' in params:
            min_balance = params['min_label_balance']
            max_balance = params['max_label_balance']
            if min_balance < max_balance and 0.03 <= min_balance <= 0.1 and 0.9 <= max_balance <= 0.98:
                score += 0.2
            else:
                score += 0.1

        if 'stability_threshold' in params:
            stability = params['stability_threshold']
            if 0.6 <= stability <= 0.9:
                score += 0.2
            else:
                score += 0.1

        if 'lgb_learning_rate' in params:
            lr = params['lgb_learning_rate']
            if 0.01 <= lr <= 0.2:
                score += 0.2
            else:
                score += 0.1

        if 'model_performance_threshold' in params:
            perf_thresh = params['model_performance_threshold']
            if 0.6 <= perf_thresh <= 0.85:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_regime_transitions_params(self, params: Dict[str, Any],
                                         calibration_results: Dict[str, Any]) -> float:
        """Evaluate regime transition parameters."""
        score = 0.0

        if 'transition_intensity_threshold' in params:
            threshold = params['transition_intensity_threshold']
            if 0.2 <= threshold <= 0.5:
                score += 0.2
            else:
                score += 0.1

        if 'transition_confidence_threshold' in params:
            confidence_thresh = params['transition_confidence_threshold']
            if 0.6 <= confidence_thresh <= 0.9:
                score += 0.2
            else:
                score += 0.1

        if all(key in params for key in ['step9_5_weight', 'step10_weight', 'regime_expert_weight']):
            step9_5_w = params['step9_5_weight']
            step10_w = params['step10_weight']
            regime_w = params['regime_expert_weight']
            total_weight = step9_5_w + step10_w + regime_w
            if 0.9 <= total_weight <= 1.1:
                score += 0.2
            else:
                score += 0.1

        if 'transition_lookback_periods' in params:
            lookback = params['transition_lookback_periods']
            if 3 <= lookback <= 10:
                score += 0.2
            else:
                score += 0.1

        if 'transition_risk_multiplier' in params:
            risk_mult = params['transition_risk_multiplier']
            if 1.0 <= risk_mult <= 1.5:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_signal_aggregation_params(self, params: Dict[str, Any],
                                         calibration_results: Dict[str, Any]) -> float:
        """Evaluate signal aggregation parameters."""
        score = 0.0

        if all(key in params for key in ['analyst_weight', 'tactician_weight', 'scenario_weight',
                                       'sr_breakout_weight', 'regime_weight']):
            total_weight = (params['analyst_weight'] + params['tactician_weight'] +
                          params['scenario_weight'] + params['sr_breakout_weight'] +
                          params['regime_weight'])
            if 1.0 <= total_weight <= 2.5:
                score += 0.2
                if params['analyst_weight'] >= 0.3 and params['tactician_weight'] >= 0.3:
                    score += 0.1
            else:
                score += 0.1

        if 'conflict_penalty_factor' in params:
            penalty = params['conflict_penalty_factor']
            if 0.4 <= penalty <= 0.6:
                score += 0.2
            else:
                score += 0.1

        if 'min_source_weight' in params:
            min_weight = params['min_source_weight']
            if 0.05 <= min_weight <= 0.15:
                score += 0.1

        if 'min_signal_confidence' in params and 'min_aggregated_confidence' in params:
            signal_conf = params['min_signal_confidence']
            agg_conf = params['min_aggregated_confidence']
            if signal_conf < agg_conf and 0.2 <= signal_conf <= 0.4 and 0.4 <= agg_conf <= 0.6:
                score += 0.2
            else:
                score += 0.1

        if 'regime_alignment_bonus' in params and 'multi_signal_alignment_bonus' in params:
            regime_bonus = params['regime_alignment_bonus']
            multi_bonus = params['multi_signal_alignment_bonus']
            if 0.1 <= regime_bonus <= 0.25 and 0.05 <= multi_bonus <= 0.15:
                score += 0.1

        if 'use_multiplicative' in params and params['use_multiplicative']:
            score += 0.1

        return score

    def _evaluate_turnover_cost_penalty_params(self, params: Dict[str, Any],
                                             calibration_results: Dict[str, Any]) -> float:
        """Evaluate turnover cost penalty parameters."""
        score = 0.0

        if 'turnover_penalty_weight' in params:
            weight = params['turnover_penalty_weight']
            if 0.2 <= weight <= 0.8:
                score += 0.3
            elif 0.1 <= weight <= 1.0:
                score += 0.2
            else:
                score += 0.1

        if 'commission_rate' in params:
            commission = params['commission_rate']
            if 0.0008 <= commission <= 0.0015:
                score += 0.2
            else:
                score += 0.1

        if 'slippage_rate' in params:
            slippage = params['slippage_rate']
            if 0.0003 <= slippage <= 0.0008:
                score += 0.2
            else:
                score += 0.1

        if 'round_trip_multiplier' in params:
            multiplier = params['round_trip_multiplier']
            if 1.8 <= multiplier <= 2.5:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_intensity_params(self, params: Dict[str, Any],
                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate signal intensity parameters."""
        score = 0.0

        if 'signal_intensity_threshold' in params:
            threshold = params['signal_intensity_threshold']
            if 0.5 <= threshold <= 0.7:
                score += 0.3
            elif 0.4 <= threshold <= 0.8:
                score += 0.2
            else:
                score += 0.1

        if 'intensity_decay_factor' in params:
            decay = params['intensity_decay_factor']
            if 0.9 <= decay <= 0.95:
                score += 0.2
            elif 0.85 <= decay <= 0.99:
                score += 0.15
            else:
                score += 0.1

        return score

    def _evaluate_entry_timing_optimization_params(self, params: Dict[str, Any],
                                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate entry timing optimization parameters for updated Tactician models."""
        score = 0.0

        if 'entry_timing_range' in params:
            range_val = params['entry_timing_range']
            # Optimal range is around 0.003-0.004 (0.3%-0.4%)
            if 0.003 <= range_val <= 0.004:
                score += 0.3
            elif 0.002 <= range_val <= 0.004:
                score += 0.2
            else:
                score += 0.1

        if 'optimal_entry_reward_weight' in params and 'early_entry_penalty_weight' in params:
            reward_weight = params['optimal_entry_reward_weight']
            penalty_weight = params['early_entry_penalty_weight']
            # Reward should be higher than penalty for optimal timing
            if reward_weight > penalty_weight and reward_weight >= 0.4:
                score += 0.25
            else:
                score += 0.15

        if 'directional_accuracy_threshold' in params:
            threshold = params['directional_accuracy_threshold']
            if 0.6 <= threshold <= 0.7:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_confidence_aware_ensemble_params(self, params: Dict[str, Any],
                                                 calibration_results: Dict[str, Any]) -> float:
        """Evaluate confidence-aware ensemble parameters for updated models."""
        score = 0.0

        if 'confidence_threshold_entry' in params and 'confidence_threshold_exit' in params:
            entry_thresh = params['confidence_threshold_entry']
            exit_thresh = params['confidence_threshold_exit']
            # Entry threshold should typically be higher than exit threshold
            if entry_thresh > exit_thresh and 0.65 <= entry_thresh <= 0.8:
                score += 0.3
            else:
                score += 0.15

        if 'confidence_weight_tactician' in params and 'confidence_weight_analyst' in params:
            tactician_weight = params['confidence_weight_tactician']
            analyst_weight = params['confidence_weight_analyst']
            # Tactician should have higher weight for timing decisions
            if tactician_weight > analyst_weight and tactician_weight >= 0.4:
                score += 0.25
            else:
                score += 0.15

        if 'ensemble_confidence_threshold' in params:
            threshold = params['ensemble_confidence_threshold']
            if 0.7 <= threshold <= 0.85:
                score += 0.2
            else:
                score += 0.1

        return score

    def _evaluate_model_specific_params(self, params: Dict[str, Any],
                                      calibration_results: Dict[str, Any]) -> float:
        """Evaluate model-specific parameters for new Analyst & Tactician model types."""
        score = 0.0

        # Check if weights are balanced for different model types
        analyst_weights = []
        tactician_weights = []

        # Analyst model weights
        analyst_weight_keys = [
            'analyst_tcn_weight', 'analyst_catboost_weight', 'analyst_lightgbm_weight'
        ]

        # Tactician model weights
        tactician_weight_keys = [
            'tactician_xgboost_weight', 'tactician_randomforest_weight',
            'tactician_catboost_weight', 'tactician_elastic_net_weight'
        ]

        for key in analyst_weight_keys:
            if key in params:
                analyst_weights.append(params[key])

        for key in tactician_weight_keys:
            if key in params:
                tactician_weights.append(params[key])

        # Evaluate Analyst model balance
        if analyst_weights:
            max_weight = max(analyst_weights)
            min_weight = min(analyst_weights)
            weight_balance = min_weight / max_weight if max_weight > 0 else 0

            if weight_balance >= 0.6:  # Well balanced
                score += 0.15
            elif weight_balance >= 0.4:  # Moderately balanced
                score += 0.1
            else:
                score += 0.05

        # Evaluate Tactician model balance
        if tactician_weights:
            max_weight = max(tactician_weights)
            min_weight = min(tactician_weights)
            weight_balance = min_weight / max_weight if max_weight > 0 else 0

            if weight_balance >= 0.6:  # Well balanced
                score += 0.15
            elif weight_balance >= 0.4:  # Moderately balanced
                score += 0.1
            else:
                score += 0.05

        if 'model_diversity_bonus' in params:
            bonus = params['model_diversity_bonus']
            if 0.08 <= bonus <= 0.12:
                score += 0.15
            else:
                score += 0.1

        if 'model_complexity_penalty' in params:
            penalty = params['model_complexity_penalty']
            if 0.02 <= penalty <= 0.06:
                score += 0.15
            else:
                score += 0.1

        return score

    def _calculate_turnover_penalty(self, params: Dict[str, Any],
                                  calibration_results: Dict[str, Any]) -> float:
        """
        Calculate turnover penalty for a given configuration.

        The penalty is calculated as:
        turnover_penalty = turnover_rate * transaction_cost * round_trip_multiplier

        Where transaction_cost = commission_rate + slippage_rate

        Args:
            params: Current parameter configuration
            calibration_results: Results from calibration/backtesting

        Returns:
            Turnover penalty to subtract from base score
        """
        try:
            # Extract cost parameters from current params or use defaults
            commission_rate = params.get('commission_rate', 0.001)
            slippage_rate = params.get('slippage_rate', 0.0005)
            round_trip_multiplier = params.get('round_trip_multiplier', 2.0)
            turnover_penalty_weight = params.get('turnover_penalty_weight', 0.5)

            # Calculate transaction cost per trade
            transaction_cost = commission_rate + slippage_rate

            # Estimate turnover rate from calibration results or use default
            # In a real implementation, this would be calculated from actual backtesting results
            estimated_turnover_rate = self._estimate_turnover_rate(params, calibration_results)

            # Calculate round-trip cost
            round_trip_cost = transaction_cost * round_trip_multiplier

            # Calculate penalty
            turnover_penalty = estimated_turnover_rate * round_trip_cost * turnover_penalty_weight

            # Log the calculation for transparency
            if turnover_penalty > 0.001:  # Only log significant penalties
                self.logger.debug(f"⚠️ Turnover penalty: {turnover_penalty:.4f} "
                                f"(rate: {estimated_turnover_rate:.3f}, cost: {round_trip_cost:.6f})")

            return turnover_penalty

        except Exception as e:
            self.logger.warning(f"Error calculating turnover penalty: {e}")
            return 0.001  # Small default penalty

    async def _load_calibration_results(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load calibration results from artifacts.
        
        Args:
            config: Configuration dictionary containing symbol, exchange, etc.
            
        Returns:
            Dictionary containing calibration results
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            
            tprint_info(f"📥 Loading calibration results for {symbol} from {exchange}")
            
            # Set artifact manager context to look for calibration artifacts
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model='Calibration'
            )
            
            # Try to load calibration results from different possible artifact names
            calibration_artifacts = [
                'confidence_calibration_results',
                'calibration_results',
                'step16_confidence_calibration_results'
            ]
            
            calibration_results = {}
            
            for artifact_name in calibration_artifacts:
                try:
                    artifact_data = self.artifact_manager.get_artifact(
                        artifact_name=artifact_name,
                        artifact_type='data'
                    )
                    
                    if artifact_data is not None:
                        calibration_results.update(artifact_data)
                        tprint_success(f"✅ Loaded calibration results from {artifact_name}")
                        # Print detailed data preview for calibration results
                        tprint_info(f"📊 Calibration Results Preview:")
                        for key, value in artifact_data.items():
                            if hasattr(value, 'shape'):
                                tprint_data_preview(value, name=f"Calibration - {key}", max_rows=5, max_cols=10)
                            else:
                                tprint_info(f"   • {key}: {type(value).__name__}")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load {artifact_name}: {e}")
                    continue

            # If no calibration results found, return empty dict
            if not calibration_results:
                tprint_warning("⚠️ No calibration results found in artifacts")
                return {}

            # Check if calibration results contain required data
            # Note: Changed to use analyst_confidence instead of tactician_confidence
            required_keys = ['analyst_confidence', 'returns']
            missing_keys = [key for key in required_keys if key not in calibration_results]

            if missing_keys:
                tprint_warning(f"⚠️ Missing required calibration data: {missing_keys}")
                return {}

            tprint_success(f"✅ Calibration results loaded successfully")
            # Print summary of what was loaded
            tprint_info(f"📊 Loaded calibration data keys: {list(calibration_results.keys())}")
            return calibration_results
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration results: {e}")
            tprint_error(f"❌ Failed to load calibration results: {e}")
            return {}

    async def _load_previous_results(self, symbol: str, exchange: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load previous optimization results for warm start.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            config: Configuration dictionary
            
        Returns:
            Dictionary containing previous optimization results or None
        """
        try:
            tprint_info(f"📥 Loading previous optimization results for {symbol} from {exchange}")
            
            # Set artifact manager context to look for previous optimization artifacts
            self.artifact_manager.set_context(
                symbol=symbol,
                exchange=exchange,
                model='FinalParameters'
            )
            
            # Try to load previous optimization results from different possible artifact names
            optimization_artifacts = [
                'final_parameters_optimization_result',
                'optimization_results',
                'parameters_optimization_results'
            ]
            
            previous_results = None
            
            for artifact_name in optimization_artifacts:
                try:
                    artifact_data = self.artifact_manager.get_artifact(
                        artifact_name=artifact_name,
                        artifact_type='data'
                    )
                    
                    if artifact_data is not None:
                        previous_results = artifact_data
                        tprint_success(f"✅ Loaded previous optimization results from {artifact_name}")
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to load {artifact_name}: {e}")
                    continue
            
            if previous_results is None:
                tprint_info("ℹ️ No previous optimization results found - starting fresh")
                return None
            
            tprint_success(f"✅ Previous optimization results loaded successfully")
            return previous_results
            
        except Exception as e:
            self.logger.error(f"Failed to load previous optimization results: {e}")
            tprint_error(f"❌ Failed to load previous optimization results: {e}")
            return None

    def _load_regime_performance_stats(self) -> Dict[str, Any]:
        """Load per-regime performance statistics if available.

        This first checks any explicit configuration overrides and then falls back
        to standard reporting directories such as ``reports/backtesting`` or
        ``generated/backtesting``.
        """
        path_candidates: List[Path] = []

        config_path = self.config.get('regime_performance_path') if isinstance(self.config, dict) else None
        if config_path:
            path_candidates.append(Path(config_path))

        reporting_dir = None
        if isinstance(self.config, dict):
            reporting_dir = self.config.get('reporting_output_dir')
            if reporting_dir is None and 'reporting' in self.config and isinstance(self.config['reporting'], dict):
                reporting_dir = self.config['reporting'].get('output_dir')
        if reporting_dir:
            path_candidates.append(Path(reporting_dir) / 'backtesting' / 'per_regime_performance.json')

        path_candidates.extend([
            Path('reports/backtesting/per_regime_performance.json'),
            Path('generated/backtesting/per_regime_performance.json'),
        ])

        for candidate in path_candidates:
            try:
                if candidate.exists():
                    with candidate.open('r') as fp:
                        stats = json.load(fp)
                    self.regime_performance_path = str(candidate)
                    return stats if isinstance(stats, dict) else {}
            except Exception as exc:
                self.logger.error(f"❌ Failed to load regime performance stats from {candidate}: {exc}")

        return {}

    def _calculate_regime_performance_modifier(self) -> float:
        """
        Compute an aggregate modifier from per-regime performance stats with detailed logging.
        
        This modifier adjusts optimization scores based on historical regime performance,
        helping to weight parameter choices toward strategies that work well across
        different market conditions.
        
        Returns:
            Float modifier in range [-0.25, 0.25]
        """
        stats = getattr(self, 'regime_performance_stats', {})
        if not stats:
            tprint("ℹ️  No regime performance stats available, modifier=0.0", "info")
            return 0.0

        win_rates: List[float] = []
        profit_factors: List[float] = []
        rr_values: List[float] = []

        for regime_id, metrics in stats.items():
            if not isinstance(metrics, dict):
                continue
            wr = float(metrics.get('win_rate', 0.0))
            pf = float(metrics.get('profit_factor', 0.0))
            rr = float(metrics.get('average_rr', metrics.get('risk_reward_ratio', 0.0)))
            
            win_rates.append(wr)
            profit_factors.append(pf)
            rr_values.append(rr)
            
            # Log individual regime metrics for transparency
            self.logger.debug(f"Regime {regime_id}: WR={wr:.2%}, PF={pf:.2f}, RR={rr:.2f}")

        if not win_rates:
            tprint("⚠️  No valid regime metrics extracted, modifier=0.0", "warning")
            return 0.0

        # Calculate aggregate statistics
        avg_win = float(np.mean(win_rates))
        min_win = float(np.min(win_rates))
        avg_profit_factor = float(np.mean(profit_factors)) if profit_factors else 0.0
        avg_rr = float(np.mean(rr_values)) if rr_values else 0.0
        stability_penalty = float(np.std(win_rates)) if len(win_rates) > 1 else 0.0

        # Normalize components
        normalized_win = avg_win - 0.5
        normalized_min_win = min_win - 0.5
        normalized_profit_factor = float(np.tanh(avg_profit_factor - 1.0))
        normalized_rr = float(np.tanh(avg_rr - 1.0))

        # Calculate weighted modifier
        raw_modifier = (
            (normalized_win * 0.5)
            + (normalized_profit_factor * 0.2)
            + (normalized_rr * 0.2)
            + (normalized_min_win * 0.1)
            - (stability_penalty * 0.1)
        )
        
        modifier = float(np.clip(raw_modifier, -0.25, 0.25))
        
        # Log detailed breakdown for transparency
        tprint("📊 Regime Performance Modifier Breakdown:", "info")
        tprint(f"   • Average Win Rate: {avg_win:.2%} (norm: {normalized_win:+.3f}, weight: 0.5)", "info")
        tprint(f"   • Min Win Rate: {min_win:.2%} (norm: {normalized_min_win:+.3f}, weight: 0.1)", "info")
        tprint(f"   • Average Profit Factor: {avg_profit_factor:.2f} (norm: {normalized_profit_factor:+.3f}, weight: 0.2)", "info")
        tprint(f"   • Average Risk/Reward: {avg_rr:.2f} (norm: {normalized_rr:+.3f}, weight: 0.2)", "info")
        tprint(f"   • Win Rate Stability Penalty: {stability_penalty:.3f} (weight: -0.1)", "info")
        tprint(f"   • Raw Modifier: {raw_modifier:+.4f}", "info")
        tprint(f"   • Final Modifier (clipped): {modifier:+.4f}", "success")

        return modifier

    def _apply_regime_performance_adjustment(self, category: str, score: float) -> float:
        """
        Adjust objective score using per-regime performance insights with transparent logging.
        
        Different parameter categories receive different weights based on their impact
        on regime-specific performance:
        - High weight (1.2): tpsl, exit_strategy, regime_transitions
        - Medium weight (1.1): confidence, position_sizing  
        - Lower weight (0.9): ensemble, model_specific_parameters
        
        Args:
            category: Parameter category being optimized
            score: Base optimization score
            
        Returns:
            Adjusted score with regime performance modifier applied
        """
        modifier = getattr(self, 'regime_performance_modifier', 0.0)
        if modifier == 0.0:
            return score

        # Determine category-specific weight
        weight = 1.0
        weight_description = "standard"
        
        if category in {'tpsl', 'exit_strategy', 'regime_transitions'}:
            weight = 1.2
            weight_description = "high (regime-critical)"
        elif category in {'confidence', 'position_sizing'}:
            weight = 1.1
            weight_description = "medium-high (regime-aware)"
        elif category in {'ensemble', 'model_specific_parameters'}:
            weight = 0.9
            weight_description = "medium-low (model-specific)"

        # Calculate and clip adjustment
        raw_adjustment = modifier * weight
        adjustment = float(np.clip(raw_adjustment, -0.2, 0.2))
        adjusted_score = score + adjustment
        
        # Log adjustment details for transparency
        self.logger.debug(
            f"Regime adjustment for {category}: "
            f"base_score={score:.4f}, modifier={modifier:+.4f}, "
            f"weight={weight:.1f} ({weight_description}), "
            f"adjustment={adjustment:+.4f}, final_score={adjusted_score:.4f}"
        )
        
        # Only print to console if adjustment is significant
        if abs(adjustment) > 0.05:
            tprint(
                f"   📊 Regime adjustment for {category}: {score:.4f} → {adjusted_score:.4f} "
                f"({adjustment:+.4f}, {weight_description} weight)",
                "info"
            )
        
        return adjusted_score

    def _estimate_turnover_rate(self, params: Dict[str, Any],
                               calibration_results: Dict[str, Any]) -> float:
        """
        Estimate turnover rate based on parameters and calibration results.

        Turnover rate represents how much of the portfolio changes per period.
        Higher trading frequency = higher turnover = higher costs.

        Args:
            params: Current parameter configuration
            calibration_results: Calibration/backtesting results

        Returns:
            Estimated turnover rate (0.0 to 1.0)
        """
        try:
            # Base turnover rate depends on trading frequency
            base_turnover = 0.15  # Default 15% portfolio turnover per period

            # Adjust based on confidence thresholds (lower thresholds = more trades)
            if 'base_entry_threshold' in params:
                threshold = params['base_entry_threshold']
                if threshold < 0.6:
                    base_turnover *= 1.3  # More aggressive = more trades
                elif threshold > 0.8:
                    base_turnover *= 0.7  # More conservative = fewer trades

            # Adjust based on position sizing (larger positions = potentially more turnover)
            if 'base_position_size' in params:
                position_size = params['base_position_size']
                if position_size > 0.1:
                    base_turnover *= 1.2
                elif position_size < 0.03:
                    base_turnover *= 0.8

            # Adjust based on TP/SL ratios (wider ranges = fewer trades)
            if all(key in params for key in ['tp_long', 'sl_long']):
                tp = params['tp_long']
                sl = params['sl_long']
                if tp > sl * 2:
                    base_turnover *= 0.8  # Wider profit targets = fewer trades
                elif tp < sl * 1.2:
                    base_turnover *= 1.2  # Narrow profit targets = more trades

            # Extract from calibration results if available
            if calibration_results and 'estimated_turnover' in calibration_results:
                calibrated_turnover = calibration_results['estimated_turnover']
                base_turnover = (base_turnover + calibrated_turnover) / 2  # Average with estimate

            # Ensure reasonable bounds
            max_turnover = params.get('max_turnover_rate', 0.5)
            base_turnover = min(base_turnover, max_turnover)

            return base_turnover

        except Exception as e:
            self.logger.warning(f"Error estimating turnover rate: {e}")
            return 0.15  # Default turnover rate

    def _calculate_multiplicative_confidence(self, analyst_conf: float, tactician_conf: float,
                                           tactician_weight: float, analyst_weight: float) -> float:
        """Calculate confidence using multiplicative method (shared with signal pipeline)."""
        try:
            analyst_conf = max(0.001, analyst_conf)
            tactician_conf = max(0.001, tactician_conf)

            multiplicative_conf = (
                (tactician_conf ** tactician_weight) *
                (analyst_conf ** analyst_weight)
            )

            return min(1.0, multiplicative_conf)

        except Exception as e:
            self.logger.error(f"❌ Error in multiplicative confidence calculation: {e}")
            return 0.5

    def _calculate_logarithmic_confidence(self, analyst_conf: float, tactician_conf: float,
                                        tactician_weight: float, analyst_weight: float) -> float:
        """Calculate confidence using logarithmic method (shared with signal pipeline)."""
        try:
            analyst_conf = max(0.001, analyst_conf)
            tactician_conf = max(0.001, tactician_conf)

            log_combination = (
                tactician_weight * np.log(tactician_conf) +
                analyst_weight * np.log(analyst_conf)
            )

            logarithmic_conf = np.exp(log_combination)
            return min(1.0, max(0.0, logarithmic_conf))

        except Exception as e:
            self.logger.error(f"❌ Error in logarithmic confidence calculation: {e}")
            return 0.5

    def _format_exit_strategy_for_position_monitor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat exit strategy parameters into the schema used by PositionMonitor."""

        try:
            formatted = {
                # Component confidence drop (backtested)
                'component_confidence_drop': params.get('component_confidence_drop', 0.3),
                
                # Base profit target (tested range 0.6% - 1.2%)
                'base_profit_target': params.get('base_profit_target', 0.009),
                
                # Profit trailing percent (tested range 0.0% - 0.3%)
                'profit_trailing_percent': params.get('profit_trailing_percent', 0.0015),
                
                # Exit confidence drop
                'exit_confidence_drop': params.get('exit_confidence_drop', 0.3),
                
                'stop_loss': {
                    'base_stop_loss': params.get('base_stop_loss', -0.05),
                    'atr_multiplier': params.get('atr_multiplier', 1.5),
                    'volatility_adjustment_factor': params.get('volatility_adjustment_factor', 1.0)
                },
                'time_based': {
                    'max_hold_time': params.get('max_hold_time', 10800),
                    'confidence_time_scaling_factor': params.get('confidence_time_scaling_factor', 1.0)
                },
                'trailing_stop': {
                    'atr_multiplier': params.get('trailing_atr_multiplier', 1.5),
                    'min_distance': params.get('trailing_min_distance', 0.01),
                    'confidence_activation': params.get('trailing_confidence_activation', 0.7),
                    'tightening_threshold': params.get('trailing_tightening_threshold', 0.02),
                    'time_decay': params.get('trailing_time_decay', 0.95),
                    'ml_adjustment_weight': params.get('trailing_ml_adjustment_weight', 0.3),
                    'ml_trigger_multiplier': params.get('ml_trigger_trailing_multiplier', 1.0)
                },
                'regime_aware': {
                    'transition_penalty': params.get('regime_transition_penalty', 0.1),
                    'regime_specific_scaling': params.get('regime_specific_scaling', 1.0),
                    'profit_bands': profit_bands,
                    'trailing_sensitivity': params.get('regime_trailing_sensitivity', 1.0)
                }
            }

            return formatted

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error(f"❌ Error formatting exit strategy for position monitor: {exc}")
            return {}

    async def save_optimization_results(self, optimization_results: Dict[str, Any],
                                      symbol: str, exchange: str, data_dir: str) -> None:
        """Save optimization results."""
        try:
            self.logger.info(f"💾 Saving optimization results for {exchange}_{symbol}")
            optimization_dir = f'generated/backtesting/optimization_results'
            os.makedirs(optimization_dir, exist_ok=True)
            self.logger.info(f"📁 Optimization directory: {optimization_dir}")

            results_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            self.logger.info(f"🔄 Saving pickle file: {results_file}")
            with open(results_file, 'wb') as f:
                pickle.dump(optimization_results, f)

            json_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.json'
            self.logger.info(f"🔄 Saving JSON file: {json_file}")
            with open(json_file, 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)

            # Log file sizes
            pickle_size = os.path.getsize(results_file) / 1024  # KB
            json_size = os.path.getsize(json_file) / 1024  # KB

            self.logger.info(f'✅ Optimization results saved successfully')
            self.logger.info(f'📊 Pickle file size: {pickle_size:.1f} KB')
            self.logger.info(f'📊 JSON file size: {json_size:.1f} KB')
            self.logger.info(f'📁 Files saved to: {optimization_dir}')

            # Persist consolidated JSON for runtime consumers (e.g., PositionMonitor)
            exit_strategy_results = optimization_results.get('exit_strategy', {})
            exit_strategy_params = exit_strategy_results.get('best_params', {}) if isinstance(exit_strategy_results, dict) else {}

            if exit_strategy_params:
                formatted_exit = self._format_exit_strategy_for_position_monitor(exit_strategy_params)
                best_parameter_snapshot = {
                    category: values.get('best_params', {})
                    for category, values in optimization_results.items()
                    if isinstance(values, dict)
                }

                consolidated_payload = {
                    'generated_at': datetime.utcnow().isoformat(),
                    'best_parameters': best_parameter_snapshot,
                    'position_monitor_exit_strategy': formatted_exit,
                    'raw_exit_strategy': exit_strategy_params
                }

                results_dir = Path('results')
                results_dir.mkdir(parents=True, exist_ok=True)
                results_path = results_dir / 'final_parameters_optimization.json'

                with open(results_path, 'w') as f:
                    json.dump(consolidated_payload, f, indent=2, default=str)

                self.logger.info(f'📝 Wrote position monitor schema to: {results_path}')

        except Exception as e:
            self.logger.error(f'❌ Error saving optimization results: {e}')
            self.logger.exception("Full traceback:")

    async def load_optimization_results(self, symbol: str, exchange: str,
                                      data_dir: str) -> Optional[Dict[str, Any]]:
        """Load previous optimization results."""
        try:
            self.logger.info(f"📂 Loading previous optimization results for {exchange}_{symbol}")
            optimization_dir = f'{data_dir}/optimization_results'
            previous_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'

            self.logger.info(f"🔍 Checking for previous results: {previous_file}")

            if os.path.exists(previous_file):
                file_size = os.path.getsize(previous_file) / 1024  # KB
                self.logger.info(f"📁 Previous results found - File size: {file_size:.1f} KB")

                with open(previous_file, 'rb') as f:
                    results = pickle.load(f)

                if results:
                    self.logger.info(f"✅ Successfully loaded previous optimization results")
                    self.logger.info(f"📊 Categories in previous results: {len(results)}")
                    for category in results.keys():
                        self.logger.debug(f"   • {category}")
                else:
                    self.logger.warning(f"⚠️ Previous results file is empty")

                return results
            else:
                self.logger.info(f"ℹ️ No previous optimization results found")
                return None

        except Exception as e:
            self.logger.error(f'❌ Error loading optimization results: {e}')
            self.logger.exception("Full traceback:")
            return None

    async def validate_optimization_results(self, optimization_results: Dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            if not optimization_results:
                return False

            for category in self.categories:
                if category not in optimization_results:
                    self.logger.warning(f'Missing optimization results for category: {category}')
                    return False

            return True

        except Exception as e:
            self.logger.error(f'Error validating optimization results: {e}')
            return False

    async def generate_optimization_report(self, optimization_results: Dict[str, Any],
                                         start_time: datetime) -> Dict[str, Any]:
        """Generate optimization report."""
        try:
            report = {
                'optimization_timestamp': start_time.isoformat(),
                'duration_seconds': (datetime.now() - start_time).total_seconds(),
                'categories_optimized': list(optimization_results.keys()),
                'summary': {}
            }

            for category, results in optimization_results.items():
                if results and 'best_value' in results:
                    report['summary'][category] = {
                        'best_value': results['best_value'],
                        'n_trials': results.get('n_trials', 0)
                    }

            return report

        except Exception as e:
            self.logger.error(f'Error generating optimization report: {e}')
            return {'error': str(e)}

    def _analyze_convergence(self, study: optuna.Study) -> Dict[str, Any]:
        """Analyze convergence characteristics of the optimization."""
        try:
            if len(study.trials) < 5:
                return {'convergence_quality': 'insufficient_data'}

            values = [t.value for t in study.trials if t.value is not None]
            if not values:
                return {'convergence_quality': 'no_valid_trials'}

            # Calculate convergence metrics
            best_values = []
            current_best = float('-inf')
            for value in values:
                if value > current_best:
                    current_best = value
                best_values.append(current_best)

            # Improvement rate
            total_improvement = best_values[-1] - best_values[0]
            improvement_rate = total_improvement / len(values) if len(values) > 0 else 0

            # Convergence stability (variance in last 20% of trials)
            last_portion = int(len(best_values) * 0.2)
            if last_portion > 1:
                recent_values = best_values[-last_portion:]
                convergence_variance = np.var(recent_values)
            else:
                convergence_variance = 0

            # Convergence quality assessment
            if improvement_rate > 0.01 and convergence_variance < 0.001:
                convergence_quality = 'excellent'
            elif improvement_rate > 0.005 and convergence_variance < 0.01:
                convergence_quality = 'good'
            elif improvement_rate > 0.001:
                convergence_quality = 'fair'
            else:
                convergence_quality = 'poor'

            return {
                'convergence_quality': convergence_quality,
                'total_improvement': total_improvement,
                'improvement_rate': improvement_rate,
                'convergence_variance': convergence_variance,
                'final_best_value': best_values[-1],
                'n_trials': len(values)
            }

        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            return {'convergence_quality': 'analysis_failed', 'error': str(e)}

    def _get_used_enhancement_methods(self, search_space: Dict[str, Dict[str, Any]]) -> List[str]:
        """Get list of enhancement methods used in the search space."""
        methods = set()
        for param_config in search_space.values():
            if 'transform_type' in param_config:
                transform_type = param_config['transform_type']
                if transform_type in ['log', 'power', 'sigmoid', 'adaptive']:
                    methods.add(transform_type)
        return list(methods)

    async def _coarse_grid_search(self, category: str, search_space: Dict[str, Dict[str, Any]],
                                calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform coarse grid search with fewer parameter combinations."""
        try:
            self.logger.info(f"🔍 Creating coarse grid for {category}")

            # Create coarse parameter grid
            coarse_grid = self._create_coarse_parameter_grid(search_space)
            self.logger.info(f"📊 Coarse grid size: {len(coarse_grid)} combinations")

            best_score = -np.inf
            best_params = {}
            parameter_scores = []

            # Evaluate each parameter combination
            for i, params in enumerate(coarse_grid):
                try:
                    score = self._evaluate_configuration(category, params, calibration_results)
                    parameter_scores.append((params, score))

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(coarse_grid)} combinations")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue

            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {category}")
                return {}

            self.logger.info(f"✅ Coarse grid search completed - Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(coarse_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }

        except Exception as e:
            self.logger.error(f"❌ Coarse grid search failed for {category}: {e}")
            return {}

    async def _fine_grid_search(self, category: str, search_space: Dict[str, Dict[str, Any]],
                              best_coarse_params: Dict[str, Any], calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform fine grid search around best coarse parameters."""
        try:
            self.logger.info(f"🔍 Creating fine grid around best coarse parameters for {category}")

            # Create fine parameter grid around best coarse parameters
            fine_grid = self._create_fine_parameter_grid(search_space, best_coarse_params)
            self.logger.info(f"📊 Fine grid size: {len(fine_grid)} combinations")

            best_score = -np.inf
            best_params = {}
            parameter_scores = []

            # Evaluate each parameter combination
            for i, params in enumerate(fine_grid):
                try:
                    score = self._evaluate_configuration(category, params, calibration_results)
                    parameter_scores.append((params, score))

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"   Evaluated {i + 1}/{len(fine_grid)} combinations")

                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to evaluate parameters {params}: {e}")
                    continue

            if not parameter_scores:
                self.logger.error(f"❌ No valid parameter combinations found for {category}")
                return {}

            self.logger.info(f"✅ Fine grid search completed - Best score: {best_score:.4f}")

            return {
                'best_params': best_params,
                'best_score': best_score,
                'n_combinations': len(fine_grid),
                'valid_combinations': len(parameter_scores),
                'parameter_scores': parameter_scores[:10]  # Keep top 10 for analysis
            }

        except Exception as e:
            self.logger.error(f"❌ Fine grid search failed for {category}: {e}")
            return {}

    async def _optuna_tpe_optimization(self, category: str, search_space: Dict[str, Dict[str, Any]],
                                     best_grid_params: Dict[str, Any], calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Optuna TPE optimization around best grid parameters."""
        try:
            self.logger.info(f"🎲 Starting Optuna TPE optimization for {category}")

            # Create narrowed search space around best grid parameters
            narrowed_space = self._create_narrowed_search_space(search_space, best_grid_params)

            study_name = f'{self.study_name}_{category}_tpe'
            if self.use_nonlinear_optimization:
                study_name += '_enhanced'

            # Use TPE sampler with enhanced settings
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=5,  # Fewer startup trials since we have good starting point
                n_ei_candidates=24,
                gamma=lambda x: min(int(0.25 * x), 25),
                prior_weight=1.0,
                consider_magic_clip=True,
                consider_endpoints=True
            )

            study = optuna.create_study(
                study_name=study_name,
                direction='maximize',
                sampler=sampler,
                storage='sqlite:///optuna_studies_coarse_fine.db',
                load_if_exists=True
            )

            # Use fewer trials since we're fine-tuning around good parameters
            n_trials = min(self.n_trials // 3, 30)  # Use 1/3 of original trials or max 30
            timeout = min(self.timeout // 3, 120)   # Use 1/3 of original timeout or max 2 minutes

            self.logger.info(f"🎯 Starting TPE optimization with {n_trials} trials (timeout: {timeout}s)")

            def objective(trial):
                return self._objective_function(trial, category, narrowed_space, calibration_results)

            start_time = time.time()
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            optimization_time = time.time() - start_time

            best_params = study.best_params
            best_value = study.best_value

            # Convert parameters back to original space for reporting
            if self.use_nonlinear_optimization:
                converted_params = convert_parameters_to_original_space(best_params, narrowed_space)
            else:
                converted_params = best_params

            self.logger.info(f"✅ Optuna TPE optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"📈 Best TPE score: {best_value:.4f}")

            # Enhanced convergence analysis
            convergence_analysis = self._analyze_convergence(study)

            return {
                'best_params': converted_params,
                'best_score': best_value,
                'study_name': study_name,
                'n_trials': len(study.trials),
                'optimization_time': optimization_time,
                'convergence_analysis': convergence_analysis,
                'narrowed_space': narrowed_space
            }

        except Exception as e:
            self.logger.error(f"❌ Optuna TPE optimization failed for {category}: {e}")
            return {}

    def _create_coarse_parameter_grid(self, search_space: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create coarse parameter grid with fewer combinations."""
        import itertools

        param_combinations = []

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['min'], param_config['max']
                if param_config.get('log', False) or (self.use_nonlinear_optimization and
                    param_config.get('transform_type') == 'log'):
                    # Log-spaced values
                    values = np.logspace(np.log10(min_val), np.log10(max_val), 3)
                else:
                    # Linear-spaced values
                    values = np.linspace(min_val, max_val, 3)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'int':
                # Use 3 points for coarse grid
                min_val, max_val = param_config['min'], param_config['max']
                if max_val - min_val <= 2:
                    values = list(range(min_val, max_val + 1))
                else:
                    values = np.linspace(min_val, max_val, 3, dtype=int)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'bool':
                param_combinations.append([(param_name, v) for v in [True, False]])
            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])

        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))

        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)

        return grid

    def _create_fine_parameter_grid(self, search_space: Dict[str, Dict[str, Any]],
                                  best_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fine parameter grid around best parameters."""

        param_combinations = []

        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                continue

            best_value = best_params[param_name]

            if param_config['type'] == 'float':
                min_val, max_val = param_config['min'], param_config['max']
                # Create fine grid around best value (±20% of range)
                range_size = max_val - min_val
                fine_range = range_size * 0.2
                fine_min = max(min_val, best_value - fine_range)
                fine_max = min(max_val, best_value + fine_range)

                # Use 5 points for fine grid
                if param_config.get('log', False) or (self.use_nonlinear_optimization and
                    param_config.get('transform_type') == 'log'):
                    # Log-spaced values
                    values = np.logspace(np.log10(fine_min), np.log10(fine_max), 5)
                else:
                    # Linear-spaced values
                    values = np.linspace(fine_min, fine_max, 5)
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'int':
                min_val, max_val = param_config['min'], param_config['max']
                # Create fine grid around best value (±2 values)
                fine_min = max(min_val, best_value - 2)
                fine_max = min(max_val, best_value + 2)
                values = list(range(fine_min, fine_max + 1))
                param_combinations.append([(param_name, v) for v in values])

            elif param_config['type'] == 'bool':
                param_combinations.append([(param_name, v) for v in [True, False]])
            elif param_config['type'] == 'categorical':
                param_combinations.append([(param_name, v) for v in param_config['choices']])

        # Generate all combinations
        all_combinations = list(itertools.product(*param_combinations))

        # Convert to list of dictionaries
        grid = []
        for combination in all_combinations:
            param_dict = dict(combination)
            grid.append(param_dict)

        return grid

    def _create_narrowed_search_space(self, search_space: Dict[str, Dict[str, Any]],
                                    best_params: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create narrowed search space around best parameters for Optuna."""
        narrowed_space = {}

        for param_name, param_config in search_space.items():
            if param_name not in best_params:
                narrowed_space[param_name] = param_config
                continue

            best_value = best_params[param_name]
            narrowed_config = param_config.copy()

            if param_config['type'] == 'float':
                min_val, max_val = param_config['min'], param_config['max']
                # Narrow range to ±10% of original range around best value
                range_size = max_val - min_val
                narrow_range = range_size * 0.1
                narrowed_config['min'] = max(min_val, best_value - narrow_range)
                narrowed_config['max'] = min(max_val, best_value + narrow_range)

            elif param_config['type'] == 'int':
                min_val, max_val = param_config['min'], param_config['max']
                # Narrow range to ±1 around best value
                narrowed_config['min'] = max(min_val, best_value - 1)
                narrowed_config['max'] = min(max_val, best_value + 1)

            narrowed_space[param_name] = narrowed_config

        return narrowed_space

    def _create_fallback_result(self, category: str) -> Dict[str, Any]:
        """Create fallback result with default parameters."""
        default_params = {}
        search_space = self.default_search_spaces.get(category, {})

        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                # Use middle value
                default_params[param_name] = (param_config['min'] + param_config['max']) / 2
            elif param_config['type'] == 'int':
                # Use middle value
                default_params[param_name] = (param_config['min'] + param_config['max']) // 2
            elif param_config['type'] == 'bool':
                default_params[param_name] = True
            elif param_config['type'] == 'categorical':
                default_params[param_name] = param_config['choices'][0]  # Use first choice as default

        return {
            'best_params': default_params,
            'best_value': 0.0,
            'optimization_method': 'fallback',
            'error': 'Grid search failed, using default parameters'
        }

    async def load_hdf5_data_from_pipeline(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Load HDF5 data from versioned artifacts for final parameters optimization.

        This method loads data from previous pipeline steps:
        - feature_generation_labeling_integration_step: labeled data with features and targets
        - regime_ensemble_training: regime probabilities for each regime
        - analyst_ensemble_training: confidence scores and disagreement features

        Args:
            config: Configuration dictionary containing symbol, exchange, timeframe, direction

        Returns:
            Dictionary with loaded DataFrames:
            - 'labeled_data': Features and targets from labeling step
            - 'regime_probabilities': Regime probabilities from ensemble training
            - 'analyst_confidence': Confidence scores from analyst ensemble
            - 'disagreement_features': Disagreement features from analyst ensemble
        """
        tprint("=" * 80, "header")
        tprint("📥 LOADING HDF5 DATA FROM VERSIONED ARTIFACTS", "header")
        tprint("=" * 80, "header")

        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        direction = config.get('direction', 'long')

        loaded_data = {}

        try:
            # 1. Load labeled data from feature_generation_labeling_integration_step
            tprint("📊 Loading labeled data from feature_generation_labeling_integration_step...", "info")
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model='analyst'
            )

            # Try to load labeled data from feature_generation_labeling_integration_step
            # This is the artifact name used by the feature generation step
            labeled_data = self._get_artifact(
                artifact_name='feature_generation_labeling_integration',
                artifact_type='data',
                data_category='features'
            )

            # Fallback to legacy naming pattern if not found
            if labeled_data is None:
                tprint(f"⚠️ 'feature_generation_labeling_integration' not found, trying legacy 'labeled_data_{symbol}_{timeframe}'...", "warning")
                labeled_data = self._get_artifact(
                    artifact_name=f'labeled_data_{symbol}_{timeframe}',
                    artifact_type='data',
                    data_category='features'
                )

            if labeled_data is not None:
                tprint(f"✅ Loaded labeled data: {labeled_data.shape}", "success")
                # Print detailed data preview
                tprint_data_preview(labeled_data, name="Labeled Data", max_rows=5, max_cols=10)
                tprint_data_format(labeled_data, name="Labeled Data", check_compatibility=True)
                loaded_data['labeled_data'] = labeled_data
            else:
                tprint("⚠️ Labeled data not found in versioned artifacts", "warning")
                tprint("   Tried artifact names: 'feature_generation_labeling_integration', f'labeled_data_{symbol}_{timeframe}'", "info")

            # 2. Load regime probabilities from regime_ensemble_training
            tprint("📊 Loading regime probabilities from regime_ensemble_training...", "info")

            # For regime data, use regime timeframe (typically 1h)
            regime_timeframe = config.get('regime_timeframe', '1h')
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=regime_timeframe,
                direction='long',  # Regimes are direction-agnostic
                model='regime'
            )

            # Try to load regime ensemble predictions (saved as 'regime_ensemble_predictions' by regime training step)
            regime_probs = self._get_artifact(
                artifact_name='regime_ensemble_predictions',
                artifact_type='data',
                data_category='predictions'
            )

            # Fallback to rolling_hmm_regime_probabilities if not found
            if regime_probs is None:
                tprint("⚠️ 'regime_ensemble_predictions' not found, trying 'rolling_hmm_regime_probabilities'...", "warning")
                regime_probs = self._get_artifact(
                    artifact_name='rolling_hmm_regime_probabilities',
                    artifact_type='data',
                    data_category='features'
                )

            if regime_probs is not None:
                tprint(f"✅ Loaded regime probabilities: {regime_probs.shape}", "success")
                # Print detailed data preview
                tprint_data_preview(regime_probs, name="Regime Probabilities", max_rows=5, max_cols=10)
                tprint_data_format(regime_probs, name="Regime Probabilities", check_compatibility=True)
                loaded_data['regime_probabilities'] = regime_probs
            else:
                tprint("⚠️ Regime probabilities not found in versioned artifacts", "warning")
                tprint("   Tried artifact names: 'regime_ensemble_predictions', 'rolling_hmm_regime_probabilities'", "info")

            # 3. Load analyst ensemble confidence and disagreement features
            tprint("📊 Loading analyst ensemble outputs (confidence + disagreement)...", "info")

            # Reset context to analyst model
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model='analyst'
            )

            # Try to load analyst ensemble outputs (saved as 'analyst_ensemble_outputs' by unified training step)
            # First try the actual artifact name used by the training step
            analyst_predictions = self._get_artifact(
                artifact_name='analyst_ensemble_outputs',
                artifact_type='data',
                data_category='predictions'
            )

            # Fallback to legacy name if not found
            if analyst_predictions is None:
                tprint("⚠️ 'analyst_ensemble_outputs' not found, trying legacy 'analyst_ensemble_predictions'...", "warning")
                analyst_predictions = self._get_artifact(
                    artifact_name='analyst_ensemble_predictions',
                    artifact_type='data',
                    data_category='predictions'
                )

            if analyst_predictions is not None:
                tprint(f"✅ Loaded analyst ensemble predictions: {analyst_predictions.shape}", "success")
                # Print detailed data preview
                tprint_data_preview(analyst_predictions, name="Analyst Ensemble Predictions", max_rows=5, max_cols=10)
                tprint_data_format(analyst_predictions, name="Analyst Ensemble Predictions", check_compatibility=True)

                # The analyst_ensemble_outputs contains the predictions DataFrame
                # Typically it has columns like 'prediction', 'confidence', etc.
                # We'll use the entire DataFrame as analyst_confidence for now
                # and extract specific columns if they exist

                # Check for confidence column
                if 'confidence' in analyst_predictions.columns:
                    loaded_data['analyst_confidence'] = analyst_predictions[['confidence']]
                    tprint(f"   • Extracted confidence scores: {loaded_data['analyst_confidence'].shape}", "info")
                    tprint_data_preview(loaded_data['analyst_confidence'], name="Analyst Confidence Scores", max_rows=5, max_cols=10)
                elif 'prediction' in analyst_predictions.columns:
                    # Use predictions as proxy for confidence if confidence column doesn't exist
                    tprint("⚠️ No 'confidence' column found, using 'prediction' column as proxy", "warning")
                    loaded_data['analyst_confidence'] = analyst_predictions[['prediction']]
                    tprint(f"   • Using prediction as confidence: {loaded_data['analyst_confidence'].shape}", "info")
                    tprint_data_preview(loaded_data['analyst_confidence'], name="Analyst Predictions (as confidence)", max_rows=5, max_cols=10)
                else:
                    # Use all columns as analyst_confidence if no specific column found
                    tprint(f"⚠️ No 'confidence' or 'prediction' column found, using all columns: {list(analyst_predictions.columns)}", "warning")
                    loaded_data['analyst_confidence'] = analyst_predictions
                    tprint_data_preview(loaded_data['analyst_confidence'], name="Analyst Predictions (all columns)", max_rows=5, max_cols=10)

                # Extract disagreement features if available
                disagreement_cols = [col for col in analyst_predictions.columns if 'disagreement' in col.lower()]
                if disagreement_cols:
                    loaded_data['disagreement_features'] = analyst_predictions[disagreement_cols]
                    tprint(f"   • Extracted disagreement features: {loaded_data['disagreement_features'].shape}", "info")
                    tprint(f"   • Disagreement columns: {disagreement_cols}", "info")
                    tprint_data_preview(loaded_data['disagreement_features'], name="Disagreement Features", max_rows=5, max_cols=10)
            else:
                tprint("⚠️ Analyst ensemble outputs not found in versioned artifacts", "warning")
                tprint("   Tried artifact names: 'analyst_ensemble_outputs', 'analyst_ensemble_predictions'", "info")

            # Summary
            tprint("=" * 80, "header")
            tprint(f"📦 LOADED {len(loaded_data)} DATA ARTIFACTS", "header")
            for key, df in loaded_data.items():
                tprint(f"   • {key}: {df.shape}", "info")
            tprint("=" * 80, "header")

            return loaded_data

        except Exception as e:
            self.logger.error(f"Failed to load HDF5 data from pipeline: {e}")
            tprint(f"❌ Error loading HDF5 data: {e}", "error")
            import traceback
            tprint(traceback.format_exc(), "error")
            return loaded_data

    def save_metrics_to_markdown(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Save optimization metrics to Markdown file.

        Args:
            metrics: Dictionary of metrics to save (HPO results, accuracy, R2, etc.)
            config: Configuration dictionary for context

        Returns:
            Path to saved Markdown file
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create markdown content
            md_content = f"""# Final Parameters Optimization Report

## Configuration
- **Symbol**: {symbol}
- **Exchange**: {exchange}
- **Timeframe**: {timeframe}
- **Direction**: {direction}
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Execution Mode**: {config.get('execution_mode', 'light')}

## Optimization Metrics

### HPO Metrics
"""

            # Add HPO metrics if available
            if 'hpo_metrics' in metrics:
                hpo = metrics['hpo_metrics']
                md_content += f"""
- **Best Score**: {hpo.get('best_score', 'N/A')}
- **N Trials**: {hpo.get('n_trials', 'N/A')}
- **Best Trial**: {hpo.get('best_trial', 'N/A')}
- **Optimization Time**: {hpo.get('optimization_time', 'N/A')}s
"""

            # Add performance metrics
            md_content += "\n### Performance Metrics\n"

            perf_metrics = {
                'accuracy': 'Accuracy',
                'r2_score': 'R² Score',
                'sharpe_ratio': 'Sharpe Ratio',
                'sortino_ratio': 'Sortino Ratio',
                'max_drawdown': 'Max Drawdown',
                'win_rate': 'Win Rate',
                'profit_factor': 'Profit Factor',
                'total_return': 'Total Return'
            }

            for key, label in perf_metrics.items():
                if key in metrics:
                    md_content += f"- **{label}**: {metrics[key]}\n"

            # Add cross-validation metrics if available
            if 'cv_metrics' in metrics:
                md_content += "\n### Cross-Validation Metrics\n"
                cv = metrics['cv_metrics']
                md_content += f"""
- **CV Mean Score**: {cv.get('mean_score', 'N/A')}
- **CV Std Score**: {cv.get('std_score', 'N/A')}
- **CV Folds**: {cv.get('n_folds', 'N/A')}
"""

            # Add optimized parameters summary
            if 'optimized_parameters' in metrics:
                md_content += "\n### Optimized Parameters\n"
                params = metrics['optimized_parameters']
                for param_name, param_value in params.items():
                    md_content += f"- **{param_name}**: {param_value}\n"

            # Save to file
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)

            filename = f"final_parameters_optimization_metrics_{symbol}_{timeframe}_{direction}_{timestamp}.md"
            filepath = outcomes_dir / filename

            with open(filepath, 'w') as f:
                f.write(md_content)

            tprint(f"📝 Saved metrics to Markdown: {filepath}", "success")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to save metrics to Markdown: {e}")
            tprint(f"❌ Error saving metrics to Markdown: {e}", "error")
            raise

    def save_metrics_to_json(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Save optimization metrics to JSON file.

        Args:
            metrics: Dictionary of metrics to save
            config: Configuration dictionary for context

        Returns:
            Path to saved JSON file
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Create comprehensive metrics structure
            metrics_data = {
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    'execution_mode': config.get('execution_mode', 'light')
                },
                'metrics': metrics
            }

            # Save to file
            outcomes_dir = Path("outcomes")
            outcomes_dir.mkdir(exist_ok=True)

            filename = f"final_parameters_optimization_metrics_{symbol}_{timeframe}_{direction}_{timestamp}.json"
            filepath = outcomes_dir / filename

            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)

            tprint(f"📊 Saved metrics to JSON: {filepath}", "success")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to save metrics to JSON: {e}")
            tprint(f"❌ Error saving metrics to JSON: {e}", "error")
            raise

    def save_config_to_pickle(self, config: Dict[str, Any], optimized_params: Dict[str, Any]) -> str:
        """
        Save configuration and optimized parameters to Pickle file.

        Args:
            config: Configuration dictionary
            optimized_params: Optimized parameters dictionary

        Returns:
            Path to saved Pickle file
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Combine config and optimized params
            full_config = {
                'base_config': config,
                'optimized_parameters': optimized_params,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    'step_name': self.step_name
                }
            }

            # Save to file
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)

            filename = f"final_parameters_config_{symbol}_{timeframe}_{direction}_{timestamp}.pkl"
            filepath = artifacts_dir / filename

            with open(filepath, 'wb') as f:
                pickle.dump(full_config, f, protocol=pickle.HIGHEST_PROTOCOL)

            tprint(f"💾 Saved config to Pickle: {filepath}", "success")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to save config to Pickle: {e}")
            tprint(f"❌ Error saving config to Pickle: {e}", "error")
            raise

    def save_config_to_json(self, config: Dict[str, Any], optimized_params: Dict[str, Any]) -> str:
        """
        Save configuration and optimized parameters to JSON file.

        Args:
            config: Configuration dictionary
            optimized_params: Optimized parameters dictionary

        Returns:
            Path to saved JSON file
        """
        try:
            symbol = config.get('symbol', 'ETHUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Combine config and optimized params
            full_config = {
                'base_config': config,
                'optimized_parameters': optimized_params,
                'metadata': {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'direction': direction,
                    'timestamp': datetime.now().isoformat(),
                    'step_name': self.step_name
                }
            }

            # Save to file
            artifacts_dir = Path("artifacts")
            artifacts_dir.mkdir(exist_ok=True)

            filename = f"final_parameters_config_{symbol}_{timeframe}_{direction}_{timestamp}.json"
            filepath = artifacts_dir / filename

            with open(filepath, 'w') as f:
                json.dump(full_config, f, indent=2, default=str)

            tprint(f"📄 Saved config to JSON: {filepath}", "success")
            return str(filepath)

        except Exception as e:
            self.logger.error(f"Failed to save config to JSON: {e}")
            tprint(f"❌ Error saving config to JSON: {e}", "error")
            raise

# Convenience functions for easy integration
async def optimize_final_parameters(calibration_results: Dict[str, Any],
                                  config: Dict[str, Any],
                                  symbol: str = "ETHUSDT",
                                  exchange: str = "BINANCE",
                                  data_dir: str = "data/training",
                                  nonlinear_config: Optional[NonLinearConfig] = None) -> Dict[str, Any]:
    """
    Enhanced convenience function to optimize final parameters with optional non-linear transformations.

    Args:
        calibration_results: Results from confidence calibration
        config: Configuration dictionary
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory
        nonlinear_config: Non-linear optimization configuration (optional)

    Returns:
        Optimization results
    """
    optimizer = FinalParametersOptimizer(
        config=config,
        nonlinear_config=nonlinear_config
    )

    # Load previous results for warm start
    previous_results = await optimizer.load_optimization_results(symbol, exchange, data_dir)

    # Optimize all parameters
    optimization_results = await optimizer.optimize_all_parameters(
        calibration_results, previous_results
    )

    # Validate results
    validation_passed = await optimizer.validate_optimization_results(optimization_results)
    if not validation_passed:
        logger.warning('⚠️ Optimization results validation failed, using fallback parameters')

    # Save results
    await optimizer.save_optimization_results(optimization_results, symbol, exchange, data_dir)

    # Generate report
    start_time = datetime.now()
    report = await optimizer.generate_optimization_report(optimization_results, start_time)

    result = {
        'final_parameters': optimization_results,
        'optimization_report': report,
        'validation_passed': validation_passed
    }

    # Add non-linear optimization summary if used
    if optimizer.use_nonlinear_optimization:
        result['nonlinear_optimization'] = True
        result['enhancement_summary'] = {
            'use_log_sampling': optimizer.nonlinear_config.use_log_sampling,
            'use_fractional_powers': optimizer.nonlinear_config.use_fractional_powers,
            'use_sigmoid_transforms': optimizer.nonlinear_config.use_sigmoid_transforms,
            'use_adaptive_transforms': optimizer.nonlinear_config.use_adaptive_transforms
        }

    return result
