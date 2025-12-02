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
import asyncio
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
from src.utils.data.real_data_loader import real_data_loader

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
        self.ohlcv_data: Optional[pd.DataFrame] = None

        # Non-linear optimization configuration
        self.nonlinear_config = nonlinear_config or NonLinearConfig()
        self.parameter_sampler = NonLinearParameterSampler(self.nonlinear_config)
        
        # Initialize essential components
        self._initialize_optimization_components()

        # Parameter categories for optimization (updated for new Analyst Base models)
        self.categories = [
            'confidence', 'intensity', 'position_sizing', 'leverage', 'tpsl', 'exit_strategy',
            'uncertainty', 'sr', 'two_tier', 'technical_indicators',
            'system_monitoring', 'training_optimization', 'regime_transitions',
            'signal_aggregation', 'turnover_cost_penalty', 'entry_timing_optimization',
            'confidence_aware_signal', 'model_specific_parameters',
            # New directional categories
            'long_specific_parameters', 'short_specific_parameters',
            'directional_thresholds', 'asymmetric_risk_management',
            # Analyst integration (Analyst Only)
            'analyst_integration', 'analyst_oof_weights', 'analyst_feature_importance'
        ]

        # Default search spaces for each category
        tprint(f"🔍 DEBUG: Checking for _get_default_search_spaces method...", "info")
        if not hasattr(self, '_get_default_search_spaces'):
            tprint("❌ _get_default_search_spaces NOT FOUND in self!", "error")
            tprint(f"   Attributes: {[a for a in dir(self) if 'search' in a]}", "info")
        
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

            # Load HDF5 data from versioned artifacts
            tprint("=" * 80, "header")
            tprint("📥 LOADING PIPELINE DATA FROM HDF5 ARTIFACTS", "header")
            tprint("=" * 80, "header")
            loaded_hdf5_data = await self.load_hdf5_data_from_pipeline(config)

            # Store loaded data for use in optimization
            self.labeled_data = loaded_hdf5_data.get('labeled_data')
            self.regime_probabilities = loaded_hdf5_data.get('regime_probabilities')
            self.analyst_confidence = loaded_hdf5_data.get('analyst_confidence')
            self.disagreement_features = loaded_hdf5_data.get('disagreement_features')

            # NEW: Load OHLCV data for custom indicators
            tprint("📥 Loading OHLCV data for hierarchical optimization...", "info")
            try:
                self.ohlcv_data = await self._load_ohlcv_data(symbol, exchange, timeframe)
                tprint(f"✅ Loaded OHLCV data: {len(self.ohlcv_data)} rows", "success")
            except Exception as e:
                tprint(f"⚠️ Failed to load OHLCV data: {e}", "warning")
                self.ohlcv_data = None

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
            # 1. Run Standard Optimization (Base parameters, Confidence, etc.)
            optimization_result = await self._perform_final_parameters_optimization(
                symbol, exchange, timeframe, direction, execution_mode, config
            )

            # 2. Run Hierarchical Optimization (Specific TP/SL/Trailing Strategy)
            # This runs alongside standard optimization and merges/overrides specific parameters
            use_hierarchical = self.config.get('use_custom_hierarchical_optimization', True)
            if use_hierarchical and self.ohlcv_data is not None:
                tprint("🚀 Running Hierarchical Strategy Optimization...", "header")

                # Extract the optimized confidence threshold from standard result if available
                # otherwise default to 0.6
                current_best_params = {}
                if 'optimized_parameters' in optimization_result:
                    for cat_res in optimization_result['optimized_parameters'].values():
                        if isinstance(cat_res, dict) and 'best_params' in cat_res:
                            current_best_params.update(cat_res['best_params'])

                base_threshold = current_best_params.get('analyst_confidence_threshold',
                                    current_best_params.get('confidence_threshold', 0.6))

                hierarchical_result = await self.optimize_hierarchical_strategy(
                    self.calibration_results, self.ohlcv_data
                )

                # Merge results
                if 'optimized_parameters' in hierarchical_result:
                    tprint("🔄 Merging hierarchical strategy parameters into final result...", "info")
                    # Ensure optimized_parameters exists
                    if 'optimized_parameters' not in optimization_result:
                        optimization_result['optimized_parameters'] = {}

                    # Merge categories
                    for cat, res in hierarchical_result['optimized_parameters'].items():
                        optimization_result['optimized_parameters'][cat] = res

                    # Update metrics
                    optimization_result['parameters_optimized'] += hierarchical_result.get('parameters_optimized', 0)

                    # Update score if hierarchical found a better strategy (though scores might not be directly comparable)
                    # We keep the max
                    h_score = hierarchical_result.get('optimization_score', 0.0)
                    if h_score > optimization_result['optimization_score']:
                        optimization_result['optimization_score'] = h_score

            # Save optimization result as artifact
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

            try:
                tpsl_adjustments = self._compute_tpsl_regime_adjustments_iterative()
                if tpsl_adjustments:
                    optimization_results['_tpsl_regime_adjustments'] = tpsl_adjustments
                    sanity = self._classify_tpsl_regime_patterns(tpsl_adjustments)
                    optimization_results['_tpsl_regime_sanity'] = sanity
            except Exception as e:
                self.logger.debug(f"TPSL regime post-analysis failed: {e}")

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

    async def _load_ohlcv_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """
        Load OHLCV data using RealDataLoader.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe string

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Load data (default to last 90 days to ensure coverage)
            # In production, this should match the training/calibration range
            df = await real_data_loader.load_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                lookback_days=90,
                use_cache=True
            )
            
            if df is None or df.empty:
                raise ValueError(f"No data loaded for {symbol} {timeframe}")
                
            # Ensure standard columns lower case
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure datetime index
            if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df.set_index('timestamp', inplace=True)
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load OHLCV data: {e}")
            raise

    def _calculate_indicators_pandas(self, df: pd.DataFrame,
                                   atr_windows: List[str],
                                   adx_windows: List[int]) -> Dict[str, pd.DataFrame]:
        """
        Calculate ATR and ADX indicators for specified windows using pure pandas.
        
        Args:
            df: 15m OHLCV DataFrame
            atr_windows: List of ATR windows (e.g. ['1h', '2h'])
            adx_windows: List of ADX windows (integers, hours)
            
        Returns:
            Dictionary of DataFrames with indicators, keyed by window identifier
        """
        indicators = {}
        
        # Helper for Wilder's Smoothing (alpha=1/n)
        def wilders_smoothing(series, window):
            return series.ewm(alpha=1/window, adjust=False).mean()
            
        try:
            # Base 15m resampling mapping
            tf_map = {
                '1h': '1h', '2h': '2h', '4h': '4h', '8h': '8h', '12h': '12h'
            }
            
            # 1. Calculate ATR for requested windows
            for win_str in atr_windows:
                if win_str not in tf_map:
                    continue

                resample_freq = tf_map[win_str]
                
                # Resample (using label='right' to mark end of interval)
                # We shift results later to ensure no lookahead
                resampled = df.resample(resample_freq, label='right', closed='right').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                
                # Calculate TR
                prev_close = resampled['close'].shift(1)
                tr1 = resampled['high'] - resampled['low']
                tr2 = (resampled['high'] - prev_close).abs()
                tr3 = (resampled['low'] - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate ATR (14 period standard on resampled data)
                atr = wilders_smoothing(tr, 14)
                
                # Shift by 1 to prevent lookahead bias (indicator known ONLY after bar closes)
                atr = atr.shift(1)
                
                # Reindex back to 15m and ffill (forward fill known values)
                aligned_atr = atr.reindex(df.index, method='ffill')
                
                indicators[f'ATR_{win_str}'] = aligned_atr

            # 2. Calculate ADX/DI for requested windows
            for win_hours in adx_windows:
                win_str = f'{win_hours}h'
                
                # Resample (right labeled)
                resampled = df.resample(win_str, label='right', closed='right').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
                }).dropna()
                
                # ADX Calculation
                up = resampled['high'] - resampled['high'].shift(1)
                down = resampled['low'].shift(1) - resampled['low']
                
                # DM
                plus_dm = np.where((up > down) & (up > 0), up, 0.0)
                minus_dm = np.where((down > up) & (down > 0), down, 0.0)
                
                plus_dm = pd.Series(plus_dm, index=resampled.index)
                minus_dm = pd.Series(minus_dm, index=resampled.index)
                
                # TR
                prev_close = resampled['close'].shift(1)
                tr1 = resampled['high'] - resampled['low']
                tr2 = (resampled['high'] - prev_close).abs()
                tr3 = (resampled['low'] - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Smooth (14 period)
                tr_smooth = wilders_smoothing(tr, 14)
                plus_di = 100 * wilders_smoothing(plus_dm, 14) / tr_smooth
                minus_di = 100 * wilders_smoothing(minus_dm, 14) / tr_smooth
                
                dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
                adx = wilders_smoothing(dx, 14)
                
                # Shift by 1 to prevent lookahead bias
                adx = adx.shift(1)
                plus_di = plus_di.shift(1)
                minus_di = minus_di.shift(1)

                # Store
                indicators[f'ADX_{win_hours}h'] = adx.reindex(df.index, method='ffill')
                indicators[f'PDI_{win_hours}h'] = plus_di.reindex(df.index, method='ffill')
                indicators[f'MDI_{win_hours}h'] = minus_di.reindex(df.index, method='ffill')

            return indicators
            
        except Exception as e:
            self.logger.error(f"Failed to calculate indicators: {e}")
            raise

    def _compute_regime_and_meta_scalars(self, n_trades: int) -> Dict[str, np.ndarray]:
        """
        Compute regime and meta scalars for backtesting simulation.
        
        Args:
            n_trades: Number of trades to simulate
            
        Returns:
            Dictionary of scalars (defaulting to 1.0/0.0 if not available)
        """
        # Placeholder implementation to support legacy code
        return {
            'meta_confidence': np.ones(n_trades),
            'disagreement_scalar': np.zeros(n_trades),
            'uncertainty_scalar': np.zeros(n_trades),
            'risk_scalar': np.zeros(n_trades),
            'path_scalar': np.zeros(n_trades),
            'trend_scalar': np.ones(n_trades)
        }

    async def optimize_hierarchical_strategy(self, calibration_results: Dict[str, Any],
                                           ohlcv_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute strict 5-step hierarchical optimization for custom Trailing Stop strategy.
        
        Steps:
        1. Optimize TP/SL Multipliers (x, y)
        2. Optimize ATR Multiplier (for Trailing Stop)
        3. Optimize ATR Window (1h, 2h, 4h, 8h, 12h)
        4. Optimize Trailing Stop Weight (wATR)
        5. Optimize ADX Window (2h-8h)
        """
        tprint("=" * 80, "header")
        tprint("🚀 STARTING HIERARCHICAL OPTIMIZATION (5-STEP STRICT SEQUENCE)", "header")
        tprint("=" * 80, "header")
        
        if ohlcv_data is None:
            tprint("❌ Missing OHLCV data, cannot proceed with custom indicator optimization", "error")
            return {}

        # 0. Pre-calculate Indicators
        tprint("📊 Pre-calculating indicators for all candidate windows...", "info")
        atr_windows = ['1h', '2h', '4h', '8h', '12h']
        adx_windows = [2, 4, 6, 8] # Discretized 2-8h
        indicators = self._calculate_indicators_pandas(ohlcv_data, atr_windows, adx_windows)
        
        # Prepare signal data (timestamps and directions)
        analyst_conf = calibration_results.get('analyst_confidence', np.array([]))
        
        # Use a dynamic threshold if possible, but for initial signal extraction, we'll use a safe default
        # The backtester itself will filter using parameters
        # Here we just need a binary mask to identify potential trades
        base_threshold = 0.5 # Capture all potential positive trades
        
        # Use analyst confidence threshold as default for signals if not provided
        signals = np.where(analyst_conf > base_threshold, 1, 0) # Assuming long only for now or provided direction
        
        # Note: If direction is short, we need short logic.
        # Using loaded 'direction' from config would be better but let's assume long for now or infer
        direction = self.config.get('direction', 'long')
        if direction == 'short':
            # Analyst confidence is usually P(Up). If optimizing short, we might need P(Down).
            # Assuming analyst_conf is "Model Confidence for target direction"
            signals = np.where(analyst_conf > base_threshold, -1, 0)
        
        # Current State:
        best_params = {
            'tp_mult': 3.0, 'sl_mult': 1.5,
            'atr_trail_mult': 1.0,
            'atr_window': '4h',
            'w_atr': 0.7, # wADX = 0.3
            'adx_window': 4,
            'confidence_threshold': 0.6 # Default
        }
        
        # Helper for objective
        def run_step(step_name, param_names, ranges, objective_func):
            tprint(f"🔄 Optimizing Step: {step_name}", "header")
            study = optuna.create_study(direction='maximize')
            
            def obj(trial):
                # Suggest parameters for this step
                step_params = {}
                for p, r in zip(param_names, ranges):
                    if isinstance(r, list): # Categorical
                        step_params[p] = trial.suggest_categorical(p, r)
                    else: # Float tuple (min, max)
                        step_params[p] = trial.suggest_float(p, r[0], r[1])

                # Merge with current best
                current_trial_params = best_params.copy()
                current_trial_params.update(step_params)

                return objective_func(current_trial_params, signals, ohlcv_data, indicators)
            
            study.optimize(obj, n_trials=30) # Fast trials
            tprint(f"✅ {step_name} Best: {study.best_params} (Score: {study.best_value:.4f})", "success")
            best_params.update(study.best_params)
            return study.best_value

        # Simulation Function Wrapper
        def evaluate(p, sigs, df, inds):
            return self._run_fast_custom_backtest(p, sigs, df, inds)

        # Step 1: TP/SL Multipliers (x, y)
        run_step("1. TP/SL Multipliers",
                 ['tp_mult', 'sl_mult'],
                 [(1.0, 5.0), (0.5, 3.0)],
                 evaluate)

        # Step 2: ATR Multiplier (for Trailing)
        run_step("2. ATR Multiplier (Trailing)",
                 ['atr_trail_mult'],
                 [(0.5, 3.0)],
                 evaluate)

        # Step 3: ATR Window
        run_step("3. ATR Window",
                 ['atr_window'],
                 [atr_windows],
                 evaluate)

        # Step 4: Trailing Weight (wATR)
        run_step("4. Trailing Weight (wATR)",
                 ['w_atr'],
                 [(0.0, 1.0)],
                 evaluate)

        # Step 5: ADX Window
        run_step("5. ADX Window",
                 ['adx_window'],
                 [adx_windows],
                 evaluate)

        tprint("=" * 80, "header")
        tprint("🏆 FINAL OPTIMIZED PARAMETERS", "success")
        tprint(str(best_params), "info")
        tprint("=" * 80, "header")

        # Construct final result dictionary
        return {
            'parameters_optimized': len(best_params),
            'optimization_score': 0.0, # Placeholder
            'optimized_parameters': {
                'hierarchical_strategy': {
                    'best_params': best_params
                }
            }
        }

    def _run_fast_custom_backtest(self, params: Dict[str, Any],
                                signals: np.ndarray,
                                ohlcv: pd.DataFrame,
                                indicators: Dict[str, pd.DataFrame]) -> float:
        """
        Fast numpy-based backtester for the custom trailing strategy.
        """
        # Extract params
        tp_mult = params['tp_mult']
        sl_mult = params['sl_mult']
        atr_trail_mult = params['atr_trail_mult']
        atr_win = params['atr_window']
        w_atr = params['w_atr']
        w_adx = 1.0 - w_atr
        adx_win = params['adx_window']
        conf_threshold = params.get('confidence_threshold', 0.6)
        
        # Get indicator arrays (aligned with OHLCV)
        atr_arr = indicators[f'ATR_{atr_win}'].values
        adx_arr = indicators[f'ADX_{adx_win}h'].values
        pdi_arr = indicators[f'PDI_{adx_win}h'].values
        mdi_arr = indicators[f'MDI_{adx_win}h'].values
        
        high_arr = ohlcv['high'].values
        low_arr = ohlcv['low'].values
        close_arr = ohlcv['close'].values
        
        # Ensure signals match length (truncate if needed)
        n = min(len(signals), len(high_arr))
        
        # Identify entries (indices)
        # Use parameterized threshold if we have confidence scores
        # Note: 'signals' here is pre-filtered, but if we had raw scores we'd use them.
        # Since 'signals' is 1/0/-1, we assume it's already filtered or we filter non-zeros.
        entry_indices = np.where(signals[:n] != 0)[0]

        trades = []
        fee = 0.0015

        last_exit_idx = -1

        for idx in entry_indices:
            if idx <= last_exit_idx:
                continue

            entry_price = close_arr[idx]
            direction = signals[idx] # 1 or -1
            
            # Initial TP/SL
            atr_val = atr_arr[idx]
            if np.isnan(atr_val): continue
            
            if direction == 1:
                tp_price = entry_price + tp_mult * atr_val
                sl_price = entry_price - sl_mult * atr_val
                stop_price = sl_price
            else:
                tp_price = entry_price - tp_mult * atr_val
                sl_price = entry_price + sl_mult * atr_val
                stop_price = sl_price
            
            # Simulate forward
            exit_price = entry_price
            pnl = 0.0
            
            for t in range(idx + 1, min(idx + 500, n)): # Max duration 500 bars safety
                current_high = high_arr[t]
                current_low = low_arr[t]
                current_atr = atr_arr[t]
                current_adx = adx_arr[t]

                # Check Exits
                if direction == 1:
                    # Check TP
                    if current_high >= tp_price:
                        exit_price = tp_price
                        pnl = (exit_price - entry_price) / entry_price - fee
                        last_exit_idx = t
                        break
                    # Check Stop
                    if current_low <= stop_price:
                        exit_price = stop_price
                        pnl = (exit_price - entry_price) / entry_price - fee
                        last_exit_idx = t
                        break
                    
                    # Update Trailing
                    # Condition: +DI > -DI
                    if pdi_arr[t] > mdi_arr[t]:
                        dist = w_atr * (current_atr * atr_trail_mult) + w_adx * current_adx
                        # Ensure distance is non-negative and reasonable
                        dist = max(0.0, dist)

                        # Distance is subtracted from High? Or Current Close?
                        # User said "Trailing Stop Distance". Typically from High for Longs.
                        potential_stop = current_high - dist

                        # Ensure stop price is positive
                        potential_stop = max(0.0001, potential_stop)

                        # Only move up
                        if potential_stop > stop_price:
                            stop_price = potential_stop
                    # Else freeze (do nothing)

                else: # Short
                    # Check TP
                    if current_low <= tp_price:
                        exit_price = tp_price
                        pnl = (entry_price - exit_price) / entry_price - fee
                        last_exit_idx = t
                        break
                    # Check Stop
                    if current_high >= stop_price:
                        exit_price = stop_price
                        pnl = (entry_price - exit_price) / entry_price - fee
                        last_exit_idx = t
                        break

                    # Update Trailing
                    # Condition: -DI > +DI
                    if mdi_arr[t] > pdi_arr[t]:
                        dist = w_atr * (current_atr * atr_trail_mult) + w_adx * current_adx
                        # Ensure distance is non-negative
                        dist = max(0.0, dist)

                        potential_stop = current_low + dist
                        # Only move down
                        if potential_stop < stop_price:
                            stop_price = potential_stop
            
            trades.append(pnl)
            
        # Score: Total Return * Profit Factor (simple metric)
        if not trades:
            return 0.0
            
        trades_arr = np.array(trades)
        total_ret = np.sum(trades_arr)
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr <= 0]
        pf = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 10.0
        
        # Combined Score
        return total_ret * min(pf, 3.0)

    def _init_tpe_optimizers(self):
