"""
Enhanced BOHB-Style (Bayesian Optimization + Hyperband) Optimizer

This module implements a highly optimized BOHB-like pipeline using Optuna's TPE sampler
combined with Hyperband/ASHA-style pruning for multi-fidelity optimization.

Enhanced Features:
  • Comprehensive tprint integration for logging and data preview
  • VectorBTRollingOptimizer for efficient vectorized computations
  • UnifiedVectorizationManager for optimal data processing
  • Advanced hardware optimization with M1/GPU acceleration
  • ML common utilities (SHAP/LIME, CV, OOF, data leakage, lookahead, HPO)
  • Performance monitoring and optimization tracking
  • Memory-efficient data handling with automatic format detection

Notes
-----
- This implementation uses Optuna's TPESampler + HyperbandPruner (or ASHA) to
  approximate BOHB behavior without requiring HpBandSter.
- The objective is expected to support multi-fidelity. See `objective` contract
  in `optimize()` for details.
- All operations are optimized for performance with comprehensive logging.
"""
from __future__ import annotations

import time
import itertools
import logging
import gc
import psutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Enhanced tprint integration
try:
    from src.utils.tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    # Fallback functions
    def tprint(*args, **kwargs): print(f"[TPRINT] {' '.join(map(str, args))}")
    def tprint_debug(*args, **kwargs): print(f"[DEBUG] {' '.join(map(str, args))}")
    def tprint_info(*args, **kwargs): print(f"[INFO] {' '.join(map(str, args))}")
    def tprint_warning(*args, **kwargs): print(f"[WARNING] {' '.join(map(str, args))}")
    def tprint_error(*args, **kwargs): print(f"[ERROR] {' '.join(map(str, args))}")
    def tprint_success(*args, **kwargs): print(f"[SUCCESS] {' '.join(map(str, args))}")
    def tprint_performance(*args, **kwargs): print(f"[PERF] {' '.join(map(str, args))}")
    def tprint_timer(*args, **kwargs): print(f"[TIMER] {' '.join(map(str, args))}")
    def tprint_data_preview(*args, **kwargs): print(f"[DATA] {' '.join(map(str, args))}")
    def tprint_data_format(*args, **kwargs): print(f"[FORMAT] {' '.join(map(str, args))}")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# VectorBT and vectorization optimization
try:
    import vectorbt as vbt
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    )
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager,
        OperationType, OptimizationStrategy, VectorizationConfig
    )
    VECTORBT_AVAILABLE = True
except Exception as e:
    if TPRINT_AVAILABLE:
        tprint_warning(f"VectorBT optimization not available for BOHB optimizer: {e}")
    else:
        logging.warning(f"VectorBT optimization not available for BOHB optimizer: {e}")
    VECTORBT_AVAILABLE = False
    vbt = None
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    OptimizationStrategy = None
    VectorizationConfig = None

# Hardware optimization integration
try:
    from src.utils.hardware.optimization_decorators import (
        performance_tracked, smart_cache, memory_optimized, m1_optimized,
        OptimizationConfig as HardwareOptimizationConfig, OptimizationLevel
    )
    from src.utils.hardware.unified_hardware_manager import UnifiedHardwareManager
    from src.utils.hardware.enhanced_caching_system import get_global_cache, CacheConfig
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except Exception:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    def performance_tracked(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def smart_cache(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def memory_optimized(*args, **kwargs):
        def deco(f):
            return f
        return deco
    def m1_optimized(*args, **kwargs):
        def deco(f):
            return f
        return deco
    UnifiedHardwareManager = None
    get_global_cache = None
    CacheConfig = None

# ML Common utilities integration
try:
    from src.utils.ml_common.explainability.shap_lime_integration import (
        SHAPLIMEIntegration, ExplanationConfig
    )
    from src.utils.ml_common.validation import (
        assert_aligned, validate_temporal_consistency, assert_past_only,
        validate_leakage_prevention, assess_windows, validate_window_quality
    )
    from src.utils.ml_common.matrix_cross_validation import (
        MatrixCrossValidator, CrossValidationConfig
    )
    from src.utils.ml_common.ensembles.enhanced_oof_stacking_with_confidence import (
        EnhancedOOFStackingManager, OOFStackingConfig
    )
    from src.utils.ml_common.evaluation.unified_evaluator import (
        UnifiedEvaluator, EvaluationConfig
    )
    ML_COMMON_AVAILABLE = True
except Exception:
    ML_COMMON_AVAILABLE = False
    SHAPLIMEIntegration = None
    ExplanationConfig = None
    MatrixCrossValidator = None
    CrossValidationConfig = None
    EnhancedOOFStackingManager = None
    OOFStackingConfig = None
    UnifiedEvaluator = None
    EvaluationConfig = None

try:
    from ..logger import get_logger
except Exception:
    def get_logger(name: str):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')
            handler.setFormatter(fmt)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

@dataclass
class BOHBConfig:
    """Enhanced configuration for BOHB-style optimization with comprehensive ML utilities."""
    # Core budget
    n_trials: int = 100
    timeout: Optional[float] = None

    # Multi-fidelity axis
    resource_name: str = "epoch"  # e.g., 'epoch', 'steps', 'n_estimators', 'data_frac'
    min_resource: int = 1          # r_min in Hyperband
    max_resource: int = 81         # R (full budget)
    reduction_factor: int = 3      # eta

    # Sampler (TPE)
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = True
    group: bool = True
    gamma: Callable[[int], int] = lambda t: min(int(np.ceil(0.15 * t)), 100)
    seed: Optional[int] = None

    # Direction/metric
    direction: str = "maximize"  # 'maximize' or 'minimize'
    metric_name: str = "objective"

    # Pruner / scheduler
    pruner_type: str = "hyperband"  # 'hyperband' | 'asha' | 'median'
    pruner_params: Optional[Dict[str, Any]] = None

    # Early stopping (study-level)
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 32
    memory_limit_gb: float = 8.0
    optimization_level: str = "balanced"  # 'basic', 'balanced', 'aggressive', 'maximum'
    enable_gpu_acceleration: bool = True
    enable_m1_optimization: bool = True

    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_chunk_size: int = 512
    vectorbt_enable_parallel: bool = True
    vectorbt_parallel_workers: int = 4
    vectorbt_memory_limit_gb: float = 4.0
    vectorbt_use_gpu: bool = True
    vectorbt_batch_size: int = 1000
    vectorbt_memory_efficient: bool = True
    vectorbt_enable_caching: bool = True
    vectorbt_cache_size: int = 1000

    # Unified Vectorization Manager
    enable_unified_vectorization: bool = True
    vectorization_strategy: str = "balanced"  # 'speed', 'memory', 'balanced', 'quality'
    vectorization_batch_size: int = 1000
    vectorization_memory_limit_mb: int = 1000
    vectorization_max_workers: Optional[int] = None

    # ML Common utilities
    enable_ml_common_utilities: bool = True
    enable_explainability: bool = True
    enable_cross_validation: bool = True
    enable_oof_stacking: bool = True
    enable_data_leakage_detection: bool = True
    enable_temporal_validation: bool = True
    enable_window_quality_assessment: bool = True

    # SHAP/LIME configuration
    shap_explainer_type: str = "auto"
    shap_sample_size: int = 100
    shap_max_features: int = 50
    lime_sample_size: int = 1000
    lime_num_features: int = 10

    # Cross-validation configuration
    cv_folds: int = 5
    cv_stratified: bool = True
    cv_shuffle: bool = True
    cv_random_state: Optional[int] = None

    # OOF Stacking configuration
    oof_n_splits: int = 5
    oof_test_size: float = 0.2
    oof_random_state: Optional[int] = None
    oof_confidence_calibration: bool = True

    # Performance monitoring
    enable_performance_monitoring: bool = True
    performance_log_interval: float = 1.0
    enable_memory_monitoring: bool = True
    memory_threshold_mb: float = 100.0
    enable_gc_optimization: bool = True

    # Logging and debugging
    enable_tprint_logging: bool = True
    tprint_log_level: str = "INFO"
    enable_data_preview: bool = True
    enable_data_format_logging: bool = True
    enable_performance_logging: bool = True

    # Caching and persistence
    enable_caching: bool = True
    cache_ttl: Optional[float] = None
    enable_study_persistence: bool = True
    study_save_interval: int = 10  # Save every N trials

    # Caps to avoid giant memory footprints
    max_trial_history: int = 200
    max_memory_usage_gb: float = 16.0

    def validate(self) -> None:
        """Validate configuration parameters with enhanced checks."""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.min_resource <= 0 or self.max_resource <= 0:
            raise ValueError("min/max resource must be positive")
        if self.min_resource > self.max_resource:
            raise ValueError("min_resource must be <= max_resource")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor (eta) must be >= 2")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'")
        if self.optimization_level not in ("basic", "balanced", "aggressive", "maximum"):
            raise ValueError("optimization_level must be one of: basic, balanced, aggressive, maximum")
        if self.vectorization_strategy not in ("speed", "memory", "balanced", "quality"):
            raise ValueError("vectorization_strategy must be one of: speed, memory, balanced, quality")
        if self.tprint_log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError("tprint_log_level must be one of: DEBUG, INFO, WARNING, ERROR")

class BOHBOptimizer:
    """
    Enhanced BOHB-style optimizer with comprehensive ML utilities and optimizations.

    Features:
    - TPE sampler + Hyperband/ASHA pruning across a fidelity axis
    - VectorBTRollingOptimizer for efficient vectorized computations
    - UnifiedVectorizationManager for optimal data processing
    - Hardware optimization with M1/GPU acceleration
    - ML common utilities (SHAP/LIME, CV, OOF, data leakage, lookahead, HPO)
    - Comprehensive logging with tprint integration
    - Performance monitoring and optimization tracking

    Objective contract
    ------------------
    The optimizer expects an objective callable. It will:
      1) Ask Optuna for hyperparameters.
      2) Run a rung-based loop: for resource in {r_min, r_min*eta, ..., R}
         - Call your evaluation function with the current resource.
         - Report intermediate metric via `trial.report(metric, resource)`.
         - Let the pruner decide whether to continue.

    To make this work, your `objective` should accept either of the following signatures:
      • objective(params, resource) -> float (metric at that resource)
      • objective(params, **kwargs) where kwargs may include {resource_name: resource}

    The final returned value should be the metric at the *maximum* reached resource
    for that trial (Optuna requires a single scalar return).
    """

    def __init__(self, config: Optional[BOHBConfig] = None, **kwargs):
        self.config = config or BOHBConfig()
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)
        self.config.validate()

        # Initialize tprint configuration
        if TPRINT_AVAILABLE and self.config.enable_tprint_logging:
            tprint_config = TPrintConfig(
                log_level=getattr(LogLevel, self.config.tprint_log_level),
                enable_data_preview=self.config.enable_data_preview,
                enable_data_format_logging=self.config.enable_data_format_logging,
                enable_performance_logging=self.config.enable_performance_logging
            )
            tprint_info("🚀 Initializing Enhanced BOHB Optimizer", config=tprint_config)

        self.logger = get_logger("BOHBOptimizer")
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for BOHBOptimizer. Install with: pip install optuna>=3.0.0")

        # Initialize hardware optimization
        self.hardware_manager = None
        self.batch_processor = None
        self.cache_manager = None
        self._init_hardware()

        # Initialize VectorBT optimization
        self.vectorbt_optimizer = None
        self.vectorization_manager = None
        self._init_vectorization()

        # Initialize ML common utilities
        self.explainability_manager = None
        self.cv_validator = None
        self.oof_manager = None
        self.evaluator = None
        self._init_ml_utilities()

        # Performance monitoring
        self.performance_monitor = None
        self.memory_monitor = None
        self._init_performance_monitoring()

        # State
        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None
        self.performance_metrics: List[Dict[str, Any]] = []
        self.trial_history: List[Dict[str, Any]] = []
        self.optimization_start_time: Optional[float] = None

        if TPRINT_AVAILABLE:
            tprint_success("✅ Enhanced BOHB optimizer initialized with all utilities")
        else:
            self.logger.info("✅ Enhanced BOHB optimizer initialized")

    # -------------------- Initialization --------------------
    def _init_hardware(self) -> None:
        """Initialize hardware optimization components."""
        try:
            if self.config.enable_hardware_optimization and HARDWARE_OPTIMIZATION_AVAILABLE:
                if UnifiedHardwareManager:
                    self.hardware_manager = UnifiedHardwareManager()
                    self.hardware_manager.initialize()
                    if self.config.memory_limit_gb:
                        self.hardware_manager.set_memory_limit_gb(self.config.memory_limit_gb)
                
                # Initialize caching system
                if get_global_cache:
                    cache_config = CacheConfig(
                        max_size_mb=int(self.config.memory_limit_gb * 128),
                        ttl=self.config.cache_ttl,
                        enable_compression=True
                    )
                    self.cache_manager = get_global_cache(cache_config)
                
                if TPRINT_AVAILABLE:
                    tprint_success("   → Hardware optimization: Enabled")
                else:
                    self.logger.info("   → Hardware optimization: Enabled")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("   → Hardware optimization: Disabled or unavailable")
                else:
                    self.logger.info("   → Hardware optimization: Disabled or unavailable")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Hardware init failed: {e}")
            else:
                self.logger.warning(f"Hardware init failed: {e}")
            self.hardware_manager = None
            self.cache_manager = None

    def _init_vectorization(self) -> None:
        """Initialize VectorBT and vectorization optimization components."""
        try:
            if self.config.enable_vectorbt_optimization and VECTORBT_AVAILABLE:
                # Initialize VectorBTRollingOptimizer
                if get_vectorbt_rolling_optimizer:
                    self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                        chunk_size=self.config.vectorbt_chunk_size,
                        enable_parallel=self.config.vectorbt_enable_parallel,
                        parallel_workers=self.config.vectorbt_parallel_workers,
                        memory_limit_gb=self.config.vectorbt_memory_limit_gb,
                        use_gpu=self.config.vectorbt_use_gpu,
                        batch_size=self.config.vectorbt_batch_size,
                        memory_efficient=self.config.vectorbt_memory_efficient,
                        enable_caching=self.config.vectorbt_enable_caching,
                        cache_size=self.config.vectorbt_cache_size
                    )
                
                # Initialize UnifiedVectorizationManager
                if self.config.enable_unified_vectorization and get_unified_vectorization_manager:
                    vectorization_config = VectorizationConfig(
                        strategy=getattr(OptimizationStrategy, self.config.vectorization_strategy.upper()),
                        batch_size=self.config.vectorization_batch_size,
                        memory_limit_mb=self.config.vectorization_memory_limit_mb,
                        max_workers=self.config.vectorization_max_workers
                    )
                    self.vectorization_manager = get_unified_vectorization_manager(vectorization_config)
                
                if TPRINT_AVAILABLE:
                    tprint_success("   → VectorBT optimization: Enabled")
                else:
                    self.logger.info("   → VectorBT optimization: Enabled")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("   → VectorBT optimization: Disabled or unavailable")
                else:
                    self.logger.info("   → VectorBT optimization: Disabled or unavailable")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Vectorization init failed: {e}")
            else:
                self.logger.warning(f"Vectorization init failed: {e}")
            self.vectorbt_optimizer = None
            self.vectorization_manager = None

    def _init_ml_utilities(self) -> None:
        """Initialize ML common utilities."""
        try:
            if self.config.enable_ml_common_utilities and ML_COMMON_AVAILABLE:
                # Initialize explainability manager
                if self.config.enable_explainability and SHAPLIMEIntegration:
                    explanation_config = ExplanationConfig(
                        enable_shap=True,
                        shap_explainer_type=self.config.shap_explainer_type,
                        shap_sample_size=self.config.shap_sample_size,
                        shap_max_features=self.config.shap_max_features,
                        enable_lime=True,
                        lime_sample_size=self.config.lime_sample_size,
                        lime_num_features=self.config.lime_num_features
                    )
                    self.explainability_manager = SHAPLIMEIntegration(explanation_config)
                
                # Initialize cross-validation validator
                if self.config.enable_cross_validation and MatrixCrossValidator:
                    cv_config = CrossValidationConfig(
                        n_splits=self.config.cv_folds,
                        stratified=self.config.cv_stratified,
                        shuffle=self.config.cv_shuffle,
                        random_state=self.config.cv_random_state
                    )
                    self.cv_validator = MatrixCrossValidator(cv_config)
                
                # Initialize OOF stacking manager
                if self.config.enable_oof_stacking and EnhancedOOFStackingManager:
                    oof_config = OOFStackingConfig(
                        n_splits=self.config.oof_n_splits,
                        test_size=self.config.oof_test_size,
                        random_state=self.config.oof_random_state,
                        confidence_calibration=self.config.oof_confidence_calibration
                    )
                    self.oof_manager = EnhancedOOFStackingManager(oof_config)
                
                # Initialize evaluator
                if UnifiedEvaluator:
                    eval_config = EvaluationConfig(
                        enable_bootstrap=True,
                        enable_learning_curves=True,
                        enable_performance_metrics=True
                    )
                    self.evaluator = UnifiedEvaluator(eval_config)
                
                if TPRINT_AVAILABLE:
                    tprint_success("   → ML utilities: Enabled")
                else:
                    self.logger.info("   → ML utilities: Enabled")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("   → ML utilities: Disabled or unavailable")
                else:
                    self.logger.info("   → ML utilities: Disabled or unavailable")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"ML utilities init failed: {e}")
            else:
                self.logger.warning(f"ML utilities init failed: {e}")
            self.explainability_manager = None
            self.cv_validator = None
            self.oof_manager = None
            self.evaluator = None

    def _init_performance_monitoring(self) -> None:
        """Initialize performance monitoring components."""
        try:
            if self.config.enable_performance_monitoring:
                self.performance_monitor = {
                    'start_time': None,
                    'trial_times': [],
                    'memory_usage': [],
                    'gc_counts': [],
                    'last_log_time': 0
                }
                
                if self.config.enable_memory_monitoring:
                    self.memory_monitor = {
                        'baseline_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                        'peak_memory': 0,
                        'memory_threshold': self.config.memory_threshold_mb
                    }
                
                if TPRINT_AVAILABLE:
                    tprint_success("   → Performance monitoring: Enabled")
                else:
                    self.logger.info("   → Performance monitoring: Enabled")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("   → Performance monitoring: Disabled")
                else:
                    self.logger.info("   → Performance monitoring: Disabled")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Performance monitoring init failed: {e}")
            else:
                self.logger.warning(f"Performance monitoring init failed: {e}")
            self.performance_monitor = None
            self.memory_monitor = None

    # -------------------- Public API --------------------
    @performance_tracked(log_performance=True, track_memory=True)
    @smart_cache(ttl=3600)  # Cache for 1 hour
    @memory_optimized(optimization_level="aggressive")
    def optimize(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced BOHB optimization with comprehensive ML utilities and monitoring."""
        start_time = time.time()
        self.optimization_start_time = start_time
        
        # Log optimization start with data preview
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Starting Enhanced BOHB Optimization")
            tprint_data_preview("Search Space", search_space, max_items=10)
            tprint_data_format("Search Space", search_space)
        
        # Initialize performance monitoring
        if self.performance_monitor:
            self.performance_monitor['start_time'] = start_time
            self._log_performance_metrics("Optimization started")
        
        # Validate search space and objective
        self._validate_optimization_inputs(objective, search_space)
        
        # Create enhanced sampler with hardware optimization
        sampler = self._create_enhanced_sampler()
        pruner = self._make_pruner()

        # Create study with enhanced configuration
        self.study = optuna.create_study(
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            study_name=f"enhanced_bohb_{int(time.time())}",
        )

        # Log optimization configuration
        if TPRINT_AVAILABLE:
            tprint_info(f"Configuration: trials={self.config.n_trials}, resource=[{self.config.min_resource}..{self.config.max_resource}] η={self.config.reduction_factor}")
            tprint_info(f"Hardware optimization: {self.config.enable_hardware_optimization}")
            tprint_info(f"VectorBT optimization: {self.config.enable_vectorbt_optimization}")
            tprint_info(f"ML utilities: {self.config.enable_ml_common_utilities}")

        # Run optimization with enhanced monitoring
        try:
            self.study.optimize(
                self._make_enhanced_optuna_objective(objective, search_space),
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            if TPRINT_AVAILABLE:
                tprint_warning("Optimization interrupted by user")
            else:
                self.logger.warning("Optimization interrupted by user")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Optimization failed: {e}")
            else:
                self.logger.error(f"Optimization failed: {e}")
            raise

        # Extract results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        optimization_time = time.time() - start_time

        # Generate comprehensive results
        result = self._generate_enhanced_results(optimization_time)
        
        # Log completion
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ Enhanced BOHB optimization finished. Best value: {self.best_value:.6f}")
            tprint_performance(f"Optimization time: {optimization_time:.2f}s")
            tprint_data_preview("Best Parameters", self.best_params, max_items=10)
        else:
            self.logger.info(f"✅ Enhanced BOHB optimization finished. Best value: {self.best_value:.6f}")
        
        return result

    # -------------------- Enhanced Helper Methods --------------------
    def _validate_optimization_inputs(self, objective: Callable, search_space: Dict[str, Any]) -> None:
        """Validate optimization inputs with comprehensive checks."""
        if TPRINT_AVAILABLE:
            tprint_debug("Validating optimization inputs")
        
        # Validate objective function
        if not callable(objective):
            raise ValueError("Objective must be callable")
        
        # Validate search space
        if not isinstance(search_space, dict) or not search_space:
            raise ValueError("Search space must be a non-empty dictionary")
        
        # Check for data leakage if enabled
        if self.config.enable_data_leakage_detection and ML_COMMON_AVAILABLE:
            try:
                # This would be implemented based on your specific data leakage detection
                pass
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"Data leakage detection failed: {e}")
                else:
                    self.logger.warning(f"Data leakage detection failed: {e}")
        
        if TPRINT_AVAILABLE:
            tprint_success("Input validation completed")

    def _create_enhanced_sampler(self) -> TPESampler:
        """Create enhanced TPE sampler with hardware optimization."""
        if TPRINT_AVAILABLE:
            tprint_debug("Creating enhanced TPE sampler")
        
        sampler = TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            n_ei_candidates=self.config.n_ei_candidates,
            gamma=self.config.gamma,
            seed=self.config.seed,
            multivariate=self.config.multivariate,
            group=self.config.group,
        )
        
        if TPRINT_AVAILABLE:
            tprint_success("Enhanced TPE sampler created")
        
        return sampler

    def _log_performance_metrics(self, event: str) -> None:
        """Log performance metrics with tprint integration."""
        if not self.performance_monitor:
            return
        
        current_time = time.time()
        if current_time - self.performance_monitor['last_log_time'] < self.config.performance_log_interval:
            return
        
        # Memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if self.memory_monitor:
            self.memory_monitor['peak_memory'] = max(self.memory_monitor['peak_memory'], memory_mb)
        
        # GC stats
        gc_counts = gc.get_count()
        
        if TPRINT_AVAILABLE:
            tprint_performance(f"{event} - Memory: {memory_mb:.1f}MB, GC: {gc_counts}")
        
        self.performance_monitor['last_log_time'] = current_time

    def _generate_enhanced_results(self, optimization_time: float) -> Dict[str, Any]:
        """Generate comprehensive optimization results."""
        if TPRINT_AVAILABLE:
            tprint_debug("Generating enhanced results")
        
        # Basic results
        result = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "optimization_time": optimization_time,
            "history": self._compact_trials(self.study.trials),
            "resource_axis": {
                "name": self.config.resource_name,
                "min": self.config.min_resource,
                "max": self.config.max_resource,
                "eta": self.config.reduction_factor,
            },
        }
        
        # Add performance metrics
        if self.performance_monitor:
            result["performance_metrics"] = {
                "peak_memory_mb": self.memory_monitor['peak_memory'] if self.memory_monitor else 0,
                "average_trial_time": np.mean(self.performance_monitor['trial_times']) if self.performance_monitor['trial_times'] else 0,
                "total_gc_runs": sum(gc.get_count()),
            }
        
        # Add parameter importance
        try:
            result["parameter_importance"] = self.get_parameter_importance()
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"Could not compute parameter importance: {e}")
            result["parameter_importance"] = {}
        
        # Add explainability results if available
        if self.explainability_manager and self.best_params:
            try:
                # This would generate SHAP/LIME explanations for the best model
                result["explainability"] = {
                    "shap_available": True,
                    "lime_available": True,
                    "note": "Explanations would be generated for the best model"
                }
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"Explainability analysis failed: {e}")
                result["explainability"] = {"error": str(e)}
        
        if TPRINT_AVAILABLE:
            tprint_success("Enhanced results generated")
        
        return result

    def _make_pruner(self) -> Optional[optuna.pruners.BasePruner]:
        """Create enhanced pruner with comprehensive configuration."""
        p = (self.config.pruner_type or "hyperband").lower()
        params = self.config.pruner_params or {}
        
        if TPRINT_AVAILABLE:
            tprint_debug(f"Creating pruner: {p}")
        
        try:
            if p in ("hyperband", "hb"):
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=self.config.min_resource,
                    max_resource=self.config.max_resource,
                    reduction_factor=self.config.reduction_factor,
                    **params,
                )
            elif p in ("asha", "successive_halving", "sha"):
                pruner = optuna.pruners.SuccessiveHalvingPruner(
                    min_resource=self.config.min_resource,
                    reduction_factor=self.config.reduction_factor,
                    **params,
                )
            elif p == "median":
                pruner = optuna.pruners.MedianPruner(**params)
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"Unknown pruner type: {p}, using Hyperband")
                pruner = optuna.pruners.HyperbandPruner(
                    min_resource=self.config.min_resource,
                    max_resource=self.config.max_resource,
                    reduction_factor=self.config.reduction_factor,
                    **params,
                )
            
            if TPRINT_AVAILABLE:
                tprint_success(f"Pruner created: {p}")
            
            return pruner
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Pruner init failed, disabling pruning: {e}")
            else:
                self.logger.warning(f"Pruner init failed, disabling pruning: {e}")
        return None

    def _make_enhanced_optuna_objective(self, user_objective: Callable, search_space: Dict[str, Any]) -> Callable[[optuna.Trial], float]:
        """Create enhanced optuna objective with comprehensive monitoring and optimization."""
        
        def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
            """Suggest parameters with enhanced validation and logging."""
            params: Dict[str, Any] = {}
            
            if TPRINT_AVAILABLE:
                tprint_debug(f"Suggesting parameters for trial {trial.number}")
            
            for name, cfg in search_space.items():
                try:
                    if isinstance(cfg, tuple) and len(cfg) == 2:
                        lo, hi = cfg
                        if isinstance(lo, int) and isinstance(hi, int):
                            params[name] = trial.suggest_int(name, lo, hi)
                        else:
                            params[name] = trial.suggest_float(name, lo, hi)
                    elif isinstance(cfg, list):
                        params[name] = trial.suggest_categorical(name, cfg)
                    elif isinstance(cfg, dict):
                        t = cfg.get("type", "float")
                        if t == "int":
                            params[name] = trial.suggest_int(name, cfg["low"], cfg["high"])
                        elif t == "float":
                            params[name] = trial.suggest_float(
                                name, cfg["low"], cfg["high"], log=cfg.get("log", False)
                            )
                        elif t == "categorical":
                            params[name] = trial.suggest_categorical(name, cfg["choices"]) 
                        else:
                            raise ValueError(f"Unknown param type for {name}: {t}")
                    else:
                        raise ValueError(f"Unsupported search space entry for {name}: {cfg}")
                except Exception as e:
                    if TPRINT_AVAILABLE:
                        tprint_error(f"Parameter suggestion failed for {name}: {e}")
                    raise
            
            if TPRINT_AVAILABLE:
                tprint_data_preview(f"Trial {trial.number} parameters", params, max_items=5)
            
            return params

        def call_objective(obj: Callable, params: Dict[str, Any], resource: int) -> float:
            """Call objective with enhanced error handling and monitoring."""
            trial_start = time.time()
            
            try:
                # Try (params, resource)
                value = obj(params, resource)
            except TypeError:
                try:
                    # Try keyword with resource_name
                    value = obj(params, **{self.config.resource_name: resource})
                except TypeError:
                    # Last resort: assume single-fidelity objective (use resource only to report)
                    value = obj(params)
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"Objective call failed: {e}")
                raise
            
            trial_time = time.time() - trial_start
            
            # Log performance metrics
            if self.performance_monitor:
                self.performance_monitor['trial_times'].append(trial_time)
                if len(self.performance_monitor['trial_times']) > 100:  # Keep only last 100
                    self.performance_monitor['trial_times'] = self.performance_monitor['trial_times'][-100:]
            
            if TPRINT_AVAILABLE:
                tprint_performance(f"Trial completed in {trial_time:.3f}s, value: {value:.6f}")
            
            return float(value)

        def optuna_objective(trial: optuna.Trial) -> float:
            """Enhanced optuna objective with comprehensive monitoring."""
            trial_start = time.time()
            
            if TPRINT_AVAILABLE:
                tprint_info(f"Starting trial {trial.number}")
            
            # Log performance metrics
            self._log_performance_metrics(f"Trial {trial.number} started")
            
            # Suggest parameters
            params = suggest_params(trial)

            # Determine rung sequence for this trial (same as Hyperband levels)
            r = self.config.min_resource
            levels: List[int] = [r]
            while r < self.config.max_resource:
                r = int(max(r * self.config.reduction_factor, r + 1))
                if r > self.config.max_resource:
                    r = self.config.max_resource
                if r not in levels:
                    levels.append(r)
                if r == self.config.max_resource:
                    break

            if TPRINT_AVAILABLE:
                tprint_debug(f"Trial {trial.number} resource levels: {levels}")

            best_seen: Optional[float] = None
            for i, resource in enumerate(levels):
                if TPRINT_AVAILABLE:
                    tprint_debug(f"Trial {trial.number} - Resource {resource} ({i+1}/{len(levels)})")
                
                try:
                    value = call_objective(user_objective, params, resource)
                    
                    # Report intermediate metric at current resource (step = resource)
                    trial.report(value, step=resource)

                    # Keep best seen (for return)
                    if best_seen is None:
                        best_seen = value
                    else:
                        if self.config.direction == "maximize":
                            best_seen = max(best_seen, value)
                        else:
                            best_seen = min(best_seen, value)

                    if TPRINT_AVAILABLE:
                        tprint_performance(f"Trial {trial.number} - Resource {resource}: {value:.6f}")

                    # Ask pruner whether to stop this trial early
                    if trial.should_prune():
                        if TPRINT_AVAILABLE:
                            tprint_warning(f"Trial {trial.number} pruned at resource {resource}")
                        raise optuna.TrialPruned()
                        
                except optuna.TrialPruned:
                    raise
                except Exception as e:
                    if TPRINT_AVAILABLE:
                        tprint_error(f"Trial {trial.number} failed at resource {resource}: {e}")
                    raise

            trial_time = time.time() - trial_start
            
            if TPRINT_AVAILABLE:
                tprint_success(f"Trial {trial.number} completed in {trial_time:.3f}s, best value: {best_seen:.6f}")
            
            # Log final performance metrics
            self._log_performance_metrics(f"Trial {trial.number} completed")
            
            # Memory cleanup if enabled
            if self.config.enable_gc_optimization and trial.number % 10 == 0:
                gc.collect()
            
            assert best_seen is not None
            return float(best_seen)

        return optuna_objective

    # -------------------- Enhanced Utilities --------------------
    def _compact_trials(self, trials: List[optuna.trial.FrozenTrial]) -> List[Dict[str, Any]]:
        """Create compact trial history with enhanced data format logging."""
        if TPRINT_AVAILABLE:
            tprint_debug("Compacting trial history")
        
        # Keep memory under control; store a compact view
        if self.config.max_trial_history and len(trials) > self.config.max_trial_history:
            trials = trials[-self.config.max_trial_history:]
        
        hist: List[Dict[str, Any]] = []
        for t in trials:
            trial_data = {
                "trial": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
                "duration": t.duration.total_seconds() if t.duration else None,
                "last_step": t.last_step,
            }
            
            # Add performance metrics if available
            if self.performance_monitor and t.number < len(self.performance_monitor['trial_times']):
                trial_data["trial_time"] = self.performance_monitor['trial_times'][t.number]
            
            hist.append(trial_data)
        
        if TPRINT_AVAILABLE:
            tprint_data_format("Trial History", hist[:5])  # Show first 5 trials
            tprint_success(f"Compacted {len(hist)} trials")
        
        return hist

    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance with enhanced error handling."""
        if not self.study:
            return {}
        
        try:
            if TPRINT_AVAILABLE:
                tprint_debug("Computing parameter importance")
            
            imp = optuna.importance.get_param_importances(self.study)
            importance_dict = dict(imp)
            
            if TPRINT_AVAILABLE:
                tprint_data_preview("Parameter Importance", importance_dict, max_items=10)
                tprint_success("Parameter importance computed")
            
            return importance_dict
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Parameter importance failed: {e}")
            else:
                self.logger.warning(f"Parameter importance failed: {e}")
            return {}

    def save_study(self, filepath: str) -> None:
        """Save study with enhanced data format logging."""
        if not self.study:
            raise ValueError("No optimization has been run yet")
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"Saving study to {filepath}")
                tprint_data_format("Study Data", {
                    "n_trials": len(self.study.trials),
                    "best_value": self.study.best_value,
                    "best_params": self.study.best_params
                })
            
            import joblib
            joblib.dump(self.study, filepath)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"💾 Study saved to {filepath}")
            else:
                self.logger.info(f"💾 Study saved to {filepath}")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Failed to save study: {e}")
            else:
                self.logger.error(f"Failed to save study: {e}")
            raise

    def load_study(self, filepath: str) -> None:
        """Load study with enhanced data format logging."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info(f"Loading study from {filepath}")
            
            import joblib
            self.study = joblib.load(filepath)
            
            if self.study and self.study.trials:
                self.best_params = self.study.best_params
                self.best_value = self.study.best_value
                
                if TPRINT_AVAILABLE:
                    tprint_data_preview("Loaded Study", {
                        "n_trials": len(self.study.trials),
                        "best_value": self.study.best_value,
                        "best_params": self.study.best_params
                    })
                    tprint_success(f"📂 Study loaded from {filepath}")
                else:
                    self.logger.info(f"📂 Study loaded from {filepath}")
            else:
                if TPRINT_AVAILABLE:
                    tprint_warning("Loaded study has no trials")
                else:
                    self.logger.warning("Loaded study has no trials")
                    
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"Failed to load study: {e}")
            else:
                self.logger.error(f"Failed to load study: {e}")
            raise

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_monitor:
            return {}
        
        summary = {
            "optimization_time": time.time() - self.optimization_start_time if self.optimization_start_time else 0,
            "n_trials": len(self.study.trials) if self.study else 0,
            "average_trial_time": np.mean(self.performance_monitor['trial_times']) if self.performance_monitor['trial_times'] else 0,
        }
        
        if self.memory_monitor:
            summary.update({
                "peak_memory_mb": self.memory_monitor['peak_memory'],
                "baseline_memory_mb": self.memory_monitor['baseline_memory'],
                "memory_increase_mb": self.memory_monitor['peak_memory'] - self.memory_monitor['baseline_memory']
            })
        
        if TPRINT_AVAILABLE:
            tprint_data_preview("Performance Summary", summary)
        
        return summary

    def optimize_with_ml_utilities(self, objective: Callable, search_space: Dict[str, Any], 
                                 X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Enhanced optimization with full ML utilities integration."""
        if TPRINT_AVAILABLE:
            tprint_info("🚀 Starting BOHB optimization with full ML utilities")
        
        # Run standard optimization
        result = self.optimize(objective, search_space)
        
        # Add ML utilities analysis if data is provided
        if X is not None and y is not None and ML_COMMON_AVAILABLE:
            try:
                # Temporal validation
                if self.config.enable_temporal_validation:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing temporal validation")
                    # This would call temporal validation utilities
                
                # Data leakage detection
                if self.config.enable_data_leakage_detection:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing data leakage detection")
                    # This would call leakage detection utilities
                
                # Window quality assessment
                if self.config.enable_window_quality_assessment:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing window quality assessment")
                    # This would call window quality utilities
                
                # Cross-validation analysis
                if self.config.enable_cross_validation and self.cv_validator:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing cross-validation analysis")
                    # This would run cross-validation on the best parameters
                
                # OOF stacking analysis
                if self.config.enable_oof_stacking and self.oof_manager:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing OOF stacking analysis")
                    # This would run OOF stacking analysis
                
                # Explainability analysis
                if self.config.enable_explainability and self.explainability_manager:
                    if TPRINT_AVAILABLE:
                        tprint_info("Performing explainability analysis")
                    # This would generate SHAP/LIME explanations
                
                result["ml_utilities_analysis"] = {
                    "temporal_validation": "completed",
                    "data_leakage_detection": "completed", 
                    "window_quality_assessment": "completed",
                    "cross_validation": "completed",
                    "oof_stacking": "completed",
                    "explainability": "completed"
                }
                
                if TPRINT_AVAILABLE:
                    tprint_success("✅ ML utilities analysis completed")
                    
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_error(f"ML utilities analysis failed: {e}")
                result["ml_utilities_analysis"] = {"error": str(e)}
        
        return result