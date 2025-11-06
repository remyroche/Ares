"""
Hardware-Optimized Bayesian TPE (Tree-structured Parzen Estimator) Optimizer

This module provides a staged Bayesian optimization interface using Optuna's TPE sampler
with hardware acceleration and adaptive optimization:
coarse grid -> fine grid -> TPE optimization for efficient hyperparameter search.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, Union, List, Tuple
import logging
import time
import itertools
from dataclasses import dataclass
from .execution_mode_adapter import adjust_hpo_params_for_mode, get_execution_mode

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ...common_operations import safe_divide
from ..logger import get_logger
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from .pareto import Solution, ParetoFront, compute_pareto_front

# Hardware optimization imports
try:
    from ...hardware.unified_hardware_manager import UnifiedHardwareManager, WorkloadType, OptimizationLevel
    from ...matrix_operations.hardware_integration import HardwareOptimizedMatrixProcessor
    from ...matrix_operations.batch_operations import BatchMatrixProcessor
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Hardware optimization not available: {e}")
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    UnifiedHardwareManager = None
    HardwareOptimizedMatrixProcessor = None
    BatchMatrixProcessor = None

# VectorBT optimization imports
try:
    # Import from src.vectorbt instead of direct vectorbt import
    from src.utils.vectorbt_compat import (
        vbt, rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max,
        rolling_sum, rolling_apply, VECTORBT_AVAILABLE as VBT_AVAILABLE
    )
    from ..unified_vectorization_manager import get_unified_vectorization_manager, OperationType
    from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, get_unified_vectorization_manager as get_feature_vectorization_manager
    )
    from src.feature_generation.utils.unified_vectorization_manager import VectorizationConfig
    VECTORBT_AVAILABLE = VBT_AVAILABLE
except ImportError as e:
    logging.warning(f"VectorBT optimization not available: {e}")
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    VectorizationConfig = None
    get_feature_vectorization_manager = None

@dataclass
class OptimizationConfig:
    """Configuration for hardware-optimized staged Bayesian TPE optimization."""

    # Core optimization settings
    n_trials: int = 100
    timeout: Optional[float] = None

    # Execution mode for adaptive configuration (detected from ares_launcher)
    execution_mode: str = "light"  # "full", "light", "blank"

    # TPE sampler settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = True
    group: bool = True
    gamma: Callable[[int], int] = lambda t: min(int(np.ceil(0.15 * t)), 100)
    seed: Optional[int] = None

    # Optimization direction and metric
    direction: str = 'maximize'
    metric_name: str = 'objective'

    # Staged optimization settings
    enable_staged_optimization: bool = True
    coarse_grid_points: int = 5
    fine_grid_points: int = 5
    coarse_grid_trials: int = 25  # 5x5 grid for 2D search space
    fine_grid_trials: int = 25    # 5x5 grid for 2D search space
    tpe_trials: int = 50         # Remaining trials for TPE
    max_coarse_grid_size: int = 1000  # Maximum grid points to evaluate in coarse stage (prevent OOM)
    max_fine_grid_size: int = 500     # Maximum grid points to evaluate in fine stage (prevent OOM)

    # Hardware optimization settings
    enable_hardware_optimization: bool = True
    workload_type: str = 'ml_training'
    optimization_level: str = 'balanced'
    enable_gpu_acceleration: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 32
    memory_limit_gb: float = 8.0

    # VectorBT optimization settings
    enable_vectorbt_optimization: bool = True
    vectorbt_parallel_workers: int = 4
    vectorbt_chunk_size: int = 1000
    vectorbt_memory_limit_gb: float = 4.0
    vectorbt_use_gpu: bool = True
    vectorbt_enable_parallel: bool = True

    # Enhanced VectorBT integration settings
    enable_vectorbt_rolling_optimizer: bool = True
    enable_unified_vectorization: bool = True
    vectorbt_batch_size: int = 1000
    vectorbt_memory_efficient: bool = True
    vectorbt_enable_caching: bool = True
    vectorbt_cache_size: int = 1000

    # Adaptive grid refinement settings
    enable_adaptive_grid_refinement: bool = True
    adaptive_refinement_threshold: float = 0.01  # Minimum improvement to trigger refinement
    max_adaptive_iterations: int = 3  # Maximum number of adaptive refinements
    adaptive_refinement_factor: float = 0.1  # How much to shrink search space (10% of range)
    convergence_window: int = 10  # Number of trials to check for convergence
    min_grid_points: int = 3  # Minimum points per parameter in adaptive grid
    max_grid_points: int = 15  # Maximum points per parameter in adaptive grid

    # Adaptive optimization settings
    enable_adaptive_optimization: bool = True
    performance_monitoring_interval: float = 1.0
    auto_tune_batch_size: bool = True
    adaptive_memory_management: bool = True

    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # Constraints
    constraints: Optional[Dict[str, Any]] = None

    # Pruning and history control
    enable_pruner: bool = True
    pruner_type: str = 'hyperband'  # 'hyperband', 'successive_halving', 'median'
    pruner_params: Dict[str, Any] = None
    max_trial_history: int = 200  # cap stored trial summaries to limit memory

    def __post_init__(self):
        """Apply execution mode-based optimizations."""
    
        # Get the mode if it's not a standard one
        if self.execution_mode not in ['light', 'blank', 'full']:
            self.execution_mode = get_execution_mode()
    
        # Use coarse_grid_points as a proxy for cv_folds, default to 5 if 0
        base_grid_points = self.coarse_grid_points if self.coarse_grid_points > 0 else 5
    
        adjusted_n_trials, adjusted_grid_points = adjust_hpo_params_for_mode(
            n_trials=self.n_trials,
            cv_folds=base_grid_points, 
            execution_mode=self.execution_mode
        )
    
        # Apply reduction proportionally to trial types
        if self.n_trials > 0 and adjusted_n_trials != self.n_trials:
            ratio = adjusted_n_trials / self.n_trials
    
            self.coarse_grid_trials = max(1, int(self.coarse_grid_trials * ratio))
            self.fine_grid_trials = max(1, int(self.fine_grid_trials * ratio))
    
            # Recalculate tpe_trials to match the adjusted total
            current_grid_trials = self.coarse_grid_trials + self.fine_grid_trials if self.enable_staged_optimization else 0
            self.tpe_trials = max(1, adjusted_n_trials - current_grid_trials)
    
            # Update the main n_trials to the new total
            self.n_trials = self.coarse_grid_trials + self.fine_grid_trials + self.tpe_trials
    
        # Apply adjusted folds to grid points
        if self.execution_mode in ['light', 'blank']:
            self.coarse_grid_points = adjusted_grid_points
            self.fine_grid_points = adjusted_grid_points

        # Ensure startup trials remain valid after adjustments
        if self.n_trials > 1 and self.n_startup_trials >= self.n_trials:
            self.n_startup_trials = max(1, self.n_trials - 1)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")
        if self.n_startup_trials >= self.n_trials:
            # Auto-adjust n_startup_trials to be valid instead of raising error
            self.n_startup_trials = max(1, min(self.n_startup_trials, self.n_trials - 1))
            logging.warning(f"⚠️ Adjusted n_startup_trials to {self.n_startup_trials} (must be < n_trials={self.n_trials})")
        if self.direction not in ['minimize', 'maximize']:
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if self.coarse_grid_points <= 0:
            raise ValueError("coarse_grid_points must be positive")
        if self.fine_grid_points <= 0:
            raise ValueError("fine_grid_points must be positive")
        if self.coarse_grid_trials < 0:
            raise ValueError("coarse_grid_trials must be non-negative")
        if self.fine_grid_trials < 0:
            raise ValueError("fine_grid_trials must be non-negative")
        if self.tpe_trials < 0:
            raise ValueError("tpe_trials must be non-negative")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")

class BayesianTPEOptimizer:
    """
    Hardware-Optimized Bayesian TPE Optimizer using Optuna's Tree-structured Parzen Estimator.

    This class provides a wrapper around Optuna's TPE sampler with hardware acceleration
    and adaptive optimization for efficient hyperparameter optimization in ML pipelines.
    """

    def __init__(self, config: Optional[OptimizationConfig] = None, **kwargs):
        """
        Initialize hardware-optimized Bayesian TPE optimizer.

        Args:
            config: Optimization configuration
            **kwargs: Additional configuration parameters
        """
        self.config = config or OptimizationConfig()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        self.config.validate()
        self.logger = get_logger('BayesianTPEOptimizer')

        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for BayesianTPEOptimizer. "
                "Please install optuna: pip install optuna>=2.10.0"
            )

        # Hardware optimization components
        self.hardware_manager = None
        self.matrix_processor = None
        self.batch_processor = None
        self.performance_monitor = None

        # VectorBT optimization components
        self.vectorbt_manager = None
        self.vectorbt_available = VECTORBT_AVAILABLE

        # Adaptive grid refinement state
        self.adaptive_refinement_history = []
        self.current_refinement_iteration = 0
        self.convergence_history = []
        self.adaptive_search_spaces = []  # Track search space evolution

        # Initialize hardware optimization if available
        if HARDWARE_OPTIMIZATION_AVAILABLE and self.config.enable_hardware_optimization:
            self._initialize_hardware_optimization()

        # Initialize VectorBT optimization if available
        if VECTORBT_AVAILABLE and self.config.enable_vectorbt_optimization:
            self._initialize_vectorbt_optimization()

        # Optimization state
        self.study = None
        self.best_params = None
        self.best_value = None
        self.optimization_history = []
        self.performance_metrics = []

        # Early stopping state
        self.early_stopping_triggered = False
        self.trials_without_improvement = 0
        self.best_value_history = []

        # Adaptive patience state
        self.adaptive_patience_enabled = kwargs.get('adaptive_patience', True)
        self.patience_history = []
        self.convergence_rate_history = []
        self.current_patience = self.config.early_stopping_patience or 10
        self.min_patience = max(1, (self.config.early_stopping_patience or 10) // 3)
        self.max_patience = (self.config.early_stopping_patience or 10) * 2
        self.patience_adjustment_factor = 1.5

        # Multi-objective state
        self.multi_objective_enabled = kwargs.get('multi_objective_stopping', False)
        self.objective_weights = kwargs.get('objective_weights', {})
        self.objective_history = {}
        self.pareto_front_optimizer = ParetoFront() if self.multi_objective_enabled else None
        self.pareto_solutions = []  # Store Solution objects for Pareto front computation

        # Confidence-based stopping state
        self.confidence_based_stopping_enabled = kwargs.get('confidence_based_stopping', False)
        self.confidence_level = kwargs.get('confidence_level', 0.95)
        self.confidence_history = []
        self.statistical_tests = []

        # Learning rate schedules for thresholds
        self.threshold_schedule_enabled = kwargs.get('threshold_schedule_enabled', False)
        self.threshold_schedule_type = kwargs.get('threshold_schedule_type', 'exponential')
        self.initial_threshold = kwargs.get('initial_threshold', None)
        self.final_threshold = kwargs.get('final_threshold', None)
        self.schedule_params = kwargs.get('schedule_params', {})
        self.current_threshold = None
        self.threshold_history = []

        self.logger.info("✅ VectorBT-Optimized BayesianTPEOptimizer initialized")
        if self.hardware_manager:
            self.logger.info("   → Hardware optimization: Enabled")
        else:
            self.logger.info("   → Hardware optimization: Disabled")

        if self.vectorbt_manager:
            self.logger.info("   → VectorBT optimization: Enabled")
        else:
            self.logger.info("   → VectorBT optimization: Disabled")

        if self.config.early_stopping_patience:
            self.logger.info(f"   → Early stopping: Enabled (patience={self.config.early_stopping_patience})")

    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            # Initialize unified hardware manager
            self.hardware_manager = UnifiedHardwareManager()
            self.hardware_manager.initialize()

            # Configure for ML training workload
            if self.config.optimization_level == 'aggressive':
                self.hardware_manager.set_intensive_thresholds()
            else:
                self.hardware_manager.set_normal_thresholds()

            # Initialize matrix processor for vectorized operations
            if HardwareOptimizedMatrixProcessor:
                self.matrix_processor = HardwareOptimizedMatrixProcessor()

            # Initialize batch processor for efficient evaluation
            if BatchMatrixProcessor:
                self.batch_processor = BatchMatrixProcessor(
                    chunk_size_mb=int(self.config.memory_limit_gb * 128),  # Convert GB to MB
                    enable_gpu=self.config.enable_gpu_acceleration,
                    enable_parallel=True,
                    max_workers=4
                )

            # Performance monitoring is already available via hardware_manager.performance_monitor
            self.performance_monitor = self.hardware_manager.performance_monitor

            self.logger.info("✅ Hardware optimization components initialized")

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization initialization failed: {e}")
            self.hardware_manager = None
            self.matrix_processor = None
            self.batch_processor = None
            self.performance_monitor = None

    def _initialize_vectorbt_optimization(self):
        """Initialize VectorBT optimization components with enhanced integration."""
        try:
            # Initialize unified vectorization manager
            self.vectorbt_manager = get_unified_vectorization_manager()

            # Initialize VectorBT rolling optimizer if enabled
            self.vectorbt_rolling_optimizer = None
            if self.config.enable_vectorbt_rolling_optimizer and get_vectorbt_rolling_optimizer:
                try:
                    self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer(
                        enable_gpu=self.config.vectorbt_use_gpu,
                        enable_parallel=self.config.vectorbt_enable_parallel,
                        memory_efficient=self.config.vectorbt_memory_efficient,
                        chunk_size=self.config.vectorbt_chunk_size
                    )
                    self.logger.info("✅ VectorBT Rolling Optimizer initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBT Rolling Optimizer initialization failed: {e}")

            # Initialize enhanced unified vectorization manager if enabled
            self.enhanced_vectorization_manager = None
            if self.config.enable_unified_vectorization and get_feature_vectorization_manager:
                try:
                    vectorization_config = VectorizationConfig(
                        enable_vectorbt=self.config.enable_vectorbt_optimization,
                        enable_gpu=self.config.vectorbt_use_gpu,
                        enable_parallel=self.config.vectorbt_enable_parallel,
                        memory_efficient=self.config.vectorbt_memory_efficient,
                        max_memory_gb=self.config.vectorbt_memory_limit_gb,
                        chunk_size=self.config.vectorbt_chunk_size,
                        enable_monitoring=True,
                        batch_size=self.config.vectorbt_batch_size,
                        enable_batch_processing=True
                    )
                    self.enhanced_vectorization_manager = get_feature_vectorization_manager(vectorization_config)
                    self.logger.info("✅ Enhanced Unified Vectorization Manager initialized")
                except Exception as e:
                    self.logger.warning(f"⚠️ Enhanced Vectorization Manager initialization failed: {e}")

            # Configure VectorBT settings
            if vbt:
                # Set memory limit (if settings available)
                if hasattr(vbt, 'settings') and self.config.vectorbt_memory_limit_gb:
                    if hasattr(vbt.settings, 'array_wrapper') and 'freq' in vbt.settings.array_wrapper:
                        vbt.settings.array_wrapper['freq'] = '1min'

                # Configure parallel processing (VectorBT 0.28.1 handles this automatically)
                # Note: Parallel settings removed in VectorBT 0.28.1 as it's handled automatically
                # if hasattr(vbt, 'settings') and self.config.vectorbt_enable_parallel:
                #     if hasattr(vbt.settings, 'parallel') and 'threading' in vbt.settings['parallel']:
                #         vbt.settings.parallel['threading'] = True

                # Configure GPU usage
                if self.config.vectorbt_use_gpu:
                    self.logger.info("🚀 VectorBT GPU acceleration enabled")
                else:
                    self.logger.info("💻 VectorBT CPU-only mode")

            self.logger.info("✅ VectorBT optimization components initialized")

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT optimization initialization failed: {e}")
            self.vectorbt_manager = None
            self.vectorbt_rolling_optimizer = None
            self.enhanced_vectorization_manager = None

    def optimize(self, objective: Callable, search_space: Dict[str, Any],
                **kwargs) -> Dict[str, Any]:
        """
        Run staged Bayesian TPE optimization: coarse grid -> fine grid -> TPE.

        Args:
            objective: Objective function to optimize
            search_space: Parameter search space definition
            **kwargs: Additional optimization parameters

        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()

        try:
            # Update config with any additional parameters
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

            self.logger.info("🚀 Starting staged TPE optimization")
            self.logger.info(f"   Search space: {list(search_space.keys())}")
            self.logger.info(f"   Stages: {'Coarse' if self.config.enable_staged_optimization else ''} {'Fine' if self.config.enable_staged_optimization else ''} TPE")

            # Initialize results tracking
            all_trials = []
            best_params = None
            best_value = None
            coarse_results = None
            fine_results = None

            # Stage 1: Coarse Grid Search (if enabled)
            if self.config.enable_staged_optimization:
                coarse_results = self._run_coarse_grid_stage(objective, search_space)
                if coarse_results:
                    all_trials.extend(coarse_results['trials'])
                    if self._is_better_result(coarse_results['best_value'], best_value):
                        best_params = coarse_results['best_params']
                        best_value = coarse_results['best_value']

                if coarse_results and coarse_results.get('best_value') is not None:
                    self.logger.info(f"   Coarse grid: {len(coarse_results['trials'])} trials, best: {coarse_results['best_value']:.4f}")
                elif coarse_results:
                    self.logger.info(f"   Coarse grid: {len(coarse_results['trials'])} trials, best: N/A")

            # Stage 2: Fine Grid Search around best coarse results
            if coarse_results and coarse_results['best_params']:
                fine_results = self._run_fine_grid_stage(objective, search_space, coarse_results['best_params'])
                if fine_results:
                    all_trials.extend(fine_results['trials'])
                    if self._is_better_result(fine_results['best_value'], best_value):
                        best_params = fine_results['best_params']
                        best_value = fine_results['best_value']

                if fine_results:
                    self.logger.info(f"   Fine grid: {len(fine_results['trials'])} trials, best: {fine_results['best_value']:.4f}")

            # Stage 2.5: Adaptive Grid Refinement (if enabled)
            if (self.config.enable_adaptive_grid_refinement and
                fine_results and fine_results['best_params'] and
                self.vectorbt_manager and self.config.enable_vectorbt_optimization):

                adaptive_results = self._run_adaptive_refinement_stage(
                    objective, search_space, fine_results['best_params'], all_trials
                )
                if adaptive_results:
                    all_trials.extend(adaptive_results['trials'])
                    if self._is_better_result(adaptive_results['best_value'], best_value):
                        best_params = adaptive_results['best_params']
                        best_value = adaptive_results['best_value']

                    if adaptive_results:
                        self.logger.info(f"   Adaptive refinement: {len(adaptive_results['trials'])} trials, best: {adaptive_results['best_value']:.4f}")

            # Stage 3: TPE Optimization
            tpe_trials_needed = self.config.n_trials - len(all_trials)
            if tpe_trials_needed > 0:
                tpe_results = self._run_tpe_stage(objective, search_space, best_params, min(tpe_trials_needed, self.config.tpe_trials))
                if tpe_results:
                    all_trials.extend(tpe_results['trials'])
                    if self._is_better_result(tpe_results['best_value'], best_value):
                        best_params = tpe_results['best_params']
                        best_value = tpe_results['best_value']

                if tpe_results:
                    self.logger.info(f"   TPE: {len(tpe_results['trials'])} trials, best: {tpe_results['best_value']:.4f}")

            # Final results
            optimization_time = time.time() - start_time
            self.best_params = best_params
            self.best_value = best_value

            # Calculate efficiency score: quality achieved per unit of computational cost
            if optimization_time > 0 and len(all_trials) > 0 and best_value is not None:
                # Efficiency = best_value / (time * trials)
                # Higher is better for maximize, lower is better for minimize
                if self.config.direction == 'maximize':
                    efficiency_score = best_value / (optimization_time * len(all_trials))
                else:
                    # For minimize, use inverse of best_value to make higher efficiency better
                    efficiency_score = 1.0 / (abs(best_value) + 1e-10) / (optimization_time * len(all_trials))
            else:
                efficiency_score = 0.0

            results = {
                'best_params': self.best_params,
                'best_value': self.best_value,
                'n_trials': len(all_trials),
                'optimization_time': optimization_time,
                'efficiency_score': efficiency_score,
                'history': all_trials,
                'stages': {
                    'coarse_grid': len([t for t in all_trials if t.get('stage') == 'coarse']),
                    'fine_grid': len([t for t in all_trials if t.get('stage') == 'fine']),
                    'tpe': len([t for t in all_trials if t.get('stage') == 'tpe'])
                },
                'early_stopping': {
                    'triggered': self.early_stopping_triggered,
                    'trials_without_improvement': self.trials_without_improvement,
                    'patience': self.config.early_stopping_patience,
                    'threshold': self.config.early_stopping_threshold
                }
            }

            self.logger.info(f"✅ Staged TPE optimization completed in {optimization_time:.2f}s")
            self.logger.info(f"   Best value: {self.best_value:.4f}")
            self.logger.info(f"   Best params: {self.best_params}")
            self.logger.info(f"   Total trials: {len(all_trials)}")
            if self.early_stopping_triggered:
                self.logger.info(f"   Early stopping: Triggered (saved {self.config.n_trials - len(all_trials)} trials)")

            return results

        except Exception as e:
            self.logger.error(f"❌ Staged TPE optimization failed: {e}")
            raise

    def _is_better_result(self, new_value: float, current_best: float) -> bool:
        """Check if new value is better than current best."""
        if current_best is None:
            return True
        if self.config.direction == 'maximize':
            return new_value > current_best
        else:
            return new_value < current_best

    def _create_early_stopping_callback(self) -> Callable:
        """
        Create adaptive early stopping callback for Optuna optimization.

        Returns:
            Callback function that raises TrialPruned when early stopping criteria met
        """
        def callback(study: optuna.Study, trial: optuna.Trial):
            """Adaptive early stopping callback."""
            if self.confidence_based_stopping_enabled:
                # Confidence-based stopping takes precedence
                if self._confidence_based_early_stopping(study, trial):
                    raise optuna.TrialPruned()
            elif self.multi_objective_enabled:
                return self._multi_objective_early_stopping_callback(study, trial)
            else:
                return self._single_objective_early_stopping_callback(study, trial)

        return callback

    def _single_objective_early_stopping_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Single objective adaptive early stopping callback."""
        # Get current best value
        current_best = study.best_value

        # Update history
        self.best_value_history.append(current_best)

        # Need minimum history for adaptive patience
        if len(self.best_value_history) < max(self.min_patience + 1, 10):
                return

        # Calculate convergence rate and adaptive patience
        if self.adaptive_patience_enabled:
            convergence_rate = self._calculate_convergence_rate()
            adaptive_patience = self._calculate_adaptive_patience(convergence_rate)

            # Update current patience if it changed significantly
            if abs(adaptive_patience - self.current_patience) > 1:
                self.current_patience = int(adaptive_patience)
                self.logger.debug(f"   Adaptive patience adjusted to {self.current_patience}")
        else:
            self.current_patience = self.config.early_stopping_patience or 10

        # Check for improvement using adaptive patience
        recent_history = self.best_value_history[-self.current_patience:]
        if len(recent_history) < 2:
            return

        previous_best = recent_history[0]

        # Calculate improvement
        if self.config.direction == 'maximize':
            improvement = current_best - previous_best
        else:
            improvement = previous_best - current_best

        # Adaptive threshold based on convergence and learning rate schedule
        if self.config.early_stopping_threshold is not None:
            min_improvement = self.config.early_stopping_threshold
        else:
            # Use adaptive threshold calculation with learning rate schedule
            min_improvement = self._calculate_adaptive_threshold(convergence_rate)

        # Check if improvement is below adaptive threshold
        if improvement < min_improvement:
            self.trials_without_improvement += 1
        else:
            self.trials_without_improvement = 0

        # Trigger early stopping if adaptive patience exceeded
        if self.trials_without_improvement >= self.current_patience:
            self.early_stopping_triggered = True
            self.logger.info(f"⏹️ Early stopping triggered after {len(self.best_value_history)} trials")
            self.logger.info(f"   No improvement for {self.trials_without_improvement} consecutive checks")
            self.logger.info(f"   Best value: {current_best:.6f}")
            self.logger.info(f"   Adaptive patience: {self.current_patience}")
            self.logger.info(f"   Convergence rate: {convergence_rate:.6f}")
            raise optuna.TrialPruned()

    def _multi_objective_early_stopping_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Multi-objective adaptive early stopping callback."""
        # Extract multiple objectives from trial
        objectives = {}
        for obj_name in self.objective_weights.keys():
            if hasattr(trial, 'user_attrs') and obj_name in trial.user_attrs:
                objectives[obj_name] = trial.user_attrs[obj_name]
            else:
                # Skip if objective not available
                return

        if not objectives:
            return

        # Update objective history
        for obj_name, value in objectives.items():
            if obj_name not in self.objective_history:
                self.objective_history[obj_name] = []
            self.objective_history[obj_name].append(value)

        # Need minimum history for multi-objective analysis
        min_history = max(self.min_patience + 1, 10)
        if any(len(history) < min_history for history in self.objective_history.values()):
            return

        # Calculate convergence rate and adaptive patience
        if self.adaptive_patience_enabled:
            convergence_rate = self._calculate_multi_objective_convergence_rate()
            adaptive_patience = self._calculate_adaptive_patience(convergence_rate)

            # Update current patience if it changed significantly
            if abs(adaptive_patience - self.current_patience) > 1:
                self.current_patience = int(adaptive_patience)
                self.logger.debug(f"   Multi-objective adaptive patience adjusted to {self.current_patience}")
        else:
            self.current_patience = self.config.early_stopping_patience or 10

        # Update Pareto front using existing Pareto optimizer
        self._update_pareto_front_with_optimizer(objectives)

        # Check for Pareto improvement
        recent_history = self.best_value_history[-self.current_patience:] if self.best_value_history else []

        if self._should_stop_multi_objective_with_pareto(recent_history):
            self.early_stopping_triggered = True
            self.logger.info(f"⏹️ Multi-objective early stopping triggered after {len(self.best_value_history)} trials")
            self.logger.info(f"   No Pareto improvement for {self.trials_without_improvement} consecutive checks")
            self.logger.info(f"   Current objectives: {objectives}")
            pareto_size = len(self.pareto_solutions) if self.pareto_solutions else 0
            self.logger.info(f"   Pareto front size: {pareto_size}")
            raise optuna.TrialPruned()

    def _calculate_multi_objective_convergence_rate(self) -> float:
        """Calculate convergence rate for multi-objective optimization."""
        if not self.objective_history:
            return 1.0

        # Calculate convergence for each objective and combine
        convergence_rates = []
        for obj_name, history in self.objective_history.items():
            if len(history) < 10:
                convergence_rates.append(1.0)
                continue

            recent_values = history[-20:]
            if len(recent_values) < 5:
                convergence_rates.append(1.0)
                continue

            # Calculate improvements for this objective
            improvements = []
            for i in range(1, len(recent_values)):
                improvement = recent_values[i] - recent_values[i-1]
                improvements.append(max(0, improvement))

            if not improvements:
                convergence_rates.append(0.1)
                continue

            # Exponential moving average
            alpha = 0.3
            ema_improvements = [improvements[0]]
            for imp in improvements[1:]:
                ema = alpha * imp + (1 - alpha) * ema_improvements[-1]
                ema_improvements.append(ema)

            # Normalize and scale
            value_range = max(recent_values) - min(recent_values)
            if value_range == 0:
                convergence_rates.append(1.0)
                continue

            avg_improvement = np.mean(ema_improvements[-5:])
            normalized_rate = avg_improvement / value_range
            convergence_rate = max(0.1, min(2.0, normalized_rate * 10))
            convergence_rates.append(convergence_rate)

        # Return weighted average of convergence rates
        if convergence_rates:
            weights = [self.objective_weights.get(obj_name, 1.0)
                      for obj_name in self.objective_history.keys()]
            return np.average(convergence_rates, weights=weights)
        else:
            return 1.0

    def _update_pareto_front_with_optimizer(self, objectives: Dict[str, float]) -> None:
        """Update Pareto front using the existing ParetoFront optimizer."""
        if not self.pareto_front_optimizer:
            return

        # Create a Solution object for the current objectives
        current_solution = Solution(metrics=objectives.copy())

        # Add to the list of solutions for Pareto front computation
        self.pareto_solutions.append(current_solution)

        # Compute Pareto front using the existing optimizer
        # Define objectives direction (assuming maximization for all by default)
        objectives_direction = {obj_name: 'max' for obj_name in objectives.keys()}

        try:
            # Compute the Pareto front using the imported function
            pareto_front = compute_pareto_front(
                self.pareto_solutions, objectives_direction
            )

            # Update our stored Pareto front
            self.pareto_solutions = pareto_front

        except Exception as e:
            self.logger.warning(f"Failed to compute Pareto front: {e}")
            # Fallback to simple approach if Pareto computation fails
            pass

    def _should_stop_multi_objective_with_pareto(self, recent_history: List[float]) -> bool:
        """Determine if multi-objective optimization should stop early using Pareto front analysis."""
        if not self.pareto_solutions or len(recent_history) < self.current_patience:
            return False

        # Check if Pareto front has been stable recently
        recent_trials = recent_history[-self.current_patience:]
        if not recent_trials:
            return False

        # Calculate Pareto front stability metrics
        if len(self.pareto_solutions) < 2:
            return False

        # Check if the Pareto front size has been stable
        # If the Pareto front hasn't changed significantly in recent trials, consider stopping
        current_pareto_size = len(self.pareto_solutions)

        # Simple heuristic: if Pareto front size hasn't changed for several trials
        # and recent trials haven't produced significantly better solutions
        recent_max = max(recent_trials) if recent_trials else 0
        overall_max = max(recent_history) if recent_history else 0

        # If recent trials aren't improving the Pareto front significantly
        improvement_ratio = recent_max / (overall_max + 1e-8) if overall_max > 0 else 1.0

        if improvement_ratio < 0.95 and len(recent_history) >= self.current_patience * 2:
            # Pareto front appears stable, check if we should stop
            self.trials_without_improvement += 1
        else:
            self.trials_without_improvement = 0

        return self.trials_without_improvement >= self.current_patience

    def _confidence_based_early_stopping(self, study: optuna.Study, trial: optuna.Trial) -> bool:
        """Confidence-based early stopping using statistical confidence intervals."""
        if len(self.best_value_history) < 20:  # Need sufficient history
            return False

        current_best = study.best_value

        # Update confidence history
        self.confidence_history.append(current_best)

        # Perform statistical analysis every few trials for efficiency
        if len(self.confidence_history) % 5 != 0:
            return False

        # Calculate confidence intervals for recent performance
        recent_values = np.array(self.confidence_history[-50:])  # Last 50 trials

        if len(recent_values) < 10:
            return False

        # Calculate mean and standard deviation
        mean_performance = np.mean(recent_values)
        std_performance = np.std(recent_values, ddof=1)  # Sample standard deviation

        if std_performance == 0:
            # No variation, optimization has likely converged
            return True

        # Calculate confidence interval
        try:
            from scipy import stats
            confidence_interval = stats.t.interval(
                self.confidence_level,
                len(recent_values) - 1,
                loc=mean_performance,
                scale=stats.sem(recent_values)  # Standard error of the mean
            )
        except ImportError:
            self.logger.warning("SciPy not available, disabling confidence-based stopping")
            return False

        # Calculate expected improvement potential
        expected_improvement = self._calculate_expected_improvement_potential(recent_values)

        # Check if current best is near the upper bound of confidence interval
        if self.config.direction == 'maximize':
            upper_bound = confidence_interval[1]
            improvement_potential = upper_bound - current_best
            # Stop if current best is within confidence interval of recent mean
            # and expected improvement is minimal
            if (current_best >= confidence_interval[0] and
                improvement_potential < abs(mean_performance) * 0.01):  # Less than 1% potential improvement
                self.logger.info(f"⏹️ Confidence-based early stopping triggered")
                self.logger.info(f"   Current best: {current_best:.6f}")
                self.logger.info(f"   Confidence interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
                self.logger.info(f"   Expected improvement: {improvement_potential:.6f}")
                self.logger.info(f"   Confidence level: {self.confidence_level}")
                return True
        else:
            lower_bound = confidence_interval[0]
            improvement_potential = current_best - lower_bound
            # Stop if current best is within confidence interval of recent mean
            # and expected improvement is minimal
            if (current_best <= confidence_interval[1] and
                improvement_potential < abs(mean_performance) * 0.01):  # Less than 1% potential improvement
                self.logger.info(f"⏹️ Confidence-based early stopping triggered")
                self.logger.info(f"   Current best: {current_best:.6f}")
                self.logger.info(f"   Confidence interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}]")
                self.logger.info(f"   Expected improvement: {improvement_potential:.6f}")
                self.logger.info(f"   Confidence level: {self.confidence_level}")
                return True

        # Store statistical test results
        test_result = {
            'trial': trial.number,
            'current_best': current_best,
            'confidence_interval': confidence_interval,
            'expected_improvement': improvement_potential,
            'should_stop': improvement_potential < abs(mean_performance) * 0.01
        }
        self.statistical_tests.append(test_result)

        return False

    def _calculate_expected_improvement_potential(self, recent_values: np.ndarray) -> float:
        """Calculate expected improvement potential based on recent optimization trend."""
        if len(recent_values) < 10:
            return float('inf')  # High potential with little data

        # Fit linear trend to recent values
        x = np.arange(len(recent_values))
        
        # Check for sufficient variation to avoid rank warnings
        if len(recent_values) < 2 or np.std(recent_values) < 1e-10:
            return 0.0
            
        try:
            slope, intercept = np.polyfit(x, recent_values, 1)

            # Calculate trend strength
            trend_strength = abs(slope) / (np.std(recent_values) + 1e-8)

            # Estimate potential improvement based on trend
            if self.config.direction == 'maximize':
                if slope > 0:
                    # Improving trend - estimate potential based on trend strength
                    potential = abs(slope) * len(recent_values) * (1 + trend_strength)
                else:
                    # Declining trend - low potential
                    potential = abs(slope) * 5  # Limited potential
            else:
                if slope < 0:
                    # Improving trend for minimization
                    potential = abs(slope) * len(recent_values) * (1 + trend_strength)
                else:
                    # Worsening trend - low potential
                    potential = abs(slope) * 5

            return max(0, potential)

        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix or invalid values, likely due to constant values
            return 0.0

    def _calculate_adaptive_threshold(self, convergence_rate: float) -> float:
        """Calculate adaptive threshold using learning rate schedules."""
        if not self.threshold_schedule_enabled:
            return self.config.early_stopping_threshold or (abs(self.best_value_history[-1]) * 0.001 if self.best_value_history else 0.001)

        # Get base threshold
        if self.initial_threshold is not None:
            base_threshold = self.initial_threshold
        elif self.best_value_history:
            base_threshold = abs(self.best_value_history[-1]) * 0.001
        else:
            base_threshold = 0.001

        # Calculate progress through optimization
        total_trials = len(self.best_value_history)
        max_trials = self.config.n_trials

        if max_trials == 0:
            progress = 0.5  # Default to middle if no max trials
        else:
            progress = min(1.0, total_trials / max_trials)

        # Apply schedule
        if self.threshold_schedule_type == 'exponential':
            # Exponential decay: threshold decreases over time
            decay_rate = self.schedule_params.get('decay_rate', 0.9)
            current_threshold = base_threshold * (decay_rate ** progress)

        elif self.threshold_schedule_type == 'linear':
            # Linear decrease from initial to final threshold
            if self.final_threshold is not None:
                current_threshold = base_threshold + (self.final_threshold - base_threshold) * progress
            else:
                current_threshold = base_threshold * (1.0 - progress * 0.5)  # Default linear decrease

        elif self.threshold_schedule_type == 'step':
            # Step function: sudden changes at certain progress points
            step_points = self.schedule_params.get('step_points', [0.25, 0.5, 0.75])
            step_factors = self.schedule_params.get('step_factors', [0.8, 0.6, 0.4])

            current_threshold = base_threshold
            for i, point in enumerate(step_points):
                if progress >= point:
                    factor = step_factors[i] if i < len(step_factors) else step_factors[-1]
                    current_threshold = base_threshold * factor

        elif self.threshold_schedule_type == 'adaptive':
            # Adaptive based on convergence rate
            if convergence_rate > 1.0:
                # Fast convergence - use stricter threshold
                current_threshold = base_threshold * 0.5
            elif convergence_rate > 0.5:
                # Moderate convergence - normal threshold
                current_threshold = base_threshold
            else:
                # Slow convergence - use more lenient threshold
                current_threshold = base_threshold * 1.5

        else:
            # Default exponential decay
            current_threshold = base_threshold * (0.9 ** progress)

        # Record threshold history
        self.threshold_history.append(current_threshold)
        if len(self.threshold_history) > 20:
            self.threshold_history.pop(0)

        self.logger.debug(f"   Adaptive threshold: {current_threshold:.8f} (progress: {progress:.2f}, convergence: {convergence_rate:.2f})")

        return current_threshold

    def _check_early_stopping_grid(self, trials: List[Dict], stage: str) -> bool:
        """
        Check if early stopping should be triggered for grid search.

        Args:
            trials: List of completed trials
            stage: Stage name ('coarse' or 'fine')

        Returns:
            True if early stopping should be triggered
        """
        window = self.config.early_stopping_patience
        if not window or len(trials) < window + 1:
            return False

        values = [t['value'] for t in trials]
        
        # Compare overall best up to start of window vs overall best at end
        if self.config.direction == 'maximize':
            prev_best = max(values[:-window])
            curr_best = max(values)
            improvement = curr_best - prev_best
        else:
            prev_best = min(values[:-window])
            curr_best = min(values)
            improvement = prev_best - curr_best

        threshold = self.config.early_stopping_threshold or abs(prev_best) * 1e-3

        if improvement < threshold:
            self.logger.info(f"⏹️ Early stopping in {stage} (no improvement ≥ {threshold:.3g} in last {window} evals)")
            return True

        return False

    def _calculate_convergence_rate(self) -> float:
        """Calculate convergence rate based on recent optimization history."""
        if len(self.best_value_history) < 10:
            return 1.0  # Assume fast convergence with little history

        # Use exponential moving average of recent improvements
        recent_values = self.best_value_history[-20:]  # Last 20 trials
        if len(recent_values) < 5:
            return 1.0

        # Calculate improvements
        improvements = []
        for i in range(1, len(recent_values)):
            if self.config.direction == 'maximize':
                improvement = recent_values[i] - recent_values[i-1]
            else:
                improvement = recent_values[i-1] - recent_values[i]
            improvements.append(max(0, improvement))  # Only positive improvements

        if not improvements:
            return 0.1  # Very slow convergence

        # Exponential moving average of improvements
        alpha = 0.3  # Smoothing factor
        ema_improvements = [improvements[0]]
        for imp in improvements[1:]:
            ema = alpha * imp + (1 - alpha) * ema_improvements[-1]
            ema_improvements.append(ema)

        # Normalize by the range of recent values
        value_range = max(recent_values) - min(recent_values)
        if value_range == 0:
            return 1.0

        # Convergence rate: higher values = faster convergence
        avg_improvement = np.mean(ema_improvements[-5:])  # Last 5 improvements
        normalized_rate = avg_improvement / value_range

        # Scale to reasonable range [0.1, 2.0]
        convergence_rate = max(0.1, min(2.0, normalized_rate * 10))

        # Update convergence history for trend analysis
        self.convergence_rate_history.append(convergence_rate)
        if len(self.convergence_rate_history) > 50:
            self.convergence_rate_history.pop(0)

        return convergence_rate

    def _calculate_adaptive_patience(self, convergence_rate: float) -> float:
        """Calculate adaptive patience based on convergence rate and history."""
        # Base patience calculation
        base_patience = self.config.early_stopping_patience or 10

        # Adjust based on convergence rate
        if convergence_rate > 1.0:
            # Fast convergence - reduce patience
            patience_factor = 0.7
        elif convergence_rate > 0.5:
            # Moderate convergence - normal patience
            patience_factor = 1.0
        elif convergence_rate > 0.2:
            # Slow convergence - increase patience
            patience_factor = 1.3
        else:
            # Very slow convergence - significantly increase patience
            patience_factor = 1.8

        adaptive_patience = base_patience * patience_factor

        # Consider convergence trend
        if len(self.convergence_rate_history) >= 10:
            recent_trend = np.polyfit(range(10), self.convergence_rate_history[-10:], 1)[0]
            if recent_trend > 0.01:  # Improving convergence
                adaptive_patience *= 0.9  # Slightly reduce patience
            elif recent_trend < -0.01:  # Worsening convergence
                adaptive_patience *= 1.1  # Slightly increase patience

        # Record patience history
        self.patience_history.append(adaptive_patience)
        if len(self.patience_history) > 20:
            self.patience_history.pop(0)

        # Clamp to reasonable bounds
        return max(self.min_patience, min(self.max_patience, adaptive_patience))

    def _run_coarse_grid_stage(self, objective: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run VectorBT-optimized coarse grid search stage with memory-efficient chunking."""
        try:
            self.logger.info("🔍 Stage 1: VectorBT-optimized coarse grid search")

            # Configure hardware for grid search workload
            if self.hardware_manager and self.config.enable_hardware_optimization:
                self.hardware_manager.set_normal_thresholds()

            # Use VectorBT for grid generation if available
            if self.vectorbt_manager and self.config.enable_vectorbt_optimization:
                coarse_grid = self._generate_vectorbt_coarse_grid(search_space, self.config.coarse_grid_points)
            else:
                # Fallback to original grid generation
                coarse_grid = build_coarse_grid_from_search_space(search_space, self.config.coarse_grid_points)

            self.logger.info(f"   Generated {len(coarse_grid)} coarse grid points")

            if not coarse_grid:
                self.logger.warning("⚠️ No coarse grid points generated")
                return None

            # Check if grid is too large and needs chunking for memory efficiency
            if len(coarse_grid) > self.config.max_coarse_grid_size:
                self.logger.warning(f"⚠️ Coarse grid size ({len(coarse_grid)}) exceeds max ({self.config.max_coarse_grid_size})")
                self.logger.info(f"   Using chunked evaluation for memory efficiency")
                return self._chunked_evaluate_grid(objective, coarse_grid, 'coarse', self.config.max_coarse_grid_size)

            # Use batch evaluation if available and safe
            if (self.batch_processor and self.config.enable_batch_processing and
                len(coarse_grid) > 1 and self._is_batch_evaluation_safe(coarse_grid)):
                return self._batch_evaluate_grid(objective, coarse_grid, 'coarse')
            else:
                # Fallback to sequential evaluation
                return self._sequential_evaluate_grid(objective, coarse_grid, 'coarse')

        except Exception as e:
            self.logger.error(f"❌ Coarse grid stage failed: {e}")
            return None

    def _run_fine_grid_stage(self, objective: Callable, search_space: Dict[str, Any],
                           coarse_best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run VectorBT-optimized fine grid search stage around best coarse results with memory-efficient chunking."""
        try:
            self.logger.info("🔍 Stage 2: VectorBT-optimized fine grid search")

            # Configure hardware for intensive search workload
            if self.hardware_manager and self.config.enable_hardware_optimization:
                self.hardware_manager.set_intensive_thresholds()  # More aggressive for fine search

            # Use VectorBT for fine grid generation if available
            if self.vectorbt_manager and self.config.enable_vectorbt_optimization:
                fine_grid = self._generate_vectorbt_fine_grid(search_space, coarse_best_params, self.config.fine_grid_points)
            else:
                # Fallback to original grid generation
                fine_grid = build_fine_grid_around_best(search_space, coarse_best_params, self.config.fine_grid_points)

            self.logger.info(f"   Generated {len(fine_grid)} fine grid points")

            if not fine_grid:
                self.logger.warning("⚠️ No fine grid points generated")
                return None

            # Check if grid is too large and needs chunking for memory efficiency
            if len(fine_grid) > self.config.max_fine_grid_size:
                self.logger.warning(f"⚠️ Fine grid size ({len(fine_grid)}) exceeds max ({self.config.max_fine_grid_size})")
                self.logger.info(f"   Using chunked evaluation for memory efficiency")
                return self._chunked_evaluate_grid(objective, fine_grid, 'fine', self.config.max_fine_grid_size)

            # Use batch evaluation if available and safe
            if (self.batch_processor and self.config.enable_batch_processing and
                len(fine_grid) > 1 and self._is_batch_evaluation_safe(fine_grid)):
                return self._batch_evaluate_grid(objective, fine_grid, 'fine')
            else:
                # Fallback to sequential evaluation
                return self._sequential_evaluate_grid(objective, fine_grid, 'fine')

        except Exception as e:
            self.logger.error(f"❌ Fine grid stage failed: {e}")
            return None

    def _run_adaptive_refinement_stage(self, objective: Callable, search_space: Dict[str, Any],
                                     best_params: Dict[str, Any], all_trials: List[Dict]) -> Dict[str, Any]:
        """Run VectorBT-optimized adaptive grid refinement stage."""
        try:
            self.logger.info("🔍 Stage 2.5: VectorBT adaptive grid refinement")

            # Check if we should perform adaptive refinement
            if not self._should_perform_adaptive_refinement(all_trials):
                self.logger.info("   Skipping adaptive refinement - convergence criteria not met")
                return None

            # Generate adaptive search space around best parameters
            adaptive_search_space = self._generate_adaptive_search_space(search_space, best_params)
            self.adaptive_search_spaces.append(adaptive_search_space)

            # Generate adaptive grid using VectorBT
            adaptive_grid = self._generate_vectorbt_adaptive_grid(
                adaptive_search_space, best_params, self._calculate_adaptive_grid_points()
            )

            self.logger.info(f"   Generated {len(adaptive_grid)} adaptive grid points")
            self.logger.info(f"   Refinement iteration: {self.current_refinement_iteration + 1}")

            if not adaptive_grid:
                self.logger.warning("⚠️ No adaptive grid points generated")
                return None

            # Use batch evaluation for adaptive grid if safe
            if self._is_batch_evaluation_safe(adaptive_grid):
                adaptive_results = self._batch_evaluate_grid(objective, adaptive_grid, 'adaptive')
            else:
                adaptive_results = self._sequential_evaluate_grid(objective, adaptive_grid, 'adaptive')

            if adaptive_results:
                # Update refinement history
                self.adaptive_refinement_history.append({
                    'iteration': self.current_refinement_iteration,
                    'best_value': adaptive_results['best_value'],
                    'improvement': adaptive_results['best_value'] - best_params.get('value', 0),
                    'grid_size': len(adaptive_grid),
                    'search_space': adaptive_search_space
                })

                self.current_refinement_iteration += 1

                # Check if we should continue refining
                if (self.current_refinement_iteration < self.config.max_adaptive_iterations and
                    self._should_continue_refinement(adaptive_results['best_value'], all_trials)):

                    # Recursively call adaptive refinement
                    recursive_results = self._run_adaptive_refinement_stage(
                        objective, adaptive_search_space, adaptive_results['best_params'],
                        all_trials + adaptive_results['trials']
                    )

                    if recursive_results:
                        # Merge results
                        adaptive_results['trials'].extend(recursive_results['trials'])
                        if self._is_better_result(recursive_results['best_value'], adaptive_results['best_value']):
                            adaptive_results['best_params'] = recursive_results['best_params']
                            adaptive_results['best_value'] = recursive_results['best_value']

            return adaptive_results

        except Exception as e:
            self.logger.error(f"❌ Adaptive refinement stage failed: {e}")
            return None

    def _should_perform_adaptive_refinement(self, all_trials: List[Dict]) -> bool:
        """Check if adaptive refinement should be performed."""
        if not all_trials or len(all_trials) < self.config.convergence_window:
            return False

        # Check if we've reached max iterations
        if self.current_refinement_iteration >= self.config.max_adaptive_iterations:
            return False

        # Check recent improvement rate
        recent_trials = all_trials[-self.config.convergence_window:]
        if len(recent_trials) < 5:
            return False

        # Calculate improvement rate
        values = [t['value'] for t in recent_trials if 'value' in t]
        if len(values) < 3:
            return False

        # Check if improvement rate is above threshold
        if self.config.direction == 'maximize':
            improvement_rate = (max(values) - min(values)) / (abs(min(values)) + 1e-8)
        else:
            improvement_rate = (min(values) - max(values)) / (abs(max(values)) + 1e-8)

        return improvement_rate > self.config.adaptive_refinement_threshold

    def _should_continue_refinement(self, current_best_value: float, all_trials: List[Dict]) -> bool:
        """Check if refinement should continue based on improvement."""
        if not all_trials or len(all_trials) < 5:
            return False

        # Get previous best value
        previous_values = [t['value'] for t in all_trials[-10:] if 'value' in t]
        if not previous_values:
            return False

        previous_best = max(previous_values) if self.config.direction == 'maximize' else min(previous_values)

        # Calculate improvement
        if self.config.direction == 'maximize':
            improvement = current_best_value - previous_best
        else:
            improvement = previous_best - current_best_value

        return improvement > self.config.adaptive_refinement_threshold

    def _generate_adaptive_search_space(self, original_search_space: Dict[str, Any],
                                      best_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptive search space around best parameters."""
        adaptive_space = {}

        for param_name, param_config in original_search_space.items():
            if param_name not in best_params:
                adaptive_space[param_name] = param_config
                continue

            best_val = best_params[param_name]

            if isinstance(param_config, tuple) and len(param_config) == 2:
                # (low, high) format
                low, high = param_config
                rng = high - low
                adaptive_rng = rng * self.config.adaptive_refinement_factor
                adaptive_low = max(low, best_val - adaptive_rng)
                adaptive_high = min(high, best_val + adaptive_rng)
                adaptive_space[param_name] = (adaptive_low, adaptive_high)

            elif isinstance(param_config, dict):
                # Advanced configuration format
                param_type = param_config.get('type', 'float')
                if param_type in ['int', 'float']:
                    low, high = param_config['low'], param_config['high']
                    rng = high - low
                    adaptive_rng = rng * self.config.adaptive_refinement_factor
                    adaptive_low = max(low, best_val - adaptive_rng)
                    adaptive_high = min(high, best_val + adaptive_rng)

                    adaptive_config = param_config.copy()
                    adaptive_config['low'] = adaptive_low
                    adaptive_config['high'] = adaptive_high
                    adaptive_space[param_name] = adaptive_config
                else:
                    # Keep categorical parameters as-is
                    adaptive_space[param_name] = param_config
            else:
                # Keep other parameter types as-is
                adaptive_space[param_name] = param_config

        return adaptive_space

    def _calculate_adaptive_grid_points(self) -> int:
        """Calculate number of grid points for adaptive refinement."""
        # Start with more points for first iteration, reduce for subsequent iterations
        base_points = self.config.fine_grid_points

        # Reduce points as we refine more
        reduction_factor = 0.8 ** self.current_refinement_iteration
        adaptive_points = int(base_points * reduction_factor)

        # Clamp to min/max bounds
        return max(self.config.min_grid_points,
                  min(self.config.max_grid_points, adaptive_points))

    def _generate_vectorbt_adaptive_grid(self, search_space: Dict[str, Any],
                                       best_params: Dict[str, Any],
                                       grid_points: int) -> List[Dict[str, Any]]:
        """Generate adaptive grid using VectorBT vectorized operations."""
        try:
            if not self.vectorbt_manager or not vbt:
                # Fallback to original method
                return build_fine_grid_around_best(search_space, best_params, grid_points)

            self.logger.debug(f"🔄 Generating VectorBT adaptive grid with {grid_points} points...")

            param_combinations = []

            for param_name, param_config in search_space.items():
                if param_name not in best_params:
                    continue

                best_val = best_params[param_name]

                if isinstance(param_config, tuple) and len(param_config) == 2:
                    # (low, high) format
                    low, high = param_config
                    if isinstance(low, int) and isinstance(high, int):
                        param_values = np.linspace(low, high, grid_points, dtype=int)
                    else:
                        param_values = np.linspace(low, high, grid_points)

                elif isinstance(param_config, dict):
                    # Advanced configuration format
                    param_type = param_config.get('type', 'float')
                    if param_type == 'int':
                        low, high = param_config['low'], param_config['high']
                        param_values = np.linspace(low, high, grid_points, dtype=int)
                    elif param_type == 'float':
                        low, high = param_config['low'], param_config['high']
                        if param_config.get('log', False) and low > 0 and high > low:
                            param_values = np.logspace(np.log10(low), np.log10(high), grid_points)
                        else:
                            param_values = np.linspace(low, high, grid_points)
                    elif param_type == 'categorical':
                        param_values = param_config.get('choices', [])
                    else:
                        continue
                else:
                    continue

                param_combinations.append([(param_name, val) for val in param_values])

            if not param_combinations:
                return []

            # Use plain NumPy for efficient adaptive grid generation
            if len(param_combinations) > 1:
                # Convert to parameter value lists
                param_values = []
                param_names = []
                for param_list in param_combinations:
                    values = [val for _, val in param_list]
                    param_values.append(values)
                    param_names.append(param_list[0][0])

                # Use itertools.product for grid generation
                combinations = list(itertools.product(*param_values))
                return [dict(zip(param_names, combo)) for combo in combinations]
            else:
                # Single parameter case
                param_name = param_combinations[0][0][0]
                values = [val for _, val in param_combinations[0]]
                return [{param_name: val} for val in values]

        except Exception as e:
            self.logger.warning(f"⚠️ Adaptive grid generation failed, using fallback: {e}")
            return build_fine_grid_around_best(search_space, best_params, grid_points)

    def _run_tpe_stage(self, objective: Callable, search_space: Dict[str, Any],
                      current_best_params: Dict[str, Any], n_trials: int) -> Dict[str, Any]:
        """Run TPE optimization stage with early stopping support."""
        try:
            self.logger.info(f"🔍 Stage 3: TPE optimization ({n_trials} trials)")

            # Create Optuna study with TPE sampler (without n_jobs parameter)
            sampler = TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                gamma=self.config.gamma,
                seed=self.config.seed,
                multivariate=self.config.multivariate,
                group=self.config.group
            )

            # Configure pruner if enabled
            pruner = None
            if self.config.enable_pruner:
                try:
                    ptype = (self.config.pruner_type or 'median').lower()
                    pparams = self.config.pruner_params or {}
                    if ptype == 'median':
                        pruner = optuna.pruners.MedianPruner(**pparams)
                    elif ptype in ('successive_halving', 'sha'):
                        pruner = optuna.pruners.SuccessiveHalvingPruner(**pparams)
                    elif ptype == 'hyperband':
                        pruner = optuna.pruners.HyperbandPruner(**pparams)
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to initialize pruner: {e}")
                    pruner = None

            self.study = optuna.create_study(
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
                study_name=f"tpe_optimization_{int(time.time())}"
            )

            # Run TPE optimization (early stopping handled in objective wrapper)
            self.study.optimize(
                self._create_objective_wrapper(objective, search_space),
                n_trials=n_trials,
                timeout=self.config.timeout,
                show_progress_bar=False  # Disable progress bar for cleaner output
            )

            # Extract results
            best_params = self.study.best_params
            best_value = self.study.best_value

            # Convert Optuna trials to our format
            # Cap trial summaries to limit memory usage
            all_trials = self.study.trials
            if self.config.max_trial_history and len(all_trials) > self.config.max_trial_history:
                all_trials = all_trials[-self.config.max_trial_history:]
            trials = [{
                'trial': t.number,
                'stage': 'tpe',
                'params': t.params,
                'value': t.value,
                'duration': t.duration.total_seconds() if t.duration else None
            } for t in all_trials]

            return {
                'best_params': best_params,
                'best_value': best_value,
                'trials': trials,
                'study': self.study
            }

        except Exception as e:
            self.logger.error(f"❌ TPE stage failed: {e}")
            return None

    def _is_batch_evaluation_safe(self, grid_points: List[Dict[str, Any]]) -> bool:
        """Check if batch evaluation is safe for the given grid points."""
        if not grid_points:
            return False

        try:
            # Check if all grid points have the same parameter names
            param_names = set(grid_points[0].keys())
            if not all(set(params.keys()) == param_names for params in grid_points):
                return False

            # Check if parameter values are compatible types for batch processing
            for param_name in param_names:
                values = [params[param_name] for params in grid_points]

                # Check for problematic types that can't be batched
                if any(isinstance(v, (dict, list)) and len(str(v)) > 50 for v in values):
                    return False

                # Check for extremely heterogeneous types
                types = [type(v) for v in values]
                if len(set(types)) > 3:  # More than 3 different types
                    return False

            return True
        except Exception:
            return False

    def _batch_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                           stage: str) -> Dict[str, Any]:
        """Evaluate grid points using hardware-accelerated batch processing."""
        try:
            self.logger.info(f"🔄 Batch evaluating {len(grid_points)} {stage} grid points")

            # Record performance metrics
            start_time = time.time()
            initial_memory = self._get_memory_usage() if self.performance_monitor else 0

            # Use batch processor for efficient evaluation
            if self.batch_processor:
                # Convert parameter dictionaries to format suitable for batch processing
                param_arrays = self._prepare_batch_parameters(grid_points)

                # Evaluate in batches for memory efficiency
                batch_size = min(self.config.batch_size, len(grid_points))
                results = []

                for i in range(0, len(grid_points), batch_size):
                    batch_params = grid_points[i:i + batch_size]
                    batch_results = []

                    for params in batch_params:
                        try:
                            # Create a mock trial object for grid evaluation
                            # Grid search passes dicts directly, not Trial objects
                            value = self._evaluate_params_dict(objective, params)
                            batch_results.append(value)
                        except Exception as e:
                            self.logger.warning(f"⚠️ Batch evaluation {i} failed: {e}")
                            batch_results.append(float('-inf') if self.config.direction == 'maximize' else float('inf'))

                    results.extend(batch_results)

            else:
                # Fallback: sequential evaluation with performance monitoring
                results = []
                for i, params in enumerate(grid_points):
                    trial_start = time.time()
                    try:
                        # Create a mock trial object for grid evaluation
                        value = self._evaluate_params_dict(objective, params)
                        results.append(value)
                    except Exception as e:
                        self.logger.warning(f"⚠️ Batch evaluation {i} failed: {e}")
                        results.append(float('-inf') if self.config.direction == 'maximize' else float('inf'))

            # Create trial records and check early stopping
            trials = []
            best_params = None
            best_value = None

            for i, (params, value) in enumerate(zip(grid_points, results)):
                trial_info = {
                    'trial': i,
                    'stage': stage,
                    'params': params,
                    'value': value,
                    'duration': None
                }
                trials.append(trial_info)

                if self._is_better_result(value, best_value):
                    best_params = params
                    best_value = value

                # Note: Early stopping for batch evaluation is less useful since
                # all evaluations are already computed. This is logged for consistency.

            # Record performance metrics
            end_time = time.time()
            final_memory = self._get_memory_usage() if self.performance_monitor else 0

            performance_info = {
                'stage': stage,
                'duration': end_time - start_time,
                'memory_used': final_memory - initial_memory,
                'evaluations_per_second': len(grid_points) / (end_time - start_time),
                'hardware_accelerated': self.batch_processor is not None
            }

            self.performance_metrics.append(performance_info)
            self.logger.info(f"   Batch evaluation completed in {performance_info['duration']:.2f}s")
            self.logger.info(f"   Evaluations/sec: {performance_info['evaluations_per_second']:.1f}")
            if self.batch_processor:
                self.logger.info("   Hardware acceleration: Enabled")

            return {
                'best_params': best_params,
                'best_value': best_value,
                'trials': trials
            }

        except Exception as e:
            self.logger.error(f"❌ Batch evaluation failed: {e}")
            return None

    def _evaluate_params_dict(self, objective: Callable, params: Dict[str, Any]) -> float:
        """
        Evaluate a parameter dictionary with an objective function.
        
        Handles both cases:
        1. Objective expects a dict directly (grid search)
        2. Objective expects an Optuna Trial object (TPE)
        
        For case 1, we pass the dict directly.
        For case 2, this will fail, so we catch it and return -inf/inf.
        """
        try:
            # Try calling objective with dict directly
            return objective(params)
        except (AttributeError, TypeError) as e:
            # If objective expects Trial object with suggest_* methods
            if 'suggest_int' in str(e) or 'suggest_float' in str(e):
                # Grid search doesn't use Optuna trials, so objective must handle dicts
                # Return worst value to skip this configuration
                return float('-inf') if self.config.direction == 'maximize' else float('inf')
            else:
                # Re-raise other errors
                raise

    def _chunked_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                              stage: str, chunk_size: int) -> Dict[str, Any]:
        """
        Evaluate grid points in chunks for memory efficiency.
        
        This method splits large grids into manageable chunks, evaluates each chunk,
        and keeps track of the best parameters across all chunks. This prevents OOM
        errors when evaluating very large parameter grids.
        
        Args:
            objective: Objective function to optimize
            grid_points: List of parameter dictionaries to evaluate
            stage: Stage name ('coarse' or 'fine')
            chunk_size: Maximum number of grid points per chunk
            
        Returns:
            Dictionary with best parameters, best value, and all trials
        """
        try:
            self.logger.info(f"🔄 Chunked evaluation: {len(grid_points)} points in chunks of {chunk_size}")
            
            # Initialize tracking variables
            all_trials = []
            best_params = None
            best_value = None
            start_time = time.time()
            
            # Split grid into chunks
            num_chunks = (len(grid_points) + chunk_size - 1) // chunk_size
            self.logger.info(f"   Processing {num_chunks} chunks")
            
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min((chunk_idx + 1) * chunk_size, len(grid_points))
                chunk_points = grid_points[chunk_start:chunk_end]
                
                self.logger.info(f"   📦 Chunk {chunk_idx + 1}/{num_chunks}: evaluating {len(chunk_points)} points")
                
                # Evaluate chunk
                chunk_results = self._sequential_evaluate_grid(objective, chunk_points, f"{stage}_chunk_{chunk_idx}")
                
                if chunk_results is None:
                    self.logger.warning(f"⚠️ Chunk {chunk_idx + 1} evaluation failed, skipping")
                    continue
                
                # Update best parameters if this chunk has better results
                if chunk_results['best_value'] is not None:
                    if self._is_better_result(chunk_results['best_value'], best_value):
                        best_params = chunk_results['best_params']
                        best_value = chunk_results['best_value']
                        self.logger.info(f"   ✨ New best found in chunk {chunk_idx + 1}: {best_value:.6f}")
                
                # Accumulate trials
                all_trials.extend(chunk_results['trials'])
                
                # Optional: Early stopping if we've found good enough results
                if self.config.enable_early_stopping and best_value is not None:
                    # Check if current best is good enough to skip remaining chunks
                    if self._should_stop_chunked_evaluation(best_value, chunk_idx, num_chunks):
                        self.logger.info(f"⏹️ Early stopping chunked evaluation at chunk {chunk_idx + 1}/{num_chunks}")
                        break
            
            end_time = time.time()
            
            # Record performance metrics
            performance_info = {
                'stage': stage,
                'duration': end_time - start_time,
                'total_evaluations': len(all_trials),
                'evaluations_per_second': len(all_trials) / (end_time - start_time),
                'num_chunks': num_chunks,
                'chunk_size': chunk_size,
                'chunked_evaluation': True
            }
            
            self.performance_metrics.append(performance_info)
            self.logger.info(f"   Chunked evaluation completed in {performance_info['duration']:.2f}s")
            self.logger.info(f"   Evaluations/sec: {performance_info['evaluations_per_second']:.1f}")
            self.logger.info(f"   Best value: {best_value if best_value is not None else 'N/A'}")
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'trials': all_trials
            }
            
        except Exception as e:
            self.logger.error(f"❌ Chunked evaluation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _should_stop_chunked_evaluation(self, current_best: float, current_chunk: int, total_chunks: int) -> bool:
        """
        Determine if chunked evaluation should stop early.
        
        This is a simple heuristic: if we've evaluated at least 25% of chunks and
        the current best is very good (based on improvement threshold), we can stop.
        """
        # Only consider early stopping after processing at least 25% of chunks
        if current_chunk < total_chunks * 0.25:
            return False
        
        # If we have early stopping threshold configured, check it
        if self.config.early_stopping_threshold is not None:
            if self.config.direction == 'maximize':
                return current_best >= self.config.early_stopping_threshold
            else:
                return current_best <= self.config.early_stopping_threshold
        
        return False

    def _sequential_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]],
                                stage: str) -> Dict[str, Any]:
        """Sequential evaluation with performance monitoring and early stopping."""
        try:
            self.logger.info(f"🔄 Sequential evaluating {len(grid_points)} {stage} grid points")

            trials = []
            best_params = None
            best_value = None

            for i, params in enumerate(grid_points):
                trial_start = time.time()
                try:
                    value = objective(params)
                    trial_duration = time.time() - trial_start

                    trial_info = {
                        'trial': i,
                        'stage': stage,
                        'params': params,
                        'value': value,
                        'duration': trial_duration
                    }
                    trials.append(trial_info)

                    if self._is_better_result(value, best_value):
                        best_params = params
                        best_value = value

                    # Check early stopping after minimum trials
                    if i >= self.config.early_stopping_patience:
                        if self._check_early_stopping_grid(trials, stage):
                            self.logger.info(f"   Stopped after {i+1}/{len(grid_points)} evaluations")
                            break

                except Exception as e:
                    self.logger.warning(f"⚠️ Sequential evaluation {i} failed: {e}")
                    trial_info = {
                        'trial': i,
                        'stage': stage,
                        'params': params,
                        'value': float('-inf') if self.config.direction == 'maximize' else float('inf'),
                        'duration': None,
                        'error': str(e)
                    }
                    trials.append(trial_info)

            return {
                'best_params': best_params,
                'best_value': best_value,
                'trials': trials
            }

        except Exception as e:
            self.logger.error(f"❌ Sequential evaluation failed: {e}")
            return None

    def _prepare_batch_parameters(self, grid_points: List[Dict[str, Any]]) -> Dict[str, List]:
        """Prepare parameters for batch processing."""
        # Extract parameter names and values for vectorized processing
        if not grid_points:
            return {}

        param_names = list(grid_points[0].keys())
        param_arrays = {name: [] for name in param_names}

        # Use vectorized operations for better performance instead of nested loops
        if self.matrix_processor and self.config.enable_batch_processing:
            try:
                # Use matrix operations for efficient parameter processing
                import numpy as np

                # Convert to numpy arrays for vectorized operations
                param_matrices = {}
                for name in param_names:
                    # Extract values for each parameter across all grid points
                    values = [params[name] for params in grid_points]

                    # Handle different data types safely
                    try:
                        # Check if all values are the same type and shape
                        if all(isinstance(v, (int, float)) for v in values):
                            # Numeric values - can be safely converted to array
                            param_matrices[name] = np.array(values)
                        elif all(isinstance(v, str) for v in values):
                            # String values - convert to object array
                            param_matrices[name] = np.array(values, dtype=object)
                        else:
                            # Mixed types - convert to object array
                            param_matrices[name] = np.array(values, dtype=object)
                    except (ValueError, TypeError) as e:
                        # If array creation fails, use object array
                        self.logger.debug(f"Array creation failed for {name}: {e}, using object array")
                        param_matrices[name] = np.array(values, dtype=object)

                # Use batch matrix operations for efficient processing
                if hasattr(self.matrix_processor, 'batch_matrix_multiply'):
                    # Process parameter matrices in batches for memory efficiency
                    batch_size = min(100, len(grid_points))  # Process in batches
                    processed_matrices = {}

                    for name, matrix in param_matrices.items():
                        if matrix.ndim == 1:
                            # Convert 1D arrays to column vectors for matrix operations
                            matrix = matrix.reshape(-1, 1)

                        # Process in batches to avoid memory issues
                        processed_batches = []
                        for i in range(0, len(matrix), batch_size):
                            batch = matrix[i:i + batch_size]
                            # Use matrix operations for processing if available
                            if hasattr(self.matrix_processor, 'optimize_matrix'):
                                processed_batch = self.matrix_processor.optimize_matrix(batch)
                            else:
                                processed_batch = batch
                            processed_batches.append(processed_batch)

                        # Combine processed batches
                        if processed_batches:
                            processed_matrices[name] = np.vstack(processed_batches)
                        else:
                            processed_matrices[name] = matrix

                    # Convert back to list format for compatibility
                    for name in param_names:
                        if name in processed_matrices:
                            param_arrays[name] = processed_matrices[name].flatten().tolist()
                        else:
                            # Fallback to original nested loop approach
                            param_arrays[name] = [params[name] for params in grid_points]
                else:
                    # Fallback to optimized list comprehension
                    for name in param_names:
                        param_arrays[name] = [params[name] for params in grid_points]

            except Exception as e:
                self.logger.warning(f"⚠️ Matrix-based parameter processing failed, using list comprehension: {e}")
                # Fallback to optimized list comprehension (still better than nested loops)
                for name in param_names:
                    param_arrays[name] = [params[name] for params in grid_points]
        else:
            # Use optimized list comprehension instead of nested loops
            for name in param_names:
                param_arrays[name] = [params[name] for params in grid_points]

        return param_arrays

    def _generate_vectorbt_coarse_grid(self, search_space: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
        """Generate coarse grid using VectorBT vectorized operations with enhanced performance."""
        try:
            # Use enhanced vectorization manager if available
            if self.enhanced_vectorization_manager:
                return self._enhanced_vectorbt_coarse_grid(search_space, grid_points)
            elif self.vectorbt_manager and vbt:
                return self._standard_vectorbt_coarse_grid(search_space, grid_points)
            else:
                # Fallback to original method
                return build_coarse_grid_from_search_space(search_space, grid_points)

        except Exception as e:
            self.logger.warning(f"VectorBT coarse grid generation failed: {e}, using fallback")
            return build_coarse_grid_from_search_space(search_space, grid_points)

    def _enhanced_vectorbt_coarse_grid(self, search_space: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
        """Generate coarse grid using enhanced VectorBT vectorization manager."""
        try:
            self.logger.debug("🔄 Generating enhanced VectorBT coarse grid...")

            # Use the enhanced vectorization manager for grid generation
            param_names = list(search_space.keys())
            param_configs = list(search_space.values())

            # Generate parameter values using VectorBT
            param_values = {}
            for name, config in zip(param_names, param_configs):
                if isinstance(config, dict):
                    param_type = config.get('type', 'float')
                    if param_type == 'float':
                        low, high = config['low'], config['high']
                        if config.get('log', False):
                            values = np.logspace(np.log10(low), np.log10(high), grid_points)
                        else:
                            values = np.linspace(low, high, grid_points)
                    elif param_type == 'int':
                        low, high = config['low'], config['high']
                        if high == low:
                            values = [low]
                        else:
                            pts = np.linspace(low, high, num=max(2, grid_points))
                            values = sorted({int(round(v)) for v in pts})
                    elif param_type == 'categorical':
                        values = config.get('choices', [])
                    else:
                        values = [config.get('default', 0)]
                else:
                    # Legacy tuple format
                    if isinstance(config, tuple) and len(config) == 2:
                        low, high = config
                        values = np.linspace(low, high, grid_points)
                    else:
                        values = [config]

                param_values[name] = values

            # Generate all combinations using VectorBT if beneficial
            if len(param_values) > 1 and self._should_use_vectorbt_combinations(param_values):
                combinations = self._vectorbt_generate_combinations(param_values)
            else:
                # Use itertools product with memory limiting
                total_combinations = 1
                for name in param_names:
                    total_combinations *= len(param_values[name])
                
                max_combinations = 10000
                if total_combinations > max_combinations:
                    self.logger.warning(f"⚠️ Too many combinations ({total_combinations}), using random sampling (max {max_combinations})")
                    # Use random sampling instead of full product
                    import random
                    combinations = []
                    # Convert numpy arrays to lists for random.choice
                    param_values_lists = {}
                    for name in param_names:
                        vals = param_values[name]
                        if hasattr(vals, 'tolist'):
                            param_values_lists[name] = vals.tolist()
                        else:
                            param_values_lists[name] = list(vals)
                    
                    for _ in range(min(max_combinations, int(total_combinations))):
                        combo = tuple(random.choice(param_values_lists[name]) for name in param_names)
                        combinations.append(combo)
                else:
                    # Convert to lists for itertools.product
                    param_lists = []
                    for name in param_names:
                        vals = param_values[name]
                        if hasattr(vals, 'tolist'):
                            param_lists.append(vals.tolist())
                        else:
                            param_lists.append(list(vals))
                    combinations = list(itertools.product(*param_lists))

            grid_points_list = [dict(zip(param_names, combo)) for combo in combinations]

            self.logger.debug(f"✅ Generated {len(grid_points_list)} enhanced VectorBT coarse grid points")
            return grid_points_list

        except Exception as e:
            self.logger.warning(f"Enhanced VectorBT coarse grid generation failed: {e}")
            raise

    def _should_use_vectorbt_combinations(self, param_values: Dict[str, List]) -> bool:
        """Determine if VectorBT should be used for combination generation."""
        if not self.enhanced_vectorization_manager:
            return False

        # Use VectorBT for large parameter spaces
        total_combinations = 1
        for values in param_values.values():
            total_combinations *= len(values)

        # Use VectorBT for parameter spaces with 100-10000 combinations
        # Above 10000, we'll use random sampling instead
        return 100 < total_combinations <= 10000

    def _vectorbt_generate_combinations(self, param_values: Dict[str, List]) -> List[Tuple]:
        """Generate combinations using VectorBT vectorized operations."""
        try:
            # Convert parameter values to arrays
            param_arrays = {}
            for name, values in param_values.items():
                param_arrays[name] = np.array(values)

            # Use VectorBT for efficient combination generation
            names = list(param_arrays.keys())
            arrays = list(param_arrays.values())

            # Generate meshgrid for all parameters
            meshgrid = np.meshgrid(*arrays, indexing='ij')

            # Reshape to get all combinations
            combinations = []
            for i in range(meshgrid[0].size):
                combo = tuple(meshgrid[j].flat[i] for j in range(len(meshgrid)))
                combinations.append(combo)

            return combinations

        except Exception as e:
            self.logger.warning(f"VectorBT combination generation failed: {e}, using itertools")
            # Fallback to itertools with memory limit
            total_combinations = 1
            for values in param_values.values():
                total_combinations *= len(values)
            
            max_combinations = 10000
            if total_combinations > max_combinations:
                self.logger.warning(f"⚠️ Too many combinations ({total_combinations}), using random sampling (max {max_combinations})")
                # Use random sampling instead of full product
                import random
                combinations = []
                # Convert to lists for random sampling
                param_lists = []
                for values in param_values.values():
                    if hasattr(values, 'tolist'):
                        param_lists.append(values.tolist())
                    else:
                        param_lists.append(list(values))
                
                for _ in range(min(max_combinations, int(total_combinations))):
                    combo = tuple(random.choice(vals) for vals in param_lists)
                    combinations.append(combo)
                return combinations
            else:
                # Convert to lists for itertools.product
                param_lists = []
                for values in param_values.values():
                    if hasattr(values, 'tolist'):
                        param_lists.append(values.tolist())
                    else:
                        param_lists.append(list(values))
                return list(itertools.product(*param_lists))

    def _standard_vectorbt_coarse_grid(self, search_space: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
        """Generate coarse grid using standard VectorBT operations."""
        try:
            self.logger.debug("🔄 Generating standard VectorBT coarse grid...")

            # Use VectorBT's advanced parameter space generation
            param_combinations = []

            for param_name, param_config in search_space.items():
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    # (low, high) format for numerical parameters
                    low, high = param_config
                    if isinstance(low, int) and isinstance(high, int):
                        # Use NumPy's optimized integer range generation
                        param_values = np.linspace(low, high, grid_points, dtype=int)
                    else:
                        # Use NumPy's optimized float range generation
                        param_values = np.linspace(low, high, grid_points)
                elif isinstance(param_config, dict):
                    # Advanced configuration format
                    param_type = param_config.get('type', 'float')
                    if param_type == 'int':
                        low, high = param_config['low'], param_config['high']
                        param_values = np.linspace(low, high, grid_points, dtype=int)
                    elif param_type == 'float':
                        low, high = param_config['low'], param_config['high']
                        if param_config.get('log', False):
                            # Use NumPy's optimized logspace
                            param_values = np.logspace(np.log10(low), np.log10(high), grid_points)
                        else:
                            param_values = np.linspace(low, high, grid_points)
                    elif param_type == 'categorical':
                        param_values = np.array(param_config.get('choices', []))
                    else:
                        continue
                elif isinstance(param_config, list):
                    # Choice format for categorical parameters
                    param_values = np.array(param_config)
                else:
                    continue

                param_combinations.append([(param_name, val) for val in param_values])

            if not param_combinations:
                return []

            # Use plain NumPy and itertools for grid generation
            if len(param_combinations) > 1:
                # Convert to parameter value lists
                param_values = []
                param_names = []
                for param_list in param_combinations:
                    values = [val for _, val in param_list]
                    param_values.append(values)
                    param_names.append(param_list[0][0])

                # Calculate total combinations to avoid memory issues
                total_combinations = 1
                for values in param_values:
                    total_combinations *= len(values)
                
                # Limit combinations to avoid memory issues (max 10,000 combinations)
                max_combinations = 10000
                if total_combinations > max_combinations:
                    self.logger.warning(f"⚠️ Too many combinations ({total_combinations}), using random sampling (max {max_combinations})")
                    # Use random sampling instead of full product
                    import random
                    combinations = []
                    for _ in range(min(max_combinations, total_combinations)):
                        combo = tuple(random.choice(values) for values in param_values)
                        combinations.append(combo)
                else:
                    # Use itertools.product for grid generation
                    combinations = list(itertools.product(*param_values))
                
                return [dict(zip(param_names, combo)) for combo in combinations]
            else:
                # Single parameter case
                param_name = param_combinations[0][0][0]
                values = [val for _, val in param_combinations[0]]
                return [{param_name: val} for val in values]

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT grid generation failed, using fallback: {e}")
            return build_coarse_grid_from_search_space(search_space, grid_points)

    def _generate_vectorbt_fine_grid(self, search_space: Dict[str, Any], best_params: Dict[str, Any], grid_points: int) -> List[Dict[str, Any]]:
        """Generate fine grid around best parameters using VectorBT vectorized operations with enhanced performance."""
        try:
            if not self.vectorbt_manager or not vbt:
                # Fallback to original method
                return build_fine_grid_around_best(search_space, best_params, grid_points)

            self.logger.debug("🔄 Generating VectorBT fine grid with enhanced vectorization...")

            param_combinations = []

            for param_name, param_config in search_space.items():
                if param_name not in best_params:
                    continue

                best_val = best_params[param_name]

                if isinstance(param_config, tuple) and len(param_config) == 2:
                    # (low, high) format
                    low, high = param_config
                    rng = high - low
                    fine_rng = rng * 0.2
                    fine_min = max(low, best_val - fine_rng)
                    fine_max = min(high, best_val + fine_rng)

                    if isinstance(low, int) and isinstance(high, int):
                        # Use NumPy's optimized integer range
                        param_values = np.linspace(fine_min, fine_max, grid_points, dtype=int)
                    else:
                        # Use NumPy's optimized float range
                        param_values = np.linspace(fine_min, fine_max, grid_points)

                elif isinstance(param_config, dict):
                    # Advanced configuration format
                    param_type = param_config.get('type', 'float')
                    if param_type == 'int':
                        low, high = param_config['low'], param_config['high']
                        fine_min = max(low, int(best_val) - 2)
                        fine_max = min(high, int(best_val) + 2)
                        param_values = np.arange(fine_min, fine_max + 1, dtype=int)
                    elif param_type == 'float':
                        low, high = param_config['low'], param_config['high']
                        rng = high - low
                        fine_rng = rng * 0.2
                        fine_min = max(low, best_val - fine_rng)
                        fine_max = min(high, best_val + fine_rng)

                        if param_config.get('log', False) and fine_min > 0 and fine_max > fine_min:
                            # Use NumPy's optimized logspace
                            param_values = np.logspace(np.log10(fine_min), np.log10(fine_max), grid_points)
                        else:
                            param_values = np.linspace(fine_min, fine_max, grid_points)
                    elif param_type == 'categorical':
                        param_values = np.array(param_config.get('choices', []))
                    else:
                        continue
                else:
                    continue

                param_combinations.append([(param_name, val) for val in param_values])

            if not param_combinations:
                return []

            # Use plain NumPy and itertools for grid generation
            if len(param_combinations) > 1:
                # Convert to parameter value lists
                param_values = []
                param_names = []
                for param_list in param_combinations:
                    values = [val for _, val in param_list]
                    param_values.append(values)
                    param_names.append(param_list[0][0])

                # Use itertools.product for grid generation
                combinations = list(itertools.product(*param_values))
                return [dict(zip(param_names, combo)) for combo in combinations]
            else:
                # Single parameter case
                param_name = param_combinations[0][0][0]
                values = [val for _, val in param_combinations[0]]
                return [{param_name: val} for val in values]

        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT fine grid generation failed, using fallback: {e}")
            return build_fine_grid_around_best(search_space, best_params, grid_points)

    def _vectorbt_batch_evaluate_grid(self, objective: Callable, grid_points: List[Dict[str, Any]], stage: str) -> Dict[str, Any]:
        """Evaluate grid points using VectorBT-optimized batch processing with enhanced performance."""
        try:
            self.logger.info(f"🔄 VectorBT batch evaluating {len(grid_points)} {stage} grid points")

            # Record performance metrics
            start_time = time.time()
            initial_memory = self._get_memory_usage() if self.performance_monitor else 0

            # Use VectorBT for efficient batch evaluation
            if self.vectorbt_manager and vbt:
                # Convert parameter dictionaries to VectorBT arrays for vectorized processing
                param_arrays = self._prepare_vectorbt_parameters(grid_points)

                # Use VectorBT's optimized parallel processing
                results = []
                chunk_size = min(self.config.vectorbt_chunk_size, len(grid_points))

                # Process in optimized chunks using VectorBT's memory management
                for i in range(0, len(grid_points), chunk_size):
                    chunk_params = grid_points[i:i + chunk_size]
                    chunk_results = []

                    # Use VectorBT's built-in parallel processing with better memory management
                    if self.config.vectorbt_enable_parallel and len(chunk_params) > 1:
                        # Create VectorBT array for batch processing
                        param_batch = np.array([list(params.values()) for params in chunk_params])

                        # Use VectorBT's vectorized operations for objective evaluation
                        try:
                            # Vectorized objective evaluation (if objective supports it)
                            if hasattr(objective, '__vectorized__'):
                                chunk_results = objective.vectorized_evaluate(chunk_params)
                            else:
                                # Parallel evaluation (VectorBT 0.28.1 handles threading automatically)
                                # Note: threading settings removed in VectorBT 0.28.1 as it's handled automatically
                                # vbt.settings.threading['num_threads'] = self.config.vectorbt_parallel_workers
                                for params in chunk_params:
                                    try:
                                        value = objective(params)
                                        chunk_results.append(value)
                                    except Exception as e:
                                        self.logger.warning(f"⚠️ VectorBT batch evaluation {i} failed: {e}")
                                        chunk_results.append(float('-inf') if self.config.direction == 'maximize' else float('inf'))
                        except Exception as e:
                            self.logger.warning(f"⚠️ VectorBT batch evaluation {i} failed: {e}")
                            chunk_results.append(float('-inf') if self.config.direction == 'maximize' else float('inf'))
                    else:
                        # Sequential processing within chunk
                        for params in chunk_params:
                            try:
                                value = objective(params)
                                chunk_results.append(value)
                            except Exception as e:
                                self.logger.warning(f"⚠️ VectorBT batch evaluation {i} failed: {e}")
                                chunk_results.append(float('-inf') if self.config.direction == 'maximize' else float('inf'))

                    results.extend(chunk_results)
            else:
                # Fallback to regular batch processing
                return self._batch_evaluate_grid(objective, grid_points, stage)

            # Create trial records
            trials = []
            best_params = None
            best_value = None

            for i, (params, value) in enumerate(zip(grid_points, results)):
                trial_info = {
                    'trial': i,
                    'stage': stage,
                    'params': params,
                    'value': value,
                    'duration': None
                }
                trials.append(trial_info)

                if self._is_better_result(value, best_value):
                    best_params = params
                    best_value = value

            # Record performance metrics
            end_time = time.time()
            final_memory = self._get_memory_usage() if self.performance_monitor else 0

            performance_info = {
                'stage': stage,
                'duration': end_time - start_time,
                'memory_used': final_memory - initial_memory,
                'evaluations_per_second': len(grid_points) / (end_time - start_time),
                'vectorbt_optimized': True,
                'chunk_size': chunk_size
            }

            self.performance_metrics.append(performance_info)
            self.logger.info(f"   VectorBT batch evaluation completed in {performance_info['duration']:.2f}s")
            self.logger.info(f"   Evaluations/sec: {performance_info['evaluations_per_second']:.1f}")
            self.logger.info("   VectorBT optimization: Enabled")

            return {
                'best_params': best_params,
                'best_value': best_value,
                'trials': trials
            }

        except Exception as e:
            self.logger.error(f"❌ VectorBT batch evaluation failed: {e}")
            # Fallback to regular batch evaluation
            return self._batch_evaluate_grid(objective, grid_points, stage)

    def _prepare_vectorbt_parameters(self, grid_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare parameters for VectorBT vectorized processing."""
        if not grid_points or not vbt:
            return {}

        # Extract parameter names and values for VectorBT processing
        param_names = list(grid_points[0].keys())
        param_arrays = {}

        try:
            # Use VectorBT arrays for efficient parameter processing
            for name in param_names:
                values = [params[name] for params in grid_points]
                # Convert to VectorBT array for vectorized operations
                param_arrays[name] = np.array(values)

            return param_arrays
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT parameter preparation failed: {e}")
            return {}

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            if self.performance_monitor:
                return self.performance_monitor.get_memory_usage()
            elif hasattr(self.hardware_manager, 'get_memory_usage'):
                return self.hardware_manager.get_memory_usage()
            else:
                # Fallback: use psutil if available
                try:
                    import psutil
                    process = psutil.Process()
                    return process.memory_info().rss / (1024 * 1024)  # Convert to MB
                except ImportError:
                    return 0.0
        except Exception:
            return 0.0

    def _create_objective_wrapper(self, objective: Callable, search_space: Dict[str, Any]) -> Callable:
        """Create wrapper for objective function to work with Optuna."""
        def optuna_objective(trial: optuna.Trial) -> float:
            """Optuna-compatible objective function."""
            try:
                # Suggest parameters based on search space
                params = {}
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, tuple) and len(param_config) == 2:
                        # (low, high) format for numerical parameters
                        low, high = param_config
                        if isinstance(low, int) and isinstance(high, int):
                            params[param_name] = trial.suggest_int(param_name, low, high)
                        else:
                            params[param_name] = trial.suggest_float(param_name, low, high)
                    elif isinstance(param_config, list):
                        # Choice format for categorical parameters
                        params[param_name] = trial.suggest_categorical(param_name, param_config)
                    elif isinstance(param_config, dict):
                        # Advanced configuration format
                        if param_config.get('type') == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name,
                                param_config['low'],
                                param_config['high']
                            )
                        elif param_config.get('type') == 'float':
                            params[param_name] = trial.suggest_float(
                                param_name,
                                param_config['low'],
                                param_config['high'],
                                log=param_config.get('log', False)
                            )
                        elif param_config.get('type') == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name,
                                param_config['choices']
                            )
                        else:
                            raise ValueError(f"Unknown parameter config for {param_name}")

                # Evaluate objective function
                value = objective(params)

                # Apply constraints if specified
                if self.config.constraints:
                    for constraint_name, constraint_func in self.config.constraints.items():
                        if not constraint_func(params, value):
                            # Return worst possible value for constraint violation
                            if self.config.direction == 'maximize':
                                return float('-inf')
                            else:
                                return float('inf')

                # Check early stopping conditions
                if self._should_stop_early(trial.number, value):
                    raise optuna.TrialPruned()

                return value

            except optuna.TrialPruned:
                raise
            except Exception as e:
                self.logger.warning(f"⚠️ Trial {trial.number} failed: {e}")
                # Return worst possible value for failed trials
                if self.config.direction == 'maximize':
                    return float('-inf')
                else:
                    return float('inf')

        return optuna_objective

    def _should_stop_early(self, trial_number: int, current_value: float) -> bool:
        """Check if optimization should stop early."""
        if self.config.early_stopping_patience is None:
            return False

        if trial_number < self.config.early_stopping_patience:
            return False

        # Check if best value hasn't improved for patience trials
        recent_trials = self.study.trials[-self.config.early_stopping_patience:]
        if self.config.direction == 'maximize':
            best_recent = max(t.value for t in recent_trials if t.value != float('-inf'))
            return current_value < best_recent and self.config.early_stopping_threshold is not None
        else:
            best_recent = min(t.value for t in recent_trials if t.value != float('inf'))
            return current_value > best_recent and self.config.early_stopping_threshold is not None

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of optimization results including hardware performance."""
        if not hasattr(self, 'optimization_history') or not self.optimization_history:
            return {'error': 'No optimization has been run yet'}

        # Basic optimization summary
        completed_trials = [t for t in self.optimization_history if t.get('stage') == 'tpe']
        coarse_trials = [t for t in self.optimization_history if t.get('stage') == 'coarse']
        fine_trials = [t for t in self.optimization_history if t.get('stage') == 'fine']
        adaptive_trials = [t for t in self.optimization_history if t.get('stage') == 'adaptive']

        summary = {
            'total_trials': len(self.optimization_history),
            'coarse_trials': len(coarse_trials),
            'fine_trials': len(fine_trials),
            'adaptive_trials': len(adaptive_trials),
            'tpe_trials': len(completed_trials),
            'best_params': self.best_params,
            'best_value': self.best_value,
            'direction': self.config.direction,
            'hardware_optimization_enabled': self.hardware_manager is not None,
            'batch_processing_enabled': self.batch_processor is not None,
            'matrix_acceleration_enabled': self.matrix_processor is not None,
            'vectorbt_optimization_enabled': self.vectorbt_manager is not None,
            'vectorbt_available': self.vectorbt_available,
            'adaptive_refinement_enabled': self.config.enable_adaptive_grid_refinement,
            'adaptive_refinement_iterations': self.current_refinement_iteration,
            'adaptive_refinement_history': self.adaptive_refinement_history
        }

        # Performance metrics summary
        if self.performance_metrics:
            stage_performance = {}
            for metric in self.performance_metrics:
                stage = metric['stage']
                if stage not in stage_performance:
                    stage_performance[stage] = []
                stage_performance[stage].append(metric)

            summary['performance_summary'] = {}
            for stage, metrics in stage_performance.items():
                if metrics:
                    avg_duration = np.mean([m['duration'] for m in metrics])
                    avg_throughput = np.mean([m['evaluations_per_second'] for m in metrics])
                    hardware_accelerated = any([m['hardware_accelerated'] for m in metrics])

                    summary['performance_summary'][stage] = {
                        'avg_duration': avg_duration,
                        'avg_throughput': avg_throughput,
                        'hardware_accelerated': hardware_accelerated,
                        'num_evaluations': len(metrics)
                    }

        # Hardware configuration summary
        if self.hardware_manager:
            summary['hardware_config'] = {
                'workload_type': self.config.workload_type,
                'optimization_level': self.config.optimization_level,
                'gpu_acceleration': self.config.enable_gpu_acceleration,
                'batch_processing': self.config.enable_batch_processing,
                'memory_limit_gb': self.config.memory_limit_gb
            }

        return summary

    def get_adaptive_refinement_stats(self) -> Dict[str, Any]:
        """Get adaptive refinement statistics and analysis."""
        if not self.adaptive_refinement_history:
            return {'message': 'No adaptive refinement performed'}

        stats = {
            'total_iterations': len(self.adaptive_refinement_history),
            'total_improvement': 0.0,
            'average_improvement_per_iteration': 0.0,
            'convergence_analysis': {},
            'search_space_evolution': [],
            'refinement_efficiency': {}
        }

        # Calculate improvement metrics
        if len(self.adaptive_refinement_history) > 0:
            initial_value = self.adaptive_refinement_history[0]['best_value']
            final_value = self.adaptive_refinement_history[-1]['best_value']

            if self.config.direction == 'maximize':
                stats['total_improvement'] = final_value - initial_value
            else:
                stats['total_improvement'] = initial_value - final_value

            stats['average_improvement_per_iteration'] = stats['total_improvement'] / len(self.adaptive_refinement_history)

        # Analyze convergence
        improvements = [h['improvement'] for h in self.adaptive_refinement_history]
        if len(improvements) > 1:
            stats['convergence_analysis'] = {
                'improvement_trend': 'increasing' if improvements[-1] > improvements[0] else 'decreasing',
                'volatility': np.std(improvements) if len(improvements) > 1 else 0.0,
                'convergence_rate': self._calculate_convergence_rate()
            }

        # Track search space evolution
        for i, history in enumerate(self.adaptive_refinement_history):
            search_space = history.get('search_space', {})
            stats['search_space_evolution'].append({
                'iteration': i,
                'search_space_size': len(search_space),
                'parameter_ranges': {k: (v[0], v[1]) if isinstance(v, tuple) else str(v)
                                   for k, v in search_space.items()}
            })

        # Calculate refinement efficiency
        if len(self.adaptive_refinement_history) > 0:
            total_trials = sum(h['grid_size'] for h in self.adaptive_refinement_history)
            stats['refinement_efficiency'] = {
                'total_adaptive_trials': total_trials,
                'improvement_per_trial': stats['total_improvement'] / total_trials if total_trials > 0 else 0.0,
                'average_grid_size': np.mean([h['grid_size'] for h in self.adaptive_refinement_history])
            }

        return stats

    def get_parameter_importance(self) -> Dict[str, float]:
        """Get parameter importance scores."""
        if not self.study:
            return {}

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception as e:
            self.logger.warning(f"⚠️ Could not calculate parameter importance: {e}")
            return {}

    def plot_optimization_history(self) -> 'optuna.visualization.matplotlib.PlotlyPlot':
        """Plot optimization history (requires plotly)."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        try:
            return optuna.visualization.plot_optimization_history(self.study)
        except ImportError:
            self.logger.warning("⚠️ Plotly not available for visualization")
            return None

    def plot_param_importances(self) -> 'optuna.visualization.matplotlib.PlotlyPlot':
        """Plot parameter importances (requires plotly)."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        try:
            return optuna.visualization.plot_param_importances(self.study)
        except ImportError:
            self.logger.warning("⚠️ Plotly not available for visualization")
            return None

    def save_study(self, filepath: str) -> None:
        """Save Optuna study to file."""
        if not self.study:
            raise ValueError("No optimization has been run yet")

        try:
            import joblib
            joblib.dump(self.study, filepath)
            self.logger.info(f"💾 Study saved to {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save study: {e}")
            raise

    def load_study(self, filepath: str) -> None:
        """Load Optuna study from file."""
        try:
            import joblib
            self.study = joblib.load(filepath)
            if self.study.trials:
                self.best_params = self.study.best_params
                self.best_value = self.study.best_value
            self.logger.info(f"📂 Study loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load study: {e}")
            raise


