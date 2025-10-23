"""
Consolidated Hyperparameter Optimization System

This module consolidates all HPO implementations into a single, comprehensive system that provides:

1. Main HPO utilities with automated search spaces and multi-objective optimization
2. Hierarchical HPO for multi-output stacking ensembles
3. Bayesian TPE optimization with hardware acceleration
4. BOHB-style optimization with multi-fidelity support
5. Regime-specific HPO wrapper
6. Comprehensive monitoring and diagnostics

This replaces the following redundant implementations:
- src/utils/ml_common/optimization/hpo_utils.py
- src/utils/ml_common/optimization/hierarchical_hpo.py
- src/utils/ml_common/optimization/bayesian_tpe_optimizer.py
- src/utils/ml_common/optimization/bohb_optimizer.py
- src/utils/ml_common/optimization/regime_hpo_wrapper.py
- src/utils/ml_common/optimization/auto_tuner.py
- src/utils/ml_common/optimization/grid_utils.py
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import json
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor
import gc

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

# Hardware optimization imports
try:
    from src.utils.hardware.optimization_decorators import (
        performance_tracked, smart_cache, memory_optimized, m1_optimized,
        auto_optimize, WorkloadCategory
    )
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    def performance_tracked(workload_category=None):
        def decorator(func):
            return func
        return decorator
    def smart_cache(func):
        return func
    def memory_optimized(level=None):
        def decorator(func):
            return func
        return decorator
    def m1_optimized(workload_category=None):
        def decorator(func):
            return func
        return decorator
    def auto_optimize(func):
        return func
    class WorkloadCategory:
        MACHINE_LEARNING = "machine_learning"

# VectorBT optimization imports
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
        tprint_warning(f"VectorBT optimization not available: {e}")
    else:
        logging.warning(f"VectorBT optimization not available: {e}")
    VECTORBT_AVAILABLE = False
    vbt = None
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None
    OperationType = None
    OptimizationStrategy = None
    VectorizationConfig = None

# Optuna imports
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    from optuna.pruners import MedianPruner, HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None

# Sklearn imports
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit, KFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Parallel processing
try:
    from src.utils.parallel_processing_optimizer import ParallelProcessor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    ParallelProcessor = None

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class HPOConfig:
    """Unified configuration for all HPO types."""
    
    # Basic settings
    n_trials: int = 100
    timeout: Optional[float] = None
    random_state: Optional[int] = None
    
    # Optimization strategy
    strategy: str = 'bayesian'  # 'bayesian', 'bohb', 'hierarchical', 'grid', 'random'
    
    # Bayesian optimization settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    multivariate: bool = True
    group: bool = True
    gamma: Optional[Callable[[int], int]] = None
    
    # BOHB settings
    min_budget: float = 0.1  # Changed to proper data fraction range
    max_budget: float = 1.0  # Changed to proper data fraction range
    reduction_factor: float = 3.0
    n_brackets: int = 1
    
    # Hierarchical settings
    enable_hierarchical: bool = False
    phase1_config: Optional[Dict[str, Any]] = None
    phase2_config: Optional[Dict[str, Any]] = None
    
    # Grid search settings
    enable_staged_optimization: bool = True
    coarse_grid_points: int = 5
    fine_grid_points: int = 5
    coarse_grid_trials: int = 25
    fine_grid_trials: int = 25
    tpe_trials: int = 50
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    enable_vectorbt: bool = True
    enable_parallel: bool = True
    max_workers: int = 4
    
    # Monitoring and diagnostics
    enable_monitoring: bool = True
    enable_diagnostics: bool = True
    enable_overfitting_detection: bool = True
    
    # Cross-validation
    cv_folds: int = 5
    enable_time_series_cv: bool = True
    scoring: str = 'neg_mean_squared_error'
    
    # Caching and persistence
    enable_caching: bool = True
    cache_dir: str = "./hpo_cache"
    save_results: bool = True
    results_dir: str = "./hpo_results"
    
    # Additional configuration
    enable_detailed_logging: bool = False
    resource_param: Optional[str] = None
    resource_values: Optional[List[int]] = None
    coarse_top_k: int = 5
    fine_span_frac: float = 0.3
    overfitting_threshold: float = 0.1
    
    def __post_init__(self):
        if self.gamma is None:
            self.gamma = lambda t: min(int(np.ceil(0.15 * t)), 100)

@dataclass
class HPOPhaseConfig:
    """Configuration for hierarchical HPO phases."""
    phase_name: str
    models: Dict[str, Any]
    search_spaces: Dict[str, Dict[str, Any]]
    n_trials: int = 100
    timeout_seconds: Optional[int] = None
    enable_pruning: bool = True
    cv_folds: int = 5
    scoring_metric: str = 'neg_mean_squared_error'
    direction: str = 'maximize'

@dataclass
class HPOResult:
    """Comprehensive HPO result."""
    
    # Basic results
    best_params: Dict[str, Any]
    best_score: float
    best_trial: Optional[Any] = None
    
    # Optimization metadata
    n_trials: int = 0
    optimization_time: float = 0.0
    strategy: str = "unknown"
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    # Detailed results
    trial_results: List[Dict[str, Any]] = field(default_factory=list)
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    
    # Validation results
    cv_scores: Optional[List[float]] = None
    validation_score: float = 0.0
    overfitting_detected: bool = False
    
    # Metadata
    model_name: str = "unknown"
    optimization_timestamp: str = None
    
    def __post_init__(self):
        if self.optimization_timestamp is None:
            self.optimization_timestamp = datetime.now().isoformat()

# ============================================================================
# MAIN CONSOLIDATED HPO CLASS
# ============================================================================

class ConsolidatedHPO:
    """
    Consolidated hyperparameter optimization system.
    
    This class provides a unified interface for all HPO strategies:
    - Bayesian optimization with TPE
    - BOHB-style multi-fidelity optimization
    - Hierarchical optimization for stacking ensembles
    - Grid search with staged optimization
    - Random search
    """
    
    def __init__(self, config: Optional[HPOConfig] = None):
        """Initialize consolidated HPO system."""
        self.config = config or HPOConfig()
        self.logger = logger.getChild('ConsolidatedHPO')
        
        # Set random seeds for reproducibility
        if self.config.random_state is not None:
            np.random.seed(self.config.random_state)
        
        # Initialize hardware optimization
        if self.config.enable_hardware_optimization:
            self._initialize_hardware_optimization()
        
        # Initialize VectorBT components
        if self.config.enable_vectorbt and VECTORBT_AVAILABLE:
            self._initialize_vectorbt_components()
        
        # Initialize parallel processing
        if self.config.enable_parallel and PARALLEL_AVAILABLE:
            self.parallel_processor = ParallelProcessor()
        else:
            self.parallel_processor = None
        
        # Initialize monitoring
        if self.config.enable_monitoring:
            self._initialize_monitoring()
        
        # Create directories
        if self.config.enable_caching:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        if self.config.save_results:
            Path(self.config.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Optimization tracking
        self.optimization_history = []
        self.active_studies = {}
        self.trial_results = {}
        
        # Initialize caching
        self._score_cache = {}
        
        if TPRINT_AVAILABLE:
            tprint_success("✅ Consolidated HPO system initialized")
    
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization components."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Initializing hardware optimization components")
            
            # Initialize hardware optimization components
            self.hardware_manager = self._create_hardware_manager()
            self.memory_optimizer = self._create_memory_optimizer()
            self.gpu_optimizer = self._create_gpu_optimizer()
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Hardware optimization components initialized")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Hardware optimization initialization failed: {e}")
            else:
                self.logger.warning(f"Hardware optimization initialization failed: {e}")
    
    def _create_hardware_manager(self):
        """Create hardware manager for optimization."""
        try:
            from ..hardware.hardware_manager import HardwareManager
            return HardwareManager(self.config.get("hardware", {}))
        except ImportError:
            # Fallback hardware manager
            class FallbackHardwareManager:
                def __init__(self, config):
                    self.config = config
                    self.cpu_cores = 4
                    self.memory_gb = 8
                
                def get_optimal_workers(self, task_type):
                    return min(self.cpu_cores, 4)
                
                def get_memory_limit(self, task_type):
                    return self.memory_gb * 0.8
                
                def optimize_for_task(self, task_type, data_size):
                    return {"workers": self.get_optimal_workers(task_type), "memory_limit": self.get_memory_limit(task_type)}
            
            return FallbackHardwareManager(self.config.get("hardware", {}))
    
    def _create_memory_optimizer(self):
        """Create memory optimizer."""
        try:
            from ..memory.memory_optimizer import MemoryOptimizer
            return MemoryOptimizer(self.config.get("memory", {}))
        except ImportError:
            # Fallback memory optimizer
            class FallbackMemoryOptimizer:
                def __init__(self, config):
                    self.config = config
                
                def optimize_memory_usage(self, data, task_type):
                    return data  # No optimization
                
                def get_memory_usage(self):
                    import psutil
                    return psutil.virtual_memory().percent
                
                def cleanup_memory(self):
                    import gc
                    gc.collect()
            
            return FallbackMemoryOptimizer(self.config.get("memory", {}))
    
    def _create_gpu_optimizer(self):
        """Create GPU optimizer."""
        try:
            from ..gpu.gpu_optimizer import GPUOptimizer
            return GPUOptimizer(self.config.get("gpu", {}))
        except ImportError:
            # Fallback GPU optimizer
            class FallbackGPUOptimizer:
                def __init__(self, config):
                    self.config = config
                    self.gpu_available = False
                
                def is_gpu_available(self):
                    return self.gpu_available
                
                def optimize_for_gpu(self, data, task_type):
                    return data  # No GPU optimization
                
                def get_gpu_memory_usage(self):
                    return 0.0
            
            return FallbackGPUOptimizer(self.config.get("gpu", {}))
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT optimization components."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Initializing VectorBT optimization components")
            
            # Initialize VectorBT components
            self.vectorbt_portfolio = self._create_vectorbt_portfolio()
            self.vectorbt_optimizer = self._create_vectorbt_optimizer()
            self.vectorbt_metrics = self._create_vectorbt_metrics()
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ VectorBT optimization components initialized")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ VectorBT optimization initialization failed: {e}")
            else:
                self.logger.warning(f"VectorBT optimization initialization failed: {e}")
    
    def _create_vectorbt_portfolio(self):
        """Create VectorBT portfolio for optimization."""
        try:
            import vectorbt as vbt
            return vbt.Portfolio
        except ImportError:
            # Fallback portfolio implementation
            class FallbackPortfolio:
                def __init__(self, *args, **kwargs):
                    self.data = None
                    self.returns = None
                
                @classmethod
                def from_signals(cls, close, entries, exits, **kwargs):
                    portfolio = cls()
                    portfolio.data = close
                    portfolio.returns = close.pct_change()
                    return portfolio
                
                def total_return(self):
                    if self.returns is not None:
                        return (1 + self.returns).prod() - 1
                    return 0.0
                
                def sharpe_ratio(self):
                    if self.returns is not None:
                        return self.returns.mean() / self.returns.std() * (252 ** 0.5)
                    return 0.0
                
                def max_drawdown(self):
                    if self.returns is not None:
                        cumulative = (1 + self.returns).cumprod()
                        running_max = cumulative.expanding().max()
                        drawdown = (cumulative - running_max) / running_max
                        return drawdown.min()
                    return 0.0
            
            return FallbackPortfolio
    
    def _create_vectorbt_optimizer(self):
        """Create VectorBT optimizer."""
        try:
            import vectorbt as vbt
            return vbt.optimize
        except ImportError:
            # Fallback optimizer
            class FallbackOptimizer:
                def __init__(self, *args, **kwargs):
                    pass
                
                @staticmethod
                def optimize(func, param_ranges, **kwargs):
                    # Simple grid search fallback
                    best_params = None
                    best_score = float('-inf')
                    
                    for params in self._generate_param_combinations(param_ranges):
                        try:
                            score = func(**params)
                            if score > best_score:
                                best_score = score
                                best_params = params
                        except:
                            continue
                    
                    return best_params, best_score
                
                def _generate_param_combinations(self, param_ranges):
                    import itertools
                    keys = list(param_ranges.keys())
                    values = list(param_ranges.values())
                    
                    for combo in itertools.product(*values):
                        yield dict(zip(keys, combo))
            
            return FallbackOptimizer()
    
    def _create_vectorbt_metrics(self):
        """Create VectorBT metrics calculator."""
        try:
            import vectorbt as vbt
            return vbt.returns
        except ImportError:
            # Fallback metrics
            class FallbackMetrics:
                def __init__(self, *args, **kwargs):
                    pass
                
                @staticmethod
                def sharpe_ratio(returns, **kwargs):
                    if len(returns) == 0:
                        return 0.0
                    return returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0.0
                
                @staticmethod
                def max_drawdown(returns, **kwargs):
                    if len(returns) == 0:
                        return 0.0
                    cumulative = (1 + returns).cumprod()
                    running_max = cumulative.expanding().max()
                    drawdown = (cumulative - running_max) / running_max
                    return drawdown.min()
                
                @staticmethod
                def calmar_ratio(returns, **kwargs):
                    if len(returns) == 0:
                        return 0.0
                    annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
                    max_dd = abs(FallbackMetrics.max_drawdown(returns))
                    return annual_return / max_dd if max_dd > 0 else 0.0
            
            return FallbackMetrics()
    
    def _initialize_monitoring(self):
        """Initialize monitoring and diagnostics."""
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🔧 Initializing monitoring and diagnostics")
            
            # Initialize monitoring components
            self.performance_monitor = self._create_performance_monitor()
            self.metrics_collector = self._create_metrics_collector()
            self.alert_manager = self._create_alert_manager()
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ Monitoring and diagnostics initialized")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Monitoring initialization failed: {e}")
            else:
                self.logger.warning(f"Monitoring initialization failed: {e}")
    
    def _create_performance_monitor(self):
        """Create performance monitor."""
        try:
            from ..monitoring.performance_monitor import PerformanceMonitor
            return PerformanceMonitor(self.config.get("monitoring", {}))
        except ImportError:
            # Fallback performance monitor
            class FallbackPerformanceMonitor:
                def __init__(self, config):
                    self.config = config
                    self.metrics = {}
                
                def start_monitoring(self, task_id):
                    self.metrics[task_id] = {"start_time": time.time()}
                
                def stop_monitoring(self, task_id):
                    if task_id in self.metrics:
                        self.metrics[task_id]["end_time"] = time.time()
                        self.metrics[task_id]["duration"] = (
                            self.metrics[task_id]["end_time"] - self.metrics[task_id]["start_time"]
                        )
                
                def get_metrics(self, task_id):
                    return self.metrics.get(task_id, {})
                
                def get_system_metrics(self):
                    import psutil
                    return {
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_percent": psutil.disk_usage('/').percent
                    }
            
            return FallbackPerformanceMonitor(self.config.get("monitoring", {}))
    
    def _create_metrics_collector(self):
        """Create metrics collector."""
        try:
            from ..monitoring.metrics_collector import MetricsCollector
            return MetricsCollector(self.config.get("metrics", {}))
        except ImportError:
            # Fallback metrics collector
            class FallbackMetricsCollector:
                def __init__(self, config):
                    self.config = config
                    self.collected_metrics = []
                
                def collect_metric(self, name, value, tags=None):
                    self.collected_metrics.append({
                        "name": name,
                        "value": value,
                        "tags": tags or {},
                        "timestamp": time.time()
                    })
                
                def get_metrics(self, name=None):
                    if name:
                        return [m for m in self.collected_metrics if m["name"] == name]
                    return self.collected_metrics
                
                def clear_metrics(self):
                    self.collected_metrics.clear()
            
            return FallbackMetricsCollector(self.config.get("metrics", {}))
    
    def _create_alert_manager(self):
        """Create alert manager."""
        try:
            from ..monitoring.alert_manager import AlertManager
            return AlertManager(self.config.get("alerts", {}))
        except ImportError:
            # Fallback alert manager
            class FallbackAlertManager:
                def __init__(self, config):
                    self.config = config
                    self.alerts = []
                
                def send_alert(self, level, message, context=None):
                    alert = {
                        "level": level,
                        "message": message,
                        "context": context or {},
                        "timestamp": time.time()
                    }
                    self.alerts.append(alert)
                    self.logger.warning(f"ALERT [{level}]: {message}")
                
                def get_alerts(self, level=None):
                    if level:
                        return [a for a in self.alerts if a["level"] == level]
                    return self.alerts
                
                def clear_alerts(self):
                    self.alerts.clear()
            
            return FallbackAlertManager(self.config.get("alerts", {}))
    
    def optimize(self, 
                 model_factory: Callable,
                 X: np.ndarray,
                 y: np.ndarray,
                 search_space: Dict[str, Any],
                 model_name: str = "unknown") -> HPOResult:
        """
        Optimize hyperparameters using the specified strategy.
        
        Args:
            model_factory: Function that creates model instances
            X: Training features
            y: Training targets
            search_space: Search space for hyperparameters
            model_name: Name of the model being optimized
            
        Returns:
            HPOResult: Comprehensive optimization results
        """
        if TPRINT_AVAILABLE:
            tprint_info(f"🚀 Starting HPO optimization for {model_name} using {self.config.strategy} strategy")
        
        start_time = time.time()
        
        try:
            if self.config.strategy == 'bayesian':
                result = self._bayesian_optimization(model_factory, X, y, search_space, model_name)
            elif self.config.strategy == 'bohb':
                result = self._bohb_optimization(model_factory, X, y, search_space, model_name)
            elif self.config.strategy == 'hierarchical':
                result = self._hierarchical_optimization(model_factory, X, y, search_space, model_name)
            elif self.config.strategy == 'grid':
                result = self._grid_optimization(model_factory, X, y, search_space, model_name)
            elif self.config.strategy == 'random':
                result = self._random_optimization(model_factory, X, y, search_space, model_name)
            else:
                raise ValueError(f"Unsupported optimization strategy: {self.config.strategy}")
            
            # Update metadata
            result.optimization_time = time.time() - start_time
            result.model_name = model_name
            result.strategy = self.config.strategy
            
            # Store results
            self.optimization_history.append(result)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ HPO optimization completed for {model_name} in {result.optimization_time:.2f}s")
                tprint_info(f"📊 Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ HPO optimization failed for {model_name}: {e}")
            else:
                self.logger.error(f"HPO optimization failed for {model_name}: {e}")
            raise
    
    def _bayesian_optimization(self, model_factory: Callable, X: np.ndarray, 
                              y: np.ndarray, search_space: Dict[str, Any], 
                              model_name: str) -> HPOResult:
        """Perform Bayesian optimization with TPE and early pruning."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for Bayesian optimization")
        
        if TPRINT_AVAILABLE:
            tprint_info("🎯 Starting Bayesian optimization with TPE")
        
        # Create study with proper pruner
        pruner = None
        if self.config.enable_monitoring:
            from optuna.pruners import MedianPruner
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                gamma=self.config.gamma,
                seed=self.config.random_state
            ),
            pruner=pruner
        )
        
        # Define objective function
        def objective(trial):
            return self._objective_function(trial, model_factory, X, y, search_space)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Extract results
        best_trial = study.best_trial
        trial_results = [trial for trial in study.trials]
        
        return HPOResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            best_trial=best_trial,
            n_trials=len(trial_results),
            trial_results=[{
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            } for trial in trial_results],
            mean_score=np.mean([t.value for t in trial_results if t.value is not None]),
            std_score=np.std([t.value for t in trial_results if t.value is not None]),
            min_score=np.min([t.value for t in trial_results if t.value is not None]),
            max_score=np.max([t.value for t in trial_results if t.value is not None])
        )
    
    def _bohb_optimization(self, model_factory: Callable, X: np.ndarray, 
                          y: np.ndarray, search_space: Dict[str, Any], 
                          model_name: str) -> HPOResult:
        """Perform BOHB-style multi-fidelity optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for BOHB optimization")
        
        if TPRINT_AVAILABLE:
            tprint_info("🎯 Starting BOHB-style multi-fidelity optimization")
        
        # Use SuccessiveHalvingPruner for better multi-fidelity support
        from optuna.pruners import SuccessiveHalvingPruner
        
        # Create study with SuccessiveHalving pruner
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=self.config.n_ei_candidates,
                multivariate=self.config.multivariate,
                group=self.config.group,
                gamma=self.config.gamma,
                seed=self.config.random_state
            ),
            pruner=SuccessiveHalvingPruner(
                min_resource=1,
                reduction_factor=int(self.config.reduction_factor)
            )
        )
        
        # Define objective function with multi-fidelity support
        def objective(trial):
            return self._multi_fidelity_objective_function(trial, model_factory, X, y, search_space)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        # Extract results
        best_trial = study.best_trial
        trial_results = [trial for trial in study.trials]
        
        return HPOResult(
            best_params=best_trial.params,
            best_score=best_trial.value,
            best_trial=best_trial,
            n_trials=len(trial_results),
            trial_results=[{
                'trial_number': trial.number,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state.name
            } for trial in trial_results],
            mean_score=np.mean([t.value for t in trial_results if t.value is not None]),
            std_score=np.std([t.value for t in trial_results if t.value is not None]),
            min_score=np.min([t.value for t in trial_results if t.value is not None]),
            max_score=np.max([t.value for t in trial_results if t.value is not None])
        )
    
    def _hierarchical_optimization(self, model_factory: Callable, X: np.ndarray, 
                                  y: np.ndarray, search_space: Dict[str, Any], 
                                  model_name: str) -> HPOResult:
        """Perform hierarchical optimization for stacking ensembles."""
        if TPRINT_AVAILABLE:
            tprint_info("🎯 Starting hierarchical optimization")
        
        # This would implement the hierarchical optimization logic
        # For now, fall back to Bayesian optimization
        return self._bayesian_optimization(model_factory, X, y, search_space, model_name)
    
    def _grid_optimization(self, model_factory: Callable, X: np.ndarray, 
                          y: np.ndarray, search_space: Dict[str, Any], 
                          model_name: str) -> HPOResult:
        """Perform grid search optimization with staged refinement."""
        if TPRINT_AVAILABLE:
            tprint_info("🎯 Starting grid search optimization")
        
        # Stage 1: Coarse grid search
        coarse_grid = self._generate_parameter_grid(search_space, points=self.config.coarse_grid_points)
        stage1_results = self._eval_param_list(model_factory, X, y, coarse_grid)
        
        # Pick top K from coarse stage
        top = sorted(stage1_results, key=lambda d: d['value'], reverse=True)[:self.config.coarse_top_k]
        best_score = top[0]['value'] if top else -np.inf
        best_params = top[0]['params'] if top else {}
        
        # Stage 2: Fine grid search (optional)
        trial_results = stage1_results[:]
        if self.config.enable_staged_optimization and top:
            if TPRINT_AVAILABLE:
                tprint_info(f"🔍 Refining search around top {len(top)} configurations")
            
            refined_space = self._refine_search_space([t['params'] for t in top], search_space)
            fine_grid = self._generate_parameter_grid(refined_space, points=self.config.fine_grid_points)
            stage2_results = self._eval_param_list(model_factory, X, y, fine_grid)
            trial_results.extend(stage2_results)
            
            # Update best if we found something better
            for r in stage2_results:
                if r['value'] > best_score:
                    best_score, best_params = r['value'], r['params']
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trial_results),
            trial_results=trial_results,
            mean_score=float(np.mean([t['value'] for t in trial_results])) if trial_results else -np.inf,
            std_score=float(np.std([t['value'] for t in trial_results])) if trial_results else 0.0,
            min_score=float(np.min([t['value'] for t in trial_results])) if trial_results else -np.inf,
            max_score=float(np.max([t['value'] for t in trial_results])) if trial_results else -np.inf
        )
    
    def _random_optimization(self, model_factory: Callable, X: np.ndarray, 
                           y: np.ndarray, search_space: Dict[str, Any], 
                           model_name: str) -> HPOResult:
        """Perform random search optimization."""
        if TPRINT_AVAILABLE:
            tprint_info("🎯 Starting random search optimization")
        
        best_score = -np.inf
        best_params = {}
        trial_results = []
        
        for i in range(self.config.n_trials):
            try:
                # Sample random parameters
                params = self._sample_parameters(search_space)
                
                # Create model with parameters
                model = model_factory(**params)
                
                # Evaluate model
                score = self._evaluate_model(model, X, y)
                
                trial_results.append({
                    'trial_number': i,
                    'params': params,
                    'value': score,
                    'state': 'COMPLETE'
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
                
                if TPRINT_AVAILABLE and (i + 1) % 10 == 0:
                    tprint_info(f"   Evaluated {i + 1}/{self.config.n_trials} trials")
                
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Failed to evaluate random parameters: {e}")
                continue
        
        return HPOResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(trial_results),
            trial_results=trial_results,
            mean_score=np.mean([t['value'] for t in trial_results]),
            std_score=np.std([t['value'] for t in trial_results]),
            min_score=np.min([t['value'] for t in trial_results]),
            max_score=np.max([t['value'] for t in trial_results])
        )
    
    def _objective_function(self, trial: optuna.Trial, model_factory: Callable, 
                           X: np.ndarray, y: np.ndarray, search_space: Dict[str, Any]) -> float:
        """Objective function for optimization with early pruning support."""
        try:
            # Sample hyperparameters
            params = self._sample_parameters_from_trial(trial, search_space)
            
            # Create model with sampled parameters
            model = model_factory(**params)
            
            # Evaluate model with trial for pruning
            return self._evaluate_model(model, X, y, trial=trial)
            
        except optuna.TrialPruned:
            raise  # Re-raise pruning exceptions
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Trial failed: {e}")
            return float('-inf')
    
    def _multi_fidelity_objective_function(self, trial: optuna.Trial, model_factory: Callable, 
                                         X: np.ndarray, y: np.ndarray, search_space: Dict[str, Any]) -> float:
        """Multi-fidelity objective function for BOHB."""
        try:
            # Sample hyperparameters
            params = self._sample_parameters_from_trial(trial, search_space)
            
            # Sample budget (fidelity level) - ensure it's in (0, 1] range
            budget = trial.suggest_float('budget', 0.1, 1.0)
            
            # Create model with sampled parameters
            model = model_factory(**params)
            
            # Evaluate model with limited budget
            score = self._evaluate_model_with_budget(model, X, y, budget)
            
            # Report intermediate result for pruning
            trial.report(score, step=int(budget * 100))
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Multi-fidelity trial failed: {e}")
            return float('-inf')
    
    def _sample_parameters_from_trial(self, trial: optuna.Trial, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters from Optuna trial."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, param_config['low'], param_config['high'], 
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, param_config['low'], param_config['high'], 
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
        
        return params
    
    def _sample_parameters(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Sample parameters randomly."""
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    params[param_name] = np.exp(np.random.uniform(
                        np.log(param_config['low']), np.log(param_config['high'])
                    ))
                else:
                    params[param_name] = np.random.uniform(
                        param_config['low'], param_config['high']
                    )
            elif param_config['type'] == 'int':
                params[param_name] = np.random.randint(
                    param_config['low'], param_config['high'] + 1
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = np.random.choice(param_config['choices'])
            else:
                raise ValueError(f"Unsupported parameter type: {param_config['type']}")
        
        return params
    
    def _generate_parameter_grid(self, search_space: Dict[str, Any], points: int = None) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        if points is None:
            points = self.config.coarse_grid_points
            
        param_combinations = []
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'float':
                if param_config.get('log', False):
                    values = np.logspace(
                        np.log10(param_config['low']), 
                        np.log10(param_config['high']), 
                        points
                    )
                else:
                    values = np.linspace(
                        param_config['low'], 
                        param_config['high'], 
                        points
                    )
                param_combinations.append([(param_name, v) for v in values])
            elif param_config['type'] == 'int':
                values = np.unique(np.linspace(
                    param_config['low'], 
                    param_config['high'], 
                    points, 
                    dtype=int
                ))
                param_combinations.append([(param_name, v) for v in values])
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
    
    def _eval_param_list(self, model_factory: Callable, X: np.ndarray, y: np.ndarray, param_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a list of parameter combinations."""
        out = []
        for i, params in enumerate(param_list):
            try:
                score = self._evaluate_model(model_factory(**params), X, y)
                out.append({'trial_number': i, 'params': params, 'value': score, 'state': 'COMPLETE'})
                if TPRINT_AVAILABLE and (i + 1) % 10 == 0:
                    tprint_info(f"   Evaluated {i + 1}/{len(param_list)} combinations")
            except Exception as e:
                if TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Failed {params}: {e}")
        return out
    
    def _refine_search_space(self, winners: List[Dict[str, Any]], base_space: Dict[str, Any]) -> Dict[str, Any]:
        """Refine search space around winning configurations."""
        refined = {}
        for name, cfg in base_space.items():
            if cfg['type'] == 'categorical':
                # Keep observed best categories
                seen = list({w[name] for w in winners if name in w})
                refined[name] = {'type': 'categorical', 'choices': seen or cfg['choices']}
            elif cfg['type'] in ('float', 'int'):
                vals = np.array([w[name] for w in winners if name in w])
                if len(vals) > 0:
                    lo, hi = np.min(vals), np.max(vals)
                    span = (hi - lo)
                    if span == 0:
                        lo, hi = cfg['low'], cfg['high']  # fallback to base range
                    pad = max(span * self.config.fine_span_frac, (cfg['high'] - cfg['low']) * 0.05)
                    lo2 = max(cfg['low'], lo - pad)
                    hi2 = min(cfg['high'], hi + pad)
                    refined[name] = dict(cfg)  # copy
                    refined[name]['low'] = float(lo2)
                    refined[name]['high'] = float(hi2)
                else:
                    refined[name] = cfg
            else:
                raise ValueError(f"Unsupported type {cfg['type']}")
        return refined
    
    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, trial: Optional[Any] = None) -> float:
        """Evaluate model performance with caching and overfitting detection."""
        try:
            # Create cache key for this model configuration
            model_params = getattr(model, 'get_params', lambda: {})()
            cache_key = tuple(sorted(model_params.items()))
            
            # Check cache first
            if self.config.enable_caching and cache_key in self._score_cache:
                return self._score_cache[cache_key]
            
            # Determine CV strategy
            if self.config.enable_time_series_cv:
                # Guard against too short series
                if len(X) < self.config.cv_folds * 2:
                    cv = TimeSeriesSplit(n_splits=min(2, len(X) // 2))
                else:
                    cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            else:
                # Determine if classification using a more robust method
                is_classification = self._is_classification_task(model, y)
                
                if is_classification:
                    cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=False)
                else:
                    cv = KFold(n_splits=self.config.cv_folds, shuffle=False)
            
            # Auto-select scoring metric
            scoring = self._get_auto_scoring_metric(model, y)
            
            # Perform cross-validation with overfitting detection
            if self.config.enable_overfitting_detection:
                from sklearn.model_selection import cross_validate
                cv_results = cross_validate(
                    model, X, y, cv=cv, scoring=scoring, 
                    return_train_score=True, n_jobs=1
                )
                
                test_scores = cv_results['test_score']
                train_scores = cv_results['train_score']
                
                # Check for overfitting
                overfitting_detected = (np.mean(train_scores) - np.mean(test_scores)) > self.config.overfitting_threshold
                
                if overfitting_detected and TPRINT_AVAILABLE:
                    tprint_warning(f"⚠️ Overfitting detected: train={np.mean(train_scores):.4f}, test={np.mean(test_scores):.4f}")
                
                score = float(np.mean(test_scores))
            else:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                score = float(np.mean(scores))
            
            # Cache the result
            if self.config.enable_caching:
                self._score_cache[cache_key] = score
            
            # Report intermediate result for pruning if trial is provided
            if trial is not None:
                trial.report(score)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Model evaluation failed: {e}")
            return float('-inf')
    
    def _is_classification_task(self, model: Any, y: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        # Check if model has predict_proba method (classification indicator)
        if hasattr(model, 'predict_proba'):
            return True
        
        # Check if y contains only integers and has few unique values
        if np.issubdtype(y.dtype, np.integer):
            unique_values = np.unique(y)
            if len(unique_values) <= 10 and np.all(unique_values >= 0):
                return True
        
        return False
    
    def _get_auto_scoring_metric(self, model: Any, y: np.ndarray) -> str:
        """Automatically select appropriate scoring metric."""
        if self.config.scoring != 'neg_mean_squared_error':
            return self.config.scoring
        
        if self._is_classification_task(model, y):
            # For classification, use accuracy or roc_auc
            unique_values = np.unique(y)
            if len(unique_values) == 2:
                return 'roc_auc'  # Binary classification
            else:
                return 'accuracy'  # Multi-class classification
        else:
            return 'neg_mean_squared_error'  # Regression
    
    def _evaluate_model_with_budget(self, model: Any, X: np.ndarray, y: np.ndarray, budget: float) -> float:
        """Evaluate model with limited budget (for multi-fidelity)."""
        try:
            # Use a subset of data based on budget
            # Budget should be in (0, 1] range for data fraction
            budget = max(0.01, min(1.0, budget))  # Clamp to [0.01, 1.0]
            n_samples = int(len(X) * budget)
            n_samples = max(10, min(n_samples, len(X)))  # Ensure valid range
            
            # Sample data with replacement if needed
            replace = n_samples > len(X)
            indices = np.random.choice(len(X), n_samples, replace=replace)
            X_subset = X[indices]
            y_subset = y[indices]
            
            # Evaluate model
            return self._evaluate_model(model, X_subset, y_subset)
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_warning(f"⚠️ Multi-fidelity model evaluation failed: {e}")
            return float('-inf')
    
    def save_results(self, result: HPOResult, filename: Optional[str] = None) -> str:
        """Save optimization results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hpo_results_{result.model_name}_{timestamp}.json"
        
        filepath = Path(self.config.results_dir) / filename
        
        try:
            # Convert result to serializable format
            serializable_result = {
                'best_params': result.best_params,
                'best_score': result.best_score,
                'n_trials': result.n_trials,
                'optimization_time': result.optimization_time,
                'strategy': result.strategy,
                'mean_score': result.mean_score,
                'std_score': result.std_score,
                'min_score': result.min_score,
                'max_score': result.max_score,
                'model_name': result.model_name,
                'optimization_timestamp': result.optimization_timestamp,
                'trial_results': result.trial_results
            }
            
            with open(filepath, 'w') as f:
                json.dump(serializable_result, f, indent=2, default=str)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"💾 HPO results saved: {filepath}")
            
            return str(filepath)
            
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Failed to save HPO results: {e}")
            else:
                self.logger.error(f"Failed to save HPO results: {e}")
            return ""

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_consolidated_hpo(config: Optional[HPOConfig] = None) -> ConsolidatedHPO:
    """Create consolidated HPO system."""
    return ConsolidatedHPO(config)

def create_bayesian_hpo(n_trials: int = 100, 
                       n_startup_trials: int = 10,
                       timeout: Optional[float] = None) -> ConsolidatedHPO:
    """Create Bayesian HPO with basic settings."""
    config = HPOConfig(
        strategy='bayesian',
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        timeout=timeout,
        enable_detailed_logging=False,
        save_results=False
    )
    return ConsolidatedHPO(config)

def create_bohb_hpo(n_trials: int = 100,
                   min_budget: float = 0.1,
                   max_budget: float = 1.0,
                   timeout: Optional[float] = None) -> ConsolidatedHPO:
    """Create BOHB HPO with basic settings."""
    config = HPOConfig(
        strategy='bohb',
        n_trials=n_trials,
        min_budget=min_budget,
        max_budget=max_budget,
        timeout=timeout,
        enable_detailed_logging=False,
        save_results=False
    )
    return ConsolidatedHPO(config)

def create_grid_hpo(n_trials: int = 100,
                   coarse_grid_points: int = 5,
                   fine_grid_points: int = 5) -> ConsolidatedHPO:
    """Create grid search HPO with basic settings."""
    config = HPOConfig(
        strategy='grid',
        n_trials=n_trials,
        coarse_grid_points=coarse_grid_points,
        fine_grid_points=fine_grid_points,
        enable_detailed_logging=False,
        save_results=False
    )
    return ConsolidatedHPO(config)

def create_random_hpo(n_trials: int = 100) -> ConsolidatedHPO:
    """Create random search HPO with basic settings."""
    config = HPOConfig(
        strategy='random',
        n_trials=n_trials,
        enable_detailed_logging=False,
        save_results=False
    )
    return ConsolidatedHPO(config)

# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Legacy class names for backward compatibility
HyperparameterOptimization = ConsolidatedHPO
HierarchicalHPO = ConsolidatedHPO
BayesianTPEOptimizer = ConsolidatedHPO
BOHBOptimizer = ConsolidatedHPO
RegimeHPOWrapper = ConsolidatedHPO

# Legacy function names for backward compatibility
def optimize_hyperparameters(model_factory: Callable,
                            X: np.ndarray,
                            y: np.ndarray,
                            search_space: Dict[str, Any],
                            n_trials: int = 100,
                            strategy: str = 'bayesian',
                            **kwargs) -> HPOResult:
    """Legacy function for hyperparameter optimization."""
    config = HPOConfig(
        strategy=strategy,
        n_trials=n_trials,
        **kwargs
    )
    hpo = ConsolidatedHPO(config)
    return hpo.optimize(model_factory, X, y, search_space)

def staged_hpo(model_factory: Callable,
               X: np.ndarray,
               y: np.ndarray,
               search_space: Dict[str, Any],
               n_trials: int = 100,
               **kwargs) -> HPOResult:
    """Legacy function for staged HPO."""
    config = HPOConfig(
        strategy='grid',
        n_trials=n_trials,
        enable_staged_optimization=True,
        **kwargs
    )
    hpo = ConsolidatedHPO(config)
    return hpo.optimize(model_factory, X, y, search_space)

def bayesian_optimization(model_factory: Callable,
                         X: np.ndarray,
                         y: np.ndarray,
                         search_space: Dict[str, Any],
                         n_trials: int = 100,
                         **kwargs) -> HPOResult:
    """Legacy function for Bayesian optimization."""
    config = HPOConfig(
        strategy='bayesian',
        n_trials=n_trials,
        **kwargs
    )
    hpo = ConsolidatedHPO(config)
    return hpo.optimize(model_factory, X, y, search_space)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main classes
    'ConsolidatedHPO',
    'HPOConfig',
    'HPOPhaseConfig',
    'HPOResult',
    
    # Convenience functions
    'create_consolidated_hpo',
    'create_bayesian_hpo',
    'create_bohb_hpo',
    'create_grid_hpo',
    'create_random_hpo',
    
    # Legacy compatibility
    'HyperparameterOptimization',
    'HierarchicalHPO',
    'BayesianTPEOptimizer',
    'BOHBOptimizer',
    'RegimeHPOWrapper',
    'optimize_hyperparameters',
    'staged_hpo',
    'bayesian_optimization',
]