"""
Refactored HPO system with backward compatibility.

This module provides the new, refactored HPO system while maintaining
full backward compatibility with the existing ConsolidatedHPO interface.
"""

from typing import Any, Dict, List, Optional, Callable, Union
import time
import logging
from dataclasses import dataclass

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a mock numpy for basic functionality
    class MockNumpy:
        def __getattr__(self, name):
            raise ImportError("NumPy is required for optimization functionality")
    np = MockNumpy()

# Import new components
from .core import HPOEngine, OptimizationStrategy, BayesianStrategy, GridStrategy, RandomStrategy, BOHBStrategy
from .core.monitoring import OptimizationMonitor
from .core.caching import OptimizationCache
from .core.pruner_factory import PrunerFactory
from .validation import HPOConfig, validate_hpo_config, validate_search_space, AresExecutionMode
from .exceptions import OptimizationError, ConfigurationError
from .results import HPOResult

# Import legacy components for backward compatibility
from .consolidated_hpo import ConsolidatedHPO as LegacyConsolidatedHPO

# Try to import hardware optimization
try:
    from ..hardware.optimization_decorators import (
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

# Try to import tprint
try:
    from ..tprint import (
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

logger = logging.getLogger(__name__)


class ConsolidatedHPO:
    """
    Refactored consolidated HPO system with backward compatibility.
    
    This class provides the same interface as the original ConsolidatedHPO
    but uses the new, refactored architecture internally.
    """
    
    def __init__(self, config: Optional[Union[HPOConfig, Dict[str, Any]]] = None):
        """Initialize consolidated HPO system."""
        # Convert legacy config format if needed
        if config is None:
            config = HPOConfig()
        elif isinstance(config, dict):
            # Handle legacy config format
            config = self._convert_legacy_config(config)
        
        # Initialize the new HPO engine
        self.engine = HPOEngine(config)
        
        # Store config for backward compatibility
        self.config = config
        
        # Initialize logging
        self.logger = logger.getChild('ConsolidatedHPO')
        
        # Backward compatibility attributes
        self.optimization_history = []
        self.active_studies = {}
        self.trial_results = {}
        
        if TPRINT_AVAILABLE:
            tprint_success("✅ Refactored Consolidated HPO system initialized")
    
    def _convert_legacy_config(self, config_dict: Dict[str, Any]) -> HPOConfig:
        """Convert legacy config format to new HPOConfig."""
        # Map legacy config keys to new format
        legacy_mapping = {
            'n_trials': 'n_trials',
            'timeout': 'timeout',
            'random_state': 'random_state',
            'strategy': 'strategy',
            'ares_execution_mode': 'ares_execution_mode',
            'enable_mode_scaling': 'enable_mode_scaling',
            'auto_detect_mode': 'auto_detect_mode',
            'n_startup_trials': 'n_startup_trials',
            'n_ei_candidates': 'n_ei_candidates',
            'multivariate': 'multivariate',
            'group': 'group',
            'min_budget': 'min_budget',
            'max_budget': 'max_budget',
            'reduction_factor': 'reduction_factor',
            'n_brackets': 'n_brackets',
            'enable_staged_optimization': 'enable_staged_optimization',
            'coarse_grid_points': 'coarse_grid_points',
            'fine_grid_points': 'fine_grid_points',
            'coarse_grid_trials': 'coarse_grid_trials',
            'fine_grid_trials': 'fine_grid_trials',
            'tpe_trials': 'tpe_trials',
            'enable_hardware_optimization': 'enable_hardware_optimization',
            'enable_vectorbt': 'enable_vectorbt',
            'enable_parallel': 'enable_parallel',
            'max_workers': 'max_workers',
            'enable_monitoring': 'enable_monitoring',
            'enable_diagnostics': 'enable_diagnostics',
            'enable_overfitting_detection': 'enable_overfitting_detection',
            'cv_folds': 'cv_folds',
            'enable_time_series_cv': 'enable_time_series_cv',
            'scoring': 'scoring',
            'enable_caching': 'enable_caching',
            'cache_dir': 'cache_dir',
            'save_results': 'save_results',
            'results_dir': 'results_dir',
            'enable_detailed_logging': 'enable_detailed_logging',
            'overfitting_threshold': 'overfitting_threshold'
        }
        
        # Convert strategy string to enum
        if 'strategy' in config_dict:
            strategy_map = {
                'bayesian': 'bayesian',
                'bohb': 'bohb',
                'grid': 'grid',
                'random': 'random',
                'hierarchical': 'hierarchical'
            }
            config_dict['strategy'] = strategy_map.get(config_dict['strategy'], 'bayesian')
        
        # Convert ares execution mode
        if 'ares_execution_mode' in config_dict:
            mode_map = {
                'light': 'light',
                'blank': 'blank',
                'full': 'full'
            }
            config_dict['ares_execution_mode'] = mode_map.get(config_dict['ares_execution_mode'], 'full')
        
        # Create new config with mapped values
        new_config = {}
        for legacy_key, new_key in legacy_mapping.items():
            if legacy_key in config_dict:
                new_config[new_key] = config_dict[legacy_key]
        
        return HPOConfig(**new_config)
    
    def optimize(self, 
                 model_factory: Callable,
                 X: Any,  # Changed from np.ndarray to Any for compatibility
                 y: Any,  # Changed from np.ndarray to Any for compatibility
                 search_space: Dict[str, Any],
                 model_name: str = "unknown") -> HPOResult:
        """
        Optimize hyperparameters using the specified strategy.
        
        This method maintains the exact same interface as the original
        ConsolidatedHPO.optimize method.
        """
        if TPRINT_AVAILABLE:
            tprint_info(f"🚀 Starting HPO optimization for {model_name} using {self.config.strategy.value} strategy")
        
        # Use the new engine for optimization
        result = self.engine.optimize(
            model_factory=model_factory,
            X=X,
            y=y,
            search_space=search_space,
            model_name=model_name
        )
        
        # Update backward compatibility attributes
        self.optimization_history.append(result)
        
        if TPRINT_AVAILABLE:
            tprint_success(f"✅ HPO optimization completed for {model_name} in {result.optimization_time:.2f}s")
            tprint_info(f"📊 Best score: {result.best_score:.4f}")
        
        return result
    
    def save_results(self, result: HPOResult, filename: Optional[str] = None) -> str:
        """Save optimization results to file."""
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hpo_results_{result.model_name}_{timestamp}.json"
        
        filepath = f"{self.config.results_dir}/{filename}"
        result.save(filepath)
        
        if TPRINT_AVAILABLE:
            tprint_success(f"💾 HPO results saved: {filepath}")
        
        return filepath
    
    # Backward compatibility methods
    def _bayesian_optimization(self, model_factory, X, y, search_space, model_name):
        """Legacy method for backward compatibility."""
        return self.optimize(model_factory, X, y, search_space, model_name)
    
    def _bohb_optimization(self, model_factory, X, y, search_space, model_name):
        """Legacy method for backward compatibility."""
        return self.optimize(model_factory, X, y, search_space, model_name)
    
    def _grid_optimization(self, model_factory, X, y, search_space, model_name):
        """Legacy method for backward compatibility."""
        return self.optimize(model_factory, X, y, search_space, model_name)
    
    def _random_optimization(self, model_factory, X, y, search_space, model_name):
        """Legacy method for backward compatibility."""
        return self.optimize(model_factory, X, y, search_space, model_name)
    
    def _hierarchical_optimization(self, model_factory, X, y, search_space, model_name):
        """Legacy method for backward compatibility."""
        return self.optimize(model_factory, X, y, search_space, model_name)
    
    # Additional backward compatibility properties
    @property
    def logger(self):
        """Backward compatibility logger property."""
        return self.engine.logger
    
    def get_optimization_history(self):
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def clear_history(self):
        """Clear optimization history."""
        self.optimization_history.clear()
        self.engine.clear_history()


# Backward compatibility aliases and functions
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


def create_ares_mode_hpo(
    ares_mode: str = 'full',
    strategy: str = 'bayesian',
    n_trials: int = 100,
    **kwargs
) -> ConsolidatedHPO:
    """Create HPO optimized for specific Ares execution mode."""
    config = HPOConfig(
        strategy=strategy,
        n_trials=n_trials,
        ares_execution_mode=ares_mode,
        enable_mode_scaling=True,
        auto_detect_mode=False,
        **kwargs
    )
    return ConsolidatedHPO(config)


def create_auto_mode_hpo(
    strategy: str = 'bayesian',
    n_trials: int = 100,
    **kwargs
) -> ConsolidatedHPO:
    """Create HPO with automatic Ares mode detection."""
    config = HPOConfig(
        strategy=strategy,
        n_trials=n_trials,
        auto_detect_mode=True,
        enable_mode_scaling=True,
        **kwargs
    )
    return ConsolidatedHPO(config)


# Legacy compatibility aliases
HyperparameterOptimization = ConsolidatedHPO
HierarchicalHPO = ConsolidatedHPO
BayesianTPEOptimizer = ConsolidatedHPO
BOHBOptimizer = ConsolidatedHPO
RegimeHPOWrapper = ConsolidatedHPO


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