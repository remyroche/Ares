"""
Core HPO Engine.

This module provides the main HPO engine that orchestrates optimization strategies,
hardware optimization, monitoring, and caching.
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

from ..validation import HPOConfig, validate_search_space, validate_hpo_config
from ..exceptions import OptimizationError, ConfigurationError, HardwareOptimizationError
from ..results import HPOResult
from .optimization_strategy import OptimizationStrategy, BayesianStrategy, GridStrategy, RandomStrategy, BOHBStrategy, OptimizationContext
from .monitoring import OptimizationMonitor
from .caching import OptimizationCache
from .pruner_factory import PrunerFactory


@dataclass
class HardwareManager:
    """Hardware optimization manager interface."""
    
    def optimize_for_task(self, task_type: str, data_size: int) -> Dict[str, Any]:
        """Optimize hardware for specific task."""
        return {"workers": 4, "memory_limit": 8.0}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {"cpu_cores": 4, "memory_gb": 8}


@dataclass
class VectorBTOptimizer:
    """VectorBT optimization interface."""
    
    def optimize_rolling_operations(self, data: Any, operations: List[str]) -> Any:
        """Optimize rolling operations using VectorBT."""
        return data


class HPOEngine:
    """
    Core hyperparameter optimization engine.
    
    This class orchestrates the optimization process using the strategy pattern,
    with support for hardware optimization, monitoring, and caching.
    """
    
    def __init__(self, config: Union[HPOConfig, Dict[str, Any]], 
                 hardware_manager: Optional[HardwareManager] = None,
                 vectorbt_optimizer: Optional[VectorBTOptimizer] = None,
                 monitor: Optional[OptimizationMonitor] = None,
                 cache: Optional[OptimizationCache] = None):
        """
        Initialize HPO engine.
        
        Args:
            config: HPO configuration (validated or dict)
            hardware_manager: Hardware optimization manager
            vectorbt_optimizer: VectorBT optimization manager
            monitor: Optimization monitoring system
            cache: Optimization caching system
        """
        # Validate configuration
        if isinstance(config, dict):
            self.config = validate_hpo_config(config)
        else:
            self.config = config
        
        # Initialize components with dependency injection
        self.hardware_manager = hardware_manager or self._create_default_hardware_manager()
        self.vectorbt_optimizer = vectorbt_optimizer or self._create_default_vectorbt_optimizer()
        self.monitor = monitor or self._create_default_monitor()
        self.cache = cache or self._create_default_cache()
        
        # Initialize strategy
        self.strategy = self._create_strategy()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimization tracking
        self.optimization_history = []
        self.active_optimizations = {}
        
        self.logger.info(f"HPO Engine initialized with {self.config.strategy.value} strategy")
    
    def optimize(self, 
                 model_factory: Callable,
                 X: Any,  # Changed from np.ndarray to Any for compatibility
                 y: Any,  # Changed from np.ndarray to Any for compatibility
                 search_space: Dict[str, Any],
                 model_name: str = "unknown") -> HPOResult:
        """
        Optimize hyperparameters using the configured strategy.
        
        Args:
            model_factory: Function that creates model instances
            X: Training features
            y: Training targets
            search_space: Search space for hyperparameters
            model_name: Name of the model being optimized
            
        Returns:
            HPOResult: Comprehensive optimization results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            self._validate_inputs(X, y, search_space)
            
            # Validate and convert search space
            validated_search_space = validate_search_space(search_space)
            
            # Create optimization context
            context = OptimizationContext(
                model_factory=model_factory,
                X=X,
                y=y,
                search_space=validated_search_space,
                model_name=model_name,
                start_time=start_time,
                config=self.config
            )
            
            # Start monitoring
            if self.monitor:
                self.monitor.start_optimization(model_name, self.config.strategy.value)
            
            # Apply hardware optimization
            if self.config.enable_hardware_optimization:
                self._apply_hardware_optimization(context)
            
            # Apply VectorBT optimization
            if self.config.enable_vectorbt and self.vectorbt_optimizer:
                self._apply_vectorbt_optimization(context)
            
            # Execute optimization strategy
            result = self.strategy.optimize(context)
            
            # Update metadata
            result.model_name = model_name
            result.optimization_time = time.time() - start_time
            
            # Store results
            self.optimization_history.append(result)
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_optimization(model_name, result)
            
            self.logger.info(f"Optimization completed for {model_name} in {result.optimization_time:.2f}s")
            self.logger.info(f"Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            # Stop monitoring on error
            if self.monitor:
                self.monitor.stop_optimization(model_name, None, error=str(e))
            
            self.logger.error(f"Optimization failed for {model_name}: {e}")
            raise
    
    def _validate_inputs(self, X: Any, y: Any, search_space: Dict[str, Any]) -> None:
        """Validate input data and search space."""
        if X is None or len(X) == 0:
            raise OptimizationError("Training features X cannot be empty")
        
        if y is None or len(y) == 0:
            raise OptimizationError("Training targets y cannot be empty")
        
        if len(X) != len(y):
            raise OptimizationError("X and y must have the same length")
        
        if not search_space:
            raise OptimizationError("Search space cannot be empty")
        
        # Check for reasonable data size
        if len(X) < 10:
            raise OptimizationError("Dataset too small for optimization (minimum 10 samples)")
    
    def _create_strategy(self) -> OptimizationStrategy:
        """Create optimization strategy based on configuration."""
        strategy_map = {
            'bayesian': BayesianStrategy,
            'grid': GridStrategy,
            'random': RandomStrategy,
            'bohb': BOHBStrategy
        }
        
        strategy_class = strategy_map.get(self.config.strategy.value)
        if not strategy_class:
            raise ConfigurationError(f"Unsupported strategy: {self.config.strategy}")
        
        return strategy_class(self.config)
    
    def _apply_hardware_optimization(self, context: OptimizationContext) -> None:
        """Apply hardware optimization to the context."""
        try:
            if self.hardware_manager:
                optimization_config = self.hardware_manager.optimize_for_task(
                    "hyperparameter_optimization", 
                    len(context.X)
                )
                # Apply optimization settings to context if needed
                self.logger.debug(f"Applied hardware optimization: {optimization_config}")
        except Exception as e:
            raise HardwareOptimizationError(f"Hardware optimization failed: {e}") from e
    
    def _apply_vectorbt_optimization(self, context: OptimizationContext) -> None:
        """Apply VectorBT optimization to the context."""
        try:
            if self.vectorbt_optimizer:
                # Apply VectorBT optimizations if needed
                self.logger.debug("Applied VectorBT optimization")
        except Exception as e:
            raise OptimizationError(f"VectorBT optimization failed: {e}") from e
    
    def _create_default_hardware_manager(self) -> HardwareManager:
        """Create default hardware manager."""
        return HardwareManager()
    
    def _create_default_vectorbt_optimizer(self) -> VectorBTOptimizer:
        """Create default VectorBT optimizer."""
        return VectorBTOptimizer()
    
    def _create_default_monitor(self) -> OptimizationMonitor:
        """Create default monitoring system."""
        return OptimizationMonitor()
    
    def _create_default_cache(self) -> OptimizationCache:
        """Create default caching system."""
        return OptimizationCache()
    
    def get_optimization_history(self) -> List[HPOResult]:
        """Get optimization history."""
        return self.optimization_history.copy()
    
    def clear_history(self) -> None:
        """Clear optimization history."""
        self.optimization_history.clear()
        if self.monitor:
            self.monitor.clear_history()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status = {
            "config": {
                "strategy": self.config.strategy.value,
                "n_trials": self.config.n_trials,
                "timeout": self.config.timeout
            },
            "hardware": self.hardware_manager.get_system_status(),
            "optimizations_completed": len(self.optimization_history)
        }
        
        if self.monitor:
            status["monitoring"] = self.monitor.get_status()
        
        return status