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

# Import tprint functions
try:
    from ...tprint import (
        tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
        tprint_success, tprint_performance, tprint_timer, tprint_data_preview,
        tprint_data_format, LogLevel, TPrintConfig
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(*args, **kwargs): pass
    def tprint_warning(*args, **kwargs): pass
    def tprint_success(*args, **kwargs): pass
    def tprint_error(*args, **kwargs): pass
    def tprint_debug(*args, **kwargs): pass
    def tprint_performance(*args, **kwargs): pass
    def tprint_data_preview(*args, **kwargs): pass
    def tprint_data_format(*args, **kwargs): pass

from ..validation import HPOConfig, validate_search_space, validate_hpo_config
from ..exceptions import OptimizationError, ConfigurationError, HardwareOptimizationError
from ..results import HPOResult
from .optimization_strategy import OptimizationStrategy, BayesianStrategy, GridStrategy, RandomStrategy, BOHBStrategy, OptimizationContext
from .monitoring import OptimizationMonitor
from .caching import OptimizationCache
from .pruner_factory import PrunerFactory

# Import enhanced features
try:
    from ..enhanced_early_stopping_integration import EarlyStoppingIntegration, create_early_stopping_integration
    from ..warm_starting_system import WarmStartManager, create_warm_start_manager
    from ..multi_objective_optimizer import MultiObjectiveOptimizer, create_multi_objective_optimizer
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False
    def create_early_stopping_integration(): return None
    def create_warm_start_manager(): return None
    def create_multi_objective_optimizer(): return None


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
                 cache: Optional[OptimizationCache] = None,
                 early_stopping_integration: Optional[Any] = None,
                 warm_start_manager: Optional[Any] = None,
                 multi_objective_optimizer: Optional[Any] = None):
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
        
        # Initialize enhanced components
        self.early_stopping_integration = early_stopping_integration
        self.warm_start_manager = warm_start_manager
        self.multi_objective_optimizer = multi_objective_optimizer
        
        # Initialize strategy
        self.strategy = self._create_strategy()
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Optimization tracking
        self.optimization_history = []
        self.active_optimizations = {}
        self.current_run_id = None
        
        if TPRINT_AVAILABLE:
            tprint_success(f"🚀 HPO Engine initialized with {self.config.strategy.value} strategy")
            tprint_info(f"📊 Configuration: {self.config.n_trials} trials, timeout: {self.config.timeout}s")
            tprint_info(f"🔧 Hardware optimization: {'enabled' if self.config.enable_hardware_optimization else 'disabled'}")
            tprint_info(f"⚡ VectorBT optimization: {'enabled' if self.config.enable_vectorbt else 'disabled'}")
        else:
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
        
        if TPRINT_AVAILABLE:
            tprint_info(f"🎯 Starting optimization for {model_name}")
            tprint_data_preview(X, f"{model_name}_features", max_rows=5)
            tprint_data_preview(y, f"{model_name}_targets", max_rows=5)
            tprint_data_format(search_space, f"{model_name}_search_space")
        
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
                self.current_run_id = self.monitor.start_optimization(model_name, self.config.strategy.value)
                if TPRINT_AVAILABLE:
                    tprint_info(f"📊 Monitoring started for run {self.current_run_id}")
            
            # Apply hardware optimization
            if self.config.enable_hardware_optimization:
                if TPRINT_AVAILABLE:
                    tprint_info("🔧 Applying hardware optimization...")
                self._apply_hardware_optimization(context)
            
            # Apply VectorBT optimization
            if self.config.enable_vectorbt and self.vectorbt_optimizer:
                if TPRINT_AVAILABLE:
                    tprint_info("⚡ Applying VectorBT optimization...")
                self._apply_vectorbt_optimization(context)
            
            # Execute optimization strategy
            if TPRINT_AVAILABLE:
                tprint_info(f"🎯 Executing {self.config.strategy.value} optimization strategy...")
            result = self.strategy.optimize(context)
            
            # Update metadata
            result.model_name = model_name
            result.optimization_time = time.time() - start_time
            
            # Store results
            self.optimization_history.append(result)
            
            # Stop monitoring
            if self.monitor and self.current_run_id:
                self.monitor.stop_optimization(self.current_run_id, result)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Optimization completed for {model_name} in {result.optimization_time:.2f}s")
                tprint_success(f"🏆 Best score: {result.best_score:.4f}")
                tprint_info(f"📈 Trials completed: {result.n_trials}")
                tprint_data_preview(result.best_params, f"{model_name}_best_params")
            else:
                self.logger.info(f"Optimization completed for {model_name} in {result.optimization_time:.2f}s")
                self.logger.info(f"Best score: {result.best_score:.4f}")
            
            return result
            
        except Exception as e:
            # Stop monitoring on error
            if self.monitor and self.current_run_id:
                self.monitor.stop_optimization(self.current_run_id, None, error=str(e))
            
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Optimization failed for {model_name}: {e}")
            else:
                self.logger.error(f"Optimization failed for {model_name}: {e}")
            raise
    
    def _validate_inputs(self, X: Any, y: Any, search_space: Dict[str, Any]) -> None:
        """Validate input data and search space."""
        # Check for None inputs
        if X is None:
            raise OptimizationError("Training features X cannot be None")
        
        if y is None:
            raise OptimizationError("Training targets y cannot be None")
        
        # Check if inputs have length (handle scalars and non-array inputs)
        try:
            x_len = len(X)
        except (TypeError, AttributeError):
            raise OptimizationError("Training features X must be array-like or have __len__ method")
        
        try:
            y_len = len(y)
        except (TypeError, AttributeError):
            raise OptimizationError("Training targets y must be array-like or have __len__ method")
        
        # Check for empty inputs
        if x_len == 0:
            raise OptimizationError("Training features X cannot be empty")
        
        if y_len == 0:
            raise OptimizationError("Training targets y cannot be empty")
        
        # Check length mismatch
        if x_len != y_len:
            raise OptimizationError(f"X and y must have the same length (X: {x_len}, y: {y_len})")
        
        if not search_space:
            raise OptimizationError("Search space cannot be empty")
        
        # Check for reasonable data size (configurable minimum)
        min_samples = getattr(self.config, 'min_samples', 10)
        if x_len < min_samples:
            raise OptimizationError(f"Dataset too small for optimization (minimum {min_samples} samples, got {x_len})")
    
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
        
        # Pass early stopping integration if available
        return strategy_class(self.config, self.early_stopping_integration)
    
    def _apply_hardware_optimization(self, context: OptimizationContext) -> None:
        """Apply hardware optimization to the context."""
        try:
            if self.hardware_manager:
                if TPRINT_AVAILABLE:
                    tprint_info(f"🔧 Optimizing hardware for {len(context.X)} samples...")
                
                optimization_config = self.hardware_manager.optimize_for_task(
                    "hyperparameter_optimization", 
                    len(context.X)
                )
                
                # Apply optimization settings to context
                if optimization_config:
                    # Update context with hardware optimization settings
                    if 'workers' in optimization_config:
                        context.max_workers = optimization_config['workers']
                        if TPRINT_AVAILABLE:
                            tprint_info(f"👥 Set max_workers to {context.max_workers}")
                        else:
                            self.logger.info(f"Set max_workers to {context.max_workers}")
                    
                    if 'memory_limit' in optimization_config:
                        context.memory_limit = optimization_config['memory_limit']
                        if TPRINT_AVAILABLE:
                            tprint_info(f"💾 Set memory_limit to {context.memory_limit}GB")
                        else:
                            self.logger.info(f"Set memory_limit to {context.memory_limit}GB")
                    
                    if 'batch_size' in optimization_config:
                        context.batch_size = optimization_config['batch_size']
                        if TPRINT_AVAILABLE:
                            tprint_info(f"📦 Set batch_size to {context.batch_size}")
                        else:
                            self.logger.info(f"Set batch_size to {context.batch_size}")
                    
                    # Update engine config if needed
                    if hasattr(self.config, 'max_workers') and 'workers' in optimization_config:
                        self.config.max_workers = optimization_config['workers']
                    
                    if TPRINT_AVAILABLE:
                        tprint_success(f"✅ Applied hardware optimization: {optimization_config}")
                    else:
                        self.logger.info(f"Applied hardware optimization: {optimization_config}")
                else:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ Hardware optimization returned no configuration")
                    else:
                        self.logger.warning("Hardware optimization returned no configuration")
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Hardware optimization failed: {e}")
            raise HardwareOptimizationError(f"Hardware optimization failed: {e}") from e
    
    def _apply_vectorbt_optimization(self, context: OptimizationContext) -> None:
        """Apply VectorBT optimization to the context."""
        try:
            if self.vectorbt_optimizer:
                # Check if VectorBT is actually available
                try:
                    import vectorbt as vbt
                    if TPRINT_AVAILABLE:
                        tprint_info("⚡ VectorBT detected, applying optimizations...")
                    
                    # Apply VectorBT optimizations if available
                    if hasattr(self.vectorbt_optimizer, 'optimize_rolling_operations'):
                        # Apply rolling operations optimization
                        context.X = self.vectorbt_optimizer.optimize_rolling_operations(
                            context.X, ['rolling_mean', 'rolling_std']
                        )
                        if TPRINT_AVAILABLE:
                            tprint_success("✅ Applied VectorBT rolling operations optimization")
                        else:
                            self.logger.info("Applied VectorBT rolling operations optimization")
                    else:
                        if TPRINT_AVAILABLE:
                            tprint_warning("⚠️ VectorBT optimizer does not implement optimize_rolling_operations")
                        else:
                            self.logger.warning("VectorBT optimizer does not implement optimize_rolling_operations")
                except ImportError:
                    if TPRINT_AVAILABLE:
                        tprint_warning("⚠️ VectorBT is not available - skipping VectorBT optimization")
                    else:
                        self.logger.warning("VectorBT is not available - skipping VectorBT optimization")
                    # Disable VectorBT for this run
                    context.config.enable_vectorbt = False
        except Exception as e:
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ VectorBT optimization failed: {e}")
            else:
                self.logger.error(f"VectorBT optimization failed: {e}")
            # Don't raise error, just disable VectorBT
            context.config.enable_vectorbt = False
    
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