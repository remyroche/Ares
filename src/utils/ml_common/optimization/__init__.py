"""
ML Common - Optimization Module

This module contains all optimization-related functionality including:
- Hyperparameter optimization
- Pareto optimization
- Regime-specific optimization
- Multi-objective optimization

Refactored for better maintainability and performance while maintaining
full backward compatibility.
"""

# Import refactored components
from .refactored_hpo import (
    ConsolidatedHPO, HPOConfig, HPOPhaseConfig, HPOResult,
    # Legacy compatibility
    HyperparameterOptimization, HierarchicalHPO, HierarchicalHPOConfig,
    optimize_hyperparameters, staged_hpo, bayesian_optimization,
    # Factory functions
    create_consolidated_hpo, create_bayesian_hpo, create_bohb_hpo,
    create_grid_hpo, create_random_hpo, create_ares_mode_hpo, create_auto_mode_hpo
)

# Import new core components
from .core import (
    HPOEngine, OptimizationStrategy, BayesianStrategy, GridStrategy, 
    RandomStrategy, BOHBStrategy, PrunerFactory, OptimizationMonitor, OptimizationCache
)

# Import validation and exceptions
from .validation import (
    HPOConfig, HPOPhaseConfig, PrunerConfig, SearchSpaceParameter,
    validate_search_space, validate_hpo_config, validate_pruner_config,
    OptimizationStrategy as StrategyEnum, AresExecutionMode, PrunerStrategy
)
from .exceptions import (
    OptimizationError, ConfigurationError, ModelEvaluationError,
    HardwareOptimizationError, PruningError, SearchSpaceError,
    ConvergenceError, TimeoutError, ValidationError, CacheError,
    MonitoringError, VectorBTError, AresModeError
)

# Import results
from .results import HPOResult
from .pareto import ParetoFront, ParetoFrontAnalyzer, ParetoOptimizer
from .regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
# Enhanced hardware optimization imports
try:
    from ...hardware import (
        get_integrated_hardware_manager,
        m1_optimized, memory_optimized, auto_optimize, smart_cache,
        performance_tracked, WorkloadCategory, OptimizationStrategy
    )
    from ...hardware.enhanced_cpu_optimizer import EnhancedCPUOptimizer
    from ...hardware.m1_memory_optimizer import M1MemoryOptimizer
    from ...hardware.enhanced_gpu_manager import EnhancedM1GPUManager
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False

# Legacy compatibility imports
try:
    from ..hardware_optimized_parallel_processor import (
        HardwareOptimizedMLProcessor,
        get_hardware_optimized_ml_processor,
        ml_training_optimized,
        feature_engineering_optimized,
        hpo_optimized
    )
    from ..gpu_acceleration_utils import (
        GPUAccelerationUtils,
        get_gpu_acceleration_utils,
        gpu_accelerated,
        adaptive_gpu_acceleration
    )
    LEGACY_HARDWARE_AVAILABLE = True
except ImportError:
    LEGACY_HARDWARE_AVAILABLE = False

__all__ = [
    # Refactored HPO (main)
    'ConsolidatedHPO', 'HPOConfig', 'HPOResult',
    
    # Legacy HPO compatibility
    'HyperparameterOptimization', 'HierarchicalHPO', 'HierarchicalHPOConfig', 'HPOPhaseConfig',
    'optimize_hyperparameters', 'staged_hpo', 'bayesian_optimization',
    
    # Factory functions
    'create_consolidated_hpo', 'create_bayesian_hpo', 'create_bohb_hpo',
    'create_grid_hpo', 'create_random_hpo', 'create_ares_mode_hpo', 'create_auto_mode_hpo',

    # Core components
    'HPOEngine', 'OptimizationStrategy', 'BayesianStrategy', 'GridStrategy', 
    'RandomStrategy', 'BOHBStrategy', 'PrunerFactory', 'OptimizationMonitor', 'OptimizationCache',
    
    # Validation and configuration
    'PrunerConfig', 'SearchSpaceParameter', 'validate_search_space', 'validate_hpo_config', 'validate_pruner_config',
    'StrategyEnum', 'AresExecutionMode', 'PrunerStrategy',
    
    # Exceptions
    'OptimizationError', 'ConfigurationError', 'ModelEvaluationError',
    'HardwareOptimizationError', 'PruningError', 'SearchSpaceError',
    'ConvergenceError', 'TimeoutError', 'ValidationError', 'CacheError',
    'MonitoringError', 'VectorBTError', 'AresModeError',

    # Pareto Optimization
    'ParetoFront', 'ParetoFrontAnalyzer', 'ParetoOptimizer',

    # Regime-specific Optimization
    'RegimeSpecificTPSLOptimizer',

    # Grid utilities
    'build_coarse_grid_from_search_space', 'build_fine_grid_around_best',
    
    # Enhanced hardware optimization (preferred)
    'HARDWARE_OPTIMIZATION_AVAILABLE',
    'get_integrated_hardware_manager',
    'm1_optimized', 'memory_optimized', 'auto_optimize', 'smart_cache',
    'performance_tracked', 'WorkloadCategory', 'OptimizationStrategy',
    'EnhancedCPUOptimizer', 'M1MemoryOptimizer', 'EnhancedM1GPUManager',
]

# Add legacy hardware imports if available
if LEGACY_HARDWARE_AVAILABLE:
    __all__.extend([
        # Legacy hardware-optimized processing
        'HardwareOptimizedMLProcessor',
        'get_hardware_optimized_ml_processor',
        'ml_training_optimized',
        'feature_engineering_optimized',
        'hpo_optimized',
        
        # Legacy GPU acceleration
        'GPUAccelerationUtils',
        'get_gpu_acceleration_utils',
        'gpu_accelerated',
        'adaptive_gpu_acceleration'
    ])
