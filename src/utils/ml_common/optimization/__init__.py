"""
ML Common - Optimization Module

This module contains all optimization-related functionality including:
- Hyperparameter optimization
- Pareto optimization
- Regime-specific optimization
- Multi-objective optimization
"""

from .hpo_utils import HyperparameterOptimization
from .pareto import ParetoFront, ParetoFrontAnalyzer, ParetoOptimizer
from .regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer
from .hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
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

__all__ = [
    # Hyperparameter Optimization
    'HyperparameterOptimization',

    # Pareto Optimization
    'ParetoFront', 'ParetoFrontAnalyzer', 'ParetoOptimizer',

    # Regime-specific Optimization
    'RegimeSpecificTPSLOptimizer',

    # Hierarchical HPO
    'HierarchicalHPO', 'HierarchicalHPOConfig', 'HPOPhaseConfig',

    # Grid utilities
    'build_coarse_grid_from_search_space', 'build_fine_grid_around_best',
    
    # Hardware-optimized processing
    'HardwareOptimizedMLProcessor',
    'get_hardware_optimized_ml_processor',
    'ml_training_optimized',
    'feature_engineering_optimized',
    'hpo_optimized',
    
    # GPU acceleration
    'GPUAccelerationUtils',
    'get_gpu_acceleration_utils',
    'gpu_accelerated',
    'adaptive_gpu_acceleration'
]
