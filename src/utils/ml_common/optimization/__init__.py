"""
ML Common - Optimization Module

This module contains all optimization-related functionality including:
- Hyperparameter optimization
- Pareto optimization
- Regime-specific optimization
- Multi-objective optimization
- Hierarchical parameter optimization
"""

from .hpo_utils import HyperparameterOptimization
from .pareto import ParetoFront, ParetoFrontAnalyzer, ParetoOptimizer
from .regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer
from .hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from .hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    OptimizationBackend,
    StageConfig,
    OptimizationResult,
    HierarchicalOptimizationResult,
    create_param_group,
    default_objective_function
)

__all__ = [
    # Hyperparameter Optimization
    'HyperparameterOptimization',

    # Pareto Optimization
    'ParetoFront', 'ParetoFrontAnalyzer', 'ParetoOptimizer',

    # Regime-specific Optimization
    'RegimeSpecificTPSLOptimizer',

    # Hierarchical HPO (for ensembles)
    'HierarchicalHPO', 'HierarchicalHPOConfig', 'HPOPhaseConfig',

    # Hierarchical Parameter Optimization (general purpose)
    'HierarchicalParameterOptimizer',
    'ParameterGroup',
    'OptimizationStage',
    'OptimizationBackend',
    'StageConfig',
    'OptimizationResult',
    'HierarchicalOptimizationResult',
    'create_param_group',
    'default_objective_function',

    # Grid utilities
    'build_coarse_grid_from_search_space', 'build_fine_grid_around_best'
]
