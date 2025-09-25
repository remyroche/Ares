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
from src.utils.nas_tas.hierarchical_hpo import HierarchicalHPO, HierarchicalHPOConfig, HPOPhaseConfig
from .grid_utils import build_coarse_grid_from_search_space, build_fine_grid_around_best
from .bayesian_tpe_optimizer import (
    BayesianTPEOptimizer, 
    BayesianTPEConfig, 
    OptimizationResult,
    optimize_with_bayesian_tpe,
    create_search_space_from_bounds
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
    
    # Bayesian TPE Optimization
    'BayesianTPEOptimizer', 'BayesianTPEConfig', 'OptimizationResult',
    'optimize_with_bayesian_tpe', 'create_search_space_from_bounds'
]