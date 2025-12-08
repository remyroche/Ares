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
# from .regime_specific_tpsl_optimizer import RegimeSpecificTPSLOptimizer  # File does not exist yet
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
    default_objective_function,
    create_custom_balanced_score_objective,
    CUSTOM_BALANCED_SCORE_AVAILABLE
)
from .execution_mode_adapter import (
    get_execution_mode,
    adjust_hpo_params_for_mode,
    adjust_model_iterations_for_mode,
    set_execution_mode
)
from .ic_snr_objective import (
    ICSNRConfig,
    ICMetrics,
    ICSNRObjective,
    compute_spearman_ic,
    compute_ic_metrics_purged,
    compute_stability_across_subsamples,
    create_ic_snr_objective_for_xgb,
    DEFAULT_REGULARIZATION_RANGES,
    is_regularization_param,
)
from .diversity_defense_objectives import (
    SpecialistType,
    SpecialistConfig,
    DiversityDefenseConfig,
    DiversityDefenseObjectives,
    DiversityDefenseAggregator,
    DiversityDefenseHPO,
    create_diversity_defense_ensemble,
)

# Backward compatibility alias
HyperparameterOptimizer = HyperparameterOptimization

__all__ = [
    # Hyperparameter Optimization
    'HyperparameterOptimization',
    'HyperparameterOptimizer',  # Backward compatibility alias

    # Pareto Optimization
    'ParetoFront', 'ParetoFrontAnalyzer', 'ParetoOptimizer',

    # Regime-specific Optimization
    # 'RegimeSpecificTPSLOptimizer',  # Not yet available

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
    'create_custom_balanced_score_objective',
    'CUSTOM_BALANCED_SCORE_AVAILABLE',

    # Grid utilities
    'build_coarse_grid_from_search_space', 'build_fine_grid_around_best',
    
    # Execution mode adapter
    'get_execution_mode',
    'adjust_hpo_params_for_mode',
    'adjust_model_iterations_for_mode',
    'set_execution_mode',
    
    # IC-SNR Objective for Regularization HPO
    'ICSNRConfig',
    'ICMetrics',
    'ICSNRObjective',
    'compute_spearman_ic',
    'compute_ic_metrics_purged',
    'compute_stability_across_subsamples',
    'create_ic_snr_objective_for_xgb',
    'DEFAULT_REGULARIZATION_RANGES',
    'is_regularization_param',
    
    # Diversity Defense Objectives for Bagged LGBM Ensemble
    'SpecialistType',
    'SpecialistConfig',
    'DiversityDefenseConfig',
    'DiversityDefenseObjectives',
    'DiversityDefenseAggregator',
    'DiversityDefenseHPO',
    'create_diversity_defense_ensemble',
]
