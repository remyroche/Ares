"""
TAS Configuration Classes

Configuration classes for the Tree Architecture Search system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
from pathlib import Path


class TreeModelType(Enum):
    """Tree model types for TAS."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    DECISION_TREE = "decision_tree"
    ADABOOST = "adaboost"
    BAGGING = "bagging"


class SearchMethod(Enum):
    """Search methods for TAS."""
    RANDOM = "random"
    GRID = "grid"
    BAYESIAN = "bayesian"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    META_LEARNING = "meta_learning"
    HYBRID = "hybrid"


class OptimizationObjective(Enum):
    """Optimization objectives for TAS."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PRECISION_RECALL_AUC = "precision_recall_auc"
    LOG_LOSS = "log_loss"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
    R2_SCORE = "r2_score"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFITABILITY = "profitability"
    ROBUSTNESS = "robustness"
    EFFICIENCY = "efficiency"
    INTERPRETABILITY = "interpretability"


@dataclass
class TASConfig:
    """Base configuration for Tree Architecture Search."""
    
    # Model settings
    model_types: List[TreeModelType] = field(default_factory=lambda: [
        TreeModelType.RANDOM_FOREST,
        TreeModelType.XGBOOST,
        TreeModelType.LIGHTGBM
    ])
    
    # Tree architecture constraints
    min_trees: int = 10
    max_trees: int = 1000
    min_depth: int = 1
    max_depth: int = 20
    min_samples_split: int = 2
    max_samples_split: int = 1000
    min_samples_leaf: int = 1
    max_samples_leaf: int = 100
    
    # Feature selection
    min_features: int = 1
    max_features: Union[int, float, str] = "auto"
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        "auto", "sqrt", "log2", "none"
    ])
    
    # Optimization settings
    primary_objective: OptimizationObjective = OptimizationObjective.ACCURACY
    secondary_objectives: List[OptimizationObjective] = field(default_factory=lambda: [
        OptimizationObjective.ROBUSTNESS,
        OptimizationObjective.EFFICIENCY
    ])
    objective_weights: List[float] = field(default_factory=lambda: [0.6, 0.2, 0.2])
    
    # Search settings
    search_method: SearchMethod = SearchMethod.BAYESIAN
    max_iterations: int = 100
    max_time_seconds: int = 3600
    early_stopping_patience: int = 10
    min_improvement_threshold: float = 0.001
    
    # Validation settings
    validation_method: str = "holdout"  # "holdout", "cross_validation", "time_series_split"
    validation_split: float = 0.2
    cv_folds: int = 5
    time_series_gap: int = 0
    
    # Performance settings
    n_jobs: int = -1
    random_state: int = 42
    verbose: bool = True
    
    # Advanced settings
    enable_pruning: bool = True
    enable_feature_importance: bool = True
    enable_early_stopping: bool = True
    enable_hyperparameter_tuning: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    save_best_models: bool = True
    output_dir: str = "tas_output"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model_types': [t.value for t in self.model_types],
            'min_trees': self.min_trees,
            'max_trees': self.max_trees,
            'min_depth': self.min_depth,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_samples_split': self.max_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_samples_leaf': self.max_samples_leaf,
            'min_features': self.min_features,
            'max_features': self.max_features,
            'feature_selection_methods': self.feature_selection_methods,
            'primary_objective': self.primary_objective.value,
            'secondary_objectives': [o.value for o in self.secondary_objectives],
            'objective_weights': self.objective_weights,
            'search_method': self.search_method.value,
            'max_iterations': self.max_iterations,
            'max_time_seconds': self.max_time_seconds,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement_threshold': self.min_improvement_threshold,
            'validation_method': self.validation_method,
            'validation_split': self.validation_split,
            'cv_folds': self.cv_folds,
            'time_series_gap': self.time_series_gap,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'enable_pruning': self.enable_pruning,
            'enable_feature_importance': self.enable_feature_importance,
            'enable_early_stopping': self.enable_early_stopping,
            'enable_hyperparameter_tuning': self.enable_hyperparameter_tuning,
            'save_intermediate_results': self.save_intermediate_results,
            'save_best_models': self.save_best_models,
            'output_dir': self.output_dir
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TASConfig':
        """Create config from dictionary."""
        # Convert string values back to enums
        if 'model_types' in config_dict:
            config_dict['model_types'] = [TreeModelType(t) for t in config_dict['model_types']]
        if 'primary_objective' in config_dict:
            config_dict['primary_objective'] = OptimizationObjective(config_dict['primary_objective'])
        if 'secondary_objectives' in config_dict:
            config_dict['secondary_objectives'] = [OptimizationObjective(o) for o in config_dict['secondary_objectives']]
        if 'search_method' in config_dict:
            config_dict['search_method'] = SearchMethod(config_dict['search_method'])
        
        return cls(**config_dict)


@dataclass
class TASSearchConfig:
    """Configuration for TAS search strategies."""
    
    # Search strategy settings
    search_strategy: SearchMethod = SearchMethod.BAYESIAN
    search_budget: int = 100
    search_time_limit: int = 3600
    
    # Bayesian optimization settings
    bayesian_acquisition_function: str = "expected_improvement"  # "expected_improvement", "upper_confidence_bound", "probability_improvement"
    bayesian_n_initial_points: int = 10
    bayesian_n_restarts_optimizer: int = 5
    bayesian_alpha: float = 1e-6
    
    # Evolutionary algorithm settings
    evolutionary_population_size: int = 50
    evolutionary_generations: int = 100
    evolutionary_mutation_rate: float = 0.1
    evolutionary_crossover_rate: float = 0.8
    evolutionary_elite_size: int = 5
    
    # Reinforcement learning settings
    rl_algorithm: str = "ppo"  # "ppo", "a2c", "dqn"
    rl_episodes: int = 1000
    rl_learning_rate: float = 0.001
    rl_discount_factor: float = 0.99
    rl_epsilon: float = 0.1
    
    # Meta-learning settings
    meta_learning_rate: float = 0.001
    meta_inner_steps: int = 5
    meta_outer_steps: int = 100
    meta_batch_size: int = 32
    
    # Hybrid search settings
    hybrid_strategies: List[SearchMethod] = field(default_factory=lambda: [
        SearchMethod.BAYESIAN,
        SearchMethod.EVOLUTIONARY
    ])
    hybrid_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    hybrid_switch_iteration: int = 50
    
    # Performance settings
    parallel_evaluations: int = 4
    memory_limit_gb: float = 8.0
    cache_evaluations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'search_strategy': self.search_strategy.value,
            'search_budget': self.search_budget,
            'search_time_limit': self.search_time_limit,
            'bayesian_acquisition_function': self.bayesian_acquisition_function,
            'bayesian_n_initial_points': self.bayesian_n_initial_points,
            'bayesian_n_restarts_optimizer': self.bayesian_n_restarts_optimizer,
            'bayesian_alpha': self.bayesian_alpha,
            'evolutionary_population_size': self.evolutionary_population_size,
            'evolutionary_generations': self.evolutionary_generations,
            'evolutionary_mutation_rate': self.evolutionary_mutation_rate,
            'evolutionary_crossover_rate': self.evolutionary_crossover_rate,
            'evolutionary_elite_size': self.evolutionary_elite_size,
            'rl_algorithm': self.rl_algorithm,
            'rl_episodes': self.rl_episodes,
            'rl_learning_rate': self.rl_learning_rate,
            'rl_discount_factor': self.rl_discount_factor,
            'rl_epsilon': self.rl_epsilon,
            'meta_learning_rate': self.meta_learning_rate,
            'meta_inner_steps': self.meta_inner_steps,
            'meta_outer_steps': self.meta_outer_steps,
            'meta_batch_size': self.meta_batch_size,
            'hybrid_strategies': [s.value for s in self.hybrid_strategies],
            'hybrid_weights': self.hybrid_weights,
            'hybrid_switch_iteration': self.hybrid_switch_iteration,
            'parallel_evaluations': self.parallel_evaluations,
            'memory_limit_gb': self.memory_limit_gb,
            'cache_evaluations': self.cache_evaluations
        }


@dataclass
class TASOptimizationConfig:
    """Configuration for TAS optimization."""
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    multi_objective_algorithm: str = "nsga2"  # "nsga2", "spea2", "moea_d"
    pareto_front_size: int = 100
    diversity_metric: str = "crowding_distance"
    
    # Regime-aware optimization
    enable_regime_aware: bool = True
    regime_detection_method: str = "clustering"  # "clustering", "changepoint", "hmm"
    regime_stability_threshold: float = 0.7
    regime_transition_cost: float = 0.1
    
    # Real-time optimization
    enable_real_time: bool = False
    adaptation_frequency: int = 100  # Adapt every N samples
    adaptation_threshold: float = 0.05  # Performance degradation threshold
    adaptation_method: str = "incremental"  # "incremental", "retrain", "meta_learning"
    
    # Continual learning
    enable_continual_learning: bool = False
    memory_size: int = 1000
    forgetting_rate: float = 0.1
    replay_method: str = "random"  # "random", "importance", "diversity"
    
    # Hardware optimization
    enable_hardware_optimization: bool = True
    target_device: str = "auto"  # "auto", "cpu", "gpu", "m1"
    memory_optimization: bool = True
    parallel_processing: bool = True
    
    # Uncertainty estimation
    enable_uncertainty_estimation: bool = True
    uncertainty_method: str = "ensemble"  # "ensemble", "dropout", "bayesian"
    uncertainty_samples: int = 100
    confidence_threshold: float = 0.8
    
    # Robustness optimization
    enable_robustness_optimization: bool = True
    robustness_perturbations: List[str] = field(default_factory=lambda: [
        "noise", "adversarial", "distribution_shift"
    ])
    robustness_weight: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enable_multi_objective': self.enable_multi_objective,
            'multi_objective_algorithm': self.multi_objective_algorithm,
            'pareto_front_size': self.pareto_front_size,
            'diversity_metric': self.diversity_metric,
            'enable_regime_aware': self.enable_regime_aware,
            'regime_detection_method': self.regime_detection_method,
            'regime_stability_threshold': self.regime_stability_threshold,
            'regime_transition_cost': self.regime_transition_cost,
            'enable_real_time': self.enable_real_time,
            'adaptation_frequency': self.adaptation_frequency,
            'adaptation_threshold': self.adaptation_threshold,
            'adaptation_method': self.adaptation_method,
            'enable_continual_learning': self.enable_continual_learning,
            'memory_size': self.memory_size,
            'forgetting_rate': self.forgetting_rate,
            'replay_method': self.replay_method,
            'enable_hardware_optimization': self.enable_hardware_optimization,
            'target_device': self.target_device,
            'memory_optimization': self.memory_optimization,
            'parallel_processing': self.parallel_processing,
            'enable_uncertainty_estimation': self.enable_uncertainty_estimation,
            'uncertainty_method': self.uncertainty_method,
            'uncertainty_samples': self.uncertainty_samples,
            'confidence_threshold': self.confidence_threshold,
            'enable_robustness_optimization': self.enable_robustness_optimization,
            'robustness_perturbations': self.robustness_perturbations,
            'robustness_weight': self.robustness_weight
        }


# Configuration presets
def create_quick_config() -> TASConfig:
    """Create a quick search configuration."""
    return TASConfig(
        max_iterations=20,
        max_time_seconds=300,
        search_method=SearchMethod.RANDOM,
        enable_hyperparameter_tuning=False
    )


def create_comprehensive_config() -> TASConfig:
    """Create a comprehensive search configuration."""
    return TASConfig(
        max_iterations=500,
        max_time_seconds=7200,
        search_method=SearchMethod.HYBRID,
        enable_hyperparameter_tuning=True,
        enable_pruning=True,
        enable_feature_importance=True
    )


def create_regime_aware_config() -> TASConfig:
    """Create a regime-aware search configuration."""
    return TASConfig(
        max_iterations=200,
        max_time_seconds=3600,
        search_method=SearchMethod.BAYESIAN,
        enable_hyperparameter_tuning=True,
        secondary_objectives=[
            OptimizationObjective.ROBUSTNESS,
            OptimizationObjective.EFFICIENCY
        ]
    )


def create_real_time_config() -> TASConfig:
    """Create a real-time search configuration."""
    return TASConfig(
        max_iterations=50,
        max_time_seconds=600,
        search_method=SearchMethod.BAYESIAN,
        enable_hyperparameter_tuning=False,
        early_stopping_patience=5
    )