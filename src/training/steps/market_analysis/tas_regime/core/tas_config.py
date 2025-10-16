"""
Advanced Tree Architecture Search Configuration

This module provides comprehensive configuration for Tree Architecture Search with:
- General tree-based architecture search capabilities
- Trading-specific optimizations
- Micro-regime detection
- Economic significance validation
- Hardware acceleration
- Multi-objective optimization with advanced constraints
- Neural architecture integration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
from pathlib import Path
import numpy as np

# General TAS enums
class TreeModelType(Enum):
    """Tree model types for TAS."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    EXTRA_TREES = "extra_trees"
    # GRADIENT_BOOSTING = "gradient_boosting"  # Removed - use XGBoost/LightGBM/CatBoost instead
    # DECISION_TREE = "decision_tree"  # Removed - use RandomForest instead
    ADABOOST = "adaboost"
    BAGGING = "bagging"
    # New advanced tree models
    NGBOOST = "ngboost"
    QUANTILE_GBDT = "quantile_gbdt"
    DART = "dart"
    DEEPGBM = "deepgbm"
    NODE = "node"

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

# Trading-specific enums
class TASArchitectureType(Enum):
    """TAS architecture types for advanced trading."""
    TREE_ONLY = "tree_only"
    CVLSA_TREE = "cvlSA_tree"  # Cascade Variable Length Selection Architecture
    HYBRID_TREE_NEURAL = "hybrid_tree_neural"
    NEURAL_ONLY = "neural_only"
    ENSEMBLE_HIERARCHICAL = "ensemble_hierarchical"
    META_LEARNING = "meta_learning"

class MicroRegimeType(Enum):
    """Micro-regime types for subtle market changes."""
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    REVERSAL = "reversal"
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    MOMENTUM_SHIFT = "momentum_shift"
    LIQUIDITY_CHANGE = "liquidity_change"

class TradingObjective(Enum):
    """Advanced trading-specific optimization objectives."""
    PROFITABILITY = "profitability"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    REGIME_STABILITY = "regime_stability"
    ADAPTATION_SPEED = "adaptation_speed"
    ROBUSTNESS = "robustness"
    TRANSACTION_COSTS = "transaction_costs"
    ECONOMIC_SIGNIFICANCE = "economic_significance"
    TRADING_VIABILITY = "trading_viability"
    MICRO_REGIME_ACCURACY = "micro_regime_accuracy"
    PREDICTION_CONFIDENCE = "prediction_confidence"

class MarketRegime(Enum):
    """Advanced market regime types for trading."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"
    CRISIS = "crisis"
    NORMAL = "normal"
    UNKNOWN = "unknown"
    # Micro-regimes
    BREAKOUT_MICRO = "breakout_micro"
    CONSOLIDATION_MICRO = "consolidation_micro"
    REVERSAL_MICRO = "reversal_micro"
    ACCELERATION_MICRO = "acceleration_micro"
    DECELERATION_MICRO = "deceleration_micro"

class ClusteringStrategy(Enum):
    """Clustering strategies for tree-based regime detection."""
    COMPLEMENTARY = "complementary"  # Feature selection + ensemble
    ENSEMBLE = "ensemble"  # Multiple tree models combined
    SEQUENTIAL = "sequential"  # Tree-first approach with refinement
    SINGLE = "single"  # Single best tree model
    AUTO = "auto"  # Data-driven strategy selection

class ClusteringMetric(Enum):
    """Clustering quality metrics."""
    SILHOUETTE = "silhouette_score"
    CALINSKI_HARABASZ = "calinski_harabasz_score"
    DAVIES_BOULDIN = "davies_bouldin_score"
    DUNN_INDEX = "dunn_index"
    ADJUSTED_RAND = "adjusted_rand_score"
    MUTUAL_INFO = "mutual_info_score"

@dataclass
class TASConfig:
    """Advanced configuration for Tree Architecture Search with trading optimizations."""

    # Architecture type and components
    architecture_type: TASArchitectureType = TASArchitectureType.HYBRID_TREE_NEURAL
    enable_micro_regime_detection: bool = True
    enable_neural_components: bool = True
    enable_hierarchical_ensembles: bool = True
    enable_meta_learning: bool = True

    # Model settings
    model_types: List[TreeModelType] = field(default_factory=lambda: [
        TreeModelType.RANDOM_FOREST,
        TreeModelType.XGBOOST,
        TreeModelType.LIGHTGBM,
        TreeModelType.EXTRA_TREES,
        TreeModelType.GRADIENT_BOOSTING,
        TreeModelType.NGBOOST,
        TreeModelType.QUANTILE_GBDT,
        TreeModelType.DART,
        TreeModelType.DEEPGBM,
        TreeModelType.NODE
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

    # Timeframe configuration
    primary_timeframe: str = "15m"
    micro_timeframe: str = "5m"
    regime_detection_window: int = 100  # Data points for regime detection
    adaptation_interval_minutes: int = 15

    # Regime configuration
    n_regimes: int = 12
    min_regime_duration: int = 15  # Minimum 15 minutes
    max_regime_duration: int = 180  # Maximum 3 hours
    data_driven_regimes: bool = True
    regime_stability_threshold: float = 0.7

    # Clustering-specific configuration
    clustering_strategy: ClusteringStrategy = ClusteringStrategy.AUTO
    clustering_metrics: List[ClusteringMetric] = field(default_factory=lambda: [
        ClusteringMetric.SILHOUETTE,
        ClusteringMetric.CALINSKI_HARABASZ,
        ClusteringMetric.DAVIES_BOULDIN
    ])
    enable_unsupervised_regime_detection: bool = True
    enable_data_driven_strategy_selection: bool = True

    # Data analysis thresholds for strategy selection
    tabular_threshold: float = 0.7
    sequential_threshold: float = 0.5
    complexity_threshold: float = 0.8
    volatility_threshold: float = 0.3
    volume_ratio_threshold: float = 2.0

    # Micro-regime configuration
    micro_regime_types: List[MicroRegimeType] = field(default_factory=lambda: [
        MicroRegimeType.BREAKOUT,
        MicroRegimeType.CONSOLIDATION,
        MicroRegimeType.REVERSAL,
        MicroRegimeType.ACCELERATION,
        MicroRegimeType.VOLUME_SPIKE,
        MicroRegimeType.VOLATILITY_SPIKE
    ])
    micro_regime_sensitivity: float = 0.7
    micro_regime_detection_threshold: float = 0.6

    # Multi-objective optimization
    trading_objectives: List[TradingObjective] = field(default_factory=lambda: [
        TradingObjective.PROFITABILITY,
        TradingObjective.SHARPE_RATIO,
        TradingObjective.ROBUSTNESS,
        TradingObjective.ECONOMIC_SIGNIFICANCE,
        TradingObjective.TRADING_VIABILITY
    ])
    trading_objective_weights: List[float] = field(default_factory=lambda: [0.25, 0.2, 0.15, 0.2, 0.2])

    # Economic significance and validation
    economic_significance_threshold: float = 0.7
    trading_viability_threshold: float = 0.6
    regime_transition_cost: float = 0.05
    min_position_size: float = 0.01
    max_position_size: float = 0.1

    # Risk management
    max_drawdown_threshold: float = 0.15
    risk_adjusted_return_threshold: float = 0.1
    transaction_cost_penalty: float = 0.001
    slippage_assumption: float = 0.0005

    # Model configuration
    min_model_confidence: float = 0.6
    max_model_complexity: int = 100
    preferred_model_types: List[str] = field(default_factory=lambda: [
        'RandomForest', 'XGBoost', 'LightGBM', 'ExtraTrees',
        'NeuralNetwork', 'LSTM', 'Attention', 'NeuralODE'
    ])

    # Advanced search parameters
    search_space_config: Dict[str, Any] = field(default_factory=dict)
    enable_bayesian_optimization: bool = True
    enable_evolutionary_search: bool = True
    enable_random_search: bool = False
    n_search_iterations: int = 50
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # CVLSA-specific parameters
    enable_cvlSA_architecture: bool = True
    cvlSA_cascade_depth: int = 3
    cvlSA_variable_selection_methods: List[str] = field(default_factory=lambda: [
        'variance_threshold', 'mutual_information', 'tree_importance',
        'correlation_filter', 'recursive_elimination'
    ])
    cvlSA_feature_ensemble_method: str = "intersection"
    cvlSA_optimization_objective: str = "cascade_efficiency"

    # Hardware acceleration
    enable_hardware_acceleration: bool = True
    enable_gpu_acceleration: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 1000
    max_memory_usage: float = 0.8

    # Meta-learning
    meta_learning_enabled: bool = True
    regime_similarity_threshold: float = 0.8
    adaptation_history_length: int = 100
    transfer_learning_enabled: bool = True

    # Performance tracking
    enable_performance_tracking: bool = True
    performance_tracking_interval: int = 60
    save_model_snapshots: bool = True
    enable_uncertainty_quantification: bool = True

    # Integration settings
    integrate_with_nas_clustering: bool = True
    use_existing_regime_detection: bool = True
    output_format: str = "comprehensive"

    # Validation settings
    validation_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'architecture_type': self.architecture_type.value,
            'enable_micro_regime_detection': self.enable_micro_regime_detection,
            'enable_neural_components': self.enable_neural_components,
            'enable_hierarchical_ensembles': self.enable_hierarchical_ensembles,
            'enable_meta_learning': self.enable_meta_learning,
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
            'output_dir': self.output_dir,
            'primary_timeframe': self.primary_timeframe,
            'micro_timeframe': self.micro_timeframe,
            'regime_detection_window': self.regime_detection_window,
            'adaptation_interval_minutes': self.adaptation_interval_minutes,
            'n_regimes': self.n_regimes,
            'min_regime_duration': self.min_regime_duration,
            'max_regime_duration': self.max_regime_duration,
            'data_driven_regimes': self.data_driven_regimes,
            'regime_stability_threshold': self.regime_stability_threshold,
            'clustering_strategy': self.clustering_strategy.value,
            'clustering_metrics': [m.value for m in self.clustering_metrics],
            'enable_unsupervised_regime_detection': self.enable_unsupervised_regime_detection,
            'enable_data_driven_strategy_selection': self.enable_data_driven_strategy_selection,
            'tabular_threshold': self.tabular_threshold,
            'sequential_threshold': self.sequential_threshold,
            'complexity_threshold': self.complexity_threshold,
            'volatility_threshold': self.volatility_threshold,
            'volume_ratio_threshold': self.volume_ratio_threshold,
            'micro_regime_types': [t.value for t in self.micro_regime_types],
            'micro_regime_sensitivity': self.micro_regime_sensitivity,
            'micro_regime_detection_threshold': self.micro_regime_detection_threshold,
            'trading_objectives': [o.value for o in self.trading_objectives],
            'trading_objective_weights': self.trading_objective_weights,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'regime_transition_cost': self.regime_transition_cost,
            'min_position_size': self.min_position_size,
            'max_position_size': self.max_position_size,
            'max_drawdown_threshold': self.max_drawdown_threshold,
            'risk_adjusted_return_threshold': self.risk_adjusted_return_threshold,
            'transaction_cost_penalty': self.transaction_cost_penalty,
            'slippage_assumption': self.slippage_assumption,
            'min_model_confidence': self.min_model_confidence,
            'max_model_complexity': self.max_model_complexity,
            'preferred_model_types': self.preferred_model_types,
            'search_space_config': self.search_space_config,
            'enable_bayesian_optimization': self.enable_bayesian_optimization,
            'enable_evolutionary_search': self.enable_evolutionary_search,
            'enable_random_search': self.enable_random_search,
            'n_search_iterations': self.n_search_iterations,
            'population_size': self.population_size,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'enable_cvlSA_architecture': self.enable_cvlSA_architecture,
            'cvlSA_cascade_depth': self.cvlSA_cascade_depth,
            'cvlSA_variable_selection_methods': self.cvlSA_variable_selection_methods,
            'cvlSA_feature_ensemble_method': self.cvlSA_feature_ensemble_method,
            'cvlSA_optimization_objective': self.cvlSA_optimization_objective,
            'enable_hardware_acceleration': self.enable_hardware_acceleration,
            'enable_gpu_acceleration': self.enable_gpu_acceleration,
            'enable_batch_processing': self.enable_batch_processing,
            'batch_size': self.batch_size,
            'max_memory_usage': self.max_memory_usage,
            'meta_learning_enabled': self.meta_learning_enabled,
            'regime_similarity_threshold': self.regime_similarity_threshold,
            'adaptation_history_length': self.adaptation_history_length,
            'transfer_learning_enabled': self.transfer_learning_enabled,
            'enable_performance_tracking': self.enable_performance_tracking,
            'performance_tracking_interval': self.performance_tracking_interval,
            'save_model_snapshots': self.save_model_snapshots,
            'enable_uncertainty_quantification': self.enable_uncertainty_quantification,
            'integrate_with_nas_clustering': self.integrate_with_nas_clustering,
            'use_existing_regime_detection': self.use_existing_regime_detection,
            'output_format': self.output_format,
            'validation_config': self.validation_config
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TASConfig':
        """Create config from dictionary."""
        # Convert string values back to enums
        if 'architecture_type' in config_dict:
            config_dict['architecture_type'] = TASArchitectureType(config_dict['architecture_type'])
        if 'model_types' in config_dict:
            config_dict['model_types'] = [TreeModelType(t) for t in config_dict['model_types']]
        if 'primary_objective' in config_dict:
            config_dict['primary_objective'] = OptimizationObjective(config_dict['primary_objective'])
        if 'secondary_objectives' in config_dict:
            config_dict['secondary_objectives'] = [OptimizationObjective(o) for o in config_dict['secondary_objectives']]
        if 'search_method' in config_dict:
            config_dict['search_method'] = SearchMethod(config_dict['search_method'])
        if 'micro_regime_types' in config_dict:
            config_dict['micro_regime_types'] = [MicroRegimeType(t) for t in config_dict['micro_regime_types']]
        if 'trading_objectives' in config_dict:
            config_dict['trading_objectives'] = [TradingObjective(o) for o in config_dict['trading_objectives']]
        if 'clustering_strategy' in config_dict:
            config_dict['clustering_strategy'] = ClusteringStrategy(config_dict['clustering_strategy'])
        if 'clustering_metrics' in config_dict:
            config_dict['clustering_metrics'] = [ClusteringMetric(m) for m in config_dict['clustering_metrics']]

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
    @classmethod
    def create_advanced_trading_config(cls) -> 'TASConfig':
        """Create configuration optimized for advanced trading."""
        return cls(
            architecture_type=TASArchitectureType.HYBRID_TREE_NEURAL,
            enable_micro_regime_detection=True,
            enable_neural_components=True,
            enable_hierarchical_ensembles=True,
            enable_meta_learning=True,
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=12,
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.7,
            micro_regime_sensitivity=0.7,
            micro_regime_detection_threshold=0.6,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS,
                TradingObjective.ECONOMIC_SIGNIFICANCE,
                TradingObjective.TRADING_VIABILITY,
                TradingObjective.MICRO_REGIME_ACCURACY
            ],
            trading_objective_weights=[0.25, 0.2, 0.15, 0.2, 0.15, 0.05],
            economic_significance_threshold=0.7,
            trading_viability_threshold=0.6,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=True,
            enable_gpu_acceleration=True,
            enable_batch_processing=True,
            batch_size=1000,
            meta_learning_enabled=True,
            regime_similarity_threshold=0.8,
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=True,
            enable_cvlSA_architecture=True,
            cvlSA_cascade_depth=3,
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10, 15],
                    'max_depth': [5, 10, 15, 20, 25],
                    'min_trees': [50, 100, 200, 300, 500],
                    'max_trees': [100, 200, 400, 600, 800],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 'auto', 0.3, 0.5, 0.8]
                },
                'neural_search_space': {
                    'hidden_dims': [
                        [32], [64], [128], [256],
                        [64, 32], [128, 64], [256, 128],
                        [128, 64, 32], [256, 128, 64]
                    ],
                    'activation_functions': ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'swish'],
                    'dropout_rates': [0.0, 0.1, 0.2, 0.3, 0.4],
                    'use_lstm': [True, False],
                    'use_attention': [True, False],
                    'use_batch_norm': [True, False]
                }
            },
            validation_config={
                'min_regime_stability': 0.6,
                'min_economic_significance': 0.7,
                'min_trading_viability': 0.6,
                'max_regime_volatility': 0.3,
                'min_prediction_confidence': 0.6,
                'max_model_complexity': 100
            }
        )

    @classmethod
    def create_cvlSA_tree_config(cls) -> 'TASConfig':
        """Create configuration optimized for CVLSA tree architecture."""
        return cls(
            architecture_type=TASArchitectureType.CVLSA_TREE,
            enable_micro_regime_detection=True,
            enable_neural_components=False,  # Tree-only
            enable_hierarchical_ensembles=True,
            enable_meta_learning=True,
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=12,
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.7,
            micro_regime_sensitivity=0.8,  # Higher sensitivity for CVLSA
            micro_regime_detection_threshold=0.7,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS,
                TradingObjective.ECONOMIC_SIGNIFICANCE,
                TradingObjective.TRADING_VIABILITY,
                TradingObjective.MICRO_REGIME_ACCURACY
            ],
            trading_objective_weights=[0.25, 0.2, 0.15, 0.2, 0.15, 0.05],
            economic_significance_threshold=0.7,
            trading_viability_threshold=0.6,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=False,  # Tree-focused, no GPU needed
            enable_gpu_acceleration=False,
            enable_batch_processing=True,
            batch_size=500,  # Smaller batches for tree models
            meta_learning_enabled=True,
            regime_similarity_threshold=0.85,  # Higher similarity for cascade
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=True,
            enable_cvlSA_architecture=True,
            cvlSA_cascade_depth=3,
            cvlSA_variable_selection_methods=[
                'variance_threshold',
                'mutual_information',
                'tree_importance',
                'correlation_filter',
                'recursive_elimination'
            ],
            cvlSA_feature_ensemble_method="intersection",
            cvlSA_optimization_objective="cascade_efficiency",
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10, 15],
                    'max_depth': [5, 10, 15, 20, 25],
                    'min_trees': [50, 100, 200, 300, 500],
                    'max_trees': [100, 200, 400, 600, 800],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 5, 10],
                    'max_features': ['sqrt', 'log2', 'auto', 0.3, 0.5, 0.8],
                    'splitting_strategies': [
                        'gini', 'entropy', 'log_loss',
                        'xgb_gbtree', 'xgb_dart',
                        'lgb_gbdt', 'lgb_rf', 'lgb_dart'
                    ]
                },
                'cvlSA_search_space': {
                    'cascade_depths': [2, 3, 4, 5],
                    'ensemble_methods': ['voting', 'stacking', 'weighted_voting'],
                    'feature_selection_methods': [
                        'variance_threshold', 'mutual_information', 'tree_importance'
                    ],
                    'optimization_objectives': ['accuracy', 'efficiency', 'robustness']
                }
            },
            validation_config={
                'min_regime_stability': 0.6,
                'min_economic_significance': 0.7,
                'min_trading_viability': 0.6,
                'max_regime_volatility': 0.3,
                'min_prediction_confidence': 0.6,
                'max_model_complexity': 100,
                'min_cascade_efficiency': 0.7,
                'min_variable_selection_accuracy': 0.8
            }
        )

    @classmethod
    def create_tree_only_config(cls) -> 'TASConfig':
        """Create configuration for tree-only architectures."""
        return cls(
            architecture_type=TASArchitectureType.TREE_ONLY,
            enable_micro_regime_detection=True,
            enable_neural_components=False,
            enable_hierarchical_ensembles=True,
            enable_meta_learning=False,  # Simpler tree-only approach
            primary_timeframe="15m",
            micro_timeframe="5m",
            n_regimes=8,  # Fewer regimes for tree-only
            min_regime_duration=15,
            max_regime_duration=180,
            data_driven_regimes=True,
            regime_stability_threshold=0.6,
            micro_regime_sensitivity=0.6,
            micro_regime_detection_threshold=0.5,
            trading_objectives=[
                TradingObjective.PROFITABILITY,
                TradingObjective.SHARPE_RATIO,
                TradingObjective.ROBUSTNESS
            ],
            trading_objective_weights=[0.4, 0.3, 0.3],
            economic_significance_threshold=0.6,  # Lower threshold for tree-only
            trading_viability_threshold=0.5,
            regime_transition_cost=0.05,
            enable_hardware_acceleration=False,
            enable_gpu_acceleration=False,
            enable_batch_processing=True,
            batch_size=500,
            meta_learning_enabled=False,
            regime_similarity_threshold=0.7,
            enable_uncertainty_quantification=True,
            integrate_with_nas_clustering=False,  # Tree-only, no neural integration
            enable_cvlSA_architecture=False,  # Use standard tree architecture
            search_space_config={
                'tree_search_space': {
                    'min_depth': [3, 5, 8, 10],
                    'max_depth': [5, 10, 15, 20],
                    'min_trees': [50, 100, 200],
                    'max_trees': [100, 200, 400],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 5],
                    'max_features': ['sqrt', 'log2', 'auto']
                }
            },
            validation_config={
                'min_regime_stability': 0.5,
                'min_economic_significance': 0.6,
                'min_trading_viability': 0.5,
                'max_regime_volatility': 0.4,
                'min_prediction_confidence': 0.5,
                'max_model_complexity': 50
            }
        )

    def get_tree_search_space(self) -> Dict[str, Any]:
        """Get tree-specific search space configuration."""
        return self.search_space_config.get('tree_search_space', {})

    def get_neural_search_space(self) -> Dict[str, Any]:
        """Get neural-specific search space configuration."""
        return self.search_space_config.get('neural_search_space', {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return {
            'regime_stability_threshold': self.regime_stability_threshold,
            'economic_significance_threshold': self.economic_significance_threshold,
            'trading_viability_threshold': self.trading_viability_threshold,
            'min_model_confidence': self.min_model_confidence,
            'max_model_complexity': self.max_model_complexity,
            **self.validation_config
        }
