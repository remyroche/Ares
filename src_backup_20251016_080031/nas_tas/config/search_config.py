"""
Search Configuration for NAS/TAS Systems

This module provides specialized configuration classes for different search strategies
and optimization approaches used in both NAS and TAS implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from .base_config import UnifiedArchitectureConfig, SearchStrategy, OptimizationMode


class BayesianAcquisitionFunction(Enum):
    """Bayesian optimization acquisition functions."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_IMPROVEMENT = "probability_improvement"
    ENTROPY_SEARCH = "entropy_search"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"


class EvolutionarySelectionMethod(Enum):
    """Evolutionary algorithm selection methods."""
    TOURNAMENT = "tournament"
    ROULETTE_WHEEL = "roulette_wheel"
    RANK_SELECTION = "rank_selection"
    ELITE_SELECTION = "elite_selection"
    TRUNCATION_SELECTION = "truncation_selection"


class CrossoverMethod(Enum):
    """Evolutionary algorithm crossover methods."""
    UNIFORM_CROSSOVER = "uniform_crossover"
    SINGLE_POINT_CROSSOVER = "single_point_crossover"
    TWO_POINT_CROSSOVER = "two_point_crossover"
    ARITHMETIC_CROSSOVER = "arithmetic_crossover"
    BLEND_CROSSOVER = "blend_crossover"


class MutationMethod(Enum):
    """Evolutionary algorithm mutation methods."""
    GAUSSIAN_MUTATION = "gaussian_mutation"
    UNIFORM_MUTATION = "uniform_mutation"
    POLYNOMIAL_MUTATION = "polynomial_mutation"
    ADAPTIVE_MUTATION = "adaptive_mutation"


@dataclass
class SearchConfig:
    """Base search configuration class."""
    
    # Basic search parameters
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN
    optimization_mode: OptimizationMode = OptimizationMode.SINGLE_OBJECTIVE
    max_iterations: int = 100
    max_time_seconds: int = 3600
    early_stopping_patience: int = 20
    min_improvement_threshold: float = 0.001
    
    # Performance settings
    parallel_evaluations: int = 4
    memory_limit_gb: float = 8.0
    cache_evaluations: bool = True
    
    # Search budget
    search_budget: int = 100
    time_limit: int = 3600
    
    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            else:
                config_dict[field_name] = field_value
        return config_dict


@dataclass
class BayesianSearchConfig(SearchConfig):
    """Configuration for Bayesian optimization search."""
    
    search_strategy: SearchStrategy = SearchStrategy.BAYESIAN
    acquisition_function: BayesianAcquisitionFunction = BayesianAcquisitionFunction.EXPECTED_IMPROVEMENT
    
    # Bayesian optimization parameters
    n_initial_points: int = 10
    n_restarts_optimizer: int = 5
    alpha: float = 1e-6
    n_restarts_acq: int = 1
    
    # Gaussian process parameters
    kernel: str = "rbf"  # rbf, matern, white, etc.
    kernel_length_scale: float = 1.0
    kernel_length_scale_bounds: Tuple[float, float] = (1e-5, 1e5)
    noise_level: float = 1e-6
    normalize_y: bool = True
    
    # Acquisition optimization
    acq_optimizer: str = "sampling"  # sampling, lbfgs
    acq_optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization setup."""
        if not self.acq_optimizer_kwargs:
            self.acq_optimizer_kwargs = {
                'n_points': 10000,
                'n_restarts_optimizer': 5,
                'n_jobs': 1
            }


@dataclass
class EvolutionarySearchConfig(SearchConfig):
    """Configuration for evolutionary algorithm search."""
    
    search_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    
    # Population parameters
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    
    # Selection parameters
    selection_method: EvolutionarySelectionMethod = EvolutionarySelectionMethod.TOURNAMENT
    tournament_size: int = 3
    selection_pressure: float = 1.5
    
    # Crossover parameters
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM_CROSSOVER
    crossover_rate: float = 0.8
    crossover_probability: float = 0.8
    
    # Mutation parameters
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN_MUTATION
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    # Diversity maintenance
    enable_diversity_mechanism: bool = True
    diversity_threshold: float = 0.1
    crowding_distance: bool = True
    
    # Multi-objective parameters (if applicable)
    enable_multi_objective: bool = False
    pareto_front_size: int = 100
    dominance_ranking: bool = True


@dataclass
class ReinforcementLearningSearchConfig(SearchConfig):
    """Configuration for reinforcement learning search."""
    
    search_strategy: SearchStrategy = SearchStrategy.REINFORCEMENT
    
    # RL algorithm parameters
    algorithm: str = "ppo"  # ppo, a2c, dqn, sac
    episodes: int = 1000
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    
    # PPO specific parameters
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Exploration parameters
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # Experience replay
    buffer_size: int = 10000
    batch_size: int = 64
    update_frequency: int = 4
    
    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"
    use_batch_norm: bool = True


@dataclass
class MetaLearningSearchConfig(SearchConfig):
    """Configuration for meta-learning search."""
    
    search_strategy: SearchStrategy = SearchStrategy.META_LEARNING
    
    # Meta-learning parameters
    meta_learning_rate: float = 0.001
    inner_steps: int = 5
    outer_steps: int = 100
    meta_batch_size: int = 32
    
    # MAML parameters
    adaptation_rate: float = 0.01
    adaptation_steps: int = 5
    first_order: bool = False
    
    # Task sampling
    task_sampling_strategy: str = "uniform"  # uniform, weighted, curriculum
    tasks_per_batch: int = 8
    task_diversity_weight: float = 0.1
    
    # Memory and experience
    enable_experience_replay: bool = True
    experience_buffer_size: int = 1000
    experience_sampling_strategy: str = "fifo"  # fifo, priority, diversity


@dataclass
class HybridSearchConfig(SearchConfig):
    """Configuration for hybrid search strategies."""
    
    search_strategy: SearchStrategy = SearchStrategy.HYBRID
    
    # Hybrid strategy components
    primary_strategy: SearchStrategy = SearchStrategy.BAYESIAN
    secondary_strategy: SearchStrategy = SearchStrategy.EVOLUTIONARY
    tertiary_strategy: Optional[SearchStrategy] = None
    
    # Strategy weights
    strategy_weights: List[float] = field(default_factory=lambda: [0.6, 0.4])
    
    # Switching criteria
    switch_iteration: int = 50
    switch_criterion: str = "iteration"  # iteration, performance, diversity
    
    # Performance thresholds for switching
    performance_threshold: float = 0.01
    diversity_threshold: float = 0.1
    
    # Individual strategy configurations
    bayesian_config: Optional[BayesianSearchConfig] = None
    evolutionary_config: Optional[EvolutionarySearchConfig] = None
    rl_config: Optional[ReinforcementLearningSearchConfig] = None
    meta_config: Optional[MetaLearningSearchConfig] = None
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Initialize individual strategy configs if not provided
        if self.bayesian_config is None:
            self.bayesian_config = BayesianSearchConfig()
        
        if self.evolutionary_config is None:
            self.evolutionary_config = EvolutionarySearchConfig()
        
        if self.rl_config is None:
            self.rl_config = ReinforcementLearningSearchConfig()
        
        if self.meta_config is None:
            self.meta_config = MetaLearningSearchConfig()


@dataclass
class NASSearchConfig(SearchConfig):
    """Specialized configuration for Neural Architecture Search."""
    
    # Neural architecture specific parameters
    max_layers: int = 10
    min_layers: int = 2
    max_neurons_per_layer: int = 512
    min_neurons_per_layer: int = 16
    
    # Layer types to consider
    layer_types: List[str] = field(default_factory=lambda: [
        'linear', 'conv1d', 'conv2d', 'lstm', 'gru', 'attention'
    ])
    
    # Activation functions
    activation_functions: List[str] = field(default_factory=lambda: [
        'relu', 'tanh', 'sigmoid', 'gelu', 'swish', 'leaky_relu'
    ])
    
    # Regularization options
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    batch_norm_options: List[bool] = field(default_factory=lambda: [True, False])
    
    # Optimization constraints
    max_parameters: int = 1000000  # 1M parameters
    max_flops: int = 1000000000  # 1B FLOPs
    min_accuracy_threshold: float = 0.7
    
    # Architecture evaluation
    evaluation_budget: int = 100
    quick_evaluation_epochs: int = 5
    full_evaluation_epochs: int = 50


@dataclass
class TASSearchConfig(SearchConfig):
    """Specialized configuration for Tree Architecture Search."""
    
    # Tree architecture specific parameters
    max_depth: int = 20
    min_depth: int = 3
    max_trees: int = 1000
    min_trees: int = 10
    
    # Tree types to consider
    tree_types: List[str] = field(default_factory=lambda: [
        'random_forest', 'xgboost', 'lightgbm', 'extra_trees', 'adaboost'
    ])
    
    # Feature selection options
    feature_selection_methods: List[str] = field(default_factory=lambda: [
        'auto', 'sqrt', 'log2', 'none', 'variance_threshold', 'mutual_information'
    ])
    
    # Splitting strategies
    splitting_strategies: List[str] = field(default_factory=lambda: [
        'gini', 'entropy', 'log_loss', 'xgb_gbtree', 'lgb_gbdt'
    ])
    
    # Ensemble methods
    ensemble_methods: List[str] = field(default_factory=lambda: [
        'voting', 'stacking', 'weighted_voting', 'bagging'
    ])
    
    # Optimization constraints
    max_model_size_mb: float = 100.0  # 100MB
    max_training_time_minutes: int = 60
    min_accuracy_threshold: float = 0.6
    
    # Tree evaluation
    evaluation_budget: int = 50
    cross_validation_folds: int = 5
    quick_evaluation_samples: int = 1000


def create_search_config(
    search_strategy: SearchStrategy,
    optimization_mode: OptimizationMode = OptimizationMode.SINGLE_OBJECTIVE,
    **kwargs
) -> SearchConfig:
    """
    Factory function to create appropriate search configuration based on strategy.
    
    Args:
        search_strategy: The search strategy to use
        optimization_mode: The optimization mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Appropriate SearchConfig instance
    """
    base_params = {
        'search_strategy': search_strategy,
        'optimization_mode': optimization_mode,
        **kwargs
    }
    
    if search_strategy == SearchStrategy.BAYESIAN:
        return BayesianSearchConfig(**base_params)
    elif search_strategy == SearchStrategy.EVOLUTIONARY:
        return EvolutionarySearchConfig(**base_params)
    elif search_strategy == SearchStrategy.REINFORCEMENT:
        return ReinforcementLearningSearchConfig(**base_params)
    elif search_strategy == SearchStrategy.META_LEARNING:
        return MetaLearningSearchConfig(**base_params)
    elif search_strategy == SearchStrategy.HYBRID:
        return HybridSearchConfig(**base_params)
    else:
        return SearchConfig(**base_params)


def create_nas_search_config(**kwargs) -> NASSearchConfig:
    """Create NAS-specific search configuration."""
    return NASSearchConfig(**kwargs)


def create_tas_search_config(**kwargs) -> TASSearchConfig:
    """Create TAS-specific search configuration."""
    return TASSearchConfig(**kwargs)