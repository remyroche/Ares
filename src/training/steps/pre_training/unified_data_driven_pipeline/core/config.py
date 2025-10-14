"""
Configuration System for Unified Data-Driven Pipeline

Provides comprehensive configuration with guardrails to prevent
brittle "zero heuristics" approach while maintaining data-driven methodology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)


class OptimizationStrategy(Enum):
    """Optimization strategies for different operations."""
    VECTORBT_CPU = "vectorbt_cpu"
    VECTORBT_GPU = "vectorbt_gpu"
    VECTORBT_PARALLEL = "vectorbt_parallel"
    PANDAS_FALLBACK = "pandas_fallback"
    NUMPY_OPTIMIZED = "numpy_optimized"


class FeatureType(Enum):
    """Types of features for categorization."""
    PRICE = "price"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    LIQUIDITY = "liquidity"
    VOLUME = "volume"
    TIME_OF_DAY = "time_of_day"
    REGIME = "regime"
    CROSS_TIMEFRAME = "cross_timeframe"
    INTERACTION = "interaction"


@dataclass
class GuardrailConfig:
    """Configuration for guardrails to prevent brittle statistical discovery."""
    
    # Maximum lookback constraints
    max_lookback_periods: Dict[FeatureType, int] = field(default_factory=lambda: {
        FeatureType.PRICE: 252,  # 1 year
        FeatureType.VOLATILITY: 63,  # 3 months
        FeatureType.MOMENTUM: 21,  # 1 month
        FeatureType.MEAN_REVERSION: 14,  # 2 weeks
        FeatureType.LIQUIDITY: 5,  # 1 week
        FeatureType.VOLUME: 5,  # 1 week
        FeatureType.TIME_OF_DAY: 1,  # No lookback
        FeatureType.REGIME: 252,  # 1 year
        FeatureType.CROSS_TIMEFRAME: 504,  # 2 years
        FeatureType.INTERACTION: 63,  # 3 months
    })
    
    # Monotonicity constraints
    enforce_monotonicity: bool = True
    monotonicity_tolerance: float = 0.01
    
    # Feature cost and latency penalties
    feature_costs: Dict[FeatureType, float] = field(default_factory=lambda: {
        FeatureType.PRICE: 1.0,
        FeatureType.VOLATILITY: 2.0,
        FeatureType.MOMENTUM: 1.5,
        FeatureType.MEAN_REVERSION: 1.5,
        FeatureType.LIQUIDITY: 3.0,
        FeatureType.VOLUME: 1.0,
        FeatureType.TIME_OF_DAY: 0.5,
        FeatureType.REGIME: 4.0,
        FeatureType.CROSS_TIMEFRAME: 5.0,
        FeatureType.INTERACTION: 2.5,
    })
    
    # Domain sanity checks
    enable_domain_checks: bool = True
    price_bounds: tuple = (0.0, float('inf'))
    volatility_bounds: tuple = (0.0, 10.0)  # Max 1000% volatility
    correlation_threshold: float = 0.99  # Max correlation between features
    stability_threshold: float = 0.8  # Min stability score
    
    # Statistical significance thresholds
    min_statistical_significance: float = 0.05
    min_effect_size: float = 0.01
    min_sample_size: int = 30


@dataclass
class TimeSeriesCVConfig:
    """Configuration for time series cross-validation."""
    
    # Basic CV parameters
    n_splits: int = 5
    test_size: float = 0.2
    train_size: float = 0.6
    purge_fraction: float = 0.1
    embargo_fraction: float = 0.05
    
    # Minimum sizes
    min_train_samples: int = 100
    min_test_samples: int = 50
    min_embargo_samples: int = 10
    
    # Validation
    strict_time_ordering: bool = True
    validate_splits: bool = True
    check_leakage: bool = True


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    # Basic parameters
    n_splits: int = 5
    min_window_size: int = 50
    min_train_samples: int = 100
    min_val_samples: int = 20
    min_train_ratio: float = 0.6
    
    # Advanced parameters
    enable_nested_cv: bool = True
    nested_outer_splits: int = 3
    nested_inner_splits: int = 2
    frozen_decision_plan: bool = True
    
    # Validation
    strict_time_ordering: bool = True
    check_leakage: bool = True
    validate_splits: bool = True


@dataclass
class LookbackOptimizationConfig:
    """Configuration for feature lookback optimization."""
    
    # Lookback search space
    min_lookback: int = 5
    max_lookback: int = 100
    step_size: int = 5
    min_samples: int = 20
    
    # Optimization parameters
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_CPU
    enable_bayesian_optimization: bool = True
    bayesian_trials: int = 50
    enable_direction_optimization: bool = True
    optimization_direction: str = 'both'  # 'longs', 'shorts', 'both'
    
    # Walk-forward validation
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    
    # Regularization
    enable_regularization: bool = True
    regularization_strength: float = 0.1
    preferred_min_lookback: float = 40.0
    preferred_max_lookback: float = 80.0
    
    # Performance constraints
    max_computation_time: float = 600.0  # 10 minutes
    memory_limit_gb: float = 8.0
    
    # Guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


@dataclass
class PeriodOptimizationConfig:
    """Configuration for period optimization."""
    
    # Period search space
    min_period: int = 2
    max_period: int = 252
    period_step: int = 1
    
    # Optimization parameters
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_CPU
    enable_parallel: bool = True
    max_workers: Optional[int] = None
    
    # Statistical criteria
    min_period_stability: float = 0.7
    min_period_significance: float = 0.05
    max_period_correlation: float = 0.95
    
    # Performance constraints
    max_computation_time: float = 300.0  # 5 minutes
    memory_limit_gb: float = 8.0
    
    # Guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


@dataclass
class InteractionGenerationConfig:
    """Configuration for interaction generation."""
    
    # Interaction discovery
    max_interactions: int = 100
    min_utility_threshold: float = 0.1
    max_correlation_threshold: float = 0.95
    
    # Statistical criteria
    min_interaction_significance: float = 0.05
    min_interaction_stability: float = 0.6
    min_effect_size: float = 0.01
    
    # Optimization
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_CPU
    enable_batch_processing: bool = True
    batch_size: int = 1000
    
    # HTF interactions
    enable_htf_interactions: bool = True
    htf_interaction_ratio: float = 0.3  # 30% of interactions can be HTF
    
    # Guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective feature selection."""
    
    # Objectives and weights
    objectives: Dict[str, float] = field(default_factory=lambda: {
        'out_of_sample_sharpe': 0.25,
        'drawdown': 0.20,
        'turnover': 0.15,
        'stability': 0.15,
        'diversity': 0.10,
        'mutual_information': 0.10,
        'profit_centered': 0.05
    })
    
    # Stability metrics
    stability_method: str = 'jaccard_similarity'  # or 'correlation_stability'
    min_stability_score: float = 0.8
    
    # Diversity metrics
    diversity_method: str = 'correlation_penalty'  # or 'dpp'
    max_pairwise_correlation: float = 0.8
    
    # Optimization
    optimization_algorithm: str = 'nsga2'  # or 'spea2', 'moea_d'
    max_generations: int = 100
    population_size: int = 50
    
    # Constraints
    max_features: int = 50
    min_features: int = 5
    max_feature_cost: float = 100.0


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""
    
    # Selection strategy
    selection_strategy: str = 'multi_objective'  # or 'single_objective', 'ensemble'
    
    # Multi-objective configuration
    multi_objective: MultiObjectiveConfig = field(default_factory=MultiObjectiveConfig)
    
    # Single objective fallback
    primary_objective: str = 'out_of_sample_sharpe'
    secondary_objective: str = 'stability'
    
    # Ensemble configuration
    ensemble_methods: List[str] = field(default_factory=lambda: ['mrmr', 'lasso', 'random_forest'])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    
    # Cross-validation
    cv_config: TimeSeriesCVConfig = field(default_factory=TimeSeriesCVConfig)
    
    # Performance constraints
    max_computation_time: float = 600.0  # 10 minutes
    memory_limit_gb: float = 16.0
    
    # Guardrails
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


@dataclass
class VectorizationConfig:
    """Configuration for vectorization and optimization."""
    
    # VectorBT settings
    enable_vectorbt: bool = True
    vectorbt_strategy: OptimizationStrategy = OptimizationStrategy.VECTORBT_CPU
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    # Memory management
    memory_efficient: bool = True
    max_memory_gb: float = 8.0
    chunk_size: int = 1000
    
    # Performance monitoring
    enable_monitoring: bool = True
    enable_profiling: bool = False
    enable_caching: bool = True
    cache_size: int = 1000


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    
    # Monitoring
    enable_performance_tracking: bool = True
    enable_memory_monitoring: bool = True
    enable_timing: bool = True
    
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    
    # Profiling
    enable_profiling: bool = False
    profile_output_dir: Optional[str] = None
    
    # Optimization
    enable_auto_optimization: bool = True
    optimization_threshold: float = 0.1  # 10% improvement threshold


@dataclass
class UnifiedPipelineConfig:
    """Main configuration for the unified data-driven pipeline."""
    
    # Component configurations
    period_optimization: PeriodOptimizationConfig = field(default_factory=PeriodOptimizationConfig)
    lookback_optimization: LookbackOptimizationConfig = field(default_factory=LookbackOptimizationConfig)
    interaction_generation: InteractionGenerationConfig = field(default_factory=InteractionGenerationConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    vectorization: VectorizationConfig = field(default_factory=VectorizationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Pipeline settings
    enable_period_optimization: bool = True
    enable_feature_lookback_optimization: bool = True
    enable_interaction_generation: bool = True
    enable_htf_interactions: bool = True
    enable_feature_selection: bool = True
    
    # Advanced lookback optimization settings
    enable_nested_cv: bool = True
    enable_direction_optimization: bool = True
    enable_bayesian_optimization: bool = True
    enable_advanced_caching: bool = True
    enable_regularization: bool = True
    
    # Data validation
    validate_input_data: bool = True
    strict_data_validation: bool = True
    
    # Output settings
    save_intermediate_results: bool = False
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration."""
        tprint_debug("Validating unified pipeline configuration")
        
        # Validate objective weights sum to 1
        total_weight = sum(self.feature_selection.multi_objective.objectives.values())
        if not np.isclose(total_weight, 1.0, atol=1e-6):
            tprint_warning(f"Objective weights sum to {total_weight:.6f}, not 1.0")
        
        # Validate ensemble weights sum to 1
        if self.feature_selection.selection_strategy == 'ensemble':
            total_ensemble_weight = sum(self.feature_selection.ensemble_weights)
            if not np.isclose(total_ensemble_weight, 1.0, atol=1e-6):
                tprint_warning(f"Ensemble weights sum to {total_ensemble_weight:.6f}, not 1.0")
        
        # Validate memory limits
        if self.vectorization.max_memory_gb > self.feature_selection.memory_limit_gb:
            tprint_warning("Vectorization memory limit exceeds feature selection limit")
        
        tprint_success("Configuration validation completed")


def create_default_config() -> UnifiedPipelineConfig:
    """Create a default configuration for the unified pipeline."""
    return UnifiedPipelineConfig()


def create_high_performance_config() -> UnifiedPipelineConfig:
    """Create a high-performance configuration."""
    config = UnifiedPipelineConfig()
    
    # Optimize for performance
    config.vectorization.enable_gpu = True
    config.vectorization.enable_parallel = True
    config.vectorization.vectorbt_strategy = OptimizationStrategy.VECTORBT_GPU
    
    config.period_optimization.enable_parallel = True
    config.period_optimization.optimization_strategy = OptimizationStrategy.VECTORBT_GPU
    
    config.interaction_generation.enable_batch_processing = True
    config.interaction_generation.optimization_strategy = OptimizationStrategy.VECTORBT_GPU
    
    return config


def create_memory_efficient_config() -> UnifiedPipelineConfig:
    """Create a memory-efficient configuration."""
    config = UnifiedPipelineConfig()
    
    # Optimize for memory
    config.vectorization.memory_efficient = True
    config.vectorization.max_memory_gb = 4.0
    config.vectorization.chunk_size = 500
    
    config.period_optimization.memory_limit_gb = 4.0
    config.feature_selection.memory_limit_gb = 4.0
    
    return config


def create_fast_config() -> UnifiedPipelineConfig:
    """Create a fast configuration with reduced complexity."""
    config = UnifiedPipelineConfig()
    
    # Reduce complexity for speed
    config.period_optimization.max_period = 63  # 3 months max
    config.interaction_generation.max_interactions = 50
    config.feature_selection.multi_objective.max_features = 25
    
    # Reduce CV complexity
    config.feature_selection.cv_config.n_splits = 3
    config.feature_selection.cv_config.test_size = 0.3
    
    return config


def load_config_from_file(config_path: Union[str, Path]) -> UnifiedPipelineConfig:
    """Load configuration from a file."""
    # This would implement loading from JSON/YAML
    # For now, return default config
    tprint_warning("Config loading from file not implemented, using default config")
    return create_default_config()


def save_config_to_file(config: UnifiedPipelineConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to a file."""
    # This would implement saving to JSON/YAML
    tprint_warning("Config saving to file not implemented")
    pass