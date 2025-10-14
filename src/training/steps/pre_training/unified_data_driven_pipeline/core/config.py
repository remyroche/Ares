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

# Import features_common configuration utilities
try:
    from src.features_common import (
        OptimizationConfig, get_optimization_config,
        VectorBTConfig as FeaturesCommonVectorBTConfig, get_vectorbt_config,
        UnifiedConfig, get_unified_config,
        validate_configuration, check_system_health
    )
    FEATURES_COMMON_CONFIG_AVAILABLE = True
except ImportError:
    FEATURES_COMMON_CONFIG_AVAILABLE = False

# Import feature_generation configuration utilities
try:
    from src.feature_generation.utils import (
        UtilityConfig, FeatureOptimizationConfig
    )
    FEATURE_GENERATION_CONFIG_AVAILABLE = True
except ImportError:
    FEATURE_GENERATION_CONFIG_AVAILABLE = False


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
    
    # Labeling system configuration
    labeling_type: str = "analyst"  # "analyst" or "tactician"
    enable_labeling_optimization: bool = True
    labeling_quality_threshold: float = 0.7
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
    try:
        from src.utils.common_operations import safe_json_load, safe_file_exists
        from src.utils.tprint import tprint_error, tprint_success, tprint_warning
        
        config_path = Path(config_path)
        
        if not safe_file_exists(config_path):
            tprint_warning(f"Config file not found: {config_path}, using default config")
            return create_default_config()
        
        # Load JSON configuration
        config_data = safe_json_load(config_path)
        if config_data is None:
            tprint_warning(f"Failed to load config from {config_path}, using default config")
            return create_default_config()
        
        # Convert to UnifiedPipelineConfig
        config = UnifiedPipelineConfig()
        
        # Update configuration with loaded data
        if 'period_optimization' in config_data:
            for key, value in config_data['period_optimization'].items():
                if hasattr(config.period_optimization, key):
                    setattr(config.period_optimization, key, value)
        
        if 'lookback_optimization' in config_data:
            for key, value in config_data['lookback_optimization'].items():
                if hasattr(config.lookback_optimization, key):
                    setattr(config.lookback_optimization, key, value)
        
        if 'interaction_generation' in config_data:
            for key, value in config_data['interaction_generation'].items():
                if hasattr(config.interaction_generation, key):
                    setattr(config.interaction_generation, key, value)
        
        if 'feature_selection' in config_data:
            for key, value in config_data['feature_selection'].items():
                if hasattr(config.feature_selection, key):
                    setattr(config.feature_selection, key, value)
        
        if 'vectorization' in config_data:
            for key, value in config_data['vectorization'].items():
                if hasattr(config.vectorization, key):
                    setattr(config.vectorization, key, value)
        
        if 'performance' in config_data:
            for key, value in config_data['performance'].items():
                if hasattr(config.performance, key):
                    setattr(config.performance, key, value)
        
        # Update pipeline settings
        pipeline_settings = ['enable_period_optimization', 'enable_feature_lookback_optimization',
                           'enable_interaction_generation', 'enable_htf_interactions',
                           'enable_feature_selection', 'enable_nested_cv', 'enable_direction_optimization',
                           'enable_bayesian_optimization', 'enable_advanced_caching', 'enable_regularization',
                           'validate_input_data', 'strict_data_validation', 'save_intermediate_results']
        
        for setting in pipeline_settings:
            if setting in config_data:
                setattr(config, setting, config_data[setting])
        
        if 'output_dir' in config_data:
            config.output_dir = config_data['output_dir']
        
        tprint_success(f"Successfully loaded configuration from {config_path}")
        return config
        
    except Exception as e:
        tprint_error(f"Error loading config from {config_path}: {e}")
        tprint_warning("Using default configuration")
        return create_default_config()


def save_config_to_file(config: UnifiedPipelineConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to a file."""
    try:
        from src.utils.common_operations import safe_json_dump, ensure_directory
        from src.utils.tprint import tprint_error, tprint_success, tprint_warning
        import json
        
        config_path = Path(config_path)
        
        # Ensure directory exists
        ensure_directory(config_path.parent)
        
        # Convert configuration to dictionary
        config_dict = {
            'period_optimization': {
                'min_period': config.period_optimization.min_period,
                'max_period': config.period_optimization.max_period,
                'period_step': config.period_optimization.period_step,
                'optimization_strategy': config.period_optimization.optimization_strategy.value,
                'enable_parallel': config.period_optimization.enable_parallel,
                'max_workers': config.period_optimization.max_workers,
                'min_period_stability': config.period_optimization.min_period_stability,
                'min_period_significance': config.period_optimization.min_period_significance,
                'max_period_correlation': config.period_optimization.max_period_correlation,
                'max_computation_time': config.period_optimization.max_computation_time,
                'memory_limit_gb': config.period_optimization.memory_limit_gb
            },
            'lookback_optimization': {
                'min_lookback': config.lookback_optimization.min_lookback,
                'max_lookback': config.lookback_optimization.max_lookback,
                'step_size': config.lookback_optimization.step_size,
                'min_samples': config.lookback_optimization.min_samples,
                'optimization_strategy': config.lookback_optimization.optimization_strategy.value,
                'enable_bayesian_optimization': config.lookback_optimization.enable_bayesian_optimization,
                'bayesian_trials': config.lookback_optimization.bayesian_trials,
                'enable_direction_optimization': config.lookback_optimization.enable_direction_optimization,
                'optimization_direction': config.lookback_optimization.optimization_direction,
                'enable_regularization': config.lookback_optimization.enable_regularization,
                'regularization_strength': config.lookback_optimization.regularization_strength,
                'labeling_type': config.lookback_optimization.labeling_type,
                'enable_labeling_optimization': config.lookback_optimization.enable_labeling_optimization,
                'labeling_quality_threshold': config.lookback_optimization.labeling_quality_threshold,
                'preferred_min_lookback': config.lookback_optimization.preferred_min_lookback,
                'preferred_max_lookback': config.lookback_optimization.preferred_max_lookback,
                'max_computation_time': config.lookback_optimization.max_computation_time,
                'memory_limit_gb': config.lookback_optimization.memory_limit_gb
            },
            'interaction_generation': {
                'max_interactions': config.interaction_generation.max_interactions,
                'min_utility_threshold': config.interaction_generation.min_utility_threshold,
                'max_correlation_threshold': config.interaction_generation.max_correlation_threshold,
                'min_interaction_significance': config.interaction_generation.min_interaction_significance,
                'min_interaction_stability': config.interaction_generation.min_interaction_stability,
                'min_effect_size': config.interaction_generation.min_effect_size,
                'optimization_strategy': config.interaction_generation.optimization_strategy.value,
                'enable_batch_processing': config.interaction_generation.enable_batch_processing,
                'batch_size': config.interaction_generation.batch_size,
                'enable_htf_interactions': config.interaction_generation.enable_htf_interactions,
                'htf_interaction_ratio': config.interaction_generation.htf_interaction_ratio
            },
            'feature_selection': {
                'selection_strategy': config.feature_selection.selection_strategy,
                'multi_objective': {
                    'objectives': config.feature_selection.multi_objective.objectives,
                    'stability_method': config.feature_selection.multi_objective.stability_method,
                    'min_stability_score': config.feature_selection.multi_objective.min_stability_score,
                    'diversity_method': config.feature_selection.multi_objective.diversity_method,
                    'max_pairwise_correlation': config.feature_selection.multi_objective.max_pairwise_correlation,
                    'optimization_algorithm': config.feature_selection.multi_objective.optimization_algorithm,
                    'max_generations': config.feature_selection.multi_objective.max_generations,
                    'population_size': config.feature_selection.multi_objective.population_size,
                    'max_features': config.feature_selection.multi_objective.max_features,
                    'min_features': config.feature_selection.multi_objective.min_features,
                    'max_feature_cost': config.feature_selection.multi_objective.max_feature_cost
                },
                'primary_objective': config.feature_selection.primary_objective,
                'secondary_objective': config.feature_selection.secondary_objective,
                'ensemble_methods': config.feature_selection.ensemble_methods,
                'ensemble_weights': config.feature_selection.ensemble_weights,
                'max_computation_time': config.feature_selection.max_computation_time,
                'memory_limit_gb': config.feature_selection.memory_limit_gb
            },
            'vectorization': {
                'enable_vectorbt': config.vectorization.enable_vectorbt,
                'vectorbt_strategy': config.vectorization.vectorbt_strategy.value,
                'enable_gpu': config.vectorization.enable_gpu,
                'enable_parallel': config.vectorization.enable_parallel,
                'memory_efficient': config.vectorization.memory_efficient,
                'max_memory_gb': config.vectorization.max_memory_gb,
                'chunk_size': config.vectorization.chunk_size,
                'enable_monitoring': config.vectorization.enable_monitoring,
                'enable_profiling': config.vectorization.enable_profiling,
                'enable_caching': config.vectorization.enable_caching,
                'cache_size': config.vectorization.cache_size
            },
            'performance': {
                'enable_performance_tracking': config.performance.enable_performance_tracking,
                'enable_memory_monitoring': config.performance.enable_memory_monitoring,
                'enable_timing': config.performance.enable_timing,
                'log_level': config.performance.log_level,
                'log_to_file': config.performance.log_to_file,
                'log_file_path': config.performance.log_file_path,
                'enable_profiling': config.performance.enable_profiling,
                'profile_output_dir': config.performance.profile_output_dir,
                'enable_auto_optimization': config.performance.enable_auto_optimization,
                'optimization_threshold': config.performance.optimization_threshold
            },
            'pipeline_settings': {
                'enable_period_optimization': config.enable_period_optimization,
                'enable_feature_lookback_optimization': config.enable_feature_lookback_optimization,
                'enable_interaction_generation': config.enable_interaction_generation,
                'enable_htf_interactions': config.enable_htf_interactions,
                'enable_feature_selection': config.enable_feature_selection,
                'enable_nested_cv': config.enable_nested_cv,
                'enable_direction_optimization': config.enable_direction_optimization,
                'enable_bayesian_optimization': config.enable_bayesian_optimization,
                'enable_advanced_caching': config.enable_advanced_caching,
                'enable_regularization': config.enable_regularization,
                'validate_input_data': config.validate_input_data,
                'strict_data_validation': config.strict_data_validation,
                'save_intermediate_results': config.save_intermediate_results,
                'output_dir': config.output_dir
            }
        }
        
        # Save to JSON file
        success = safe_json_dump(config_dict, config_path, indent=2)
        
        if success:
            tprint_success(f"Successfully saved configuration to {config_path}")
        else:
            tprint_warning(f"Failed to save configuration to {config_path}")
            
    except Exception as e:
        tprint_error(f"Error saving config to {config_path}: {e}")
        raise