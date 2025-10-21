"""
Data-Driven Configuration for Clustering Parameters

This module provides configuration classes for making clustering parameters
data-driven rather than hardcoded, including feature group weights,
merging thresholds, temporal windows, and validation cutoffs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
from enum import Enum

# Import tprint utilities for extensive logging
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, 
    tprint_debug, tprint_performance, tprint_progress, tprint_timer,
    tprint_logged, LogLevel
)


class OptimizationStrategy(Enum):
    """Strategy for parameter optimization."""
    BAYESIAN_TPE = "bayesian_tpe"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    ADAPTIVE = "adaptive"


class ValidationMetric(Enum):
    """Metrics for validation and optimization."""
    SILHOUETTE = "silhouette"
    DAVIES_BOULDIN = "davies_bouldin"
    CALINSKI_HARABASZ = "calinski_harabasz"
    ECONOMIC_RETURN = "economic_return"
    SHARPE_RATIO = "sharpe_ratio"
    STABILITY_INDEX = "stability_index"
    COMBINED = "combined"


@dataclass
class FeatureGroupWeightConfig:
    """Configuration for data-driven feature group weight optimization."""
    
    # Enable data-driven optimization
    enable_optimization: bool = True
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_TPE
    
    # Feature groups to optimize
    feature_groups: List[str] = field(default_factory=lambda: [
        'returns', 'volatility', 'volume', 'other'
    ])
    
    # Weight constraints
    min_weight: float = 0.05
    max_weight: float = 0.80
    weight_sum_constraint: bool = True  # Weights must sum to 1.0
    
    # Optimization parameters
    n_trials: int = 100
    n_startup_trials: int = 20
    timeout_seconds: Optional[float] = 300.0
    
    # Validation metrics
    primary_metric: ValidationMetric = ValidationMetric.SILHOUETTE
    secondary_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.DAVIES_BOULDIN, ValidationMetric.STABILITY_INDEX
    ])
    
    # Economic validation
    enable_economic_validation: bool = True
    economic_weight: float = 0.3  # Weight for economic metrics in combined score
    
    # Stability requirements
    min_stability_threshold: float = 0.7
    bootstrap_samples: int = 50
    
    # Feature importance fallback
    enable_feature_importance_fallback: bool = True
    importance_method: str = 'mutual_info'  # 'mutual_info', 'l1_regularization', 'permutation'
    
    # Caching
    enable_caching: bool = True
    cache_ttl_hours: int = 24


@dataclass
class RegimeMergingThresholdConfig:
    """Configuration for data-driven regime merging threshold optimization."""
    
    # Enable data-driven optimization
    enable_optimization: bool = True
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_TPE
    
    # Thresholds to optimize
    similarity_threshold_range: Tuple[float, float] = (0.5, 0.95)
    distance_threshold_range: Tuple[float, float] = (0.1, 0.5)
    p_value_threshold_range: Tuple[float, float] = (0.01, 0.1)
    
    # Optimization parameters
    n_trials: int = 80
    n_startup_trials: int = 15
    timeout_seconds: Optional[float] = 240.0
    
    # Validation metrics
    primary_metric: ValidationMetric = ValidationMetric.SILHOUETTE
    secondary_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.DAVIES_BOULDIN, ValidationMetric.STABILITY_INDEX
    ])
    
    # Merging constraints
    min_clusters_after_merge: int = 2
    max_clusters_after_merge: int = 10
    min_merge_improvement: float = 0.01
    
    # Statistical validation
    enable_statistical_validation: bool = True
    multiple_testing_correction: str = 'bonferroni'  # 'bonferroni', 'fdr', 'none'
    
    # Stability requirements
    min_stability_threshold: float = 0.6
    bootstrap_samples: int = 30


@dataclass
class TemporalWindowConfig:
    """Configuration for data-driven temporal window size optimization."""
    
    # Enable data-driven optimization
    enable_optimization: bool = True
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_TPE
    
    # Window size ranges
    window_size_range: Tuple[int, int] = (50, 500)
    smoothing_window_range: Tuple[int, int] = (3, 20)
    
    # Optimization parameters
    n_trials: int = 60
    n_startup_trials: int = 12
    timeout_seconds: Optional[float] = 180.0
    
    # Validation metrics
    primary_metric: ValidationMetric = ValidationMetric.STABILITY_INDEX
    secondary_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.SILHOUETTE, ValidationMetric.ECONOMIC_RETURN
    ])
    
    # Temporal constraints
    min_window_size: int = 20
    max_window_size: int = 1000
    min_smoothing_window: int = 1
    max_smoothing_window: int = 50
    
    # Volatility adaptation
    enable_volatility_adaptation: bool = True
    volatility_lookback: int = 100
    high_volatility_threshold: float = 0.02  # 2% daily volatility
    low_volatility_threshold: float = 0.005  # 0.5% daily volatility
    
    # Economic validation
    enable_economic_validation: bool = True
    economic_weight: float = 0.4
    
    # Stability requirements
    min_stability_threshold: float = 0.65
    bootstrap_samples: int = 40


@dataclass
class ClusterValidationThresholdConfig:
    """Configuration for data-driven cluster validation threshold optimization."""
    
    # Enable data-driven optimization
    enable_optimization: bool = True
    
    # Optimization strategy
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN_TPE
    
    # Threshold ranges
    min_silhouette_range: Tuple[float, float] = (0.1, 0.5)
    max_dbi_range: Tuple[float, float] = (1.0, 4.0)
    min_stability_range: Tuple[float, float] = (0.5, 0.9)
    
    # Optimization parameters
    n_trials: int = 70
    n_startup_trials: int = 15
    timeout_seconds: Optional[float] = 200.0
    
    # Validation metrics
    primary_metric: ValidationMetric = ValidationMetric.COMBINED
    secondary_metrics: List[ValidationMetric] = field(default_factory=lambda: [
        ValidationMetric.SILHOUETTE, ValidationMetric.DAVIES_BOULDIN, 
        ValidationMetric.STABILITY_INDEX
    ])
    
    # Threshold constraints
    min_silhouette_floor: float = 0.05
    max_dbi_ceiling: float = 5.0
    min_stability_floor: float = 0.3
    
    # Statistical validation
    enable_permutation_testing: bool = True
    permutation_samples: int = 100
    significance_level: float = 0.05
    
    # Bootstrap validation
    enable_bootstrap_validation: bool = True
    bootstrap_samples: int = 50
    confidence_level: float = 0.95
    
    # Economic validation
    enable_economic_validation: bool = True
    economic_weight: float = 0.3


@dataclass
class DataDrivenClusteringConfig:
    """Main configuration for data-driven clustering parameters."""
    
    # Feature group weight optimization
    feature_weights: FeatureGroupWeightConfig = field(default_factory=FeatureGroupWeightConfig)
    
    # Regime merging threshold optimization
    merging_thresholds: RegimeMergingThresholdConfig = field(default_factory=RegimeMergingThresholdConfig)
    
    # Temporal window optimization
    temporal_windows: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)
    
    # Cluster validation threshold optimization
    validation_thresholds: ClusterValidationThresholdConfig = field(default_factory=ClusterValidationThresholdConfig)
    
    # Global optimization settings
    enable_global_optimization: bool = True
    optimization_order: List[str] = field(default_factory=lambda: [
        'feature_weights', 'temporal_windows', 'merging_thresholds', 'validation_thresholds'
    ])
    
    # Cross-validation
    enable_cross_validation: bool = True
    cv_folds: int = 3
    cv_strategy: str = 'time_series'  # 'time_series', 'kfold', 'stratified'
    
    # Caching and persistence
    enable_caching: bool = True
    cache_directory: str = 'cache/data_driven_clustering'
    save_optimization_results: bool = True
    
    # Logging and monitoring
    enable_detailed_logging: bool = True
    log_optimization_progress: bool = True
    save_optimization_plots: bool = True
    
    # Performance constraints
    max_optimization_time_hours: float = 2.0
    memory_limit_gb: float = 8.0
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def validate(self) -> None:
        """Validate configuration parameters."""
        tprint_debug("🔍 Validating DataDrivenClusteringConfig parameters")
        
        # Validate feature weights
        if self.feature_weights.enable_optimization:
            tprint_debug("Validating feature weights configuration")
            assert 0 < self.feature_weights.min_weight < self.feature_weights.max_weight < 1.0
            assert self.feature_weights.n_trials > 0
            assert self.feature_weights.n_startup_trials < self.feature_weights.n_trials
            tprint_debug(f"Feature weights validation passed: {self.feature_weights.n_trials} trials")
        
        # Validate merging thresholds
        if self.merging_thresholds.enable_optimization:
            tprint_debug("Validating merging thresholds configuration")
            assert 0 < self.merging_thresholds.similarity_threshold_range[0] < self.merging_thresholds.similarity_threshold_range[1] < 1.0
            assert 0 < self.merging_thresholds.distance_threshold_range[0] < self.merging_thresholds.distance_threshold_range[1] < 1.0
            assert 0 < self.merging_thresholds.p_value_threshold_range[0] < self.merging_thresholds.p_value_threshold_range[1] < 1.0
            tprint_debug(f"Merging thresholds validation passed: {self.merging_thresholds.similarity_threshold_range}")
        
        # Validate temporal windows
        if self.temporal_windows.enable_optimization:
            tprint_debug("Validating temporal windows configuration")
            assert 0 < self.temporal_windows.window_size_range[0] < self.temporal_windows.window_size_range[1]
            assert 0 < self.temporal_windows.smoothing_window_range[0] < self.temporal_windows.smoothing_window_range[1]
            tprint_debug(f"Temporal windows validation passed: {self.temporal_windows.window_size_range}")
        
        # Validate validation thresholds
        if self.validation_thresholds.enable_optimization:
            tprint_debug("Validating validation thresholds configuration")
            assert 0 < self.validation_thresholds.min_silhouette_range[0] < self.validation_thresholds.min_silhouette_range[1] < 1.0
            assert 0 < self.validation_thresholds.max_dbi_range[0] < self.validation_thresholds.max_dbi_range[1]
            assert 0 < self.validation_thresholds.min_stability_range[0] < self.validation_thresholds.min_stability_range[1] < 1.0
            tprint_debug(f"Validation thresholds validation passed: {self.validation_thresholds.min_silhouette_range}")
        
        tprint_success("✅ DataDrivenClusteringConfig validation completed successfully")
    
    @tprint_logged(LogLevel.DEBUG, include_result=True)
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        tprint_debug("📋 Converting DataDrivenClusteringConfig to dictionary")
        
        config_dict = {
            'feature_weights': self.feature_weights.__dict__,
            'merging_thresholds': self.merging_thresholds.__dict__,
            'temporal_windows': self.temporal_windows.__dict__,
            'validation_thresholds': self.validation_thresholds.__dict__,
            'enable_global_optimization': self.enable_global_optimization,
            'optimization_order': self.optimization_order,
            'enable_cross_validation': self.enable_cross_validation,
            'cv_folds': self.cv_folds,
            'cv_strategy': self.cv_strategy,
            'enable_caching': self.enable_caching,
            'cache_directory': self.cache_directory,
            'save_optimization_results': self.save_optimization_results,
            'enable_detailed_logging': self.enable_detailed_logging,
            'log_optimization_progress': self.log_optimization_progress,
            'save_optimization_plots': self.save_optimization_plots,
            'max_optimization_time_hours': self.max_optimization_time_hours,
            'memory_limit_gb': self.memory_limit_gb
        }
        
        tprint_debug(f"Configuration converted to dictionary with {len(config_dict)} top-level keys")
        return config_dict
    
    @classmethod
    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataDrivenClusteringConfig':
        """Create configuration from dictionary."""
        tprint_debug("🔧 Creating DataDrivenClusteringConfig from dictionary")
        
        # Create sub-configurations
        tprint_debug("Creating feature weights configuration")
        feature_weights = FeatureGroupWeightConfig(**config_dict.get('feature_weights', {}))
        
        tprint_debug("Creating merging thresholds configuration")
        merging_thresholds = RegimeMergingThresholdConfig(**config_dict.get('merging_thresholds', {}))
        
        tprint_debug("Creating temporal windows configuration")
        temporal_windows = TemporalWindowConfig(**config_dict.get('temporal_windows', {}))
        
        tprint_debug("Creating validation thresholds configuration")
        validation_thresholds = ClusterValidationThresholdConfig(**config_dict.get('validation_thresholds', {}))
        
        # Create main configuration
        tprint_debug("Creating main configuration object")
        config = cls(
            feature_weights=feature_weights,
            merging_thresholds=merging_thresholds,
            temporal_windows=temporal_windows,
            validation_thresholds=validation_thresholds,
            enable_global_optimization=config_dict.get('enable_global_optimization', True),
            optimization_order=config_dict.get('optimization_order', [
                'feature_weights', 'temporal_windows', 'merging_thresholds', 'validation_thresholds'
            ]),
            enable_cross_validation=config_dict.get('enable_cross_validation', True),
            cv_folds=config_dict.get('cv_folds', 3),
            cv_strategy=config_dict.get('cv_strategy', 'time_series'),
            enable_caching=config_dict.get('enable_caching', True),
            cache_directory=config_dict.get('cache_directory', 'cache/data_driven_clustering'),
            save_optimization_results=config_dict.get('save_optimization_results', True),
            enable_detailed_logging=config_dict.get('enable_detailed_logging', True),
            log_optimization_progress=config_dict.get('log_optimization_progress', True),
            save_optimization_plots=config_dict.get('save_optimization_plots', True),
            max_optimization_time_hours=config_dict.get('max_optimization_time_hours', 2.0),
            memory_limit_gb=config_dict.get('memory_limit_gb', 8.0)
        )
        
        tprint_success("✅ DataDrivenClusteringConfig created from dictionary successfully")
        return config