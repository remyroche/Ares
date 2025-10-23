"""
Enhanced Configuration Classes for Advanced Feature Selection

This module provides comprehensive configuration classes for the enhanced
feature selection methods with adaptive weighting, confidence scoring, and
native validation framework integration.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np

@dataclass
class AdaptiveWeightingConfig:
    """Configuration for adaptive weighting based on performance."""
    enable_adaptive_weighting: bool = True
    performance_metric: str = 'r2'  # 'r2', 'mse', 'accuracy', 'f1'
    weight_smoothing: float = 0.1  # Smoothing factor for weight updates
    min_weight: float = 0.1  # Minimum weight for any method
    max_weight: float = 0.8  # Maximum weight for any method
    weight_update_frequency: int = 10  # Update weights every N selections
    enable_weight_history: bool = True
    weight_decay: float = 0.95  # Decay factor for historical weights

@dataclass
class ConfidenceScoringConfig:
    """Configuration for confidence scoring based on method agreement."""
    enable_confidence_scoring: bool = True
    agreement_threshold: float = 0.5  # Minimum agreement for high confidence
    consensus_bonus: float = 0.2  # Bonus for features selected by multiple methods
    stability_weight: float = 0.3  # Weight for stability in confidence calculation
    performance_weight: float = 0.4  # Weight for performance in confidence calculation
    agreement_weight: float = 0.3  # Weight for agreement in confidence calculation
    min_confidence: float = 0.1  # Minimum confidence score
    max_confidence: float = 1.0  # Maximum confidence score

@dataclass
class NativeValidationConfig:
    """Configuration for native validation framework integration."""
    enable_native_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = 'kfold'  # 'kfold', 'timeseries', 'stratified'
    enable_stability_metrics: bool = True
    stability_n_bootstrap: int = 10
    stability_threshold: float = 0.8
    enable_consensus_scoring: bool = True
    consensus_min_methods: int = 2  # Minimum methods for consensus
    enable_performance_validation: bool = True
    performance_validation_models: List[str] = field(default_factory=lambda: ['linear', 'random_forest'])
    validation_timeout: float = 300.0  # Timeout in seconds

@dataclass
class DynamicFeatureSelectionConfig:
    """Configuration for dynamic feature selection."""
    enable_dynamic_selection: bool = True
    default_target_type: str = 'percentage'  # 'absolute', 'percentage', 'performance_threshold'
    default_target_value: float = 0.5  # 50% of features by default
    enable_elbow_method: bool = True
    elbow_method_range: Tuple[int, int] = (5, 50)  # Range for elbow method
    elbow_method_step: int = 5
    enable_statistical_thresholding: bool = True
    statistical_significance_level: float = 0.05
    enable_progressive_selection: bool = False  # Not implemented as requested
    performance_degradation_threshold: float = 0.05  # 5% degradation threshold

@dataclass
class ElbowMethodConfig:
    """Configuration for elbow method feature count detection."""
    enable_elbow_method: bool = True
    min_features: int = 5
    max_features: int = 50
    step_size: int = 5
    scoring_metric: str = 'r2'
    elbow_detection_method: str = 'curvature'  # 'curvature', 'knee', 'elbow'
    curvature_threshold: float = 0.1
    enable_plotting: bool = False
    plot_save_path: Optional[str] = None

@dataclass
class StatisticalThresholdingConfig:
    """Configuration for statistical significance testing."""
    enable_statistical_thresholding: bool = True
    significance_level: float = 0.05
    test_method: str = 'permutation'  # 'permutation', 'bootstrap', 't_test'
    n_permutations: int = 1000
    correction_method: str = 'fdr'  # 'fdr', 'bonferroni', 'holm'
    min_p_value: float = 1e-6
    enable_effect_size: bool = True
    effect_size_threshold: float = 0.1

@dataclass
class EnhancedEnsembleConfig:
    """Enhanced configuration for ensemble advanced selector."""
    # Adaptive weighting
    adaptive_weighting: AdaptiveWeightingConfig = field(default_factory=AdaptiveWeightingConfig)

    # Confidence scoring
    confidence_scoring: ConfidenceScoringConfig = field(default_factory=ConfidenceScoringConfig)

    # Native validation
    native_validation: NativeValidationConfig = field(default_factory=NativeValidationConfig)

    # Dynamic feature selection
    dynamic_selection: DynamicFeatureSelectionConfig = field(default_factory=DynamicFeatureSelectionConfig)

    # Elbow method
    elbow_method: ElbowMethodConfig = field(default_factory=ElbowMethodConfig)

    # Statistical thresholding
    statistical_thresholding: StatisticalThresholdingConfig = field(default_factory=StatisticalThresholdingConfig)

    # General settings
    random_state: int = 42
    n_jobs: int = -1
    enable_hardware_optimization: bool = True
    enable_logging: bool = True
    log_level: str = 'INFO'

@dataclass
class EnhancedAdvancedConfig:
    """Enhanced configuration for advanced feature selector."""
    # Ensemble configuration
    ensemble_config: EnhancedEnsembleConfig = field(default_factory=EnhancedEnsembleConfig)

    # Method selection
    enable_auto_method_selection: bool = True
    method_selection_metrics: List[str] = field(default_factory=lambda: ['r2', 'mse', 'stability'])

    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_interval: int = 10  # Monitor every N selections
    performance_history_size: int = 100

    # Error handling
    enable_error_recovery: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0

    # General settings
    random_state: int = 42
    n_jobs: int = -1
    enable_hardware_optimization: bool = True
