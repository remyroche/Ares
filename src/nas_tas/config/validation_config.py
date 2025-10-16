"""
Validation Configuration for NAS/TAS Systems

This module provides specialized configuration classes for validation, testing,
and evaluation of both NAS and TAS implementations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
from .base_config import ValidationMethod

class ValidationType(Enum):
    """Types of validation to perform."""
    PERFORMANCE = "performance"
    FINANCIAL = "financial"
    REGIME = "regime"
    ROBUSTNESS = "robustness"
    UNCERTAINTY = "uncertainty"
    COMPREHENSIVE = "comprehensive"

class FinancialMetric(Enum):
    """Financial metrics for validation."""
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    WIN_RATE = "win_rate"
    TOTAL_RETURN = "total_return"
    VOLATILITY = "volatility"
    VALUE_AT_RISK = "value_at_risk"
    CONDITIONAL_VAR = "conditional_var"

class PerformanceMetric(Enum):
    """Performance metrics for validation."""
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

class RegimeMetric(Enum):
    """Regime-specific metrics for validation."""
    REGIME_ACCURACY = "regime_accuracy"
    REGIME_STABILITY = "regime_stability"
    REGIME_SEPARATION = "regime_separation"
    REGIME_CONSISTENCY = "regime_consistency"
    REGIME_TRANSITION_ACCURACY = "regime_transition_accuracy"
    ADAPTATION_SPEED = "adaptation_speed"

@dataclass
class ValidationConfig:
    """Base validation configuration class."""

    # Validation type and method
    validation_type: ValidationType = ValidationType.COMPREHENSIVE
    validation_method: ValidationMethod = ValidationMethod.TIME_SERIES_SPLIT

    # Data splitting
    validation_split: float = 0.2
    test_split: float = 0.2
    cv_folds: int = 5

    # Time series specific
    time_series_gap: int = 0
    walk_forward_window: int = 100
    walk_forward_step: int = 10

    # Validation metrics
    performance_metrics: List[PerformanceMetric] = field(default_factory=lambda: [
        PerformanceMetric.ACCURACY,
        PerformanceMetric.PRECISION,
        PerformanceMetric.RECALL,
        PerformanceMetric.F1_SCORE
    ])

    financial_metrics: List[FinancialMetric] = field(default_factory=lambda: [
        FinancialMetric.SHARPE_RATIO,
        FinancialMetric.MAX_DRAWDOWN,
        FinancialMetric.SORTINO_RATIO,
        FinancialMetric.WIN_RATE
    ])

    regime_metrics: List[RegimeMetric] = field(default_factory=lambda: [
        RegimeMetric.REGIME_ACCURACY,
        RegimeMetric.REGIME_STABILITY,
        RegimeMetric.ADAPTATION_SPEED
    ])

    # Thresholds for validation
    min_accuracy_threshold: float = 0.6
    min_sharpe_ratio_threshold: float = 1.0
    max_drawdown_threshold: float = 0.15
    min_regime_accuracy_threshold: float = 0.7

    # Statistical validation
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bootstrap_samples: int = 1000

    # Performance requirements
    min_sample_size: int = 100
    max_evaluation_time_seconds: int = 300
    enable_early_stopping: bool = True

    # Custom parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            elif isinstance(field_value, list) and field_value and isinstance(field_value[0], Enum):
                config_dict[field_name] = [item.value for item in field_value]
            else:
                config_dict[field_name] = field_value
        return config_dict

@dataclass
class FinancialValidationConfig(ValidationConfig):
    """Configuration for financial validation."""

    validation_type: ValidationType = ValidationType.FINANCIAL

    # Trading simulation parameters
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    risk_free_rate: float = 0.02

    # Position sizing
    min_position_size: float = 0.01
    max_position_size: float = 0.1
    position_sizing_method: str = "fixed"  # fixed, kelly, risk_parity

    # Risk management
    stop_loss_threshold: float = 0.05
    take_profit_threshold: float = 0.1
    max_concurrent_positions: int = 5

    # Performance evaluation periods
    evaluation_periods: List[str] = field(default_factory=lambda: [
        "1d", "1w", "1M", "3M", "6M", "1Y"
    ])

    # Benchmark comparison
    enable_benchmark_comparison: bool = True
    benchmark_symbol: str = "SPY"

    # Financial thresholds
    min_annual_return: float = 0.05
    max_volatility: float = 0.25
    min_profit_factor: float = 1.2
    min_win_rate: float = 0.4

    # Stress testing
    enable_stress_testing: bool = True
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        "market_crash", "high_volatility", "low_volatility", "trending_market"
    ])

@dataclass
class PerformanceValidationConfig(ValidationConfig):
    """Configuration for performance validation."""

    validation_type: ValidationType = ValidationType.PERFORMANCE

    # Cross-validation parameters
    cv_strategy: str = "stratified"  # stratified, kfold, time_series
    cv_scoring: str = "accuracy"

    # Model evaluation
    enable_learning_curves: bool = True
    enable_validation_curves: bool = True
    enable_confusion_matrix: bool = True
    enable_classification_report: bool = True

    # Statistical tests
    enable_statistical_tests: bool = True
    statistical_tests: List[str] = field(default_factory=lambda: [
        "t_test", "wilcoxon", "mann_whitney"
    ])

    # Performance thresholds
    min_accuracy: float = 0.7
    min_precision: float = 0.6
    min_recall: float = 0.6
    min_f1_score: float = 0.6
    min_roc_auc: float = 0.7

    # Overfitting detection
    enable_overfitting_detection: bool = True
    overfitting_threshold: float = 0.1  # Max difference between train and validation
    overfitting_patience: int = 5

    # Model comparison
    enable_model_comparison: bool = True
    comparison_models: List[str] = field(default_factory=lambda: [
        "random_forest", "xgboost", "lightgbm", "logistic_regression"
    ])

@dataclass
class RegimeValidationConfig(ValidationConfig):
    """Configuration for regime-specific validation."""

    validation_type: ValidationType = ValidationType.REGIME

    # Regime detection validation
    regime_detection_methods: List[str] = field(default_factory=lambda: [
        "clustering", "changepoint", "hmm", "markov_switching"
    ])

    # Regime stability analysis
    stability_window: int = 20
    stability_threshold: float = 0.7
    transition_window: int = 10

    # Regime-specific performance
    min_regimes: int = 3
    max_regimes: int = 15
    regime_balance_threshold: float = 0.1  # Min proportion per regime

    # Adaptation testing
    adaptation_scenarios: List[str] = field(default_factory=lambda: [
        "gradual_change", "sudden_change", "cyclic_change", "trend_change"
    ])

    # Regime transition analysis
    enable_transition_analysis: bool = True
    transition_probability_threshold: float = 0.1
    transition_duration_threshold: int = 5

    # Regime-specific thresholds
    min_regime_accuracy: float = 0.6
    min_regime_stability: float = 0.7
    max_regime_volatility: float = 0.3
    min_regime_separation: float = 0.5

@dataclass
class RobustnessValidationConfig(ValidationConfig):
    """Configuration for robustness validation."""

    validation_type: ValidationType = ValidationType.ROBUSTNESS

    # Noise and perturbation testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    perturbation_types: List[str] = field(default_factory=lambda: [
        "gaussian_noise", "uniform_noise", "outliers", "missing_data"
    ])

    # Adversarial testing
    enable_adversarial_testing: bool = True
    adversarial_epsilon: float = 0.01
    adversarial_iterations: int = 10

    # Distribution shift testing
    enable_distribution_shift: bool = True
    shift_types: List[str] = field(default_factory=lambda: [
        "covariate_shift", "label_shift", "concept_drift"
    ])

    # Temporal robustness
    enable_temporal_robustness: bool = True
    temporal_windows: List[int] = field(default_factory=lambda: [7, 30, 90, 365])

    # Robustness thresholds
    min_robustness_score: float = 0.7
    max_performance_degradation: float = 0.2
    min_correlation_stability: float = 0.8

@dataclass
class UncertaintyValidationConfig(ValidationConfig):
    """Configuration for uncertainty validation."""

    validation_type: ValidationType = ValidationType.UNCERTAINTY

    # Uncertainty estimation methods
    uncertainty_methods: List[str] = field(default_factory=lambda: [
        "ensemble", "dropout", "bayesian", "bootstrap"
    ])

    # Calibration testing
    enable_calibration_testing: bool = True
    calibration_bins: int = 10
    calibration_methods: List[str] = field(default_factory=lambda: [
        "platt_scaling", "isotonic_regression", "temperature_scaling"
    ])

    # Confidence interval analysis
    confidence_levels: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95, 0.99])
    interval_methods: List[str] = field(default_factory=lambda: [
        "bootstrap", "jackknife", "delta_method"
    ])

    # Reliability metrics
    enable_reliability_metrics: bool = True
    reliability_metrics: List[str] = field(default_factory=lambda: [
        "confidence_calibration", "reliability_diagram", "brier_score"
    ])

    # Uncertainty thresholds
    min_confidence_calibration: float = 0.8
    max_brier_score: float = 0.25
    min_interval_coverage: float = 0.9

def create_validation_config(
    validation_type: ValidationType = ValidationType.COMPREHENSIVE,
    **kwargs
) -> ValidationConfig:
    """
    Factory function to create appropriate validation configuration.

    Args:
        validation_type: The type of validation to perform
        **kwargs: Additional configuration parameters

    Returns:
        Appropriate ValidationConfig instance
    """
    base_params = {
        'validation_type': validation_type,
        **kwargs
    }

    if validation_type == ValidationType.FINANCIAL:
        return FinancialValidationConfig(**base_params)
    elif validation_type == ValidationType.PERFORMANCE:
        return PerformanceValidationConfig(**base_params)
    elif validation_type == ValidationType.REGIME:
        return RegimeValidationConfig(**base_params)
    elif validation_type == ValidationType.ROBUSTNESS:
        return RobustnessValidationConfig(**base_params)
    elif validation_type == ValidationType.UNCERTAINTY:
        return UncertaintyValidationConfig(**base_params)
    else:
        return ValidationConfig(**base_params)

def create_comprehensive_validation_config(**kwargs) -> ValidationConfig:
    """Create comprehensive validation configuration."""
    return ValidationConfig(
        validation_type=ValidationType.COMPREHENSIVE,
        **kwargs
    )

def create_quick_validation_config(**kwargs) -> ValidationConfig:
    """Create quick validation configuration for fast testing."""
    return ValidationConfig(
        validation_type=ValidationType.PERFORMANCE,
        cv_folds=3,
        bootstrap_samples=100,
        max_evaluation_time_seconds=60,
        **kwargs
    )
