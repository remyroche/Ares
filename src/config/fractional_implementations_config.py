# src/config/fractional_implementations_config.py

"""Configuration for fractional labeling and fractional differentiation implementations."""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class FractionalLabelingConfig:
    pass  # TODO: Add implementation
class FractionalLabelingConfig:
class FractionalLabelingConfig:
    """Configuration for fractional triple barrier labeling."""

# Enable/disable fractional labeling
enable_fractional_labels: bool = True

# Component weights for fractional label calculation
distance_weight: float = 0.4
time_weight: float = 0.3
volatility_weight: float = 0.3

# Confidence thresholds
min_confidence_threshold: float = 0.1
max_confidence_threshold: float = 0.95

# Component enablement
enable_distance_scaling: bool = True
enable_time_decay: bool = True
enable_volatility_normalization: bool = True
enable_regime_scaling: bool = False

# Regime-specific configurations
regime_specific_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
"trending": {
"distance_weight": 0.5,
"time_weight": 0.3,
"volatility_weight": 0.2,
"min_confidence_threshold": 0.15,
},
"ranging": {
"distance_weight": 0.3,
"time_weight": 0.4,
"volatility_weight": 0.3,
"min_confidence_threshold": 0.1,
},
"volatile": {
"distance_weight": 0.2,
"time_weight": 0.2,
"volatility_weight": 0.6,
"min_confidence_threshold": 0.2,
}
})


@dataclass
class FractionalDifferentiationConfig:
    pass  # TODO: Add implementation
class FractionalDifferentiationConfig:
class FractionalDifferentiationConfig:
    """Configuration for fractional differentiation."""

# Enable/disable fractional differentiation
enable_fractional_diff: bool = True

# Default fractional order
default_d: float = 0.5

# Optimization settings
optimize_order: bool = True
min_d: float = 0.1
max_d: float = 0.9
optimization_steps: int = 10

# Computational settings
window: int = 100
threshold: float = 1e-5

# Column configurations
price_columns: List[str] = field(default_factory=lambda: ["close", "high", "low", "open"])
volume_columns: List[str] = field(default_factory=lambda: ["volume"])
exclude_columns: List[str] = field(default_factory=lambda: ["timestamp", "datetime", "date"])

# Performance settings
enable_batch_processing: bool = True
enable_parallel_processing: bool = True
max_parallel_workers: int = 4


@dataclass
class FractionalImplementationsConfig:
    pass  # TODO: Add implementation
class FractionalImplementationsConfig:
class FractionalImplementationsConfig:
    """Main configuration for fractional implementations."""

# General settings
enable_fractional_implementations: bool = True
enable_gradual_rollout: bool = True
enable_performance_monitoring: bool = True

# Phase settings
current_phase: str = "phase1"  # phase1, phase2, phase3, phase4

# Performance targets
target_sharpe_improvement: float = 0.15  # 15%
target_drawdown_reduction: float = 0.20  # 20%
target_accuracy_improvement: float = 0.10  # 10%

# Monitoring settings
performance_check_interval: int = 1000  # samples
alert_threshold: float = 0.05  # 5% performance degradation

# Logging settings
enable_detailed_logging: bool = True
log_performance_metrics: bool = True
log_feature_statistics: bool = True

# Testing settings
enable_comprehensive_testing: bool = True
test_data_size: int = 10000
validation_split: float = 0.2

# Sub-configurations
fractional_labeling: FractionalLabelingConfig = field(default_factory=FractionalLabelingConfig)
fractional_differentiation: FractionalDifferentiationConfig = field(default_factory=FractionalDifferentiationConfig)


# Default configuration instance
DEFAULT_FRACTIONAL_CONFIG = FractionalImplementationsConfig()


def get_fractional_config(config_dict: Dict[str, Any] = None) -> FractionalImplementationsConfig:
    """Get fractional implementations configuration.

Args:
        config_dict: Optional configuration dictionary to override defaults

Returns:
        FractionalImplementationsConfig instance
"""
if config_dict is None:
        return DEFAULT_FRACTIONAL_CONFIG

# Create config from dictionary
config = FractionalImplementationsConfig()

# Update general settings
for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

# Update sub-configurations
if "fractional_labeling" in config_dict:
        for key, value in config_dict["fractional_labeling"].items():
            if hasattr(config.fractional_labeling, key):
                setattr(config.fractional_labeling, key, value)

if "fractional_differentiation" in config_dict:
        for key, value in config_dict["fractional_differentiation"].items():
            if hasattr(config.fractional_differentiation, key):
                setattr(config.fractional_differentiation, key, value)

return config


def validate_fractional_config(config: FractionalImplementationsConfig) -> List[str]:
    """Validate fractional implementations configuration.

Args:
        config: Configuration to validate

Returns:
        List of validation errors (empty if valid)
"""
errors = []

# Validate fractional labeling config
if config.fractional_labeling.enable_fractional_labels:
        if not (0 <= config.fractional_labeling.distance_weight <= 1):
            errors.append("distance_weight must be between 0 and 1")
if not (0 <= config.fractional_labeling.time_weight <= 1):
            errors.append("time_weight must be between 0 and 1")
if not (0 <= config.fractional_labeling.volatility_weight <= 1):
            errors.append("volatility_weight must be between 0 and 1")

total_weight = (config.fractional_labeling.distance_weight +
config.fractional_labeling.time_weight +
config.fractional_labeling.volatility_weight)
if abs(total_weight - 1.0) > 1e-6:
            errors.append("Component weights must sum to 1.0")

if not (0 <= config.fractional_labeling.min_confidence_threshold <=
config.fractional_labeling.max_confidence_threshold <= 1):
            errors.append("Confidence thresholds must be between 0 and 1, with min <= max")

# Validate fractional differentiation config
if config.fractional_differentiation.enable_fractional_diff:
        if not (0 < config.fractional_differentiation.default_d < 1):
            errors.append("default_d must be between 0 and 1")
if not (0 < config.fractional_differentiation.min_d <
config.fractional_differentiation.max_d < 1):
            errors.append("min_d must be < max_d, both between 0 and 1")
if config.fractional_differentiation.window <= 0:
            errors.append("window must be positive")
if config.fractional_differentiation.threshold <= 0:
            errors.append("threshold must be positive")

# Validate performance targets
if config.target_sharpe_improvement <= 0:
        errors.append("target_sharpe_improvement must be positive")
if config.target_drawdown_reduction <= 0:
        errors.append("target_drawdown_reduction must be positive")
if config.target_accuracy_improvement <= 0:
        errors.append("target_accuracy_improvement must be positive")

return errors