"""
import warnings
Configuration System for Feature Lookback Optimization

This module provides a comprehensive configuration system for feature lookback
optimization parameters, including validation, defaults, and environment-specific settings.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Optimization methods for feature lookback periods."""
    SIGNAL_STRENGTH = "signal_strength"
    NOISE_REDUCTION = "noise_reduction"
    TREND_FOLLOWING = "trend_following"
    INFORMATION_CONTENT = "information_content"
    REGIME_ADAPTATION = "regime_adaptation"
    CROSS_VALIDATION = "cross_validation"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ADAPTIVE = "adaptive"

class ValidationLevel(Enum):
    """Validation levels for optimization results."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"

@dataclass
class FeatureOptimizationConfig:
    """Configuration for individual feature optimization."""
    name: str
    periods: List[int]
    method: OptimizationMethod
    weight: float = 1.0
    enabled: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationSystemConfig:
    """Comprehensive configuration for feature lookback optimization system."""

    # General settings
    optimization_method: OptimizationMethod = OptimizationMethod.STATISTICAL_ANALYSIS
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    parallel_processing: bool = True
    max_workers: int = 4

    # Data settings
    min_lookback: int = 5
    max_lookback: int = 252
    step_size: int = 1
    min_data_points: int = 100

    # Performance settings
    performance_threshold: float = 0.3
    stability_threshold: float = 0.5
    confidence_level: float = 0.95

    # Feature configurations
    features: List[FeatureOptimizationConfig] = field(default_factory=list)

    # Validation settings
    enable_validation: bool = True
    enable_performance_metrics: bool = True
    enable_recommendations: bool = True

    # Output settings
    save_results: bool = True
    save_metrics: bool = True
    output_directory: str = "optimization_results"

    # Advanced settings
    memory_efficient: bool = True
    chunk_size: int = 1000
    cache_results: bool = True
    max_cache_size: int = 100

    def __post_init__(self):
        """Initialize default feature configurations if none provided."""
        if not self.features:
            self.features = self._get_default_feature_configs()

    def _get_default_feature_configs(self) -> List[FeatureOptimizationConfig]:
        """Get default feature configurations."""
        return [
            FeatureOptimizationConfig(
                name="rsi",
                periods=[7, 14, 21, 28],
                method=OptimizationMethod.SIGNAL_STRENGTH,
                weight=1.0
            ),
            FeatureOptimizationConfig(
                name="sma",
                periods=[10, 20, 30, 50],
                method=OptimizationMethod.NOISE_REDUCTION,
                weight=1.0
            ),
            FeatureOptimizationConfig(
                name="ema",
                periods=[8, 12, 20, 26],
                method=OptimizationMethod.TREND_FOLLOWING,
                weight=1.0
            ),
            FeatureOptimizationConfig(
                name="bollinger_bands",
                periods=[15, 20, 25, 30],
                method=OptimizationMethod.INFORMATION_CONTENT,
                weight=0.8
            ),
            FeatureOptimizationConfig(
                name="macd",
                periods=[7, 9, 12, 15],
                method=OptimizationMethod.SIGNAL_STRENGTH,
                weight=0.9
            ),
            FeatureOptimizationConfig(
                name="volatility",
                periods=[10, 15, 20, 25],
                method=OptimizationMethod.REGIME_ADAPTATION,
                weight=0.7
            )
        ]

    def get_enabled_features(self) -> List[FeatureOptimizationConfig]:
        """Get list of enabled feature configurations."""
        return [f for f in self.features if f.enabled]

    def get_feature_config(self, feature_name: str) -> Optional[FeatureOptimizationConfig]:
        """Get configuration for a specific feature."""
        for feature in self.features:
            if feature.name == feature_name:
                return feature
        return None

    def update_feature_config(self, feature_name: str, **kwargs) -> bool:
        """Update configuration for a specific feature."""
        feature = self.get_feature_config(feature_name)
        if feature:
            for key, value in kwargs.items():
                if hasattr(feature, key):
                    setattr(feature, key, value)
            return True
        return False

    def add_feature_config(self, feature_config: FeatureOptimizationConfig) -> None:
        """Add a new feature configuration."""
        # Remove existing config if it exists
        self.features = [f for f in self.features if f.name != feature_config.name]
        self.features.append(feature_config)

    def remove_feature_config(self, feature_name: str) -> bool:
        """Remove a feature configuration."""
        original_count = len(self.features)
        self.features = [f for f in self.features if f.name != feature_name]
        return len(self.features) < original_count

    def validate_config(self) -> List[str]:
        """Validate the configuration and return any errors."""
        errors = []

        # Validate general settings
        if self.min_lookback >= self.max_lookback:
            errors.append("min_lookback must be less than max_lookback")

        if self.step_size <= 0:
            errors.append("step_size must be positive")

        if self.min_data_points < 10:
            errors.append("min_data_points should be at least 10")

        if self.max_workers < 1:
            errors.append("max_workers must be at least 1")

        # Validate thresholds
        if not 0 <= self.performance_threshold <= 1:
            errors.append("performance_threshold must be between 0 and 1")

        if not 0 <= self.stability_threshold <= 1:
            errors.append("stability_threshold must be between 0 and 1")

        if not 0 <= self.confidence_level <= 1:
            errors.append("confidence_level must be between 0 and 1")

        # Validate feature configurations
        feature_names = set()
        for feature in self.features:
            if feature.name in feature_names:
                errors.append(f"Duplicate feature name: {feature.name}")
            feature_names.add(feature.name)

            if not feature.periods:
                errors.append(f"No periods specified for feature: {feature.name}")

            if any(p <= 0 for p in feature.periods):
                errors.append(f"Invalid periods for feature {feature.name}: {feature.periods}")

            if not 0 <= feature.weight <= 2:
                errors.append(f"Invalid weight for feature {feature.name}: {feature.weight}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        # Convert enums to strings
        config_dict['optimization_method'] = self.optimization_method.value
        config_dict['validation_level'] = self.validation_level.value

        for feature in config_dict['features']:
            feature['method'] = feature['method'].value if hasattr(feature['method'], 'value') else feature['method']

        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OptimizationSystemConfig':
        """Create configuration from dictionary."""
        # Convert string enums back to enum objects
        if 'optimization_method' in config_dict:
            config_dict['optimization_method'] = OptimizationMethod(config_dict['optimization_method'])

        if 'validation_level' in config_dict:
            config_dict['validation_level'] = ValidationLevel(config_dict['validation_level'])

        # Convert feature configs
        if 'features' in config_dict:
            features = []
            for feature_dict in config_dict['features']:
                if 'method' in feature_dict:
                    feature_dict['method'] = OptimizationMethod(feature_dict['method'])
                features.append(FeatureOptimizationConfig(**feature_dict))
            config_dict['features'] = features

        return cls(**config_dict)

    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to JSON file."""
        try:
            config_dict = self.to_dict()

            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False

    @classmethod
    def load_from_file(cls, filepath: str) -> Optional['OptimizationSystemConfig']:
        """Load configuration from JSON file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)

            config = cls.from_dict(config_dict)
            logger.info(f"Configuration loaded from {filepath}")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return None

class OptimizationConfigManager:
    """Manager for optimization configurations."""

    def __init__(self, config_dir: str = "config/optimization"):
        """Initialize the configuration manager."""
        self.logger = logger.getChild('OptimizationConfigManager')
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.default_config = OptimizationSystemConfig()
        self.current_config = self.default_config

        self.logger.info(f"Initialized OptimizationConfigManager with config directory: {self.config_dir}")

    def load_config(self, config_name: str = "default") -> Optional[OptimizationSystemConfig]:
        """Load a configuration by name."""
        config_file = self.config_dir / f"{config_name}.json"

        if config_file.exists():
            config = OptimizationSystemConfig.load_from_file(str(config_file))
            if config:
                self.current_config = config
                self.logger.info(f"Loaded configuration: {config_name}")
                return config
        else:
            self.logger.warning(f"Configuration file not found: {config_file}")

        return None

    def save_config(self, config: OptimizationSystemConfig, config_name: str = "default") -> bool:
        """Save a configuration with a given name."""
        config_file = self.config_dir / f"{config_name}.json"

        if config.save_to_file(str(config_file)):
            self.current_config = config
            self.logger.info(f"Saved configuration: {config_name}")
            return True

        return False

    def get_current_config(self) -> OptimizationSystemConfig:
        """Get the current configuration."""
        return self.current_config

    def update_current_config(self, **kwargs) -> bool:
        """Update the current configuration with new values."""
        try:
            for key, value in kwargs.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                else:
                    self.logger.warning(f"Unknown configuration parameter: {key}")

            # Validate updated configuration
            errors = self.current_config.validate_config()
            if errors:
                self.logger.error(f"Configuration validation errors: {errors}")
                return False

            self.logger.info("Configuration updated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return False

    def list_configs(self) -> List[str]:
        """List available configuration files."""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]

    def create_environment_config(self, environment: str) -> OptimizationSystemConfig:
        """Create environment-specific configuration."""
        if environment == "development":
            config = OptimizationSystemConfig(
                validation_level=ValidationLevel.BASIC,
                parallel_processing=False,
                max_workers=2,
                enable_performance_metrics=True,
                save_results=False
            )
        elif environment == "testing":
            config = OptimizationSystemConfig(
                validation_level=ValidationLevel.STANDARD,
                parallel_processing=True,
                max_workers=2,
                min_lookback=3,
                max_lookback=20,
                step_size=2,
                enable_performance_metrics=True,
                save_results=True
            )
        elif environment == "production":
            config = OptimizationSystemConfig(
                validation_level=ValidationLevel.COMPREHENSIVE,
                parallel_processing=True,
                max_workers=8,
                enable_performance_metrics=True,
                save_results=True,
                save_metrics=True,
                cache_results=True
            )
        else:
            config = self.default_config

        self.logger.info(f"Created {environment} configuration")
        return config

# Convenience functions
def get_default_config() -> OptimizationSystemConfig:
    """Get the default optimization configuration."""
    return OptimizationSystemConfig()

def load_config_from_file(filepath: str) -> Optional[OptimizationSystemConfig]:
    """Load configuration from file."""
    return OptimizationSystemConfig.load_from_file(filepath)

def create_config_for_environment(environment: str) -> OptimizationSystemConfig:
    """Create configuration for specific environment."""
    manager = OptimizationConfigManager()
    return manager.create_environment_config(environment)

def validate_config_file(filepath: str) -> Tuple[bool, List[str]]:
    """Validate a configuration file."""
    config = load_config_from_file(filepath)
    if config:
        errors = config.validate_config()
        return len(errors) == 0, errors
    else:
        return False, ["Failed to load configuration file"]
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and
                VECTORBT_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
