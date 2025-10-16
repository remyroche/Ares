"""
Unified Configuration Manager

This module provides a unified configuration management system that combines
configuration classes from both TAS and NAS regime detection systems.

Features:
- Unified configuration classes
- Parameter validation
- Common default values
- Configuration serialization
- Environment-based configuration
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    PYTHON = "python"

@dataclass
class UnifiedRegimeConfig:
    """Unified configuration for regime detection systems."""

    # System configuration
    system_name: str = "unified_regime_system"
    version: str = "1.0.0"
    environment: str = "development"

    # Regime detection parameters
    n_regimes: int = 3
    regime_detection_method: str = "hybrid"
    enable_tree_based: bool = True
    enable_neural_based: bool = True

    # Economic evaluation
    economic_significance_threshold: float = 0.6
    trading_viability_threshold: float = 0.6
    enable_economic_indicators: bool = True

    # Optimization parameters
    max_iterations: int = 100
    population_size: int = 50
    convergence_threshold: float = 1e-6

    # Hardware optimization
    enable_hardware_optimization: bool = True
    max_memory_usage_gb: float = 8.0
    enable_gpu_acceleration: bool = True

    # Analysis parameters
    stability_window: int = 20
    transition_window: int = 10
    uncertainty_method: str = "entropy"

    # Meta-learning
    enable_meta_learning: bool = True
    adaptation_rate: float = 0.1
    learning_threshold: float = 0.05

    # Logging and monitoring
    log_level: str = "INFO"
    enable_performance_monitoring: bool = True
    monitoring_interval: int = 100

    # Advanced features
    enable_uncertainty_quantification: bool = True
    enable_bootstrap_analysis: bool = True
    enable_position_aware_analysis: bool = True

    # Data processing
    data_preprocessing: Dict[str, Any] = field(default_factory=lambda: {
        'normalize': True,
        'standardize': True,
        'handle_missing': True,
        'outlier_detection': True
    })

    # Model parameters
    model_parameters: Dict[str, Any] = field(default_factory=lambda: {
        'learning_rate': 0.001,
        'batch_size': 64,
        'dropout_rate': 0.2,
        'hidden_size': 128
    })

class UnifiedConfigManager:
    """
    Unified Configuration Manager.

    Manages configuration for both TAS and NAS regime detection systems.
    """

    def __init__(self, config: Optional[UnifiedRegimeConfig] = None):
        """Initialize unified configuration manager."""
        self.config = config or UnifiedRegimeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info("✅ Unified Configuration Manager initialized")
        self.logger.info(f"   System: {self.config.system_name}")
        self.logger.info(f"   Version: {self.config.version}")
        self.logger.info(f"   Environment: {self.config.environment}")

    def load_from_file(self, filepath: Union[str, Path], format: ConfigFormat = ConfigFormat.JSON) -> 'UnifiedConfigManager':
        """Load configuration from file."""
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                self.logger.warning(f"Configuration file not found: {filepath}")
                return self

            with open(filepath, 'r') as f:
                if format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            # Update configuration
            self._update_config_from_dict(config_data)

            self.logger.info(f"✅ Configuration loaded from {filepath}")
            return self

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {filepath}: {e}")
            return self

    def save_to_file(self, filepath: Union[str, Path], format: ConfigFormat = ConfigFormat.JSON) -> bool:
        """Save configuration to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            config_dict = asdict(self.config)

            with open(filepath, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            self.logger.info(f"✅ Configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {filepath}: {e}")
            return False

    def load_from_environment(self) -> 'UnifiedConfigManager':
        """Load configuration from environment variables."""
        try:
            # System configuration
            if 'REGIME_SYSTEM_NAME' in os.environ:
                self.config.system_name = os.environ['REGIME_SYSTEM_NAME']

            if 'REGIME_VERSION' in os.environ:
                self.config.version = os.environ['REGIME_VERSION']

            if 'REGIME_ENVIRONMENT' in os.environ:
                self.config.environment = os.environ['REGIME_ENVIRONMENT']

            # Regime detection parameters
            if 'REGIME_N_REGIMES' in os.environ:
                self.config.n_regimes = int(os.environ['REGIME_N_REGIMES'])

            if 'REGIME_DETECTION_METHOD' in os.environ:
                self.config.regime_detection_method = os.environ['REGIME_DETECTION_METHOD']

            # Economic evaluation
            if 'ECONOMIC_SIGNIFICANCE_THRESHOLD' in os.environ:
                self.config.economic_significance_threshold = float(os.environ['ECONOMIC_SIGNIFICANCE_THRESHOLD'])

            if 'TRADING_VIABILITY_THRESHOLD' in os.environ:
                self.config.trading_viability_threshold = float(os.environ['TRADING_VIABILITY_THRESHOLD'])

            # Optimization parameters
            if 'MAX_ITERATIONS' in os.environ:
                self.config.max_iterations = int(os.environ['MAX_ITERATIONS'])

            if 'POPULATION_SIZE' in os.environ:
                self.config.population_size = int(os.environ['POPULATION_SIZE'])

            # Hardware optimization
            if 'ENABLE_HARDWARE_OPTIMIZATION' in os.environ:
                self.config.enable_hardware_optimization = os.environ['ENABLE_HARDWARE_OPTIMIZATION'].lower() == 'true'

            if 'MAX_MEMORY_USAGE_GB' in os.environ:
                self.config.max_memory_usage_gb = float(os.environ['MAX_MEMORY_USAGE_GB'])

            # Logging
            if 'LOG_LEVEL' in os.environ:
                self.config.log_level = os.environ['LOG_LEVEL']

            self.logger.info("✅ Configuration loaded from environment variables")
            return self

        except Exception as e:
            self.logger.error(f"Failed to load configuration from environment: {e}")
            return self

    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        try:
            for key, value in config_dict.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    self.logger.warning(f"Unknown configuration parameter: {key}")

        except Exception as e:
            self.logger.error(f"Failed to update configuration from dictionary: {e}")

    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        try:
            # Validate system parameters
            if self.config.n_regimes < 2:
                self.logger.error("Number of regimes must be at least 2")
                return False

            if self.config.economic_significance_threshold < 0 or self.config.economic_significance_threshold > 1:
                self.logger.error("Economic significance threshold must be between 0 and 1")
                return False

            if self.config.trading_viability_threshold < 0 or self.config.trading_viability_threshold > 1:
                self.logger.error("Trading viability threshold must be between 0 and 1")
                return False

            # Validate optimization parameters
            if self.config.max_iterations < 1:
                self.logger.error("Max iterations must be at least 1")
                return False

            if self.config.population_size < 1:
                self.logger.error("Population size must be at least 1")
                return False

            # Validate hardware parameters
            if self.config.max_memory_usage_gb < 0.1:
                self.logger.error("Max memory usage must be at least 0.1 GB")
                return False

            # Validate analysis parameters
            if self.config.stability_window < 1:
                self.logger.error("Stability window must be at least 1")
                return False

            if self.config.transition_window < 1:
                self.logger.error("Transition window must be at least 1")
                return False

            self.logger.info("✅ Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return asdict(self.config)

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'system_name': self.config.system_name,
            'version': self.config.version,
            'environment': self.config.environment,
            'n_regimes': self.config.n_regimes,
            'regime_detection_method': self.config.regime_detection_method,
            'economic_significance_threshold': self.config.economic_significance_threshold,
            'trading_viability_threshold': self.config.trading_viability_threshold,
            'max_iterations': self.config.max_iterations,
            'population_size': self.config.population_size,
            'enable_hardware_optimization': self.config.enable_hardware_optimization,
            'max_memory_usage_gb': self.config.max_memory_usage_gb,
            'log_level': self.config.log_level,
            'enable_performance_monitoring': self.config.enable_performance_monitoring
        }

    def create_tas_config(self) -> Dict[str, Any]:
        """Create TAS-specific configuration."""
        return {
            'n_regimes': self.config.n_regimes,
            'economic_significance_threshold': self.config.economic_significance_threshold,
            'trading_viability_threshold': self.config.trading_viability_threshold,
            'enable_tree_based': self.config.enable_tree_based,
            'enable_uncertainty_quantification': self.config.enable_uncertainty_quantification,
            'enable_bootstrap_analysis': self.config.enable_bootstrap_analysis,
            'enable_position_aware_analysis': self.config.enable_position_aware_analysis,
            'stability_window': self.config.stability_window,
            'transition_window': self.config.transition_window,
            'uncertainty_method': self.config.uncertainty_method,
            'enable_meta_learning': self.config.enable_meta_learning,
            'adaptation_rate': self.config.adaptation_rate,
            'learning_threshold': self.config.learning_threshold,
            'model_parameters': self.config.model_parameters,
            'data_preprocessing': self.config.data_preprocessing
        }

    def create_nas_config(self) -> Dict[str, Any]:
        """Create NAS-specific configuration."""
        return {
            'n_regimes': self.config.n_regimes,
            'economic_significance_threshold': self.config.economic_significance_threshold,
            'trading_viability_threshold': self.config.trading_viability_threshold,
            'enable_neural_based': self.config.enable_neural_based,
            'enable_uncertainty_quantification': self.config.enable_uncertainty_quantification,
            'enable_bootstrap_analysis': self.config.enable_bootstrap_analysis,
            'enable_position_aware_analysis': self.config.enable_position_aware_analysis,
            'stability_window': self.config.stability_window,
            'transition_window': self.config.transition_window,
            'uncertainty_method': self.config.uncertainty_method,
            'enable_meta_learning': self.config.enable_meta_learning,
            'adaptation_rate': self.config.adaptation_rate,
            'learning_threshold': self.config.learning_threshold,
            'model_parameters': self.config.model_parameters,
            'data_preprocessing': self.config.data_preprocessing
        }

    def create_optimization_config(self) -> Dict[str, Any]:
        """Create optimization-specific configuration."""
        return {
            'max_iterations': self.config.max_iterations,
            'population_size': self.config.population_size,
            'convergence_threshold': self.config.convergence_threshold,
            'enable_hardware_optimization': self.config.enable_hardware_optimization,
            'max_memory_usage_gb': self.config.max_memory_usage_gb,
            'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
            'enable_performance_monitoring': self.config.enable_performance_monitoring,
            'monitoring_interval': self.config.monitoring_interval
        }

    def create_economic_config(self) -> Dict[str, Any]:
        """Create economic evaluation configuration."""
        return {
            'economic_significance_threshold': self.config.economic_significance_threshold,
            'trading_viability_threshold': self.config.trading_viability_threshold,
            'enable_economic_indicators': self.config.enable_economic_indicators,
            'enable_position_aware_analysis': self.config.enable_position_aware_analysis
        }

    def create_analysis_config(self) -> Dict[str, Any]:
        """Create analysis-specific configuration."""
        return {
            'stability_window': self.config.stability_window,
            'transition_window': self.config.transition_window,
            'uncertainty_method': self.config.uncertainty_method,
            'enable_uncertainty_quantification': self.config.enable_uncertainty_quantification,
            'enable_bootstrap_analysis': self.config.enable_bootstrap_analysis,
            'enable_meta_learning': self.config.enable_meta_learning,
            'adaptation_rate': self.config.adaptation_rate,
            'learning_threshold': self.config.learning_threshold
        }

    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        try:
            merged_config = {}

            for config in configs:
                merged_config.update(config)

            return merged_config

        except Exception as e:
            self.logger.error(f"Failed to merge configurations: {e}")
            return {}

    def get_default_config(self) -> UnifiedRegimeConfig:
        """Get default configuration."""
        return UnifiedRegimeConfig()

    def create_development_config(self) -> UnifiedRegimeConfig:
        """Create development configuration."""
        config = UnifiedRegimeConfig()
        config.environment = "development"
        config.log_level = "DEBUG"
        config.enable_performance_monitoring = True
        config.max_iterations = 50
        config.population_size = 25
        return config

    def create_production_config(self) -> UnifiedRegimeConfig:
        """Create production configuration."""
        config = UnifiedRegimeConfig()
        config.environment = "production"
        config.log_level = "INFO"
        config.enable_performance_monitoring = True
        config.max_iterations = 200
        config.population_size = 100
        config.enable_hardware_optimization = True
        return config

    def create_testing_config(self) -> UnifiedRegimeConfig:
        """Create testing configuration."""
        config = UnifiedRegimeConfig()
        config.environment = "testing"
        config.log_level = "WARNING"
        config.enable_performance_monitoring = False
        config.max_iterations = 10
        config.population_size = 5
        config.n_regimes = 2
        return config

# Convenience functions
def create_unified_config_manager(config: Optional[UnifiedRegimeConfig] = None) -> UnifiedConfigManager:
    """Create a unified configuration manager."""
    return UnifiedConfigManager(config)

def load_config_from_file(filepath: Union[str, Path], format: ConfigFormat = ConfigFormat.JSON) -> UnifiedConfigManager:
    """Load configuration from file."""
    manager = UnifiedConfigManager()
    return manager.load_from_file(filepath, format)

def create_environment_config(environment: str) -> UnifiedConfigManager:
    """Create configuration for specific environment."""
    manager = UnifiedConfigManager()

    if environment == "development":
        manager.config = manager.create_development_config()
    elif environment == "production":
        manager.config = manager.create_production_config()
    elif environment == "testing":
        manager.config = manager.create_testing_config()
    else:
        manager.logger.warning(f"Unknown environment: {environment}")

    return manager
