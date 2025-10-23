"""
Analytical Process Configuration Loader

This module loads and manages the configuration for the three main analytical components:
- Regime Detection (4h timeframe, run every 1h)
- Analyst (1h timeframe, run every 15m)
- Tactician (15m timeframe, run every 3m)
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ComponentType(Enum):
    """Analytical component types."""
    REGIME_DETECTION = "regime_detection"
    ANALYST = "analyst"
    TACTICIAN = "tactician"

@dataclass
class ComponentConfig:
    """Configuration for an analytical component."""
    timeframe: str
    run_frequency: str
    description: str
    min_bars: int
    lookback_days: int
    model_types: list
    confidence_threshold: Optional[float] = None
    analyst_filtering: Optional[bool] = None

@dataclass
class ExecutionSchedule:
    """Execution schedule configuration."""
    cron_expression: str
    timezone: str
    max_execution_time: str
    retry_attempts: int
    retry_delay: str

class AnalyticalProcessConfig:
    """Configuration manager for analytical process timeframes and frequencies."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "analytical_process_timeframes.yaml"

        self.config_path = Path(config_path)
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")

    def get_component_config(self, component: ComponentType) -> ComponentConfig:
        """Get configuration for a specific component.

        Args:
            component: The component type

        Returns:
            ComponentConfig object
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        component_key = component.value
        if component_key not in self._config['analytical_process']:
            raise ValueError(f"Component {component_key} not found in configuration")

        config_data = self._config['analytical_process'][component_key]

        return ComponentConfig(
            timeframe=config_data['timeframe'],
            run_frequency=config_data['run_frequency'],
            description=config_data['description'],
            min_bars=config_data['data_requirements']['min_bars'],
            lookback_days=config_data['data_requirements']['lookback_days'],
            model_types=config_data['model_types'],
            confidence_threshold=config_data.get('confidence_threshold'),
            analyst_filtering=config_data.get('analyst_filtering')
        )

    def get_execution_schedule(self, component: ComponentType) -> ExecutionSchedule:
        """Get execution schedule for a specific component.

        Args:
            component: The component type

        Returns:
            ExecutionSchedule object
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        component_key = component.value
        if component_key not in self._config['execution_schedule']:
            raise ValueError(f"Execution schedule for {component_key} not found in configuration")

        schedule_data = self._config['execution_schedule'][component_key]

        return ExecutionSchedule(
            cron_expression=schedule_data['cron_expression'],
            timezone=schedule_data['timezone'],
            max_execution_time=schedule_data['max_execution_time'],
            retry_attempts=schedule_data['retry_attempts'],
            retry_delay=schedule_data['retry_delay']
        )

    def get_data_pipeline_config(self, component: ComponentType) -> Dict[str, str]:
        """Get data pipeline configuration for a specific component.

        Args:
            component: The component type

        Returns:
            Dictionary with data collection and feature engineering timeframes
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        component_key = component.value
        data_config = self._config['data_pipeline']

        return {
            'data_collection': data_config['data_collection'][f"{component_key}_data"],
            'base_timeframe': data_config['feature_engineering'][component_key]['base_timeframe'],
            'additional_timeframes': data_config['feature_engineering'][component_key]['additional_timeframes']
        }

    def get_model_training_config(self, component: ComponentType) -> Dict[str, Any]:
        """Get model training configuration for a specific component.

        Args:
            component: The component type

        Returns:
            Dictionary with model training configuration
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        component_key = component.value
        if component_key not in self._config['model_training']:
            raise ValueError(f"Model training configuration for {component_key} not found")

        return self._config['model_training'][component_key]

    def get_monitoring_config(self, component: ComponentType) -> Dict[str, Any]:
        """Get monitoring configuration for a specific component.

        Args:
            component: The component type

        Returns:
            Dictionary with monitoring configuration
        """
        if self._config is None:
            raise RuntimeError("Configuration not loaded")

        component_key = component.value
        monitoring_config = self._config['monitoring']

        return {
            'performance_thresholds': monitoring_config['performance_thresholds'][component_key],
            'alerts': monitoring_config['alerts']
        }

    def get_all_components(self) -> Dict[ComponentType, ComponentConfig]:
        """Get configuration for all components.

        Returns:
            Dictionary mapping component types to their configurations
        """
        return {
            ComponentType.REGIME_DETECTION: self.get_component_config(ComponentType.REGIME_DETECTION),
            ComponentType.ANALYST: self.get_component_config(ComponentType.ANALYST),
            ComponentType.TACTICIAN: self.get_component_config(ComponentType.TACTICIAN)
        }

    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()

    def validate_config(self) -> bool:
        """Validate the loaded configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if self._config is None:
            return False

        required_sections = ['analytical_process', 'execution_schedule', 'data_pipeline', 'model_training', 'monitoring']
        for section in required_sections:
            if section not in self._config:
                return False

        required_components = ['regime_detection', 'analyst', 'tactician']
        for component in required_components:
            if component not in self._config['analytical_process']:
                return False

        return True

# Global configuration instance
_config_instance: Optional[AnalyticalProcessConfig] = None

def get_analytical_process_config() -> AnalyticalProcessConfig:
    """Get the global analytical process configuration instance.

    Returns:
        AnalyticalProcessConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = AnalyticalProcessConfig()
    return _config_instance

def reload_analytical_process_config() -> None:
    """Reload the global analytical process configuration."""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload_config()

# Convenience functions for quick access
def get_regime_detection_config() -> ComponentConfig:
    """Get regime detection configuration."""
    return get_analytical_process_config().get_component_config(ComponentType.REGIME_DETECTION)

def get_analyst_config() -> ComponentConfig:
    """Get analyst configuration."""
    return get_analytical_process_config().get_component_config(ComponentType.ANALYST)

def get_tactician_config() -> ComponentConfig:
    """Get tactician configuration."""
    return get_analytical_process_config().get_component_config(ComponentType.TACTICIAN)

def get_regime_detection_schedule() -> ExecutionSchedule:
    """Get regime detection execution schedule."""
    return get_analytical_process_config().get_execution_schedule(ComponentType.REGIME_DETECTION)

def get_analyst_schedule() -> ExecutionSchedule:
    """Get analyst execution schedule."""
    return get_analytical_process_config().get_execution_schedule(ComponentType.ANALYST)

def get_tactician_schedule() -> ExecutionSchedule:
    """Get tactician execution schedule."""
    return get_analytical_process_config().get_execution_schedule(ComponentType.TACTICIAN)
