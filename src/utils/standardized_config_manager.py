from __future__ import annotations
'Standardized Configuration Management System.\n\nThis module provides centralized configuration management with validation,\nversioning, and standardized access patterns across all pipeline steps.\n'
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from .pipeline_standards import pipeline_standards
from copy import copy

class StandardizedConfigManager:
    """Centralized configuration manager with validation and versioning."""

    def __init__(self, base_config_path: str='config') -> None:
        self.base_config_path = Path(base_config_path)
        self.logger = pipeline_standards.get_logger(__name__)
        self.config_cache = {}
        self.config_versions = {}
        self.schemas = {'pipeline': {'required': ['symbol', 'exchange', 'timeframe'], 'optional': ['data_dir', 'force_rerun', 'enable_mlflow'], 'defaults': {'data_dir': None, 'force_rerun': False, 'enable_mlflow': True}}, 'training': {'required': ['model_type', 'epochs'], 'optional': ['batch_size', 'learning_rate', 'validation_split'], 'defaults': {'batch_size': 32, 'learning_rate': 0.001, 'validation_split': 0.2}}, 'data_quality': {'required': [], 'optional': ['min_quality_score', 'max_missing_ratio', 'min_rows'], 'defaults': {'min_quality_score': 0.8, 'max_missing_ratio': 0.1, 'min_rows': 1000}}}

    def load_config(self, config_type: str, config_name: str='default') -> dict[str, Any]:
        """Load configuration with validation and caching.

        Args:
            config_type: Type of configuration (pipeline, training, data_quality)
            config_name: Name of the configuration file

        Returns:
            Validated configuration dictionary
        """
        cache_key = f'{config_type}_{config_name}'
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        config_path = self.base_config_path / config_type / f'{config_name}.json'
        if not config_path.exists():
            self.logger.warning(f'⚠️ Config file not found: {config_path}')
            config = self._get_default_config(config_type)
        else:
            try:
                with open(config_path) as f:
                    config = json.load(f)
                self.logger.info(f'✅ Loaded config: {config_path}')
            except Exception as e:
                self.logger.exception(f'❌ Error loading config {config_path}: {e}')
                config = self._get_default_config(config_type)
        validated_config = self._validate_config(config, config_type)
        self.config_cache[cache_key] = validated_config
        return validated_config

    def _get_default_config(self, config_type: str) -> dict[str, Any]:
        """Get default configuration for a given type."""
        if config_type in self.schemas:
            return self.schemas[config_type]['defaults'].copy()
        return {}

    def _validate_config(self, config: dict[str, Any], config_type: str) -> dict[str, Any]:
        """Validate configuration against schema and apply defaults."""
        if config_type not in self.schemas:
            self.logger.warning(f'⚠️ Unknown config type: {config_type}')
            return config
        schema = self.schemas[config_type]
        validated_config = schema['defaults'].copy()
        for key, value in config.items():
            if key in schema['required'] or key in schema['optional']:
                validated_config[key] = value
            else:
                self.logger.warning(f'⚠️ Unknown config key: {key}')
        missing_required = []
        for required_key in schema['required']:
            if required_key not in validated_config:
                missing_required.append(required_key)
        if missing_required:
            self.logger.error(f'❌ Missing required config keys: {missing_required}')
            msg = f'Missing required configuration keys: {missing_required}'
            raise ValueError(msg)
        return validated_config

    def create_step_config(self, step_name: str, base_config: dict[str, Any]) -> dict[str, Any]:
        """Create standardized configuration for a specific step.

        Args:
            step_name: Name of the step (e.g., "step01", "step09")
            base_config: Base configuration dictionary

        Returns:
            Step-specific configuration
        """
        step_config = base_config.copy()
        step_defaults = {'step_name': step_name, 'timestamp': datetime.now().isoformat(), 'version': '1.0.0'}
        step_config.update(step_defaults)
        if step_name.startswith('step'):
            step_config.update({'enable_validation': True, 'enable_logging': True, 'enable_mlflow': step_config.get('enable_mlflow', True)})
        return step_config

    def save_config(self, config: dict[str, Any], config_type: str, config_name: str) -> bool:
        """Save configuration to file.

        Args:
            config: Configuration dictionary
            config_type: Type of configuration
            config_name: Name of the configuration file

        Returns:
            True if successful, False otherwise
        """
        try:
            config_dir = self.base_config_path / config_type
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / f'{config_name}.json'
            config_with_metadata = {'metadata': {'created_at': datetime.now().isoformat(), 'version': '1.0.0', 'config_type': config_type}, 'config': config}
            with open(config_path, 'w') as f:
                json.dump(config_with_metadata, f, indent=2)
            self.logger.info(f'✅ Saved config: {config_path}')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Error saving config: {e}')
            return False

    def get_standardized_paths(self, exchange: str, symbol: str) -> dict[str, str]:
        """Get standardized paths for a given exchange and symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Dictionary of standardized paths
        """
        return {'raw_data': pipeline_standards.build_path('raw_data', exchange, symbol), 'processed_data': pipeline_standards.build_path('processed_data', exchange, symbol), 'unified_data': pipeline_standards.build_path('unified_data', exchange, symbol), 'training_data': pipeline_standards.build_path('training_data', exchange, symbol), 'models': pipeline_standards.build_path('models', exchange, symbol), 'logs': pipeline_standards.build_path('logs', exchange, symbol)}

    def validate_environment_config(self) -> dict[str, bool]:
        """Validate environment configuration.

        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        required_dirs = ['config', 'data_cache', 'logs']
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            validation_results[f'dir_{dir_name}'] = dir_path.exists()
        required_files = ['config/pipeline/default.json']
        for file_path in required_files:
            file_path_obj = Path(file_path)
            validation_results[f'file_{file_path}'] = file_path_obj.exists()
        return validation_results
config_manager = StandardizedConfigManager()

def get_standardized_config(step_name: str, config_overrides: dict[str, Any] | None=None) -> dict[str, Any]:
    """Get standardized configuration for a step.

    Args:
        step_name: Name of the step
        config_overrides: Optional configuration overrides

    Returns:
        Standardized configuration dictionary
    """
    base_config = config_manager.load_config('pipeline')
    if config_overrides:
        base_config.update(config_overrides)
    return config_manager.create_step_config(step_name, base_config)

def validate_step_config(step_config: dict[str, Any], step_name: str) -> bool:
    """Validate step configuration.

    Args:
        step_config: Step configuration dictionary
        step_name: Name of the step

    Returns:
        True if valid, False otherwise
    """
    required_keys = ['symbol', 'exchange', 'timeframe']
    for key in required_keys:
        if key not in step_config:
            config_manager.logger.error(f'❌ Missing required config key for {step_name}: {key}')
            return False
    return True