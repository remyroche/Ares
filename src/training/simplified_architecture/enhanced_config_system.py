"""
Enhanced Configuration-Driven Architecture System

This module provides a comprehensive configuration system that replaces
hardcoded parameters and complex initialization with flexible, version-controlled
configuration files.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import yaml
from enum import Enum
import asyncio

class ConfigFormat(Enum):
    """Supported configuration formats."""
    YAML = "yaml"
    JSON = "json"
    PYTHON = "python"

class Environment(Enum):
    """Environment types for configuration management."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class GlobalSettings:
    """Global pipeline settings."""
    data_source: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    security: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StepConfiguration:
    """Configuration for a single pipeline step."""
    name: str
    class_name: str
    enabled: bool = True
    timeout_seconds: Optional[int] = None
    retry_count: int = 0
    retry_delay_seconds: int = 1
    fail_fast: bool = True
    priority: int = 2  # 1=low, 2=normal, 3=high, 4=critical
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineConfiguration:
    """Complete pipeline configuration."""
    name: str
    version: str
    description: str = ""
    environment: Environment = Environment.DEVELOPMENT
    global_settings: GlobalSettings = field(default_factory=GlobalSettings)
    steps: List[StepConfiguration] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()

class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass

class ConfigurationManager:
    """
    Enhanced configuration manager for pipeline configurations.
    
    Features:
    - Multiple format support (YAML, JSON, Python)
    - Environment-specific configurations
    - Configuration validation and schema checking
    - Configuration versioning and history
    - Hot-reloading of configurations
    - Configuration templates and inheritance
    """

    def __init__(self, config_dir: Optional[Path] = None, logger: Optional[logging.Logger] = None):
        self.config_dir = config_dir or Path("config")
        self.logger = logger or logging.getLogger(__name__)
        self._config_cache: Dict[str, PipelineConfiguration] = {}
        self._config_schemas: Dict[str, Dict[str, Any]] = {}
        self._watchers: List[asyncio.Task] = []

    def load_config(self, config_path: Union[str, Path], environment: Optional[Environment] = None) -> PipelineConfiguration:
        """
        Load pipeline configuration from file.
        
        Args:
            config_path: Path to configuration file
            environment: Environment override
            
        Returns:
            PipelineConfiguration object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        # Check cache first
        cache_key = f"{config_path}_{environment}"
        if cache_key in self._config_cache:
            return self._config_cache[cache_key]

        try:
            # Load based on file extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = self._load_yaml(config_path)
            elif config_path.suffix.lower() == '.json':
                config_data = self._load_json(config_path)
            elif config_path.suffix.lower() == '.py':
                config_data = self._load_python(config_path)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {config_path.suffix}")

            # Apply environment-specific overrides
            if environment:
                config_data = self._apply_environment_overrides(config_data, environment)

            # Convert to PipelineConfiguration
            config = self._parse_configuration(config_data)
            
            # Validate configuration
            self._validate_configuration(config)
            
            # Cache the configuration
            self._config_cache[cache_key] = config
            
            self.logger.info(f"Loaded configuration: {config.name} v{config.version}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def save_config(self, config: PipelineConfiguration, config_path: Union[str, Path], format: ConfigFormat = ConfigFormat.YAML) -> None:
        """
        Save pipeline configuration to file.
        
        Args:
            config: PipelineConfiguration to save
            config_path: Path to save configuration
            format: Output format
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Update timestamps
        config.updated_at = datetime.now()

        try:
            config_data = self._serialize_configuration(config)
            
            if format == ConfigFormat.YAML:
                self._save_yaml(config_data, config_path)
            elif format == ConfigFormat.JSON:
                self._save_json(config_data, config_path)
            else:
                raise ConfigurationError(f"Unsupported save format: {format}")

            self.logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            self.logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise ConfigurationError(f"Configuration saving failed: {e}")

    def create_config_template(self, template_name: str, output_path: Union[str, Path]) -> None:
        """Create a configuration template for common use cases."""
        templates = {
            "basic_ml_pipeline": self._create_basic_ml_template(),
            "advanced_trading_pipeline": self._create_advanced_trading_template(),
            "hmm_regime_pipeline": self._create_hmm_regime_template(),
            "ensemble_training_pipeline": self._create_ensemble_training_template()
        }

        if template_name not in templates:
            raise ConfigurationError(f"Unknown template: {template_name}")

        template = templates[template_name]
        self.save_config(template, output_path)

    def validate_config(self, config: PipelineConfiguration) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Basic validation
        if not config.name:
            errors.append("Pipeline name is required")
        
        if not config.version:
            errors.append("Pipeline version is required")

        # Validate steps
        step_names = set()
        for i, step in enumerate(config.steps):
            if not step.name:
                errors.append(f"Step {i}: name is required")
            elif step.name in step_names:
                errors.append(f"Step {i}: duplicate name '{step.name}'")
            else:
                step_names.add(step.name)

            if not step.class_name:
                errors.append(f"Step {i}: class_name is required")

            # Validate dependencies
            for dep in step.dependencies:
                if dep not in step_names:
                    errors.append(f"Step {i}: dependency '{dep}' not found")

        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies(config.steps)
        if circular_deps:
            errors.append(f"Circular dependencies detected: {circular_deps}")

        return errors

    def get_available_templates(self) -> List[str]:
        """Get list of available configuration templates."""
        return [
            "basic_ml_pipeline",
            "advanced_trading_pipeline", 
            "hmm_regime_pipeline",
            "ensemble_training_pipeline"
        ]

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_python(self, path: Path) -> Dict[str, Any]:
        """Load Python configuration file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CONFIG

    def _save_yaml(self, data: Dict[str, Any], path: Path) -> None:
        """Save configuration as YAML."""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)

    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save configuration as JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _apply_environment_overrides(self, config_data: Dict[str, Any], environment: Environment) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        # Look for environment-specific sections
        env_key = f"{environment.value}_overrides"
        if env_key in config_data:
            overrides = config_data[env_key]
            config_data = self._deep_merge(config_data, overrides)
            # Remove the override section
            del config_data[env_key]
        
        return config_data

    def _parse_configuration(self, data: Dict[str, Any]) -> PipelineConfiguration:
        """Parse configuration data into PipelineConfiguration object."""
        # Parse global settings
        global_settings_data = data.get('global_settings', {})
        global_settings = GlobalSettings(
            data_source=global_settings_data.get('data_source', {}),
            model=global_settings_data.get('model', {}),
            logging=global_settings_data.get('logging', {}),
            performance=global_settings_data.get('performance', {}),
            security=global_settings_data.get('security', {}),
            monitoring=global_settings_data.get('monitoring', {})
        )

        # Parse steps
        steps = []
        for step_data in data.get('steps', []):
            step = StepConfiguration(
                name=step_data['name'],
                class_name=step_data['class_name'],
                enabled=step_data.get('enabled', True),
                timeout_seconds=step_data.get('timeout_seconds'),
                retry_count=step_data.get('retry_count', 0),
                retry_delay_seconds=step_data.get('retry_delay_seconds', 1),
                fail_fast=step_data.get('fail_fast', True),
                priority=step_data.get('priority', 2),
                parameters=step_data.get('parameters', {}),
                dependencies=step_data.get('dependencies', []),
                output_schema=step_data.get('output_schema'),
                validation_rules=step_data.get('validation_rules', []),
                resource_limits=step_data.get('resource_limits', {}),
                metadata=step_data.get('metadata', {})
            )
            steps.append(step)

        # Create configuration
        config = PipelineConfiguration(
            name=data['name'],
            version=data['version'],
            description=data.get('description', ''),
            environment=Environment(data.get('environment', 'development')),
            global_settings=global_settings,
            steps=steps,
            metadata=data.get('metadata', {})
        )

        return config

    def _serialize_configuration(self, config: PipelineConfiguration) -> Dict[str, Any]:
        """Serialize PipelineConfiguration to dictionary."""
        return {
            'name': config.name,
            'version': config.version,
            'description': config.description,
            'environment': config.environment.value,
            'global_settings': {
                'data_source': config.global_settings.data_source,
                'model': config.global_settings.model,
                'logging': config.global_settings.logging,
                'performance': config.global_settings.performance,
                'security': config.global_settings.security,
                'monitoring': config.global_settings.monitoring
            },
            'steps': [
                {
                    'name': step.name,
                    'class_name': step.class_name,
                    'enabled': step.enabled,
                    'timeout_seconds': step.timeout_seconds,
                    'retry_count': step.retry_count,
                    'retry_delay_seconds': step.retry_delay_seconds,
                    'fail_fast': step.fail_fast,
                    'priority': step.priority,
                    'parameters': step.parameters,
                    'dependencies': step.dependencies,
                    'output_schema': step.output_schema,
                    'validation_rules': step.validation_rules,
                    'resource_limits': step.resource_limits,
                    'metadata': step.metadata
                }
                for step in config.steps
            ],
            'metadata': config.metadata,
            'created_at': config.created_at.isoformat() if config.created_at else None,
            'updated_at': config.updated_at.isoformat() if config.updated_at else None
        }

    def _validate_configuration(self, config: PipelineConfiguration) -> None:
        """Validate configuration and raise errors if invalid."""
        errors = self.validate_config(config)
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

    def _detect_circular_dependencies(self, steps: List[StepConfiguration]) -> Optional[List[str]]:
        """Detect circular dependencies in step configuration."""
        # Build dependency graph
        graph = {step.name: step.dependencies for step in steps}
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return list(rec_stack)
        
        return None

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

    def _create_basic_ml_template(self) -> PipelineConfiguration:
        """Create basic ML pipeline template."""
        return PipelineConfiguration(
            name="Basic_ML_Pipeline",
            version="1.0.0",
            description="Basic machine learning pipeline for data processing and model training",
            global_settings=GlobalSettings(
                data_source={"type": "file", "format": "parquet"},
                model={"type": "lightgbm", "hyperparameters": {"n_estimators": 1000}}
            ),
            steps=[
                StepConfiguration(
                    name="data_loading",
                    class_name="DataLoadingStep",
                    parameters={"source": "data/raw/training_data.parquet"}
                ),
                StepConfiguration(
                    name="feature_engineering",
                    class_name="FeatureEngineeringStep",
                    dependencies=["data_loading"],
                    parameters={"feature_types": ["technical", "statistical"]}
                ),
                StepConfiguration(
                    name="model_training",
                    class_name="ModelTrainingStep",
                    dependencies=["feature_engineering"],
                    parameters={"model_type": "lightgbm"}
                )
            ]
        )

    def _create_advanced_trading_template(self) -> PipelineConfiguration:
        """Create advanced trading pipeline template."""
        return PipelineConfiguration(
            name="Advanced_Trading_Pipeline",
            version="2.0.0",
            description="Advanced trading pipeline with HMM regime detection and ensemble models",
            global_settings=GlobalSettings(
                data_source={"type": "exchange", "exchange": "binance"},
                model={"type": "ensemble", "models": ["lightgbm", "xgboost", "neural_network"]}
            ),
            steps=[
                StepConfiguration(
                    name="data_collection",
                    class_name="DataCollectionStep",
                    parameters={"symbol": "BTCUSDT", "timeframe": "1h", "lookback_days": 90}
                ),
                StepConfiguration(
                    name="hmm_regime_discovery",
                    class_name="HMMRegimeDiscoveryStep",
                    dependencies=["data_collection"],
                    parameters={"n_components": 3, "max_iterations": 100}
                ),
                StepConfiguration(
                    name="feature_engineering",
                    class_name="FeatureEngineeringStep",
                    dependencies=["hmm_regime_discovery"],
                    parameters={"feature_types": ["technical", "statistical", "wavelet", "regime_based"]}
                ),
                StepConfiguration(
                    name="ensemble_training",
                    class_name="EnsembleTrainingStep",
                    dependencies=["feature_engineering"],
                    parameters={
                        "models": [
                            {"type": "lightgbm", "hyperparameters": {"n_estimators": 1000}},
                            {"type": "xgboost", "hyperparameters": {"n_estimators": 1000}},
                            {"type": "neural_network", "hyperparameters": {"hidden_layers": [128, 64]}}
                        ]
                    }
                )
            ]
        )

    def _create_hmm_regime_template(self) -> PipelineConfiguration:
        """Create HMM regime detection pipeline template."""
        return PipelineConfiguration(
            name="HMM_Regime_Pipeline",
            version="1.5.0",
            description="HMM-based regime detection and per-regime model training",
            global_settings=GlobalSettings(
                data_source={"type": "exchange", "exchange": "binance"},
                model={"type": "hmm_ensemble", "regime_models": ["lightgbm", "xgboost"]}
            ),
            steps=[
                StepConfiguration(
                    name="data_collection",
                    class_name="DataCollectionStep",
                    parameters={"symbol": "BTCUSDT", "timeframe": "1h"}
                ),
                StepConfiguration(
                    name="hmm_regime_discovery",
                    class_name="HMMRegimeDiscoveryStep",
                    dependencies=["data_collection"],
                    parameters={"n_components": 4, "covariance_type": "full"}
                ),
                StepConfiguration(
                    name="regime_data_splitting",
                    class_name="RegimeDataSplittingStep",
                    dependencies=["hmm_regime_discovery"],
                    parameters={"min_samples_per_regime": 1000}
                ),
                StepConfiguration(
                    name="per_regime_training",
                    class_name="PerRegimeTrainingStep",
                    dependencies=["regime_data_splitting"],
                    parameters={"model_type": "lightgbm", "regime_specific": True}
                )
            ]
        )

    def _create_ensemble_training_template(self) -> PipelineConfiguration:
        """Create ensemble training pipeline template."""
        return PipelineConfiguration(
            name="Ensemble_Training_Pipeline",
            version="1.2.0",
            description="Multi-model ensemble training with stacking and blending",
            global_settings=GlobalSettings(
                data_source={"type": "file", "format": "parquet"},
                model={"type": "stacking_ensemble", "meta_learner": "logistic_regression"}
            ),
            steps=[
                StepConfiguration(
                    name="data_loading",
                    class_name="DataLoadingStep",
                    parameters={"source": "data/processed/features.parquet"}
                ),
                StepConfiguration(
                    name="base_model_training",
                    class_name="BaseModelTrainingStep",
                    dependencies=["data_loading"],
                    parameters={
                        "models": ["lightgbm", "xgboost", "random_forest", "neural_network"],
                        "cross_validation": True
                    }
                ),
                StepConfiguration(
                    name="meta_learner_training",
                    class_name="MetaLearnerTrainingStep",
                    dependencies=["base_model_training"],
                    parameters={"meta_learner_type": "logistic_regression"}
                ),
                StepConfiguration(
                    name="ensemble_validation",
                    class_name="EnsembleValidationStep",
                    dependencies=["meta_learner_training"],
                    parameters={"validation_method": "time_series_split"}
                )
            ]
        )