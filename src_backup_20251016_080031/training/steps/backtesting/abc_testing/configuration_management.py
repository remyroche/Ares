"""
Configuration Management System for A/B/C Testing

This module provides a comprehensive configuration management system
for A/B/C testing with validation, inheritance, and dynamic updates.

Key Features:
- Hierarchical configuration system
- Configuration validation and schema enforcement
- Environment-specific configurations
- Dynamic configuration updates
- Configuration versioning and rollback
- Template-based configuration generation
- Configuration inheritance and overrides
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from pathlib import Path
import copy
import jsonschema
from jsonschema import validate, ValidationError
import uuid

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)

# Core decorators and validation
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time, 
    timeout, error_boundary, compose, validate_data_quality, 
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    MathValidationError, TimeoutError
)

logger = logging.getLogger(__name__)


class ConfigurationFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    ENV = "env"


class ConfigurationScope(Enum):
    """Configuration scope levels."""
    GLOBAL = "global"
    ENVIRONMENT = "environment"
    TEST = "test"
    MODEL = "model"
    RUNTIME = "runtime"


@dataclass
class ConfigurationSchema:
    """Configuration schema definition."""
    schema_id: str
    name: str
    description: str
    version: str
    schema: Dict[str, Any]
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    default_values: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationEntry:
    """Configuration entry with metadata."""
    config_id: str
    name: str
    scope: ConfigurationScope
    format: ConfigurationFormat
    content: Dict[str, Any]
    schema_id: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    is_active: bool = True
    parent_config_id: Optional[str] = None
    environment: str = "default"


@dataclass
class ConfigurationTemplate:
    """Configuration template for generating new configurations."""
    template_id: str
    name: str
    description: str
    template_type: str
    schema_id: str
    template_data: Dict[str, Any]
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


class ConfigurationValidator:
    """Advanced configuration validator."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.logger = logger.getChild('ConfigurationValidator')
        self.schemas: Dict[str, ConfigurationSchema] = {}
        
        # Initialize default schemas
        self._initialize_default_schemas()
        
        self.logger.info("🚀 ConfigurationValidator initialized")
        self.logger.info(f"📊 Schemas loaded: {len(self.schemas)}")
    
    def _initialize_default_schemas(self) -> None:
        """Initialize default configuration schemas."""
        
        # A/B/C Testing Configuration Schema
        abc_testing_schema = {
            "type": "object",
            "properties": {
                "test_name": {"type": "string", "minLength": 1},
                "test_description": {"type": "string"},
                "symbol": {"type": "string", "pattern": "^[A-Z0-9]+$"},
                "exchange": {"type": "string", "enum": ["BINANCE", "COINBASE", "KRAKEN"]},
                "timeframe": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "4h", "1d"]},
                "data_dir": {"type": "string"},
                "start_date": {"type": "string", "format": "date-time"},
                "end_date": {"type": "string", "format": "date-time"},
                "test_mode": {"type": "string", "enum": ["paper_trading", "backtesting", "hybrid"]},
                "model_configs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string"},
                            "model_name": {"type": "string"},
                            "model_type": {"type": "string"},
                            "initial_capital": {"type": "number", "minimum": 1000},
                            "max_position_size": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                            "risk_per_trade": {"type": "number", "minimum": 0.001, "maximum": 0.1}
                        },
                        "required": ["model_id", "model_name", "model_type"]
                    }
                },
                "statistical_testing": {
                    "type": "object",
                    "properties": {
                        "enable_statistical_testing": {"type": "boolean"},
                        "confidence_level": {"type": "number", "minimum": 0.5, "maximum": 0.99},
                        "alpha": {"type": "number", "minimum": 0.001, "maximum": 0.1},
                        "min_sample_size": {"type": "integer", "minimum": 10}
                    }
                },
                "risk_management": {
                    "type": "object",
                    "properties": {
                        "global_risk_limit": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                        "max_concurrent_positions": {"type": "integer", "minimum": 1, "maximum": 20},
                        "correlation_threshold": {"type": "number", "minimum": 0.1, "maximum": 1.0}
                    }
                }
            },
            "required": ["test_name", "symbol", "exchange", "timeframe", "model_configs"]
        }
        
        self.schemas["abc_testing"] = ConfigurationSchema(
            schema_id="abc_testing",
            name="A/B/C Testing Configuration",
            description="Schema for A/B/C testing configurations",
            version="1.0.0",
            schema=abc_testing_schema,
            required_fields=["test_name", "symbol", "exchange", "timeframe", "model_configs"],
            default_values={
                "test_mode": "paper_trading",
                "statistical_testing": {
                    "enable_statistical_testing": True,
                    "confidence_level": 0.95,
                    "alpha": 0.05,
                    "min_sample_size": 100
                },
                "risk_management": {
                    "global_risk_limit": 0.2,
                    "max_concurrent_positions": 5,
                    "correlation_threshold": 0.7
                }
            }
        )
        
        # Model Configuration Schema
        model_schema = {
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "model_name": {"type": "string"},
                "model_type": {"type": "string"},
                "model_params": {"type": "object"},
                "initial_capital": {"type": "number", "minimum": 1000},
                "max_position_size": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                "risk_per_trade": {"type": "number", "minimum": 0.001, "maximum": 0.1},
                "stop_loss_pct": {"type": "number", "minimum": 0.001, "maximum": 0.5},
                "take_profit_pct": {"type": "number", "minimum": 0.001, "maximum": 1.0},
                "enable_risk_management": {"type": "boolean"},
                "enable_position_sizing": {"type": "boolean"}
            },
            "required": ["model_id", "model_name", "model_type"]
        }
        
        self.schemas["model"] = ConfigurationSchema(
            schema_id="model",
            name="Model Configuration",
            description="Schema for individual model configurations",
            version="1.0.0",
            schema=model_schema,
            required_fields=["model_id", "model_name", "model_type"],
            default_values={
                "initial_capital": 100000.0,
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
                "stop_loss_pct": 0.05,
                "take_profit_pct": 0.1,
                "enable_risk_management": True,
                "enable_position_sizing": True
            }
        )
        
        # Monitoring Configuration Schema
        monitoring_schema = {
            "type": "object",
            "properties": {
                "monitoring_interval": {"type": "integer", "minimum": 10, "maximum": 3600},
                "enable_alerting": {"type": "boolean"},
                "alert_cooldown_minutes": {"type": "integer", "minimum": 1, "maximum": 1440},
                "performance_thresholds": {
                    "type": "object",
                    "properties": {
                        "max_drawdown": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                        "min_sharpe_ratio": {"type": "number", "minimum": 0.0, "maximum": 5.0},
                        "max_volatility": {"type": "number", "minimum": 0.01, "maximum": 1.0},
                        "min_win_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    }
                },
                "email_settings": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "smtp_server": {"type": "string"},
                        "smtp_port": {"type": "integer", "minimum": 1, "maximum": 65535},
                        "username": {"type": "string"},
                        "recipients": {"type": "array", "items": {"type": "string", "format": "email"}}
                    }
                }
            },
            "required": ["monitoring_interval", "enable_alerting"]
        }
        
        self.schemas["monitoring"] = ConfigurationSchema(
            schema_id="monitoring",
            name="Monitoring Configuration",
            description="Schema for monitoring and alerting configurations",
            version="1.0.0",
            schema=monitoring_schema,
            required_fields=["monitoring_interval", "enable_alerting"],
            default_values={
                "monitoring_interval": 30,
                "enable_alerting": True,
                "alert_cooldown_minutes": 15,
                "performance_thresholds": {
                    "max_drawdown": 0.15,
                    "min_sharpe_ratio": 0.5,
                    "max_volatility": 0.3,
                    "min_win_rate": 0.4
                }
            }
        )
    
    def register_schema(self, schema: ConfigurationSchema) -> bool:
        """Register a new configuration schema."""
        try:
            self.schemas[schema.schema_id] = schema
            self.logger.info(f"✅ Registered schema: {schema.name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Error registering schema {schema.name}: {e}")
            return False
    
    def validate_configuration(self, config: Dict[str, Any], schema_id: str) -> Tuple[bool, List[str]]:
        """Validate configuration against schema."""
        try:
            if schema_id not in self.schemas:
                return False, [f"Schema {schema_id} not found"]
            
            schema = self.schemas[schema_id]
            errors = []
            
            # JSON Schema validation
            try:
                validate(instance=config, schema=schema.schema)
            except ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")
            
            # Custom validation rules
            for field, rules in schema.validation_rules.items():
                if field in config:
                    field_errors = self._validate_field(config[field], rules)
                    errors.extend(field_errors)
            
            # Required fields check
            for field in schema.required_fields:
                if field not in config:
                    errors.append(f"Required field missing: {field}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"❌ Error validating configuration: {e}")
            return False, [str(e)]
    
    def _validate_field(self, value: Any, rules: Dict[str, Any]) -> List[str]:
        """Validate a single field against custom rules."""
        errors = []
        
        for rule_type, rule_value in rules.items():
            if rule_type == "min_length" and isinstance(value, str):
                if len(value) < rule_value:
                    errors.append(f"String too short: minimum length {rule_value}")
            elif rule_type == "max_length" and isinstance(value, str):
                if len(value) > rule_value:
                    errors.append(f"String too long: maximum length {rule_value}")
            elif rule_type == "min_value" and isinstance(value, (int, float)):
                if value < rule_value:
                    errors.append(f"Value too small: minimum {rule_value}")
            elif rule_type == "max_value" and isinstance(value, (int, float)):
                if value > rule_value:
                    errors.append(f"Value too large: maximum {rule_value}")
            elif rule_type == "pattern" and isinstance(value, str):
                import re
                if not re.match(rule_value, value):
                    errors.append(f"Pattern mismatch: {rule_value}")
        
        return errors
    
    def apply_defaults(self, config: Dict[str, Any], schema_id: str) -> Dict[str, Any]:
        """Apply default values to configuration."""
        if schema_id not in self.schemas:
            return config
        
        schema = self.schemas[schema_id]
        result = copy.deepcopy(config)
        
        def apply_defaults_recursive(target: Dict[str, Any], defaults: Dict[str, Any]) -> None:
            for key, default_value in defaults.items():
                if key not in target:
                    target[key] = default_value
                elif isinstance(default_value, dict) and isinstance(target[key], dict):
                    apply_defaults_recursive(target[key], default_value)
        
        apply_defaults_recursive(result, schema.default_values)
        return result


class ConfigurationManager:
    """Comprehensive configuration management system."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        ensure_directory(self.config_dir)
        
        self.logger = logger.getChild('ConfigurationManager')
        self.validator = ConfigurationValidator()
        
        # Configuration storage
        self.configurations: Dict[str, ConfigurationEntry] = {}
        self.templates: Dict[str, ConfigurationTemplate] = {}
        self.environments: Dict[str, Dict[str, Any]] = {}
        
        # Configuration history
        self.config_history: Dict[str, List[ConfigurationEntry]] = {}
        
        # Load existing configurations
        self._load_configurations()
        
        self.logger.info("🚀 ConfigurationManager initialized")
        self.logger.info(f"📁 Config directory: {self.config_dir}")
        self.logger.info(f"📊 Configurations loaded: {len(self.configurations)}")
    
    def _load_configurations(self) -> None:
        """Load existing configurations from disk."""
        try:
            config_files = list(self.config_dir.glob("*.json")) + list(self.config_dir.glob("*.yaml"))
            
            for config_file in config_files:
                try:
                    if config_file.suffix == '.json':
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                    elif config_file.suffix == '.yaml':
                        with open(config_file, 'r') as f:
                            config_data = yaml.safe_load(f)
                    else:
                        continue
                    
                    # Create configuration entry
                    config_entry = ConfigurationEntry(
                        config_id=config_data.get('config_id', str(uuid.uuid4())),
                        name=config_data.get('name', config_file.stem),
                        scope=ConfigurationScope(config_data.get('scope', 'test')),
                        format=ConfigurationFormat(config_data.get('format', 'json')),
                        content=config_data.get('content', {}),
                        schema_id=config_data.get('schema_id'),
                        version=config_data.get('version', '1.0.0'),
                        created_at=datetime.fromisoformat(config_data.get('created_at', datetime.now().isoformat())),
                        updated_at=datetime.fromisoformat(config_data.get('updated_at', datetime.now().isoformat())),
                        created_by=config_data.get('created_by', 'system'),
                        description=config_data.get('description', ''),
                        tags=config_data.get('tags', []),
                        is_active=config_data.get('is_active', True),
                        parent_config_id=config_data.get('parent_config_id'),
                        environment=config_data.get('environment', 'default')
                    )
                    
                    self.configurations[config_entry.config_id] = config_entry
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not load configuration {config_file}: {e}")
        
        except Exception as e:
            self.logger.error(f"❌ Error loading configurations: {e}")
    
    def save_configuration(self, config: ConfigurationEntry) -> bool:
        """Save configuration to disk."""
        try:
            # Validate configuration
            if config.schema_id:
                is_valid, errors = self.validator.validate_configuration(config.content, config.schema_id)
                if not is_valid:
                    self.logger.error(f"❌ Configuration validation failed: {errors}")
                    return False
            
            # Update metadata
            config.updated_at = datetime.now()
            
            # Store in memory
            self.configurations[config.config_id] = config
            
            # Add to history
            if config.config_id not in self.config_history:
                self.config_history[config.config_id] = []
            self.config_history[config.config_id].append(copy.deepcopy(config))
            
            # Save to disk
            config_data = {
                'config_id': config.config_id,
                'name': config.name,
                'scope': config.scope.value,
                'format': config.format.value,
                'content': config.content,
                'schema_id': config.schema_id,
                'version': config.version,
                'created_at': config.created_at.isoformat(),
                'updated_at': config.updated_at.isoformat(),
                'created_by': config.created_by,
                'description': config.description,
                'tags': config.tags,
                'is_active': config.is_active,
                'parent_config_id': config.parent_config_id,
                'environment': config.environment
            }
            
            # Determine file extension
            if config.format == ConfigurationFormat.JSON:
                file_extension = '.json'
            elif config.format == ConfigurationFormat.YAML:
                file_extension = '.yaml'
            else:
                file_extension = '.json'
            
            config_file = self.config_dir / f"{config.name}{file_extension}"
            
            if config.format == ConfigurationFormat.JSON:
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
            elif config.format == ConfigurationFormat.YAML:
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            
            self.logger.info(f"✅ Configuration saved: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving configuration {config.name}: {e}")
            return False
    
    def load_configuration(self, config_id: str) -> Optional[ConfigurationEntry]:
        """Load configuration by ID."""
        return self.configurations.get(config_id)
    
    def get_configuration_by_name(self, name: str, environment: str = "default") -> Optional[ConfigurationEntry]:
        """Get configuration by name and environment."""
        for config in self.configurations.values():
            if config.name == name and config.environment == environment and config.is_active:
                return config
        return None
    
    def list_configurations(self, scope: Optional[ConfigurationScope] = None, 
                          environment: Optional[str] = None) -> List[ConfigurationEntry]:
        """List configurations with optional filtering."""
        configs = list(self.configurations.values())
        
        if scope:
            configs = [c for c in configs if c.scope == scope]
        
        if environment:
            configs = [c for c in configs if c.environment == environment]
        
        return [c for c in configs if c.is_active]
    
    def create_configuration_from_template(self, template_id: str, 
                                         parameters: Dict[str, Any],
                                         name: str,
                                         environment: str = "default") -> Optional[ConfigurationEntry]:
        """Create configuration from template."""
        if template_id not in self.templates:
            self.logger.error(f"❌ Template {template_id} not found")
            return None
        
        try:
            template = self.templates[template_id]
            
            # Generate configuration content
            config_content = self._generate_config_from_template(template, parameters)
            
            # Apply defaults
            if template.schema_id:
                config_content = self.validator.apply_defaults(config_content, template.schema_id)
            
            # Create configuration entry
            config = ConfigurationEntry(
                config_id=str(uuid.uuid4()),
                name=name,
                scope=ConfigurationScope.TEST,
                format=ConfigurationFormat.JSON,
                content=config_content,
                schema_id=template.schema_id,
                environment=environment,
                description=f"Generated from template {template.name}"
            )
            
            # Save configuration
            if self.save_configuration(config):
                return config
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error creating configuration from template: {e}")
            return None
    
    def _generate_config_from_template(self, template: ConfigurationTemplate, 
                                     parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate configuration content from template."""
        config_content = copy.deepcopy(template.template_data)
        
        # Replace parameters
        def replace_parameters(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: replace_parameters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_parameters(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                param_name = obj[2:-1]
                return parameters.get(param_name, obj)
            else:
                return obj
        
        return replace_parameters(config_content)
    
    def merge_configurations(self, base_config_id: str, override_config_id: str) -> Optional[ConfigurationEntry]:
        """Merge two configurations with override taking precedence."""
        try:
            base_config = self.load_configuration(base_config_id)
            override_config = self.load_configuration(override_config_id)
            
            if not base_config or not override_config:
                self.logger.error("❌ Base or override configuration not found")
                return None
            
            # Deep merge configurations
            merged_content = self._deep_merge(base_config.content, override_config.content)
            
            # Create merged configuration
            merged_config = ConfigurationEntry(
                config_id=str(uuid.uuid4()),
                name=f"{base_config.name}_merged_{override_config.name}",
                scope=base_config.scope,
                format=base_config.format,
                content=merged_content,
                schema_id=base_config.schema_id,
                environment=base_config.environment,
                description=f"Merged from {base_config.name} and {override_config.name}",
                parent_config_id=base_config.config_id
            )
            
            return merged_config
            
        except Exception as e:
            self.logger.error(f"❌ Error merging configurations: {e}")
            return None
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def validate_configuration(self, config_id: str) -> Tuple[bool, List[str]]:
        """Validate configuration against its schema."""
        config = self.load_configuration(config_id)
        if not config:
            return False, ["Configuration not found"]
        
        if not config.schema_id:
            return True, []  # No schema to validate against
        
        return self.validator.validate_configuration(config.content, config.schema_id)
    
    def get_configuration_history(self, config_id: str) -> List[ConfigurationEntry]:
        """Get configuration history."""
        return self.config_history.get(config_id, [])
    
    def rollback_configuration(self, config_id: str, version: int = -1) -> bool:
        """Rollback configuration to a previous version."""
        try:
            history = self.get_configuration_history(config_id)
            if not history:
                self.logger.error(f"❌ No history found for configuration {config_id}")
                return False
            
            if version < 0:
                version = len(history) + version
            
            if version < 0 or version >= len(history):
                self.logger.error(f"❌ Invalid version {version}")
                return False
            
            # Get the version to rollback to
            rollback_config = history[version]
            
            # Update current configuration
            current_config = self.configurations[config_id]
            current_config.content = rollback_config.content
            current_config.updated_at = datetime.now()
            
            # Save the rollback
            return self.save_configuration(current_config)
            
        except Exception as e:
            self.logger.error(f"❌ Error rolling back configuration: {e}")
            return False
    
    def export_configuration(self, config_id: str, format: ConfigurationFormat = ConfigurationFormat.JSON) -> str:
        """Export configuration to string."""
        config = self.load_configuration(config_id)
        if not config:
            return ""
        
        try:
            if format == ConfigurationFormat.JSON:
                return json.dumps(config.content, indent=2)
            elif format == ConfigurationFormat.YAML:
                return yaml.dump(config.content, default_flow_style=False)
            else:
                return json.dumps(config.content, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Error exporting configuration: {e}")
            return ""
    
    def import_configuration(self, config_data: str, format: ConfigurationFormat = ConfigurationFormat.JSON,
                           name: str = "imported_config", environment: str = "default") -> Optional[ConfigurationEntry]:
        """Import configuration from string."""
        try:
            if format == ConfigurationFormat.JSON:
                content = json.loads(config_data)
            elif format == ConfigurationFormat.YAML:
                content = yaml.safe_load(config_data)
            else:
                content = json.loads(config_data)
            
            # Create configuration entry
            config = ConfigurationEntry(
                config_id=str(uuid.uuid4()),
                name=name,
                scope=ConfigurationScope.TEST,
                format=format,
                content=content,
                environment=environment,
                description="Imported configuration"
            )
            
            # Save configuration
            if self.save_configuration(config):
                return config
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error importing configuration: {e}")
            return None


# Convenience function for easy integration
def create_configuration_manager(config_dir: str = "config") -> ConfigurationManager:
    """Create a configuration manager instance."""
    return ConfigurationManager(config_dir)