"""
Configuration Utilities

This module provides comprehensive configuration management utilities
for environment variables, configuration files, and runtime settings.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class ConfigSource(Enum):
    """Configuration source types."""
    ENV = "environment"
    FILE = "file"
    DICT = "dictionary"
    DEFAULT = "default"

@dataclass
class ConfigValue:
    """Represents a configuration value with metadata."""
    value: Any
    source: ConfigSource
    key: str
    description: Optional[str] = None
    required: bool = False
    default: Any = None
    type_hint: Optional[Type] = None

class EnvironmentConfig:
    """Environment variable configuration manager."""
    
    def __init__(self, prefix: str = "", case_sensitive: bool = False):
        """
        Initialize environment config manager.
        
        Args:
            prefix: Prefix for environment variables
            case_sensitive: Whether to use case-sensitive matching
        """
        self.prefix = prefix.upper()
        self.case_sensitive = case_sensitive
        self._cache: Dict[str, ConfigValue] = {}
    
    def get(self, key: str, default: Any = None, required: bool = False,
            type_hint: Optional[Type] = None, description: Optional[str] = None) -> Any:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            required: Whether the variable is required
            type_hint: Expected type for type conversion
            description: Description of the variable
            
        Returns:
            Environment variable value or default
        """
        env_key = self._format_key(key)
        
        # Check cache first
        if env_key in self._cache:
            return self._cache[env_key].value
        
        # Get from environment
        value = os.environ.get(env_key)
        
        if value is None:
            if required:
                raise ConfigError(f"Required environment variable not found: {env_key}")
            value = default
        
        # Type conversion
        if value is not None and type_hint:
            value = self._convert_type(value, type_hint)
        
        # Cache the result
        self._cache[env_key] = ConfigValue(
            value=value,
            source=ConfigSource.ENV,
            key=env_key,
            description=description,
            required=required,
            default=default,
            type_hint=type_hint
        )
        
        return value
    
    def get_bool(self, key: str, default: bool = False, required: bool = False) -> bool:
        """Get boolean environment variable."""
        return self.get(key, default, required, bool)
    
    def get_int(self, key: str, default: int = 0, required: bool = False) -> int:
        """Get integer environment variable."""
        return self.get(key, default, required, int)
    
    def get_float(self, key: str, default: float = 0.0, required: bool = False) -> float:
        """Get float environment variable."""
        return self.get(key, default, required, float)
    
    def get_list(self, key: str, default: List[str] = None, required: bool = False,
                 separator: str = ",") -> List[str]:
        """Get list environment variable (comma-separated by default)."""
        if default is None:
            default = []
        
        value = self.get(key, default, required, str)
        if isinstance(value, str):
            return [item.strip() for item in value.split(separator) if item.strip()]
        return value
    
    def get_dict(self, key: str, default: Dict[str, Any] = None, required: bool = False) -> Dict[str, Any]:
        """Get dictionary environment variable (JSON format)."""
        if default is None:
            default = {}
        
        value = self.get(key, default, required, str)
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ConfigError(f"Invalid JSON in environment variable {key}: {e}")
        return value
    
    def _format_key(self, key: str) -> str:
        """Format key with prefix and case handling."""
        formatted_key = key.upper() if not self.case_sensitive else key
        if self.prefix:
            formatted_key = f"{self.prefix}_{formatted_key}"
        return formatted_key
    
    def _convert_type(self, value: str, type_hint: Type) -> Any:
        """Convert string value to specified type."""
        try:
            if type_hint == bool:
                return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
            elif type_hint == int:
                return int(value)
            elif type_hint == float:
                return float(value)
            elif type_hint == str:
                return str(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ConfigError(f"Failed to convert '{value}' to {type_hint.__name__}: {e}")
    
    def get_all_with_prefix(self) -> Dict[str, ConfigValue]:
        """Get all environment variables with the configured prefix."""
        prefix = self.prefix + "_" if self.prefix else ""
        result = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                result[clean_key] = ConfigValue(
                    value=value,
                    source=ConfigSource.ENV,
                    key=key,
                    description=None
                )
        
        return result

class FileConfig:
    """File-based configuration manager."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize file config manager.
        
        Args:
            config_dir: Directory to look for config files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / "config"
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load_json(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """Load JSON configuration file."""
        file_path = self.config_dir / filename
        return self._load_file(file_path, json.load, required)
    
    def load_yaml(self, filename: str, required: bool = True) -> Dict[str, Any]:
        """Load YAML configuration file."""
        file_path = self.config_dir / filename
        return self._load_file(file_path, yaml.safe_load, required)
    
    def save_json(self, config: Dict[str, Any], filename: str) -> bool:
        """Save configuration to JSON file."""
        file_path = self.config_dir / filename
        return self._save_file(file_path, config, json.dump)
    
    def save_yaml(self, config: Dict[str, Any], filename: str) -> bool:
        """Save configuration to YAML file."""
        file_path = self.config_dir / filename
        return self._save_file(file_path, config, yaml.dump)
    
    def _load_file(self, file_path: Path, loader_func, required: bool) -> Dict[str, Any]:
        """Load configuration file with caching."""
        if str(file_path) in self._cache:
            return self._cache[str(file_path)]
        
        if not file_path.exists():
            if required:
                raise ConfigError(f"Required config file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r') as f:
                config = loader_func(f)
            
            self._cache[str(file_path)] = config
            logger.debug(f"Loaded config from {file_path}")
            return config
            
        except Exception as e:
            raise ConfigError(f"Failed to load config from {file_path}: {e}")
    
    def _save_file(self, file_path: Path, config: Dict[str, Any], dumper_func) -> bool:
        """Save configuration to file."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                dumper_func(config, f, indent=2)
            
            self._cache[str(file_path)] = config
            logger.debug(f"Saved config to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            return False

class ConfigManager:
    """Unified configuration manager."""
    
    def __init__(self, env_prefix: str = "", config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            env_prefix: Prefix for environment variables
            config_dir: Directory for configuration files
        """
        self.env_config = EnvironmentConfig(env_prefix)
        self.file_config = FileConfig(config_dir)
        self._overrides: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None, required: bool = False,
            type_hint: Optional[Type] = None, description: Optional[str] = None) -> Any:
        """
        Get configuration value from multiple sources.
        
        Priority order:
        1. Overrides (set via set_override)
        2. Environment variables
        3. Default value
        
        Args:
            key: Configuration key
            default: Default value
            required: Whether the value is required
            type_hint: Expected type
            description: Description of the value
            
        Returns:
            Configuration value
        """
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]
        
        # Try environment variable
        try:
            return self.env_config.get(key, default, required, type_hint, description)
        except ConfigError:
            if required:
                raise
            return default
    
    def set_override(self, key: str, value: Any) -> None:
        """Set configuration override."""
        self._overrides[key] = value
    
    def clear_override(self, key: str) -> None:
        """Clear configuration override."""
        self._overrides.pop(key, None)
    
    def clear_all_overrides(self) -> None:
        """Clear all configuration overrides."""
        self._overrides.clear()
    
    def load_from_file(self, filename: str, file_type: str = "json") -> Dict[str, Any]:
        """Load configuration from file."""
        if file_type.lower() == "json":
            return self.file_config.load_json(filename)
        elif file_type.lower() in ("yaml", "yml"):
            return self.file_config.load_yaml(filename)
        else:
            raise ConfigError(f"Unsupported file type: {file_type}")
    
    def save_to_file(self, config: Dict[str, Any], filename: str, file_type: str = "json") -> bool:
        """Save configuration to file."""
        if file_type.lower() == "json":
            return self.file_config.save_json(config, filename)
        elif file_type.lower() in ("yaml", "yml"):
            return self.file_config.save_yaml(config, filename)
        else:
            raise ConfigError(f"Unsupported file type: {file_type}")
    
    def get_all_env_vars(self) -> Dict[str, ConfigValue]:
        """Get all environment variables with the configured prefix."""
        return self.env_config.get_all_with_prefix()
    
    def validate_required(self, required_keys: List[str]) -> Dict[str, Any]:
        """Validate that all required configuration keys are present."""
        missing = []
        values = {}
        
        for key in required_keys:
            try:
                values[key] = self.get(key, required=True)
            except ConfigError:
                missing.append(key)
        
        if missing:
            raise ConfigError(f"Missing required configuration keys: {missing}")
        
        return values

# Convenience functions
def get_env_var(key: str, default: Any = None, required: bool = False, 
                type_hint: Optional[Type] = None) -> Any:
    """Convenience function to get environment variable."""
    env_config = EnvironmentConfig()
    return env_config.get(key, default, required, type_hint)

def get_env_bool(key: str, default: bool = False, required: bool = False) -> bool:
    """Convenience function to get boolean environment variable."""
    return get_env_var(key, default, required, bool)

def get_env_int(key: str, default: int = 0, required: bool = False) -> int:
    """Convenience function to get integer environment variable."""
    return get_env_var(key, default, required, int)

def get_env_float(key: str, default: float = 0.0, required: bool = False) -> float:
    """Convenience function to get float environment variable."""
    return get_env_var(key, default, required, float)

def get_env_list(key: str, default: List[str] = None, required: bool = False,
                 separator: str = ",") -> List[str]:
    """Convenience function to get list environment variable."""
    if default is None:
        default = []
    env_config = EnvironmentConfig()
    return env_config.get_list(key, default, required, separator)

def load_config_file(filename: str, config_dir: Optional[Union[str, Path]] = None,
                     file_type: str = "json") -> Dict[str, Any]:
    """Convenience function to load configuration file."""
    file_config = FileConfig(config_dir)
    if file_type.lower() == "json":
        return file_config.load_json(filename)
    elif file_type.lower() in ("yaml", "yml"):
        return file_config.load_yaml(filename)
    else:
        raise ConfigError(f"Unsupported file type: {file_type}")

# Global configuration manager instance
global_config = ConfigManager()

# ============================================================
#  Step-specific lightweight config objects
# ============================================================

@dataclass(slots=True, frozen=True)
class Step06LabelParams:
    """Constant parameters for Step-06 labeling logic.

    Keeping them in a dataclass lets tests inject alternate values without
    touching production code.  The class is *frozen* to prevent accidental
    mutation at runtime and *slots* to minimise memory usage.
    """

    profit_take: float = 0.004   # 0.4 % upward threshold
    stop_loss:   float = 0.003   # 0.3 % downward threshold
    tx_cost:     float = 0.0008  # 0.08 % transaction cost (future use)

__all__ = [
    'ConfigError',
    'ConfigSource',
    'ConfigValue',
    'EnvironmentConfig',
    'FileConfig',
    'ConfigManager',
    'get_env_var',
    'get_env_bool',
    'get_env_int',
    'get_env_float',
    'get_env_list',
    'load_config_file',
    'Step06LabelParams',
    'global_config'
]