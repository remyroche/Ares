"""
Configuration utilities for loading and managing configuration files.

This module provides utilities for loading, validating, and managing
configuration files and settings.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List

# Setup logging
logger = logging.getLogger(__name__)

def load_config_file(file_path: Union[str, Path], config_type: str = "auto") -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Configuration file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if config_type == "auto":
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    logger.error(f"Unsupported configuration file type: {file_path.suffix}")
                    return {}
            elif config_type == "yaml":
                return yaml.safe_load(f)
            elif config_type == "json":
                return json.load(f)
            else:
                logger.error(f"Unsupported config_type: {config_type}")
                return {}
    except Exception as e:
        logger.error(f"Error loading configuration file {file_path}: {e}")
        return {}

def save_config_file(config: Dict[str, Any], file_path: Union[str, Path], config_type: str = "auto") -> bool:
    """Save configuration to file."""
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if config_type == "auto":
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config, f, default_flow_style=False)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    logger.error(f"Unsupported configuration file type: {file_path.suffix}")
                    return False
            elif config_type == "yaml":
                yaml.dump(config, f, default_flow_style=False)
            elif config_type == "json":
                json.dump(config, f, indent=2)
            else:
                logger.error(f"Unsupported config_type: {config_type}")
                return False
        return True
    except Exception as e:
        logger.error(f"Error saving configuration file {file_path}: {e}")
        return False

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate configuration has required keys."""
    try:
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            logger.error(f"Missing required configuration keys: {missing_keys}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Get configuration value with default."""
    try:
        return config.get(key, default)
    except Exception:
        return default

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence."""
    try:
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    except Exception as e:
        logger.error(f"Error merging configurations: {e}")
        return base_config

def get_nested_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation."""
    try:
        keys = key_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    except Exception:
        return default

def set_nested_config_value(config: Dict[str, Any], key_path: str, value: Any) -> bool:
    """Set nested configuration value using dot notation."""
    try:
        keys = key_path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
        return True
    except Exception as e:
        logger.error(f"Error setting nested configuration value: {e}")
        return False
