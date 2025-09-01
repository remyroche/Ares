"""
Configuration Security Framework

This module provides standardized configuration security including:
    pass - Secure configuration loading and validation - Environment - specific configuration management - Sensitive configuration encryption - Configuration access control - Configuration audit logging - Configuration backup and recovery
"""

import os
import json
import yaml
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from .pipeline_standards import PipelineStandards, pipeline_standards
from .security_framework import security_framework, SecurityLevel
from .logger import system_logger
from .error_handler import handle_errors

class ConfigurationSecurityManager:
    """Manages configuration security and validation."""

    def __init__(self):
        """Initialize configuration security manager."""
        self.standards, pipeline_standards
        self.logger, system_logger.getChild("ConfigurationSecurity")
        self.security, security_framework
        self.config_cache = {}
        self.config_hashes = {}

        # Configuration security policies
        self.security_policies = {
            "encrypt_sensitive_configs": True,
            "validate_config_schemas": True,
            "audit_config_access": True,
            "backup_configs": True,
            "environment_isolation": True,
            "config_versioning": True
        }

        # Sensitive configuration keys
        self.sensitive_keys = [
            "api_key",
            "api_secret",
            "password",
            "secret",
            "token",
            "private_key",
            "database_password",
            "encryption_key"
        ]

        # Configuration schemas
        self.config_schemas = {
            "database": {
                "required": ["host", "port", "database", "username"],
                "optional": ["password", "ssl_mode", "connection_timeout"],
                "types": {
                    "host": "str",
                    "port": "int",
                    "database": "str",
                    "username": "str",
                    "password": "str",
                    "ssl_mode": "str",
                    "connection_timeout": "int"
                }
            },
            "exchange": {
                "required": ["name", "api_key", "api_secret"],
                "optional": ["testnet", "rate_limit", "timeout"],
                "types": {
                    "name": "str",
                    "api_key": "str",
                    "api_secret": "str",
                    "testnet": "bool",
                    "rate_limit": "int",
                    "timeout": "int"
                }
            },
            "security": {
                "required": ["encryption_enabled", "audit_logging"],
                "optional": ["session_timeout", "max_login_attempts"],
                "types": {
                    "encryption_enabled": "bool",
                    "audit_logging": "bool",
                    "session_timeout": "int",
                    "max_login_attempts": "int"
                }
            }
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return = None,
        context="configuration loading"
    )
    def load_secure_configuration(self, config_path: str, config_type: str = "yaml",
                                environment: str = "production") -> Dict[str, Any]:
        """Load configuration securely.

        Args:
            config_path: Path to configuration file
            config_type: Type of configuration file (yaml, json)
            environment: Environment name

        Returns:
            Loaded configuration
        """
        try:
        # Check if configuration is cached
            cache_key, f"{config_path}_{environment}"
        if cache_key in self.config_cache:
        self.logger.debug(f"Using cached configuration: {config_path}")
        return self.config_cache[cache_key]

        # Load configuration file
            config_data, self._load_config_file(config_path, config_type)

        # Apply environment - specific overrides
        if self.security_policies["environment_isolation"]:
                config_data, self._apply_environment_overrides(config_data, environment)

        # Validate configuration schema
        if self.security_policies["validate_config_schemas"]:
        self._validate_config_schema(config_data, config_type)

        # Encrypt sensitive values
        if self.security_policies["encrypt_sensitive_configs"]:
                config_data, self._encrypt_sensitive_configs(config_data)

        # Calculate configuration hash
            config_hash, self._calculate_config_hash(config_data)
        self.config_hashes[cache_key] = config_hash

        # Cache configuration
        self.config_cache[cache_key] = config_data

        # Log configuration access
        if self.security_policies["audit_config_access"]:
        self.security.audit_logger.log_security_event(
                    "config_access",
                    "system",
                    f"Loaded configuration: {config_path}",
                    {"config_path": config_path, "environment": environment, "config_hash": config_hash},
                    SecurityLevel.LOW
                )

        self.logger.info(f"Configuration loaded securely: {config_path}")
        return config_data

        except Exception as e:
        self.logger.error(f"Failed to load configuration {config_path}: {e}")
            raise

    def _load_config_file(self, config_path: str, config_type: str) -> Dict[str, Any]:
        """Load configuration file based on type."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
        if config_type.lower() == "yaml":
        return yaml.safe_load(f)
            elif config_type.lower() == "json":
        return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration type: {config_type}")

    def _apply_environment_overrides(self, config_data: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply environment - specific configuration overrides."""
        if "environments" in config_data and environment in config_data["environments"]:
            env_overrides, config_data["environments"][environment]

        # Merge environment overrides
        for key, value in env_overrides.items():
        if isinstance(value, dict) and key in config_data:
                    config_data[key].update(value)
                else:
                    config_data[key] = value

        return config_data

    def _validate_config_schema(self, config_data: Dict[str, Any], config_type: str) -> None:
        """Validate configuration against schema."""
        # This is a simplified validation - in practice, you'd use a proper schema validation library
        for section, schema in self.config_schemas.items():
        if section in config_data:
                section_data, config_data[section]

        # Check required fields
        for required_field in schema["required"]:
        if required_field not in section_data:
                        raise ValueError(f"Missing required field '{required_field}' in {section}")

        # Check field types
        for field, expected_type in schema["types"].items():
        if field in section_data:
                        actual_type, type(section_data[field]).__name__
        if actual_type != expected_type:
        self.logger.warning(f"Type mismatch in {section}.{field}: expected {expected_type}, got {actual_type}")

    def _encrypt_sensitive_configs(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values."""
        encrypted_config, config_data.copy()

        def encrypt_dict(data: Dict[str, Any]) -> Dict[str, Any]:
            encrypted = {}
        for key, value in data.items():
        if isinstance(value, dict):
                    encrypted[key] = encrypt_dict(value)
                elif isinstance(value, str) and any(sensitive in key.lower() for sensitive in self.sensitive_keys):
                    encrypted[key] = self.security.data_encryption.encrypt_data(value)
                else:
                    encrypted[key] = value
        return encrypted

        return encrypt_dict(encrypted_config)

    def _calculate_config_hash(self, config_data: Dict[str, Any]) -> str:
        """Calculate hash of configuration data."""
        config_str, json.dumps(config_data, sort_keys = True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def get_config_value(self, config_data: Dict[str, Any], key_path: str, default: Any, None) -> Any:
        """Get configuration value by key path.

        Args:
            config_data: Configuration data
            key_path: Dot - separated key path (e.g., "database.host")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        try:
            keys, key_path.split('.')
            value, config_data

        for key in keys:
        if isinstance(value, dict) and key in value:
                    value, value[key]
                else:
        return default

        # Decrypt if necessary
        if isinstance(value, bytes) and self.security_policies["encrypt_sensitive_configs"]:
        try:
                    value, self.security.data_encryption.decrypt_data(value)
        except Exception as e:
        self.logger.warning(f"Failed to decrypt config value {key_path}: {e}")

        return value

        except Exception as e:
        self.logger.error(f"Error accessing config value {key_path}: {e}")
        return default

    def set_config_value(self, config_data: Dict[str, Any], key_path: str, value: Any,
                        encrypt: bool, True) -> Dict[str, Any]:
        """Set configuration value by key path.

        Args:
            config_data: Configuration data
            key_path: Dot - separated key path
            value: Value to set
            encrypt: Whether to encrypt sensitive values

        Returns:
            Updated configuration data
        """
        try:
            keys, key_path.split('.')
            updated_config, config_data.copy()
            current, updated_config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
        if key not in current:
                    current[key] = {}
                current, current[key]

        # Set the value
            target_key, keys[-1]

        # Encrypt if necessary
        if encrypt and any(sensitive in target_key.lower() for sensitive in self.sensitive_keys):
                value, self.security.data_encryption.encrypt_data(str(value))

            current[target_key] = value

        # Log configuration change
        if self.security_policies["audit_config_access"]:
        self.security.audit_logger.log_security_event(
                    "config_change",
                    "system",
                    f"Updated configuration: {key_path}",
                    {"key_path": key_path, "encrypted": encrypt},
                    SecurityLevel.MEDIUM
                )

        return updated_config

        except Exception as e:
        self.logger.error(f"Error setting config value {key_path}: {e}")
            raise

    def save_secure_configuration(self, config_data: Dict[str, Any], config_path: str,
                                config_type: str = "yaml") -> None:
        """Save configuration securely.

        Args:
            config_data: Configuration data to save
            config_path: Path to save configuration
            config_type: Type of configuration file
        """
        try:
        # Create backup if enabled
        if self.security_policies["backup_configs"]:
        self._backup_configuration(config_path)

        # Save configuration
        with open(config_path, 'w') as f:
        if config_type.lower() == "yaml":
                    yaml.dump(config_data, f, default_flow_style = False, indent = 2)
                elif config_type.lower() == "json":
                    json.dump(config_data, f, indent = 2)
                else:
                    raise ValueError(f"Unsupported configuration type: {config_type}")

        # Update cache
            cache_key, f"{config_path}_production"  # Default environment
        self.config_cache[cache_key] = config_data
        self.config_hashes[cache_key] = self._calculate_config_hash(config_data)

        # Log configuration save
        if self.security_policies["audit_config_access"]:
        self.security.audit_logger.log_security_event(
                    "config_save",
                    "system",
                    f"Saved configuration: {config_path}",
                    {"config_path": config_path, "config_type": config_type},
                    SecurityLevel.MEDIUM
                )

        self.logger.info(f"Configuration saved securely: {config_path}")

        except Exception as e:
        self.logger.error(f"Failed to save configuration {config_path}: {e}")
            raise

    def _backup_configuration(self, config_path: str) -> None:
        """Create backup of configuration file."""
        try:
        if os.path.exists(config_path):
                backup_path, f"{config_path}.backup.{int(datetime.now().timestamp())}"

        # Copy file
                import shutil
                shutil.copy2(config_path, backup_path)

        # Encrypt backup
                encrypted_backup_path, self.security.data_encryption.encrypt_file(backup_path)

        # Remove unencrypted backup
                os.remove(backup_path)

        self.logger.info(f"Configuration backup created: {encrypted_backup_path}")

        except Exception as e:
        self.logger.warning(f"Failed to create configuration backup: {e}")

    def validate_configuration_integrity(self, config_path: str, environment: str = "production") -> bool:
        """Validate configuration integrity.

        Args:
            config_path: Path to configuration file
            environment: Environment name

        Returns:
            True if configuration is valid
        """
        try:
            cache_key, f"{config_path}_{environment}"

        # Load configuration
            config_data, self.load_secure_configuration(config_path, environment = environment)

        # Check hash integrity
            current_hash, self._calculate_config_hash(config_data)
            stored_hash, self.config_hashes.get(cache_key)

        if stored_hash and current_hash != stored_hash:
        self.logger.warning(f"Configuration integrity check failed for {config_path}")
        return False

        self.logger.info(f"Configuration integrity validated: {config_path}")
        return True

        except Exception as e:
        self.logger.error(f"Configuration integrity validation failed: {e}")
        return False

    def get_configuration_security_report(self) -> Dict[str, Any]:
        """Get configuration security report.

        Returns:
            Configuration security report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "security_policies": self.security_policies,
            "cached_configs": len(self.config_cache),
            "config_hashes": len(self.config_hashes),
            "sensitive_keys": self.sensitive_keys,
            "config_schemas": list(self.config_schemas.keys()),
            "encryption_enabled": self.security_policies["encrypt_sensitive_configs"],
            "audit_logging_enabled": self.security_policies["audit_config_access"],
            "backup_enabled": self.security_policies["backup_configs"]
        }

        return report

# Global configuration security manager instance
configuration_security_manager, ConfigurationSecurityManager()