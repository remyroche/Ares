"""
Configuration Security Framework

This module provides standardized configuration security including:
- Secure configuration loading and validation
- Environment-specific configuration management
- Sensitive configuration encryption
- Configuration access control
- Configuration audit logging
- Configuration backup and recovery
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
        self.standards = pipeline_standards
        self.logger = system_logger.getChild("ConfigurationSecurity")
        self.security = security_framework
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
        default_return=None,
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
            cache_key = f"{config_path}_{environment}"
            if cache_key in self.config_cache:
                self.logger.debug(f"Using cached configuration: {config_path}")
                return self.config_cache[cache_key]

            # Load configuration file
            config_data = self._load_config_file(config_path, config_type)

            # Apply environment-specific overrides
            if self.security_policies["environment_isolation"]:
                config_data = self._apply_environment_overrides(config_data, environment)

            # Validate configuration schema
            if self.security_policies["validate_config_schemas"]:
                self._validate_config_schema(config_data, config_type)

            # Encrypt sensitive values
            if self.security_policies["encrypt_sensitive_configs"]:
                config_data = self._encrypt_sensitive_configs(config_data)

            # Calculate configuration hash
            config_hash = self._calculate_config_hash(config_data)
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
        """Apply environment-specific configuration overrides."""
        if "environments" in config_data and environment in config_data["environments"]:
            env_overrides = config_data["environments"][environment]

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
                section_data = config_data[section]

                # Check required fields
                for required_field in schema["required"]:
                    if required_field not in section_data:
                        raise ValueError(f"Missing required field '{required_field}' in {section}")

                # Check field types
                for field, expected_type in schema["types"].items():
                    if field in section_data:
                        actual_type = type(section_data[field]).__name__
                        if actual_type != expected_type:
                            self.logger.warning(f"Type mismatch in {section}.{field}: expected {expected_type}, got {actual_type}")

    def _encrypt_sensitive_configs(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration values."""
        encrypted_config = config_data.copy()

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
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _backup_configuration(self, config_path: str) -> None:
        """Create backup of configuration file."""
        try:
            if os.path.exists(config_path):
                backup_path = f"{config_path}.backup.{int(datetime.now().timestamp())}"

                # Copy file
                import shutil
                shutil.copy2(config_path, backup_path)

                # Encrypt backup
                encrypted_backup_path = self.security.data_encryption.encrypt_file(backup_path)

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
            cache_key = f"{config_path}_{environment}"

            # Load configuration
            config_data = self.load_secure_configuration(config_path, environment=environment)

            # Check hash integrity
            current_hash = self._calculate_config_hash(config_data)
            stored_hash = self.config_hashes.get(cache_key)

            if stored_hash and current_hash != stored_hash:
                self.logger.warning(f"Configuration integrity check failed for {config_path}")
                return False

            self.logger.info(f"Configuration integrity validated: {config_path}")
            return True

        except Exception as e:
            self.logger.error(f"Configuration integrity validation failed: {e}")
            return False


# Global configuration security manager instance
configuration_security_manager = ConfigurationSecurityManager()