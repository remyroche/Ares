"""
Unified Configuration Service

This service provides a unified interface for loading configurations from both
the old (configs/) and new (config/) directory structures, ensuring backward
compatibility while supporting the new unified structure.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Try to import system logger, fall back to standard logging
try:
    from ..utils.logger import system_logger as logger
except ImportError:
    logger = logging.getLogger(__name__)

class UnifiedConfigService:
    """
    Unified configuration service that handles both old and new configuration paths.

    Provides backward compatibility while supporting the new unified structure:
    - config/environments/ for environment-specific configs
    - config/features/ for feature-specific configs
    - config/ for system configs
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the unified configuration service.

        Args:
            base_path: Base path for configurations (defaults to workspace root)
        """
        self.base_path = base_path or Path("/workspace")
        self.config_path = self.base_path / "config"
        self.legacy_configs_path = self.base_path / "configs"

        # Path mapping for backward compatibility
        self.legacy_path_mapping = {
            # Environment configs
            "configs/development_config.json": "config/environments/development.json",
            "configs/production_config.json": "config/environments/production.json",
            "configs/testing_config.json": "config/environments/testing.json",

            # Feature configs (these were already in config/)
            "config/enhanced_reporting_config.yaml": "config/features/enhanced_reporting_config.yaml",
            "config/explainability_config.yaml": "config/features/explainability_config.yaml",
            "config/probabilistic_optimization.yaml": "config/features/probabilistic_optimization.yaml",
            "config/sr_levels_config.yaml": "config/features/sr_levels_config.yaml",
            "config/training_config.json": "config/features/training_config.json",
            "config/training_modes.yaml": "config/features/training_modes.yaml",
        }

    def resolve_path(self, config_path: Union[str, Path]) -> Path:
        """
        Resolve configuration path, handling both old and new structures.

        Args:
            config_path: Path to configuration file (can be old or new format)

        Returns:
            Resolved Path object pointing to the actual configuration file
        """
        config_path = Path(config_path)

        # Convert to string for mapping lookup
        path_str = str(config_path)

        # Check if this is a legacy path that needs mapping
        if path_str in self.legacy_path_mapping:
            new_path = self.base_path / self.legacy_path_mapping[path_str]
            logger.info(f"Resolved legacy path '{path_str}' to '{new_path}'")
            return new_path

        # Check if it's a relative path that needs base path prepending
        if not config_path.is_absolute():
            full_path = self.base_path / config_path
        else:
            full_path = config_path

        return full_path

    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file, supporting both old and new paths.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If file format is not supported
        """
        resolved_path = self.resolve_path(config_path)

        if not resolved_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {resolved_path}")

        # Load based on file extension
        if resolved_path.suffix.lower() == '.json':
            with open(resolved_path, 'r') as f:
                return json.load(f)
        elif resolved_path.suffix.lower() in ['.yaml', '.yml']:
            with open(resolved_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {resolved_path.suffix}")

    def load_environment_config(self, environment: str) -> Dict[str, Any]:
        """
        Load environment-specific configuration.

        Args:
            environment: Environment name (development, production, testing)

        Returns:
            Environment configuration dictionary
        """
        config_path = f"config/environments/{environment}.json"
        return self.load_config(config_path)

    def load_feature_config(self, feature_name: str) -> Dict[str, Any]:
        """
        Load feature-specific configuration.

        Args:
            feature_name: Feature name (without extension)

        Returns:
            Feature configuration dictionary
        """
        # Try different extensions
        for ext in ['.yaml', '.yml', '.json']:
            config_path = f"config/features/{feature_name}{ext}"
            try:
                return self.load_config(config_path)
            except FileNotFoundError:
                continue

        raise FileNotFoundError(f"Feature configuration not found: {feature_name}")

    def load_version_config(self) -> Dict[str, Any]:
        """
        Load version configuration.

        Returns:
            Version configuration dictionary
        """
        return self.load_config("config/version_config.json")

    def list_available_configs(self) -> Dict[str, list]:
        """
        List all available configurations.

        Returns:
            Dictionary with categories and available configs
        """
        configs = {
            'environments': [],
            'features': [],
            'system': []
        }

        # List environment configs
        env_path = self.config_path / "environments"
        if env_path.exists():
            configs['environments'] = [f.stem for f in env_path.glob("*.json")]

        # List feature configs
        features_path = self.config_path / "features"
        if features_path.exists():
            configs['features'] = [f.stem for f in features_path.glob("*")]

        # List system configs
        system_configs = list(self.config_path.glob("*.json")) + list(self.config_path.glob("*.yaml"))
        configs['system'] = [f.stem for f in system_configs if f.name != "README.md"]

        return configs

    def validate_config(self, config: Dict[str, Any], config_type: str = "general") -> bool:
        """
        Validate configuration structure.

        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration (environment, feature, system)

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        if config_type == "environment":
            required_fields = ["environment", "version", "symbol", "exchange"]
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Environment config missing required field: {field}")

        return True

# Global instance for easy access
config_service = UnifiedConfigService()

# Convenience functions for backward compatibility
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file (backward compatibility)."""
    return config_service.load_config(config_path)

def load_environment_config(environment: str) -> Dict[str, Any]:
    """Load environment-specific configuration."""
    return config_service.load_environment_config(environment)

def load_feature_config(feature_name: str) -> Dict[str, Any]:
    """Load feature-specific configuration."""
    return config_service.load_feature_config(feature_name)

def load_version_config() -> Dict[str, Any]:
    """Load version configuration."""
    return config_service.load_version_config()

def validate_config(config: Dict[str, Any], config_type: str = "general") -> bool:
    """Validate configuration structure."""
    return config_service.validate_config(config, config_type)
