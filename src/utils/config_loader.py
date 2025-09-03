# src/utils/config_loader.py

import os
import os.path
from typing import Any

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.pipeline_standards import PipelineStandards, pipeline_standards
from src.utils.warning_symbols import error, missing, yaml

class ConfigLoader:
    """Utility class for loading YAML configuration files."""

    def __init__(self):
        self.logger = system_logger.getChild("ConfigLoader")

    @handles_errors
    def load_yaml_config(self, config_path: str) -> dict[str, Any]:
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Dictionary containing the configuration
        """
        if not os.path.exists(config_path):
            self.print(missing(f"Config file not found: {config_path}"))
            return {}

        try:
            with open(config_path, encoding="utf-8") as file:
                config = yaml.safe_load(file)

            self.logger.info(f"Successfully loaded config from: {config_path}")
            return config or {}

        except Exception as e:
            self.print(error(f"Error loading config from {config_path}: {e}"))
            return {}

    @handles_errors
    def load_position_sizing_config(self, config_dir: str = "config") -> dict[str, Any]:
        """Load position sizing configuration.

        Args:
            config_dir: Directory containing config files

        Returns:
            Position sizing configuration dictionary
        """
        config_path = os.path.join(config_dir, "position_sizing.yaml")
        return self.load_yaml_config(config_path)

    @handles_errors
    def load_leverage_sizing_config(self, config_dir: str = "config") -> dict[str, Any]:
        """Load leverage sizing configuration.

        Args:
            config_dir: Directory containing config files

        Returns:
            Leverage sizing configuration dictionary
        """
        config_path = os.path.join(config_dir, "leverage_sizing.yaml")
        return self.load_yaml_config(config_path)

    @handles_errors
    def load_combined_sizing_config(self, config_dir: str = "config") -> dict[str, Any]:
        """Load combined position and leverage sizing configuration.

        Args:
            config_dir: Directory containing config files

        Returns:
            Combined sizing configuration dictionary
        """
        config_path = os.path.join(config_dir, "combined_sizing.yaml")
        return self.load_yaml_config(config_path)

    @handles_errors
    def validate_config(self, config: dict[str, Any], config_type: str) -> bool:
        """Validate configuration structure.

        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration ("position", "leverage", or "combined")

        Returns:
            True if configuration is valid = False otherwise
        """
        if not config:
            self.print(error(f"Empty {config_type} configuration"))
            return False

        # Check for required sections
        if "risk_management" not in config:
            self.logger.error(
                f"Missing 'risk_management' section in {config_type} config",
            )
            return False

        risk_management = config["risk_management"]

        if config_type in ["position", "combined"]:
            if "position_sizing" not in risk_management:
                self.logger.error(
                    f"Missing 'position_sizing' section in {config_type} config",
                )
                return False

        if config_type in ["leverage", "combined"]:
            if "leverage_sizing" not in risk_management:
                self.logger.error(
                    f"Missing 'leverage_sizing' section in {config_type} config",
                )
                return False

        if "dynamic_risk_management" not in risk_management:
            self.logger.error(
                f"Missing 'dynamic_risk_management' section in {config_type} config",
            )
            return False

        if "liquidation_risk" not in risk_management:
            self.logger.error(
                f"Missing 'liquidation_risk' section in {config_type} config",
            )
            return False

        self.logger.info(f"✅ {config_type} configuration validation passed")
        return True

    @handles_errors
    def merge_configs(self, *configs: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple configuration dictionaries.

        Args:
            *configs: Configuration dictionaries to merge

        Returns:
            Merged configuration dictionary
        """
        merged_config = {}

        for config in configs:
            if config:
                self._deep_merge(merged_config, config)

        return merged_config

    def _deep_merge(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """Deep merge source dictionary into target dictionary.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if (
                key in target
                and isinstance(target[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    @handles_errors
    def load_config_with_fallback(
        self,
        primary_config: str,
        fallback_config: str = "config",
        config_dir: str = "config",
    ) -> dict[str, Any]:
        """Load configuration with fallback to another config file.

        Args:
            primary_config: Primary configuration file name
            fallback_config: Fallback configuration file name
            config_dir: Directory containing config files

        Returns:
            Configuration dictionary
        """
        # Try to load primary config
        primary_path = os.path.join(config_dir, primary_config)
        config = self.load_yaml_config(primary_path)

        if config:
            self.logger.info(f"Loaded primary config: {primary_config}")
            return config

        # Try to load fallback config
        fallback_path = os.path.join(config_dir, fallback_config)
        config = self.load_yaml_config(fallback_path)

        if config:
            self.logger.info(f"Loaded fallback config: {fallback_config}")
            return config

        self.logger.warning(
            f"No configuration found in {primary_config} or {fallback_config}",
        )
        return {}

    # Convenience functions

    def load_position_sizing_config(config_dir: str = "config") -> dict[str, Any]:
        """Load position sizing configuration."""
        loader = ConfigLoader()
        return loader.load_position_sizing_config(config_dir)

    def load_leverage_sizing_config(config_dir: str = "config") -> dict[str, Any]:
        """Load leverage sizing configuration."""
        loader = ConfigLoader()
        return loader.load_leverage_sizing_config(config_dir)

    def load_combined_sizing_config(config_dir: str = "config") -> dict[str, Any]:
        """Load combined sizing configuration."""
        loader = ConfigLoader()
        return loader.load_combined_sizing_config(config_dir)

    def load_config_with_fallback(
        primary_config: str,
        fallback_config: str,
        config_dir: str = "config",
    ) -> dict[str, Any]:
        """Load configuration with fallback."""
        loader = ConfigLoader()
        return loader.load_config_with_fallback(
            primary_config, fallback_config, config_dir
        )
