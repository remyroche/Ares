# src/utils/config_loader.py

import os
import yaml
from typing import Any, Dict, List, Optional, Union
import logging
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading YAML configuration files with validation and error handling.
    
    This class provides methods to load various types of configuration files
    including position sizing, leverage sizing, and combined configurations.
    It includes validation, fallback mechanisms, and proper error handling.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigLoader.
        
        Args:
            config_dir: Directory containing configuration files. 
                       If None, uses current working directory.
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd()
        self.logger = logger.getChild("ConfigLoader")
        self.is_initialized = False
        
        # Validate config directory exists
        if not self.config_dir.exists():
            self.logger.warning(f"Config directory does not exist: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created config directory: {self.config_dir}")
    
    async def initialize(self) -> bool:
        """Initialize ConfigLoader asynchronously."""
        try:
            self.logger.info(f"🚀 Initializing {self.__class__.__name__}...")
            self.is_initialized = True
            self.logger.info(f"✅ {self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {self.__class__.__name__}: {e}")
            return False
    
    def load_yaml_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with error handling.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Configuration dictionary or empty dict if loading fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return {}
        
        try:
            with open(config_path, encoding="utf-8") as file:
                config = yaml.safe_load(file)
            
            self.logger.info(f"Successfully loaded config from: {config_path}")
            return config or {}
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {config_path}: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config from {config_path}: {e}")
            return {}
    
    def load_position_sizing_config(self, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load position sizing configuration.
        
        Args:
            config_dir: Directory containing config files. Uses instance config_dir if None.
            
        Returns:
            Position sizing configuration dictionary
        """
        config_dir = Path(config_dir) if config_dir else self.config_dir
        config_path = config_dir / "position_sizing.yaml"
        return self.load_yaml_config(config_path)
    
    def load_leverage_sizing_config(self, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load leverage sizing configuration.
        
        Args:
            config_dir: Directory containing config files. Uses instance config_dir if None.
            
        Returns:
            Leverage sizing configuration dictionary
        """
        config_dir = Path(config_dir) if config_dir else self.config_dir
        config_path = config_dir / "leverage_sizing.yaml"
        return self.load_yaml_config(config_path)
    
    def load_combined_sizing_config(self, config_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Load combined sizing configuration.
        
        Args:
            config_dir: Directory containing config files. Uses instance config_dir if None.
            
        Returns:
            Combined sizing configuration dictionary
        """
        config_dir = Path(config_dir) if config_dir else self.config_dir
        config_path = config_dir / "combined_sizing.yaml"
        return self.load_yaml_config(config_path)
    
    def validate_config(self, config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration ("position", "leverage", or "combined")
            
        Returns:
            True if configuration is valid, False otherwise
        """
        if not config:
            self.logger.error(f"Empty {config_type} configuration")
            return False
        
        # Check for required sections
        if "risk_management" not in config:
            self.logger.error(
                f"Missing 'risk_management' section in {config_type} config"
            )
            return False
        
        risk_management = config["risk_management"]
        
        if config_type in ["position", "combined"]:
            if "position_sizing" not in risk_management:
                self.logger.error(
                    f"Missing 'position_sizing' section in {config_type} config"
                )
                return False
        
        if config_type in ["leverage", "combined"]:
            if "leverage_sizing" not in risk_management:
                self.logger.error(
                    f"Missing 'leverage_sizing' section in {config_type} config"
                )
                return False
        
        if "dynamic_risk_management" not in risk_management:
            self.logger.error(
                f"Missing 'dynamic_risk_management' section in {config_type} config"
            )
            return False
        
        if "liquidation_risk" not in risk_management:
            self.logger.error(
                f"Missing 'liquidation_risk' section in {config_type} config"
            )
            return False
        
        self.logger.info(f"✅ {config_type} configuration validation passed")
        return True
    
    def merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            configs: List of configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged_config = {}
        
        for config in configs:
            if config:
                self._deep_merge(merged_config, config)
        
        return merged_config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Recursively merge source dictionary into target dictionary.
        
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
    
    def load_config_with_fallback(
        self, 
        primary_config: str, 
        fallback_config: str, 
        config_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load configuration with fallback mechanism.
        
        Args:
            primary_config: Primary configuration filename
            fallback_config: Fallback configuration filename
            config_dir: Directory containing config files. Uses instance config_dir if None.
            
        Returns:
            Configuration dictionary from primary or fallback config
        """
        config_dir = Path(config_dir) if config_dir else self.config_dir
        
        # Try to load primary config
        primary_path = config_dir / primary_config
        config = self.load_yaml_config(primary_path)
        
        if config:
            self.logger.info(f"Loaded primary config: {primary_config}")
            return config
        
        # Try to load fallback config
        fallback_path = config_dir / fallback_config
        config = self.load_yaml_config(fallback_path)
        
        if config:
            self.logger.info(f"Loaded fallback config: {fallback_config}")
            return config
        
        self.logger.warning(
            f"No configuration found in {primary_config} or {fallback_config}"
        )
        return {}
    
    def create_default_configs(self, config_dir: Optional[str] = None) -> None:
        """
        Create default configuration files if they don't exist.
        
        Args:
            config_dir: Directory to create config files in. Uses instance config_dir if None.
        """
        config_dir = Path(config_dir) if config_dir else self.config_dir
        
        # Default position sizing config
        position_config = {
            "risk_management": {
                "position_sizing": {
                    "max_position_size": 0.02,  # 2% of portfolio
                    "position_scaling": 0.5,
                    "volatility_adjustment": True
                },
                "dynamic_risk_management": {
                    "enabled": True,
                    "max_drawdown": 0.15,
                    "correlation_threshold": 0.7
                },
                "liquidation_risk": {
                    "max_leverage": 3.0,
                    "margin_buffer": 0.1
                }
            }
        }
        
        # Default leverage sizing config
        leverage_config = {
            "risk_management": {
                "leverage_sizing": {
                    "base_leverage": 1.0,
                    "max_leverage": 5.0,
                    "volatility_scaling": True
                },
                "dynamic_risk_management": {
                    "enabled": True,
                    "max_drawdown": 0.15,
                    "correlation_threshold": 0.7
                },
                "liquidation_risk": {
                    "max_leverage": 3.0,
                    "margin_buffer": 0.1
                }
            }
        }
        
        # Create config files
        position_path = config_dir / "position_sizing.yaml"
        leverage_path = config_dir / "leverage_sizing.yaml"
        
        if not position_path.exists():
            with open(position_path, 'w') as f:
                yaml.dump(position_config, f, default_flow_style=False)
            self.logger.info(f"Created default position sizing config: {position_path}")
        
        if not leverage_path.exists():
            with open(leverage_path, 'w') as f:
                yaml.dump(leverage_config, f, default_flow_style=False)
            self.logger.info(f"Created default leverage sizing config: {leverage_path}")


# Convenience functions for module-level usage

def load_position_sizing_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load position sizing configuration.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Position sizing configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_position_sizing_config(config_dir)


def load_leverage_sizing_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load leverage sizing configuration.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Leverage sizing configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_leverage_sizing_config(config_dir)


def load_combined_sizing_config(config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load combined sizing configuration.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Combined sizing configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_combined_sizing_config(config_dir)


def load_config_with_fallback(
    primary_config: str, 
    fallback_config: str, 
    config_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to load configuration with fallback.
    
    Args:
        primary_config: Primary configuration filename
        fallback_config: Fallback configuration filename
        config_dir: Directory containing config files
        
    Returns:
        Configuration dictionary from primary or fallback config
    """
    loader = ConfigLoader()
    return loader.load_config_with_fallback(primary_config, fallback_config, config_dir)
