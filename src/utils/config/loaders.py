"""
Parameter Loader Utilities

This module provides comprehensive parameter loading and management utilities
to replace the functionality from sr_parameter_loader.py.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParameterSet:
    """Represents a set of parameters with metadata."""
    name: str
    parameters: Dict[str, Any]
    source: str
    timestamp: float
    version: Optional[str] = None
    description: Optional[str] = None

class ParameterLoader:
    """Generic parameter loader for various parameter types."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize parameter loader.

        Args:
            config: Configuration for parameter loading
        """
        self.config = config or {}
        self.parameter_cache: Dict[str, ParameterSet] = {}
        self.default_paths = {
            'models': 'models',
            'config': 'config',
            'parameters': 'parameters'
        }

    def load_parameters(self, parameter_name: str,
                       file_path: Optional[Union[str, Path]] = None,
                       config_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Load parameters from various sources.

        Args:
            parameter_name: Name of the parameter set
            file_path: Specific file path to load from
            config_key: Key in config to look for parameters
            default: Default parameters if loading fails

        Returns:
            Dictionary containing parameters
        """
        # Check cache first
        if parameter_name in self.parameter_cache:
            logger.debug(f"Using cached parameters for {parameter_name}")
            return self.parameter_cache[parameter_name].parameters

        # Try to load from config first
        if config_key and config_key in self.config:
            logger.info(f"✅ Using parameters from config key: {config_key}")
            parameters = self.config[config_key]
            self._cache_parameters(parameter_name, parameters, "config")
            return parameters

        # Try to load from file
        if file_path:
            parameters = self._load_from_file(file_path)
            if parameters:
                self._cache_parameters(parameter_name, parameters, str(file_path))
                return parameters

        # Try default file paths
        for path_type, base_path in self.default_paths.items():
            default_file = Path(base_path) / f"{parameter_name}_parameters.json"
            if default_file.exists():
                parameters = self._load_from_file(default_file)
                if parameters:
                    self._cache_parameters(parameter_name, parameters, str(default_file))
                    return parameters

        logger.warning(f"No parameters found for {parameter_name}")
        return {}

    def _load_from_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Load parameters from file."""
        try:
            file_path = Path(file_path)

            if not file_path.exists():
                logger.debug(f"Parameter file not found: {file_path}")
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle different file formats
            if isinstance(data, dict):
                if 'parameters' in data:
                    parameters = data['parameters']
                elif 'config' in data:
                    parameters = data['config']
                else:
                    parameters = data
            else:
                logger.error(f"Invalid parameter file format: {file_path}")
                return None

            logger.debug(f"Successfully loaded parameters from {file_path}")
            return parameters

        except Exception as e:
            logger.error(f"Failed to load parameters from {file_path}: {e}")
            return None

    def _cache_parameters(self, name: str, parameters: Dict[str, Any], source: str) -> None:
        """Cache parameters."""
        import time
        self.parameter_cache[name] = ParameterSet(
            name=name,
            parameters=parameters,
            source=source,
            timestamp=time.time()
        )

    def save_parameters(self, parameters: Dict[str, Any],
                       file_path: Union[str, Path],
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Save parameters to file.

        Args:
            parameters: Parameters to save
            file_path: Path to save the file
            metadata: Additional metadata to save

        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                'parameters': parameters,
                'metadata': metadata or {},
                'timestamp': time.time()
            }

            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=2)

            logger.info(f"Successfully saved parameters to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save parameters to {file_path}: {e}")
            return False

    def get_parameter_summary(self, parameter_name: str) -> Dict[str, Any]:
        """Get summary of loaded parameters."""
        if parameter_name not in self.parameter_cache:
            return {}

        param_set = self.parameter_cache[parameter_name]
        return {
            'name': param_set.name,
            'source': param_set.source,
            'timestamp': param_set.timestamp,
            'parameter_count': len(param_set.parameters),
            'parameter_keys': list(param_set.parameters.keys())
        }

    def clear_cache(self) -> None:
        """Clear parameter cache."""
        self.parameter_cache.clear()
        logger.debug("Parameter cache cleared")

class SRParameterLoader(ParameterLoader):
    """Specialized parameter loader for S/R parameters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize S/R parameter loader."""
        super().__init__(config)
        self.sr_config_keys = [
            'sr_probability_calculation',
            'sr_breakout_predictor',
            'sr_strength_parameters'
        ]

    def load_optimized_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load optimized S/R parameters from step 2.5.

        Args:
            config: System configuration

        Returns:
            Dictionary containing optimized S/R parameters
        """
        logger = logging.getLogger("SRParameterLoader")

        try:
            # Check if parameters are already in config (from training)
            for key in self.sr_config_keys:
                if key in config and config[key]:
                    logger.info(f"✅ Using S/R parameters from config key: {key}")
                    return config[key]

            # Try to load from file
            param_file = Path(config.get("model_save_path", "models")) / "optimized_sr_strength_parameters.json"

            if param_file.exists():
                parameters = self._load_from_file(param_file)
                if parameters:
                    # Update config with loaded parameters
                    config["sr_probability_calculation"] = parameters
                    config.setdefault("sr_breakout_predictor", {})["optimized_parameters"] = parameters
                    config["sr_breakout_predictor"]["use_optimized_params"] = True

                    logger.info(f"✅ Loaded S/R parameters from {param_file}")
                    return parameters

            # Try alternative file locations
            alternative_paths = [
                Path("models") / "sr_parameters.json",
                Path("config") / "sr_parameters.json",
                Path("parameters") / "sr_parameters.json"
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    parameters = self._load_from_file(alt_path)
                    if parameters:
                        logger.info(f"✅ Loaded S/R parameters from {alt_path}")
                        return parameters

            logger.warning("⚠️ No optimized S/R parameters found, using defaults")
            return self._get_default_sr_parameters()

        except Exception as e:
            logger.error(f"❌ Error loading S/R parameters: {e}")
            return self._get_default_sr_parameters()

    def _get_default_sr_parameters(self) -> Dict[str, Any]:
        """Get default S/R parameters."""
        return {
            'strength_threshold': 0.5,
            'probability_threshold': 0.6,
            'breakout_threshold': 0.7,
            'confluence_weight': 0.8,
            'timeframe_weight': 0.6,
            'volume_weight': 0.4
        }

    def ensure_parameters_loaded(self, config: Dict[str, Any]) -> None:
        """Ensure S/R parameters are loaded in config."""
        if 'sr_probability_calculation' not in config:
            config['sr_probability_calculation'] = self.load_optimized_parameters(config)

    def get_parameter_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get summary of S/R parameters."""
        parameters = self.load_optimized_parameters(config)
        return {
            'parameter_count': len(parameters),
            'parameter_keys': list(parameters.keys()),
            'has_optimized_params': 'sr_probability_calculation' in config,
            'parameters': parameters
        }

# Convenience functions
def initialize_sr_parameters(config: Dict[str, Any]) -> None:
    """Initialize S/R parameters in config."""
    loader = SRParameterLoader()
    loader.ensure_parameters_loaded(config)

def load_sr_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load S/R parameters."""
    loader = SRParameterLoader()
    return loader.load_optimized_parameters(config)

def load_parameters(parameter_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load generic parameters."""
    loader = ParameterLoader(config)
    return loader.load_parameters(parameter_name)

# Global parameter loader instance
global_parameter_loader = ParameterLoader()

__all__ = [
    'ParameterSet',
    'ParameterLoader',
    'SRParameterLoader',
    'initialize_sr_parameters',
    'load_sr_parameters',
    'load_parameters',
    'global_parameter_loader'
]
