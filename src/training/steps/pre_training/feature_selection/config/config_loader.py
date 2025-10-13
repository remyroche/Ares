"""
Configuration loading utilities for feature selection.

This module provides configuration loading and management capabilities
for the feature selection system, including YAML file loading and
environment variable overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


@dataclass
class ConfigLoadResult:
    """Result of configuration loading operation."""
    config: Dict[str, Any]
    source_file: Optional[str]
    success: bool
    error_message: Optional[str] = None
    warnings: List[str] = None


class ConfigLoader:
    """Configuration loader for feature selection system."""
    
    def __init__(self, config_root: Optional[str] = None):
        self.config_root = config_root or self._get_default_config_root()
        self.logger = get_logger("ConfigLoader")
        
        # Configuration file paths
        self.config_files = {
            'feature_selection': 'feature_selection_config.yaml',
            'optimized_feature_selection': 'optimized_feature_selection_config.yaml',
            'wavelet_feature_selection': 'wavelet_feature_selection_config.yaml'
        }
    
    def _get_default_config_root(self) -> str:
        """Get default configuration root directory."""
        # Try environment variable first
        config_root_env = os.environ.get("ARES_CONFIG_ROOT")
        if config_root_env:
            return config_root_env
        
        # Default to config directory relative to this module
        return str(Path(__file__).resolve().parents[4] / "config")
    
    def load_config(self, config_name: str, model_type: str = 'default') -> ConfigLoadResult:
        """
        Load configuration for feature selection.
        
        Args:
            config_name: Name of the configuration to load
            model_type: Model type for model-specific configuration
            
        Returns:
            ConfigLoadResult with loaded configuration
        """
        tprint_info(f"📁 Loading configuration: {config_name} for model: {model_type}")
        
        try:
            # Get config file path
            config_file = self.config_files.get(config_name)
            if not config_file:
                return ConfigLoadResult(
                    config={},
                    source_file=None,
                    success=False,
                    error_message=f"Unknown configuration name: {config_name}"
                )
            
            config_path = Path(self.config_root) / config_file
            
            if not config_path.exists():
                return ConfigLoadResult(
                    config={},
                    source_file=str(config_path),
                    success=False,
                    error_message=f"Configuration file not found: {config_path}"
                )
            
            # Load YAML configuration
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if not config_data:
                return ConfigLoadResult(
                    config={},
                    source_file=str(config_path),
                    success=False,
                    error_message="Configuration file is empty or invalid"
                )
            
            # Extract feature selection configuration
            if 'feature_selection' in config_data:
                fs_config = config_data['feature_selection']
            else:
                fs_config = config_data
            
            # Apply model-specific configuration if available
            if model_type != 'default' and 'model_profiles' in fs_config:
                model_profiles = fs_config['model_profiles']
                if model_type in model_profiles:
                    model_config = model_profiles[model_type]
                    # Merge model-specific config with base config
                    merged_config = self._merge_configs(fs_config, model_config)
                    tprint_success(f"   ✅ Loaded model-specific config for {model_type}")
                else:
                    merged_config = fs_config
                    tprint_warning(f"   ⚠️ Model type {model_type} not found, using default config")
            else:
                merged_config = fs_config
            
            # Apply environment variable overrides
            merged_config = self._apply_env_overrides(merged_config)
            
            return ConfigLoadResult(
                config=merged_config,
                source_file=str(config_path),
                success=True
            )
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load configuration: {e}")
            return ConfigLoadResult(
                config={},
                source_file=None,
                success=False,
                error_message=str(e)
            )
    
    def _merge_configs(self, base_config: Dict[str, Any], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model-specific configuration with base configuration."""
        tprint_debug("🔧 Merging model-specific configuration")
        
        merged = base_config.copy()
        
        # Override base config with model-specific values
        for key, value in model_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides to configuration."""
        tprint_debug("🔧 Applying environment variable overrides")
        
        overrides = {}
        
        # Common environment variable mappings
        env_mappings = {
            'ARES_FEATURE_SELECTION_TARGET_FEATURES': 'target_features',
            'ARES_FEATURE_SELECTION_MIN_FEATURES': 'min_features',
            'ARES_FEATURE_SELECTION_MAX_FEATURES': 'max_features',
            'ARES_FEATURE_SELECTION_VIF_THRESHOLD': 'vif_threshold',
            'ARES_FEATURE_SELECTION_CORRELATION_THRESHOLD': 'correlation_threshold',
            'ARES_FEATURE_SELECTION_ENABLE_VECTORBT': 'enable_vectorbt_optimization',
            'ARES_FEATURE_SELECTION_MEMORY_EFFICIENT': 'vectorbt_memory_efficient',
            'ARES_FEATURE_SELECTION_CHUNK_SIZE': 'vectorbt_chunk_size'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                overrides[config_key] = converted_value
                tprint_debug(f"   🔧 Override: {config_key} = {converted_value}")
        
        # Apply overrides
        if overrides:
            config.update(overrides)
            tprint_success(f"   ✅ Applied {len(overrides)} environment overrides")
        
        return config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer values
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float values
        try:
            return float(value)
        except ValueError:
            pass
        
        # String values (default)
        return value
    
    def get_available_configs(self) -> List[str]:
        """Get list of available configuration names."""
        return list(self.config_files.keys())
    
    def get_config_file_path(self, config_name: str) -> Optional[str]:
        """Get file path for a configuration."""
        config_file = self.config_files.get(config_name)
        if config_file:
            return str(Path(self.config_root) / config_file)
        return None
    
    def validate_config_file(self, config_name: str) -> bool:
        """Validate that a configuration file exists and is readable."""
        config_path = self.get_config_file_path(config_name)
        if not config_path:
            return False
        
        try:
            with open(config_path, 'r') as f:
                yaml.safe_load(f)
            return True
        except Exception:
            return False
    
    def load_all_configs(self) -> Dict[str, ConfigLoadResult]:
        """Load all available configurations."""
        tprint_info("📁 Loading all available configurations")
        
        results = {}
        for config_name in self.get_available_configs():
            results[config_name] = self.load_config(config_name)
        
        successful_configs = sum(1 for result in results.values() if result.success)
        tprint_success(f"   ✅ Loaded {successful_configs}/{len(results)} configurations")
        
        return results
    
    def create_config_template(self, config_name: str, output_path: str):
        """Create a configuration template file."""
        tprint_info(f"📝 Creating configuration template: {config_name}")
        
        try:
            # Get template based on config name
            template = self._get_config_template(config_name)
            
            # Write template to file
            with open(output_path, 'w') as f:
                yaml.dump(template, f, default_flow_style=False, indent=2)
            
            tprint_success(f"   ✅ Template created at {output_path}")
            
        except Exception as e:
            tprint_warning(f"   ⚠️ Failed to create template: {e}")
    
    def _get_config_template(self, config_name: str) -> Dict[str, Any]:
        """Get configuration template based on config name."""
        if config_name == 'feature_selection':
            return {
                'feature_selection': {
                    'target_features': 80,
                    'min_features': 60,
                    'max_features': 100,
                    'vif_threshold': 10.0,
                    'correlation_threshold': 0.95,
                    'enable_vectorbt_optimization': True,
                    'vectorbt_memory_efficient': True,
                    'vectorbt_chunk_size': 1000,
                    'model_profiles': {
                        'default': {
                            'target_features': 80,
                            'min_features': 60,
                            'max_features': 100
                        }
                    }
                }
            }
        elif config_name == 'optimized_feature_selection':
            return {
                'feature_selection': {
                    'target_features': {
                        'neural_networks': 80,
                        'linear_models': 60,
                        'ensemble_models': 90
                    },
                    'vif_threshold': 10.0,
                    'correlation_threshold': 0.95,
                    'enable_vectorbt_optimization': True,
                    'performance': {
                        'enable_parallel_processing': True,
                        'max_workers': -1,
                        'chunk_size': 1000
                    }
                }
            }
        else:
            return {
                'feature_selection': {
                    'target_features': 80,
                    'min_features': 60,
                    'max_features': 100
                }
            }