"""
Configuration Migration Helper

This module provides utilities to help migrate existing NAS/TAS configurations
to use the unified configuration system.
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict

from ..config.base_config import (
    UnifiedArchitectureConfig, 
    create_quick_config, 
    create_comprehensive_config,
    create_regime_aware_config,
    create_real_time_config,
    ArchitectureType,
    OptimizationMode,
    SearchStrategy,
    ValidationMethod
)


class ConfigMigrationHelper:
    """Helper class for migrating configurations to unified system."""
    
    @staticmethod
    def migrate_nas_config_to_unified(
        nas_config: Dict[str, Any],
        preserve_custom: bool = True
    ) -> UnifiedArchitectureConfig:
        """
        Migrate NAS-specific configuration to unified configuration.
        
        Args:
            nas_config: Original NAS configuration dictionary
            preserve_custom: Whether to preserve custom parameters
            
        Returns:
            UnifiedArchitectureConfig instance
        """
        # Start with comprehensive config as base
        unified_config = create_comprehensive_config()
        
        # Map common NAS parameters
        mapping = {
            'n_regimes': 'n_regimes',
            'population_size': 'population_size',
            'generations': 'generations',
            'max_search_iterations': 'max_search_iterations',
            'max_search_time_seconds': 'max_search_time_seconds',
            'early_stopping_patience': 'early_stopping_patience',
            'validation_split': 'validation_split',
            'test_size': 'test_size',
            'cv_folds': 'cv_folds',
            'random_state': 'random_state',
            'verbose': 'verbose',
            'enable_parallel_processing': 'enable_parallel_processing',
            'max_memory_usage_gb': 'max_memory_usage_gb',
            'enable_gpu_acceleration': 'enable_gpu_acceleration',
            'batch_size': 'batch_size',
            'save_intermediate_results': 'save_intermediate_results',
            'save_best_models': 'save_best_models',
            'output_dir': 'output_dir',
            'log_level': 'log_level',
            'enable_detailed_logging': 'enable_detailed_logging'
        }
        
        # Apply mapped parameters
        for old_key, new_key in mapping.items():
            if old_key in nas_config:
                setattr(unified_config, new_key, nas_config[old_key])
        
        # Set architecture type to neural
        unified_config.architecture_type = ArchitectureType.NEURAL_ONLY
        
        # Map optimization mode
        if 'optimization_mode' in nas_config:
            try:
                unified_config.optimization_mode = OptimizationMode(nas_config['optimization_mode'])
            except ValueError:
                unified_config.optimization_mode = OptimizationMode.REGIME_AWARE
        
        # Map search strategy
        if 'search_strategy' in nas_config:
            try:
                unified_config.search_strategy = SearchStrategy(nas_config['search_strategy'])
            except ValueError:
                unified_config.search_strategy = SearchStrategy.EVOLUTIONARY
        
        # Map validation method
        if 'validation_method' in nas_config:
            try:
                unified_config.validation_method = ValidationMethod(nas_config['validation_method'])
            except ValueError:
                unified_config.validation_method = ValidationMethod.TIME_SERIES_SPLIT
        
        # Preserve custom parameters
        if preserve_custom:
            custom_params = {}
            for key, value in nas_config.items():
                if key not in mapping and key not in ['optimization_mode', 'search_strategy', 'validation_method']:
                    custom_params[key] = value
            
            if custom_params:
                unified_config.custom_parameters.update(custom_params)
        
        return unified_config
    
    @staticmethod
    def migrate_tas_config_to_unified(
        tas_config: Dict[str, Any],
        preserve_custom: bool = True
    ) -> UnifiedArchitectureConfig:
        """
        Migrate TAS-specific configuration to unified configuration.
        
        Args:
            tas_config: Original TAS configuration dictionary
            preserve_custom: Whether to preserve custom parameters
            
        Returns:
            UnifiedArchitectureConfig instance
        """
        # Start with regime-aware config as base for TAS
        unified_config = create_regime_aware_config()
        
        # Map common TAS parameters
        mapping = {
            'n_regimes': 'n_regimes',
            'regime_stability_threshold': 'regime_stability_threshold',
            'data_driven_regimes': 'data_driven_regimes',
            'min_regime_duration': 'min_regime_duration',
            'max_regime_duration': 'max_regime_duration',
            'population_size': 'population_size',
            'generations': 'generations',
            'max_search_iterations': 'max_search_iterations',
            'max_search_time_seconds': 'max_search_time_seconds',
            'early_stopping_patience': 'early_stopping_patience',
            'validation_split': 'validation_split',
            'test_size': 'test_size',
            'cv_folds': 'cv_folds',
            'random_state': 'random_state',
            'verbose': 'verbose',
            'enable_parallel_processing': 'enable_parallel_processing',
            'max_memory_usage_gb': 'max_memory_usage_gb',
            'batch_size': 'batch_size',
            'save_intermediate_results': 'save_intermediate_results',
            'save_best_models': 'save_best_models',
            'output_dir': 'output_dir',
            'log_level': 'log_level',
            'enable_detailed_logging': 'enable_detailed_logging'
        }
        
        # Apply mapped parameters
        for old_key, new_key in mapping.items():
            if old_key in tas_config:
                setattr(unified_config, new_key, tas_config[old_key])
        
        # Set architecture type to tree
        unified_config.architecture_type = ArchitectureType.TREE_ONLY
        
        # Map optimization mode
        if 'optimization_mode' in tas_config:
            try:
                unified_config.optimization_mode = OptimizationMode(tas_config['optimization_mode'])
            except ValueError:
                unified_config.optimization_mode = OptimizationMode.REGIME_AWARE
        
        # Map search strategy
        if 'search_strategy' in tas_config:
            try:
                unified_config.search_strategy = SearchStrategy(tas_config['search_strategy'])
            except ValueError:
                unified_config.search_strategy = SearchStrategy.EVOLUTIONARY
        
        # Map validation method
        if 'validation_method' in tas_config:
            try:
                unified_config.validation_method = ValidationMethod(tas_config['validation_method'])
            except ValueError:
                unified_config.validation_method = ValidationMethod.TIME_SERIES_SPLIT
        
        # Preserve custom parameters
        if preserve_custom:
            custom_params = {}
            for key, value in tas_config.items():
                if key not in mapping and key not in ['optimization_mode', 'search_strategy', 'validation_method']:
                    custom_params[key] = value
            
            if custom_params:
                unified_config.custom_parameters.update(custom_params)
        
        return unified_config
    
    @staticmethod
    def migrate_hybrid_config_to_unified(
        hybrid_config: Dict[str, Any],
        preserve_custom: bool = True
    ) -> UnifiedArchitectureConfig:
        """
        Migrate hybrid NAS/TAS configuration to unified configuration.
        
        Args:
            hybrid_config: Original hybrid configuration dictionary
            preserve_custom: Whether to preserve custom parameters
            
        Returns:
            UnifiedArchitectureConfig instance
        """
        # Start with comprehensive config as base for hybrid
        unified_config = create_comprehensive_config()
        
        # Map common hybrid parameters
        mapping = {
            'n_regimes': 'n_regimes',
            'regime_stability_threshold': 'regime_stability_threshold',
            'data_driven_regimes': 'data_driven_regimes',
            'min_regime_duration': 'min_regime_duration',
            'max_regime_duration': 'max_regime_duration',
            'population_size': 'population_size',
            'generations': 'generations',
            'max_search_iterations': 'max_search_iterations',
            'max_search_time_seconds': 'max_search_time_seconds',
            'early_stopping_patience': 'early_stopping_patience',
            'validation_split': 'validation_split',
            'test_size': 'test_size',
            'cv_folds': 'cv_folds',
            'random_state': 'random_state',
            'verbose': 'verbose',
            'enable_parallel_processing': 'enable_parallel_processing',
            'max_memory_usage_gb': 'max_memory_usage_gb',
            'enable_gpu_acceleration': 'enable_gpu_acceleration',
            'batch_size': 'batch_size',
            'save_intermediate_results': 'save_intermediate_results',
            'save_best_models': 'save_best_models',
            'output_dir': 'output_dir',
            'log_level': 'log_level',
            'enable_detailed_logging': 'enable_detailed_logging'
        }
        
        # Apply mapped parameters
        for old_key, new_key in mapping.items():
            if old_key in hybrid_config:
                setattr(unified_config, new_key, hybrid_config[old_key])
        
        # Set architecture type to hybrid
        unified_config.architecture_type = ArchitectureType.HYBRID_NEURAL_TREE
        
        # Map optimization mode
        if 'optimization_mode' in hybrid_config:
            try:
                unified_config.optimization_mode = OptimizationMode(hybrid_config['optimization_mode'])
            except ValueError:
                unified_config.optimization_mode = OptimizationMode.REGIME_AWARE
        
        # Map search strategy
        if 'search_strategy' in hybrid_config:
            try:
                unified_config.search_strategy = SearchStrategy(hybrid_config['search_strategy'])
            except ValueError:
                unified_config.search_strategy = SearchStrategy.HYBRID
        
        # Map validation method
        if 'validation_method' in hybrid_config:
            try:
                unified_config.validation_method = ValidationMethod(hybrid_config['validation_method'])
            except ValueError:
                unified_config.validation_method = ValidationMethod.TIME_SERIES_SPLIT
        
        # Preserve custom parameters
        if preserve_custom:
            custom_params = {}
            for key, value in hybrid_config.items():
                if key not in mapping and key not in ['optimization_mode', 'search_strategy', 'validation_method']:
                    custom_params[key] = value
            
            if custom_params:
                unified_config.custom_parameters.update(custom_params)
        
        return unified_config
    
    @staticmethod
    def create_config_from_preset(
        preset_name: str,
        custom_overrides: Optional[Dict[str, Any]] = None
    ) -> UnifiedArchitectureConfig:
        """
        Create unified configuration from preset.
        
        Args:
            preset_name: Name of preset ('quick', 'comprehensive', 'regime_aware', 'real_time')
            custom_overrides: Custom parameter overrides
            
        Returns:
            UnifiedArchitectureConfig instance
        """
        presets = {
            'quick': create_quick_config,
            'comprehensive': create_comprehensive_config,
            'regime_aware': create_regime_aware_config,
            'real_time': create_real_time_config
        }
        
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        config = presets[preset_name]()
        
        if custom_overrides:
            config = config.update(custom_overrides)
        
        return config
    
    @staticmethod
    def save_config_to_file(
        config: UnifiedArchitectureConfig,
        filepath: str,
        format: str = 'json'
    ) -> bool:
        """
        Save unified configuration to file.
        
        Args:
            config: UnifiedArchitectureConfig instance
            filepath: Path to save file
            format: File format ('json', 'yaml')
            
        Returns:
            True if successful
        """
        try:
            if format == 'json':
                with open(filepath, 'w') as f:
                    json.dump(config.to_dict(), f, indent=2, default=str)
            elif format == 'yaml':
                import yaml
                with open(filepath, 'w') as f:
                    yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Failed to save config: {e}")
            return False
    
    @staticmethod
    def load_config_from_file(
        filepath: str,
        format: str = 'json'
    ) -> Optional[UnifiedArchitectureConfig]:
        """
        Load unified configuration from file.
        
        Args:
            filepath: Path to config file
            format: File format ('json', 'yaml')
            
        Returns:
            UnifiedArchitectureConfig instance or None if failed
        """
        try:
            if format == 'json':
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
            elif format == 'yaml':
                import yaml
                with open(filepath, 'r') as f:
                    config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return UnifiedArchitectureConfig.from_dict(config_dict)
        except Exception as e:
            print(f"Failed to load config: {e}")
            return None
    
    @staticmethod
    def compare_configs(
        config1: UnifiedArchitectureConfig,
        config2: UnifiedArchitectureConfig
    ) -> Dict[str, Any]:
        """
        Compare two unified configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary with comparison results
        """
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        comparison = {
            'identical': dict1 == dict2,
            'differences': {},
            'only_in_config1': {},
            'only_in_config2': {}
        }
        
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            if key not in dict1:
                comparison['only_in_config2'][key] = dict2[key]
            elif key not in dict2:
                comparison['only_in_config1'][key] = dict1[key]
            elif dict1[key] != dict2[key]:
                comparison['differences'][key] = {
                    'config1': dict1[key],
                    'config2': dict2[key]
                }
        
        return comparison


def migrate_config_file(
    input_file: str,
    output_file: str,
    config_type: str = 'auto',
    preserve_custom: bool = True
) -> bool:
    """
    Migrate a configuration file to unified format.
    
    Args:
        input_file: Path to input configuration file
        output_file: Path to output configuration file
        config_type: Type of config ('nas', 'tas', 'hybrid', 'auto')
        preserve_custom: Whether to preserve custom parameters
        
    Returns:
        True if successful
    """
    try:
        # Load original config
        with open(input_file, 'r') as f:
            if input_file.endswith('.json'):
                original_config = json.load(f)
            elif input_file.endswith('.yaml') or input_file.endswith('.yml'):
                import yaml
                original_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {input_file}")
        
        # Determine config type if auto
        if config_type == 'auto':
            if 'neural' in str(original_config).lower() or 'nas' in str(original_config).lower():
                config_type = 'nas'
            elif 'tree' in str(original_config).lower() or 'tas' in str(original_config).lower():
                config_type = 'tas'
            elif 'hybrid' in str(original_config).lower():
                config_type = 'hybrid'
            else:
                config_type = 'nas'  # Default fallback
        
        # Migrate config
        if config_type == 'nas':
            unified_config = ConfigMigrationHelper.migrate_nas_config_to_unified(
                original_config, preserve_custom
            )
        elif config_type == 'tas':
            unified_config = ConfigMigrationHelper.migrate_tas_config_to_unified(
                original_config, preserve_custom
            )
        elif config_type == 'hybrid':
            unified_config = ConfigMigrationHelper.migrate_hybrid_config_to_unified(
                original_config, preserve_custom
            )
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        
        # Save migrated config
        format = 'json' if output_file.endswith('.json') else 'yaml'
        return ConfigMigrationHelper.save_config_to_file(unified_config, output_file, format)
        
    except Exception as e:
        print(f"Migration failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("Configuration Migration Helper")
    print("Available functions:")
    print("- migrate_config_file()")
    print("- ConfigMigrationHelper.migrate_nas_config_to_unified()")
    print("- ConfigMigrationHelper.migrate_tas_config_to_unified()")
    print("- ConfigMigrationHelper.migrate_hybrid_config_to_unified()")
    print("- ConfigMigrationHelper.create_config_from_preset()")