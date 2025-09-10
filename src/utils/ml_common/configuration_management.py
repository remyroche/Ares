"""
Configuration Management Utilities

This module provides standardized configuration validation and management utilities
extracted from training steps to eliminate code duplication and provide consistent
configuration handling across all steps.

Key Features:
- StandardizedConfigValidator for unified configuration validation
- Step-specific validation rules and schemas
- Default value application and configuration fixing
- Comprehensive error reporting and logging
- Integration with ML Common utilities
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

# Import ML Common utilities
from src.utils.ml_common import (
    ConfigurationValidator,
    ValidationError
)

# Import common operations
from src.utils.common_operations import get_logger

logger = get_logger(__name__)


class StandardizedConfigValidator:
    """
    Standardized configuration validator for all training steps.
    
    This replaces custom configuration validation logic in individual steps
    with a unified approach using ConfigurationValidator from ml_common.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize standardized config validator."""
        self.logger = logger or get_logger(f"{__name__}.StandardizedConfigValidator")
        
        # Initialize ML Common ConfigurationValidator
        self.config_validator = ConfigurationValidator(self.logger)
        
        # Standard validation rules for all steps
        self.standard_rules = {
            'required_keys': ['symbol', 'exchange', 'timeframe'],
            'optional_keys': [
                'data_dir', 'output_dir', 'model_dir', 'log_dir',
                'enable_gpu', 'enable_parallel', 'max_workers',
                'memory_limit', 'timeout_seconds', 'random_state'
            ],
            'valid_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'],
            'valid_exchanges': ['binance', 'coinbase', 'kraken', 'bitfinex'],
            'default_values': {
                'data_dir': 'data',
                'output_dir': 'output',
                'model_dir': 'models',
                'log_dir': 'logs',
                'enable_gpu': True,
                'enable_parallel': True,
                'max_workers': 4,
                'memory_limit': 0.8,
                'timeout_seconds': 3600,
                'random_state': 42
            }
        }
        
        # Step-specific validation rules
        self.step_specific_rules = {
            'data_collection': {
                'required_keys': ['symbol', 'exchange', 'timeframe', 'data_dir'],
                'optional_keys': ['start_date', 'end_date', 'intervals']
            },
            'feature_engineering': {
                'required_keys': ['symbol', 'exchange', 'timeframe'],
                'optional_keys': ['feature_config', 'enable_advanced_features']
            },
            'model_training': {
                'required_keys': ['symbol', 'exchange', 'timeframe', 'model_config'],
                'optional_keys': ['train_test_split', 'validation_split', 'hyperparameters']
            },
            'feature_selection': {
                'required_keys': ['symbol', 'exchange', 'timeframe'],
                'optional_keys': ['selection_method', 'n_features', 'selection_threshold']
            },
            'model_evaluation': {
                'required_keys': ['symbol', 'exchange', 'timeframe'],
                'optional_keys': ['evaluation_config', 'metrics', 'validation_method']
            },
            'optimization': {
                'required_keys': ['symbol', 'exchange', 'timeframe'],
                'optional_keys': ['optimization_config', 'optimization_type', 'target_metric']
            }
        }
        
        self.logger.info("🚀 Standardized Config Validator initialized")
    
    def validate_standard_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration using standard rules.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: For critical configuration issues
        """
        try:
            self.logger.info("🔍 Validating standard configuration...")
            
            # Use ML Common ConfigurationValidator for basic validation
            basic_validation = self.config_validator.validate_ml_config(config)
            
            if not basic_validation['passed']:
                raise ValidationError(
                    f"Basic configuration validation failed: {basic_validation.get('errors', [])}",
                    "configuration",
                    basic_validation
                )
            
            # Apply standard validation rules
            validation_result = {
                'passed': True,
                'errors': [],
                'warnings': [],
                'validated_config': config.copy()
            }
            
            # Check required keys
            missing_keys = [key for key in self.standard_rules['required_keys'] if key not in config]
            if missing_keys:
                error_msg = f"Missing required configuration keys: {missing_keys}"
                validation_result['errors'].append(error_msg)
                validation_result['passed'] = False
            
            # Validate symbol format
            symbol = config.get('symbol', '')
            if not isinstance(symbol, str) or len(symbol) < 2:
                error_msg = f"Invalid symbol format: {symbol}"
                validation_result['errors'].append(error_msg)
                validation_result['passed'] = False
            
            # Validate exchange
            exchange = config.get('exchange', '')
            if exchange not in self.standard_rules['valid_exchanges']:
                warning_msg = f"Exchange '{exchange}' not in standard list: {self.standard_rules['valid_exchanges']}"
                validation_result['warnings'].append(warning_msg)
            
            # Validate timeframe
            timeframe = config.get('timeframe', '')
            if timeframe not in self.standard_rules['valid_timeframes']:
                error_msg = f"Invalid timeframe '{timeframe}'. Must be one of: {self.standard_rules['valid_timeframes']}"
                validation_result['errors'].append(error_msg)
                validation_result['passed'] = False
            
            # Apply default values
            for key, default_value in self.standard_rules['default_values'].items():
                if key not in config:
                    validation_result['validated_config'][key] = default_value
                    self.logger.info(f"Applied default value for '{key}': {default_value}")
            
            # Validate numeric values
            numeric_validations = [
                ('max_workers', int, 1, 32),
                ('memory_limit', float, 0.1, 1.0),
                ('timeout_seconds', int, 60, 86400),
                ('random_state', int, 0, 2**31-1)
            ]
            
            for key, value_type, min_val, max_val in numeric_validations:
                if key in config:
                    try:
                        converted_value = value_type(config[key])
                        if not (min_val <= converted_value <= max_val):
                            error_msg = f"Value for '{key}' must be between {min_val} and {max_val}, got {converted_value}"
                            validation_result['errors'].append(error_msg)
                            validation_result['passed'] = False
                        else:
                            validation_result['validated_config'][key] = converted_value
                    except (ValueError, TypeError):
                        error_msg = f"Invalid value type for '{key}': {config[key]}"
                        validation_result['errors'].append(error_msg)
                        validation_result['passed'] = False
            
            # Validate boolean values
            boolean_keys = ['enable_gpu', 'enable_parallel']
            for key in boolean_keys:
                if key in config:
                    if not isinstance(config[key], bool):
                        try:
                            validation_result['validated_config'][key] = bool(config[key])
                            self.logger.info(f"Converted '{key}' to boolean: {validation_result['validated_config'][key]}")
                        except (ValueError, TypeError):
                            error_msg = f"Invalid boolean value for '{key}': {config[key]}"
                            validation_result['errors'].append(error_msg)
                            validation_result['passed'] = False
            
            if validation_result['passed']:
                self.logger.info("✅ Standard configuration validation passed")
            else:
                self.logger.error(f"❌ Standard configuration validation failed: {validation_result['errors']}")
            
            return validation_result
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.exception(f"Standard configuration validation error: {e}")
            raise ValidationError(f"Configuration validation failed: {e}", "configuration", {"error": str(e)})
    
    def validate_step_config(self, config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
        """
        Validate configuration for a specific step.
        
        Args:
            config: Configuration dictionary to validate
            step_name: Name of the step
            
        Returns:
            Validation result dictionary
            
        Raises:
            ValidationError: For critical configuration issues
        """
        try:
            self.logger.info(f"🔍 Validating configuration for step: {step_name}")
            
            # First validate standard configuration
            standard_validation = self.validate_standard_config(config)
            
            if not standard_validation['passed']:
                return standard_validation
            
            # Get step-specific rules
            step_rules = self.step_specific_rules.get(step_name, {})
            
            if not step_rules:
                self.logger.warning(f"No specific validation rules found for step '{step_name}'")
                return standard_validation
            
            # Apply step-specific validation
            step_validation = standard_validation.copy()
            
            # Check step-specific required keys
            step_required_keys = step_rules.get('required_keys', [])
            missing_step_keys = [key for key in step_required_keys if key not in config]
            
            if missing_step_keys:
                error_msg = f"Missing required keys for step '{step_name}': {missing_step_keys}"
                step_validation['errors'].append(error_msg)
                step_validation['passed'] = False
            
            # Validate step-specific configurations
            if step_name == 'data_collection':
                step_validation = self._validate_data_collection_config(config, step_validation)
            elif step_name == 'feature_engineering':
                step_validation = self._validate_feature_engineering_config(config, step_validation)
            elif step_name == 'model_training':
                step_validation = self._validate_model_training_config(config, step_validation)
            elif step_name == 'feature_selection':
                step_validation = self._validate_feature_selection_config(config, step_validation)
            elif step_name == 'model_evaluation':
                step_validation = self._validate_model_evaluation_config(config, step_validation)
            elif step_name == 'optimization':
                step_validation = self._validate_optimization_config(config, step_validation)
            
            if step_validation['passed']:
                self.logger.info(f"✅ Step configuration validation passed for '{step_name}'")
            else:
                self.logger.error(f"❌ Step configuration validation failed for '{step_name}': {step_validation['errors']}")
            
            return step_validation
            
        except Exception as e:
            self.logger.exception(f"Step configuration validation error for '{step_name}': {e}")
            raise ValidationError(f"Step configuration validation failed for '{step_name}': {e}", "configuration", {"step_name": step_name, "error": str(e)})
    
    def _validate_data_collection_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data collection specific configuration."""
        try:
            # Validate data directory
            data_dir = config.get('data_dir', 'data')
            if not isinstance(data_dir, str) or len(data_dir) == 0:
                validation_result['errors'].append(f"Invalid data directory: {data_dir}")
                validation_result['passed'] = False
            
            # Validate date ranges if provided
            if 'start_date' in config:
                try:
                    from datetime import datetime
                    datetime.strptime(config['start_date'], '%Y-%m-%d')
                except ValueError:
                    validation_result['errors'].append(f"Invalid start_date format: {config['start_date']}. Use YYYY-MM-DD")
                    validation_result['passed'] = False
            
            if 'end_date' in config:
                try:
                    from datetime import datetime
                    datetime.strptime(config['end_date'], '%Y-%m-%d')
                except ValueError:
                    validation_result['errors'].append(f"Invalid end_date format: {config['end_date']}. Use YYYY-MM-DD")
                    validation_result['passed'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Data collection config validation error: {e}")
            validation_result['errors'].append(f"Data collection config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def _validate_feature_engineering_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature engineering specific configuration."""
        try:
            # Validate feature configuration
            feature_config = config.get('feature_config', {})
            if not isinstance(feature_config, dict):
                validation_result['errors'].append(f"Feature config must be a dictionary, got: {type(feature_config)}")
                validation_result['passed'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Feature engineering config validation error: {e}")
            validation_result['errors'].append(f"Feature engineering config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def _validate_model_training_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model training specific configuration."""
        try:
            # Validate model configuration
            model_config = config.get('model_config', {})
            if not isinstance(model_config, dict):
                validation_result['errors'].append(f"Model config must be a dictionary, got: {type(model_config)}")
                validation_result['passed'] = False
            
            # Validate train/test split
            train_test_split = config.get('train_test_split', 0.8)
            if not isinstance(train_test_split, (int, float)) or not (0 < train_test_split < 1):
                validation_result['errors'].append(f"Train test split must be between 0 and 1, got: {train_test_split}")
                validation_result['passed'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Model training config validation error: {e}")
            validation_result['errors'].append(f"Model training config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def _validate_feature_selection_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature selection specific configuration."""
        try:
            # Validate selection method
            selection_method = config.get('selection_method', 'mrmr')
            valid_methods = ['mrmr', 'importance', 'rfe', 'correlation', 'mutual_info']
            if selection_method not in valid_methods:
                validation_result['warnings'].append(f"Selection method '{selection_method}' not in standard list: {valid_methods}")
            
            # Validate number of features
            n_features = config.get('n_features')
            if n_features is not None:
                if not isinstance(n_features, int) or n_features <= 0:
                    validation_result['errors'].append(f"Number of features must be a positive integer, got: {n_features}")
                    validation_result['passed'] = False
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Feature selection config validation error: {e}")
            validation_result['errors'].append(f"Feature selection config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def _validate_model_evaluation_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model evaluation specific configuration."""
        try:
            # Validate evaluation configuration
            evaluation_config = config.get('evaluation_config', {})
            if not isinstance(evaluation_config, dict):
                validation_result['errors'].append(f"Evaluation config must be a dictionary, got: {type(evaluation_config)}")
                validation_result['passed'] = False
            
            # Validate evaluation type
            evaluation_type = config.get('evaluation_type', 'comprehensive')
            valid_types = ['basic', 'standard', 'comprehensive']
            if evaluation_type not in valid_types:
                validation_result['warnings'].append(f"Evaluation type '{evaluation_type}' not in standard list: {valid_types}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Model evaluation config validation error: {e}")
            validation_result['errors'].append(f"Model evaluation config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def _validate_optimization_config(self, config: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate optimization specific configuration."""
        try:
            # Validate optimization configuration
            optimization_config = config.get('optimization_config', {})
            if not isinstance(optimization_config, dict):
                validation_result['errors'].append(f"Optimization config must be a dictionary, got: {type(optimization_config)}")
                validation_result['passed'] = False
            
            # Validate optimization type
            optimization_type = config.get('optimization_type', 'comprehensive')
            valid_types = ['basic', 'standard', 'comprehensive']
            if optimization_type not in valid_types:
                validation_result['warnings'].append(f"Optimization type '{optimization_type}' not in standard list: {valid_types}")
            
            return validation_result
            
        except Exception as e:
            self.logger.exception(f"Optimization config validation error: {e}")
            validation_result['errors'].append(f"Optimization config validation error: {e}")
            validation_result['passed'] = False
            return validation_result
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """Get a summary of validation results."""
        if validation_result['passed']:
            summary = "✅ Configuration validation passed"
            if validation_result.get('warnings'):
                summary += f" with {len(validation_result['warnings'])} warnings"
        else:
            summary = f"❌ Configuration validation failed with {len(validation_result['errors'])} errors"
        
        return summary


# Global instance for easy access
_global_validator = None

def get_standardized_validator() -> StandardizedConfigValidator:
    """Get the global standardized config validator instance."""
    global _global_validator
    if _global_validator is None:
        _global_validator = StandardizedConfigValidator()
    return _global_validator


# Convenience functions
def validate_config(config: Dict[str, Any], step_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate configuration using standardized validator.
    
    Args:
        config: Configuration dictionary to validate
        step_name: Optional step name for step-specific validation
        
    Returns:
        Validation result dictionary
    """
    validator = get_standardized_validator()
    
    if step_name:
        return validator.validate_step_config(config, step_name)
    else:
        return validator.validate_standard_config(config)


def validate_and_fix_config(config: Dict[str, Any], step_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate configuration and return fixed version with defaults applied.
    
    Args:
        config: Configuration dictionary to validate
        step_name: Optional step name for step-specific validation
        
    Returns:
        Fixed configuration dictionary
    """
    validation_result = validate_config(config, step_name)
    return validation_result['validated_config']


# Example usage
if __name__ == "__main__":
    # Example configurations
    test_configs = [
        {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1m'
        },
        {
            'symbol': 'ETHUSDT',
            'exchange': 'coinbase',
            'timeframe': '5m',
            'data_dir': 'custom_data',
            'enable_gpu': True
        },
        {
            'symbol': 'INVALID',
            'timeframe': 'invalid_timeframe'
        }
    ]
    
    validator = StandardizedConfigValidator()
    
    for i, config in enumerate(test_configs):
        print(f"\n--- Test Config {i+1} ---")
        print(f"Input: {config}")
        
        try:
            result = validator.validate_standard_config(config)
            print(f"Result: {validator.get_validation_summary(result)}")
            print(f"Fixed config: {result['validated_config']}")
            
            if result.get('warnings'):
                print(f"Warnings: {result['warnings']}")
            
        except ValidationError as e:
            print(f"Validation Error: {e}")