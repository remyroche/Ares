"""
Configuration Validator for ML Components.

This module provides validation for configuration parameters to prevent
runtime failures and ensure proper setup.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates configuration parameters for ML components.
    """
    
    def __init__(self):
        """Initialize configuration validator."""
        self.validation_rules = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default validation rules."""
        self.validation_rules = {
            'test_size': {
                'type': float,
                'min': 0.01,
                'max': 0.5,
                'description': 'Test set size as fraction of total data'
            },
            'validation_size': {
                'type': float,
                'min': 0.01,
                'max': 0.4,
                'description': 'Validation set size as fraction of training data'
            },
            'cv_folds': {
                'type': int,
                'min': 2,
                'max': 10,
                'description': 'Number of cross-validation folds'
            },
            'random_state': {
                'type': (int, type(None)),
                'description': 'Random state for reproducibility'
            },
            'gap_size': {
                'type': int,
                'min': 0,
                'max': 10,
                'description': 'Gap size between train and test sets'
            },
            'min_regime_samples': {
                'type': int,
                'min': 2,
                'max': 1000,
                'description': 'Minimum samples per regime'
            },
            'n_estimators': {
                'type': int,
                'min': 10,
                'max': 10000,
                'description': 'Number of estimators for ensemble methods'
            },
            'max_depth': {
                'type': (int, type(None)),
                'min': 1,
                'max': 50,
                'description': 'Maximum depth for tree-based models'
            },
            'learning_rate': {
                'type': float,
                'min': 0.001,
                'max': 1.0,
                'description': 'Learning rate for gradient-based methods'
            },
            'min_samples_split': {
                'type': int,
                'min': 2,
                'max': 1000,
                'description': 'Minimum samples required to split a node'
            },
            'min_samples_leaf': {
                'type': int,
                'min': 1,
                'max': 1000,
                'description': 'Minimum samples required in a leaf node'
            }
        }
    
    def validate_config(self, config: Dict[str, Any], 
                       required_params: Optional[List[str]] = None,
                       strict: bool = True) -> Dict[str, Any]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            required_params: List of required parameters
            strict: If True, raise errors for invalid values. If False, log warnings.
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ValueError: If validation fails and strict=True
        """
        logger.info("Starting configuration validation")
        
        # Check required parameters
        if required_params:
            missing_params = [param for param in required_params if param not in config]
            if missing_params:
                error_msg = f"Missing required parameters: {missing_params}"
                if strict:
                    raise ValueError(error_msg)
                else:
                    logger.warning(error_msg)
        
        validated_config = {}
        errors = []
        warnings = []
        
        # Validate each parameter
        for param, value in config.items():
            try:
                validated_value = self._validate_parameter(param, value)
                validated_config[param] = validated_value
                logger.debug(f"Validated {param}: {validated_value}")
            except ValueError as e:
                error_msg = f"Invalid value for {param}: {e}"
                if strict:
                    errors.append(error_msg)
                else:
                    warnings.append(error_msg)
                    validated_config[param] = value  # Keep original value
        
        # Log warnings if not strict
        if warnings:
            for warning in warnings:
                logger.warning(warning)
        
        # Raise errors if strict
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        # Validate parameter combinations
        self._validate_parameter_combinations(validated_config, strict)
        
        logger.info("Configuration validation completed successfully")
        return validated_config
    
    def _validate_parameter(self, param: str, value: Any) -> Any:
        """
        Validate a single parameter.
        
        Args:
            param: Parameter name
            value: Parameter value
            
        Returns:
            Validated value
            
        Raises:
            ValueError: If validation fails
        """
        if param not in self.validation_rules:
            logger.debug(f"No validation rules for parameter: {param}")
            return value
        
        rule = self.validation_rules[param]
        
        # Check type
        expected_type = rule['type']
        if isinstance(expected_type, tuple):
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected type {expected_type}, got {type(value)}")
        else:
            if not isinstance(value, expected_type):
                raise ValueError(f"Expected type {expected_type}, got {type(value)}")
        
        # Check range constraints
        if 'min' in rule and value < rule['min']:
            raise ValueError(f"Value {value} is below minimum {rule['min']}")
        
        if 'max' in rule and value > rule['max']:
            raise ValueError(f"Value {value} is above maximum {rule['max']}")
        
        # Check specific constraints
        if param == 'test_size' and 'validation_size' in self.validation_rules:
            # This will be checked in parameter combinations
            pass
        
        return value
    
    def _validate_parameter_combinations(self, config: Dict[str, Any], strict: bool):
        """
        Validate parameter combinations.
        
        Args:
            config: Configuration dictionary
            strict: If True, raise errors for invalid combinations
        """
        errors = []
        
        # Check that test_size + validation_size < 1
        if 'test_size' in config and 'validation_size' in config:
            total_size = config['test_size'] + config['validation_size']
            if total_size >= 1:
                error_msg = f"test_size ({config['test_size']}) + validation_size ({config['validation_size']}) must be < 1"
                errors.append(error_msg)
        
        # Check that cv_folds is reasonable for data size
        if 'cv_folds' in config and 'test_size' in config:
            # This is a heuristic - in practice, you'd need actual data size
            min_test_size = 1.0 / config['cv_folds']
            if config['test_size'] < min_test_size:
                warning_msg = f"test_size ({config['test_size']}) might be too small for {config['cv_folds']} CV folds"
                if strict:
                    errors.append(warning_msg)
                else:
                    logger.warning(warning_msg)
        
        if errors:
            if strict:
                raise ValueError(f"Parameter combination validation failed: {'; '.join(errors)}")
            else:
                for error in errors:
                    logger.warning(error)
    
    def get_default_config(self, component_type: str = 'regime_training') -> Dict[str, Any]:
        """
        Get default configuration for a component type.
        
        Args:
            component_type: Type of component
            
        Returns:
            Default configuration dictionary
        """
        if component_type == 'regime_training':
            return {
                'test_size': 0.3,
                'validation_size': 0.2,
                'cv_folds': 5,
                'random_state': 42,
                'gap_size': 1,
                'min_regime_samples': 10,
                'n_estimators': 100,
                'max_depth': None,
                'learning_rate': 0.1,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        else:
            return {}


def validate_regime_training_config(config: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """
    Validate regime training configuration.
    
    Args:
        config: Configuration dictionary
        strict: If True, raise errors for invalid values
        
    Returns:
        Validated configuration dictionary
    """
    validator = ConfigValidator()
    
    required_params = [
        'test_size',
        'cv_folds',
        'random_state'
    ]
    
    return validator.validate_config(config, required_params, strict)


def create_default_regime_training_config() -> Dict[str, Any]:
    """
    Create default regime training configuration.
    
    Returns:
        Default configuration dictionary
    """
    validator = ConfigValidator()
    return validator.get_default_config('regime_training')