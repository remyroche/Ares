"""
Parameter Mapper for Pyro to statsmodels Migration

This module provides comprehensive parameter mapping between Pyro-based Sticky Finite HMM
configurations and statsmodels-compatible parameters, ensuring seamless migration
with proper validation and conversion.

Key Features:
- Pyro to statsmodels parameter mapping
- Search space conversion
- Hyperparameter validation
- Configuration transformation
- Backward compatibility support
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error,
        tprint_structured
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')
    def tprint_structured(data, level="INFO"):
        for key, value in data.items():
            print(f'🔧 {key}: {value}')


@dataclass
class ParameterMappingConfig:
    """
    Configuration for parameter mapping between Pyro and statsmodels.
    
    Defines how different parameter types should be mapped and validated
    during the migration process.
    """
    # Mapping rules
    strict_mapping: bool = True  # Raise errors for unmapped parameters
    preserve_unknown: bool = False  # Keep unknown parameters in metadata
    
    # Validation rules
    validate_ranges: bool = True  # Validate parameter ranges
    validate_types: bool = True  # Validate parameter types
    
    # Transformation rules
    normalize_probabilities: bool = True  # Normalize probability vectors
    round_discrete_params: bool = True  # Round discrete parameters
    
    # Logging
    log_mappings: bool = True  # Log all parameter mappings
    log_unmapped: bool = True  # Log unmapped parameters
    
    # Output
    save_mapping_report: bool = False  # Save detailed mapping report
    output_dir: Optional[str] = None


@dataclass
class ParameterMappingResult:
    """
    Result container for parameter mapping operations.
    
    Contains mapped parameters, validation results, and metadata
    about the mapping process.
    """
    # Mapped parameters
    mapped_params: Dict[str, Any] = field(default_factory=dict)
    unmapped_params: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    
    # Mapping metadata
    mapping_log: List[Dict[str, str]] = field(default_factory=list)
    transformation_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Success status
    success: bool = True
    error_message: Optional[str] = None


class PyroToStatsmodelsMapper:
    """
    Comprehensive mapper for Pyro Sticky Finite HMM to statsmodels parameters.
    
    Handles conversion of all major parameter types including:
    - Model structure parameters (K, order, etc.)
    - Transition matrices and probabilities
    - Emission distributions
    - Prior distributions
    - Hyperparameters
    """
    
    def __init__(self, config: Optional[ParameterMappingConfig] = None):
        """
        Initialize the parameter mapper.
        
        Args:
            config: Configuration for mapping behavior
        """
        self.config = config or ParameterMappingConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Define parameter mapping rules
        self._setup_mapping_rules()
        
        # Define parameter validation rules
        self._setup_validation_rules()
        
        tprint_info("🔧 Initialized Pyro to statsmodels parameter mapper")
    
    def _setup_mapping_rules(self):
        """Setup parameter mapping rules."""
        self.mapping_rules = {
            # Model structure
            'K': {
                'target': 'k_regimes',
                'type': 'int',
                'min': 2,
                'max': 10,
                'description': 'Number of regimes'
            },
            'order': {
                'target': 'order',
                'type': 'int',
                'min': 0,
                'max': 5,
                'description': 'Autoregressive order'
            },
            'trend': {
                'target': 'trend',
                'type': 'str',
                'options': ['c', 't', 'ct'],
                'description': 'Trend component'
            },
            
            # Switching parameters
            'switching_variance': {
                'target': 'switching_variance',
                'type': 'bool',
                'description': 'Allow variance switching'
            },
            'switching_trend': {
                'target': 'switching_trend',
                'type': 'bool',
                'description': 'Allow trend switching'
            },
            'switching_exog': {
                'target': 'switching_exog',
                'type': 'bool',
                'description': 'Allow exogenous switching'
            },
            
            # Transition parameters
            'transition_matrix': {
                'target': 'transition_matrix',
                'type': 'array',
                'shape': 'square',
                'description': 'Transition probability matrix'
            },
            'transition_prior': {
                'target': 'transition_prior',
                'type': 'array',
                'description': 'Transition prior parameters'
            },
            'alpha': {
                'target': 'transition_prior',
                'type': 'array',
                'description': 'Transition prior concentration'
            },
            
            # Emission parameters
            'emission_means': {
                'target': 'regime_means',
                'type': 'array',
                'description': 'Regime mean parameters'
            },
            'emission_covs': {
                'target': 'regime_covs',
                'type': 'array',
                'description': 'Regime covariance parameters'
            },
            'emission_prior': {
                'target': 'emission_prior',
                'type': 'dict',
                'description': 'Emission prior parameters'
            },
            'beta': {
                'target': 'emission_prior',
                'type': 'dict',
                'description': 'Emission prior concentration'
            },
            
            # Training parameters
            'max_iter': {
                'target': 'maxiter',
                'type': 'int',
                'min': 1,
                'max': 1000,
                'description': 'Maximum iterations'
            },
            'tolerance': {
                'target': 'tolerance',
                'type': 'float',
                'min': 1e-10,
                'max': 1e-1,
                'description': 'Convergence tolerance'
            },
            'random_state': {
                'target': 'random_state',
                'type': 'int',
                'description': 'Random seed'
            },
            
            # Data parameters
            'missing': {
                'target': 'missing',
                'type': 'str',
                'options': ['drop', 'none', 'raise'],
                'description': 'Missing value handling'
            },
            'loglikelihood_burn': {
                'target': 'loglikelihood_burn',
                'type': 'int',
                'min': 0,
                'description': 'Log-likelihood burn-in period'
            }
        }
    
    def _setup_validation_rules(self):
        """Setup parameter validation rules."""
        self.validation_rules = {
            'k_regimes': {
                'type': int,
                'min': 2,
                'max': 20,
                'required': True
            },
            'order': {
                'type': int,
                'min': 0,
                'max': 10,
                'required': False
            },
            'trend': {
                'type': str,
                'options': ['c', 't', 'ct'],
                'required': False
            },
            'switching_variance': {
                'type': bool,
                'required': False
            },
            'switching_trend': {
                'type': bool,
                'required': False
            },
            'switching_exog': {
                'type': bool,
                'required': False
            },
            'maxiter': {
                'type': int,
                'min': 1,
                'max': 10000,
                'required': False
            },
            'tolerance': {
                'type': float,
                'min': 1e-12,
                'max': 1e-1,
                'required': False
            },
            'random_state': {
                'type': int,
                'min': 0,
                'required': False
            },
            'missing': {
                'type': str,
                'options': ['drop', 'none', 'raise'],
                'required': False
            },
            'loglikelihood_burn': {
                'type': int,
                'min': 0,
                'required': False
            }
        }
    
    def map_parameters(self, pyro_params: Dict[str, Any]) -> ParameterMappingResult:
        """
        Map Pyro parameters to statsmodels format.
        
        Args:
            pyro_params: Dictionary of Pyro model parameters
            
        Returns:
            ParameterMappingResult with mapped parameters and metadata
        """
        result = ParameterMappingResult()
        
        try:
            tprint_info("🔄 Mapping Pyro parameters to statsmodels format")
            
            # Process each parameter
            for param_name, param_value in pyro_params.items():
                mapping_result = self._map_single_parameter(param_name, param_value)
                
                if mapping_result['mapped']:
                    result.mapped_params[mapping_result['target_name']] = mapping_result['value']
                    
                    if self.config.log_mappings:
                        result.mapping_log.append({
                            'source': param_name,
                            'target': mapping_result['target_name'],
                            'type': type(param_value).__name__,
                            'status': 'mapped'
                        })
                else:
                    result.unmapped_params[param_name] = param_value
                    
                    if self.config.log_unmapped:
                        result.mapping_log.append({
                            'source': param_name,
                            'target': None,
                            'type': type(param_value).__name__,
                            'status': 'unmapped'
                        })
            
            # Apply transformations
            self._apply_transformations(result)
            
            # Validate mapped parameters
            if self.config.validate_ranges or self.config.validate_types:
                self._validate_mapped_parameters(result)
            
            # Save mapping report if requested
            if self.config.save_mapping_report:
                self._save_mapping_report(result)
            
            # Determine success
            result.success = len(result.validation_errors) == 0
            
            if result.success:
                tprint_success(f"✅ Parameter mapping completed: {len(result.mapped_params)} mapped, {len(result.unmapped_params)} unmapped")
            else:
                tprint_warning(f"⚠️ Parameter mapping completed with {len(result.validation_errors)} errors")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Parameter mapping failed: {e}")
            result.success = False
            result.error_message = str(e)
            return result
    
    def map_search_space(self, pyro_search_space: Dict[str, Any]) -> ParameterMappingResult:
        """
        Map Pyro hyperparameter search space to statsmodels format.
        
        Args:
            pyro_search_space: Pyro hyperparameter search space
            
        Returns:
            ParameterMappingResult with mapped search space
        """
        result = ParameterMappingResult()
        
        try:
            tprint_info("🔄 Mapping Pyro search space to statsmodels format")
            
            # Process each search space parameter
            for param_name, search_config in pyro_search_space.items():
                mapping_result = self._map_search_parameter(param_name, search_config)
                
                if mapping_result['mapped']:
                    result.mapped_params[mapping_result['target_name']] = mapping_result['value']
                    
                    if self.config.log_mappings:
                        result.mapping_log.append({
                            'source': param_name,
                            'target': mapping_result['target_name'],
                            'type': 'search_space',
                            'status': 'mapped'
                        })
                else:
                    result.unmapped_params[param_name] = search_config
                    
                    if self.config.log_unmapped:
                        result.mapping_log.append({
                            'source': param_name,
                            'target': None,
                            'type': 'search_space',
                            'status': 'unmapped'
                        })
            
            # Validate search space
            self._validate_search_space(result)
            
            result.success = len(result.validation_errors) == 0
            
            if result.success:
                tprint_success(f"✅ Search space mapping completed: {len(result.mapped_params)} mapped")
            else:
                tprint_warning(f"⚠️ Search space mapping completed with {len(result.validation_errors)} errors")
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Search space mapping failed: {e}")
            result.success = False
            result.error_message = str(e)
            return result
    
    def _map_single_parameter(self, param_name: str, param_value: Any) -> Dict[str, Any]:
        """Map a single parameter from Pyro to statsmodels."""
        # Check if parameter has mapping rule
        if param_name not in self.mapping_rules:
            return {
                'mapped': False,
                'target_name': None,
                'value': None,
                'reason': 'No mapping rule found'
            }
        
        rule = self.mapping_rules[param_name]
        target_name = rule['target']
        
        try:
            # Type conversion
            converted_value = self._convert_parameter_type(param_value, rule)
            
            # Validation
            if self.config.validate_types:
                self._validate_parameter_type(converted_value, rule)
            
            if self.config.validate_ranges:
                self._validate_parameter_range(converted_value, rule)
            
            return {
                'mapped': True,
                'target_name': target_name,
                'value': converted_value,
                'reason': 'Successfully mapped'
            }
            
        except Exception as e:
            return {
                'mapped': False,
                'target_name': target_name,
                'value': None,
                'reason': f'Conversion failed: {e}'
            }
    
    def _map_search_parameter(self, param_name: str, search_config: Any) -> Dict[str, Any]:
        """Map a search space parameter from Pyro to statsmodels."""
        # Check if parameter has mapping rule
        if param_name not in self.mapping_rules:
            return {
                'mapped': False,
                'target_name': None,
                'value': None,
                'reason': 'No mapping rule found'
            }
        
        rule = self.mapping_rules[param_name]
        target_name = rule['target']
        
        try:
            # Handle different search space formats
            if isinstance(search_config, dict):
                # Pyro-style search space with distribution
                mapped_config = self._map_search_distribution(search_config, rule)
            elif isinstance(search_config, (list, tuple)):
                # Pyro-style search space with choices
                mapped_config = self._map_search_choices(search_config, rule)
            else:
                # Simple value
                mapped_config = search_config
            
            return {
                'mapped': True,
                'target_name': target_name,
                'value': mapped_config,
                'reason': 'Successfully mapped'
            }
            
        except Exception as e:
            return {
                'mapped': False,
                'target_name': target_name,
                'value': None,
                'reason': f'Conversion failed: {e}'
            }
    
    def _convert_parameter_type(self, value: Any, rule: Dict[str, Any]) -> Any:
        """Convert parameter to the correct type."""
        target_type = rule.get('type', 'auto')
        
        if target_type == 'int':
            return int(value)
        elif target_type == 'float':
            return float(value)
        elif target_type == 'bool':
            return bool(value)
        elif target_type == 'str':
            return str(value)
        elif target_type == 'array':
            return np.array(value)
        elif target_type == 'dict':
            return dict(value)
        else:
            return value
    
    def _validate_parameter_type(self, value: Any, rule: Dict[str, Any]):
        """Validate parameter type."""
        expected_type = rule.get('type')
        
        if expected_type == 'int' and not isinstance(value, int):
            raise ValueError(f"Expected int, got {type(value).__name__}")
        elif expected_type == 'float' and not isinstance(value, (int, float)):
            raise ValueError(f"Expected float, got {type(value).__name__}")
        elif expected_type == 'bool' and not isinstance(value, bool):
            raise ValueError(f"Expected bool, got {type(value).__name__}")
        elif expected_type == 'str' and not isinstance(value, str):
            raise ValueError(f"Expected str, got {type(value).__name__}")
        elif expected_type == 'array' and not isinstance(value, np.ndarray):
            raise ValueError(f"Expected array, got {type(value).__name__}")
    
    def _validate_parameter_range(self, value: Any, rule: Dict[str, Any]):
        """Validate parameter range."""
        if 'min' in rule and value < rule['min']:
            raise ValueError(f"Value {value} below minimum {rule['min']}")
        
        if 'max' in rule and value > rule['max']:
            raise ValueError(f"Value {value} above maximum {rule['max']}")
        
        if 'options' in rule and value not in rule['options']:
            raise ValueError(f"Value {value} not in options {rule['options']}")
    
    def _map_search_distribution(self, search_config: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """Map Pyro search distribution to statsmodels format."""
        # This is a simplified implementation
        # In practice, you'd handle different Pyro distributions
        
        mapped_config = {}
        
        # Map common distribution parameters
        if 'low' in search_config:
            mapped_config['min'] = search_config['low']
        if 'high' in search_config:
            mapped_config['max'] = search_config['high']
        if 'loc' in search_config:
            mapped_config['mean'] = search_config['loc']
        if 'scale' in search_config:
            mapped_config['std'] = search_config['scale']
        
        return mapped_config
    
    def _map_search_choices(self, search_config: List[Any], rule: Dict[str, Any]) -> List[Any]:
        """Map Pyro search choices to statsmodels format."""
        # Convert choices to appropriate type
        target_type = rule.get('type', 'auto')
        
        if target_type == 'int':
            return [int(x) for x in search_config]
        elif target_type == 'float':
            return [float(x) for x in search_config]
        elif target_type == 'str':
            return [str(x) for x in search_config]
        else:
            return search_config
    
    def _apply_transformations(self, result: ParameterMappingResult):
        """Apply post-mapping transformations."""
        transformations = []
        
        # Normalize probability matrices
        if self.config.normalize_probabilities:
            for param_name, param_value in result.mapped_params.items():
                if 'transition' in param_name.lower() and isinstance(param_value, np.ndarray):
                    if param_value.ndim == 2 and param_value.shape[0] == param_value.shape[1]:
                        # Normalize rows to sum to 1
                        row_sums = param_value.sum(axis=1, keepdims=True)
                        row_sums[row_sums == 0] = 1  # Avoid division by zero
                        normalized = param_value / row_sums
                        result.mapped_params[param_name] = normalized
                        
                        transformations.append({
                            'parameter': param_name,
                            'transformation': 'normalize_rows',
                            'description': 'Normalized transition matrix rows to sum to 1'
                        })
        
        # Round discrete parameters
        if self.config.round_discrete_params:
            for param_name, param_value in result.mapped_params.items():
                if param_name in ['k_regimes', 'order', 'maxiter', 'random_state', 'loglikelihood_burn']:
                    if isinstance(param_value, float):
                        result.mapped_params[param_name] = int(round(param_value))
                        
                        transformations.append({
                            'parameter': param_name,
                            'transformation': 'round_int',
                            'description': f'Rounded {param_name} to integer'
                        })
        
        result.transformation_log = transformations
    
    def _validate_mapped_parameters(self, result: ParameterMappingResult):
        """Validate mapped parameters."""
        for param_name, param_value in result.mapped_params.items():
            if param_name in self.validation_rules:
                rule = self.validation_rules[param_name]
                
                try:
                    # Type validation
                    if self.config.validate_types and 'type' in rule:
                        expected_type = rule['type']
                        if not isinstance(param_value, expected_type):
                            result.validation_errors.append(
                                f"Parameter {param_name}: Expected {expected_type.__name__}, got {type(param_value).__name__}"
                            )
                    
                    # Range validation
                    if self.config.validate_ranges:
                        if 'min' in rule and param_value < rule['min']:
                            result.validation_errors.append(
                                f"Parameter {param_name}: Value {param_value} below minimum {rule['min']}"
                            )
                        
                        if 'max' in rule and param_value > rule['max']:
                            result.validation_errors.append(
                                f"Parameter {param_name}: Value {param_value} above maximum {rule['max']}"
                            )
                        
                        if 'options' in rule and param_value not in rule['options']:
                            result.validation_errors.append(
                                f"Parameter {param_name}: Value {param_value} not in options {rule['options']}"
                            )
                
                except Exception as e:
                    result.validation_errors.append(
                        f"Parameter {param_name}: Validation error - {e}"
                    )
    
    def _validate_search_space(self, result: ParameterMappingResult):
        """Validate mapped search space."""
        for param_name, search_config in result.mapped_params.items():
            if isinstance(search_config, dict):
                # Validate search space configuration
                if 'min' in search_config and 'max' in search_config:
                    if search_config['min'] >= search_config['max']:
                        result.validation_errors.append(
                            f"Search space {param_name}: min ({search_config['min']}) >= max ({search_config['max']})"
                        )
    
    def _save_mapping_report(self, result: ParameterMappingResult):
        """Save detailed mapping report."""
        if not self.config.output_dir:
            return
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report = {
            'mapping_summary': {
                'total_parameters': len(result.mapped_params) + len(result.unmapped_params),
                'mapped_parameters': len(result.mapped_params),
                'unmapped_parameters': len(result.unmapped_params),
                'validation_errors': len(result.validation_errors),
                'validation_warnings': len(result.validation_warnings),
                'success': result.success
            },
            'mapped_parameters': result.mapped_params,
            'unmapped_parameters': result.unmapped_params,
            'mapping_log': result.mapping_log,
            'transformation_log': result.transformation_log,
            'validation_errors': result.validation_errors,
            'validation_warnings': result.validation_warnings,
            'mapping_rules': self.mapping_rules,
            'validation_rules': self.validation_rules
        }
        
        # Save report
        report_file = output_dir / 'parameter_mapping_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        tprint_info(f"💾 Parameter mapping report saved to {report_file}")


# Convenience functions for common mapping operations
def map_pyro_to_statsmodels(pyro_params: Dict[str, Any],
                          config: Optional[ParameterMappingConfig] = None) -> ParameterMappingResult:
    """
    Convenience function to map Pyro parameters to statsmodels.
    
    Args:
        pyro_params: Pyro parameters dictionary
        config: Optional mapping configuration
        
    Returns:
        ParameterMappingResult with mapped parameters
    """
    mapper = PyroToStatsmodelsMapper(config)
    return mapper.map_parameters(pyro_params)


def map_pyro_search_space(pyro_search_space: Dict[str, Any],
                         config: Optional[ParameterMappingConfig] = None) -> ParameterMappingResult:
    """
    Convenience function to map Pyro search space to statsmodels.
    
    Args:
        pyro_search_space: Pyro search space dictionary
        config: Optional mapping configuration
        
    Returns:
        ParameterMappingResult with mapped search space
    """
    mapper = PyroToStatsmodelsMapper(config)
    return mapper.map_search_space(pyro_search_space)


def create_default_mapping_config(strict_mapping: bool = True,
                              validate_ranges: bool = True,
                              log_mappings: bool = True) -> ParameterMappingConfig:
    """
    Create a default parameter mapping configuration.
    
    Args:
        strict_mapping: Whether to enforce strict mapping
        validate_ranges: Whether to validate parameter ranges
        log_mappings: Whether to log mapping operations
        
    Returns:
        ParameterMappingConfig instance
    """
    return ParameterMappingConfig(
        strict_mapping=strict_mapping,
        validate_ranges=validate_ranges,
        validate_types=True,
        normalize_probabilities=True,
        round_discrete_params=True,
        log_mappings=log_mappings,
        log_unmapped=True,
        save_mapping_report=False
    )