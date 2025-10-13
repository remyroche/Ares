"""
Configuration validation utilities for feature selection.

This module provides comprehensive validation of feature selection
configurations to ensure they are valid and consistent.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import re

from src.utils.tprint import tprint_debug, tprint_info, tprint_warning, tprint_success


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validated_config: Optional[Dict[str, Any]] = None


class ConfigValidator:
    """Validator for feature selection configurations."""
    
    def __init__(self):
        self.logger = get_logger("ConfigValidator")
        
        # Define validation rules
        self.validation_rules = {
            'target_features': {
                'type': int,
                'min_value': 1,
                'max_value': 1000,
                'required': True
            },
            'min_features': {
                'type': int,
                'min_value': 1,
                'max_value': 500,
                'required': True
            },
            'max_features': {
                'type': int,
                'min_value': 1,
                'max_value': 1000,
                'required': True
            },
            'vif_threshold': {
                'type': float,
                'min_value': 1.0,
                'max_value': 100.0,
                'required': False,
                'default': 10.0
            },
            'correlation_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'required': False,
                'default': 0.95
            },
            'mutual_info_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'required': False,
                'default': 0.001
            },
            'variance_threshold': {
                'type': float,
                'min_value': 0.0,
                'max_value': 1.0,
                'required': False,
                'default': 0.01
            },
            'priority_categories': {
                'type': list,
                'item_type': str,
                'required': False,
                'default': ['momentum', 'volatility']
            },
            'enable_parallel_processing': {
                'type': bool,
                'required': False,
                'default': True
            },
            'memory_efficient': {
                'type': bool,
                'required': False,
                'default': True
            },
            'chunk_size': {
                'type': int,
                'min_value': 100,
                'max_value': 10000,
                'required': False,
                'default': 1000
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a feature selection configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation results
        """
        tprint_info("🔍 Validating feature selection configuration")
        
        errors = []
        warnings = []
        validated_config = config.copy()
        
        try:
            # Validate required fields
            errors.extend(self._validate_required_fields(config))
            
            # Validate field types and values
            errors.extend(self._validate_field_types_and_values(config))
            
            # Validate logical consistency
            errors.extend(self._validate_logical_consistency(config))
            
            # Validate model-specific parameters
            errors.extend(self._validate_model_specific_params(config))
            
            # Apply defaults for missing optional fields
            validated_config = self._apply_defaults(validated_config)
            
            # Generate warnings for potential issues
            warnings.extend(self._generate_warnings(validated_config))
            
            is_valid = len(errors) == 0
            
            if is_valid:
                tprint_success(f"   ✅ Configuration validation passed")
                if warnings:
                    tprint_warning(f"   ⚠️ {len(warnings)} warnings generated")
            else:
                tprint_warning(f"   ⚠️ Configuration validation failed: {len(errors)} errors")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validated_config=validated_config if is_valid else None
            )
            
        except Exception as e:
            error_msg = f"Validation failed with exception: {e}"
            tprint_warning(f"   ⚠️ {error_msg}")
            return ValidationResult(
                is_valid=False,
                errors=[error_msg],
                warnings=[],
                validated_config=None
            )
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> List[str]:
        """Validate required fields are present."""
        errors = []
        
        for field, rules in self.validation_rules.items():
            if rules.get('required', False) and field not in config:
                errors.append(f"Required field '{field}' is missing")
        
        return errors
    
    def _validate_field_types_and_values(self, config: Dict[str, Any]) -> List[str]:
        """Validate field types and value ranges."""
        errors = []
        
        for field, value in config.items():
            if field not in self.validation_rules:
                continue  # Skip unknown fields
            
            rules = self.validation_rules[field]
            
            # Check type
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                errors.append(f"Field '{field}' must be of type {expected_type.__name__}, got {type(value).__name__}")
                continue
            
            # Check value ranges for numeric types
            if expected_type in (int, float):
                if 'min_value' in rules and value < rules['min_value']:
                    errors.append(f"Field '{field}' must be >= {rules['min_value']}, got {value}")
                
                if 'max_value' in rules and value > rules['max_value']:
                    errors.append(f"Field '{field}' must be <= {rules['max_value']}, got {value}")
            
            # Check list item types
            if expected_type == list and 'item_type' in rules:
                if not all(isinstance(item, rules['item_type']) for item in value):
                    errors.append(f"Field '{field}' must contain only {rules['item_type'].__name__} items")
        
        return errors
    
    def _validate_logical_consistency(self, config: Dict[str, Any]) -> List[str]:
        """Validate logical consistency between fields."""
        errors = []
        
        # Check min_features <= target_features <= max_features
        if all(field in config for field in ['min_features', 'target_features', 'max_features']):
            min_features = config['min_features']
            target_features = config['target_features']
            max_features = config['max_features']
            
            if min_features > target_features:
                errors.append("min_features must be <= target_features")
            
            if target_features > max_features:
                errors.append("target_features must be <= max_features")
            
            if min_features > max_features:
                errors.append("min_features must be <= max_features")
        
        # Check threshold consistency
        if 'vif_threshold' in config and 'correlation_threshold' in config:
            vif_threshold = config['vif_threshold']
            correlation_threshold = config['correlation_threshold']
            
            # VIF threshold should be reasonable relative to correlation threshold
            if vif_threshold < 1.0 / (1.0 - correlation_threshold + 1e-10):
                errors.append("vif_threshold may be too low relative to correlation_threshold")
        
        # Check chunk size relative to expected data size
        if 'chunk_size' in config and 'target_features' in config:
            chunk_size = config['chunk_size']
            target_features = config['target_features']
            
            if chunk_size < target_features:
                errors.append("chunk_size should be >= target_features for efficient processing")
        
        return errors
    
    def _validate_model_specific_params(self, config: Dict[str, Any]) -> List[str]:
        """Validate model-specific parameters."""
        errors = []
        
        # Check model_profiles if present
        if 'model_profiles' in config:
            model_profiles = config['model_profiles']
            if not isinstance(model_profiles, dict):
                errors.append("model_profiles must be a dictionary")
            else:
                for model_name, model_config in model_profiles.items():
                    if not isinstance(model_config, dict):
                        errors.append(f"Model profile '{model_name}' must be a dictionary")
                        continue
                    
                    # Validate each model profile
                    model_errors = self._validate_model_profile(model_config, model_name)
                    errors.extend(model_errors)
        
        # Check performance settings
        if 'performance' in config:
            performance = config['performance']
            if not isinstance(performance, dict):
                errors.append("performance must be a dictionary")
            else:
                if 'max_workers' in performance:
                    max_workers = performance['max_workers']
                    if not isinstance(max_workers, int) or max_workers < -1:
                        errors.append("performance.max_workers must be an integer >= -1")
        
        return errors
    
    def _validate_model_profile(self, model_config: Dict[str, Any], model_name: str) -> List[str]:
        """Validate a single model profile."""
        errors = []
        
        # Required fields for model profiles
        required_fields = ['target_features', 'min_features', 'max_features']
        for field in required_fields:
            if field not in model_config:
                errors.append(f"Model profile '{model_name}' missing required field '{field}'")
        
        # Validate numeric fields
        numeric_fields = ['target_features', 'min_features', 'max_features', 'vif_threshold', 'correlation_threshold']
        for field in numeric_fields:
            if field in model_config:
                value = model_config[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"Model profile '{model_name}' field '{field}' must be numeric")
        
        return errors
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing optional fields."""
        validated_config = config.copy()
        
        for field, rules in self.validation_rules.items():
            if field not in validated_config and 'default' in rules:
                validated_config[field] = rules['default']
                tprint_debug(f"   🔧 Applied default for {field}: {rules['default']}")
        
        return validated_config
    
    def _generate_warnings(self, config: Dict[str, Any]) -> List[str]:
        """Generate warnings for potential configuration issues."""
        warnings = []
        
        # Check for potentially inefficient settings
        if config.get('chunk_size', 1000) < 500:
            warnings.append("chunk_size < 500 may cause inefficient processing")
        
        if config.get('target_features', 80) > 200:
            warnings.append("target_features > 200 may cause memory issues")
        
        # Check for potentially aggressive thresholds
        if config.get('vif_threshold', 10.0) < 5.0:
            warnings.append("vif_threshold < 5.0 may be too aggressive")
        
        if config.get('correlation_threshold', 0.95) < 0.8:
            warnings.append("correlation_threshold < 0.8 may remove too many features")
        
        # Check for missing performance optimizations
        if not config.get('enable_parallel_processing', True):
            warnings.append("Parallel processing is disabled, may impact performance")
        
        if not config.get('memory_efficient', True):
            warnings.append("Memory efficiency is disabled, may cause memory issues")
        
        # Check priority categories
        priority_categories = config.get('priority_categories', [])
        if len(priority_categories) == 0:
            warnings.append("No priority categories specified, using default selection")
        
        # Check model-specific warnings
        if 'model_profiles' in config:
            for model_name, model_config in config['model_profiles'].items():
                if model_config.get('target_features', 80) < 20:
                    warnings.append(f"Model '{model_name}' has very low target_features")
        
        return warnings
    
    def validate_model_type(self, model_type: str) -> bool:
        """Validate that a model type is supported."""
        valid_model_types = [
            'neural_network', 'linear_model', 'ensemble_model', 'time_series',
            'regime_detection', 'AdvancedMambaHybrid', 'FinancialResNet',
            'DeepScaler', 'NBEATS', 'default'
        ]
        
        return model_type.lower() in [t.lower() for t in valid_model_types]
    
    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results."""
        return {
            'is_valid': result.is_valid,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings),
            'errors': result.errors,
            'warnings': result.warnings,
            'has_validated_config': result.validated_config is not None
        }
    
    def create_validation_report(self, result: ValidationResult) -> str:
        """Create a human-readable validation report."""
        report = []
        report.append("=" * 50)
        report.append("FEATURE SELECTION CONFIGURATION VALIDATION REPORT")
        report.append("=" * 50)
        
        if result.is_valid:
            report.append("✅ VALIDATION PASSED")
        else:
            report.append("❌ VALIDATION FAILED")
        
        report.append(f"Errors: {len(result.errors)}")
        report.append(f"Warnings: {len(result.warnings)}")
        report.append("")
        
        if result.errors:
            report.append("ERRORS:")
            for i, error in enumerate(result.errors, 1):
                report.append(f"  {i}. {error}")
            report.append("")
        
        if result.warnings:
            report.append("WARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"  {i}. {warning}")
            report.append("")
        
        if result.validated_config:
            report.append("VALIDATED CONFIGURATION:")
            for key, value in result.validated_config.items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)