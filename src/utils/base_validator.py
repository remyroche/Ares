"""
Base Validator

This module provides a base validator class for step validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import time

class BaseValidator(ABC):
    """
    Base validator class for step validation.

    This class provides a common interface for all step validators.
    """

    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the base validator."""
        self.step_name = step_name
        self.config = config or {}
        self.validation_history = []
        self.last_validation_result = None
        self.validation_count = 0

    @abstractmethod
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate the step data.

        Args:
            data: The data to validate
            context: Additional context for validation

        Returns:
            Validation results
        """
        pass

    @abstractmethod
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the validation results.

        Returns:
            Summary of validation results
        """
        pass
    
    def _record_validation(self, result: Dict[str, Any], context: Optional[Dict[str, Any]] = None):
        """Record validation result for tracking."""
        self.validation_count += 1
        validation_record = {
            'validation_id': self.validation_count,
            'step_name': self.step_name,
            'timestamp': time.time(),
            'result': result,
            'context': context or {},
            'success': result.get('success', False)
        }
        self.validation_history.append(validation_record)
        self.last_validation_result = result
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()
    
    def get_last_validation(self) -> Optional[Dict[str, Any]]:
        """Get the last validation result."""
        return self.last_validation_result
    
    def clear_history(self):
        """Clear validation history."""
        self.validation_history = []
        self.last_validation_result = None
        self.validation_count = 0
    
    def is_valid(self, data: Any, context: Optional[Dict[str, Any]] = None) -> bool:
        """Synchronous validation check."""
        try:
            import asyncio
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # If we're in an async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.validate(data, context))
                    result = future.result(timeout=30)  # 30 second timeout
            except RuntimeError:
                # No event loop running, we can create one
                result = asyncio.run(self.validate(data, context))
            return result.get('success', False)
        except Exception as e:
            print(f"Validation error: {e}")
            return False


class DataValidator(BaseValidator):
    """Concrete validator for data validation."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        self.required_fields = self.config.get('required_fields', [])
        self.data_types = self.config.get('data_types', {})
        self.value_ranges = self.config.get('value_ranges', {})
    
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate data according to configuration."""
        try:
            validation_result = {
                'success': True,
                'step_name': self.step_name,
                'validation_type': 'data_validation',
                'errors': [],
                'warnings': [],
                'validated_fields': [],
                'timestamp': time.time()
            }
            
            # Check if data is None or empty
            if data is None:
                validation_result['success'] = False
                validation_result['errors'].append("Data is None")
                self._record_validation(validation_result, context)
                return validation_result
            
            # Check if data is empty
            if hasattr(data, '__len__') and len(data) == 0:
                validation_result['success'] = False
                validation_result['errors'].append("Data is empty")
                self._record_validation(validation_result, context)
                return validation_result
            
            # Validate required fields if data is a dictionary
            if isinstance(data, dict):
                for field in self.required_fields:
                    if field not in data:
                        validation_result['success'] = False
                        validation_result['errors'].append(f"Required field '{field}' not found")
                    else:
                        validation_result['validated_fields'].append(field)
                
                # Validate data types
                for field, expected_type in self.data_types.items():
                    if field in data:
                        if not isinstance(data[field], expected_type):
                            validation_result['success'] = False
                            validation_result['errors'].append(
                                f"Field '{field}' has wrong type. Expected {expected_type.__name__}, got {type(data[field]).__name__}"
                            )
                
                # Validate value ranges
                for field, (min_val, max_val) in self.value_ranges.items():
                    if field in data:
                        value = data[field]
                        if isinstance(value, (int, float)):
                            if not (min_val <= value <= max_val):
                                validation_result['success'] = False
                                validation_result['errors'].append(
                                    f"Field '{field}' value {value} is outside range [{min_val}, {max_val}]"
                                )
            
            # Validate data structure
            if hasattr(data, 'shape'):
                if len(data.shape) == 0:
                    validation_result['success'] = False
                    validation_result['errors'].append("Data has zero dimensions")
                elif data.shape[0] == 0:
                    validation_result['success'] = False
                    validation_result['errors'].append("Data has zero samples")
            
            self._record_validation(validation_result, context)
            return validation_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'step_name': self.step_name,
                'validation_type': 'data_validation',
                'errors': [f"Validation failed with exception: {str(e)}"],
                'warnings': [],
                'validated_fields': [],
                'timestamp': time.time()
            }
            self._record_validation(error_result, context)
            return error_result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_history:
            return {
                'step_name': self.step_name,
                'total_validations': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success_rate': 0.0,
                'last_validation': None
            }
        
        successful = sum(1 for v in self.validation_history if v['success'])
        total = len(self.validation_history)
        
        return {
            'step_name': self.step_name,
            'total_validations': total,
            'successful_validations': successful,
            'failed_validations': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'last_validation': self.last_validation_result,
            'validation_history_length': len(self.validation_history)
        }


class ModelValidator(BaseValidator):
    """Concrete validator for model validation."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        self.required_methods = self.config.get('required_methods', ['fit', 'predict'])
        self.performance_thresholds = self.config.get('performance_thresholds', {})
    
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate model according to configuration."""
        try:
            validation_result = {
                'success': True,
                'step_name': self.step_name,
                'validation_type': 'model_validation',
                'errors': [],
                'warnings': [],
                'validated_methods': [],
                'timestamp': time.time()
            }
            
            # Check if model is None
            if data is None:
                validation_result['success'] = False
                validation_result['errors'].append("Model is None")
                self._record_validation(validation_result, context)
                return validation_result
            
            # Validate required methods
            for method in self.required_methods:
                if hasattr(data, method):
                    validation_result['validated_methods'].append(method)
                else:
                    validation_result['success'] = False
                    validation_result['errors'].append(f"Required method '{method}' not found")
            
            # Check if model is fitted (if it has the attribute)
            if hasattr(data, 'is_fitted'):
                if not data.is_fitted:
                    validation_result['warnings'].append("Model is not fitted")
            
            # Validate performance if provided in context
            if context and 'performance_metrics' in context:
                metrics = context['performance_metrics']
                for metric, threshold in self.performance_thresholds.items():
                    if metric in metrics:
                        if metrics[metric] < threshold:
                            validation_result['warnings'].append(
                                f"Performance metric '{metric}' ({metrics[metric]:.4f}) is below threshold ({threshold:.4f})"
                            )
            
            self._record_validation(validation_result, context)
            return validation_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'step_name': self.step_name,
                'validation_type': 'model_validation',
                'errors': [f"Validation failed with exception: {str(e)}"],
                'warnings': [],
                'validated_methods': [],
                'timestamp': time.time()
            }
            self._record_validation(error_result, context)
            return error_result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_history:
            return {
                'step_name': self.step_name,
                'total_validations': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success_rate': 0.0,
                'last_validation': None
            }
        
        successful = sum(1 for v in self.validation_history if v['success'])
        total = len(self.validation_history)
        
        return {
            'step_name': self.step_name,
            'total_validations': total,
            'successful_validations': successful,
            'failed_validations': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'last_validation': self.last_validation_result,
            'validation_history_length': len(self.validation_history)
        }


class ConfigValidator(BaseValidator):
    """Concrete validator for configuration validation."""
    
    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(step_name, config)
        self.required_keys = self.config.get('required_keys', [])
        self.optional_keys = self.config.get('optional_keys', [])
        self.value_validators = self.config.get('value_validators', {})
    
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate configuration according to specification."""
        try:
            validation_result = {
                'success': True,
                'step_name': self.step_name,
                'validation_type': 'config_validation',
                'errors': [],
                'warnings': [],
                'validated_keys': [],
                'timestamp': time.time()
            }
            
            # Check if config is a dictionary
            if not isinstance(data, dict):
                validation_result['success'] = False
                validation_result['errors'].append("Configuration must be a dictionary")
                self._record_validation(validation_result, context)
                return validation_result
            
            # Validate required keys
            for key in self.required_keys:
                if key not in data:
                    validation_result['success'] = False
                    validation_result['errors'].append(f"Required key '{key}' not found in configuration")
                else:
                    validation_result['validated_keys'].append(key)
            
            # Validate optional keys
            for key in self.optional_keys:
                if key in data:
                    validation_result['validated_keys'].append(key)
            
            # Validate values using custom validators
            for key, validator_func in self.value_validators.items():
                if key in data:
                    try:
                        if not validator_func(data[key]):
                            validation_result['success'] = False
                            validation_result['errors'].append(f"Value validation failed for key '{key}'")
                    except Exception as e:
                        validation_result['success'] = False
                        validation_result['errors'].append(f"Value validation error for key '{key}': {str(e)}")
            
            self._record_validation(validation_result, context)
            return validation_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'step_name': self.step_name,
                'validation_type': 'config_validation',
                'errors': [f"Validation failed with exception: {str(e)}"],
                'warnings': [],
                'validated_keys': [],
                'timestamp': time.time()
            }
            self._record_validation(error_result, context)
            return error_result
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        if not self.validation_history:
            return {
                'step_name': self.step_name,
                'total_validations': 0,
                'successful_validations': 0,
                'failed_validations': 0,
                'success_rate': 0.0,
                'last_validation': None
            }
        
        successful = sum(1 for v in self.validation_history if v['success'])
        total = len(self.validation_history)
        
        return {
            'step_name': self.step_name,
            'total_validations': total,
            'successful_validations': successful,
            'failed_validations': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'last_validation': self.last_validation_result,
            'validation_history_length': len(self.validation_history)
        }
