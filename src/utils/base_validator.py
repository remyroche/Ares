"""
Base Validator

This module provides a base validator class for step validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
import time
from datetime import datetime

class BaseValidator(ABC):
    """
    Base validator class for step validation.

    This class provides a common interface for all step validators.
    """

    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the base validator."""
        self.step_name = step_name
        self.config = config or {}
        self.logger = logging.getLogger(f"validator.{step_name}")
        self.validation_history: List[Dict[str, Any]] = []
        self.last_validation_result: Optional[Dict[str, Any]] = None

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

    def _record_validation(self, result: Dict[str, Any]) -> None:
        """Record validation result in history."""
        validation_record = {
            'timestamp': datetime.now().isoformat(),
            'step_name': self.step_name,
            'result': result,
            'config': self.config
        }
        self.validation_history.append(validation_record)
        self.last_validation_result = result

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_history.copy()

    def clear_validation_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()
        self.last_validation_result = None


class ConcreteValidator(BaseValidator):
    """
    Concrete implementation of BaseValidator for general use.
    """

    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the concrete validator."""
        super().__init__(step_name, config)
        self.validation_rules = self.config.get('validation_rules', {})
        self.required_fields = self.config.get('required_fields', [])
        self.data_types = self.config.get('data_types', {})

    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate the step data with comprehensive checks.

        Args:
            data: The data to validate
            context: Additional context for validation

        Returns:
            Validation results
        """
        start_time = time.time()
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'validation_time': 0.0,
            'step_name': self.step_name,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Basic data existence check
            if data is None:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Data is None")
                return validation_result

            # Check required fields if data is a dictionary
            if isinstance(data, dict):
                missing_fields = [field for field in self.required_fields if field not in data]
                if missing_fields:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(f"Missing required fields: {missing_fields}")

            # Type validation
            for field, expected_type in self.data_types.items():
                if isinstance(data, dict) and field in data:
                    if not isinstance(data[field], expected_type):
                        validation_result['warnings'].append(
                            f"Field '{field}' expected type {expected_type.__name__}, got {type(data[field]).__name__}"
                        )

            # Custom validation rules
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    rule_result = rule_func(data, context)
                    if not rule_result:
                        validation_result['is_valid'] = False
                        validation_result['errors'].append(f"Validation rule '{rule_name}' failed")
                except Exception as e:
                    validation_result['warnings'].append(f"Validation rule '{rule_name}' error: {str(e)}")

            # Data quality checks
            if hasattr(data, '__len__'):
                if len(data) == 0:
                    validation_result['warnings'].append("Data is empty")
                elif len(data) > self.config.get('max_size', float('inf')):
                    validation_result['warnings'].append(f"Data size {len(data)} exceeds maximum {self.config.get('max_size')}")

        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Validation error for {self.step_name}: {e}")

        finally:
            validation_result['validation_time'] = time.time() - start_time
            self._record_validation(validation_result)

        return validation_result

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the validation results.

        Returns:
            Summary of validation results
        """
        if not self.validation_history:
            return {
                'total_validations': 0,
                'success_rate': 0.0,
                'average_validation_time': 0.0,
                'last_validation': None
            }

        total_validations = len(self.validation_history)
        successful_validations = sum(1 for record in self.validation_history if record['result']['is_valid'])
        success_rate = successful_validations / total_validations
        average_time = sum(record['result']['validation_time'] for record in self.validation_history) / total_validations

        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': success_rate,
            'average_validation_time': average_time,
            'last_validation': self.last_validation_result,
            'step_name': self.step_name
        }
