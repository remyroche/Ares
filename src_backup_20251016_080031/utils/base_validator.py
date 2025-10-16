"""
Base Validator

This module provides a base validator class for step validation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseValidator(ABC):
    """
    Base validator class for step validation.

    This class provides a common interface for all step validators.
    """

    def __init__(self, step_name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the base validator."""
        self.step_name = step_name
        self.config = config or {}

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
