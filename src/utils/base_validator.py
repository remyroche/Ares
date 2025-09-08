"""
Base Validator Module

This module provides the base validation framework for all validators in the system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class BaseValidator(ABC):
    """Base class for all validators in the system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the validator with optional configuration."""
        self.config = config or {}
        self.logger = logger.getChild(self.__class__.__name__)
        self.validation_count = 0
        self.error_count = 0
    
    @abstractmethod
    def validate(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Validate the provided data.
        
        Args:
            data: The data to validate
            **kwargs: Additional validation parameters
            
        Returns:
            Dict containing validation results
        """
        pass
    
    def validate_batch(self, data_list: List[Any], **kwargs) -> List[Dict[str, Any]]:
        """
        Validate a batch of data items.
        
        Args:
            data_list: List of data items to validate
            **kwargs: Additional validation parameters
            
        Returns:
            List of validation results
        """
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.validate(data, **kwargs)
                result['index'] = i
                results.append(result)
            except Exception as e:
                self.error_count += 1
                self.logger.error(f"Validation error for item {i}: {e}")
                results.append({
                    'index': i,
                    'valid': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'validation_count': self.validation_count,
            'error_count': self.error_count,
            'success_rate': (self.validation_count - self.error_count) / max(self.validation_count, 1)
        }
    
    def reset_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_count = 0
        self.error_count = 0

__all__ = ['BaseValidator']
