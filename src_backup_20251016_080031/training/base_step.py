"""
Base Step for Training Pipeline

This module defines a simple BaseStep class that can be inherited by other training steps.
It provides a basic structure for steps in the training pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseStep(ABC):
    """
    Abstract Base Class for a training step.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    async def execute(self, data: Any) -> Any:
        """
        Execute the logic of the training step.
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the configuration for the step.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status and metrics of the step.
        """
        pass
