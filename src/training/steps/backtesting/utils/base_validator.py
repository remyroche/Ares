from ..standardized_parquet_handler import standardized_parquet_handler
"""Base validator class for backtesting pipeline."""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseValidator(ABC):
    """Base class for all validators in the backtesting pipeline."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize the validator.

        Args:
            name: Name of the validator
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.validation_results = {}

    @abstractmethod
    async def validate(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> bool:
        """Validate the given inputs and pipeline state.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            True if validation passes, False otherwise
        """
        pass

    def _log_validation_success(self, message: str) -> None:
        """Log a validation success message."""
        self.logger.info(f"✅ {self.name}: {message}")

    def _log_validation_error(self, message: str) -> None:
        """Log a validation error message."""
        self.logger.error(f"❌ {self.name}: {message}")

    def _log_validation_warning(self, message: str) -> None:
        """Log a validation warning message."""
        self.logger.warning(f"⚠️ {self.name}: {message}")
