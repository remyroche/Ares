from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
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
        
        self.logger.info(f"🚀 BaseValidator '{name}' initialized")
        self.logger.info(f"📊 Configuration keys: {list(config.keys())}")
        self.logger.info(f"🔧 Validator type: {self.__class__.__name__}")

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
    
    def _log_validation_info(self, message: str) -> None:
        """Log a validation info message."""
        self.logger.info(f"ℹ️ {self.name}: {message}")
    
    def _log_validation_debug(self, message: str) -> None:
        """Log a validation debug message."""
        self.logger.debug(f"🔍 {self.name}: {message}")
    
    def _log_validation_start(self, validation_type: str) -> None:
        """Log the start of a validation process."""
        self.logger.info(f"🔄 {self.name}: Starting {validation_type} validation")
    
    def _log_validation_complete(self, validation_type: str, success: bool) -> None:
        """Log the completion of a validation process."""
        if success:
            self.logger.info(f"✅ {self.name}: {validation_type} validation completed successfully")
        else:
            self.logger.error(f"❌ {self.name}: {validation_type} validation failed")
    
    def _log_validation_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log validation metrics."""
        self.logger.info(f"📊 {self.name}: Validation metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"   • {key}: {value:,}" if isinstance(value, int) else f"   • {key}: {value:.4f}")
            else:
                self.logger.info(f"   • {key}: {value}")
    
    def _log_validation_summary(self, total_checks: int, passed_checks: int, failed_checks: int) -> None:
        """Log validation summary."""
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        self.logger.info(f"📋 {self.name}: Validation Summary")
        self.logger.info(f"   • Total checks: {total_checks}")
        self.logger.info(f"   • Passed: {passed_checks}")
        self.logger.info(f"   • Failed: {failed_checks}")
        self.logger.info(f"   • Success rate: {success_rate:.1f}%")
