"""
Logging utilities for pre-training pipeline.
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from src.utils.logger import LoggingConfig, UnifiedLogger


class PreTrainingEventLogger:
    """Event logger for pre-training pipeline steps."""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = UnifiedLogger(config)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger for pre-training events."""
        self.logger.get_logger('PreTrainingEventLogger')
    
    def log_step_start(self, step_name: str, **kwargs):
        """Log the start of a step."""
        self.logger.info(f"🚀 Starting step: {step_name}", **kwargs)
    
    def log_step_complete(self, step_name: str, **kwargs):
        """Log the completion of a step."""
        self.logger.info(f"✅ Completed step: {step_name}", **kwargs)
    
    def log_step_error(self, step_name: str, error: Exception, **kwargs):
        """Log an error in a step."""
        self.logger.error(f"❌ Error in step {step_name}: {str(error)}", **kwargs)
    
    def log_step_warning(self, step_name: str, message: str, **kwargs):
        """Log a warning in a step."""
        self.logger.warning(f"⚠️ Warning in step {step_name}: {message}", **kwargs)
    
    def log_step_info(self, step_name: str, message: str, **kwargs):
        """Log info for a step."""
        self.logger.info(f"ℹ️ {step_name}: {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning."""
        self.logger.log_warning(f"⚠️ {message}")
    
    def info(self, message: str, **kwargs):
        """Log info."""
        self.logger.log_info(f"ℹ️ {message}")
    
    def error(self, message: str, **kwargs):
        """Log an error."""
        self.logger.log_error(f"❌ {message}")


class StepLogContext:
    """Context manager for step logging."""
    
    def __init__(self, event_logger: PreTrainingEventLogger, step_name: str):
        self.event_logger = event_logger
        self.step_name = step_name
    
    def __enter__(self):
        self.event_logger.log_step_start(self.step_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.event_logger.log_step_error(self.step_name, exc_val)
        else:
            self.event_logger.log_step_complete(self.step_name)
        return False


def configure_pre_training_logging() -> LoggingConfig:
    """Configure logging for pre-training pipeline."""
    config = LoggingConfig()
    config.log_level = logging.INFO
    config.console_output = True
    config.file_output = True
    config.log_file = "logs/pre_training.log"
    return config
