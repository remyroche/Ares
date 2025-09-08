"""Step 10 Logging Setup.

This module provides logging configuration and utilities for the
unified regime intelligence system.
"""

import logging
from typing import Optional

STEP10_LOGGER_NAME = "Step10_UnifiedRegimeIntelligence"

def setup_step10_logger(level: Optional[str] = None) -> logging.Logger:
    """Setup and configure Step 10 logger.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(STEP10_LOGGER_NAME)

    # Set level if provided
    if level:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        logger.setLevel(level_map.get(level.upper(), logging.INFO))

    # Add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure propagation is enabled to inherit from parent loggers
    logger.propagate = True

    return logger

def get_step10_logger() -> logging.Logger:
    """Get the Step 10 logger instance.

    Returns:
        Step 10 logger instance
    """
    return logging.getLogger(STEP10_LOGGER_NAME)

def log_step10_initialization(config: dict) -> None:
    """Log Step 10 initialization details.

    Args:
        config: Configuration dictionary
    """
    logger = get_step10_logger()

    logger.info("🚀 Step 10 Unified Regime Intelligence - Initialization")
    logger.info(f"   Symbol: {config.get('symbol', 'N/A')}")
    logger.info(f"   Exchange: {config.get('exchange', 'N/A')}")
    logger.info(f"   Timeframes: {config.get('timeframes', [])}")
    logger.info(f"   Model Dim: {config.get('d_model', 'N/A')}")
    logger.info(f"   Sequence Length: {config.get('sequence_length', 'N/A')}")

    # Log enhancement features
    enhancement = config.get('enhancement', {})
    if enhancement.get('hpo_enabled'):
        logger.info("   HPO: ✅ Enabled")
    else:
        logger.info("   HPO: ❌ Disabled")

    if enhancement.get('architecture_optimization_enabled'):
        logger.info("   Architecture Optimization: ✅ Enabled")
    else:
        logger.info("   Architecture Optimization: ❌ Disabled")

def log_step10_training_start(config: dict) -> None:
    """Log Step 10 training start.

    Args:
        config: Training configuration
    """
    logger = get_step10_logger()

    logger.info("🚀 Step 10 Training Started")
    logger.info(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
    logger.info(f"   Batch Size: {config.get('batch_size', 'N/A')}")
    logger.info(f"   Epochs: {config.get('epochs', 'N/A')}")
    logger.info(f"   Validation Split: {config.get('validation_split', 'N/A')}")

def log_step10_training_complete(metrics: dict) -> None:
    """Log Step 10 training completion.

    Args:
        metrics: Training metrics
    """
    logger = get_step10_logger()

    logger.info("✅ Step 10 Training Completed")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"   {key}: {value:.4f}")
        else:
            logger.info(f"   {key}: {value}")

def log_step10_error(error: Exception, context: str = "") -> None:
    """Log Step 10 errors with context.

    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger = get_step10_logger()

    error_msg = f"❌ Step 10 Error"
    if context:
        error_msg += f" ({context})"
    error_msg += f": {str(error)}"

    logger.error(error_msg)
    logger.exception("Full traceback:")
