"""
Backward compatibility layer for features_common.

This module ensures that all existing code continues to work
while providing optional enhanced logging and error handling.
"""

import logging
from typing import Any, Dict, Optional, Union, Callable
from functools import wraps

# Import the enhanced components
from .transforms.base_scaler import BaseScaler as EnhancedBaseScaler
from .mixins import (
    OptimizationMixin, PerformanceMixin, VectorBTMixin,
    ValidationMixin, CachingMixin, MonitoringMixin
)
from .logging_enhancements import enable_verbose_logging, get_logging_enhancements

logger = logging.getLogger(__name__)

class BackwardCompatibleBaseScaler(EnhancedBaseScaler):
    """
    Backward-compatible BaseScaler that maintains the original interface
    while providing optional enhanced logging and error handling.
    """

    def __init__(self, use_vectorbt: bool = True, enable_gpu: bool = True,
                 vectorbt_threshold: int = 100, use_optimizer: bool = True,
                 use_unified_manager: bool = True, enable_verbose_logging: bool = True,
                 enable_hardware_optimization: bool = True, **kwargs):
        """
        Initialize the backward-compatible scaler with enhanced features enabled by default.

        Args:
            use_vectorbt: Whether to use VectorBT optimizations (default: True)
            enable_gpu: Whether to enable GPU acceleration (default: True)
            vectorbt_threshold: Minimum data size for VectorBT optimization (default: 100)
            use_optimizer: Whether to use VectorBTRollingOptimizer (default: True)
            use_unified_manager: Whether to use UnifiedVectorizationManager (default: True)
            enable_verbose_logging: Whether to enable verbose logging (default: True)
            enable_hardware_optimization: Whether to enable hardware optimization (default: True)
            **kwargs: Additional configuration parameters
        """
        # Store verbose logging preference
        self.enable_verbose_logging = enable_verbose_logging

        # Initialize the enhanced base class with hardware optimization enabled by default
        super().__init__(
            use_vectorbt=use_vectorbt,
            enable_gpu=enable_gpu,
            vectorbt_threshold=vectorbt_threshold,
            use_optimizer=use_optimizer,
            use_unified_manager=use_unified_manager,
            enable_hardware_optimization=enable_hardware_optimization,
            **kwargs
        )

        # Set up logging enhancements if enabled
        if enable_verbose_logging:
            self._setup_verbose_logging()

    def _setup_verbose_logging(self):
        """Set up verbose logging for this instance."""
        logging_enhancements = get_logging_enhancements()
        logging_enhancements.enable_verbose_logging_for_instance(self, True)

    def fit_transform(self, data, *args, **kwargs):
        """
        Backward-compatible fit_transform with optional verbose logging.

        This method maintains the exact same interface as the original
        while providing optional enhanced logging.
        """
        if self.enable_verbose_logging:
            from .logging_enhancements import log_operation, log_success, log_failure
            log_operation("fit_transform", "BaseScaler", data_length=len(data))

        try:
            result = super().fit_transform(data, *args, **kwargs)

            if self.enable_verbose_logging:
                from .logging_enhancements import log_success
                log_success("fit_transform", "BaseScaler", result_shape=result.shape)

            return result

        except Exception as e:
            if self.enable_verbose_logging:
                from .logging_enhancements import log_failure
                log_failure("fit_transform", "BaseScaler", e)
            raise

    def transform(self, data, *args, **kwargs):
        """
        Backward-compatible transform with optional verbose logging.

        This method maintains the exact same interface as the original
        while providing optional enhanced logging.
        """
        if self.enable_verbose_logging:
            from .logging_enhancements import log_operation, log_success, log_failure
            log_operation("transform", "BaseScaler", data_length=len(data))

        try:
            result = super().transform(data, *args, **kwargs)

            if self.enable_verbose_logging:
                from .logging_enhancements import log_success
                log_success("transform", "BaseScaler", result_shape=result.shape)

            return result

        except Exception as e:
            if self.enable_verbose_logging:
                from .logging_enhancements import log_failure
                log_failure("transform", "BaseScaler", e)
            raise

# Create backward-compatible aliases
BaseScaler = BackwardCompatibleBaseScaler

def create_backward_compatible_scaler(scaler_class, enable_verbose_logging: bool = False, **kwargs):
    """
    Create a backward-compatible scaler instance.

    Args:
        scaler_class: The scaler class to instantiate
        enable_verbose_logging: Whether to enable verbose logging
        **kwargs: Additional arguments for the scaler

    Returns:
        Backward-compatible scaler instance
    """
    if enable_verbose_logging:
        kwargs['enable_verbose_logging'] = True

    return scaler_class(**kwargs)

def enable_enhanced_logging(enable: bool = True):
    """
    Enable or disable enhanced logging globally.

    Args:
        enable: Whether to enable enhanced logging
    """
    enable_verbose_logging(enable)

def create_enhanced_scaler(method: str = 'zscore', enable_verbose_logging: bool = True, 
                          enable_hardware_optimization: bool = True, enable_gpu: bool = True, **kwargs):
    """
    Create an enhanced scaler with all optimizations enabled by default.

    Args:
        method: Scaling method ('zscore', 'minmax', 'robust', 'quantile')
        enable_verbose_logging: Whether to enable verbose logging (default: True)
        enable_hardware_optimization: Whether to enable hardware optimization (default: True)
        enable_gpu: Whether to enable GPU acceleration (default: True)
        **kwargs: Additional arguments for the scaler

    Returns:
        Enhanced scaler instance
    """
    from .factories import create_optimized_scaler

    # Create the enhanced scaler with all optimizations enabled by default
    scaler = create_optimized_scaler(
        method=method, 
        enable_hardware_optimization=enable_hardware_optimization,
        enable_gpu=enable_gpu,
        **kwargs
    )

    # Enable verbose logging by default
    if enable_verbose_logging:
        scaler.enable_verbose_logging = True
        scaler._setup_verbose_logging()

    return scaler

# Backward compatibility functions
def get_original_interface():
    """
    Get the original interface without enhanced logging.

    Returns:
        Dictionary containing the original interface components
    """
    return {
        'BaseScaler': BackwardCompatibleBaseScaler,
        'create_scaler': create_enhanced_scaler,
        'enable_logging': enable_enhanced_logging
    }

def ensure_backward_compatibility():
    """
    Ensure backward compatibility by setting up the original interface.

    This function should be called to ensure that existing code
    continues to work without modification.
    """
    # Enable backward compatibility mode
    import sys

    # Add backward compatibility aliases to the module
    current_module = sys.modules[__name__]

    # Ensure the original BaseScaler is available
    if not hasattr(current_module, 'BaseScaler'):
        current_module.BaseScaler = BackwardCompatibleBaseScaler

    logger.info("Backward compatibility ensured for features_common")

# Ensure backward compatibility on import
ensure_backward_compatibility()
