#!/usr/bin/env python3
"""
NAS Utilities Package

This package provides comprehensive utilities for Neural Architecture Search (NAS)
components including error handling, resource management, threading, performance
optimization, testing, logging, and validation.
"""

# Version information
__version__ = "1.0.0"
__author__ = "NAS Development Team"
__description__ = "Comprehensive utilities for Neural Architecture Search components"

# Package initialization
def initialize_nas_utilities(
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_performance_monitoring: bool = True,
    enable_resource_monitoring: bool = True
) -> None:
    """Initialize NAS utilities with default configuration."""
    try:
        # Setup logging
        from .nas_logging import setup_logging, LogLevel
        log_level_enum = LogLevel[log_level.upper()]
        setup_logging(log_dir=log_dir, level=log_level_enum)
        
        # Start resource monitoring
        if enable_resource_monitoring:
            from .nas_resource_manager import get_resource_manager
            resource_manager = get_resource_manager()
            resource_manager.start_monitoring()
        
        print(f"NAS Utilities initialized successfully (v{__version__})")
        
    except Exception as e:
        print(f"Warning: Failed to initialize NAS utilities: {e}")


def cleanup_nas_utilities() -> None:
    """Clean up NAS utilities resources."""
    try:
        from .nas_logging import cleanup_logging
        from .nas_resource_manager import get_resource_manager
        
        # Cleanup logging
        cleanup_logging()
        
        # Cleanup resources
        resource_manager = get_resource_manager()
        resource_manager.cleanup_all_resources()
        
        print("NAS Utilities cleaned up successfully")
        
    except Exception as e:
        print(f"Warning: Failed to cleanup NAS utilities: {e}")


# Export main functions
__all__ = [
    'initialize_nas_utilities',
    'cleanup_nas_utilities'
]