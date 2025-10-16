#!/usr/bin/env python3

"""
TPrint Integration with System Logger

This module provides utilities to properly integrate tprint with the system logger,
ensuring all tprint statements are captured in the unified logging system.
"""

from .tprint import configure_tprint_with_system_logger, LogLevel
from .logger import system_logger


def setup_tprint_logging_integration():
    """Setup tprint to integrate with the system logger."""
    try:
        # Configure tprint to use the system logger
        configure_tprint_with_system_logger(
            enable_logging=True,
            enable_file_output=True
        )
        
        # Log the integration setup
        system_logger.info("✅ TPrint integration with system logger configured successfully")
        return True
        
    except Exception as e:
        system_logger.error(f"❌ Failed to configure tprint integration: {e}")
        return False


def setup_tprint_for_component(component_name: str):
    """Setup tprint for a specific component with proper logging integration."""
    try:
        # Configure tprint with system logger integration
        configure_tprint_with_system_logger(
            enable_logging=True,
            enable_file_output=True
        )
        
        # Get component logger
        component_logger = system_logger.getChild(component_name)
        component_logger.info(f"🔧 TPrint configured for component: {component_name}")
        
        return True
        
    except Exception as e:
        system_logger.error(f"❌ Failed to configure tprint for component {component_name}: {e}")
        return False


def verify_tprint_integration():
    """Verify that tprint is properly integrated with the logging system."""
    try:
        from .tprint import tprint, tprint_info, tprint_error
        
        # Test tprint integration
        tprint_info("🧪 Testing tprint integration with system logger")
        
        # Check if the global manager is configured
        from .tprint import _global_manager
        if hasattr(_global_manager, 'logger') and _global_manager.logger:
            system_logger.info("✅ TPrint integration verified successfully")
            return True
        else:
            system_logger.warning("⚠️ TPrint integration not properly configured")
            return False
            
    except Exception as e:
        system_logger.error(f"❌ TPrint integration verification failed: {e}")
        return False


# Auto-setup when module is imported
def _auto_setup_tprint_integration():
    """Automatically setup tprint integration when this module is imported."""
    try:
        setup_tprint_logging_integration()
    except Exception:
        # Silently fail during auto-setup to avoid import errors
        pass


# Auto-setup on import
_auto_setup_tprint_integration()
