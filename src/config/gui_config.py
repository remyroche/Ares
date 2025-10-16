"""
GUI Configuration Utilities

This module provides utilities for the GUI to access centralized pipeline mode configurations.
"""

from .pipeline_modes import get_mode_summary, get_mode_config

def get_gui_mode_configs():
    """Get mode configurations formatted for GUI display."""
    return get_mode_summary()

def get_gui_mode_lookback_days(mode: str) -> int:
    """Get lookback days for GUI mode selection."""
    return get_mode_config(mode).lookback_days

def get_gui_mode_description(mode: str) -> str:
    """Get description for GUI mode selection."""
    return get_mode_config(mode).description

def get_gui_mode_estimated_duration(mode: str) -> int:
    """Get estimated duration for GUI mode selection."""
    return get_mode_config(mode).estimated_duration_minutes
