"""
Training Reports Module

This module provides centralized reporting utilities and enhanced reporting
capabilities for training steps.
"""

# Re-export functions from the main reports module
try:
    from .reports import (
        save_training_report,
        CentralizedReportManager,
        get_report_path,
        list_reports
    )
except ImportError as e:
    # Fallback if the main reports module is not available
    print(f"Warning: Main reports module not available: {e}")

    def save_training_report(*args, **kwargs):
        """Fallback function when main reports module is not available."""
        print("Warning: Main reports module not available, using fallback")
        return None

    def get_report_path(*args, **kwargs):
        """Fallback function when main reports module is not available."""
        return None

    def list_reports(*args, **kwargs):
        """Fallback function when main reports module is not available."""
        return {}

    class CentralizedReportManager:
        """Fallback class when main reports module is not available."""
        def __init__(self, *args, **kwargs):
            pass

        def save_report(self, *args, **kwargs):
            return None

# Export all components
__all__ = [
    'save_training_report',
    'CentralizedReportManager',
    'get_report_path',
    'list_reports'
]
