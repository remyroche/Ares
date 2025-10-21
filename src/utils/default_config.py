"""
Default configuration setup for Ares trading system.
This module sets default environment variables for debug mode and data preview.
"""

import os

def set_default_environment():
    """Set default environment variables if they are not already set."""
    
    # Debug mode - enabled by default
    if 'DEBUG' not in os.environ:
        os.environ['DEBUG'] = 'true'
    
    # Data preview settings - enabled by default with specified values
    if 'ENABLE_DATA_PREVIEW' not in os.environ:
        os.environ['ENABLE_DATA_PREVIEW'] = 'true'
    
    if 'DATA_PREVIEW_MAX_ROWS' not in os.environ:
        os.environ['DATA_PREVIEW_MAX_ROWS'] = '5'
    
    if 'DATA_PREVIEW_MAX_COLS' not in os.environ:
        os.environ['DATA_PREVIEW_MAX_COLS'] = '10'
    
    if 'DATA_PREVIEW_LARGE_THRESHOLD' not in os.environ:
        os.environ['DATA_PREVIEW_LARGE_THRESHOLD'] = '10000'

# Automatically set defaults when this module is imported
set_default_environment()