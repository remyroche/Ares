"""
Data Storage for TAS - ML Common Implementation

This module provides an ml_common-specific wrapper for the common data storage functionality.
It imports and uses the shared data storage implementation from src.utils.nas_tas.
"""

# Import the common data storage implementation
from src.utils.nas_tas.data_storage import (
    StorageFormat, StorageType, StorageConfig, StorageResult, DataStorageManager
)