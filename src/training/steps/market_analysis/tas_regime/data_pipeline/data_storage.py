"""
Data Storage for TAS - Training Step Implementation

This module provides a training-specific wrapper for the common data storage functionality.
It imports and uses the shared data storage implementation from src.utils.nas_tas.
"""

# Import the common data storage implementation
from src.utils.nas_tas.data_storage import (
    StorageFormat, StorageType, StorageConfig, StorageResult, DataStorageManager
)