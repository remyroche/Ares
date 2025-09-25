"""
Data Manager for TAS Backtesting - Training Step Implementation

This module provides a training-specific wrapper for the common data management functionality.
It imports and uses the shared data management implementation from src.utils.nas_tas.
"""

# Import the common data management implementation
from src.utils.nas_tas.data_manager import (
    DataSource, DataConfig, DataResult, BacktestingDataManager
)