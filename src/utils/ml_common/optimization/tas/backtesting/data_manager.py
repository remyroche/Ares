"""
Data Manager for TAS Backtesting - ML Common Implementation

This module provides an ml_common-specific wrapper for the common data management functionality.
It imports and uses the shared data management implementation from src.utils.nas_tas.
"""

# Import the common data management implementation
from src.utils.nas_tas.data_manager import (
    DataSource, DataConfig, DataResult, BacktestingDataManager
)