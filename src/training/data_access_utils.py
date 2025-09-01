# src/training/data_access_utils.py

"""Utility functions for accessing the unified training database across different steps."""

import os
from typing import Any

import numpy as np
import pandas as pd

from src.training.data_manager import UnifiedDataManager
from src.utils.logger import system_logger



def load_training_data(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    split_type: str = "train",
    label_column: str = "tactician_label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Load training data for a specific split.

    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        split_type: Type of split ('train', 'validation', 'test', 'full')
        label_column: Name of the label column

    Returns:
        Tuple of (features_df, labels_series)

    """
    logger = system_logger.getChild("DataAccessUtils")

    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)
        return data_manager.get_features_and_labels(split_type, label_column)
    except Exception as e:
        error_msg = f"Error loading {split_type} data for {symbol} on {exchange}: {e}"
        logger.exception(error_msg)
        raise




def validate_dataset_integrity(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
) -> dict[str, Any]:
    """Validate the integrity of the dataset.

    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name

    Returns:
        Dictionary containing validation results

    """
    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)
        return data_manager.validate_database_integrity()
    except Exception as e:
        logger = system_logger.getChild("DataAccessUtils")
        error_msg = (
            f"Error validating dataset integrity for {symbol} on {exchange}: {e}"
        )
        logger.exception(error_msg)
        return {
            "status": "FAILED",
            "issues": [f"Validation error: {e!s}"],
            "warnings": [],
        }



def check_unified_database_exists(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
) -> bool:
    """Check if the unified database exists and is accessible.

    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name

    Returns:
        True if unified database exists and is accessible

    """
    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)

        # Check if main database file exists
        if not os.path.exists(data_manager.database_file):
            return False

        # Check if metadata file exists
        if not os.path.exists(data_manager.metadata_file):
            return False

        # Try to load a small sample to verify accessibility
        return data_manager.get_metadata()

    except Exception as e:
        logger = system_logger.getChild("DataAccessUtils")
        logger.warning(
            f"Error checking unified database existence for {symbol} on {exchange}: {e}",
        )
        return False




# Convenience functions for common use cases
def get_training_features_and_labels(
    data_dir: str,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Get training features and labels."""
    return load_training_data(data_dir, split_type="train", **kwargs)



def get_test_features_and_labels(
    data_dir: str,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series]:
    """Get test features and labels."""
    return load_training_data(data_dir, split_type="test", **kwargs)

