# src/training/data_access_utils.py

"""
Utility functions for accessing the unified training database across different steps.
"""

import os
from typing import Tuple, Optional, Any, Dict

import pandas as pd
import numpy as np

from src.training.data_manager import UnifiedDataManager
from src.utils.logger import system_logger


def get_data_manager(
    data_dir: str, 
    symbol: str = "ETHUSDT", 
    exchange: str = "BINANCE",
    lookback_days: Optional[int] = None
) -> UnifiedDataManager:
    """
    Get a unified data manager instance.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        lookback_days: Optional lookback period
        
    Returns:
        UnifiedDataManager instance
    """
    return UnifiedDataManager(
        data_dir=data_dir,
        symbol=symbol,
        exchange=exchange,
        lookback_days=lookback_days or 730
    )


def load_training_data(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    split_type: str = "train",
    label_column: str = "tactician_label"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load training data for a specific split.
    
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
        logger.error(f"Error loading {split_type} data: {e}")
        raise


def load_validation_data_for_optimization(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE",
    label_column: str = "tactician_label"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load validation data specifically formatted for hyperparameter optimization.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        label_column: Name of the label column
        
    Returns:
        Tuple of (X_val, y_val) as numpy arrays
    """
    logger = system_logger.getChild("DataAccessUtils")
    
    try:
        X_val, y_val = load_training_data(
            data_dir, symbol, exchange, "validation", label_column
        )
        
        # Convert to numpy arrays and handle missing values
        X_val_np = X_val.fillna(0).values
        y_val_np = y_val.fillna(0).astype(int).values
        
        # Ensure targets are in proper range
        y_val_np = np.clip(y_val_np, -1, 1)
        
        logger.info(f"Loaded validation data: X={X_val_np.shape}, y={y_val_np.shape}")
        logger.info(f"Target distribution: {np.unique(y_val_np, return_counts=True)}")
        
        return X_val_np, y_val_np
        
    except Exception as e:
        logger.error(f"Error loading validation data for optimization: {e}")
        raise


def get_dataset_metadata(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> Dict[str, Any]:
    """
    Get metadata about the dataset.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        Dictionary containing dataset metadata
    """
    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)
        return data_manager.get_metadata()
    except Exception as e:
        logger = system_logger.getChild("DataAccessUtils")
        logger.error(f"Error loading dataset metadata: {e}")
        raise


def validate_dataset_integrity(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> Dict[str, Any]:
    """
    Validate the integrity of the dataset.
    
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
        logger.error(f"Error validating dataset integrity: {e}")
        return {
            "status": "FAILED",
            "issues": [f"Validation error: {str(e)}"],
            "warnings": []
        }


def update_dataset_with_new_features(
    data_dir: str,
    updated_data: pd.DataFrame,
    split_type: str = "full",
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> None:
    """
    Update the dataset with new features or modifications.
    
    Args:
        data_dir: Data directory path
        updated_data: Updated DataFrame with new features
        split_type: Which split to update ('train', 'validation', 'test', 'full')
        symbol: Trading symbol
        exchange: Exchange name
    """
    logger = system_logger.getChild("DataAccessUtils")
    
    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)
        data_manager.update_data_split(split_type, updated_data)
        logger.info(f"Successfully updated {split_type} dataset with new features")
    except Exception as e:
        logger.error(f"Error updating dataset: {e}")
        raise


def check_unified_database_exists(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> bool:
    """
    Check if the unified database exists and is accessible.
    
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
        metadata = data_manager.get_metadata()
        if not metadata:
            return False
        
        return True
        
    except Exception:
        return False


def get_time_splits_info(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> Dict[str, Any]:
    """
    Get information about the time-based data splits.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        Dictionary containing split information
    """
    try:
        metadata = get_dataset_metadata(data_dir, symbol, exchange)
        return metadata.get("splits", {})
    except Exception as e:
        logger = system_logger.getChild("DataAccessUtils")
        logger.error(f"Error getting time splits info: {e}")
        return {}


def ensure_temporal_consistency(
    data_dir: str,
    symbol: str = "ETHUSDT",
    exchange: str = "BINANCE"
) -> bool:
    """
    Ensure that the temporal ordering is maintained across all splits.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        
    Returns:
        True if temporal consistency is maintained
    """
    logger = system_logger.getChild("DataAccessUtils")
    
    try:
        data_manager = get_data_manager(data_dir, symbol, exchange)
        validation_results = data_manager.validate_database_integrity()
        
        # Check for temporal ordering issues
        temporal_issues = [
            issue for issue in validation_results.get("issues", [])
            if "temporal" in issue.lower()
        ]
        
        if temporal_issues:
            logger.error(f"Temporal consistency issues found: {temporal_issues}")
            return False
        
        logger.info("✅ Temporal consistency verified")
        return True
        
    except Exception as e:
        logger.error(f"Error checking temporal consistency: {e}")
        return False


# Convenience functions for common use cases
def get_training_features_and_labels(data_dir: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Get training features and labels."""
    return load_training_data(data_dir, split_type="train", **kwargs)


def get_validation_features_and_labels(data_dir: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Get validation features and labels."""
    return load_training_data(data_dir, split_type="validation", **kwargs)


def get_test_features_and_labels(data_dir: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
    """Get test features and labels."""
    return load_training_data(data_dir, split_type="test", **kwargs)


def get_full_dataset(data_dir: str, **kwargs) -> pd.DataFrame:
    """Get the full dataset."""
    data_manager = get_data_manager(data_dir, **kwargs)
    return data_manager.load_data_split("full")