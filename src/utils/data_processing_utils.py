"""
Data Processing Utilities Module

This module provides utilities for data processing and transformation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessingUtils:
    """Utilities for data processing and transformation."""

    def __init__(self):
        """Initialize data processing utilities."""
        self.logger = logger

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing NaN values and duplicates."""
        try:
            # Remove duplicate rows
            cleaned = data.drop_duplicates()

            # Remove rows with all NaN values
            cleaned = cleaned.dropna(how='all')

            return cleaned
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            return data

    def normalize_data(self, data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Normalize data using specified method."""
        try:
            if method == 'standard':
                # Standard normalization (z-score)
                return (data - data.mean()) / data.std()
            elif method == 'minmax':
                # Min-max normalization
                return (data - data.min()) / (data.max() - data.min())
            else:
                logger.warning(f"Unknown normalization method: {method}")
                return data
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}")
            return data

    def handle_outliers(self, data: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers using specified method."""
        try:
            if method == 'iqr':
                # IQR method
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                # Replace outliers with bounds
                cleaned = data.clip(lower=lower_bound, upper=upper_bound, axis=1)
                return cleaned
            else:
                logger.warning(f"Unknown outlier handling method: {method}")
                return data
        except Exception as e:
            logger.error(f"Failed to handle outliers: {e}")
            return data

    def resample_data(self, data: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
        """Resample time series data."""
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                return data.resample(freq).mean()
            else:
                logger.warning("Data does not have DatetimeIndex, cannot resample")
                return data
        except Exception as e:
            logger.error(f"Failed to resample data: {e}")
            return data

    def validate_data_integrity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data integrity and return summary."""
        try:
            summary = {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': data.dtypes.to_dict(),
                'null_counts': data.isnull().sum().to_dict(),
                'duplicate_rows': data.duplicated().sum(),
                'memory_usage': data.memory_usage(deep=True).sum()
            }

            # Check for issues
            issues = []
            if data.isnull().any().any():
                issues.append("Data contains null values")
            if data.duplicated().any():
                issues.append("Data contains duplicate rows")
            if data.empty:
                issues.append("Data is empty")

            summary['issues'] = issues
            summary['is_valid'] = len(issues) == 0

            return summary
        except Exception as e:
            logger.error(f"Failed to validate data integrity: {e}")
            return {'error': str(e), 'is_valid': False}
