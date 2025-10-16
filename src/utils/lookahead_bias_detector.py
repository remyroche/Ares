"""
Lookahead bias detection and prevention mechanisms.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LookaheadBiasError(Exception):
    """Exception raised when lookahead bias is detected."""
    pass

class LookaheadBiasDetector:
    """Detects and prevents lookahead bias in trading algorithms."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.detection_log: List[Dict[str, Any]] = []
        self.current_timestamp: Optional[datetime] = None

    def set_current_timestamp(self, timestamp: datetime) -> None:
        """Set the current timestamp for bias detection."""
        self.current_timestamp = timestamp

    def validate_dataframe_timestamps(self, df: pd.DataFrame,
                                    timestamp_column: str = 'timestamp') -> bool:
        """Validate that all timestamps in a DataFrame are not in the future."""
        if self.current_timestamp is None:
            return True

        if timestamp_column not in df.columns:
            return True

        future_timestamps = df[df[timestamp_column] > self.current_timestamp]

        if len(future_timestamps) > 0:
            error_msg = f"Lookahead bias detected! DataFrame contains {len(future_timestamps)} future timestamps"

            if self.strict_mode:
                logger.error(error_msg)
                raise LookaheadBiasError(error_msg)
            else:
                logger.warning(error_msg)

            return False

        return True

# Global detector instance
_global_detector = LookaheadBiasDetector()

def get_global_detector() -> LookaheadBiasDetector:
    """Get the global lookahead bias detector."""
    return _global_detector

def validate_no_future_data(df: pd.DataFrame,
                           timestamp_column: str = 'timestamp',
                           current_timestamp: Optional[datetime] = None) -> pd.DataFrame:
    """Validate and filter out future data from a DataFrame."""
    detector = get_global_detector()

    if current_timestamp is None:
        current_timestamp = detector.current_timestamp

    if current_timestamp is None:
        return df

    if timestamp_column not in df.columns:
        return df

    valid_data = df[df[timestamp_column] <= current_timestamp].copy()

    if len(valid_data) < len(df):
        removed_count = len(df) - len(valid_data)
        logger.info(f"Removed {removed_count} future data points from DataFrame")

    return valid_data
