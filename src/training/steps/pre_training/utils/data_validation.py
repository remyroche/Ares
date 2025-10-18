"""
Data validation utilities for pre-training steps.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def validate_training_data(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 100) -> bool:
    """Validate training data structure and content.
    
    Args:
        df: DataFrame containing training data
        required_columns: List of required columns
        min_rows: Minimum number of rows required
        
    Returns:
        True if data is valid, False otherwise
    """
    if df is None or df.empty:
        logger.error("DataFrame is None or empty")
        return False
    
    if len(df) < min_rows:
        logger.error(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
        return False
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
    
    return True

def validate_features(df: pd.DataFrame, 
                     feature_columns: List[str],
                     target_column: str) -> Dict[str, Any]:
    """Validate feature data quality.
    
    Args:
        df: DataFrame containing features
        feature_columns: List of feature column names
        target_column: Name of target column
        
    Returns:
        Dictionary containing validation results
    """
    results = {
        'valid': True,
        'issues': [],
        'feature_stats': {}
    }
    
    # Check for missing features
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        results['valid'] = False
        results['issues'].append(f"Missing features: {missing_features}")
    
    # Check for target column
    if target_column not in df.columns:
        results['valid'] = False
        results['issues'].append(f"Missing target column: {target_column}")
    
    # Check for infinite values
    for col in feature_columns:
        if col in df.columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                results['issues'].append(f"Column {col} has {inf_count} infinite values")
                results['feature_stats'][col] = {'infinite_values': inf_count}
    
    return results
