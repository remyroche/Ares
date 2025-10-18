"""
Model preparation utilities for pre-training steps.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def prepare_model_data(df: pd.DataFrame,
                      feature_columns: List[str],
                      target_column: str,
                      test_size: float = 0.2,
                      random_state: int = 42) -> Dict[str, Any]:
    """Prepare data for model training.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature column names
        target_column: Name of target column
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing prepared data
    """
    # Remove rows with missing values
    df_clean = df.dropna()
    
    # Separate features and target
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_columns': feature_columns,
        'target_column': target_column
    }

def split_train_test(df: pd.DataFrame,
                    target_column: str,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        target_column: Name of target column
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    return train_df, test_df
