"""
Negative learning pipeline integration for feature generation.

This module provides integration with negative learning pipelines
for enhanced feature generation and model training.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class NegativeLearningPipelineIntegration:
    """Integration class for negative learning pipelines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the negative learning pipeline integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def generate_negative_features(self, df: pd.DataFrame, 
                                 feature_columns: List[str]) -> pd.DataFrame:
        """Generate negative learning features.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            
        Returns:
            DataFrame with negative learning features
        """
        df_negative = df.copy()
        
        # Create negative correlation features
        for col in feature_columns:
            if col in df.columns:
                df_negative[f"{col}_negative"] = -df[col]
                df_negative[f"{col}_inverse"] = 1 / (df[col] + 1e-8)
        
        return df_negative
    
    def apply_negative_learning(self, df: pd.DataFrame,
                               target_column: str,
                               negative_ratio: float = 0.5) -> pd.DataFrame:
        """Apply negative learning to the dataset.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            negative_ratio: Ratio of negative samples to generate
            
        Returns:
            DataFrame with negative learning applied
        """
        # Generate negative samples
        n_negative = int(len(df) * negative_ratio)
        negative_indices = np.random.choice(df.index, n_negative, replace=False)
        
        df_negative = df.copy()
        df_negative.loc[negative_indices, target_column] = -df_negative.loc[negative_indices, target_column]
        
        return df_negative

class NegativeLearningPipelineIntegrator:
    """Advanced integrator for negative learning pipelines with enhanced features."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the negative learning pipeline integrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.integration = NegativeLearningPipelineIntegration(config)
    
    def integrate_advanced_negative_learning(self, df: pd.DataFrame,
                                          feature_columns: List[str],
                                          target_column: str,
                                          negative_ratio: float = 0.3,
                                          feature_interactions: bool = True) -> pd.DataFrame:
        """Integrate advanced negative learning with feature interactions.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature columns
            target_column: Name of target column
            negative_ratio: Ratio of negative samples to generate
            feature_interactions: Whether to create feature interactions
            
        Returns:
            DataFrame with advanced negative learning integration
        """
        self.logger.info("Starting advanced negative learning integration...")
        
        # Generate negative features
        df_with_negative_features = self.integration.generate_negative_features(df, feature_columns)
        
        # Create feature interactions if requested
        if feature_interactions:
            df_with_negative_features = self._create_feature_interactions(
                df_with_negative_features, feature_columns
            )
        
        # Apply negative learning with custom ratio
        df_with_negative_learning = self.integration.apply_negative_learning(
            df_with_negative_features, target_column, negative_ratio
        )
        
        # Add negative learning metadata
        df_with_negative_learning = self._add_negative_learning_metadata(
            df_with_negative_learning, negative_ratio
        )
        
        self.logger.info("Advanced negative learning integration completed")
        return df_with_negative_learning
    
    def _create_feature_interactions(self, df: pd.DataFrame, 
                                   feature_columns: List[str]) -> pd.DataFrame:
        """Create feature interactions for negative learning."""
        df_interactions = df.copy()
        
        # Create pairwise interactions
        for i, col1 in enumerate(feature_columns):
            for j, col2 in enumerate(feature_columns[i+1:], i+1):
                if col1 in df.columns and col2 in df.columns:
                    df_interactions[f"{col1}_x_{col2}_neg"] = df[col1] * df[col2]
                    df_interactions[f"{col1}_div_{col2}_neg"] = df[col1] / (df[col2] + 1e-8)
        
        return df_interactions
    
    def _add_negative_learning_metadata(self, df: pd.DataFrame, 
                                      negative_ratio: float) -> pd.DataFrame:
        """Add metadata about negative learning process."""
        df_metadata = df.copy()
        df_metadata['negative_learning_applied'] = True
        df_metadata['negative_ratio'] = negative_ratio
        df_metadata['integration_timestamp'] = pd.Timestamp.now()
        
        return df_metadata

def integrate_negative_learning(df: pd.DataFrame,
                               feature_columns: List[str],
                               target_column: str,
                               config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Integrate negative learning into feature generation.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        target_column: Name of target column
        config: Configuration dictionary
        
    Returns:
        DataFrame with negative learning integration
    """
    integration = NegativeLearningPipelineIntegration(config)
    
    # Generate negative features
    df_with_negative_features = integration.generate_negative_features(df, feature_columns)
    
    # Apply negative learning
    df_with_negative_learning = integration.apply_negative_learning(
        df_with_negative_features, target_column
    )
    
    return df_with_negative_learning
