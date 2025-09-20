#!/usr/bin/env python3
"""
Standardized Feature Calculation Module for HMM Clustering

This module provides centralized, consistent feature calculations for the 5 core features
used throughout the HMM regime discovery process. All features are calculated in a single 
place to ensure consistency from regime discovery to merging.

Core Features (5 only):
- volume_ratio_48h: Current volume / average 48h volume
- volatility_5: 5 period rolling volatility
- volatility_3: 3 period rolling volatility  
- momentum_5: 5 period price momentum
- momentum_3: 3 period price momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StandardizedFeatureCalculator:
    """
    Centralized feature calculator that ensures consistency across all HMM operations.
    Only calculates the 5 core features specified.
    """
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the 5 core standardized features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with only the 5 core features
        """
        logger.info("Calculating 5 core standardized features for HMM clustering...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Volume feature: current volume / average 48h volume
        volume_48h_avg = df['volume'].rolling(window=48).mean()
        volume_48h_safe = volume_48h_avg.replace(0, np.nan)
        volume_48h_safe = volume_48h_safe.fillna(method='bfill').fillna(1.0)
        features['volume_ratio_48h'] = df['volume'] / volume_48h_safe
        features['volume_ratio_48h'] = features['volume_ratio_48h'].clip(-100, 100)
        
        # 2-3. Volatility features: 5 and 3 period rolling volatility
        price_returns = df['close'].pct_change().fillna(0)
        features['volatility_5'] = price_returns.rolling(window=5).std()
        features['volatility_3'] = price_returns.rolling(window=3).std()
        
        # 4-5. Momentum features: 5 and 3 period price momentum
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_3'] = df['close'].pct_change(3)
        
        # Add timestamp if available
        if 'timestamp' in df.columns:
            features['timestamp'] = df['timestamp']
        
        logger.info(f"Calculated 5 core standardized features: {list(features.columns)}")
        
        return features
    
    @staticmethod
    def get_primary_features() -> Dict[str, List[str]]:
        """
        Get the 5 core features used for dimension-aware clustering.
        
        Returns:
            Dict mapping dimension names to feature names
        """
        return {
            'volume': ['volume_ratio_48h'],
            'volatility': ['volatility_5', 'volatility_3'],
            'momentum': ['momentum_5', 'momentum_3']
        }
    
    @staticmethod
    def validate_features(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all 5 core features are present and non-null.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dict with validation results
        """
        primary_features = StandardizedFeatureCalculator.get_primary_features()
        validation_results = {}
        
        for dimension, feature_names in primary_features.items():
            for feature_name in feature_names:
                if feature_name not in df.columns:
                    validation_results[f"{feature_name}_exists"] = False
                    logger.warning(f"Missing required feature: {feature_name}")
                else:
                    validation_results[f"{feature_name}_exists"] = True
                    
                    # Check for excessive nulls
                    null_ratio = df[feature_name].isnull().sum() / len(df)
                    if null_ratio > 0.1:  # More than 10% nulls
                        validation_results[f"{feature_name}_nulls"] = False
                        logger.warning(f"High null ratio in {feature_name}: {null_ratio:.2%}")
                    else:
                        validation_results[f"{feature_name}_nulls"] = True
        
        return validation_results


# Convenience functions for direct import
def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the 5 core standardized features."""
    return StandardizedFeatureCalculator.calculate_all_features(df)


def get_primary_features() -> Dict[str, List[str]]:
    """Get the 5 core features for dimension-aware clustering."""
    return StandardizedFeatureCalculator.get_primary_features()