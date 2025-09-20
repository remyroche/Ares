#!/usr/bin/env python3
"""
Standardized Feature Calculation Module for HMM Clustering

This module provides centralized, consistent feature calculations for volatility,
momentum, and volume features used throughout the HMM regime discovery process.
All features are calculated in a single place to ensure consistency from regime
discovery to merging.

Features:
- Volume: Current volume / average 48h volume
- Volatility: 5 and 3 period rolling volatility
- Momentum: 5 and 3 period price momentum
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StandardizedFeatureCalculator:
    """
    Centralized feature calculator that ensures consistency across all HMM operations.
    """
    
    @staticmethod
    def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standardized volume features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=df.index)
        
        # Primary volume feature: current volume / average 48h volume
        volume_48h_avg = df['volume'].rolling(window=48).mean()
        volume_48h_safe = volume_48h_avg.replace(0, np.nan)
        volume_48h_safe = volume_48h_safe.fillna(method='bfill').fillna(1.0)
        features['volume_ratio_48h'] = df['volume'] / volume_48h_safe
        features['volume_ratio_48h'] = features['volume_ratio_48h'].clip(-100, 100)
        
        # Additional volume features for completeness
        features['volume_change'] = df['volume'].pct_change()
        features['volume_ma_6'] = df['volume'].rolling(6).mean()
        features['volume_std_6'] = df['volume'].rolling(6).std()
        
        # Volume patterns
        features['volume_spike'] = df['volume'] > features['volume_ma_6'] * 2
        features['volume_dry_up'] = df['volume'] < features['volume_ma_6'] * 0.5
        
        logger.debug(f"Calculated {len([col for col in features.columns if 'volume' in col])} volume features")
        return features
    
    @staticmethod
    def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standardized volatility features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=df.index)
        
        # Calculate price returns
        price_returns = df['close'].pct_change().fillna(0)
        
        # Primary volatility features: 5 and 3 period volatility
        features['volatility_5'] = price_returns.rolling(window=5).std()
        features['volatility_3'] = price_returns.rolling(window=3).std()
        
        # Additional volatility features
        features['volatility_1'] = price_returns.rolling(window=1).std()
        features['volatility_2'] = price_returns.rolling(window=2).std()
        features['volatility_6'] = price_returns.rolling(window=6).std()
        
        # EWMA volatility
        features['ewma_volatility_5'] = price_returns.ewm(span=5).std()
        features['ewma_volatility_3'] = price_returns.ewm(span=3).std()
        
        # Volatility ratios (safe division)
        vol_5_safe = features['volatility_5'].replace(0, np.nan)
        vol_5_safe = vol_5_safe.fillna(method='bfill').fillna(1e-6)
        features['volatility_ratio_3_5'] = features['volatility_3'] / vol_5_safe
        features['volatility_ratio_3_5'] = features['volatility_ratio_3_5'].clip(-1000, 1000)
        
        # Volatility momentum and acceleration
        features['volatility_momentum'] = features['volatility_5'] - features['volatility_5'].shift(3)
        features['volatility_acceleration'] = features['volatility_momentum'].diff()
        
        # GARCH-like features
        features['volatility_clustering'] = (price_returns ** 2).rolling(6).mean()
        features['volatility_persistence'] = features['volatility_clustering'].rolling(3).corr(
            features['volatility_clustering'].shift(1)
        )
        
        logger.debug(f"Calculated {len([col for col in features.columns if 'volatility' in col])} volatility features")
        return features
    
    @staticmethod
    def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate standardized momentum features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=df.index)
        
        # Primary momentum features: 5 and 3 period momentum
        features['momentum_5'] = df['close'].pct_change(5)
        features['momentum_3'] = df['close'].pct_change(3)
        
        # Additional momentum features
        features['momentum_1'] = df['close'].pct_change(1)
        features['momentum_2'] = df['close'].pct_change(2)
        features['momentum_6'] = df['close'].pct_change(6)
        
        # Momentum moving averages
        features['momentum_ma_5'] = features['momentum_5'].rolling(3).mean()
        features['momentum_ma_3'] = features['momentum_3'].rolling(3).mean()
        
        # Volume momentum
        features['volume_momentum_5'] = df['volume'].pct_change(5)
        features['volume_momentum_3'] = df['volume'].pct_change(3)
        
        # Momentum ratios (safe division)
        momentum_5_safe = features['momentum_5'].replace(0, np.nan)
        momentum_5_safe = momentum_5_safe.fillna(method='bfill').fillna(1e-6)
        features['momentum_ratio_3_5'] = features['momentum_3'] / momentum_5_safe
        features['momentum_ratio_3_5'] = features['momentum_ratio_3_5'].clip(-1000, 1000)
        
        logger.debug(f"Calculated {len([col for col in features.columns if 'momentum' in col])} momentum features")
        return features
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all standardized features in one call.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features combined
        """
        logger.info("Calculating standardized features for HMM clustering...")
        
        # Calculate each feature group
        volume_features = StandardizedFeatureCalculator.calculate_volume_features(df)
        volatility_features = StandardizedFeatureCalculator.calculate_volatility_features(df)
        momentum_features = StandardizedFeatureCalculator.calculate_momentum_features(df)
        
        # Combine all features
        all_features = pd.concat([
            volume_features,
            volatility_features,
            momentum_features
        ], axis=1)
        
        # Add timestamp if available
        if 'timestamp' in df.columns:
            all_features['timestamp'] = df['timestamp']
        
        logger.info(f"Calculated {len(all_features.columns)} standardized features total")
        logger.info(f"  - Volume features: {len([col for col in all_features.columns if 'volume' in col])}")
        logger.info(f"  - Volatility features: {len([col for col in all_features.columns if 'volatility' in col])}")
        logger.info(f"  - Momentum features: {len([col for col in all_features.columns if 'momentum' in col])}")
        
        return all_features
    
    @staticmethod
    def get_primary_features() -> Dict[str, List[str]]:
        """
        Get the primary features used for dimension-aware clustering.
        
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
        Validate that all required features are present and non-null.
        
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
def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized volume features."""
    return StandardizedFeatureCalculator.calculate_volume_features(df)


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized volatility features."""
    return StandardizedFeatureCalculator.calculate_volatility_features(df)


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate standardized momentum features."""
    return StandardizedFeatureCalculator.calculate_momentum_features(df)


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all standardized features."""
    return StandardizedFeatureCalculator.calculate_all_features(df)


def get_primary_features() -> Dict[str, List[str]]:
    """Get primary features for dimension-aware clustering."""
    return StandardizedFeatureCalculator.get_primary_features()