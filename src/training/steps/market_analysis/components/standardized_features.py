#!/usr/bin/env python3
"""
Standardized Feature Calculation Module for HMM Clustering

This module provides centralized, consistent feature calculations for the 6 core features
used throughout the HMM regime discovery process. All features are calculated in a single 
place to ensure consistency from regime discovery to merging.

Optimized for ETH 15m timeframe with 4x more bars than 1h.

Core Features (6 only):
- volume_ratio_192m: Current volume / average 192m volume (192 minutes for 15m)
- volatility_20: 20 period rolling volatility (5 * 4 for 15m)
- volatility_12: 12 period rolling volatility (3 * 4 for 15m)  
- momentum_20: 20 period price momentum (5 * 4 for 15m)
- momentum_12: 12 period price momentum (3 * 4 for 15m)
- trend_score: Directional Signal normalized × ADX
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StandardizedFeatureCalculator:
    """
    Centralized feature calculator that ensures consistency across all HMM operations.
    Only calculates the 6 core features specified for 15m timeframe.
    """
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the 6 core standardized features for 15m timeframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with only the 6 core features
        """
        logger.info("Calculating 6 core standardized features for HMM clustering (15m timeframe)...")
        
        features = pd.DataFrame(index=df.index)
        
        # 1. Volume feature: current volume / average 192m volume (192 minutes for 15m)
        volume_192m_avg = df['volume'].rolling(window=192).mean()
        volume_192m_safe = volume_192m_avg.replace(0, np.nan)
        volume_192m_safe = volume_192m_safe.fillna(method='bfill').fillna(1.0)
        features['volume_ratio_192m'] = df['volume'] / volume_192m_safe
        features['volume_ratio_192m'] = features['volume_ratio_192m'].clip(-100, 100)
        
        # 2-3. Volatility features: 20 and 12 period rolling volatility (5*4 and 3*4 for 15m)
        price_returns = df['close'].pct_change().fillna(0)
        features['volatility_20'] = self._vectorbt_rolling_operation(price_returns, "std", 20)
        features['volatility_12'] = self._vectorbt_rolling_operation(price_returns, "std", 12)
        
        # 4-5. Momentum features: 20 and 12 period price momentum (5*4 and 3*4 for 15m)
        features['momentum_20'] = df['close'].pct_change(20)
        features['momentum_12'] = df['close'].pct_change(12)
        
        # 6. Trend feature: Directional Signal normalized × ADX
        # Keep EMA8 and EMA20 as specified (not scaled for timeframe)
        ema_8 = df['close'].ewm(span=8).mean()
        ema_20 = df['close'].ewm(span=20).mean()
        
        # Directional Signal: DS = EMA8 - EMA20
        directional_signal = ema_8 - ema_20
        
        # Calculate ADX (Average Directional Index) with standard 20-period
        features['adx'] = StandardizedFeatureCalculator._calculate_adx(df, period=20)
        
        # Normalize DS using rolling z-score with 20-period window
        ds_mean = self._vectorbt_rolling_operation(directional_signal, "mean", 20)
        ds_std = self._vectorbt_rolling_operation(directional_signal, "std", 20)
        ds_normalized = (directional_signal - ds_mean) / (ds_std + 1e-8)
        
        # Trend Score = DS_normalized × ADX
        features['trend_score'] = ds_normalized * features['adx']
        features['trend_score'] = features['trend_score'].clip(-1000, 1000)  # Prevent extreme values
        
        # Add timestamp if available
        if 'timestamp' in df.columns:
            features['timestamp'] = df['timestamp']
        
        logger.info(f"Calculated 6 core standardized features: {list(features.columns)}")
        
        return features
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Average Directional Index (ADX) with specified period.
        
        Args:
            df: DataFrame with OHLCV data
            period: Period for ADX calculation (default 20)
            
        Returns:
            Series with ADX values
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        # Set negative values to 0
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate smoothed averages
        atr = tr.ewm(span=period).mean()
        dm_plus_smooth = dm_plus.ewm(span=period).mean()
        dm_minus_smooth = dm_minus.ewm(span=period).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr)
        di_minus = 100 * (dm_minus_smooth / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.ewm(span=period).mean()
        
        return adx.fillna(0)
    
    @staticmethod
    def get_primary_features() -> Dict[str, List[str]]:
        """
        Get the 6 core features used for 4D dimension-aware clustering.
        
        Returns:
            Dict mapping dimension names to feature names
        """
        return {
            'volume': ['volume_ratio_192m'],
            'volatility': ['volatility_20', 'volatility_12'],
            'momentum': ['momentum_20', 'momentum_12'],
            'trend': ['trend_score']
        }
    
    @staticmethod
    def validate_features(df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all 6 core features are present and non-null.
        
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

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    """Calculate the 6 core standardized features for 15m timeframe."""
    return StandardizedFeatureCalculator.calculate_all_features(df)


def get_primary_features() -> Dict[str, List[str]]:
    """Get the 6 core features for 4D dimension-aware clustering."""
    return StandardizedFeatureCalculator.get_primary_features()