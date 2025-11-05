"""
Rank Normalization for Robust Feature Scaling

This module implements various normalization techniques including
cross-sectional rank normalization to reduce outliers.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import warnings
from scipy import stats

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


@dataclass
class NormalizationConfig:
    """Configuration for feature normalization."""
    method: str = 'rank'  # 'rank', 'zscore', 'minmax', 'robust'
    axis: int = 0  # 0 for cross-sectional, 1 for time-series
    handle_missing: str = 'interpolate'  # 'interpolate', 'forward_fill', 'backward_fill'
    clip_outliers: bool = True
    outlier_threshold: float = 3.0  # For robust scaling
    
    # Rank-specific settings
    rank_method: str = 'average'  # 'average', 'min', 'max', 'dense'
    ascending: bool = True
    
    # Z-score specific settings
    robust_zscore: bool = True  # Use median and MAD instead of mean and std


class RankNormalizer:
    """
    Feature normalizer using multiple normalization techniques.
    
    Implements rank normalization, z-score scaling, min-max scaling,
    and robust scaling to handle outliers and improve clustering.
    """
    
    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize rank normalizer.
        
        Args:
            config: Configuration for normalization
        """
        self.config = config or NormalizationConfig()
        
        tprint_info(f"🔧 Initialized Rank Normalizer (method: {self.config.method})")
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using configured method.
        
        Args:
            features: DataFrame with features to normalize
            
        Returns:
            DataFrame with normalized features
        """
        tprint_info(f"🔍 Normalizing features using {self.config.method} method")
        
        try:
            if self.config.method == 'rank':
                normalized_features = self._rank_normalization(features)
            elif self.config.method == 'zscore':
                normalized_features = self._zscore_normalization(features)
            elif self.config.method == 'minmax':
                normalized_features = self._minmax_normalization(features)
            elif self.config.method == 'robust':
                normalized_features = self._robust_normalization(features)
            else:
                raise ValueError(f"Unknown normalization method: {self.config.method}")
            
            # Handle missing values
            normalized_features = self._handle_missing_values(normalized_features)
            
            # Clip outliers if requested
            if self.config.clip_outliers:
                normalized_features = self._clip_outliers(normalized_features)
            
            tprint_success(f"✅ Feature normalization complete: {normalized_features.shape}")
            return normalized_features
            
        except Exception as e:
            tprint_error(f"❌ Feature normalization failed: {e}")
            raise
    
    def _rank_normalization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-sectional rank normalization."""
        tprint_info(f"📊 Applying rank normalization (method: {self.config.rank_method}, axis: {self.config.axis})")
        
        normalized_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        if self.config.axis == 0:  # Cross-sectional normalization
            tprint_info("📈 Using cross-sectional rank normalization")
            for col in features.columns:
                if self.config.rank_method == 'average':
                    # Handle ties by averaging ranks
                    ranks = features[col].rank(method='average', ascending=self.config.ascending)
                elif self.config.rank_method == 'min':
                    ranks = features[col].rank(method='min', ascending=self.config.ascending)
                elif self.config.rank_method == 'max':
                    ranks = features[col].rank(method='max', ascending=self.config.ascending)
                elif self.config.rank_method == 'dense':
                    ranks = features[col].rank(method='dense', ascending=self.config.ascending)
                else:
                    ranks = features[col].rank(method='average', ascending=self.config.ascending)
                
                # Normalize ranks to [0, 1]
                valid_ranks = ranks.dropna()
                if len(valid_ranks) > 0:
                    min_rank = valid_ranks.min()
                    max_rank = valid_ranks.max()
                    if max_rank > min_rank:
                        normalized_features[col] = (ranks - min_rank) / (max_rank - min_rank)
                    else:
                        normalized_features[col] = 0.5  # All ranks are the same
                else:
                    normalized_features[col] = 0.5  # No valid ranks
        
        else:  # Time-series normalization
            tprint_info("📈 Using time-series rank normalization")
            for col in features.columns:
                # Use rolling window for time-series rank normalization
                window_size = max(10, len(features) // 10)  # Adaptive window
                tprint_info(f"📊 Using adaptive window size: {window_size}")
                ranks = features[col].rolling(window=window_size, min_periods=1).rank(
                    method=self.config.rank_method, ascending=self.config.ascending
                )
                
                # Normalize ranks to [0, 1]
                valid_ranks = ranks.dropna()
                if len(valid_ranks) > 0:
                    min_rank = valid_ranks.min()
                    max_rank = valid_ranks.max()
                    if max_rank > min_rank:
                        normalized_features[col] = (ranks - min_rank) / (max_rank - min_rank)
                    else:
                        normalized_features[col] = 0.5
                else:
                    normalized_features[col] = 0.5
        
        tprint_success("✅ Rank normalization complete")
        return normalized_features
    
    def _zscore_normalization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization."""
        tprint_info(f"📊 Applying z-score normalization (robust: {self.config.robust_zscore})")
        
        normalized_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        for col in features.columns:
            if self.config.robust_zscore:
                # Use median and MAD for robust z-score
                tprint_info(f"📈 Applying robust z-score to {col}")
                median = features[col].median()
                mad = (features[col] - median).abs().median()
                if mad > 0:
                    normalized_features[col] = (features[col] - median) / mad
                else:
                    normalized_features[col] = 0.0
            else:
                # Use mean and std for standard z-score
                tprint_info(f"📈 Applying standard z-score to {col}")
                mean = features[col].mean()
                std = features[col].std()
                if std > 0:
                    normalized_features[col] = (features[col] - mean) / std
                else:
                    normalized_features[col] = 0.0
        
        tprint_success("✅ Z-score normalization complete")
        return normalized_features
    
    def _minmax_normalization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply min-max normalization."""
        tprint_info("📊 Applying min-max normalization")
        
        normalized_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        for col in features.columns:
            min_val = features[col].min()
            max_val = features[col].max()
            
            if max_val > min_val:
                normalized_features[col] = (features[col] - min_val) / (max_val - min_val)
            else:
                normalized_features[col] = 0.5  # All values are the same
        
        tprint_success("✅ Min-max normalization complete")
        return normalized_features
    
    def _robust_normalization(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply robust scaling using quantiles."""
        tprint_info("📊 Applying robust scaling using quantiles")
        
        normalized_features = pd.DataFrame(index=features.index, columns=features.columns)
        
        for col in features.columns:
            # Use 25th and 75th percentiles for robust scaling
            q25 = features[col].quantile(0.25)
            q75 = features[col].quantile(0.75)
            iqr = q75 - q25
            
            if iqr > 0:
                # Robust scaling: (x - median) / IQR
                median = features[col].median()
                normalized_features[col] = (features[col] - median) / iqr
            else:
                normalized_features[col] = 0.0
        
        tprint_success("✅ Robust normalization complete")
        return normalized_features
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        tprint_info(f"🔍 Handling missing values (method: {self.config.handle_missing})")
        
        if self.config.handle_missing == 'interpolate':
            # Linear interpolation
            tprint_info("📈 Applying linear interpolation")
            features = features.interpolate(method='linear')
        elif self.config.handle_missing == 'forward_fill':
            # Forward fill
            tprint_info("📈 Applying forward fill")
            features = features.fillna(method='ffill')
        elif self.config.handle_missing == 'backward_fill':
            # Backward fill
            tprint_info("📈 Applying backward fill")
            features = features.fillna(method='bfill')
        
        tprint_success("✅ Missing value handling complete")
        return features
    
    def _clip_outliers(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clip outliers using threshold."""
        tprint_info(f"🔍 Clipping outliers (method: {self.config.method})")
        
        if self.config.method == 'zscore' or self.config.method == 'robust':
            # Clip extreme values for z-score and robust methods
            tprint_info("📈 Applying percentile-based clipping")
            for col in features.columns:
                # Use percentile-based clipping
                lower_bound = features[col].quantile(0.01)
                upper_bound = features[col].quantile(0.99)
                features[col] = features[col].clip(lower_bound, upper_bound)
        
        elif self.config.method == 'rank':
            # Rank normalization already handles outliers naturally
            tprint_info("⏭️ Rank normalization already handles outliers naturally")
            pass
        
        elif self.config.method == 'minmax':
            # Min-max normalization already bounded to [0, 1]
            tprint_info("⏭️ Min-max normalization already bounded to [0, 1]")
            pass
        
        tprint_success("✅ Outlier clipping complete")
        return features


def create_rank_normalizer(
    method: str = 'rank',
    axis: int = 0,
    handle_missing: str = 'interpolate',
    clip_outliers: bool = True,
    rank_method: str = 'average',
    robust_zscore: bool = True
) -> RankNormalizer:
    """
    Factory function to create rank normalizer.
    
    Args:
        method: Normalization method
        axis: Axis for normalization (0 for cross-sectional, 1 for time-series)
        handle_missing: How to handle missing values
        clip_outliers: Whether to clip outliers
        rank_method: Rank calculation method
        robust_zscore: Use robust z-score calculation
        
    Returns:
        RankNormalizer instance
    """
    tprint_info("🏭 Creating Rank Normalizer with factory function")
    
    config = NormalizationConfig(
        method=method,
        axis=axis,
        handle_missing=handle_missing,
        clip_outliers=clip_outliers,
        rank_method=rank_method,
        robust_zscore=robust_zscore
    )
    
    tprint_info(f"📊 Configuration: method={method}, axis={axis}, handle_missing={handle_missing}")
    tprint_info(f"📊 Configuration: clip_outliers={clip_outliers}, rank_method={rank_method}")
    
    normalizer = RankNormalizer(config)
    tprint_success("✅ Rank Normalizer created successfully")
    return normalizer