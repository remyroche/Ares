"""
Temporal Feature Extraction with Anti-Leakage Safeguards

This module implements rolling statistical features with proper temporal handling
to avoid look-ahead bias in regime detection.
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
class TemporalFeatureConfig:
    """Configuration for temporal feature extraction."""
    windows: List[int] = None
    shift_periods: int = 1
    include_momentum: bool = True
    include_volatility: bool = True
    include_trend: bool = True
    include_reversal: bool = True
    include_volume_features: bool = True
    
    def __post_init__(self):
        if self.windows is None:
            self.windows = [5, 10, 20]


class TemporalFeatureExtractor:
    """
    Temporal feature extractor with anti-leakage safeguards.
    
    This class implements rolling statistical features while ensuring
    no look-ahead bias through proper temporal handling.
    """
    
    def __init__(self, 
                 windows: Optional[List[int]] = None,
                 shift_periods: int = 1):
        """
        Initialize temporal feature extractor.
        
        Args:
            windows: List of rolling windows
            shift_periods: Number of periods to shift for anti-leakage
        """
        self.config = TemporalFeatureConfig(
            windows=windows,
            shift_periods=shift_periods
        )
        
        tprint_info(f"🔧 Initialized Temporal Feature Extractor (shift: {shift_periods})")
    
    def extract_rolling_features(self, 
                             price_data: pd.DataFrame,
                             volume_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract rolling features with anti-leakage safeguards.
        
        Args:
            price_data: DataFrame with OHLC price data
            volume_data: Optional volume data
            
        Returns:
            DataFrame with rolling features
        """
        tprint_info("🔍 Extracting rolling features with anti-leakage safeguards")
        
        try:
            # Initialize features DataFrame
            features = pd.DataFrame(index=price_data.index)
            
            # Extract price-based features
            price_features = self._extract_price_features(price_data)
            features = pd.concat([features, price_features], axis=1)
            
            # Extract volume-based features
            if volume_data is not None and self.config.include_volume_features:
                volume_features = self._extract_volume_features(volume_data)
                features = pd.concat([features, volume_features], axis=1)
            
            # Apply anti-leakage shifts
            features = self._apply_anti_leakage_shifts(features)
            
            # Clean features
            features = self._clean_features(features)
            
            tprint_success(f"✅ Rolling feature extraction complete: {features.shape[1]} features")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Rolling feature extraction failed: {e}")
            raise
    
    def _extract_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract rolling price-based features."""
        tprint_info("📈 Extracting rolling price-based features")
        
        close_prices = price_data['close']
        high_prices = price_data['high']
        low_prices = price_data['low']
        open_prices = price_data['open']
        
        features = pd.DataFrame(index=price_data.index)
        
        tprint_info(f"📊 Using windows: {self.config.windows}")
        for window in self.config.windows:
            tprint_info(f"🔄 Calculating {window}-period price features")
            
            # Basic rolling statistics
            rolling_mean = close_prices.rolling(window).mean()
            rolling_std = close_prices.rolling(window).std()
            rolling_var = close_prices.rolling(window).var()
            rolling_min = close_prices.rolling(window).min()
            rolling_max = close_prices.rolling(window).max()
            
            # Z-score (distance from mean in standard deviations)
            rolling_z = (close_prices - rolling_mean) / rolling_std
            
            # Price range features
            rolling_range = high_prices.rolling(window).max() - low_prices.rolling(window).min()
            rolling_hl_ratio = (high_prices.rolling(window).max() - low_prices.rolling(window).min()) / close_prices
            
            # Trend features
            rolling_slope = self._calculate_rolling_slope(close_prices, window)
            rolling_acceleration = self._calculate_rolling_acceleration(close_prices, window)
            
            # Momentum features
            rolling_momentum = close_prices / close_prices.rolling(window).mean() - 1
            rolling_rate_of_change = close_prices.pct_change(window)
            
            # Reversal features
            rolling_reversal = self._calculate_reversal_signals(close_prices, window)
            
            # Volatility features
            rolling_volatility = close_prices.rolling(window).std()
            rolling_vol_ratio = rolling_volatility / rolling_volatility.rolling(window*2).mean()
            
            # Add features with window suffix
            features[f'close_mean_{window}'] = rolling_mean
            features[f'close_std_{window}'] = rolling_std
            features[f'close_var_{window}'] = rolling_var
            features[f'close_min_{window}'] = rolling_min
            features[f'close_max_{window}'] = rolling_max
            features[f'close_z_{window}'] = rolling_z
            features[f'price_range_{window}'] = rolling_range
            features[f'hl_ratio_{window}'] = rolling_hl_ratio
            features[f'price_slope_{window}'] = rolling_slope
            features[f'price_acceleration_{window}'] = rolling_acceleration
            features[f'price_momentum_{window}'] = rolling_momentum
            features[f'price_roc_{window}'] = rolling_rate_of_change
            features[f'price_reversal_{window}'] = rolling_reversal
            features[f'price_volatility_{window}'] = rolling_volatility
            features[f'vol_ratio_{window}'] = rolling_vol_ratio
        
        tprint_success("✅ Price features extracted successfully")
        return features.add_prefix('price_')
    
    def _extract_volume_features(self, volume_data: pd.DataFrame) -> pd.DataFrame:
        """Extract rolling volume-based features."""
        # Handle different volume column names
        if 'volume' in volume_data.columns:
            volume = volume_data['volume']
        else:
            # Use first column if 'volume' not found
            volume = volume_data.iloc[:, 0]
        
        features = pd.DataFrame(index=volume_data.index)
        
        for window in self.config.windows:
            # Basic volume statistics
            volume_mean = volume.rolling(window).mean()
            volume_std = volume.rolling(window).std()
            volume_sum = volume.rolling(window).sum()
            
            # Volume ratio features
            volume_ratio = volume / volume_mean
            volume_spike = volume / volume.rolling(window*2).mean()
            
            # Volume trend features
            volume_slope = self._calculate_rolling_slope(volume, window)
            volume_momentum = volume / volume.rolling(window).mean() - 1
            
            # Volume-price interaction features
            price_volume_ratio = volume / volume.rolling(window).mean()  # Simplified
            
            # Add features with window suffix
            features[f'volume_mean_{window}'] = volume_mean
            features[f'volume_std_{window}'] = volume_std
            features[f'volume_sum_{window}'] = volume_sum
            features[f'volume_ratio_{window}'] = volume_ratio
            features[f'volume_spike_{window}'] = volume_spike
            features[f'volume_slope_{window}'] = volume_slope
            features[f'volume_momentum_{window}'] = volume_momentum
            features[f'price_volume_ratio_{window}'] = price_volume_ratio
        
        return features.add_prefix('volume_')
    
    def _calculate_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling slope using linear regression."""
        tprint_info(f"📈 Calculating {window}-period rolling slope")
        
        # Use numpy for efficient calculation
        values = series.values
        n = len(values)
        
        if n < window:
            tprint_warning(f"⚠️ Insufficient data for {window}-period slope: {n} < {window}")
            return pd.Series(np.nan, index=series.index)
        
        slopes = np.full(n, np.nan)
        
        for i in range(window - 1, n):
            # Get window data
            window_values = values[i - window + 1:i + 1]
            window_x = np.arange(window)
            
            # Calculate slope using least squares
            if len(window_values) == window and not np.any(np.isnan(window_values)):
                # Simple linear regression: y = mx + b
                x_mean = np.mean(window_x)
                y_mean = np.mean(window_values)
                
                numerator = np.sum((window_x - x_mean) * (window_values - y_mean))
                denominator = np.sum((window_x - x_mean) ** 2)
                
                if denominator != 0:
                    slopes[i] = numerator / denominator
        
        tprint_success("✅ Rolling slope calculated successfully")
        return pd.Series(slopes, index=series.index)
    
    def _calculate_rolling_acceleration(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling acceleration (second derivative)."""
        tprint_info(f"📈 Calculating {window}-period rolling acceleration")
        
        slopes = self._calculate_rolling_slope(series, window)
        acceleration = self._calculate_rolling_slope(slopes, 3)  # Acceleration is slope of slopes
        
        tprint_success("✅ Rolling acceleration calculated successfully")
        return acceleration
    
    def _calculate_reversal_signals(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate reversal signals based on price action."""
        # Simple reversal detection based on local extrema
        values = series.values
        n = len(values)
        
        if n < window:
            return pd.Series(0, index=series.index)
        
        reversals = np.zeros(n)
        
        for i in range(window, n):
            # Check for local maximum followed by decline
            if i > 0 and i < n - 1:
                prev_values = values[i-window:i]
                current = values[i]
                next_values = values[i+1:i+window//2+1]
                
                # Local maximum detection
                if len(prev_values) > 0 and len(next_values) > 0:
                    is_local_max = current == np.max(prev_values) and current > np.max(next_values)
                    reversals[i] = 1 if is_local_max else 0
        
        return pd.Series(reversals, index=series.index)
    
    def _apply_anti_leakage_shifts(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply shifts to prevent look-ahead bias."""
        tprint_info(f"🔄 Applying anti-leakage shifts ({self.config.shift_periods} periods)")
        
        if self.config.shift_periods <= 0:
            tprint_warning("⚠️ No shift periods specified, skipping anti-leakage")
            return features
        
        shifted_features = features.copy()
        
        # Apply shift to all features except those that should be current
        tprint_info("🔄 Shifting features to prevent look-ahead bias")
        shifted_count = 0
        for col in features.columns:
            # Don't shift features that are already lagged or are identifiers
            if not any(suffix in col.lower() for suffix in ['_lag', '_shift', '_id']):
                shifted_features[col] = features[col].shift(self.config.shift_periods)
                shifted_count += 1
        
        tprint_success(f"✅ Applied anti-leakage shifts to {shifted_count} features")
        return shifted_features
    
    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate final features."""
        tprint_info("🧹 Cleaning and validating final features")
        
        # Remove infinite values
        tprint_info("🔄 Removing infinite values")
        features = features.replace([np.inf, -np.inf], np.nan)
        
        # Remove columns with too many NaN values
        tprint_info("🔍 Checking for columns with excessive NaN values")
        nan_ratio = features.isna().sum() / len(features)
        valid_columns = nan_ratio[nan_ratio < 0.5].index
        removed_cols = set(features.columns) - set(valid_columns)
        if removed_cols:
            tprint_warning(f"⚠️ Removing {len(removed_cols)} columns with excessive NaN values")
        features = features[valid_columns]
        
        # Forward fill remaining NaN values (limited)
        tprint_info("🔄 Applying forward fill (limit=3)")
        features = features.fillna(method='ffill', limit=3)
        
        # Backward fill remaining NaN values (limited)
        tprint_info("🔄 Applying backward fill (limit=1)")
        features = features.fillna(method='bfill', limit=1)
        
        # Drop any remaining NaN rows
        initial_rows = len(features)
        features = features.dropna()
        final_rows = len(features)
        if initial_rows != final_rows:
            tprint_info(f"🧹 Dropped {initial_rows - final_rows} rows with remaining NaN values")
        
        tprint_success(f"✅ Feature cleaning complete: {features.shape}")
        return features


def create_temporal_feature_extractor(
    windows: Optional[List[int]] = None,
    shift_periods: int = 1,
    include_momentum: bool = True,
    include_volatility: bool = True,
    include_volume_features: bool = True
) -> TemporalFeatureExtractor:
    """
    Factory function to create temporal feature extractor.
    
    Args:
        windows: List of rolling windows
        shift_periods: Number of periods to shift for anti-leakage
        include_momentum: Include momentum features
        include_volatility: Include volatility features
        include_volume_features: Include volume features
        
    Returns:
        TemporalFeatureExtractor instance
    """
    tprint_info("🏭 Creating Temporal Feature Extractor with factory function")
    
    config = TemporalFeatureConfig(
        windows=windows,
        shift_periods=shift_periods,
        include_momentum=include_momentum,
        include_volatility=include_volatility,
        include_volume_features=include_volume_features
    )
    
    tprint_info(f"📊 Configuration: windows={windows}, shift_periods={shift_periods}")
    tprint_info(f"📊 Configuration: momentum={include_momentum}, volatility={include_volatility}")
    tprint_info(f"📊 Configuration: volume_features={include_volume_features}")
    
    extractor = TemporalFeatureExtractor(config)
    tprint_success("✅ Temporal Feature Extractor created successfully")
    return extractor