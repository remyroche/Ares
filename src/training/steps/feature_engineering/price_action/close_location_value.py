"""
Close-Location Value (CLV) Feature Engineering

This module implements the Close-Location Value feature for tracking buying/selling pressure
and control in 15-minute timeframe data.

Formula: CLV_t = (2*close_t - high_t - low_t) / (high_t - low_t)
Rolling mean with volatility check
Sustained positive CLV → bullish control, sustained negative → bearish control
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import existing utilities
from src.utils.tprint import tprint_info, tprint_warning, tprint_error
from src.utils.common_operations import safe_divide, safe_mean, safe_std
from src.utils.matrix_operations import vectorized_rolling_features


@dataclass
class CLVConfig:
    """Configuration for Close-Location Value feature."""
    
    # Feature settings
    window: int = 8  # Rolling window for CLV smoothing
    min_periods: int = 1  # Minimum periods for rolling calculation
    
    # Thresholds for interpretation
    positive_threshold: float = 0.2   # Sustained positive CLV = bullish
    negative_threshold: float = -0.2  # Sustained negative CLV = bearish
    volatility_threshold: float = 0.5  # Avoid when CLV fluctuates rapidly
    
    # Output settings
    include_raw_clv: bool = True  # Include raw CLV values
    include_rolling_clv: bool = True  # Include rolling mean CLV
    include_clv_volatility: bool = True  # Include CLV volatility
    include_clv_grade: bool = True  # Include normalized grade (0.0-1.0)
    include_clv_class: bool = True  # Include CLV classification


class CloseLocationValueFeature:
    """
    Close-Location Value (CLV) Feature Engineering
    
    Tracks buying/selling pressure and control within each bar.
    Positive CLV indicates buying pressure, negative CLV indicates selling pressure.
    """
    
    def __init__(self, config: Optional[CLVConfig] = None):
        """Initialize Close-Location Value feature."""
        self.config = config or CLVConfig()
        tprint_info("📊 Close-Location Value feature initialized")
        tprint_info(f"   → Window: {self.config.window} bars")
        tprint_info(f"   → Positive threshold: {self.config.positive_threshold}")
        tprint_info(f"   → Negative threshold: {self.config.negative_threshold}")
        tprint_info(f"   → Volatility threshold: {self.config.volatility_threshold}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Close-Location Value features.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating Close-Location Value features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Calculate raw CLV
            price_range = data['high'] - data['low']
            price_range = price_range.replace(0, np.nan)  # Avoid division by zero
            
            clv_numerator = 2 * data['close'] - data['high'] - data['low']
            raw_clv = clv_numerator / price_range
            raw_clv = raw_clv.fillna(0)  # Set to 0 for zero-range bars
            raw_clv = raw_clv.replace([np.inf, -np.inf], 0)  # Replace infinite values
            
            if self.config.include_raw_clv:
                features['clv_raw'] = raw_clv
                tprint_info(f"   → Raw CLV: mean={raw_clv.mean():.3f}, std={raw_clv.std():.3f}")
            
            # Calculate rolling mean CLV
            if self.config.include_rolling_clv:
                rolling_clv = vectorized_rolling_features(
                    raw_clv.values,
                    windows=self.config.window,
                    operation='mean'
                )
                rolling_clv = pd.Series(rolling_clv, index=data.index)
                features['clv_rolling'] = rolling_clv
                tprint_info(f"   → Rolling CLV: mean={rolling_clv.mean():.3f}, std={rolling_clv.std():.3f}")
            
            # Calculate CLV volatility
            if self.config.include_clv_volatility:
                clv_volatility = vectorized_rolling_features(
                    raw_clv.values,
                    windows=self.config.window,
                    operation='std'
                )
                clv_volatility = pd.Series(clv_volatility, index=data.index)
                features['clv_volatility'] = clv_volatility
                tprint_info(f"   → CLV volatility: mean={clv_volatility.mean():.3f}, std={clv_volatility.std():.3f}")
            
            # Calculate CLV grade (0.0-1.0)
            if self.config.include_clv_grade:
                # Grade based on directional strength and stability
                clv_strength = np.abs(rolling_clv)
                clv_stability = 1.0 - np.clip(clv_volatility / self.config.volatility_threshold, 0.0, 1.0)
                clv_grade = (clv_strength * clv_stability).clip(0.0, 1.0)
                features['clv_grade'] = clv_grade
                tprint_info(f"   → CLV grade: mean={clv_grade.mean():.3f}, std={clv_grade.std():.3f}")
            
            # Calculate CLV classification
            if self.config.include_clv_class and self.config.include_rolling_clv:
                clv_class = pd.Series('neutral', index=data.index)
                clv_class[rolling_clv >= self.config.positive_threshold] = 'bullish'
                clv_class[rolling_clv <= self.config.negative_threshold] = 'bearish'
                
                # Mark as unstable if volatility is too high
                if self.config.include_clv_volatility:
                    clv_class[clv_volatility > self.config.volatility_threshold] = 'unstable'
                
                features['clv_class'] = clv_class
                
                # Count classifications
                class_counts = clv_class.value_counts()
                tprint_info(f"   → CLV classification: {dict(class_counts)}")
            
            tprint_info("✅ Close-Location Value features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating Close-Location Value features: {e}")
            raise
    
    def get_feature_names(self) -> list:
        """Get list of feature names this class produces."""
        features = []
        if self.config.include_raw_clv:
            features.append('clv_raw')
        if self.config.include_rolling_clv:
            features.append('clv_rolling')
        if self.config.include_clv_volatility:
            features.append('clv_volatility')
        if self.config.include_clv_grade:
            features.append('clv_grade')
        if self.config.include_clv_class:
            features.append('clv_class')
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about the features."""
        return {
            'clv_raw': {
                'description': 'Raw Close-Location Value ((2*close-high-low) / (high-low))',
                'range': '[-1, 1]',
                'interpretation': 'Positive = buying pressure, Negative = selling pressure'
            },
            'clv_rolling': {
                'description': f'Rolling mean CLV over {self.config.window} bars',
                'range': '[-1, 1]',
                'interpretation': 'Smoothed CLV for trend analysis'
            },
            'clv_volatility': {
                'description': f'Rolling standard deviation of CLV over {self.config.window} bars',
                'range': '[0, inf)',
                'interpretation': 'Higher values indicate more volatile CLV'
            },
            'clv_grade': {
                'description': 'Normalized CLV grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = strong directional CLV with low volatility'
            },
            'clv_class': {
                'description': 'CLV classification (bullish/bearish/neutral/unstable)',
                'values': ['bullish', 'bearish', 'neutral', 'unstable'],
                'interpretation': 'Categorical classification based on thresholds'
            }
        }


# Convenience function for external usage
def calculate_clv_features(
    data: pd.DataFrame,
    config: Optional[CLVConfig] = None
) -> Dict[str, pd.Series]:
    """
    Calculate Close-Location Value features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = CloseLocationValueFeature(config)
    return feature_engine.calculate_features(data)