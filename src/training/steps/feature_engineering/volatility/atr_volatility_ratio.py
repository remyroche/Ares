"""
ATR Volatility Ratio Feature Engineering

This module implements the ATR Volatility Ratio feature for normalizing volatility
and identifying appropriate trading conditions in 15-minute timeframe data.

Formula: r_t = ATR_short / ATR_long
Short-term (1 hour) vs long-term (5 hours) ATR comparison
Skip when r_t > 1.5-2.0 (too jumpy) - no "too quiet" filter
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
class ATRVolatilityRatioConfig:
    """Configuration for ATR Volatility Ratio feature."""
    
    # ATR calculation settings
    short_window: int = 4   # Short-term ATR window (1 hour)
    long_window: int = 20   # Long-term ATR window (5 hours)
    min_periods: int = 1    # Minimum periods for rolling calculation
    
    # Thresholds for interpretation
    high_ratio_threshold: float = 1.5  # Too jumpy - skip signals
    # Removed low_ratio_threshold - no "too quiet" filter
    
    # Output settings
    include_atr_short: bool = True  # Include short-term ATR
    include_atr_long: bool = True  # Include long-term ATR
    include_atr_ratio: bool = True  # Include ATR ratio
    include_atr_grade: bool = True  # Include normalized grade (0.0-1.0)
    include_atr_class: bool = True  # Include ATR classification


class ATRVolatilityRatioFeature:
    """
    ATR Volatility Ratio Feature Engineering
    
    Compares short-term vs long-term Average True Range to identify appropriate
    volatility conditions for trading. Higher ratios indicate more volatile conditions.
    """
    
    def __init__(self, config: Optional[ATRVolatilityRatioConfig] = None):
        """Initialize ATR Volatility Ratio feature."""
        self.config = config or ATRVolatilityRatioConfig()
        tprint_info("📊 ATR Volatility Ratio feature initialized")
        tprint_info(f"   → Short window: {self.config.short_window} bars")
        tprint_info(f"   → Long window: {self.config.long_window} bars")
        tprint_info(f"   → High ratio threshold: {self.config.high_ratio_threshold}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate ATR Volatility Ratio features.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating ATR Volatility Ratio features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Calculate True Range
            tr1 = data['high'] - data['low']
            tr2 = np.abs(data['high'] - data['close'].shift(1))
            tr3 = np.abs(data['low'] - data['close'].shift(1))
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate short-term ATR
            if self.config.include_atr_short:
                atr_short = vectorized_rolling_features(
                    true_range.values,
                    windows=self.config.short_window,
                    operation='mean'
                )
                atr_short = pd.Series(atr_short, index=data.index)
                features['atr_short'] = atr_short
                tprint_info(f"   → Short-term ATR: mean={atr_short.mean():.3f}, std={atr_short.std():.3f}")
            
            # Calculate long-term ATR
            if self.config.include_atr_long:
                atr_long = vectorized_rolling_features(
                    true_range.values,
                    windows=self.config.long_window,
                    operation='mean'
                )
                atr_long = pd.Series(atr_long, index=data.index)
                features['atr_long'] = atr_long
                tprint_info(f"   → Long-term ATR: mean={atr_long.mean():.3f}, std={atr_long.std():.3f}")
            
            # Calculate ATR ratio
            if self.config.include_atr_ratio:
                atr_ratio = atr_short / atr_long
                atr_ratio = atr_ratio.fillna(1.0)  # Fill NaN values with 1.0
                atr_ratio = atr_ratio.replace([np.inf, -np.inf], 1.0)  # Replace infinite values
                features['atr_ratio'] = atr_ratio
                tprint_info(f"   → ATR ratio: mean={atr_ratio.mean():.3f}, std={atr_ratio.std():.3f}")
            
            # Calculate ATR grade (0.0-1.0)
            if self.config.include_atr_grade:
                # Grade decreases as ratio approaches the threshold (too jumpy)
                # No penalty for low ratios (no "too quiet" filter)
                atr_grade = np.clip(1.0 - (atr_ratio / self.config.high_ratio_threshold), 0.0, 1.0)
                features['atr_grade'] = atr_grade
                tprint_info(f"   → ATR grade: mean={atr_grade.mean():.3f}, std={atr_grade.std():.3f}")
            
            # Calculate ATR classification
            if self.config.include_atr_class and self.config.include_atr_ratio:
                atr_class = pd.Series('moderate', index=data.index)
                atr_class[atr_ratio > self.config.high_ratio_threshold] = 'too_jumpy'
                # No "too_quiet" classification - removed as per requirements
                features['atr_class'] = atr_class
                
                # Count classifications
                class_counts = atr_class.value_counts()
                tprint_info(f"   → ATR classification: {dict(class_counts)}")
            
            tprint_info("✅ ATR Volatility Ratio features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating ATR Volatility Ratio features: {e}")
            raise
    
    def get_feature_names(self) -> list:
        """Get list of feature names this class produces."""
        features = []
        if self.config.include_atr_short:
            features.append('atr_short')
        if self.config.include_atr_long:
            features.append('atr_long')
        if self.config.include_atr_ratio:
            features.append('atr_ratio')
        if self.config.include_atr_grade:
            features.append('atr_grade')
        if self.config.include_atr_class:
            features.append('atr_class')
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about the features."""
        return {
            'atr_short': {
                'description': f'Short-term Average True Range over {self.config.short_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Recent volatility measure'
            },
            'atr_long': {
                'description': f'Long-term Average True Range over {self.config.long_window} bars',
                'range': '[0, inf)',
                'interpretation': 'Baseline volatility measure'
            },
            'atr_ratio': {
                'description': 'Ratio of short-term to long-term ATR',
                'range': '[0, inf)',
                'interpretation': 'Higher values indicate increased volatility'
            },
            'atr_grade': {
                'description': 'Normalized ATR grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = moderate volatility, 0.0 = too jumpy'
            },
            'atr_class': {
                'description': 'ATR classification (moderate/too_jumpy)',
                'values': ['moderate', 'too_jumpy'],
                'interpretation': 'Categorical classification based on thresholds'
            }
        }


# Convenience function for external usage
def calculate_atr_volatility_ratio_features(
    data: pd.DataFrame,
    config: Optional[ATRVolatilityRatioConfig] = None
) -> Dict[str, pd.Series]:
    """
    Calculate ATR Volatility Ratio features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = ATRVolatilityRatioFeature(config)
    return feature_engine.calculate_features(data)