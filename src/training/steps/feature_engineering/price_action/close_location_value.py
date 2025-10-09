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


class CloseLocationValueGenerator(FeatureGenerator):
    """
    Framework-compatible Close-Location Value feature generator.
    
    Implements the FeatureGenerator interface for integration with the feature bank
    and period lookback optimization system.
    """
    
    def __init__(self, lookback: int = 8, **kwargs):
        """
        Initialize the Close-Location Value feature generator.
        
        Args:
            lookback: Number of periods for rolling calculation
            **kwargs: Additional configuration parameters
        """
        config = FeatureConfig(
            name="close_location_value",
            category=FeatureCategory.PRICE_ACTION,
            description="Close-Location Value measuring buying/selling pressure and control",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=20,
            parameters={
                'window': lookback,
                'positive_threshold': kwargs.get('positive_threshold', 0.2),
                'negative_threshold': kwargs.get('negative_threshold', -0.2),
                'volatility_threshold': kwargs.get('volatility_threshold', 0.5),
                'include_raw_clv': kwargs.get('include_raw_clv', True),
                'include_rolling_clv': kwargs.get('include_rolling_clv', True),
                'include_clv_volatility': kwargs.get('include_clv_volatility', True),
                'include_clv_grade': kwargs.get('include_clv_grade', True),
                'include_clv_class': kwargs.get('include_clv_class', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config)
        
        # Initialize the feature engine
        feature_config = CLVConfig(
            window=lookback,
            positive_threshold=kwargs.get('positive_threshold', 0.2),
            negative_threshold=kwargs.get('negative_threshold', -0.2),
            volatility_threshold=kwargs.get('volatility_threshold', 0.5),
            include_raw_clv=kwargs.get('include_raw_clv', True),
            include_rolling_clv=kwargs.get('include_rolling_clv', True),
            include_clv_volatility=kwargs.get('include_clv_volatility', True),
            include_clv_grade=kwargs.get('include_clv_grade', True),
            include_clv_class=kwargs.get('include_clv_class', True)
        )
        self.feature_engine = CloseLocationValueFeature(feature_config)
    
    def generate(self, data: pd.DataFrame, lookback: Optional[int] = None) -> FeatureResult:
        """
        Generate Close-Location Value features.
        
        Args:
            data: OHLCV data with required columns
            lookback: Override default lookback period
            
        Returns:
            FeatureResult with generated features
        """
        start_time = time.time()
        
        try:
            # Use provided lookback or default
            effective_lookback = lookback or self.config.default_lookback
            
            # Update feature engine configuration if lookback changed
            if effective_lookback != self.config.default_lookback:
                self.feature_engine.config.window = effective_lookback
            
            # Generate features
            features = self.feature_engine.calculate_features(data)
            
            # Select the primary feature (rolling CLV)
            if 'clv_rolling' in features:
                primary_feature = features['clv_rolling']
            elif 'clv_raw' in features:
                primary_feature = features['clv_raw']
            else:
                raise ValueError("No primary CLV feature generated")
            
            computation_time = time.time() - start_time
            
            return FeatureResult(
                name=self.config.name,
                data=primary_feature,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata={
                    'lookback_used': effective_lookback,
                    'all_features': list(features.keys()),
                    'feature_stats': {
                        'mean': float(primary_feature.mean()),
                        'std': float(primary_feature.std()),
                        'min': float(primary_feature.min()),
                        'max': float(primary_feature.max())
                    }
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=str(e)
            )
    
    def get_all_features(self, data: pd.DataFrame, lookback: Optional[int] = None) -> Dict[str, pd.Series]:
        """
        Generate all Close-Location Value features.
        
        Args:
            data: OHLCV data with required columns
            lookback: Override default lookback period
            
        Returns:
            Dictionary of all generated features
        """
        # Use provided lookback or default
        effective_lookback = lookback or self.config.default_lookback
        
        # Update feature engine configuration if lookback changed
        if effective_lookback != self.config.default_lookback:
            self.feature_engine.config.window = effective_lookback
        
        # Generate all features
        return self.feature_engine.calculate_features(data)


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