"""
Trend Coherence Feature Engineering

This module implements the Trend Coherence feature for ensuring trend continuity
and direction consistency in 15-minute timeframe data.

Combines direction consistency and EMA slope for trend continuity.
Direction consistency: % of bars closing in same direction within last N bars
EMA slope: Rolling slope of EMA for trend continuity
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import existing utilities
from src.utils.tprint import tprint_info, tprint_warning, tprint_error
from src.utils.common_operations import safe_divide, safe_mean, safe_std
from src.utils.matrix_operations import vectorized_rolling_features

# Import framework components
from src.feature_generation.core.feature_generator import FeatureGenerator, FeatureCategory, FeatureConfig, FeatureResult, VectorizedFeatureGenerator

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

@dataclass
class TrendCoherenceConfig:
    """Configuration for Trend Coherence feature."""
    
    # Direction consistency settings
    direction_window: int = 8   # Window for direction consistency check
    min_periods: int = 1        # Minimum periods for rolling calculation
    
    # EMA settings
    ema_period: int = 12        # EMA period for slope calculation
    
    # Thresholds for interpretation
    min_direction_consistency: float = 0.6  # 60% of bars in same direction
    min_slope_threshold: float = 0.001      # Minimum slope for trend continuity
    
    # Output settings
    include_direction_consistency: bool = True  # Include direction consistency
    include_ema_slope: bool = True             # Include EMA slope
    include_trend_coherence_grade: bool = True  # Include combined grade (0.0-1.0)
    include_trend_class: bool = True           # Include trend classification

class TrendCoherenceFeature:
    """
    Trend Coherence Feature Engineering
    
    Ensures trend continuity and direction consistency by combining:
    1. Direction consistency: % of bars closing in same direction within last N bars
    2. EMA slope: Rolling slope of EMA for trend continuity
    """
    
    def __init__(self, config: Optional[TrendCoherenceConfig] = None):
        """Initialize Trend Coherence feature."""
        self.config = config or TrendCoherenceConfig()
        tprint_info("📊 Trend Coherence feature initialized")
        tprint_info(f"   → Direction window: {self.config.direction_window} bars")
        tprint_info(f"   → EMA period: {self.config.ema_period}")
        tprint_info(f"   → Min direction consistency: {self.config.min_direction_consistency}")
        tprint_info(f"   → Min slope threshold: {self.config.min_slope_threshold}")
    
    def calculate_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Trend Coherence features.
        
        Args:
            data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary of feature Series
        """
        tprint_info("📊 Calculating Trend Coherence features")
        
        # Validate input data
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        features = {}
        
        try:
            # Calculate direction consistency
            if self.config.include_direction_consistency:
                close_direction = np.sign(data['close'].diff())
                close_direction_series = pd.Series(close_direction, index=data.index)
                
                # Calculate direction consistency as the absolute value of the mean direction
                # (higher values = more consistent directional movement)
                direction_consistency = close_direction_series.rolling(
                    window=self.config.direction_window,
                    min_periods=1
                ).apply(lambda x: np.abs(x.mean()), raw=True)
                features['trend_direction_consistency'] = direction_consistency
                tprint_info(f"   → Direction consistency: mean={direction_consistency.mean():.3f}, std={direction_consistency.std():.3f}")
            
            # Calculate EMA slope
            if self.config.include_ema_slope:
                ema = data['close'].ewm(span=self.config.ema_period, min_periods=1).mean()
                ema_slope = ema.diff()
                features['trend_ema_slope'] = ema_slope
                tprint_info(f"   → EMA slope: mean={ema_slope.mean():.3f}, std={ema_slope.std():.3f}")
            
            # Calculate trend coherence grade (0.0-1.0)
            if self.config.include_trend_coherence_grade:
                # Convert to grade based on direction consistency and slope strength
                direction_grade = np.clip(direction_consistency, 0.0, 1.0)
                slope_grade = np.clip(ema_slope / self.config.min_slope_threshold, 0.0, 1.0)
                trend_coherence_grade = (direction_grade * slope_grade).clip(0.0, 1.0)
                features['trend_coherence_grade'] = trend_coherence_grade
                tprint_info(f"   → Trend coherence grade: mean={trend_coherence_grade.mean():.3f}, std={trend_coherence_grade.std():.3f}")
            
            # Calculate trend classification
            if self.config.include_trend_class:
                trend_class = pd.Series('incoherent', index=data.index)
                
                # Check direction consistency
                direction_consistent = direction_consistency >= self.config.min_direction_consistency
                slope_positive = ema_slope >= self.config.min_slope_threshold
                
                # Classify based on both criteria
                trend_class[direction_consistent & slope_positive] = 'coherent_uptrend'
                trend_class[direction_consistent & (ema_slope <= -self.config.min_slope_threshold)] = 'coherent_downtrend'
                trend_class[direction_consistent & (np.abs(ema_slope) < self.config.min_slope_threshold)] = 'coherent_sideways'
                trend_class[~direction_consistent] = 'incoherent'
                
                features['trend_class'] = trend_class
                
                # Count classifications
                class_counts = trend_class.value_counts()
                tprint_info(f"   → Trend classification: {dict(class_counts)}")
            
            tprint_info("✅ Trend Coherence features calculated successfully")
            return features
            
        except Exception as e:
            tprint_error(f"❌ Error calculating Trend Coherence features: {e}")
            raise
    
    def get_feature_names(self) -> list:
        """Get list of feature names this class produces."""
        features = []
        if self.config.include_direction_consistency:
            features.append('trend_direction_consistency')
        if self.config.include_ema_slope:
            features.append('trend_ema_slope')
        if self.config.include_trend_coherence_grade:
            features.append('trend_coherence_grade')
        if self.config.include_trend_class:
            features.append('trend_class')
        return features
    
    def get_feature_info(self) -> Dict[str, Dict[str, any]]:
        """Get detailed information about the features."""
        return {
            'trend_direction_consistency': {
                'description': f'Percentage of bars closing in same direction over {self.config.direction_window} bars',
                'range': '[0, 1]',
                'interpretation': 'Higher values indicate more consistent direction'
            },
            'trend_ema_slope': {
                'description': f'Slope of EMA({self.config.ema_period})',
                'range': '(-inf, inf)',
                'interpretation': 'Positive = uptrend, Negative = downtrend'
            },
            'trend_coherence_grade': {
                'description': 'Combined trend coherence grade (0.0-1.0)',
                'range': '[0, 1]',
                'interpretation': '1.0 = high coherence, 0.0 = low coherence'
            },
            'trend_class': {
                'description': 'Trend classification based on coherence',
                'values': ['coherent_uptrend', 'coherent_downtrend', 'coherent_sideways', 'incoherent'],
                'interpretation': 'Categorical classification of trend state'
            }
        }

class TrendCoherenceGenerator(VectorizedFeatureGenerator):
    """
    Framework-compatible Trend Coherence feature generator.
    
    Implements the FeatureGenerator interface for integration with the feature bank
    and period lookback optimization system.
    """
    
    def __init__(self, lookback: int = 8, **kwargs):
        """
        Initialize the Trend Coherence feature generator.
        
        Args:
            lookback: Number of periods for direction consistency check
            **kwargs: Additional configuration parameters
        """
        config = FeatureConfig(
            name="trend_coherence",
            category=FeatureCategory.TREND,
            description="Trend coherence ensuring trend continuity and direction consistency",
            required_columns=['open', 'high', 'low', 'close'],
            optional_columns=['volume'],
            default_lookback=lookback,
            min_lookback=1,
            max_lookback=20,
            parameters={
                'direction_window': lookback,
                'ema_period': kwargs.get('ema_period', 12),
                'min_direction_consistency': kwargs.get('min_direction_consistency', 0.6),
                'min_slope_threshold': kwargs.get('min_slope_threshold', 0.001),
                'include_direction_consistency': kwargs.get('include_direction_consistency', True),
                'include_ema_slope': kwargs.get('include_ema_slope', True),
                'include_trend_coherence_grade': kwargs.get('include_trend_coherence_grade', True),
                'include_trend_class': kwargs.get('include_trend_class', True)
            },
            matrix_optimized=True,
            gpu_accelerated=False,
            enable_feature_selection=True
        )
        
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize the feature engine
        feature_config = TrendCoherenceConfig(
            direction_window=lookback,
            ema_period=kwargs.get('ema_period', 12),
            min_direction_consistency=kwargs.get('min_direction_consistency', 0.6),
            min_slope_threshold=kwargs.get('min_slope_threshold', 0.001),
            include_direction_consistency=kwargs.get('include_direction_consistency', True),
            include_ema_slope=kwargs.get('include_ema_slope', True),
            include_trend_coherence_grade=kwargs.get('include_trend_coherence_grade', True),
            include_trend_class=kwargs.get('include_trend_class', True)
        )
        self.feature_engine = TrendCoherenceFeature(feature_config)
    
    def generate(self, data: pd.DataFrame, lookback: Optional[int] = None) -> FeatureResult:
        """
        Generate Trend Coherence features.
        
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
                self.feature_engine.config.direction_window = effective_lookback
            
            # Generate features
            features = self.feature_engine.calculate_features(data)
            
            # Select the primary feature (trend coherence grade)
            if 'trend_coherence_grade' in features:
                primary_feature = features['trend_coherence_grade']
            elif 'trend_direction_consistency' in features:
                primary_feature = features['trend_direction_consistency']
            else:
                raise ValueError("No primary trend coherence feature generated")
            
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
        Generate all Trend Coherence features.
        
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
            self.feature_engine.config.direction_window = effective_lookback
        
        # Generate all features
        return self.feature_engine.calculate_features(data)

# Convenience function for external usage
def calculate_trend_coherence_features(
    data: pd.DataFrame,
    config: Optional[TrendCoherenceConfig] = None
) -> Dict[str, pd.Series]:
    """
    Calculate Trend Coherence features.
    
    Args:
        data: OHLCV data with columns ['open', 'high', 'low', 'close', 'volume']
        config: Optional configuration
        
    Returns:
        Dictionary of feature Series
    """
    feature_engine = TrendCoherenceFeature(config)
    return feature_engine.calculate_features(data)

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
