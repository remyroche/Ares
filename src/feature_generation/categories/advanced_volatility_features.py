"""
Advanced Volatility Features with VectorBT Optimization

This module provides comprehensive volatility feature generation using VectorBT's
optimized indicators and advanced statistical measures for enhanced market analysis.

Key Features:
- ATR (Average True Range) with VectorBT optimization
- Bollinger Bands with advanced statistical measures
- Keltner Channels and volatility clustering detection
- GARCH-based volatility modeling
- Regime-based volatility analysis
- GPU acceleration support
"""

import numpy as np
import pandas as pd
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

# VectorBT imports for optimization
try:
    import vectorbt as vbt
    from vectorbt.indicators import ATR, BBANDS, KC, STOCH, WILLR, CCI, MFI, ADX
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max
    from vectorbt.generic import rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    ATR = None
    BBANDS = None
    KC = None
    STOCH = None
    WILLR = None
    CCI = None
    MFI = None
    ADX = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for advanced volatility features."""
    # ATR settings
    atr_periods: List[int] = None
    atr_multipliers: List[float] = None
    
    # Bollinger Bands settings
    bb_periods: List[int] = None
    bb_std_devs: List[float] = None
    
    # Keltner Channels settings
    kc_periods: List[int] = None
    kc_multipliers: List[float] = None
    
    # Advanced settings
    enable_garch: bool = True
    enable_regime_analysis: bool = True
    enable_clustering_detection: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    def __post_init__(self):
        if self.atr_periods is None:
            self.atr_periods = [14, 21, 30]
        if self.atr_multipliers is None:
            self.atr_multipliers = [1.0, 2.0, 3.0]
        if self.bb_periods is None:
            self.bb_periods = [20, 30, 50]
        if self.bb_std_devs is None:
            self.bb_std_devs = [1.5, 2.0, 2.5]
        if self.kc_periods is None:
            self.kc_periods = [20, 30, 50]
        if self.kc_multipliers is None:
            self.kc_multipliers = [1.0, 1.5, 2.0]


class AdvancedVolatilityFeatures(VectorBTFeatureGenerator):
    """
    Advanced volatility feature generator using VectorBT's optimized indicators.
    
    Provides comprehensive volatility analysis including:
    - ATR-based features
    - Bollinger Bands analysis
    - Keltner Channels
    - Volatility clustering detection
    - GARCH modeling
    - Regime-based analysis
    """
    
    def __init__(self, config: Optional[VolatilityConfig] = None, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize advanced volatility feature generator.
        
        Args:
            config: Volatility configuration
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        self.config = config or VolatilityConfig()
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        
        # Initialize VectorBT optimizers for consistent usage
        self.vectorbt_optimizer = None
        self.unified_optimizer = None
        if VECTORBT_AVAILABLE:
            try:
                from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
                from ..utils.unified_optimization_system import get_unified_optimization_system
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.enable_gpu,
                    enable_parallel=self.enable_parallel
                )
                self.unified_optimizer = get_unified_optimization_system()
                logger.info("✅ VectorBT optimizers initialized for AdvancedVolatilityFeatures")
            except Exception as e:
                logger.warning(f"⚠️ VectorBT optimizer initialization failed: {e}")
        
        # Create feature config
        feature_config = FeatureConfig(
            name="advanced_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Advanced volatility features using VectorBT indicators",
            required_columns=["high", "low", "close"],
            optional_columns=["open", "volume"],
            default_lookback=max(self.config.atr_periods + self.config.bb_periods + self.config.kc_periods),
            min_lookback=14,
            max_lookback=200,
            parameters={
                "atr_periods": self.config.atr_periods,
                "bb_periods": self.config.bb_periods,
                "kc_periods": self.config.kc_periods,
                "enable_garch": self.config.enable_garch,
                "enable_regime_analysis": self.config.enable_regime_analysis
            },
            matrix_optimized=True,
            gpu_accelerated=self.enable_gpu
        )
        
        super().__init__(feature_config, enable_gpu=self.enable_gpu, enable_parallel=self.enable_parallel)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if self.vectorbt_optimizer:
            try:
                if operation == 'mean':
                    return self.vectorbt_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    return self.vectorbt_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    return self.vectorbt_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    return self.vectorbt_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    return self.vectorbt_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    return self.vectorbt_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    if func is not None:
                        return self.vectorbt_optimizer.rolling_apply(data, func, window, **kwargs)
                    else:
                        raise ValueError("Function must be provided for rolling apply operation")
                elif operation == 'quantile':
                    q = kwargs.get('q', 0.5)
                    return self.vectorbt_optimizer.rolling_quantile(data, window, q=q, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        elif operation == 'apply':
            func = kwargs.get('func')
            if func is not None:
                return rolling_obj.apply(func)
            else:
                raise ValueError("Function must be provided for rolling apply operation")
        elif operation == 'quantile':
            q = kwargs.get('q', 0.5)
            return rolling_obj.quantile(q)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate comprehensive volatility features using VectorBT.
        
        Args:
            data: OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with volatility features
        """
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, falling back to basic volatility features")
            return self._generate_basic_volatility_features(data)
        
        try:
            features = pd.DataFrame(index=data.index)
            
            # Generate ATR-based features
            features = self._generate_atr_features(data, features)
            
            # Generate Bollinger Bands features
            features = self._generate_bollinger_bands_features(data, features)
            
            # Generate Keltner Channels features
            features = self._generate_keltner_channels_features(data, features)
            
            # Generate volatility clustering features
            if self.config.enable_clustering_detection:
                features = self._generate_volatility_clustering_features(data, features)
            
            # Generate regime-based volatility features
            if self.config.enable_regime_analysis:
                features = self._generate_regime_volatility_features(data, features)
            
            # Generate GARCH-based features
            if self.config.enable_garch:
                features = self._generate_garch_features(data, features)
            
            # Generate advanced statistical features
            features = self._generate_advanced_statistical_features(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating advanced volatility features: {e}")
            return self._generate_basic_volatility_features(data)
    
    def _generate_atr_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate ATR-based volatility features."""
        try:
            for period in self.config.atr_periods:
                # Calculate ATR using VectorBT
                atr = ATR.run(data['high'], data['low'], data['close'], window=period)
                
                # Basic ATR
                features[f'atr_{period}'] = atr.atr
                
                # ATR as percentage of close price
                features[f'atr_pct_{period}'] = (atr.atr / data['close']) * 100
                
                # ATR moving averages - use optimized rolling operations
                features[f'atr_sma_{period}'] = self._vectorbt_rolling_operation(atr.atr, 'mean', period)
                features[f'atr_ema_{period}'] = vbt.MA.run(atr.atr, window=period, short_window=period//2).ma
                
                # ATR volatility (volatility of volatility)
                features[f'atr_vol_{period}'] = self._vectorbt_rolling_operation(atr.atr, 'std', period)
                
                # ATR position relative to recent range
                atr_high = self._vectorbt_rolling_operation(atr.atr, 'max', period)
                atr_low = self._vectorbt_rolling_operation(atr.atr, 'min', period)
                features[f'atr_position_{period}'] = (atr.atr - atr_low) / (atr_high - atr_low)
                
                # ATR trend - use optimized rolling apply
                features[f'atr_trend_{period}'] = self._vectorbt_rolling_operation(
                    atr.atr, 
                    'apply', 
                    period//2,
                    func=lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating ATR features: {e}")
            return features
    
    def _generate_bollinger_bands_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands-based volatility features."""
        try:
            for period in self.config.bb_periods:
                for std_dev in self.config.bb_std_devs:
                    # Calculate Bollinger Bands using VectorBT
                    bb = BBANDS.run(data['close'], window=period, alpha=std_dev)
                    
                    # Basic Bollinger Bands
                    features[f'bb_upper_{period}_{std_dev}'] = bb.upper
                    features[f'bb_lower_{period}_{std_dev}'] = bb.lower
                    features[f'bb_middle_{period}_{std_dev}'] = bb.middle
                    
                    # Bollinger Bands width and position
                    bb_width = (bb.upper - bb.lower) / bb.middle
                    features[f'bb_width_{period}_{std_dev}'] = bb_width
                    features[f'bb_position_{period}_{std_dev}'] = (data['close'] - bb.lower) / (bb.upper - bb.lower)
                    
                    # Bollinger Bands squeeze detection using VectorBT
                    features[f'bb_squeeze_{period}_{std_dev}'] = (bb_width < rolling_apply(bb_width, lambda x: x.quantile(0.2), window=period)).astype(int)
                    
                    # Bollinger Bands breakout detection
                    features[f'bb_breakout_upper_{period}_{std_dev}'] = (data['close'] > bb.upper).astype(int)
                    features[f'bb_breakout_lower_{period}_{std_dev}'] = (data['close'] < bb.lower).astype(int)
                    
                    # Bollinger Bands momentum
                    features[f'bb_momentum_{period}_{std_dev}'] = rolling_apply(
                        bb_position, 
                        lambda x: x.iloc[-1] - x.iloc[0], 
                        window=period//2
                    )
                    
                    # Bollinger Bands volatility
                    features[f'bb_volatility_{period}_{std_dev}'] = rolling_std(bb_width, window=period)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating Bollinger Bands features: {e}")
            return features
    
    def _generate_keltner_channels_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate Keltner Channels-based volatility features."""
        try:
            for period in self.config.kc_periods:
                for multiplier in self.config.kc_multipliers:
                    # Calculate Keltner Channels using VectorBT
                    kc = KC.run(data['high'], data['low'], data['close'], window=period, alpha=multiplier)
                    
                    # Basic Keltner Channels
                    features[f'kc_upper_{period}_{multiplier}'] = kc.upper
                    features[f'kc_lower_{period}_{multiplier}'] = kc.lower
                    features[f'kc_middle_{period}_{multiplier}'] = kc.middle
                    
                    # Keltner Channels width and position
                    kc_width = (kc.upper - kc.lower) / kc.middle
                    features[f'kc_width_{period}_{multiplier}'] = kc_width
                    features[f'kc_position_{period}_{multiplier}'] = (data['close'] - kc.lower) / (kc.upper - kc.lower)
                    
                    # Keltner Channels vs Bollinger Bands
                    if f'bb_width_{period}_{2.0}' in features.columns:
                        bb_width = features[f'bb_width_{period}_{2.0}']
                        features[f'kc_bb_ratio_{period}_{multiplier}'] = kc_width / bb_width
                    
                    # Keltner Channels trend
                    features[f'kc_trend_{period}_{multiplier}'] = rolling_apply(
                        kc.middle, 
                        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, 
                        window=period//2
                    )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating Keltner Channels features: {e}")
            return features
    
    def _generate_volatility_clustering_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility clustering detection features."""
        try:
            # Calculate returns using optimized rolling apply
            returns = self._vectorbt_rolling_operation(
                data['close'], 
                'apply', 
                1,
                func=lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
            )
            
            # Volatility clustering indicators - use optimized rolling operations
            vol_short = self._vectorbt_rolling_operation(returns, 'std', 5)
            vol_long = self._vectorbt_rolling_operation(returns, 'std', 20)
            
            # Volatility clustering ratio
            features['vol_cluster_ratio'] = vol_short / vol_long
            
            # Volatility clustering momentum - use optimized rolling apply
            features['vol_cluster_momentum'] = self._vectorbt_rolling_operation(
                vol_short, 
                'apply', 
                10,
                func=lambda x: x.iloc[-1] - x.iloc[0]
            )
            
            # Volatility clustering persistence - use optimized rolling apply
            features['vol_cluster_persistence'] = self._vectorbt_rolling_operation(
                vol_short, 
                'apply', 
                20,
                func=lambda x: (x > self._vectorbt_rolling_operation(x, 'mean', 5)).sum() / len(x)
            )
            
            # Volatility clustering regime detection
            vol_threshold = vol_long.quantile(0.7)
            features['vol_cluster_regime'] = (vol_short > vol_threshold).astype(int)
            
            # Volatility clustering intensity - use optimized rolling apply
            features['vol_cluster_intensity'] = self._vectorbt_rolling_operation(
                returns.abs(), 
                'apply', 
                10,
                func=lambda x: x.sum()
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volatility clustering features: {e}")
            return features
    
    def _generate_regime_volatility_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-based volatility features."""
        try:
            # Calculate returns using VectorBT
            returns = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
            
            # Volatility regimes based on different timeframes
            vol_short = rolling_std(returns, window=5)
            vol_medium = rolling_std(returns, window=20)
            vol_long = rolling_std(returns, window=50)
            
            # Regime classification
            high_vol_threshold = vol_medium.quantile(0.7)
            low_vol_threshold = vol_medium.quantile(0.3)
            
            features['vol_regime'] = np.where(
                vol_medium > high_vol_threshold, 2,  # High volatility
                np.where(vol_medium < low_vol_threshold, 0, 1)  # Low volatility, Medium volatility
            )
            
            # Regime persistence
            features['vol_regime_persistence'] = rolling_apply(
                features['vol_regime'], 
                lambda x: (x == x.iloc[-1]).sum() / len(x), 
                window=20
            )
            
            # Regime transition probability
            features['vol_regime_transition'] = rolling_apply(
                features['vol_regime'], 
                lambda x: 1 if x.iloc[-1] != x.iloc[0] else 0, 
                window=10
            )
            
            # Regime-specific volatility measures
            for regime in [0, 1, 2]:
                regime_mask = features['vol_regime'] == regime
                features[f'vol_regime_{regime}_intensity'] = np.where(
                    regime_mask, vol_short, np.nan
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating regime volatility features: {e}")
            return features
    
    def _generate_garch_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate GARCH-based volatility features."""
        try:
            # Calculate returns using VectorBT
            returns = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
            returns = returns.dropna()
            
            if len(returns) < 50:  # Need sufficient data for GARCH
                return features
            
            # Simple GARCH(1,1) approximation using rolling windows
            # This is a simplified version - for full GARCH, use specialized libraries
            
            # Rolling variance (GARCH approximation)
            rolling_var = rolling_var(returns, window=20)
            features['garch_variance'] = rolling_var
            
            # GARCH volatility
            features['garch_volatility'] = np.sqrt(rolling_var)
            
            # GARCH persistence (simplified)
            features['garch_persistence'] = rolling_apply(
                rolling_var, 
                lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0, 
                window=30
            )
            
            # GARCH mean reversion
            features['garch_mean_reversion'] = rolling_apply(
                rolling_var, 
                lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, 
                window=20
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating GARCH features: {e}")
            return features
    
    def _generate_advanced_statistical_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced statistical volatility features."""
        try:
            # Calculate returns using VectorBT
            returns = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
            
            # Higher moments
            features['vol_skewness'] = rolling_apply(returns, lambda x: x.skew(), window=20)
            features['vol_kurtosis'] = rolling_apply(returns, lambda x: x.kurtosis(), window=20)
            
            # Volatility of volatility
            vol_short = rolling_std(returns, window=5)
            features['vol_of_vol'] = rolling_std(vol_short, window=20)
            
            # Volatility percentiles
            features['vol_percentile_25'] = rolling_apply(vol_short, lambda x: x.quantile(0.25), window=20)
            features['vol_percentile_75'] = rolling_apply(vol_short, lambda x: x.quantile(0.75), window=20)
            features['vol_percentile_90'] = rolling_apply(vol_short, lambda x: x.quantile(0.90), window=20)
            
            # Volatility momentum
            features['vol_momentum_5'] = rolling_apply(vol_short, lambda x: x.iloc[-1] - x.iloc[0], window=5)
            features['vol_momentum_10'] = rolling_apply(vol_short, lambda x: x.iloc[-1] - x.iloc[0], window=10)
            
            # Volatility acceleration
            vol_momentum = rolling_apply(vol_short, lambda x: x.iloc[-1] - x.iloc[0], window=5)
            features['vol_acceleration'] = rolling_apply(vol_momentum, lambda x: x.iloc[-1] - x.iloc[0], window=5)
            
            # Volatility mean reversion
            vol_mean = rolling_mean(vol_short, window=20)
            features['vol_mean_reversion'] = (vol_short - vol_mean) / vol_mean
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating advanced statistical features: {e}")
            return features
    
    def _generate_basic_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic volatility features as fallback."""
        features = pd.DataFrame(index=data.index)
        
        # Basic rolling standard deviation using VectorBT
        returns = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
        features['volatility_20'] = rolling_std(returns, window=20)
        features['volatility_50'] = rolling_std(returns, window=50)
        
        # Basic ATR approximation using VectorBT
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - rolling_apply(data['close'], lambda x: x.iloc[0], window=1))
        low_close = np.abs(data['low'] - rolling_apply(data['close'], lambda x: x.iloc[0], window=1))
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        features['atr_14'] = rolling_mean(true_range, window=14)
        
        return features


def create_advanced_volatility_generator(
    config: Optional[VolatilityConfig] = None,
    enable_gpu: bool = False,
    enable_parallel: bool = True
) -> AdvancedVolatilityFeatures:
    """
    Create an advanced volatility feature generator.
    
    Args:
        config: Volatility configuration
        enable_gpu: Whether to enable GPU acceleration
        enable_parallel: Whether to enable parallel processing
        
    Returns:
        AdvancedVolatilityFeatures instance
    """
    return AdvancedVolatilityFeatures(config, enable_gpu, enable_parallel)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1min')
    np.random.seed(42)
    
    # Generate sample OHLCV data
    returns = np.random.normal(0.001, 0.02, 1000)
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, 1000)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, 1000)
    }, index=dates)
    
    # Create and test the generator
    generator = create_advanced_volatility_generator(enable_gpu=False, enable_parallel=True)
    features = generator.generate_features(data)
    
    print(f"Generated {len(features.columns)} volatility features")
    print("Feature names:", list(features.columns))
    print("\nFirst few rows:")
    print(features.head())