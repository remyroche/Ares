"""
Advanced Volume Features with VectorBT Optimization

This module provides comprehensive volume feature generation using VectorBT's
optimized indicators and advanced volume analysis techniques.

Key Features:
- On-Balance Volume (OBV) with VectorBT optimization
- Accumulation/Distribution Line (AD) with advanced metrics
- Money Flow Index (MFI) and related indicators
- Volume Rate of Change and momentum indicators
- Volume-weighted average price (VWAP) with VectorBT
- Volume profile analysis and clustering
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
    from vectorbt.indicators import OBV, AD, MFI, ADOSC, AROONOSC
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max
    from vectorbt.generic import rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    OBV = None
    AD = None
    MFI = None
    ADOSC = None
    AROONOSC = None
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
class VolumeConfig:
    """Configuration for advanced volume features."""
    # OBV settings
    obv_periods: List[int] = None
    obv_sma_periods: List[int] = None
    
    # AD settings
    ad_periods: List[int] = None
    ad_sma_periods: List[int] = None
    
    # MFI settings
    mfi_periods: List[int] = None
    mfi_thresholds: List[float] = None
    
    # VWAP settings
    vwap_periods: List[int] = None
    vwap_types: List[str] = None
    
    # Volume momentum settings
    volume_roc_periods: List[int] = None
    volume_momentum_periods: List[int] = None
    
    # Advanced settings
    enable_volume_profile: bool = True
    enable_volume_clustering: bool = True
    enable_volume_regime_analysis: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    
    def __post_init__(self):
        if self.obv_periods is None:
            self.obv_periods = [14, 21, 30]
        if self.obv_sma_periods is None:
            self.obv_sma_periods = [5, 10, 20]
        if self.ad_periods is None:
            self.ad_periods = [14, 21, 30]
        if self.ad_sma_periods is None:
            self.ad_sma_periods = [5, 10, 20]
        if self.mfi_periods is None:
            self.mfi_periods = [14, 21, 30]
        if self.mfi_thresholds is None:
            self.mfi_thresholds = [20, 30, 50, 70, 80]
        if self.vwap_periods is None:
            self.vwap_periods = [20, 50, 100]
        if self.vwap_types is None:
            self.vwap_types = ['standard', 'exponential', 'volume_weighted']
        if self.volume_roc_periods is None:
            self.volume_roc_periods = [1, 5, 10, 20]
        if self.volume_momentum_periods is None:
            self.volume_momentum_periods = [5, 10, 20]


class AdvancedVolumeFeatures(VectorBTFeatureGenerator):
    """
    Advanced volume feature generator using VectorBT's optimized indicators.
    
    Provides comprehensive volume analysis including:
    - OBV-based features
    - Accumulation/Distribution analysis
    - Money Flow Index and related indicators
    - VWAP analysis
    - Volume momentum and rate of change
    - Volume profile analysis
    - Volume clustering detection
    """
    
    def __init__(self, config: Optional[VolumeConfig] = None, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize advanced volume feature generator.
        
        Args:
            config: Volume configuration
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        self.config = config or VolumeConfig()
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        
        # Create feature config
        feature_config = FeatureConfig(
            name="advanced_volume_features",
            category=FeatureCategory.VOLUME,
            description="Advanced volume features using VectorBT indicators",
            required_columns=["close", "volume"],
            optional_columns=["high", "low", "open"],
            default_lookback=max(self.config.obv_periods + self.config.ad_periods + self.config.mfi_periods),
            min_lookback=14,
            max_lookback=200,
            parameters={
                "obv_periods": self.config.obv_periods,
                "ad_periods": self.config.ad_periods,
                "mfi_periods": self.config.mfi_periods,
                "vwap_periods": self.config.vwap_periods,
                "enable_volume_profile": self.config.enable_volume_profile,
                "enable_volume_clustering": self.config.enable_volume_clustering
            },
            matrix_optimized=True,
            gpu_accelerated=self.enable_gpu
        )
        
        super().__init__(feature_config, enable_gpu=self.enable_gpu, enable_parallel=self.enable_parallel)
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate comprehensive volume features using VectorBT.
        
        Args:
            data: OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with volume features
        """
        if not VECTORBT_AVAILABLE:
            logger.warning("VectorBT not available, falling back to basic volume features")
            return self._generate_basic_volume_features(data)
        
        try:
            features = pd.DataFrame(index=data.index)
            
            # Generate OBV-based features
            features = self._generate_obv_features(data, features)
            
            # Generate Accumulation/Distribution features
            features = self._generate_ad_features(data, features)
            
            # Generate Money Flow Index features
            features = self._generate_mfi_features(data, features)
            
            # Generate VWAP features
            features = self._generate_vwap_features(data, features)
            
            # Generate volume momentum features
            features = self._generate_volume_momentum_features(data, features)
            
            # Generate volume rate of change features
            features = self._generate_volume_roc_features(data, features)
            
            # Generate volume profile features
            if self.config.enable_volume_profile:
                features = self._generate_volume_profile_features(data, features)
            
            # Generate volume clustering features
            if self.config.enable_volume_clustering:
                features = self._generate_volume_clustering_features(data, features)
            
            # Generate volume regime analysis features
            if self.config.enable_volume_regime_analysis:
                features = self._generate_volume_regime_features(data, features)
            
            # Generate advanced volume statistical features
            features = self._generate_advanced_volume_statistical_features(data, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating advanced volume features: {e}")
            return self._generate_basic_volume_features(data)
    
    def _generate_obv_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate On-Balance Volume (OBV) features."""
        try:
            # Calculate OBV using VectorBT
            obv = OBV.run(data['close'], data['volume'])
            
            # Basic OBV
            features['obv'] = obv.obv
            
            # OBV moving averages
            for period in self.config.obv_sma_periods:
                features[f'obv_sma_{period}'] = rolling_mean(obv.obv, window=period)
                features[f'obv_ema_{period}'] = vbt.MA.run(obv.obv, window=period, short_window=period//2).ma
            
            # OBV rate of change
            for period in self.config.obv_periods:
                features[f'obv_roc_{period}'] = rolling_apply(
                    obv.obv, 
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
                    window=period
                )
            
            # OBV divergence detection
            price_roc = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=20)
            obv_roc = rolling_apply(obv.obv, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=20)
            features['obv_divergence'] = np.where(
                (price_roc > 0) & (obv_roc < 0), -1,  # Bearish divergence
                np.where((price_roc < 0) & (obv_roc > 0), 1, 0)  # Bullish divergence
            )
            
            # OBV momentum
            features['obv_momentum'] = rolling_apply(
                obv.obv, 
                lambda x: x.iloc[-1] - x.iloc[0], 
                window=10
            )
            
            # OBV volatility
            features['obv_volatility'] = rolling_std(obv.obv, window=20)
            
            # OBV position relative to recent range
            obv_high = rolling_max(obv.obv, window=50)
            obv_low = rolling_min(obv.obv, window=50)
            features['obv_position'] = (obv.obv - obv_low) / (obv_high - obv_low)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating OBV features: {e}")
            return features
    
    def _generate_ad_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate Accumulation/Distribution (AD) features."""
        try:
            # Calculate AD using VectorBT
            ad = AD.run(data['high'], data['low'], data['close'], data['volume'])
            
            # Basic AD
            features['ad'] = ad.ad
            
            # AD moving averages
            for period in self.config.ad_sma_periods:
                features[f'ad_sma_{period}'] = rolling_mean(ad.ad, window=period)
                features[f'ad_ema_{period}'] = vbt.MA.run(ad.ad, window=period, short_window=period//2).ma
            
            # AD rate of change
            for period in self.config.ad_periods:
                features[f'ad_roc_{period}'] = rolling_apply(
                    ad.ad, 
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
                    window=period
                )
            
            # AD divergence detection
            price_roc = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=20)
            ad_roc = rolling_apply(ad.ad, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=20)
            features['ad_divergence'] = np.where(
                (price_roc > 0) & (ad_roc < 0), -1,  # Bearish divergence
                np.where((price_roc < 0) & (ad_roc > 0), 1, 0)  # Bullish divergence
            )
            
            # AD momentum
            features['ad_momentum'] = rolling_apply(
                ad.ad, 
                lambda x: x.iloc[-1] - x.iloc[0], 
                window=10
            )
            
            # AD volatility
            features['ad_volatility'] = rolling_std(ad.ad, window=20)
            
            # AD position relative to recent range
            ad_high = rolling_max(ad.ad, window=50)
            ad_low = rolling_min(ad.ad, window=50)
            features['ad_position'] = (ad.ad - ad_low) / (ad_high - ad_low)
            
            # AD oscillator
            ad_osc = ADOSC.run(data['high'], data['low'], data['close'], data['volume'], 
                             fast_period=3, slow_period=10)
            features['ad_oscillator'] = ad_osc.adosc
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating AD features: {e}")
            return features
    
    def _generate_mfi_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate Money Flow Index (MFI) features."""
        try:
            for period in self.config.mfi_periods:
                # Calculate MFI using VectorBT
                mfi = MFI.run(data['high'], data['low'], data['close'], data['volume'], window=period)
                
                # Basic MFI
                features[f'mfi_{period}'] = mfi.mfi
                
                # MFI overbought/oversold signals
                for threshold in self.config.mfi_thresholds:
                    features[f'mfi_overbought_{period}_{threshold}'] = (mfi.mfi > threshold).astype(int)
                    features[f'mfi_oversold_{period}_{threshold}'] = (mfi.mfi < (100 - threshold)).astype(int)
                
                # MFI momentum
                features[f'mfi_momentum_{period}'] = rolling_apply(
                    mfi.mfi, 
                    lambda x: x.iloc[-1] - x.iloc[0], 
                    window=period//2
                )
                
                # MFI volatility
                features[f'mfi_volatility_{period}'] = rolling_std(mfi.mfi, window=period)
                
                # MFI divergence detection
                price_roc = rolling_apply(data['close'], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=period)
                mfi_roc = rolling_apply(mfi.mfi, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0], window=period)
                features[f'mfi_divergence_{period}'] = np.where(
                    (price_roc > 0) & (mfi_roc < 0), -1,  # Bearish divergence
                    np.where((price_roc < 0) & (mfi_roc > 0), 1, 0)  # Bullish divergence
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating MFI features: {e}")
            return features
    
    def _generate_vwap_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate Volume-Weighted Average Price (VWAP) features."""
        try:
            for period in self.config.vwap_periods:
                for vwap_type in self.config.vwap_types:
                    if vwap_type == 'standard':
                        # Standard VWAP
                        vwap = (data['close'] * data['volume']).rolling(window=period).sum() / data['volume'].rolling(window=period).sum()
                        features[f'vwap_{period}'] = vwap
                        
                        # VWAP deviation
                        features[f'vwap_deviation_{period}'] = (data['close'] - vwap) / vwap
                        
                        # VWAP position
                        vwap_high = rolling_max(vwap, window=period)
                        vwap_low = rolling_min(vwap, window=period)
                        features[f'vwap_position_{period}'] = (vwap - vwap_low) / (vwap_high - vwap_low)
                        
                    elif vwap_type == 'exponential':
                        # Exponential VWAP
                        alpha = 2 / (period + 1)
                        vwap_ema = data['close'].ewm(alpha=alpha).mean()
                        features[f'vwap_ema_{period}'] = vwap_ema
                        
                        # VWAP EMA deviation
                        features[f'vwap_ema_deviation_{period}'] = (data['close'] - vwap_ema) / vwap_ema
                        
                    elif vwap_type == 'volume_weighted':
                        # Volume-weighted VWAP with additional weighting
                        volume_weight = data['volume'] / data['volume'].rolling(window=period).mean()
                        weighted_price = data['close'] * volume_weight
                        vwap_weighted = weighted_price.rolling(window=period).sum() / volume_weight.rolling(window=period).sum()
                        features[f'vwap_weighted_{period}'] = vwap_weighted
                        
                        # Weighted VWAP deviation
                        features[f'vwap_weighted_deviation_{period}'] = (data['close'] - vwap_weighted) / vwap_weighted
                
                # VWAP bands
                vwap = features[f'vwap_{period}']
                vwap_std = rolling_std(data['close'], window=period)
                features[f'vwap_upper_band_{period}'] = vwap + (2 * vwap_std)
                features[f'vwap_lower_band_{period}'] = vwap - (2 * vwap_std)
                
                # VWAP band position
                features[f'vwap_band_position_{period}'] = (data['close'] - features[f'vwap_lower_band_{period}']) / (features[f'vwap_upper_band_{period}'] - features[f'vwap_lower_band_{period}'])
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating VWAP features: {e}")
            return features
    
    def _generate_volume_momentum_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volume momentum features."""
        try:
            for period in self.config.volume_momentum_periods:
                # Volume momentum
                features[f'volume_momentum_{period}'] = rolling_apply(
                    data['volume'], 
                    lambda x: x.iloc[-1] - x.iloc[0], 
                    window=period
                )
                
                # Volume momentum rate
                features[f'volume_momentum_rate_{period}'] = rolling_apply(
                    data['volume'], 
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
                    window=period
                )
                
                # Volume acceleration
                volume_momentum = rolling_apply(data['volume'], lambda x: x.iloc[-1] - x.iloc[0], window=period)
                features[f'volume_acceleration_{period}'] = rolling_apply(
                    volume_momentum, 
                    lambda x: x.iloc[-1] - x.iloc[0], 
                    window=period//2
                )
                
                # Volume trend
                features[f'volume_trend_{period}'] = rolling_apply(
                    data['volume'], 
                    lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1, 
                    window=period
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volume momentum features: {e}")
            return features
    
    def _generate_volume_roc_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volume rate of change features."""
        try:
            for period in self.config.volume_roc_periods:
                # Volume rate of change
                features[f'volume_roc_{period}'] = rolling_apply(
                    data['volume'], 
                    lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
                    window=period
                )
                
                # Volume rate of change momentum
                features[f'volume_roc_momentum_{period}'] = rolling_apply(
                    features[f'volume_roc_{period}'], 
                    lambda x: x.iloc[-1] - x.iloc[0], 
                    window=period
                )
                
                # Volume rate of change volatility
                features[f'volume_roc_volatility_{period}'] = rolling_std(
                    features[f'volume_roc_{period}'], 
                    window=period * 2
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volume ROC features: {e}")
            return features
    
    def _generate_volume_profile_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volume profile features."""
        try:
            # Volume profile analysis
            price_bins = 20  # Number of price bins for volume profile
            
            for period in [20, 50, 100]:
                # Create price bins
                price_min = data['close'].rolling(window=period).min()
                price_max = data['close'].rolling(window=period).max()
                price_range = price_max - price_min
                
                # Volume profile
                volume_profile = rolling_apply(
                    data[['close', 'volume']], 
                    lambda x: self._calculate_volume_profile(x, price_bins), 
                    window=period
                )
                
                features[f'volume_profile_{period}'] = volume_profile
                
                # Volume profile concentration
                features[f'volume_profile_concentration_{period}'] = rolling_apply(
                    volume_profile, 
                    lambda x: x.max() / x.sum() if x.sum() > 0 else 0, 
                    window=period//2
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volume profile features: {e}")
            return features
    
    def _calculate_volume_profile(self, data: pd.DataFrame, bins: int) -> float:
        """Calculate volume profile for a given period."""
        try:
            if len(data) < 2:
                return 0.0
            
            # Create price bins
            price_min = data['close'].min()
            price_max = data['close'].max()
            if price_max == price_min:
                return 0.0
            
            bin_edges = np.linspace(price_min, price_max, bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Assign volumes to bins
            bin_volumes = np.zeros(bins)
            for _, row in data.iterrows():
                bin_idx = np.digitize(row['close'], bin_edges) - 1
                bin_idx = max(0, min(bin_idx, bins - 1))
                bin_volumes[bin_idx] += row['volume']
            
            # Return the bin with maximum volume
            return bin_centers[np.argmax(bin_volumes)]
            
        except Exception:
            return 0.0
    
    def _generate_volume_clustering_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volume clustering features."""
        try:
            # Volume clustering detection
            volume_short = rolling_mean(data['volume'], window=5)
            volume_long = rolling_mean(data['volume'], window=20)
            
            # Volume clustering ratio
            features['volume_cluster_ratio'] = volume_short / volume_long
            
            # Volume clustering momentum
            features['volume_cluster_momentum'] = rolling_apply(
                volume_short, 
                lambda x: x.iloc[-1] - x.iloc[0], 
                window=10
            )
            
            # Volume clustering persistence
            features['volume_cluster_persistence'] = rolling_apply(
                volume_short, 
                lambda x: (x > x.rolling(window=5).mean()).sum() / len(x), 
                window=20
            )
            
            # Volume clustering regime detection
            volume_threshold = volume_long.quantile(0.7)
            features['volume_cluster_regime'] = (volume_short > volume_threshold).astype(int)
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volume clustering features: {e}")
            return features
    
    def _generate_volume_regime_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate volume regime analysis features."""
        try:
            # Volume regimes based on different timeframes
            volume_short = rolling_mean(data['volume'], window=5)
            volume_medium = rolling_mean(data['volume'], window=20)
            volume_long = rolling_mean(data['volume'], window=50)
            
            # Regime classification
            high_vol_threshold = volume_medium.quantile(0.7)
            low_vol_threshold = volume_medium.quantile(0.3)
            
            features['volume_regime'] = np.where(
                volume_medium > high_vol_threshold, 2,  # High volume
                np.where(volume_medium < low_vol_threshold, 0, 1)  # Low volume, Medium volume
            )
            
            # Regime persistence
            features['volume_regime_persistence'] = rolling_apply(
                features['volume_regime'], 
                lambda x: (x == x.iloc[-1]).sum() / len(x), 
                window=20
            )
            
            # Regime transition probability
            features['volume_regime_transition'] = rolling_apply(
                features['volume_regime'], 
                lambda x: 1 if x.iloc[-1] != x.iloc[0] else 0, 
                window=10
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating volume regime features: {e}")
            return features
    
    def _generate_advanced_volume_statistical_features(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced volume statistical features."""
        try:
            # Volume statistics
            features['volume_skewness'] = rolling_apply(data['volume'], lambda x: x.skew(), window=20)
            features['volume_kurtosis'] = rolling_apply(data['volume'], lambda x: x.kurtosis(), window=20)
            
            # Volume percentiles
            features['volume_percentile_25'] = rolling_apply(data['volume'], lambda x: x.quantile(0.25), window=20)
            features['volume_percentile_75'] = rolling_apply(data['volume'], lambda x: x.quantile(0.75), window=20)
            features['volume_percentile_90'] = rolling_apply(data['volume'], lambda x: x.quantile(0.90), window=20)
            
            # Volume mean reversion
            volume_mean = rolling_mean(data['volume'], window=20)
            features['volume_mean_reversion'] = (data['volume'] - volume_mean) / volume_mean
            
            # Volume-price correlation
            features['volume_price_correlation'] = rolling_corr(data['volume'], data['close'], window=20)
            
            # Volume volatility
            features['volume_volatility'] = rolling_std(data['volume'], window=20)
            
            # Volume relative strength
            volume_sma = rolling_mean(data['volume'], window=20)
            features['volume_relative_strength'] = data['volume'] / volume_sma
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating advanced volume statistical features: {e}")
            return features
    
    def _generate_basic_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate basic volume features as fallback."""
        features = pd.DataFrame(index=data.index)
        
        # Basic volume moving averages
        features['volume_sma_20'] = data['volume'].rolling(window=20).mean()
        features['volume_sma_50'] = data['volume'].rolling(window=50).mean()
        
        # Basic volume rate of change
        features['volume_roc_5'] = data['volume'].pct_change(5)
        features['volume_roc_20'] = data['volume'].pct_change(20)
        
        # Basic volume volatility
        features['volume_volatility'] = data['volume'].rolling(window=20).std()
        
        return features


def create_advanced_volume_generator(
    config: Optional[VolumeConfig] = None,
    enable_gpu: bool = False,
    enable_parallel: bool = True
) -> AdvancedVolumeFeatures:
    """
    Create an advanced volume feature generator.
    
    Args:
        config: Volume configuration
        enable_gpu: Whether to enable GPU acceleration
        enable_parallel: Whether to enable parallel processing
        
    Returns:
        AdvancedVolumeFeatures instance
    """
    return AdvancedVolumeFeatures(config, enable_gpu, enable_parallel)


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
    generator = create_advanced_volume_generator(enable_gpu=False, enable_parallel=True)
    features = generator.generate_features(data)
    
    print(f"Generated {len(features.columns)} volume features")
    print("Feature names:", list(features.columns))
    print("\nFirst few rows:")
    print(features.head())