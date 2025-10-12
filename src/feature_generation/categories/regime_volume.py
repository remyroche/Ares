"""
Regime Volume Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on volume regime
characteristics rather than short-term trading signals.

Key Features:
- Volume regime persistence and stability
- Volume clustering patterns
- Volume-price relationship consistency
- Volume regime transitions
- Volume regime strength indicators
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    from ..utils.unified_optimization_system import get_unified_optimization_system, UnifiedOptimizationSystem
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Import tprint for consistent logging
from src.utils.tprint import tprint

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

class RegimeVolumeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volume regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizers
        self.vectorbt_optimizer = None
        self.unified_optimizer = None
        if OPTIMIZATION_AVAILABLE:
            try:
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True
                )
                self.unified_optimizer = get_unified_optimization_system()
                tprint("✅ VectorBT optimizers initialized for RegimeVolumeFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_volume_features",
            category=FeatureCategory.VOLUME,
            description="Volume regime features for 15m timeframe regime classification",
            required_columns=["volume"],
            optional_columns=["close", "high", "low", "open"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 30, 80],  # 3h, 7.5h, 20h in 15m periods (original min, middle, new max)
                "persistence_windows": [8, 20, 64],  # 2h, 5h, 16h (original min, middle, new max)
                "clustering_windows": [16, 40, 128],  # 4h, 10h, 32h (original min, middle, new max)
                "transition_windows": [4, 12, 32]  # 1h, 3h, 8h (original min, middle, new max)
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single volume regime feature as required by the base class."""
        try:
            # Generate all volume features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple volume feature if no features generated
                if 'volume' in data.columns and len(data) > 0:
                    volume_feature = data['volume'].pct_change().fillna(0).values
                    return pd.Series(volume_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            tprint(f"_generate_feature: Volume feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate volume regime features."""
        features = {}
        
        # Validate volume data
        if 'volume' not in data.columns:
            return features
        
        volume = data['volume'].values
        if len(volume) < 4:
            return features
        
        # 1. Volume Regime Persistence
        features.update(self._generate_volume_persistence_features(volume, data))
        
        # 2. Volume Clustering Features
        features.update(self._generate_volume_clustering_features(volume, data))
        
        # 3. Volume-Price Relationship Features
        features.update(self._generate_volume_price_features(volume, data))
        
        # 4. Volume Regime Transitions
        features.update(self._generate_volume_transition_features(volume, data))
        
        # 5. Volume Regime Stability
        features.update(self._generate_volume_stability_features(volume, data))
        
        return features
    
    def _generate_volume_persistence_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume persistence features for regime detection."""
        features = {}
        windows = self.config.parameters["regime_windows"]
        
        for window in windows:
            if len(volume) < window:
                continue
            
            # Rolling volume statistics
            vol_mean = self._rolling_mean(volume, window)
            vol_std = self._rolling_std(volume, window)
            
            # Volume persistence (autocorrelation of volume)
            vol_persistence = self._calculate_volume_persistence(volume, window)
            
            # Volume regime strength
            vol_regime_strength = self._calculate_volume_regime_strength(volume, window)
            
            # Volume regime consistency
            vol_consistency = self._calculate_volume_consistency(volume, window)
            
            # Pad to match data length
            vol_mean_padded = np.full(len(data), np.nan)
            vol_std_padded = np.full(len(data), np.nan)
            vol_persistence_padded = np.full(len(data), np.nan)
            vol_strength_padded = np.full(len(data), np.nan)
            vol_consistency_padded = np.full(len(data), np.nan)
            
            # Rolling functions return len(volume) - window + 1, aligned to volume[window-1:]
            vol_mean_padded[window-1:] = vol_mean
            vol_std_padded[window-1:] = vol_std
            # Full-length functions return len(volume) with valid values from index window onwards
            vol_persistence_padded[window:] = vol_persistence[window:]
            vol_strength_padded[window:] = vol_regime_strength[window:]
            vol_consistency_padded[window:] = vol_consistency[window:]
            
            features[f'vol_mean_{window}'] = vol_mean_padded
            features[f'vol_std_{window}'] = vol_std_padded
            features[f'vol_persistence_{window}'] = vol_persistence_padded
            features[f'vol_regime_strength_{window}'] = vol_strength_padded
            features[f'vol_consistency_{window}'] = vol_consistency_padded
        
        return features
    
    def _generate_volume_clustering_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume clustering features."""
        features = {}
        windows = self.config.parameters["clustering_windows"]
        
        for window in windows:
            if len(volume) < window:
                continue
            
            # Volume clustering (similar to volatility clustering)
            vol_clustering = self._calculate_volume_clustering(volume, window)
            
            # Volume regime patterns
            vol_patterns = self._calculate_volume_patterns(volume, window)
            
            # Volume regime intensity
            vol_intensity = self._calculate_volume_intensity(volume, window)
            
            # Pad to match data length
            clustering_padded = np.full(len(data), np.nan)
            patterns_padded = np.full(len(data), np.nan)
            intensity_padded = np.full(len(data), np.nan)
            
            # Functions return len(volume) with valid values from index window onwards
            clustering_padded[window:] = vol_clustering[window:]
            patterns_padded[window:] = vol_patterns[window:]
            intensity_padded[window:] = vol_intensity[window:]
            
            features[f'vol_clustering_{window}'] = clustering_padded
            features[f'vol_patterns_{window}'] = patterns_padded
            features[f'vol_intensity_{window}'] = intensity_padded
        
        return features
    
    def _generate_volume_price_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume-price relationship features."""
        features = {}
        windows = self.config.parameters["regime_windows"]

        if 'close' not in data.columns:
            return features

        close_prices = data['close'].values
        if len(close_prices) != len(volume):
            # Log the mismatch and return empty features - this shouldn't happen
            tprint(f"⚠️ WARNING: Length mismatch detected - volume={len(volume)}, close_prices={len(close_prices)}")
            tprint(f"⚠️ This indicates a data integrity issue. Skipping volume-price features.")
            return features
        
        for window in windows:
            if len(volume) < window:
                continue
            
            # Volume-price correlation
            vol_price_corr = self._calculate_volume_price_correlation(volume, close_prices, window)
            
            # Volume-weighted price change
            vol_weighted_price = self._calculate_volume_weighted_price_change(volume, close_prices, window)
            
            # Volume regime price impact
            vol_price_impact = self._calculate_volume_price_impact(volume, close_prices, window)
            
            # Pad to match data length
            corr_padded = np.full(len(data), np.nan)
            weighted_padded = np.full(len(data), np.nan)
            impact_padded = np.full(len(data), np.nan)
            
            # Functions return len(volume) with valid values from index window onwards
            corr_padded[window:] = vol_price_corr[window:]
            weighted_padded[window:] = vol_weighted_price[window:]
            impact_padded[window:] = vol_price_impact[window:]
            
            features[f'vol_price_corr_{window}'] = corr_padded
            features[f'vol_weighted_price_{window}'] = weighted_padded
            features[f'vol_price_impact_{window}'] = impact_padded
        
        return features
    
    def _generate_volume_transition_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(volume) < window * 2:
                continue
            
            # Volume regime change detection
            vol_change = self._detect_volume_regime_changes(volume, window)
            
            # Volume regime transition probability
            transition_prob = self._calculate_volume_transition_probability(volume, window)
            
            # Volume regime momentum
            vol_momentum = self._calculate_volume_momentum(volume, window)
            
            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)
            momentum_padded = np.full(len(data), np.nan)
            
            # Functions return len(volume) with valid values from index window*2 onwards
            change_padded[window*2:] = vol_change[window*2:]
            prob_padded[window*2:] = transition_prob[window*2:]
            momentum_padded[window*2:] = vol_momentum[window*2:]
            
            features[f'vol_regime_change_{window}'] = change_padded
            features[f'vol_transition_prob_{window}'] = prob_padded
            features[f'vol_momentum_{window}'] = momentum_padded
        
        return features
    
    def _generate_volume_stability_features(self, volume: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volume regime stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(volume) < window:
                continue
            
            # Volume regime stability
            vol_stability = self._calculate_volume_stability(volume, window)
            
            # Volume regime persistence score
            persistence_score = self._calculate_volume_persistence_score(volume, window)
            
            # Volume regime entropy
            vol_entropy = self._calculate_volume_entropy(volume, window)
            
            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)
            entropy_padded = np.full(len(data), np.nan)
            
            # These functions return len(volume) with valid values from index window onwards
            stability_padded[window:] = vol_stability[window:]
            persistence_padded[window:] = persistence_score[window:]
            entropy_padded[window:] = vol_entropy[window:]
            
            features[f'vol_stability_{window}'] = stability_padded
            features[f'vol_persistence_score_{window}'] = persistence_padded
            features[f'vol_entropy_{window}'] = entropy_padded
        
        return features
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean - VECTORIZED."""
        if len(data) < window:
            return np.array([])
        
        # Vectorized approach using pandas rolling
        data_series = pd.Series(data)
        result = self._vectorbt_rolling_operation(data_series, "mean", window).dropna().values
        
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation - VECTORIZED."""
        if len(data) < window:
            return np.array([])
        
        # Vectorized approach using pandas rolling
        data_series = pd.Series(data)
        result = self._vectorbt_rolling_operation(data_series, "std", window).dropna().values
        
        return result
    
    def _calculate_volume_persistence(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence using autocorrelation - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        persistence = np.zeros(len(volume))
        
        # Calculate autocorrelation using vectorized operations
        vol_series = pd.Series(volume)
        
        # Vectorized autocorrelation using rolling correlation with shifted series
        vol_shifted = vol_series.shift(1)
        autocorr = vol_series.rolling(window=window).corr(vol_shifted).fillna(0)
        
        persistence[window:] = autocorr[window:]
        return persistence
    
    def _calculate_volume_regime_strength(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime strength - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # Vectorized regime strength calculation
        vol_series = pd.Series(volume)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)
        
        # Regime strength based on consistency of volume level
        vol_consistency = 1.0 - (rolling_std / (rolling_mean + 1e-8))
        strength = vol_consistency.clip(0, 1)
        
        return strength.fillna(0).values
    
    def _calculate_volume_consistency(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime consistency - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # Vectorized consistency calculation
        vol_series = pd.Series(volume)
        rolling_std = self._vectorbt_rolling_operation(vol_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)
        
        # Consistency based on low coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)
        
        return consistency.fillna(0).values
    
    def _calculate_volume_clustering(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume clustering - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        vol_series = pd.Series(volume)
        
        # Vectorized autocorrelation using rolling correlation with shifted series
        vol_shifted = vol_series.shift(1)
        clustering = vol_series.rolling(window=window).corr(vol_shifted).fillna(0)
        
        return clustering.values
    
    def _calculate_volume_patterns(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime patterns - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized pattern calculation
        vol_series = pd.Series(volume)
        
        # Vectorized trend calculation using rolling linear regression
        # Create index arrays for linear regression
        x = np.arange(window)
        
        # Calculate rolling trend using vectorized operations
        patterns = np.full(len(volume), 0.5)  # Default value
        
        for i in range(window, len(volume) + 1):
            vol_window = volume[i-window:i]
            if len(vol_window) >= 3:
                # Calculate trend using linear regression
                trend = np.polyfit(x, vol_window, 1)[0]
                # Normalize trend to 0-1 range
                mean_vol = np.mean(vol_window)
                if mean_vol > 0:
                    normalized_trend = (np.tanh(trend / mean_vol) + 1) / 2
                    patterns[i-1] = normalized_trend
        
        return patterns
    
    def _calculate_trend_pattern(self, vol_window: pd.Series) -> float:
        """Calculate trend pattern for a volume window."""
        if len(vol_window) < 3:
            return 0.5
        
        # Pattern based on volume trend
        x = np.arange(len(vol_window))
        trend = np.polyfit(x, vol_window, 1)[0]
        # Normalize trend to 0-1 range
        mean_vol = vol_window.mean()
        return (np.tanh(trend / (mean_vol + 1e-8)) + 1) / 2
    
    def _calculate_volume_intensity(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime intensity - VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # Vectorized intensity calculation
        vol_series = pd.Series(volume)
        rolling_mean = self._vectorbt_rolling_operation(vol_series, "mean", window)
        
        # Intensity based on volume relative to historical average
        intensity = (vol_series / (rolling_mean + 1e-8)).clip(0, 2)
        
        return intensity.fillna(0).values
    
    def _calculate_volume_price_correlation(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-price correlation - VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        # Vectorized correlation calculation
        vol_series = pd.Series(volume)
        price_series = pd.Series(prices)
        
        # Fix: corr() method expects a Series, not a Rolling object
        correlation = vol_series.rolling(window=window).corr(price_series).fillna(0)
        
        return correlation.values
    
    def _calculate_volume_weighted_price_change(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-weighted price change - OPTIMIZED VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        prices_series = pd.Series(prices)
        
        # Calculate price changes
        price_changes = prices_series.diff()
        
        # Vectorized volume-weighted calculation
        # Calculate rolling volume sums and price change sums
        vol_sum = self._vectorbt_rolling_operation(volume_series, "sum", window)
        vol_price_sum = (volume_series * price_changes).rolling(window=window).sum()
        
        # Volume-weighted price change
        weighted_change = vol_price_sum / (vol_sum + 1e-8)
        
        return weighted_change.fillna(0).values
    
    def _calculate_weighted_change_window(self, vol_window: pd.Series, price_changes_window: np.ndarray) -> float:
        """Helper function for volume-weighted price change calculation."""
        if len(vol_window) < 2 or len(price_changes_window) < 1:
            return 0.0
        
        # Ensure we have matching lengths
        min_len = min(len(vol_window), len(price_changes_window))
        vol_vals = vol_window.iloc[:min_len].values
        price_changes_vals = price_changes_window[:min_len]
        
        # Calculate weights
        weights = vol_vals[1:] / (np.sum(vol_vals[1:]) + 1e-8)
        
        # Calculate weighted change
        if len(weights) > 0 and len(price_changes_vals[1:]) > 0:
            return np.sum(price_changes_vals[1:] * weights)
        return 0.0
    
    def _calculate_volume_price_impact(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume price impact - OPTIMIZED VECTORIZED."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        prices_series = pd.Series(prices)
        
        # Calculate price and volume changes
        price_changes = prices_series.diff().abs()
        vol_changes = volume_series.diff()
        
        # Vectorized impact calculation
        # Calculate rolling correlation between volume changes and price changes
        impact = vol_changes.rolling(window=window).corr(price_changes).fillna(0)
        
        return impact.values
    
    def _calculate_impact_window(self, vol_window: pd.Series, price_changes_window: np.ndarray) -> float:
        """Helper function for volume price impact calculation."""
        if len(vol_window) < 2 or len(price_changes_window) < 1:
            return 0.0
        
        # Ensure we have matching lengths
        min_len = min(len(vol_window), len(price_changes_window))
        vol_vals = vol_window.iloc[:min_len].values
        price_changes_vals = price_changes_window[:min_len]
        
        # Calculate volume changes
        vol_changes = np.diff(vol_vals)
        price_changes = price_changes_vals[1:]  # Skip first element to match vol_changes
        
        # Avoid division by zero
        vol_changes = np.where(vol_changes == 0, 1e-8, vol_changes)
        
        # Calculate impact ratio
        if len(vol_changes) > 0 and len(price_changes) > 0:
            impact_ratio = price_changes / vol_changes
            return np.mean(impact_ratio)
        return 0.0
    
    def _detect_volume_regime_changes(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Detect volume regime changes - VECTORIZED."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))
        
        # Vectorized approach using pandas rolling
        volume_series = pd.Series(volume)
        
        # Calculate rolling means for both windows
        vol1 = self._vectorbt_rolling_operation(volume_series, "mean", window)
        vol2 = vol1.shift(-window)  # Second window
        
        # Calculate change ratios
        change_ratios = ((vol2 - vol1).abs() / (vol1 + 1e-8)).fillna(0)
        
        # Apply threshold (30% change)
        changes = (change_ratios > 0.3).astype(int)
        
        return changes.fillna(0).values
    
    def _calculate_volume_transition_probability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime transition probability - OPTIMIZED VECTORIZED."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        
        # Calculate rolling volume changes for transition probability
        vol_changes = volume_series.diff()
        vol_volatility = self._vectorbt_rolling_operation(vol_changes, "std", window)
        vol_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)
        
        # Vectorized transition probability
        transition_prob = vol_volatility / (vol_mean + 1e-8)
        transition_prob = transition_prob.clip(0, 1)
        
        return transition_prob.fillna(0).values
    
    def _calculate_transition_prob_window(self, vol_window: pd.Series) -> float:
        """Helper function for volume transition probability calculation."""
        if len(vol_window) < 2:
            return 0.0
        
        try:
            # Calculate trend using linear regression
            x = np.arange(len(vol_window))
            trend = np.polyfit(x, vol_window, 1)[0]
            mean_vol = vol_window.mean()
            
            # Probability based on trend strength
            prob = min(1, max(0, abs(trend) / (mean_vol + 1e-8)))
            return prob
        except:
            return 0.0
    
    def _calculate_volume_momentum(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume momentum - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        
        # Calculate volume changes for momentum
        vol_changes = volume_series.diff()
        vol_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)
        
        # Vectorized momentum calculation
        momentum = self._vectorbt_rolling_operation(vol_changes, "mean", window) / (vol_mean + 1e-8)
        
        return momentum.fillna(0).values
    
    def _calculate_momentum_window(self, vol_window: pd.Series) -> float:
        """Helper function for volume momentum calculation."""
        if len(vol_window) < 2:
            return 0.0
        
        try:
            # Calculate trend using linear regression
            x = np.arange(len(vol_window))
            trend = np.polyfit(x, vol_window, 1)[0]
            mean_vol = vol_window.mean()
            
            # Momentum based on trend strength
            momentum = trend / (mean_vol + 1e-8)
            return momentum
        except:
            return 0.0
    
    def _calculate_volume_stability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime stability - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        
        # Calculate rolling statistics for stability
        rolling_std = self._vectorbt_rolling_operation(volume_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(volume_series, "mean", window)
        
        # Vectorized stability calculation using coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)
        
        return stability.fillna(0).values
    
    def _calculate_stability_window(self, vol_window: pd.Series) -> float:
        """Helper function for volume stability calculation."""
        if len(vol_window) < 2:
            return 0.0
        
        try:
            # Stability based on low coefficient of variation
            cv = vol_window.std() / (vol_window.mean() + 1e-8)
            stability = max(0, 1 - cv)
            return stability
        except:
            return 0.0
    
    def _calculate_volume_persistence_score(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence score - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        
        # Calculate volume changes for autocorrelation
        vol_changes = volume_series.diff()
        
        # Vectorized persistence using rolling correlation with shifted series
        vol_changes_shifted = vol_changes.shift(1)
        persistence = vol_changes.rolling(window=window).corr(vol_changes_shifted).fillna(0)
        
        return persistence.values
    
    def _calculate_persistence_window(self, vol_window: pd.Series) -> float:
        """Helper function for volume persistence calculation."""
        if len(vol_window) < 3:
            return 0.0
        
        try:
            # Persistence based on autocorrelation of volume
            corr = vol_window.autocorr(lag=1)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _calculate_volume_entropy(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime entropy - OPTIMIZED VECTORIZED."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        volume_series = pd.Series(volume)
        
        # Calculate volume changes for entropy
        vol_changes = volume_series.diff()
        
        # Vectorized entropy using rolling coefficient of variation
        rolling_std = self._vectorbt_rolling_operation(vol_changes, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(vol_changes, "mean", window).abs()
        
        # Entropy proxy using coefficient of variation
        entropy = rolling_std / (rolling_mean + 1e-8)
        
        return entropy.fillna(0).values
    
    def _calculate_entropy_window(self, vol_window: pd.Series) -> float:
        """Helper function for volume entropy calculation."""
        if len(vol_window) < 2:
            return 0.0
        
        try:
            # Calculate entropy of volume distribution
            # Discretize volume into bins
            bins = np.linspace(vol_window.min(), vol_window.max(), 10)
            hist, _ = np.histogram(vol_window, bins=bins)
            # Normalize to probabilities
            probs = hist / (np.sum(hist) + 1e-8)
            # Calculate entropy
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            return entropy
        except:
            return 0.0
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
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
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
            except Exception as e:
                tprint(f"VectorBT operation failed: {e}, using pandas fallback")
                return self._pandas_rolling_operation(data, operation, window, **kwargs)
        else:
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
