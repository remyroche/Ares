"""
Regime Structural Trend Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on structural trend
characteristics rather than short-term momentum or trading signals.

Key Features:
- Structural trend persistence and strength
- Trend regime transitions
- Market structure indicators
- Trend regime stability
- Long-term trend characteristics (not momentum)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

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
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

class RegimeStructuralTrendFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for structural trend regime features optimized for 15m timeframe."""
    
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
                tprint("✅ VectorBT optimizers initialized for RegimeStructuralTrendFeatureGenerator")
            except Exception as e:
                tprint(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_structural_trend_features",
            category=FeatureCategory.TREND,
            description="Structural trend regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=40,  # 10 hours in 15m periods
            min_lookback=8,       # 2 hours minimum
            max_lookback=160,     # 40 hours maximum
            parameters={
                "structural_windows": [20, 60, 160],  # 5h, 15h, 40h in 15m periods (original min, middle, new max)
                "persistence_windows": [16, 40, 128],  # 4h, 10h, 32h (original min, middle, new max)
                "transition_windows": [8, 20, 64],  # 2h, 5h, 16h (original min, middle, new max)
                "structure_windows": [24, 60, 192]  # 6h, 15h, 48h (original min, middle, new max)
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate a single structural trend feature as required by the base class."""
        try:
            # Generate all structural trend features
            features_dict = self.generate_features(data, **kwargs)
            
            # Combine all features into a single series (use first feature as representative)
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index[:len(features_dict[first_feature_name])])
            else:
                # Return a simple trend feature if no features generated
                if 'close' in data.columns and len(data) > 1:
                    trend_feature = data['close'].pct_change().fillna(0).values
                    return pd.Series(trend_feature, index=data.index)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index)
                
        except Exception as e:
            print(f"_generate_feature: Structural trend feature generation failed: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate structural trend regime features."""
        features = {}
        
        # Validate price data
        if 'close' not in data.columns:
            return features
        
        close_prices = data['close'].values
        if len(close_prices) < 8:
            return features
        
        # 1. Structural Trend Persistence
        features.update(self._generate_structural_persistence_features(close_prices, data))
        
        # 2. Trend Regime Strength
        features.update(self._generate_trend_strength_features(close_prices, data))
        
        # 3. Market Structure Features
        features.update(self._generate_market_structure_features(close_prices, data))
        
        # 4. Trend Regime Transitions
        features.update(self._generate_trend_transition_features(close_prices, data))
        
        # 5. Trend Regime Stability
        features.update(self._generate_trend_stability_features(close_prices, data))
        
        return features
    
    def _generate_structural_persistence_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate structural trend persistence features."""
        features = {}
        windows = self.config.parameters["structural_windows"]
        
        for window in windows:
            if len(prices) < window:
                continue
            
            # Structural trend persistence
            trend_persistence = self._calculate_structural_trend_persistence(prices, window)
            
            # Trend direction consistency
            direction_consistency = self._calculate_trend_direction_consistency(prices, window)
            
            # Trend regime persistence
            regime_persistence = self._calculate_trend_regime_persistence(prices, window)
            
            # Pad to match data length
            persistence_padded = np.full(len(data), np.nan)
            direction_padded = np.full(len(data), np.nan)
            regime_padded = np.full(len(data), np.nan)
            
            # Functions return len(prices) with valid values from index window onwards
            persistence_padded[window:] = trend_persistence[window:]
            direction_padded[window:] = direction_consistency[window:]
            regime_padded[window:] = regime_persistence[window:]
            
            features[f'structural_persistence_{window}'] = persistence_padded
            features[f'trend_direction_consistency_{window}'] = direction_padded
            features[f'trend_regime_persistence_{window}'] = regime_padded
        
        return features
    
    def _generate_trend_strength_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend strength features."""
        features = {}
        windows = self.config.parameters["structural_windows"]
        
        for window in windows:
            if len(prices) < window:
                continue
            
            # Structural trend strength
            trend_strength = self._calculate_structural_trend_strength(prices, window)
            
            # Trend acceleration
            trend_acceleration = self._calculate_trend_acceleration(prices, window)
            
            # Trend regime intensity
            trend_intensity = self._calculate_trend_intensity(prices, window)
            
            # Pad to match data length
            strength_padded = np.full(len(data), np.nan)
            acceleration_padded = np.full(len(data), np.nan)
            intensity_padded = np.full(len(data), np.nan)
            
            # Functions return len(prices) with valid values from index window onwards
            strength_padded[window:] = trend_strength[window:]
            acceleration_padded[window:] = trend_acceleration[window:]
            intensity_padded[window:] = trend_intensity[window:]
            
            features[f'structural_trend_strength_{window}'] = strength_padded
            features[f'trend_acceleration_{window}'] = acceleration_padded
            features[f'trend_intensity_{window}'] = intensity_padded
        
        return features
    
    def _generate_market_structure_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate market structure features."""
        features = {}
        windows = self.config.parameters["structure_windows"]
        
        for window in windows:
            if len(prices) < window:
                continue
            
            # Market structure strength
            structure_strength = self._calculate_market_structure_strength(prices, window)
            
            # Support/Resistance strength
            support_resistance = self._calculate_support_resistance_strength(prices, window)
            
            # Market structure consistency
            structure_consistency = self._calculate_market_structure_consistency(prices, window)
            
            # Pad to match data length
            strength_padded = np.full(len(data), np.nan)
            sr_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)
            
            # Functions return len(prices) with valid values from index window onwards
            strength_padded[window:] = structure_strength[window:]
            sr_padded[window:] = support_resistance[window:]
            consistency_padded[window:] = structure_consistency[window:]
            
            features[f'market_structure_strength_{window}'] = strength_padded
            features[f'support_resistance_strength_{window}'] = sr_padded
            features[f'market_structure_consistency_{window}'] = consistency_padded
        
        return features
    
    def _generate_trend_transition_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(prices) < window * 2:
                continue
            
            # Trend regime change detection
            trend_change = self._detect_trend_regime_changes(prices, window)
            
            # Trend transition probability
            transition_prob = self._calculate_trend_transition_probability(prices, window)
            
            # Trend regime momentum (structural, not trading)
            trend_momentum = self._calculate_structural_trend_momentum(prices, window)
            
            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)
            momentum_padded = np.full(len(data), np.nan)
            
            # trend_change and transition_prob have valid values from index window*2 onwards
            change_padded[window*2:] = trend_change[window*2:]
            prob_padded[window*2:] = transition_prob[window*2:]
            # trend_momentum has valid values from index window onwards
            momentum_padded[window:] = trend_momentum[window:]
            
            features[f'trend_regime_change_{window}'] = change_padded
            features[f'trend_transition_prob_{window}'] = prob_padded
            features[f'structural_trend_momentum_{window}'] = momentum_padded
        
        return features
    
    def _generate_trend_stability_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate trend stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(prices) < window:
                continue
            
            # Trend regime stability
            trend_stability = self._calculate_trend_stability(prices, window)
            
            # Trend persistence score
            persistence_score = self._calculate_trend_persistence_score(prices, window)
            
            # Trend regime entropy
            trend_entropy = self._calculate_trend_entropy(prices, window)
            
            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)
            entropy_padded = np.full(len(data), np.nan)
            
            # Functions return len(prices) with valid values from index window onwards
            stability_padded[window:] = trend_stability[window:]
            persistence_padded[window:] = persistence_score[window:]
            entropy_padded[window:] = trend_entropy[window:]
            
            features[f'trend_stability_{window}'] = stability_padded
            features[f'trend_persistence_score_{window}'] = persistence_padded
            features[f'trend_entropy_{window}'] = entropy_padded
        
        return features
    
    def _calculate_structural_trend_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend persistence - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        prices_series = pd.Series(prices)
        
        # Pre-calculate rolling statistics for efficiency
        rolling_mean = self._vectorbt_rolling_operation(prices_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(prices_series, "std", window)
        
        # Vectorized trend persistence using slope approximation
        x = np.arange(window)
        x_mean = x.mean()
        x_var = np.var(x)
        
        # OPTIMIZED: Use vectorized slope calculation
        # Calculate rolling slope using linear regression approximation
        rolling_slope = prices_series.rolling(window=window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
            raw=False
        ).fillna(0)
        
        # Persistence based on slope consistency (simplified)
        persistence = (rolling_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)
        
        return persistence.values
    
    def _calculate_trend_consistency(self, price_window: pd.Series) -> float:
        """Calculate trend consistency for a price window."""
        if len(price_window) < 2:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(price_window))
        slope, _ = np.polyfit(x, price_window, 1)
        
        # Persistence based on trend consistency
        trend_consistency = abs(slope) / (price_window.std() + 1e-8)
        return min(1, trend_consistency)
    
    def _calculate_trend_direction_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend direction consistency - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized operations instead of rolling apply
        prices_series = pd.Series(prices)
        
        # Calculate price changes vectorized
        price_changes = prices_series.diff()
        
        # OPTIMIZED: Use vectorized direction consistency calculation
        # Calculate rolling sums of positive and negative changes
        positive_changes = (price_changes > 0).rolling(window=window).sum().fillna(0)
        negative_changes = (price_changes < 0).rolling(window=window).sum().fillna(0)
        
        total_changes = positive_changes + negative_changes
        
        # Vectorized consistency calculation
        consistency = np.where(
            total_changes > 0,
            np.maximum(positive_changes, negative_changes) / total_changes,
            0
        )
        
        return consistency
    
    def _calculate_direction_consistency(self, price_window: pd.Series) -> float:
        """Calculate direction consistency for a price window."""
        if len(price_window) < 3:
            return 0.0
        
        # Calculate direction changes
        price_changes = price_window.diff().dropna()
        positive_changes = (price_changes > 0).sum()
        negative_changes = (price_changes < 0).sum()
        
        # Consistency based on direction dominance
        total_changes = positive_changes + negative_changes
        if total_changes > 0:
            return max(positive_changes, negative_changes) / total_changes
        return 0.0
    
    def _calculate_trend_regime_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend regime persistence - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified autocorrelation calculation
        prices_series = pd.Series(prices)
        
        # Calculate price changes for autocorrelation
        price_changes = prices_series.diff()
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        # Calculate rolling autocorrelation using pandas built-in method
        persistence = price_changes.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        return persistence.values
    
    def _calculate_trend_autocorrelation(self, price_window: pd.Series) -> float:
        """Calculate trend autocorrelation for a price window."""
        if len(price_window) < 3:
            return 0.0
        
        # Calculate trend autocorrelation
        x = np.arange(len(price_window))
        slopes = []
        for j in range(1, len(price_window)):
            if j > 1:
                slope, _ = np.polyfit(x[:j], price_window.iloc[:j], 1)
                slopes.append(slope)
        
        if len(slopes) > 1:
            corr = np.corrcoef(slopes[:-1], slopes[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        return 0.0
    
    def _calculate_structural_trend_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend strength - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized R-squared calculation
        prices_series = pd.Series(prices)
        
        # Pre-calculate rolling statistics
        rolling_mean = self._vectorbt_rolling_operation(prices_series, "mean", window)
        rolling_std = self._vectorbt_rolling_operation(prices_series, "std", window)
        
        # Simplified R-squared using variance ratio
        # R² ≈ 1 - (variance_around_trend / total_variance)
        # Approximate trend variance using rolling std
        trend_strength = (1 - (rolling_std / (rolling_mean + 1e-8))).clip(0, 1)
        
        return trend_strength.fillna(0).values
    
    def _calculate_r_squared(self, price_window: pd.Series) -> float:
        """Calculate R-squared for a price window."""
        if len(price_window) < 2:
            return 0.0
        
        # Calculate R-squared of linear trend
        x = np.arange(len(price_window))
        slope, intercept = np.polyfit(x, price_window, 1)
        y_pred = slope * x + intercept
        
        # R-squared as trend strength
        ss_res = np.sum((price_window - y_pred) ** 2)
        ss_tot = np.sum((price_window - price_window.mean()) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        return max(0, r_squared)
    
    def _calculate_trend_acceleration(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend acceleration - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized second derivative calculation
        prices_series = pd.Series(prices)
        
        # Calculate first and second differences vectorized
        first_diff = prices_series.diff()
        second_diff = first_diff.diff()
        
        # Rolling acceleration using second differences
        acceleration = self._vectorbt_rolling_operation(second_diff, "mean", window)
        
        return acceleration.fillna(0).values
    
    def _calculate_second_derivative(self, price_window: pd.Series) -> float:
        """Calculate second derivative for a price window."""
        if len(price_window) < 3:
            return 0.0
        
        # Calculate second derivative (acceleration)
        x = np.arange(len(price_window))
        coeffs = np.polyfit(x, price_window, 2)
        return 2 * coeffs[0]  # Second derivative
    
    def _calculate_trend_intensity(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend intensity - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized intensity calculation
        prices_series = pd.Series(prices)
        
        # Calculate rolling volatility and price changes
        rolling_vol = self._vectorbt_rolling_operation(prices_series, "std", window)
        price_change = (prices_series - prices_series.shift(window)).abs()
        
        # Vectorized intensity calculation
        intensity = price_change / (rolling_vol + 1e-8)
        
        return intensity.fillna(0).values
    
    def _calculate_market_structure_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure strength - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized structure strength calculation
        prices_series = pd.Series(prices)
        
        # Calculate rolling highs and lows
        rolling_highs = self._vectorbt_rolling_operation(prices_series, "max", window)
        rolling_lows = self._vectorbt_rolling_operation(prices_series, "min", window)
        
        # Structure strength based on position within range
        price_range = rolling_highs - rolling_lows
        position = (prices_series - rolling_lows) / (price_range + 1e-8)
        
        # Strength based on how well-defined the structure is
        strength = (1 - (position - 0.5).abs() * 2).clip(0, 1)
        
        return strength.fillna(0).values
    
    def _calculate_support_resistance_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate support/resistance strength - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified support/resistance calculation
        prices_series = pd.Series(prices)
        
        # Calculate rolling price levels and their consistency
        rolling_highs = self._vectorbt_rolling_operation(prices_series, "max", window)
        rolling_lows = self._vectorbt_rolling_operation(prices_series, "min", window)
        
        # Strength based on price level consistency (simplified)
        price_range = rolling_highs - rolling_lows
        level_consistency = 1 / (1 + price_range / (prices_series + 1e-8))
        
        return level_consistency.fillna(0).values
    
    def _calculate_level_strength(self, price_window: pd.Series) -> float:
        """Calculate level strength for a price window."""
        if len(price_window) < 3:
            return 0.0
        
        try:
            # Find local peaks and troughs
            peaks, _ = find_peaks(price_window, distance=2)
            troughs, _ = find_peaks(-price_window, distance=2)
            
            # Strength based on number of significant levels
            all_levels = np.concatenate([price_window.iloc[peaks], price_window.iloc[troughs]])
            if len(all_levels) > 0:
                # Calculate how clustered the levels are
                level_std = np.std(all_levels)
                level_mean = np.mean(all_levels)
                return 1 / (1 + level_std / (level_mean + 1e-8))
        except:
            return 0.0
        return 0.0
    
    def _calculate_market_structure_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure consistency - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use vectorized structure consistency calculation
        prices_series = pd.Series(prices)
        
        # Calculate rolling price level consistency
        rolling_std = self._vectorbt_rolling_operation(prices_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(prices_series, "mean", window)
        
        # Consistency based on coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        consistency = (1 - cv).clip(0, 1)
        
        return consistency.fillna(0).values
    
    def _calculate_level_consistency(self, price_window: pd.Series) -> float:
        """Calculate level consistency for a price window."""
        if len(price_window) < 3:
            return 0.0
        
        # Calculate structure consistency
        # Look for repeated patterns in price levels
        price_levels = np.round(price_window, 2)  # Round to reduce noise
        unique_levels, counts = np.unique(price_levels, return_counts=True)
        
        # Consistency based on level repetition
        if len(unique_levels) > 0:
            max_count = np.max(counts)
            return max_count / len(price_window)
        return 0.0
    
    def _detect_trend_regime_changes(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Detect trend regime changes - OPTIMIZED VECTORIZED."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified trend change detection
        price_series = pd.Series(prices)
        
        # Calculate rolling price changes for trend detection
        price_changes = price_series.diff()
        
        # Calculate rolling means for both windows
        trend1 = self._vectorbt_rolling_operation(price_changes, "mean", window)
        trend2 = trend1.shift(-window)
        
        # Vectorized change detection using simplified approach
        change_ratios = ((trend2 - trend1).abs() / (trend1.abs() + 1e-8)).fillna(0)
        changes = (change_ratios > 0.5).astype(int)
        
        return changes.fillna(0).values
    
    def _calculate_window_trend(self, price_window: np.ndarray) -> float:
        """Calculate trend for a price window."""
        if len(price_window) < 2:
            return 0
        
        x = np.arange(len(price_window))
        slope, _ = np.polyfit(x, price_window, 1)
        return slope
    
    def _calculate_quadratic_coefficient(self, price_window: np.ndarray) -> float:
        """Calculate quadratic coefficient for a price window."""
        if len(price_window) < 3:
            return 0
        
        x = np.arange(len(price_window))
        coeffs = np.polyfit(x, price_window, 2)
        return coeffs[0]  # Quadratic coefficient
    
    def _calculate_r_squared(self, price_window: np.ndarray) -> float:
        """Calculate R-squared for a price window."""
        if len(price_window) < 3:
            return 0
        
        x = np.arange(len(price_window))
        slope, intercept = np.polyfit(x, price_window, 1)
        y_pred = slope * x + intercept
        
        # Calculate R-squared
        ss_res = np.sum((price_window - y_pred) ** 2)
        ss_tot = np.sum((price_window - np.mean(price_window)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        return max(0, r_squared)
    
    def _calculate_slope_autocorr(self, price_window: np.ndarray) -> float:
        """Calculate slope autocorrelation for a price window."""
        if len(price_window) < 4:
            return 0
        
        x = np.arange(len(price_window))
        slopes = []
        
        # Calculate slopes for different sub-windows
        for j in range(2, len(price_window)):
            slope, _ = np.polyfit(x[:j], price_window[:j], 1)
            slopes.append(slope)
        
        if len(slopes) > 1:
            corr = np.corrcoef(slopes[:-1], slopes[1:])[0, 1]
            return corr if not np.isnan(corr) else 0
        
        return 0
    
    def _calculate_window_entropy(self, price_window: np.ndarray) -> float:
        """Calculate entropy for a price window."""
        if len(price_window) < 2:
            return 0
        
        # Calculate entropy of price changes
        price_changes = np.diff(price_window)
        
        if len(price_changes) == 0:
            return 0
        
        # Discretize changes into bins
        bins = np.linspace(np.min(price_changes), np.max(price_changes), 10)
        hist, _ = np.histogram(price_changes, bins=bins)
        
        # Normalize to probabilities
        probs = hist / (np.sum(hist) + 1e-8)
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return entropy
    
    def _calculate_trend_transition_probability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend transition probability - OPTIMIZED VECTORIZED."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified transition probability calculation
        price_series = pd.Series(prices)
        
        # Calculate rolling price changes for trend analysis
        price_changes = price_series.diff()
        
        # Calculate rolling volatility of price changes
        trend_vol = self._vectorbt_rolling_operation(price_changes, "std", window)
        trend_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()
        
        # Vectorized transition probability calculation
        transition_prob = np.minimum(1, trend_vol / (trend_mean + 1e-8))
        
        return transition_prob.fillna(0).values
    
    def _calculate_structural_trend_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend momentum - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified momentum calculation
        price_series = pd.Series(prices)
        
        # Calculate second differences for momentum
        first_diff = price_series.diff()
        second_diff = first_diff.diff()
        
        # Rolling momentum using second differences
        momentum = self._vectorbt_rolling_operation(second_diff, "mean", window)
        
        return momentum.fillna(0).values
    
    def _calculate_trend_stability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend stability - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified stability calculation
        price_series = pd.Series(prices)
        
        # Calculate rolling coefficient of variation for stability
        rolling_std = self._vectorbt_rolling_operation(price_series, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(price_series, "mean", window)
        
        # Stability based on low coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)
        
        return stability.fillna(0).values
    
    def _calculate_trend_persistence_score(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend persistence score - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified persistence calculation
        price_series = pd.Series(prices)
        
        # Calculate price changes for autocorrelation
        price_changes = price_series.diff()
        
        # OPTIMIZED: Use vectorized autocorrelation calculation
        # Calculate rolling autocorrelation using pandas built-in method
        persistence = price_changes.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
            raw=False
        ).fillna(0)
        
        return persistence.values
    
    def _calculate_trend_entropy(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend entropy - OPTIMIZED VECTORIZED."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use simplified entropy calculation
        price_series = pd.Series(prices)
        
        # Calculate price changes for entropy
        price_changes = price_series.diff()
        
        # Rolling entropy using rolling standard deviation as proxy
        rolling_std = self._vectorbt_rolling_operation(price_changes, "std", window)
        rolling_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()
        
        # Entropy proxy using coefficient of variation
        entropy = rolling_std / (rolling_mean + 1e-8)
        
        return entropy.fillna(0).values
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
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
