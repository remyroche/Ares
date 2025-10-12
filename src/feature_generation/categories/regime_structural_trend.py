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
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

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

class RegimeStructuralTrendFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for structural trend regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT optimizers
        self.vectorbt_optimizer = None
        self.unified_optimizer = None
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0,
            'feature_generation_time': 0.0,
            'optimization_time': 0.0,
            'memory_usage_mb': 0.0,
            'features_generated': 0,
            'vectorbt_success_rate': 0.0,
            'unified_optimization_success_rate': 0.0,
            'avg_feature_generation_time': 0.0,
            'total_features_generated': 0
        }
        
        if OPTIMIZATION_AVAILABLE:
            try:
                # Initialize VectorBT Rolling Optimizer
                self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=getattr(config, 'gpu_accelerated', False),
                    enable_parallel=True,
                    memory_efficient=True
                )
                
                # Initialize Unified Optimization System
                from ..utils.unified_optimization_system import UnifiedOptimizationConfig
                unified_config = UnifiedOptimizationConfig(
                    enable_normalization=True,
                    enable_scaling=True,
                    enable_vectorization=True,
                    enable_hardware_optimization=getattr(config, 'gpu_accelerated', False),
                    memory_limit_gb=8.0
                )
                self.unified_optimizer = get_unified_optimization_system(unified_config)
                
                print("✅ VectorBT optimizers and UnifiedVectorizationManager initialized for RegimeStructuralTrendFeatureGenerator")
            except Exception as e:
                print(f"⚠️ VectorBT optimizer initialization failed: {e}")
    
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
        """Generate structural trend regime features with unified optimization."""
        import time
        start_time = time.time()
        
        features = {}
        
        # Validate price data
        if 'close' not in data.columns:
            return features
        
        close_prices = data['close'].values
        if len(close_prices) < 8:
            return features
        
        # Apply unified optimization to data if available
        optimization_start = time.time()
        if self.unified_optimizer:
            try:
                optimized_result = self.unified_optimizer.process_features_unified(
                    data, 
                    categories=['regime_structural_trend'],
                    **kwargs
                )
                if optimized_result.success:
                    data = optimized_result.data
                    self.performance_stats['unified_optimizations'] += 1
                    self.performance_stats['optimization_time'] += time.time() - optimization_start
            except Exception as e:
                print(f"⚠️ Unified optimization failed: {e}")
                self.performance_stats['optimization_time'] += time.time() - optimization_start
        
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
        
        # 6. Enhanced VectorBT Features (if available)
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            features.update(self._calculate_enhanced_trend_features(close_prices, data))
            features.update(self._calculate_vectorbt_optimized_correlations(close_prices, data))
            features.update(self._calculate_vectorbt_advanced_features(close_prices, data))
        
        # Update performance stats
        feature_generation_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['processing_time'] += feature_generation_time
        self.performance_stats['feature_generation_time'] += feature_generation_time
        self.performance_stats['features_generated'] = len(features)
        self.performance_stats['total_features_generated'] += len(features)
        
        # Log performance if verbose
        if hasattr(self.config, 'verbose') and self.config.verbose:
            print(f"Generated {len(features)} features in {feature_generation_time:.3f}s")
        
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
        """Calculate structural trend persistence - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Pre-calculate rolling statistics using VectorBT
        rolling_mean = self.vectorbt_optimizer.rolling_mean(prices_series, window) if self.vectorbt_optimizer else self._vectorbt_rolling_operation(prices_series, "mean", window)
        rolling_std = self.vectorbt_optimizer.rolling_std(prices_series, window) if self.vectorbt_optimizer else self._vectorbt_rolling_operation(prices_series, "std", window)
        
        # OPTIMIZED: Use VectorBT native functions for enhanced calculations
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            # Use VectorBT's built-in trend analysis
            try:
                # Calculate rolling linear regression using VectorBT
                rolling_slope = self.vectorbt_optimizer.rolling_apply(
                    prices_series, 
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
                    window
                ).fillna(0)
                
                # Use VectorBT's scale function for normalization
                normalized_slope = scale(rolling_slope, axis=0)
                
                # Persistence based on normalized slope consistency
                persistence = (normalized_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)
                
            except Exception:
                # Fallback to simpler calculation
                rolling_slope = self.vectorbt_optimizer.rolling_apply(
                    prices_series, 
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0,
                    window
                ).fillna(0)
                persistence = (rolling_slope.abs() / (rolling_std + 1e-8)).clip(0, 1)
        else:
            rolling_slope = self._vectorbt_rolling_operation(prices_series, "apply", window, func=lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else 0).fillna(0)
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
        """Calculate trend direction consistency - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate price changes vectorized
        price_changes = prices_series.diff()
        
        # OPTIMIZED: Use VectorBT native functions for enhanced calculations
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT's rank function for direction analysis
                price_ranks = self.vectorbt_optimizer.rolling_rank(price_changes, window)
                
                # Calculate direction consistency using ranks
                positive_ranks = (price_ranks > 0.5).astype(int)
                negative_ranks = (price_ranks < 0.5).astype(int)
                
                positive_changes = self.vectorbt_optimizer.rolling_sum(positive_ranks, window).fillna(0)
                negative_changes = self.vectorbt_optimizer.rolling_sum(negative_ranks, window).fillna(0)
                
            except Exception:
                # Fallback to standard calculation
                positive_changes = self.vectorbt_optimizer.rolling_sum((price_changes > 0).astype(int), window).fillna(0)
                negative_changes = self.vectorbt_optimizer.rolling_sum((price_changes < 0).astype(int), window).fillna(0)
        else:
            positive_changes = self._vectorbt_rolling_operation((price_changes > 0).astype(int), "sum", window).fillna(0)
            negative_changes = self._vectorbt_rolling_operation((price_changes < 0).astype(int), "sum", window).fillna(0)
        
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
        """Calculate trend regime persistence - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate price changes for autocorrelation
        price_changes = prices_series.diff()
        
        # OPTIMIZED: Use VectorBT rolling apply for autocorrelation
        if self.vectorbt_optimizer:
            persistence = self.vectorbt_optimizer.rolling_apply(
                price_changes,
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                window
            ).fillna(0)
        else:
            persistence = self._vectorbt_rolling_operation(price_changes, "apply", window, func=lambda x: x.autocorr(lag=1) if len(x) > 1 else 0).fillna(0)
        
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
        """Calculate structural trend strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Pre-calculate rolling statistics using VectorBT
        if self.vectorbt_optimizer:
            rolling_mean = self.vectorbt_optimizer.rolling_mean(prices_series, window)
            rolling_std = self.vectorbt_optimizer.rolling_std(prices_series, window)
        else:
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
        """Calculate trend acceleration - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate first and second differences vectorized
        first_diff = prices_series.diff()
        second_diff = first_diff.diff()
        
        # Rolling acceleration using VectorBT
        if self.vectorbt_optimizer:
            acceleration = self.vectorbt_optimizer.rolling_mean(second_diff, window)
        else:
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
        """Calculate trend intensity - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate rolling volatility using VectorBT
        if self.vectorbt_optimizer:
            rolling_vol = self.vectorbt_optimizer.rolling_std(prices_series, window)
        else:
            rolling_vol = self._vectorbt_rolling_operation(prices_series, "std", window)
        
        price_change = (prices_series - prices_series.shift(window)).abs()
        
        # Vectorized intensity calculation
        intensity = price_change / (rolling_vol + 1e-8)
        
        return intensity.fillna(0).values
    
    def _calculate_market_structure_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate rolling highs and lows using VectorBT
        if self.vectorbt_optimizer:
            rolling_highs = self.vectorbt_optimizer.rolling_max(prices_series, window)
            rolling_lows = self.vectorbt_optimizer.rolling_min(prices_series, window)
        else:
            rolling_highs = self._vectorbt_rolling_operation(prices_series, "max", window)
            rolling_lows = self._vectorbt_rolling_operation(prices_series, "min", window)
        
        # Structure strength based on position within range
        price_range = rolling_highs - rolling_lows
        position = (prices_series - rolling_lows) / (price_range + 1e-8)
        
        # Strength based on how well-defined the structure is
        strength = (1 - (position - 0.5).abs() * 2).clip(0, 1)
        
        return strength.fillna(0).values
    
    def _calculate_support_resistance_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate support/resistance strength - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate rolling price levels using VectorBT
        if self.vectorbt_optimizer:
            rolling_highs = self.vectorbt_optimizer.rolling_max(prices_series, window)
            rolling_lows = self.vectorbt_optimizer.rolling_min(prices_series, window)
        else:
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
        """Calculate market structure consistency - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        prices_series = pd.Series(prices)
        
        # Calculate rolling price level consistency using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_optimizer.rolling_std(prices_series, window)
            rolling_mean = self.vectorbt_optimizer.rolling_mean(prices_series, window)
        else:
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
        """Detect trend regime changes - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate rolling price changes for trend detection
        price_changes = price_series.diff()
        
        # Calculate rolling means using VectorBT
        if self.vectorbt_optimizer:
            trend1 = self.vectorbt_optimizer.rolling_mean(price_changes, window)
        else:
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
        """Calculate trend transition probability - OPTIMIZED VECTORBT."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate rolling price changes for trend analysis
        price_changes = price_series.diff()
        
        # Calculate rolling volatility using VectorBT
        if self.vectorbt_optimizer:
            trend_vol = self.vectorbt_optimizer.rolling_std(price_changes, window)
            trend_mean = self.vectorbt_optimizer.rolling_mean(price_changes, window).abs()
        else:
            trend_vol = self._vectorbt_rolling_operation(price_changes, "std", window)
            trend_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()
        
        # Vectorized transition probability calculation
        transition_prob = np.minimum(1, trend_vol / (trend_mean + 1e-8))
        
        return transition_prob.fillna(0).values
    
    def _calculate_structural_trend_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend momentum - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate second differences for momentum
        first_diff = price_series.diff()
        second_diff = first_diff.diff()
        
        # Rolling momentum using VectorBT
        if self.vectorbt_optimizer:
            momentum = self.vectorbt_optimizer.rolling_mean(second_diff, window)
        else:
            momentum = self._vectorbt_rolling_operation(second_diff, "mean", window)
        
        return momentum.fillna(0).values
    
    def _calculate_trend_stability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend stability - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate rolling coefficient of variation using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_optimizer.rolling_std(price_series, window)
            rolling_mean = self.vectorbt_optimizer.rolling_mean(price_series, window)
        else:
            rolling_std = self._vectorbt_rolling_operation(price_series, "std", window)
            rolling_mean = self._vectorbt_rolling_operation(price_series, "mean", window)
        
        # Stability based on low coefficient of variation
        cv = rolling_std / (rolling_mean + 1e-8)
        stability = (1 - cv).clip(0, 1)
        
        return stability.fillna(0).values
    
    def _calculate_trend_persistence_score(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend persistence score - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate price changes for autocorrelation
        price_changes = price_series.diff()
        
        # OPTIMIZED: Use VectorBT rolling apply for autocorrelation
        if self.vectorbt_optimizer:
            persistence = self.vectorbt_optimizer.rolling_apply(
                price_changes,
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                window
            ).fillna(0)
        else:
            persistence = self._vectorbt_rolling_operation(price_changes, "apply", window, func=lambda x: x.autocorr(lag=1) if len(x) > 1 else 0).fillna(0)
        
        return persistence.values
    
    def _calculate_trend_entropy(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend entropy - OPTIMIZED VECTORBT."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        # OPTIMIZED: Use VectorBT rolling operations
        price_series = pd.Series(prices)
        
        # Calculate price changes for entropy
        price_changes = price_series.diff()
        
        # Rolling entropy using VectorBT
        if self.vectorbt_optimizer:
            rolling_std = self.vectorbt_optimizer.rolling_std(price_changes, window)
            rolling_mean = self.vectorbt_optimizer.rolling_mean(price_changes, window).abs()
        else:
            rolling_std = self._vectorbt_rolling_operation(price_changes, "std", window)
            rolling_mean = self._vectorbt_rolling_operation(price_changes, "mean", window).abs()
        
        # Entropy proxy using coefficient of variation
        entropy = rolling_std / (rolling_mean + 1e-8)
        
        return entropy.fillna(0).values
    
    def _calculate_enhanced_trend_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate enhanced trend features using VectorBT native functions."""
        features = {}
        
        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features
        
        try:
            prices_series = pd.Series(prices)
            
            # Use VectorBT native functions for enhanced calculations
            if self.vectorbt_optimizer:
                # Rolling quantiles for trend distribution analysis
                q25 = self.vectorbt_optimizer.rolling_quantile(prices_series, window=20, q=0.25)
                q75 = self.vectorbt_optimizer.rolling_quantile(prices_series, window=20, q=0.75)
                
                # Rolling skewness and kurtosis for trend shape analysis
                rolling_skew = self.vectorbt_optimizer.rolling_skew(prices_series, window=20)
                rolling_kurt = self.vectorbt_optimizer.rolling_kurt(prices_series, window=20)
                
                # Enhanced trend features
                features['trend_quartile_range'] = (q75 - q25).fillna(0).values
                features['trend_skewness'] = rolling_skew.fillna(0).values
                features['trend_kurtosis'] = rolling_kurt.fillna(0).values
                
                # Rolling rank for trend position analysis
                rolling_rank = self.vectorbt_optimizer.rolling_rank(prices_series, window=20)
                features['trend_rank'] = rolling_rank.fillna(0).values
                
                # Price position within rolling range
                rolling_min = self.vectorbt_optimizer.rolling_min(prices_series, window=20)
                rolling_max = self.vectorbt_optimizer.rolling_max(prices_series, window=20)
                price_position = (prices_series - rolling_min) / (rolling_max - rolling_min + 1e-8)
                features['trend_position'] = price_position.fillna(0.5).values
                
        except Exception as e:
            print(f"Enhanced trend features calculation failed: {e}")
        
        return features
    
    def _calculate_vectorbt_optimized_correlations(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate rolling correlations using VectorBT optimization."""
        features = {}
        
        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features
        
        try:
            prices_series = pd.Series(prices)
            
            # Calculate price changes for correlation analysis
            price_changes = prices_series.diff()
            
            if self.vectorbt_optimizer:
                # Rolling autocorrelation using VectorBT
                rolling_autocorr = self.vectorbt_optimizer.rolling_apply(
                    price_changes,
                    lambda x: x.autocorr(lag=1) if len(x) > 1 else 0,
                    window=20
                )
                features['price_autocorrelation'] = rolling_autocorr.fillna(0).values
                
                # Rolling correlation with lagged values
                if len(price_changes) > 40:
                    lagged_changes = price_changes.shift(1)
                    rolling_corr = self.vectorbt_optimizer.rolling_corr(
                        price_changes, lagged_changes, window=20
                    )
                    features['price_lag_correlation'] = rolling_corr.fillna(0).values
                
                # Use VectorBT native functions for additional correlations
                if VECTORBT_AVAILABLE:
                    try:
                        # Rolling correlation with volume if available
                        if 'volume' in data.columns and len(data['volume']) > 20:
                            volume_series = data['volume']
                            volume_changes = volume_series.diff()
                            
                            price_volume_corr = self.vectorbt_optimizer.rolling_corr(
                                price_changes, volume_changes, window=20
                            )
                            features['price_volume_correlation'] = price_volume_corr.fillna(0).values
                        
                        # Rolling correlation with high-low range if available
                        if 'high' in data.columns and 'low' in data.columns:
                            hl_range = data['high'] - data['low']
                            range_changes = hl_range.diff()
                            
                            price_range_corr = self.vectorbt_optimizer.rolling_corr(
                                price_changes, range_changes, window=20
                            )
                            features['price_range_correlation'] = price_range_corr.fillna(0).values
                            
                    except Exception as e:
                        print(f"Additional VectorBT correlation calculations failed: {e}")
                
        except Exception as e:
            print(f"VectorBT correlation calculation failed: {e}")
        
        return features
    
    def _calculate_vectorbt_advanced_features(self, prices: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate advanced features using VectorBT native functions."""
        features = {}
        
        if not VECTORBT_AVAILABLE or len(prices) < 20:
            return features
        
        try:
            prices_series = pd.Series(prices)
            
            if self.vectorbt_optimizer:
                # Use VectorBT's winsorize function for outlier handling
                winsorized_prices = winsorize(prices_series, limits=(0.05, 0.05))
                
                # Calculate rolling statistics on winsorized data
                rolling_mean_win = self.vectorbt_optimizer.rolling_mean(winsorized_prices, window=20)
                rolling_std_win = self.vectorbt_optimizer.rolling_std(winsorized_prices, window=20)
                
                features['winsorized_mean'] = rolling_mean_win.fillna(0).values
                features['winsorized_std'] = rolling_std_win.fillna(0).values
                
                # Use VectorBT's clip function for bounded calculations
                clipped_prices = clip(prices_series, 
                                    prices_series.quantile(0.01), 
                                    prices_series.quantile(0.99))
                
                # Calculate rolling quantiles using VectorBT
                rolling_q25 = self.vectorbt_optimizer.rolling_quantile(clipped_prices, window=20, q=0.25)
                rolling_q75 = self.vectorbt_optimizer.rolling_quantile(clipped_prices, window=20, q=0.75)
                
                features['clipped_iqr'] = (rolling_q75 - rolling_q25).fillna(0).values
                
                # Use VectorBT's zscore function for normalization
                zscored_prices = zscore(prices_series, axis=0)
                rolling_zscore_mean = self.vectorbt_optimizer.rolling_mean(zscored_prices, window=20)
                
                features['rolling_zscore_mean'] = rolling_zscore_mean.fillna(0).values
                
        except Exception as e:
            print(f"VectorBT advanced features calculation failed: {e}")
        
        return features
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (self.vectorbt_optimizer is not None and 
                len(data) >= 100 and 
                VECTORBT_AVAILABLE)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Add VectorBT optimizer stats if available
        if self.vectorbt_optimizer:
            vectorbt_stats = self.vectorbt_optimizer.get_performance_stats()
            stats['vectorbt_stats'] = vectorbt_stats
        
        # Add unified optimizer stats if available
        if self.unified_optimizer:
            unified_stats = self.unified_optimizer.get_performance_report()
            stats['unified_stats'] = unified_stats
        
        # Calculate efficiency metrics
        if stats['total_operations'] > 0:
            stats['avg_processing_time'] = stats['processing_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0
            stats['unified_optimization_rate'] = stats['unified_optimizations'] / stats['total_operations'] if stats['total_operations'] > 0 else 0
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'unified_optimizations': 0,
            'total_operations': 0,
            'processing_time': 0.0
        }
        
        if self.vectorbt_optimizer:
            self.vectorbt_optimizer.reset_stats()
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            # Use VectorBT Rolling Optimizer if available
            if self.vectorbt_optimizer:
                if operation == 'mean':
                    result = self.vectorbt_optimizer.rolling_mean(data, window, **kwargs)
                elif operation == 'std':
                    result = self.vectorbt_optimizer.rolling_std(data, window, **kwargs)
                elif operation == 'var':
                    result = self.vectorbt_optimizer.rolling_var(data, window, **kwargs)
                elif operation == 'min':
                    result = self.vectorbt_optimizer.rolling_min(data, window, **kwargs)
                elif operation == 'max':
                    result = self.vectorbt_optimizer.rolling_max(data, window, **kwargs)
                elif operation == 'sum':
                    result = self.vectorbt_optimizer.rolling_sum(data, window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    result = self.vectorbt_optimizer.rolling_apply(data, window, func, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                self.performance_stats['vectorbt_operations'] += 1
                return result
            else:
                # Direct VectorBT usage
                if operation == 'mean':
                    result = rolling_mean(data, window=window, **kwargs)
                elif operation == 'std':
                    result = rolling_std(data, window=window, **kwargs)
                elif operation == 'var':
                    result = rolling_var(data, window=window, **kwargs)
                elif operation == 'min':
                    result = rolling_min(data, window=window, **kwargs)
                elif operation == 'max':
                    result = rolling_max(data, window=window, **kwargs)
                elif operation == 'sum':
                    result = rolling_sum(data, window=window, **kwargs)
                elif operation == 'apply':
                    func = kwargs.get('func')
                    result = rolling_apply(data, window=window, func=func, **kwargs)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")
                
                self.performance_stats['vectorbt_operations'] += 1
                return result
                
        except Exception as e:
            print(f"VectorBT operation failed: {e}, using pandas fallback")
            self.performance_stats['pandas_fallbacks'] += 1
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas with VectorBT optimization when available."""
        # Try VectorBT first if available
        if VECTORBT_AVAILABLE and len(data) > 100:  # Use VectorBT for larger datasets
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
                elif operation == 'apply':
                    func = kwargs.get('func')
                    return rolling_apply(data, window=window, func=func, **kwargs)
            except Exception:
                pass  # Fall back to pandas
        
        # Pandas fallback
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
        elif operation == 'apply':
            func = kwargs.get('func')
            return data.rolling(window=window).apply(func, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
