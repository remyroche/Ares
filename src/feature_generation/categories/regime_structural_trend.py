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

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)
from ..base_calculations import (
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
        super().__init__(config, enable_matrix_ops=True)
    
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
                "structural_windows": [20, 40, 80],  # 5h, 10h, 20h in 15m periods
                "persistence_windows": [16, 32, 64],  # 4h, 8h, 16h
                "transition_windows": [8, 16, 32],  # 2h, 4h, 8h
                "structure_windows": [24, 48, 96]  # 6h, 12h, 24h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
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
            
            persistence_padded[window-1:] = trend_persistence
            direction_padded[window-1:] = direction_consistency
            regime_padded[window-1:] = regime_persistence
            
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
            
            strength_padded[window-1:] = trend_strength
            acceleration_padded[window-1:] = trend_acceleration
            intensity_padded[window-1:] = trend_intensity
            
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
            
            strength_padded[window-1:] = structure_strength
            sr_padded[window-1:] = support_resistance
            consistency_padded[window-1:] = structure_consistency
            
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
            
            change_padded[window*2-1:] = trend_change
            prob_padded[window*2-1:] = transition_prob
            momentum_padded[window*2-1:] = trend_momentum
            
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
            
            stability_padded[window-1:] = trend_stability
            persistence_padded[window-1:] = persistence_score
            entropy_padded[window-1:] = trend_entropy
            
            features[f'trend_stability_{window}'] = stability_padded
            features[f'trend_persistence_score_{window}'] = persistence_padded
            features[f'trend_entropy_{window}'] = entropy_padded
        
        return features
    
    def _calculate_structural_trend_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend persistence."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        persistence = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 1:
                # Calculate trend using linear regression
                x = np.arange(len(price_window))
                slope, _ = np.polyfit(x, price_window, 1)
                
                # Persistence based on trend consistency
                trend_consistency = abs(slope) / (np.std(price_window) + 1e-8)
                persistence[i] = min(1, trend_consistency)
        
        return persistence
    
    def _calculate_trend_direction_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend direction consistency."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        consistency = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Calculate direction changes
                price_changes = np.diff(price_window)
                positive_changes = np.sum(price_changes > 0)
                negative_changes = np.sum(price_changes < 0)
                
                # Consistency based on direction dominance
                total_changes = positive_changes + negative_changes
                if total_changes > 0:
                    consistency[i] = max(positive_changes, negative_changes) / total_changes
        
        return consistency
    
    def _calculate_trend_regime_persistence(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend regime persistence."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        persistence = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Calculate trend autocorrelation
                x = np.arange(len(price_window))
                slopes = []
                for j in range(1, len(price_window)):
                    if j > 1:
                        slope, _ = np.polyfit(x[:j], price_window[:j], 1)
                        slopes.append(slope)
                
                if len(slopes) > 1:
                    corr = np.corrcoef(slopes[:-1], slopes[1:])[0, 1]
                    persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_structural_trend_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend strength."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        strength = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 1:
                # Calculate R-squared of linear trend
                x = np.arange(len(price_window))
                slope, intercept = np.polyfit(x, price_window, 1)
                y_pred = slope * x + intercept
                
                # R-squared as trend strength
                ss_res = np.sum((price_window - y_pred) ** 2)
                ss_tot = np.sum((price_window - np.mean(price_window)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                strength[i] = max(0, r_squared)
        
        return strength
    
    def _calculate_trend_acceleration(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend acceleration."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        acceleration = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Calculate second derivative (acceleration)
                x = np.arange(len(price_window))
                coeffs = np.polyfit(x, price_window, 2)
                acceleration[i] = 2 * coeffs[0]  # Second derivative
        
        return acceleration
    
    def _calculate_trend_intensity(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend intensity."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        intensity = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 1:
                # Intensity based on price change relative to volatility
                price_change = abs(prices[i-1] - prices[i-window])
                volatility = np.std(price_window)
                intensity[i] = price_change / (volatility + 1e-8)
        
        return intensity
    
    def _calculate_market_structure_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure strength."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        strength = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Calculate structure based on price levels
                highs = np.max(price_window)
                lows = np.min(price_window)
                current_price = prices[i-1]
                
                # Structure strength based on position within range
                if highs != lows:
                    position = (current_price - lows) / (highs - lows)
                    # Strength based on how well-defined the structure is
                    strength[i] = 1 - abs(position - 0.5) * 2
        
        return strength
    
    def _calculate_support_resistance_strength(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate support/resistance strength."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        strength = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Find local peaks and troughs
                peaks, _ = find_peaks(price_window, distance=2)
                troughs, _ = find_peaks(-price_window, distance=2)
                
                # Strength based on number of significant levels
                all_levels = np.concatenate([price_window[peaks], price_window[troughs]])
                if len(all_levels) > 0:
                    # Calculate how clustered the levels are
                    level_std = np.std(all_levels)
                    level_mean = np.mean(all_levels)
                    strength[i] = 1 / (1 + level_std / (level_mean + 1e-8))
        
        return strength
    
    def _calculate_market_structure_consistency(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate market structure consistency."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        consistency = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Calculate structure consistency
                # Look for repeated patterns in price levels
                price_levels = np.round(price_window, 2)  # Round to reduce noise
                unique_levels, counts = np.unique(price_levels, return_counts=True)
                
                # Consistency based on level repetition
                if len(unique_levels) > 0:
                    max_count = np.max(counts)
                    consistency[i] = max_count / len(price_window)
        
        return consistency
    
    def _detect_trend_regime_changes(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Detect trend regime changes."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        changes = np.zeros(len(prices))
        for i in range(window * 2, len(prices)):
            # Compare trends in two consecutive windows
            trend1 = self._calculate_window_trend(prices[i-window*2:i-window])
            trend2 = self._calculate_window_trend(prices[i-window:i])
            
            # Significant change threshold
            if abs(trend1) > 1e-8 and abs(trend2) > 1e-8:
                change_ratio = abs(trend2 - trend1) / (abs(trend1) + 1e-8)
                changes[i] = 1 if change_ratio > 0.5 else 0
        
        return changes
    
    def _calculate_window_trend(self, price_window: np.ndarray) -> float:
        """Calculate trend for a price window."""
        if len(price_window) < 2:
            return 0
        
        x = np.arange(len(price_window))
        slope, _ = np.polyfit(x, price_window, 1)
        return slope
    
    def _calculate_trend_transition_probability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend transition probability."""
        if len(prices) < window * 2:
            return np.zeros(len(prices))
        
        transition_prob = np.zeros(len(prices))
        for i in range(window * 2, len(prices)):
            # Calculate transition probability based on recent trend changes
            recent_trends = []
            for j in range(window):
                if i - window - j >= window:
                    trend = self._calculate_window_trend(prices[i-window-j:i-j])
                    recent_trends.append(trend)
            
            if len(recent_trends) > 1:
                # Probability based on trend volatility
                trend_vol = np.std(recent_trends)
                trend_mean = np.mean(np.abs(recent_trends))
                transition_prob[i] = min(1, trend_vol / (trend_mean + 1e-8))
        
        return transition_prob
    
    def _calculate_structural_trend_momentum(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate structural trend momentum (not trading momentum)."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        momentum = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 1:
                # Structural momentum based on trend acceleration
                x = np.arange(len(price_window))
                coeffs = np.polyfit(x, price_window, 2)
                momentum[i] = coeffs[0]  # Quadratic coefficient
        
        return momentum
    
    def _calculate_trend_stability(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend stability."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        stability = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Stability based on trend consistency
                x = np.arange(len(price_window))
                slope, _ = np.polyfit(x, price_window, 1)
                y_pred = slope * x + np.mean(price_window)
                
                # Stability based on R-squared
                ss_res = np.sum((price_window - y_pred) ** 2)
                ss_tot = np.sum((price_window - np.mean(price_window)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                stability[i] = max(0, r_squared)
        
        return stability
    
    def _calculate_trend_persistence_score(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend persistence score."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        persistence = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 2:
                # Persistence based on trend autocorrelation
                x = np.arange(len(price_window))
                slopes = []
                for j in range(2, len(price_window)):
                    slope, _ = np.polyfit(x[:j], price_window[:j], 1)
                    slopes.append(slope)
                
                if len(slopes) > 1:
                    corr = np.corrcoef(slopes[:-1], slopes[1:])[0, 1]
                    persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_trend_entropy(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate trend entropy."""
        if len(prices) < window:
            return np.zeros(len(prices))
        
        entropy = np.zeros(len(prices))
        for i in range(window, len(prices)):
            price_window = prices[i-window:i]
            if len(price_window) > 1:
                # Calculate entropy of price changes
                price_changes = np.diff(price_window)
                # Discretize changes into bins
                bins = np.linspace(np.min(price_changes), np.max(price_changes), 10)
                hist, _ = np.histogram(price_changes, bins=bins)
                # Normalize to probabilities
                probs = hist / (np.sum(hist) + 1e-8)
                # Calculate entropy
                entropy[i] = -np.sum(probs * np.log(probs + 1e-8))
        
        return entropy