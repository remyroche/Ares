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

class RegimeVolumeFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volume regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
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
                "regime_windows": [12, 20, 40],  # 3h, 5h, 10h in 15m periods
                "persistence_windows": [8, 16, 32],  # 2h, 4h, 8h
                "clustering_windows": [16, 32, 64],  # 4h, 8h, 16h
                "transition_windows": [4, 8, 16]  # 1h, 2h, 4h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
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
            
            vol_mean_padded[window-1:] = vol_mean
            vol_std_padded[window-1:] = vol_std
            vol_persistence_padded[window-1:] = vol_persistence
            vol_strength_padded[window-1:] = vol_regime_strength
            vol_consistency_padded[window-1:] = vol_consistency
            
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
            
            clustering_padded[window-1:] = vol_clustering
            patterns_padded[window-1:] = vol_patterns
            intensity_padded[window-1:] = vol_intensity
            
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
            
            corr_padded[window-1:] = vol_price_corr
            weighted_padded[window-1:] = vol_weighted_price
            impact_padded[window-1:] = vol_price_impact
            
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
            
            change_padded[window*2-1:] = vol_change
            prob_padded[window*2-1:] = transition_prob
            momentum_padded[window*2-1:] = vol_momentum
            
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
            
            stability_padded[window-1:] = vol_stability
            persistence_padded[window-1:] = persistence_score
            entropy_padded[window-1:] = vol_entropy
            
            features[f'vol_stability_{window}'] = stability_padded
            features[f'vol_persistence_score_{window}'] = persistence_padded
            features[f'vol_entropy_{window}'] = entropy_padded
        
        return features
    
    def _rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean."""
        if len(data) < window:
            return np.array([])
        
        result = np.zeros(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.mean(data[i:i+window])
        
        return result
    
    def _rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        if len(data) < window:
            return np.array([])
        
        result = np.zeros(len(data) - window + 1)
        for i in range(len(result)):
            result[i] = np.std(data[i:i+window])
        
        return result
    
    def _calculate_volume_persistence(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence using autocorrelation."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        persistence = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                corr = np.corrcoef(vol_window[:-1], vol_window[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_volume_regime_strength(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime strength."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        strength = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            # Regime strength based on consistency of volume level
            vol_consistency = 1.0 - (np.std(vol_window) / (np.mean(vol_window) + 1e-8))
            strength[i] = max(0, min(1, vol_consistency))
        
        return strength
    
    def _calculate_volume_consistency(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime consistency."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        consistency = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                # Consistency based on low coefficient of variation
                cv = np.std(vol_window) / (np.mean(vol_window) + 1e-8)
                consistency[i] = max(0, 1 - cv)
        
        return consistency
    
    def _calculate_volume_clustering(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume clustering."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        clustering = np.zeros(len(volume))
        for i in range(window, len(volume)):
            # Calculate volume autocorrelation
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                corr = np.corrcoef(vol_window[:-1], vol_window[1:])[0, 1]
                clustering[i] = corr if not np.isnan(corr) else 0
        
        return clustering
    
    def _calculate_volume_patterns(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime patterns."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        patterns = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 2:
                # Pattern based on volume trend
                trend = np.polyfit(range(len(vol_window)), vol_window, 1)[0]
                # Normalize trend to 0-1 range
                patterns[i] = (np.tanh(trend / (np.mean(vol_window) + 1e-8)) + 1) / 2
        
        return patterns
    
    def _calculate_volume_intensity(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime intensity."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        intensity = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            # Intensity based on volume relative to historical average
            avg_vol = np.mean(vol_window)
            current_vol = volume[i-1] if i > 0 else vol_window[-1]
            intensity[i] = min(2, current_vol / (avg_vol + 1e-8))
        
        return intensity
    
    def _calculate_volume_price_correlation(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-price correlation."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        correlation = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            price_window = prices[i-window:i]
            if len(vol_window) > 1 and len(price_window) > 1:
                corr = np.corrcoef(vol_window, price_window)[0, 1]
                correlation[i] = corr if not np.isnan(corr) else 0
        
        return correlation
    
    def _calculate_volume_weighted_price_change(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume-weighted price change."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        weighted_change = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            price_window = prices[i-window:i]
            if len(vol_window) > 1 and len(price_window) > 1:
                # Volume-weighted price change
                price_changes = np.diff(price_window)
                weights = vol_window[1:] / (np.sum(vol_window[1:]) + 1e-8)
                weighted_change[i] = np.sum(price_changes * weights)
        
        return weighted_change
    
    def _calculate_volume_price_impact(self, volume: np.ndarray, prices: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume price impact."""
        if len(volume) < window or len(prices) < window:
            return np.zeros(len(volume))
        
        impact = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            price_window = prices[i-window:i]
            if len(vol_window) > 1 and len(price_window) > 1:
                # Price impact per unit volume
                price_changes = np.abs(np.diff(price_window))
                vol_changes = np.diff(vol_window)
                # Avoid division by zero
                vol_changes = np.where(vol_changes == 0, 1e-8, vol_changes)
                impact_ratio = price_changes / vol_changes
                impact[i] = np.mean(impact_ratio)
        
        return impact
    
    def _detect_volume_regime_changes(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Detect volume regime changes."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))
        
        changes = np.zeros(len(volume))
        for i in range(window * 2, len(volume)):
            # Compare volume in two consecutive windows
            vol1 = np.mean(volume[i-window*2:i-window])
            vol2 = np.mean(volume[i-window:i])
            
            # Significant change threshold (30% change)
            change_ratio = abs(vol2 - vol1) / (vol1 + 1e-8)
            changes[i] = 1 if change_ratio > 0.3 else 0
        
        return changes
    
    def _calculate_volume_transition_probability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime transition probability."""
        if len(volume) < window * 2:
            return np.zeros(len(volume))
        
        transition_prob = np.zeros(len(volume))
        for i in range(window * 2, len(volume)):
            # Calculate transition probability based on recent volume changes
            recent_vol = volume[i-window*2:i]
            if len(recent_vol) > 1:
                # Probability based on volume trend
                trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
                transition_prob[i] = min(1, max(0, abs(trend) / (np.mean(recent_vol) + 1e-8)))
        
        return transition_prob
    
    def _calculate_volume_momentum(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume momentum."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        momentum = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                # Momentum based on volume trend
                trend = np.polyfit(range(len(vol_window)), vol_window, 1)[0]
                momentum[i] = trend / (np.mean(vol_window) + 1e-8)
        
        return momentum
    
    def _calculate_volume_stability(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime stability."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        stability = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                # Stability based on low coefficient of variation
                cv = np.std(vol_window) / (np.mean(vol_window) + 1e-8)
                stability[i] = max(0, 1 - cv)
        
        return stability
    
    def _calculate_volume_persistence_score(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume persistence score."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        persistence = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 2:
                # Persistence based on autocorrelation of volume
                corr = np.corrcoef(vol_window[:-1], vol_window[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_volume_entropy(self, volume: np.ndarray, window: int) -> np.ndarray:
        """Calculate volume regime entropy."""
        if len(volume) < window:
            return np.zeros(len(volume))
        
        entropy = np.zeros(len(volume))
        for i in range(window, len(volume)):
            vol_window = volume[i-window:i]
            if len(vol_window) > 1:
                # Calculate entropy of volume distribution
                # Discretize volume into bins
                bins = np.linspace(np.min(vol_window), np.max(vol_window), 10)
                hist, _ = np.histogram(vol_window, bins=bins)
                # Normalize to probabilities
                probs = hist / (np.sum(hist) + 1e-8)
                # Calculate entropy
                entropy[i] = -np.sum(probs * np.log(probs + 1e-8))
        
        return entropy