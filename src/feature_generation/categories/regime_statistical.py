"""
Regime Statistical Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on statistical
regime characteristics rather than short-term trading signals.

Key Features:
- Distribution shape changes (skewness, kurtosis)
- Regime persistence measures
- Cross-correlation stability
- Regime transition probabilities
- Statistical regime stability
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis, jarque_bera

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

class RegimeStatisticalFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for statistical regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_statistical_features",
            category=FeatureCategory.STATISTICAL,
            description="Statistical regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=32,  # 8 hours in 15m periods
            min_lookback=8,       # 2 hours minimum
            max_lookback=128,     # 32 hours maximum
            parameters={
                "distribution_windows": [16, 32, 64],  # 4h, 8h, 16h in 15m periods
                "correlation_windows": [20, 40, 80],  # 5h, 10h, 20h
                "persistence_windows": [12, 24, 48],  # 3h, 6h, 12h
                "transition_windows": [8, 16, 32]  # 2h, 4h, 8h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate statistical regime features."""
        features = {}
        
        # Validate price data
        if 'close' not in data.columns:
            return features
        
        close_prices = data['close'].values
        if len(close_prices) < 8:
            return features
        
        # Calculate returns for statistical analysis
        returns = np.diff(np.log(close_prices))
        
        # 1. Distribution Shape Features
        features.update(self._generate_distribution_features(returns, data))
        
        # 2. Statistical Regime Persistence
        features.update(self._generate_statistical_persistence_features(returns, data))
        
        # 3. Cross-Correlation Features
        features.update(self._generate_correlation_features(returns, data))
        
        # 4. Statistical Regime Transitions
        features.update(self._generate_statistical_transition_features(returns, data))
        
        # 5. Statistical Regime Stability
        features.update(self._generate_statistical_stability_features(returns, data))
        
        return features
    
    def _generate_distribution_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate distribution shape features."""
        features = {}
        windows = self.config.parameters["distribution_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Skewness regime features
            skewness = self._calculate_rolling_skewness(returns, window)
            skewness_persistence = self._calculate_skewness_persistence(returns, window)
            
            # Kurtosis regime features
            kurtosis = self._calculate_rolling_kurtosis(returns, window)
            kurtosis_persistence = self._calculate_kurtosis_persistence(returns, window)
            
            # Distribution normality
            normality = self._calculate_distribution_normality(returns, window)
            
            # Pad to match data length
            skewness_padded = np.full(len(data), np.nan)
            skew_persist_padded = np.full(len(data), np.nan)
            kurtosis_padded = np.full(len(data), np.nan)
            kurt_persist_padded = np.full(len(data), np.nan)
            normality_padded = np.full(len(data), np.nan)
            
            skewness_padded[window:] = skewness
            skew_persist_padded[window:] = skewness_persistence
            kurtosis_padded[window:] = kurtosis
            kurt_persist_padded[window:] = kurtosis_persistence
            normality_padded[window:] = normality
            
            features[f'returns_skewness_{window}'] = skewness_padded
            features[f'skewness_persistence_{window}'] = skew_persist_padded
            features[f'returns_kurtosis_{window}'] = kurtosis_padded
            features[f'kurtosis_persistence_{window}'] = kurt_persist_padded
            features[f'distribution_normality_{window}'] = normality_padded
        
        return features
    
    def _generate_statistical_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical persistence features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Statistical regime persistence
            stat_persistence = self._calculate_statistical_persistence(returns, window)
            
            # Distribution stability
            dist_stability = self._calculate_distribution_stability(returns, window)
            
            # Statistical regime strength
            stat_strength = self._calculate_statistical_strength(returns, window)
            
            # Pad to match data length
            persistence_padded = np.full(len(data), np.nan)
            stability_padded = np.full(len(data), np.nan)
            strength_padded = np.full(len(data), np.nan)
            
            persistence_padded[window:] = stat_persistence
            stability_padded[window:] = dist_stability
            strength_padded[window:] = stat_strength
            
            features[f'statistical_persistence_{window}'] = persistence_padded
            features[f'distribution_stability_{window}'] = stability_padded
            features[f'statistical_strength_{window}'] = strength_padded
        
        return features
    
    def _generate_correlation_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-correlation features."""
        features = {}
        windows = self.config.parameters["correlation_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Returns autocorrelation
            autocorr = self._calculate_returns_autocorrelation(returns, window)
            
            # Correlation stability
            corr_stability = self._calculate_correlation_stability(returns, window)
            
            # Cross-correlation regime features
            cross_corr = self._calculate_cross_correlation_features(returns, window)
            
            # Pad to match data length
            autocorr_padded = np.full(len(data), np.nan)
            stability_padded = np.full(len(data), np.nan)
            cross_corr_padded = np.full(len(data), np.nan)
            
            autocorr_padded[window:] = autocorr
            stability_padded[window:] = corr_stability
            cross_corr_padded[window:] = cross_corr
            
            features[f'returns_autocorr_{window}'] = autocorr_padded
            features[f'correlation_stability_{window}'] = stability_padded
            features[f'cross_correlation_{window}'] = cross_corr_padded
        
        return features
    
    def _generate_statistical_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            # Statistical regime change detection
            stat_change = self._detect_statistical_regime_changes(returns, window)
            
            # Distribution transition probability
            dist_transition = self._calculate_distribution_transition_probability(returns, window)
            
            # Statistical regime momentum
            stat_momentum = self._calculate_statistical_momentum(returns, window)
            
            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            transition_padded = np.full(len(data), np.nan)
            momentum_padded = np.full(len(data), np.nan)
            
            change_padded[window*2:] = stat_change
            transition_padded[window*2:] = dist_transition
            momentum_padded[window*2:] = stat_momentum
            
            features[f'statistical_regime_change_{window}'] = change_padded
            features[f'distribution_transition_{window}'] = transition_padded
            features[f'statistical_momentum_{window}'] = momentum_padded
        
        return features
    
    def _generate_statistical_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate statistical stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Statistical regime stability
            stat_stability = self._calculate_statistical_stability(returns, window)
            
            # Distribution entropy
            dist_entropy = self._calculate_distribution_entropy(returns, window)
            
            # Statistical regime consistency
            stat_consistency = self._calculate_statistical_consistency(returns, window)
            
            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            entropy_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)
            
            stability_padded[window:] = stat_stability
            entropy_padded[window:] = dist_entropy
            consistency_padded[window:] = stat_consistency
            
            features[f'statistical_stability_{window}'] = stability_padded
            features[f'distribution_entropy_{window}'] = entropy_padded
            features[f'statistical_consistency_{window}'] = consistency_padded
        
        return features
    
    def _calculate_rolling_skewness(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling skewness."""
        if len(returns) < window:
            return np.array([])
        
        skewness = np.zeros(len(returns) - window + 1)
        for i in range(len(skewness)):
            skewness[i] = skew(returns[i:i+window])
        
        return skewness
    
    def _calculate_rolling_kurtosis(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling kurtosis."""
        if len(returns) < window:
            return np.array([])
        
        kurt = np.zeros(len(returns) - window + 1)
        for i in range(len(kurt)):
            kurt[i] = kurtosis(returns[i:i+window])
        
        return kurt
    
    def _calculate_skewness_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate skewness persistence."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        persistence = np.zeros(len(returns))
        for i in range(window, len(returns)):
            skew_window = self._calculate_rolling_skewness(returns[i-window:i], window // 4)
            if len(skew_window) > 1:
                corr = np.corrcoef(skew_window[:-1], skew_window[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_kurtosis_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate kurtosis persistence."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        persistence = np.zeros(len(returns))
        for i in range(window, len(returns)):
            kurt_window = self._calculate_rolling_kurtosis(returns[i-window:i], window // 4)
            if len(kurt_window) > 1:
                corr = np.corrcoef(kurt_window[:-1], kurt_window[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_distribution_normality(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution normality using Jarque-Bera test."""
        if len(returns) < window:
            return np.array([])
        
        normality = np.zeros(len(returns) - window + 1)
        for i in range(len(normality)):
            try:
                jb_stat, p_value = jarque_bera(returns[i:i+window])
                # Convert p-value to normality score (higher = more normal)
                normality[i] = p_value
            except:
                normality[i] = 0
        
        return normality
    
    def _calculate_statistical_persistence(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime persistence."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        persistence = np.zeros(len(returns))
        for i in range(window, len(returns)):
            # Calculate persistence of statistical properties
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Persistence based on autocorrelation of squared returns
                squared_returns = returns_window ** 2
                corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_distribution_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution stability."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        stability = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Stability based on consistency of statistical moments
                skew_vals = self._calculate_rolling_skewness(returns_window, window // 4)
                kurt_vals = self._calculate_rolling_kurtosis(returns_window, window // 4)
                
                if len(skew_vals) > 1 and len(kurt_vals) > 1:
                    skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
                    kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
                    stability[i] = max(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability
    
    def _calculate_statistical_strength(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical regime strength."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        strength = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Strength based on how well-defined the distribution is
                skewness = skew(returns_window)
                kurtosis_val = kurtosis(returns_window)
                
                # Strength based on deviation from normal distribution
                deviation = abs(skewness) + abs(kurtosis_val - 3)
                strength[i] = max(0, 1 - deviation / 10)  # Normalize to 0-1
        
        return strength
    
    def _calculate_returns_autocorrelation(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate returns autocorrelation."""
        if len(returns) < window:
            return np.array([])
        
        autocorr = np.zeros(len(returns) - window + 1)
        for i in range(len(autocorr)):
            returns_window = returns[i:i+window]
            if len(returns_window) > 1:
                corr = np.corrcoef(returns_window[:-1], returns_window[1:])[0, 1]
                autocorr[i] = corr if not np.isnan(corr) else 0
        
        return autocorr
    
    def _calculate_correlation_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate correlation stability."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        stability = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Calculate rolling autocorrelation
                autocorr_vals = []
                sub_window = window // 4
                for j in range(0, len(returns_window) - sub_window, sub_window // 2):
                    sub_returns = returns_window[j:j+sub_window]
                    if len(sub_returns) > 1:
                        corr = np.corrcoef(sub_returns[:-1], sub_returns[1:])[0, 1]
                        if not np.isnan(corr):
                            autocorr_vals.append(corr)
                
                if len(autocorr_vals) > 1:
                    # Stability based on low variance of autocorrelations
                    stability[i] = max(0, 1 - np.std(autocorr_vals))
        
        return stability
    
    def _calculate_cross_correlation_features(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate cross-correlation features."""
        if len(returns) < window:
            return np.array([])
        
        cross_corr = np.zeros(len(returns) - window + 1)
        for i in range(len(cross_corr)):
            returns_window = returns[i:i+window]
            if len(returns_window) > 2:
                # Cross-correlation between returns and absolute returns
                abs_returns = np.abs(returns_window)
                corr = np.corrcoef(returns_window, abs_returns)[0, 1]
                cross_corr[i] = corr if not np.isnan(corr) else 0
        
        return cross_corr
    
    def _detect_statistical_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect statistical regime changes."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        changes = np.zeros(len(returns))
        for i in range(window * 2, len(returns)):
            # Compare statistical properties in two consecutive windows
            returns1 = returns[i-window*2:i-window]
            returns2 = returns[i-window:i]
            
            if len(returns1) > 2 and len(returns2) > 2:
                # Compare skewness and kurtosis
                skew1, skew2 = skew(returns1), skew(returns2)
                kurt1, kurt2 = kurtosis(returns1), kurtosis(returns2)
                
                # Significant change threshold
                skew_change = abs(skew2 - skew1) / (abs(skew1) + 1e-8)
                kurt_change = abs(kurt2 - kurt1) / (abs(kurt1) + 1e-8)
                
                changes[i] = 1 if (skew_change > 0.5 or kurt_change > 0.5) else 0
        
        return changes
    
    def _calculate_distribution_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution transition probability."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        transition_prob = np.zeros(len(returns))
        for i in range(window * 2, len(returns)):
            # Calculate transition probability based on recent distribution changes
            recent_returns = returns[i-window*2:i]
            if len(recent_returns) > 2:
                # Calculate rolling skewness and kurtosis
                skew_vals = self._calculate_rolling_skewness(recent_returns, window // 2)
                kurt_vals = self._calculate_rolling_kurtosis(recent_returns, window // 2)
                
                if len(skew_vals) > 1 and len(kurt_vals) > 1:
                    # Probability based on volatility of statistical moments
                    skew_vol = np.std(skew_vals)
                    kurt_vol = np.std(kurt_vals)
                    transition_prob[i] = min(1, (skew_vol + kurt_vol) / 2)
        
        return transition_prob
    
    def _calculate_statistical_momentum(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical momentum."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        momentum = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Momentum based on trend in statistical moments
                skew_vals = self._calculate_rolling_skewness(returns_window, window // 4)
                kurt_vals = self._calculate_rolling_kurtosis(returns_window, window // 4)
                
                if len(skew_vals) > 1 and len(kurt_vals) > 1:
                    # Calculate trend in statistical moments
                    x = np.arange(len(skew_vals))
                    skew_trend = np.polyfit(x, skew_vals, 1)[0]
                    kurt_trend = np.polyfit(x, kurt_vals, 1)[0]
                    momentum[i] = (skew_trend + kurt_trend) / 2
        
        return momentum
    
    def _calculate_statistical_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical stability."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        stability = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Stability based on consistency of statistical properties
                skew_vals = self._calculate_rolling_skewness(returns_window, window // 4)
                kurt_vals = self._calculate_rolling_kurtosis(returns_window, window // 4)
                
                if len(skew_vals) > 1 and len(kurt_vals) > 1:
                    # Stability based on low coefficient of variation
                    skew_cv = np.std(skew_vals) / (np.mean(np.abs(skew_vals)) + 1e-8)
                    kurt_cv = np.std(kurt_vals) / (np.mean(np.abs(kurt_vals)) + 1e-8)
                    stability[i] = max(0, 1 - (skew_cv + kurt_cv) / 2)
        
        return stability
    
    def _calculate_distribution_entropy(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate distribution entropy."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        entropy = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 1:
                # Calculate entropy of returns distribution
                # Discretize returns into bins
                bins = np.linspace(np.min(returns_window), np.max(returns_window), 10)
                hist, _ = np.histogram(returns_window, bins=bins)
                # Normalize to probabilities
                probs = hist / (np.sum(hist) + 1e-8)
                # Calculate entropy
                entropy[i] = -np.sum(probs * np.log(probs + 1e-8))
        
        return entropy
    
    def _calculate_statistical_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate statistical consistency."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        consistency = np.zeros(len(returns))
        for i in range(window, len(returns)):
            returns_window = returns[i-window:i]
            if len(returns_window) > 2:
                # Consistency based on autocorrelation of statistical moments
                skew_vals = self._calculate_rolling_skewness(returns_window, window // 4)
                kurt_vals = self._calculate_rolling_kurtosis(returns_window, window // 4)
                
                if len(skew_vals) > 1 and len(kurt_vals) > 1:
                    skew_corr = np.corrcoef(skew_vals[:-1], skew_vals[1:])[0, 1]
                    kurt_corr = np.corrcoef(kurt_vals[:-1], kurt_vals[1:])[0, 1]
                    
                    skew_corr = skew_corr if not np.isnan(skew_corr) else 0
                    kurt_corr = kurt_corr if not np.isnan(kurt_corr) else 0
                    
                    consistency[i] = (skew_corr + kurt_corr) / 2
        
        return consistency