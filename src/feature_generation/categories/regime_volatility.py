"""
Regime Volatility Feature Generator

This module provides feature generators specifically designed for regime classification
in 15-minute timeframes with 5-30 minute trade durations. Focuses on volatility
regime characteristics rather than short-term trading signals.

Key Features:
- Volatility clustering and persistence
- Volatility regime transitions
- Volatility-of-volatility measures
- Regime stability indicators
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

class RegimeVolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Feature generator for volatility regime features optimized for 15m timeframe."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="regime_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Volatility regime features for 15m timeframe regime classification",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=20,  # 5 hours in 15m periods
            min_lookback=4,       # 1 hour minimum
            max_lookback=80,      # 20 hours maximum
            parameters={
                "regime_windows": [12, 20, 40],  # 3h, 5h, 10h in 15m periods
                "persistence_windows": [8, 16, 32],  # 2h, 4h, 8h
                "vol_of_vol_windows": [16, 32, 64],  # 4h, 8h, 16h
                "transition_windows": [4, 8, 16]  # 1h, 2h, 4h
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate volatility regime features."""
        features = {}
        
        # Get base calculations
        returns = self._get_returns(data)
        if returns is None:
            return features
        
        # 1. Volatility Regime Persistence
        features.update(self._generate_volatility_persistence_features(returns, data))
        
        # 2. Volatility Clustering Features
        features.update(self._generate_volatility_clustering_features(returns, data))
        
        # 3. Volatility-of-Volatility Features
        features.update(self._generate_vol_of_vol_features(returns, data))
        
        # 4. Volatility Regime Transitions
        features.update(self._generate_volatility_transition_features(returns, data))
        
        # 5. Volatility Regime Stability
        features.update(self._generate_volatility_stability_features(returns, data))
        
        return features
    
    def _get_returns(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Calculate log returns for volatility analysis."""
        if 'close' not in data.columns:
            return None
        
        close_prices = data['close'].values
        if len(close_prices) < 2:
            return None
        
        # Use log returns for better volatility regime analysis
        returns = np.diff(np.log(close_prices))
        return returns
    
    def _generate_volatility_persistence_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility persistence features for regime detection."""
        features = {}
        windows = self.config.parameters["regime_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Rolling volatility
            vol = self._rolling_volatility(returns, window)
            
            # Volatility persistence (autocorrelation of volatility)
            vol_persistence = self._calculate_volatility_persistence(vol, window // 4)
            
            # Volatility regime strength
            vol_regime_strength = self._calculate_volatility_regime_strength(vol, window)
            
            # Pad to match data length
            vol_padded = np.full(len(data), np.nan)
            vol_persistence_padded = np.full(len(data), np.nan)
            vol_strength_padded = np.full(len(data), np.nan)
            
            vol_padded[window-1:] = vol
            vol_persistence_padded[window-1:] = vol_persistence
            vol_strength_padded[window-1:] = vol_regime_strength
            
            features[f'vol_persistence_{window}'] = vol_persistence_padded
            features[f'vol_regime_strength_{window}'] = vol_strength_padded
        
        return features
    
    def _generate_volatility_clustering_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility clustering features."""
        features = {}
        windows = self.config.parameters["regime_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # GARCH-like volatility clustering
            vol_clustering = self._calculate_volatility_clustering(returns, window)
            
            # Volatility regime consistency
            vol_consistency = self._calculate_volatility_consistency(returns, window)
            
            # Pad to match data length
            clustering_padded = np.full(len(data), np.nan)
            consistency_padded = np.full(len(data), np.nan)
            
            clustering_padded[window-1:] = vol_clustering
            consistency_padded[window-1:] = vol_consistency
            
            features[f'vol_clustering_{window}'] = clustering_padded
            features[f'vol_consistency_{window}'] = consistency_padded
        
        return features
    
    def _generate_vol_of_vol_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility-of-volatility features."""
        features = {}
        windows = self.config.parameters["vol_of_vol_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            # Calculate volatility of volatility
            vol_of_vol = self._calculate_volatility_of_volatility(returns, window)
            
            # Volatility regime uncertainty
            vol_uncertainty = self._calculate_volatility_uncertainty(returns, window)
            
            # Pad to match data length
            vol_of_vol_padded = np.full(len(data), np.nan)
            uncertainty_padded = np.full(len(data), np.nan)
            
            vol_of_vol_padded[window*2-1:] = vol_of_vol
            uncertainty_padded[window*2-1:] = vol_uncertainty
            
            features[f'vol_of_vol_{window}'] = vol_of_vol_padded
            features[f'vol_uncertainty_{window}'] = uncertainty_padded
        
        return features
    
    def _generate_volatility_transition_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime transition features."""
        features = {}
        windows = self.config.parameters["transition_windows"]
        
        for window in windows:
            if len(returns) < window * 2:
                continue
            
            # Volatility regime change detection
            vol_change = self._detect_volatility_regime_changes(returns, window)
            
            # Transition probability
            transition_prob = self._calculate_volatility_transition_probability(returns, window)
            
            # Pad to match data length
            change_padded = np.full(len(data), np.nan)
            prob_padded = np.full(len(data), np.nan)
            
            change_padded[window*2-1:] = vol_change
            prob_padded[window*2-1:] = transition_prob
            
            features[f'vol_regime_change_{window}'] = change_padded
            features[f'vol_transition_prob_{window}'] = prob_padded
        
        return features
    
    def _generate_volatility_stability_features(self, returns: np.ndarray, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate volatility regime stability features."""
        features = {}
        windows = self.config.parameters["persistence_windows"]
        
        for window in windows:
            if len(returns) < window:
                continue
            
            # Volatility regime stability
            vol_stability = self._calculate_volatility_stability(returns, window)
            
            # Regime persistence score
            persistence_score = self._calculate_regime_persistence_score(returns, window)
            
            # Pad to match data length
            stability_padded = np.full(len(data), np.nan)
            persistence_padded = np.full(len(data), np.nan)
            
            stability_padded[window-1:] = vol_stability
            persistence_padded[window-1:] = persistence_score
            
            features[f'vol_stability_{window}'] = stability_padded
            features[f'regime_persistence_{window}'] = persistence_padded
        
        return features
    
    def _rolling_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling volatility."""
        if len(returns) < window:
            return np.array([])
        
        vol = np.zeros(len(returns) - window + 1)
        for i in range(len(vol)):
            vol[i] = np.std(returns[i:i+window])
        
        return vol
    
    def _calculate_volatility_persistence(self, vol: np.ndarray, lag: int) -> np.ndarray:
        """Calculate volatility persistence using autocorrelation."""
        if len(vol) < lag + 1:
            return np.zeros(len(vol))
        
        persistence = np.zeros(len(vol))
        for i in range(lag, len(vol)):
            if i >= lag:
                corr = np.corrcoef(vol[i-lag:i], vol[i-lag+1:i+1])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence
    
    def _calculate_volatility_regime_strength(self, vol: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime strength."""
        if len(vol) < window:
            return np.zeros(len(vol))
        
        strength = np.zeros(len(vol))
        for i in range(window, len(vol)):
            vol_window = vol[i-window:i]
            # Regime strength based on consistency of volatility level
            vol_consistency = 1.0 - (np.std(vol_window) / (np.mean(vol_window) + 1e-8))
            strength[i] = max(0, min(1, vol_consistency))
        
        return strength
    
    def _calculate_volatility_clustering(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate GARCH-like volatility clustering."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        clustering = np.zeros(len(returns))
        for i in range(window, len(returns)):
            # Calculate squared returns autocorrelation
            squared_returns = returns[i-window:i] ** 2
            if len(squared_returns) > 1:
                corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                clustering[i] = corr if not np.isnan(corr) else 0
        
        return clustering
    
    def _calculate_volatility_consistency(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime consistency."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        consistency = np.zeros(len(returns))
        for i in range(window, len(returns)):
            vol_window = self._rolling_volatility(returns[i-window:i], window // 4)
            if len(vol_window) > 1:
                # Consistency based on low coefficient of variation
                cv = np.std(vol_window) / (np.mean(vol_window) + 1e-8)
                consistency[i] = max(0, 1 - cv)
        
        return consistency
    
    def _calculate_volatility_of_volatility(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility of volatility."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        vol_of_vol = np.zeros(len(returns))
        for i in range(window * 2, len(returns)):
            # Calculate volatility over first window
            vol1 = np.std(returns[i-window*2:i-window])
            # Calculate volatility over second window
            vol2 = np.std(returns[i-window:i])
            # Volatility of volatility
            vol_of_vol[i] = abs(vol2 - vol1) / (vol1 + 1e-8)
        
        return vol_of_vol
    
    def _calculate_volatility_uncertainty(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime uncertainty."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        uncertainty = np.zeros(len(returns))
        for i in range(window, len(returns)):
            vol_window = self._rolling_volatility(returns[i-window:i], window // 4)
            if len(vol_window) > 2:
                # Uncertainty based on volatility of volatility
                vol_vol = np.std(vol_window) / (np.mean(vol_window) + 1e-8)
                uncertainty[i] = min(1, vol_vol)
        
        return uncertainty
    
    def _detect_volatility_regime_changes(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Detect volatility regime changes."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        changes = np.zeros(len(returns))
        for i in range(window * 2, len(returns)):
            # Compare volatility in two consecutive windows
            vol1 = np.std(returns[i-window*2:i-window])
            vol2 = np.std(returns[i-window:i])
            
            # Significant change threshold (50% change)
            change_ratio = abs(vol2 - vol1) / (vol1 + 1e-8)
            changes[i] = 1 if change_ratio > 0.5 else 0
        
        return changes
    
    def _calculate_volatility_transition_probability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime transition probability."""
        if len(returns) < window * 2:
            return np.zeros(len(returns))
        
        transition_prob = np.zeros(len(returns))
        for i in range(window * 2, len(returns)):
            # Calculate transition probability based on recent volatility changes
            recent_vol = self._rolling_volatility(returns[i-window*2:i], window // 2)
            if len(recent_vol) > 1:
                # Probability based on volatility trend
                trend = np.polyfit(range(len(recent_vol)), recent_vol, 1)[0]
                transition_prob[i] = min(1, max(0, abs(trend) / (np.mean(recent_vol) + 1e-8)))
        
        return transition_prob
    
    def _calculate_volatility_stability(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate volatility regime stability."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        stability = np.zeros(len(returns))
        for i in range(window, len(returns)):
            vol_window = self._rolling_volatility(returns[i-window:i], window // 4)
            if len(vol_window) > 1:
                # Stability based on low volatility of volatility
                vol_vol = np.std(vol_window) / (np.mean(vol_window) + 1e-8)
                stability[i] = max(0, 1 - vol_vol)
        
        return stability
    
    def _calculate_regime_persistence_score(self, returns: np.ndarray, window: int) -> np.ndarray:
        """Calculate regime persistence score."""
        if len(returns) < window:
            return np.zeros(len(returns))
        
        persistence = np.zeros(len(returns))
        for i in range(window, len(returns)):
            vol_window = self._rolling_volatility(returns[i-window:i], window // 4)
            if len(vol_window) > 2:
                # Persistence based on autocorrelation of volatility
                corr = np.corrcoef(vol_window[:-1], vol_window[1:])[0, 1]
                persistence[i] = corr if not np.isnan(corr) else 0
        
        return persistence