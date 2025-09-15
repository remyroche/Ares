"""
Advanced HMM Regime Feature Generator

This module provides sophisticated HMM regime-based features using Hidden Markov Models
to detect multiple market regimes with high precision. Supports 8-16+ market states
based on volatility, trend, momentum, and volume characteristics.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import (
    FeatureGenerator, 
    FeatureConfig, 
    FeatureCategory,
    VectorizedFeatureGenerator
)

# Try to import HMM libraries, fall back to simplified implementation if not available
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

# Advanced HMM Regime Detection Classes

class HMMRegimeDetector:
    """Advanced HMM-based regime detector supporting 8-16+ market states."""
    
    def __init__(self, n_states: int = 8, window: int = 50):
        self.n_states = n_states
        self.window = window
        self.scaler = StandardScaler() if HMM_AVAILABLE else None
        self.model = None
        self.is_fitted = False
        
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for HMM regime detection."""
        close = data['close']
        high = data.get('high', close)
        low = data.get('low', close)
        volume = data.get('volume', pd.Series(1, index=close.index))
        
        # Calculate comprehensive features
        returns = close.pct_change().fillna(0)
        
        features = []
        
        # Volatility features
        vol_5 = returns.rolling(5).std()
        vol_10 = returns.rolling(10).std()
        vol_20 = returns.rolling(20).std()
        features.extend([vol_5, vol_10, vol_20])
        
        # Trend features
        sma_5 = close.rolling(5).mean()
        sma_20 = close.rolling(20).mean()
        trend_5 = (close - sma_5) / sma_5
        trend_20 = (close - sma_20) / sma_20
        features.extend([trend_5, trend_20])
        
        # Momentum features
        momentum_5 = returns.rolling(5).sum()
        momentum_10 = returns.rolling(10).sum()
        features.extend([momentum_5, momentum_10])
        
        # Volume features
        vol_ratio = volume / volume.rolling(20).mean()
        vol_vol = volume.pct_change().rolling(5).std()
        features.extend([vol_ratio, vol_vol])
        
        # Range features
        daily_range = (high - low) / close
        range_5 = daily_range.rolling(5).mean()
        features.extend([daily_range, range_5])
        
        # Combine all features
        feature_matrix = np.column_stack([f.fillna(0).values for f in features])
        return feature_matrix
    
    def fit(self, data: pd.DataFrame):
        """Fit the HMM model to data."""
        if len(data) < self.window:
            return
            
        feature_matrix = self._extract_features(data)
        
        if HMM_AVAILABLE:
            # Use Gaussian Mixture Model as HMM approximation
            self.model = GaussianMixture(n_components=self.n_states, random_state=42)
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            self.model.fit(feature_matrix_scaled)
            self.is_fitted = True
        else:
            # Simplified clustering-based approach
            self.is_fitted = True
    
    def predict_regimes(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime labels."""
        if not self.is_fitted or len(data) < self.window:
            return np.zeros(len(data), dtype=int)
            
        feature_matrix = self._extract_features(data)
        
        if HMM_AVAILABLE and self.model is not None:
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            regimes = self.model.predict(feature_matrix_scaled)
        else:
            # Simplified regime detection
            returns = data['close'].pct_change().fillna(0)
            vol = returns.rolling(self.window).std()
            trend = data['close'].rolling(self.window).apply(lambda x: (x[-1] - x[0]) / x[0])
            
            # Create regimes based on volatility and trend
            regimes = np.zeros(len(data), dtype=int)
            vol_threshold = vol.quantile(0.75)
            trend_threshold = 0.02
            
            regimes[(vol > vol_threshold) & (trend > trend_threshold)] = 1   # High vol, bullish
            regimes[(vol > vol_threshold) & (trend < -trend_threshold)] = 2  # High vol, bearish
            regimes[(vol <= vol_threshold) & (trend > trend_threshold)] = 3  # Low vol, bullish
            regimes[(vol <= vol_threshold) & (trend < -trend_threshold)] = 4 # Low vol, bearish
            
        return regimes
    
    def predict_probabilities(self, data: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities."""
        if not self.is_fitted or len(data) < self.window:
            return np.ones((len(data), self.n_states)) / self.n_states
            
        feature_matrix = self._extract_features(data)
        
        if HMM_AVAILABLE and self.model is not None:
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            probabilities = self.model.predict_proba(feature_matrix_scaled)
        else:
            # Simplified probability calculation
            probabilities = np.ones((len(data), self.n_states)) / self.n_states
            
        return probabilities

# HMM Regime Feature Generators

class HMMRegimeLabelGenerator(FeatureGenerator):
    """Generator for HMM-based regime labels (8-16+ states)."""
    
    def __init__(self, n_states: int = 8, window: int = 50):
        config = FeatureConfig(
            name=f"hmm_regime_label_{n_states}_{window}",
            category=FeatureCategory.HMM_REGIME,
            description=f"HMM regime label with {n_states} states over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'n_states': n_states, 'window': window}
        )
        super().__init__(config)
        self.n_states = n_states
        self.window = window
        self.detector = HMMRegimeDetector(n_states, window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if len(data) < self.window:
            return pd.Series(0, index=data.index, name=self.config.name)
            
        # Fit model on historical data
        self.detector.fit(data)
        
        # Predict regimes
        regimes = self.detector.predict_regimes(data)
        
        return pd.Series(regimes, index=data.index, name=self.config.name)

class HMMRegimeProbabilityGenerator(FeatureGenerator):
    """Generator for HMM regime probabilities."""
    
    def __init__(self, regime_id: int, n_states: int = 8, window: int = 50):
        config = FeatureConfig(
            name=f"hmm_regime_{regime_id}_probability_{n_states}_{window}",
            category=FeatureCategory.HMM_REGIME,
            description=f"HMM probability of being in regime {regime_id} with {n_states} states over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'regime_id': regime_id, 'n_states': n_states, 'window': window}
        )
        super().__init__(config)
        self.regime_id = regime_id
        self.n_states = n_states
        self.window = window
        self.detector = HMMRegimeDetector(n_states, window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if len(data) < self.window:
            return pd.Series(0.0, index=data.index, name=self.config.name)
            
        # Fit model on historical data
        self.detector.fit(data)
        
        # Predict probabilities
        probabilities = self.detector.predict_probabilities(data)
        
        # Return probability for specific regime
        if self.regime_id < self.n_states:
            regime_probs = probabilities[:, self.regime_id]
        else:
            regime_probs = np.zeros(len(data))
            
        return pd.Series(regime_probs, index=data.index, name=self.config.name)

class HMMRegimeTransitionGenerator(FeatureGenerator):
    """Generator for HMM regime transition probabilities."""
    
    def __init__(self, n_states: int = 8, window: int = 50):
        config = FeatureConfig(
            name=f"hmm_regime_transition_{n_states}_{window}",
            category=FeatureCategory.HMM_REGIME,
            description=f"HMM regime transition probability with {n_states} states over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'n_states': n_states, 'window': window}
        )
        super().__init__(config)
        self.n_states = n_states
        self.window = window
        self.detector = HMMRegimeDetector(n_states, window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if len(data) < self.window:
            return pd.Series(0.0, index=data.index, name=self.config.name)
            
        # Fit model on historical data
        self.detector.fit(data)
        
        # Predict regimes
        regimes = self.detector.predict_regimes(data)
        
        # Calculate transition probability (simplified)
        regime_changes = np.diff(regimes) != 0
        transition_prob = np.concatenate([[0], regime_changes.astype(float)])
        
        # Smooth transition probability
        transition_prob = pd.Series(transition_prob, index=data.index).rolling(window=5).mean().fillna(0)
        
        return transition_prob

class HMMRegimeDurationGenerator(FeatureGenerator):
    """Generator for HMM regime duration features."""
    
    def __init__(self, n_states: int = 8, window: int = 50):
        config = FeatureConfig(
            name=f"hmm_regime_duration_{n_states}_{window}",
            category=FeatureCategory.HMM_REGIME,
            description=f"HMM regime duration with {n_states} states over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'n_states': n_states, 'window': window}
        )
        super().__init__(config)
        self.n_states = n_states
        self.window = window
        self.detector = HMMRegimeDetector(n_states, window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if len(data) < self.window:
            return pd.Series(1, index=data.index, name=self.config.name)
            
        # Fit model on historical data
        self.detector.fit(data)
        
        # Predict regimes
        regimes = self.detector.predict_regimes(data)
        
        # Calculate duration in current regime
        regime_changes = np.diff(regimes) != 0
        regime_changes = np.concatenate([[True], regime_changes])
        
        duration = np.zeros(len(data))
        current_duration = 0
        
        for i, is_change in enumerate(regime_changes):
            if is_change:
                current_duration = 1
            else:
                current_duration += 1
            duration[i] = current_duration
        
        return pd.Series(duration, index=data.index, name=self.config.name)

class HMMRegimeStabilityGenerator(FeatureGenerator):
    """Generator for HMM regime stability features."""
    
    def __init__(self, n_states: int = 8, window: int = 50):
        config = FeatureConfig(
            name=f"hmm_regime_stability_{n_states}_{window}",
            category=FeatureCategory.HMM_REGIME,
            description=f"HMM regime stability with {n_states} states over {window} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window,
            parameters={'n_states': n_states, 'window': window}
        )
        super().__init__(config)
        self.n_states = n_states
        self.window = window
        self.detector = HMMRegimeDetector(n_states, window)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        if len(data) < self.window:
            return pd.Series(1.0, index=data.index, name=self.config.name)
            
        # Fit model on historical data
        self.detector.fit(data)
        
        # Predict probabilities
        probabilities = self.detector.predict_probabilities(data)
        
        # Calculate stability as max probability (how confident we are in the regime)
        stability = np.max(probabilities, axis=1)
        
        return pd.Series(stability, index=data.index, name=self.config.name)

class HMMRegimeFeatureGenerator(VectorizedFeatureGenerator):
    """Comprehensive HMM regime feature generator."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True)
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="hmm_regime_features",
            category=FeatureCategory.HMM_REGIME,
            description="Comprehensive HMM regime features with multiple market states",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=50,
            min_lookback=20,
            max_lookback=100,
            parameters={
                "n_states": [8, 12, 16],
                "regime_windows": [30, 50, 100],
                "transition_features": True
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'HMMRegimeFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Placeholder implementation - returns regime stability as main feature
        if len(data) < 20:
            return pd.Series(0.0, index=data.index, name='hmm_regime_stability')
            
        detector = HMMRegimeDetector(n_states=8, window=50)
        detector.fit(data)
        probabilities = detector.predict_probabilities(data)
        stability = np.max(probabilities, axis=1)
        
        return pd.Series(stability, index=data.index, name='hmm_regime_stability')

def create_hmm_regime_generators(parameters: Dict[str, Any] = None) -> List[FeatureGenerator]:
    """Create a comprehensive set of HMM regime feature generators with 8-16+ states."""
    if parameters is None:
        parameters = {
            "n_states": [8, 12, 16],
            "regime_windows": [30, 50, 100]
        }
    
    generators = []
    
    for n_states in parameters["n_states"]:
        for window in parameters["regime_windows"]:
            # Core HMM regime features
            generators.extend([
                HMMRegimeLabelGenerator(n_states, window),
                HMMRegimeTransitionGenerator(n_states, window),
                HMMRegimeDurationGenerator(n_states, window),
                HMMRegimeStabilityGenerator(n_states, window),
            ])
            
            # Regime probabilities for each state
            for regime_id in range(n_states):
                generators.append(HMMRegimeProbabilityGenerator(regime_id, n_states, window))
    
    return generators

def create_default_hmm_regime_generators() -> List[FeatureGenerator]:
    """Create default HMM regime generators with comprehensive state coverage."""
    return create_hmm_regime_generators()

def create_advanced_hmm_regime_generators() -> List[FeatureGenerator]:
    """Create advanced HMM regime generators with maximum state coverage."""
    parameters = {
        "n_states": [8, 12, 16, 20],  # Up to 20 market states
        "regime_windows": [20, 30, 50, 100]
    }
    return create_hmm_regime_generators(parameters)

def create_minimal_hmm_regime_generators() -> List[FeatureGenerator]:
    """Create minimal HMM regime generators for testing."""
    parameters = {
        "n_states": [8],
        "regime_windows": [50]
    }
    return create_hmm_regime_generators(parameters)