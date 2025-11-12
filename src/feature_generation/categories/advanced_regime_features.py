"""
Advanced Regime Features

This module provides advanced regime-based feature generators that analyze
market regimes and generate regime-specific features for enhanced trading
strategies and risk management.
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..core.feature_generator import FeatureGenerator, FeatureCategory, FeatureResult, FeatureConfig, VectorizedFeatureGenerator
from src.utils.tprint import tprint_info, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime-based features."""
    # Required base config fields
    name: str = "regime_features"
    category: str = "REGIME"
    description: str = "Advanced regime-based features"
    required_columns: List[str] = None

    # Regime-specific config
    regime_detection_method: str = "hmm"  # "hmm", "kmeans", "gmm"
    n_regimes: int = 3
    lookback_period: int = 50  # Default for backwards compatibility
    window_sizes: List[int] = None  # Multiple responsive windows [2, 4, 8, 16]
    regime_persistence_threshold: float = 0.7
    enable_regime_transitions: bool = True
    enable_regime_persistence: bool = True
    enable_regime_volatility: bool = True
    enable_regime_momentum: bool = True

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = ["close", "volume"]
        # Default to responsive windows if not specified
        if self.window_sizes is None:
            self.window_sizes = [2, 4, 8, 16]

class RegimeEntropyGenerator(FeatureGenerator):
    """Generator for regime-based entropy features."""

    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        feature_config = FeatureConfig(
            name="regime_entropy",
            category=FeatureCategory.REGIME,
            description="Regime-based entropy features",
            required_columns=["close", "volume"],
            default_lookback=self.config.lookback_period,
            min_lookback=20,
            max_lookback=200
        )
        super().__init__(feature_config)
        tprint_info(f"   🔧 RegimeEntropyGenerator initialized with generate_features method: {hasattr(self, 'generate_features')}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime entropy features."""
        tprint_info(f"   📊 RegimeEntropyGenerator._generate_feature() called")
        features = self.generate_features(data, **kwargs)

        # Return the first feature as the primary series (following base class expectation)
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index, name=self.config.name)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime entropy features for multiple window sizes."""
        features = {}
        try:
            # Generate features for each window size
            for window in self.config.window_sizes:
                # Calculate price entropy
                price_entropy = self._calculate_price_entropy(data, window)
                features[f'regime_price_entropy_{window}'] = price_entropy.values

                # Calculate volume entropy
                volume_entropy = self._calculate_volume_entropy(data, window)
                features[f'regime_volume_entropy_{window}'] = volume_entropy.values

                # Calculate regime transition entropy
                transition_entropy = self._calculate_transition_entropy(data, window)
                features[f'regime_transition_entropy_{window}'] = transition_entropy.values

        except Exception as e:
            logger.error(f"Error generating regime entropy features: {e}")

        return features
    
    def _calculate_price_entropy(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate price-based entropy for given window."""
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=window).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )

    def _calculate_volume_entropy(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volume-based entropy for given window."""
        volume = data['volume'].dropna()
        return volume.rolling(window=window).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )
    
    def _calculate_transition_entropy(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate regime transition entropy for given window."""
        # Simplified regime detection using volatility
        volatility = data['close'].pct_change().rolling(window=min(20, window*2)).std()
        regimes = (volatility > volatility.quantile(0.7)).astype(int)

        # Calculate transition entropy
        transitions = regimes.diff().dropna()
        return transitions.rolling(window=window).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )

class RegimeComplexityGenerator(FeatureGenerator):
    """Generator for regime complexity features."""

    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        feature_config = FeatureConfig(
            name="regime_complexity",
            category=FeatureCategory.REGIME,
            description="Regime-based complexity features",
            required_columns=["close", "volume"],
            default_lookback=self.config.lookback_period,
            min_lookback=20,
            max_lookback=200
        )
        super().__init__(feature_config)
        tprint_info(f"   🔧 RegimeComplexityGenerator initialized with generate_features method: {hasattr(self, 'generate_features')}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime complexity features."""
        tprint_info(f"   📊 RegimeComplexityGenerator._generate_feature() called")
        features = self.generate_features(data, **kwargs)

        # Return the first feature as the primary series (following base class expectation)
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index, name=self.config.name)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime complexity features."""
        features = {}
        try:
            # Calculate regime complexity using multiple methods
            complexity_lz = self._calculate_lz_complexity(data)
            features['regime_lz_complexity'] = complexity_lz.values
            
            complexity_perm = self._calculate_permutation_complexity(data)
            features['regime_permutation_complexity'] = complexity_perm.values
            
            complexity_sample = self._calculate_sample_complexity(data)
            features['regime_sample_complexity'] = complexity_sample.values
            
        except Exception as e:
            logger.error(f"Error generating regime complexity features: {e}")
            
        return features
    
    def _calculate_lz_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Lempel-Ziv complexity."""
        def lz_complexity(sequence):
            try:
                if len(sequence) < 2:
                    return 0
                # Simplified LZ complexity calculation
                complexity = 1
                for i in range(1, len(sequence)):
                    if sequence[i] not in sequence[:i]:
                        complexity += 1
                return complexity / len(sequence)
            except Exception:
                return 0
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(lz_complexity, raw=False)
    
    def _calculate_permutation_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate permutation complexity."""
        def perm_complexity(sequence):
            try:
                if len(sequence) < 3:
                    return 0
                # Simplified permutation complexity
                diffs = np.diff(sequence)
                if len(diffs) == 0:
                    return 0
                return len(set(diffs)) / len(diffs)
            except Exception:
                return 0
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(perm_complexity, raw=False)
    
    def _calculate_sample_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sample complexity."""
        def sample_complexity(sequence):
            try:
                if len(sequence) < 2:
                    return 0
                # Simplified sample complexity
                mean_abs = np.mean(np.abs(sequence))
                if mean_abs == 0:
                    return 0
                return np.std(sequence) / mean_abs
            except Exception:
                return 0
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(sample_complexity, raw=False)

class RegimeFractalDimensionGenerator(FeatureGenerator):
    """Generator for regime fractal dimension features."""

    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        feature_config = FeatureConfig(
            name="regime_fractal_dimension",
            category=FeatureCategory.REGIME,
            description="Regime-based fractal dimension features",
            required_columns=["close"],
            default_lookback=self.config.lookback_period,
            min_lookback=20,
            max_lookback=200
        )
        super().__init__(feature_config)
        tprint_info(f"   🔧 RegimeFractalDimensionGenerator initialized with generate_features method: {hasattr(self, 'generate_features')}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime fractal dimension features."""
        tprint_info(f"   📊 RegimeFractalDimensionGenerator._generate_feature() called")
        features = self.generate_features(data, **kwargs)

        # Return the first feature as the primary series (following base class expectation)
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index, name=self.config.name)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime fractal dimension features."""
        features = {}
        try:
            # Calculate fractal dimension using box-counting method
            fractal_dim = self._calculate_fractal_dimension(data)
            features['regime_fractal_dimension'] = fractal_dim.values
            
        except Exception as e:
            logger.error(f"Error generating regime fractal dimension features: {e}")
            
        return features
    
    def _calculate_fractal_dimension(self, data: pd.DataFrame) -> pd.Series:
        """Calculate fractal dimension using box-counting method."""
        def box_counting_dimension(sequence):
            if len(sequence) < 10:
                return 1.0
            
            # Simplified box-counting method
            n_boxes = [2, 4, 8, 16]
            counts = []
            
            for n in n_boxes:
                if len(sequence) < n:
                    continue
                box_size = len(sequence) // n
                count = 0
                for i in range(0, len(sequence) - box_size, box_size):
                    box = sequence[i:i+box_size]
                    if len(box) > 0 and not np.all(np.isnan(box)):
                        count += 1
                counts.append(count)
            
            if len(counts) < 2:
                return 1.0
            
            # Calculate dimension from slope
            log_n = np.log(n_boxes[:len(counts)])
            log_counts = np.log(np.array(counts) + 1e-10)
            slope = np.polyfit(log_n, log_counts, 1)[0]
            return max(1.0, min(2.0, -slope))
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(box_counting_dimension)

class RegimeHurstExponentGenerator(FeatureGenerator):
    """Generator for regime Hurst exponent features."""

    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        feature_config = FeatureConfig(
            name="regime_hurst_exponent",
            category=FeatureCategory.REGIME,
            description="Regime-based Hurst exponent features",
            required_columns=["close"],
            default_lookback=self.config.lookback_period,
            min_lookback=20,
            max_lookback=200
        )
        super().__init__(feature_config)
        tprint_info(f"   🔧 RegimeHurstExponentGenerator initialized with generate_features method: {hasattr(self, 'generate_features')}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime Hurst exponent features."""
        tprint_info(f"   📊 RegimeHurstExponentGenerator._generate_feature() called")
        features = self.generate_features(data, **kwargs)

        # Return the first feature as the primary series (following base class expectation)
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index, name=self.config.name)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime Hurst exponent features."""
        features = {}
        try:
            # Calculate Hurst exponent
            hurst_exp = self._calculate_hurst_exponent(data)
            features['regime_hurst_exponent'] = hurst_exp.values
            
        except Exception as e:
            logger.error(f"Error generating regime Hurst exponent features: {e}")
            
        return features
    
    def _calculate_hurst_exponent(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Hurst exponent using R/S analysis."""
        def hurst_exponent(sequence):
            if len(sequence) < 10:
                return 0.5
            
            # Simplified Hurst exponent calculation
            n = len(sequence)
            mean_seq = np.mean(sequence)
            deviations = sequence - mean_seq
            cumulative_deviations = np.cumsum(deviations)
            range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            std_val = np.std(sequence)
            
            if std_val == 0:
                return 0.5
            
            rs = range_val / std_val
            return np.log(rs) / np.log(n) if rs > 0 else 0.5
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(hurst_exponent)

class RegimeMemoryStrengthGenerator(FeatureGenerator):
    """Generator for regime memory strength features."""

    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        self.config = config or RegimeFeatureConfig()
        feature_config = FeatureConfig(
            name="regime_memory_strength",
            category=FeatureCategory.REGIME,
            description="Regime-based memory strength features",
            required_columns=["close"],
            default_lookback=self.config.lookback_period,
            min_lookback=20,
            max_lookback=200
        )
        super().__init__(feature_config)
        tprint_info(f"   🔧 RegimeMemoryStrengthGenerator initialized with generate_features method: {hasattr(self, 'generate_features')}")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate regime memory strength features."""
        tprint_info(f"   📊 RegimeMemoryStrengthGenerator._generate_feature() called")
        features = self.generate_features(data, **kwargs)

        # Return the first feature as the primary series (following base class expectation)
        if features:
            first_feature_name = list(features.keys())[0]
            return pd.Series(features[first_feature_name], index=data.index, name=self.config.name)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

    def generate_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate regime memory strength features."""
        features = {}
        try:
            # Calculate memory strength using autocorrelation
            memory_strength = self._calculate_memory_strength(data)
            features['regime_memory_strength'] = memory_strength.values
            
        except Exception as e:
            logger.error(f"Error generating regime memory strength features: {e}")
            
        return features
    
    def _calculate_memory_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate memory strength using autocorrelation."""
        def memory_strength(sequence):
            if len(sequence) < 5:
                return 0.0
            
            # Calculate autocorrelation at lag 1
            if len(sequence) < 2:
                return 0.0
            
            autocorr = np.corrcoef(sequence[:-1], sequence[1:])[0, 1]
            return autocorr if not np.isnan(autocorr) else 0.0
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(memory_strength)

def create_advanced_regime_generators(config: Optional[RegimeFeatureConfig] = None) -> List[FeatureGenerator]:
    """Create a list of advanced regime feature generators."""
    return [
        RegimeEntropyGenerator(config),
        RegimeComplexityGenerator(config),
        RegimeFractalDimensionGenerator(config),
        RegimeHurstExponentGenerator(config),
        RegimeMemoryStrengthGenerator(config)
    ]

# Export the generators
__all__ = [
    'RegimeEntropyGenerator',
    'RegimeComplexityGenerator', 
    'RegimeFractalDimensionGenerator',
    'RegimeHurstExponentGenerator',
    'RegimeMemoryStrengthGenerator',
    'create_advanced_regime_generators',
    'RegimeFeatureConfig'
]
