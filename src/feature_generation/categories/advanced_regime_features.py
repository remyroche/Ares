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

logger = logging.getLogger(__name__)

@dataclass
class RegimeFeatureConfig(FeatureConfig):
    """Configuration for regime-based features."""
    regime_detection_method: str = "hmm"  # "hmm", "kmeans", "gmm"
    n_regimes: int = 3
    lookback_period: int = 50
    regime_persistence_threshold: float = 0.7
    enable_regime_transitions: bool = True
    enable_regime_persistence: bool = True
    enable_regime_volatility: bool = True
    enable_regime_momentum: bool = True

class RegimeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for regime-based entropy features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        super().__init__()
        self.config = config or RegimeFeatureConfig()
        self.category = FeatureCategory.REGIME
        self.name = "regime_entropy"
        
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate regime entropy features."""
        try:
            # Calculate price entropy
            price_entropy = self._calculate_price_entropy(data)
            
            # Calculate volume entropy
            volume_entropy = self._calculate_volume_entropy(data)
            
            # Calculate regime transition entropy
            transition_entropy = self._calculate_transition_entropy(data)
            
            features = pd.DataFrame({
                'regime_price_entropy': price_entropy,
                'regime_volume_entropy': volume_entropy,
                'regime_transition_entropy': transition_entropy
            }, index=data.index)
            
            return FeatureResult(
                features=features,
                metadata={
                    'generator': self.name,
                    'category': self.category.value,
                    'config': self.config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating regime entropy features: {e}")
            return FeatureResult(
                features=pd.DataFrame(index=data.index),
                metadata={'error': str(e)}
            )
    
    def _calculate_price_entropy(self, data: pd.DataFrame) -> pd.Series:
        """Calculate price-based entropy."""
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )
    
    def _calculate_volume_entropy(self, data: pd.DataFrame) -> pd.Series:
        """Calculate volume-based entropy."""
        volume = data['volume'].dropna()
        return volume.rolling(window=self.config.lookback_period).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )
    
    def _calculate_transition_entropy(self, data: pd.DataFrame) -> pd.Series:
        """Calculate regime transition entropy."""
        # Simplified regime detection using volatility
        volatility = data['close'].pct_change().rolling(window=20).std()
        regimes = (volatility > volatility.quantile(0.7)).astype(int)
        
        # Calculate transition entropy
        transitions = regimes.diff().dropna()
        return transitions.rolling(window=self.config.lookback_period).apply(
            lambda x: -np.sum(x.value_counts(normalize=True) * np.log2(x.value_counts(normalize=True) + 1e-10))
        )

class RegimeComplexityGenerator(VectorizedFeatureGenerator):
    """Generator for regime complexity features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        super().__init__()
        self.config = config or RegimeFeatureConfig()
        self.category = FeatureCategory.REGIME
        self.name = "regime_complexity"
        
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate regime complexity features."""
        try:
            # Calculate regime complexity using multiple methods
            complexity_lz = self._calculate_lz_complexity(data)
            complexity_perm = self._calculate_permutation_complexity(data)
            complexity_sample = self._calculate_sample_complexity(data)
            
            features = pd.DataFrame({
                'regime_lz_complexity': complexity_lz,
                'regime_permutation_complexity': complexity_perm,
                'regime_sample_complexity': complexity_sample
            }, index=data.index)
            
            return FeatureResult(
                features=features,
                metadata={
                    'generator': self.name,
                    'category': self.category.value,
                    'config': self.config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating regime complexity features: {e}")
            return FeatureResult(
                features=pd.DataFrame(index=data.index),
                metadata={'error': str(e)}
            )
    
    def _calculate_lz_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Lempel-Ziv complexity."""
        def lz_complexity(sequence):
            if len(sequence) < 2:
                return 0
            # Simplified LZ complexity calculation
            complexity = 1
            for i in range(1, len(sequence)):
                if sequence[i] not in sequence[:i]:
                    complexity += 1
            return complexity / len(sequence)
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(lz_complexity)
    
    def _calculate_permutation_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate permutation complexity."""
        def perm_complexity(sequence):
            if len(sequence) < 3:
                return 0
            # Simplified permutation complexity
            diffs = np.diff(sequence)
            return len(set(diffs)) / len(diffs)
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(perm_complexity)
    
    def _calculate_sample_complexity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate sample complexity."""
        def sample_complexity(sequence):
            if len(sequence) < 2:
                return 0
            # Simplified sample complexity
            return np.std(sequence) / (np.mean(np.abs(sequence)) + 1e-10)
        
        returns = data['close'].pct_change().dropna()
        return returns.rolling(window=self.config.lookback_period).apply(sample_complexity)

class RegimeFractalDimensionGenerator(VectorizedFeatureGenerator):
    """Generator for regime fractal dimension features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        super().__init__()
        self.config = config or RegimeFeatureConfig()
        self.category = FeatureCategory.REGIME
        self.name = "regime_fractal_dimension"
        
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate regime fractal dimension features."""
        try:
            # Calculate fractal dimension using box-counting method
            fractal_dim = self._calculate_fractal_dimension(data)
            
            features = pd.DataFrame({
                'regime_fractal_dimension': fractal_dim
            }, index=data.index)
            
            return FeatureResult(
                features=features,
                metadata={
                    'generator': self.name,
                    'category': self.category.value,
                    'config': self.config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating regime fractal dimension features: {e}")
            return FeatureResult(
                features=pd.DataFrame(index=data.index),
                metadata={'error': str(e)}
            )
    
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

class RegimeHurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for regime Hurst exponent features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        super().__init__()
        self.config = config or RegimeFeatureConfig()
        self.category = FeatureCategory.REGIME
        self.name = "regime_hurst_exponent"
        
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate regime Hurst exponent features."""
        try:
            # Calculate Hurst exponent
            hurst_exp = self._calculate_hurst_exponent(data)
            
            features = pd.DataFrame({
                'regime_hurst_exponent': hurst_exp
            }, index=data.index)
            
            return FeatureResult(
                features=features,
                metadata={
                    'generator': self.name,
                    'category': self.category.value,
                    'config': self.config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating regime Hurst exponent features: {e}")
            return FeatureResult(
                features=pd.DataFrame(index=data.index),
                metadata={'error': str(e)}
            )
    
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

class RegimeMemoryStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for regime memory strength features."""
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        super().__init__()
        self.config = config or RegimeFeatureConfig()
        self.category = FeatureCategory.REGIME
        self.name = "regime_memory_strength"
        
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """Generate regime memory strength features."""
        try:
            # Calculate memory strength using autocorrelation
            memory_strength = self._calculate_memory_strength(data)
            
            features = pd.DataFrame({
                'regime_memory_strength': memory_strength
            }, index=data.index)
            
            return FeatureResult(
                features=features,
                metadata={
                    'generator': self.name,
                    'category': self.category.value,
                    'config': self.config.__dict__
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating regime memory strength features: {e}")
            return FeatureResult(
                features=pd.DataFrame(index=data.index),
                metadata={'error': str(e)}
            )
    
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
