"""
Advanced Regime-Specific Feature Generators

This module provides advanced feature generators specifically designed for
regime detection and clustering, focusing on entropy, fractal, and chaos
indicators that improve silhouette scores and regime separation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


class RegimeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for regime entropy features."""
    
    def __init__(self, window: int = 10):
        config = FeatureConfig(
            name=f"regime_entropy_{window}",
            category=FeatureCategory.STATISTICAL,
            description=f"Regime entropy over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate regime entropy."""
        close = data['close'].values
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        entropy_values = []
        for i in range(len(close)):
            if i < window - 1:
                entropy_values.append(np.nan)
            else:
                segment = close[i-window+1:i+1]
                # Calculate Shannon entropy
                hist, _ = np.histogram(segment, bins=10, density=True)
                hist = hist[hist > 0]  # Remove zero bins
                entropy = -np.sum(hist * np.log2(hist))
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=data.index)


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class RegimeComplexityGenerator(VectorizedFeatureGenerator):
    """Generator for regime complexity features."""
    
    def __init__(self, window: int = 5):
        config = FeatureConfig(
            name=f"regime_complexity_{window}",
            category=FeatureCategory.STATISTICAL,
            description=f"Regime complexity over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate regime complexity using sample entropy."""
        close = data['close'].values
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        complexity_values = []
        for i in range(len(close)):
            if i < window - 1:
                complexity_values.append(np.nan)
            else:
                segment = close[i-window+1:i+1]
                # Calculate sample entropy
                complexity = self._sample_entropy(segment, m=2, r=0.2)
                complexity_values.append(complexity)
        
        return pd.Series(complexity_values, index=data.index)
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        try:
            N = len(data)
            if N < m + 1:
                return 0.0
            
            # Normalize data
            data = (data - np.mean(data)) / np.std(data)
            
            # Create template vectors
            patterns = np.array([data[i:i+m] for i in range(N-m+1)])
            
            # Calculate distances
            distances = []
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j:
                        dist = np.max(np.abs(patterns[i] - patterns[j]))
                        distances.append(dist)
            
            if not distances:
                return 0.0
            
            # Count matches
            r_threshold = r * np.std(data)
            matches = np.sum(np.array(distances) <= r_threshold)
            
            if matches == 0:
                return 0.0
            
            # Calculate sample entropy
            phi = matches / (N - m + 1)
            return -np.log(phi) if phi > 0 else 0.0
            
        except Exception:
            return 0.0


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class RegimeFractalDimensionGenerator(VectorizedFeatureGenerator):
    """Generator for regime fractal dimension features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_fractal_dimension_{window}",
            category=FeatureCategory.STATISTICAL,
            description=f"Regime fractal dimension over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate regime fractal dimension using Higuchi method."""
        close = data['close'].values
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        fractal_values = []
        for i in range(len(close)):
            if i < window - 1:
                fractal_values.append(np.nan)
            else:
                segment = close[i-window+1:i+1]
                fractal_dim = self._higuchi_fractal_dimension(segment)
                fractal_values.append(fractal_dim)
        
        return pd.Series(fractal_values, index=data.index)
    
    def _higuchi_fractal_dimension(self, data: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension."""
        try:
            N = len(data)
            if N < 10:
                return 1.0
            
            # Normalize data
            data = (data - np.mean(data)) / np.std(data)
            
            # Calculate L(k) for different k values
            k_values = range(1, min(10, N//4))
            L_values = []
            
            for k in k_values:
                L_sum = 0
                for m in range(k):
                    L = 0
                    for i in range(1, (N - m) // k):
                        L += abs(data[m + i*k] - data[m + (i-1)*k])
                    L = L * (N - 1) / ((N - m) // k * k)
                    L_sum += L
                
                L_values.append(L_sum / k)
            
            if len(L_values) < 2:
                return 1.0
            
            # Calculate fractal dimension
            k_log = np.log(k_values)
            L_log = np.log(L_values)
            
            # Linear regression
            slope, _ = np.polyfit(k_log, L_log, 1)
            fractal_dim = -slope
            
            return max(1.0, min(2.0, fractal_dim))  # Bound between 1 and 2
            
        except Exception:
            return 1.0


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class RegimeHurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for regime Hurst exponent features."""
    
    def __init__(self, window: int = 20):
        config = FeatureConfig(
            name=f"regime_hurst_exponent_{window}",
            category=FeatureCategory.STATISTICAL,
            description=f"Regime Hurst exponent over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate regime Hurst exponent."""
        close = data['close'].values
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        hurst_values = []
        for i in range(len(close)):
            if i < window - 1:
                hurst_values.append(np.nan)
            else:
                segment = close[i-window+1:i+1]
                hurst = self._calculate_hurst_exponent(segment)
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=data.index)
    
    def _calculate_hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        try:
            N = len(data)
            if N < 10:
                return 0.5
            
            # Calculate returns
            returns = np.diff(data)
            
            # R/S analysis
            n_values = [N//4, N//2, N]
            rs_values = []
            
            for n in n_values:
                if n < 5:
                    continue
                
                # Calculate R/S for this n
                rs_sum = 0
                for i in range(0, N - n, n):
                    segment = returns[i:i+n]
                    mean_segment = np.mean(segment)
                    deviations = segment - mean_segment
                    cumulative_deviations = np.cumsum(deviations)
                    
                    R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
                    S = np.std(segment)
                    
                    if S > 0:
                        rs_sum += R / S
                
                if rs_sum > 0:
                    rs_values.append(rs_sum / (N // n))
            
            if len(rs_values) < 2:
                return 0.5
            
            # Calculate Hurst exponent
            n_log = np.log(n_values[:len(rs_values)])
            rs_log = np.log(rs_values)
            
            # Linear regression
            slope, _ = np.polyfit(n_log, rs_log, 1)
            hurst = slope
            
            return max(0.0, min(1.0, hurst))  # Bound between 0 and 1
            
        except Exception:
            return 0.5


    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        return data

class RegimeMemoryStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for regime memory strength features."""
    
    def __init__(self, window: int = 10):
        config = FeatureConfig(
            name=f"regime_memory_strength_{window}",
            category=FeatureCategory.STATISTICAL,
            description=f"Regime memory strength over {window} periods",
            required_columns=["close"],
            default_lookback=window,
            min_lookback=window,
            max_lookback=window * 2,
            parameters={"window": window}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Calculate regime memory strength using autocorrelation."""
        close = data['close'].values
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        memory_values = []
        for i in range(len(close)):
            if i < window - 1:
                memory_values.append(np.nan)
            else:
                segment = close[i-window+1:i+1]
                memory_strength = self._calculate_memory_strength(segment)
                memory_values.append(memory_strength)
        
        return pd.Series(memory_values, index=data.index)
    
    def _calculate_memory_strength(self, data: np.ndarray) -> float:
        """Calculate memory strength using autocorrelation."""
        try:
            N = len(data)
            if N < 5:
                return 0.0
            
            # Calculate autocorrelation for different lags
            autocorrs = []
            for lag in range(1, min(5, N//2)):
                if lag < N:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorrs.append(abs(corr))
            
            if not autocorrs:
                return 0.0
            
            # Memory strength is the average autocorrelation
            memory_strength = np.mean(autocorrs)
            return max(0.0, min(1.0, memory_strength))
            
        except Exception:
            return 0.0


def create_advanced_regime_generators() -> List[FeatureGenerator]:
    """Create advanced regime feature generators."""
    generators = []
    
    # Regime entropy features
    for window in [10, 20]:
        generators.append(RegimeEntropyGenerator(window))
    
    # Regime complexity features
    for window in [5, 10]:
        generators.append(RegimeComplexityGenerator(window))
    
    # Regime fractal dimension features
    for window in [20, 30]:
        generators.append(RegimeFractalDimensionGenerator(window))
    
    # Regime Hurst exponent features
    for window in [20, 30]:
        generators.append(RegimeHurstExponentGenerator(window))
    
    # Regime memory strength features
    for window in [10, 20]:
        generators.append(RegimeMemoryStrengthGenerator(window))
    
    return generators


__all__ = [
    'RegimeEntropyGenerator',
    'RegimeComplexityGenerator', 
    'RegimeFractalDimensionGenerator',
    'RegimeHurstExponentGenerator',
    'RegimeMemoryStrengthGenerator',
    'create_advanced_regime_generators'
]

