"""
Advanced Regime-Specific Feature Generators

This module provides advanced feature generators specifically designed for
regime detection and clustering, focusing on entropy, fractal, and chaos
indicators that improve silhouette scores and regime separation.

Enhanced with VectorBT optimizations for high-performance regime analysis.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

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

# VectorBT optimization utilities
try:
    from ..utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
    from ..utils.vectorbt_batch_processor import VectorBTBatchProcessor, create_vectorbt_batch_processor
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    VectorBTRollingOptimizer = None
    get_vectorbt_rolling_optimizer = None
    VectorBTBatchProcessor = None
    create_vectorbt_batch_processor = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
    get_unified_vectorization_manager = None

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_UTILS_AVAILABLE = True
except ImportError:
    OPTIMIZATION_UTILS_AVAILABLE = False
    get_vectorization_optimizer = None
    get_optimized_feature_pipeline = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class RegimeEntropyGenerator(VectorizedFeatureGenerator):
    """Generator for regime entropy features with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime entropy using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        # Use VectorBT rolling apply for optimized entropy calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized entropy calculation
                entropy_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_shannon_entropy_vectorized, 
                    window=window
                )
                return entropy_series
            except Exception as e:
                warnings.warn(f"VectorBT entropy calculation failed: {e}, using fallback")
                return self._calculate_entropy_fallback(close, window, data.index)
        else:
            return self._calculate_entropy_fallback(close, window, data.index)
    
    def _calculate_shannon_entropy_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Shannon entropy for a segment (vectorized)."""
        if len(segment) == 0:
            return np.nan
        
        # Calculate histogram with fixed bins for consistency
        hist, _ = np.histogram(segment, bins=10, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))  # Add small epsilon to avoid log(0)
        return entropy
    
    def _calculate_entropy_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback entropy calculation using pandas rolling."""
        entropy_values = []
        for i in range(len(close)):
            if i < window - 1:
                entropy_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                entropy = self._calculate_shannon_entropy_vectorized(segment)
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=index)


    
class RegimeComplexityGenerator(VectorizedFeatureGenerator):
    """Generator for regime complexity features with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime complexity using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        # Use VectorBT rolling apply for optimized complexity calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized complexity calculation
                complexity_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_sample_entropy_vectorized, 
                    window=window
                )
                return complexity_series
            except Exception as e:
                warnings.warn(f"VectorBT complexity calculation failed: {e}, using fallback")
                return self._calculate_complexity_fallback(close, window, data.index)
        else:
            return self._calculate_complexity_fallback(close, window, data.index)
    
    def _calculate_sample_entropy_vectorized(self, segment: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy for a segment (vectorized)."""
        return self._sample_entropy(segment, m, r)
    
    def _calculate_complexity_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback complexity calculation using pandas rolling."""
        complexity_values = []
        for i in range(len(close)):
            if i < window - 1:
                complexity_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                complexity = self._sample_entropy(segment, m=2, r=0.2)
                complexity_values.append(complexity)
        
        return pd.Series(complexity_values, index=index)
    
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


    
class RegimeFractalDimensionGenerator(VectorizedFeatureGenerator):
    """Generator for regime fractal dimension features with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime fractal dimension using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        # Use VectorBT rolling apply for optimized fractal dimension calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized fractal dimension calculation
                fractal_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_higuchi_fractal_dimension_vectorized, 
                    window=window
                )
                return fractal_series
            except Exception as e:
                warnings.warn(f"VectorBT fractal dimension calculation failed: {e}, using fallback")
                return self._calculate_fractal_fallback(close, window, data.index)
        else:
            return self._calculate_fractal_fallback(close, window, data.index)
    
    def _calculate_higuchi_fractal_dimension_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Higuchi fractal dimension for a segment (vectorized)."""
        return self._higuchi_fractal_dimension(segment)
    
    def _calculate_fractal_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback fractal dimension calculation using pandas rolling."""
        fractal_values = []
        for i in range(len(close)):
            if i < window - 1:
                fractal_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                fractal_dim = self._higuchi_fractal_dimension(segment)
                fractal_values.append(fractal_dim)
        
        return pd.Series(fractal_values, index=index)
    
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


    
class RegimeHurstExponentGenerator(VectorizedFeatureGenerator):
    """Generator for regime Hurst exponent features with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime Hurst exponent using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        # Use VectorBT rolling apply for optimized Hurst exponent calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized Hurst exponent calculation
                hurst_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_hurst_exponent_vectorized, 
                    window=window
                )
                return hurst_series
            except Exception as e:
                warnings.warn(f"VectorBT Hurst exponent calculation failed: {e}, using fallback")
                return self._calculate_hurst_fallback(close, window, data.index)
        else:
            return self._calculate_hurst_fallback(close, window, data.index)
    
    def _calculate_hurst_exponent_vectorized(self, segment: np.ndarray) -> float:
        """Calculate Hurst exponent for a segment (vectorized)."""
        return self._calculate_hurst_exponent(segment)
    
    def _calculate_hurst_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback Hurst exponent calculation using pandas rolling."""
        hurst_values = []
        for i in range(len(close)):
            if i < window - 1:
                hurst_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                hurst = self._calculate_hurst_exponent(segment)
                hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=index)
    
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


    
class RegimeMemoryStrengthGenerator(VectorizedFeatureGenerator):
    """Generator for regime memory strength features with VectorBT optimization."""
    
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
        
        # Initialize VectorBT optimizer
        self.vectorbt_optimizer = None
        self.unified_manager = None
        
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer()
        
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = get_unified_vectorization_manager()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate regime memory strength using VectorBT optimizations."""
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        window = self.config.parameters["window"]
        
        if len(close) < window:
            return pd.Series([np.nan] * len(close), index=data.index)
        
        # Use VectorBT rolling apply for optimized memory strength calculation
        if VECTORBT_AVAILABLE and self.vectorbt_optimizer:
            try:
                # Use VectorBT rolling apply for vectorized memory strength calculation
                memory_series = self.vectorbt_optimizer.rolling_apply(
                    close, 
                    self._calculate_memory_strength_vectorized, 
                    window=window
                )
                return memory_series
            except Exception as e:
                warnings.warn(f"VectorBT memory strength calculation failed: {e}, using fallback")
                return self._calculate_memory_fallback(close, window, data.index)
        else:
            return self._calculate_memory_fallback(close, window, data.index)
    
    def _calculate_memory_strength_vectorized(self, segment: np.ndarray) -> float:
        """Calculate memory strength for a segment (vectorized)."""
        return self._calculate_memory_strength(segment)
    
    def _calculate_memory_fallback(self, close: pd.Series, window: int, index: pd.Index) -> pd.Series:
        """Fallback memory strength calculation using pandas rolling."""
        memory_values = []
        for i in range(len(close)):
            if i < window - 1:
                memory_values.append(np.nan)
            else:
                segment = close.iloc[i-window+1:i+1].values
                memory_strength = self._calculate_memory_strength(segment)
                memory_values.append(memory_strength)
        
        return pd.Series(memory_values, index=index)
    
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
    """Create advanced regime feature generators with VectorBT optimization."""
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


def create_vectorbt_optimized_regime_generators() -> List[FeatureGenerator]:
    """Create VectorBT-optimized regime feature generators with enhanced performance."""
    generators = []
    
    # Enhanced regime entropy features with multiple windows
    for window in [5, 10, 15, 20, 25, 30]:
        generators.append(RegimeEntropyGenerator(window))
    
    # Enhanced regime complexity features
    for window in [3, 5, 7, 10, 15]:
        generators.append(RegimeComplexityGenerator(window))
    
    # Enhanced regime fractal dimension features
    for window in [10, 15, 20, 25, 30, 40]:
        generators.append(RegimeFractalDimensionGenerator(window))
    
    # Enhanced regime Hurst exponent features
    for window in [10, 15, 20, 25, 30, 40]:
        generators.append(RegimeHurstExponentGenerator(window))
    
    # Enhanced regime memory strength features
    for window in [5, 8, 10, 12, 15, 20]:
        generators.append(RegimeMemoryStrengthGenerator(window))
    
    return generators


def process_regime_features_batch(data: pd.DataFrame, 
                                generators: Optional[List[FeatureGenerator]] = None,
                                use_vectorbt: bool = True,
                                **kwargs) -> pd.DataFrame:
    """
    Process regime features in batch using VectorBT optimizations.
    
    Args:
        data: Input OHLCV data
        generators: List of feature generators (uses default if None)
        use_vectorbt: Whether to use VectorBT batch processing
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with generated regime features
    """
    if generators is None:
        generators = create_vectorbt_optimized_regime_generators()
    
    if use_vectorbt and VECTORBT_OPTIMIZER_AVAILABLE and create_vectorbt_batch_processor:
        try:
            # Use VectorBT batch processor for optimal performance
            batch_processor = create_vectorbt_batch_processor()
            
            # Convert generators to batch processors
            batch_generators = []
            for generator in generators:
                if hasattr(generator, 'process_batch'):
                    batch_generators.append(generator)
                else:
                    # Create a wrapper for batch processing
                    class BatchWrapper:
                        def __init__(self, gen):
                            self.generator = gen
                        
                        def process_batch(self, data, **kwargs):
                            return self.generator._generate_feature(data, **kwargs)
                        
                        def get_required_columns(self):
                            return getattr(self.generator.config, 'required_columns', ['close'])
                    
                    batch_generators.append(BatchWrapper(generator))
            
            # Process features in batch
            result = batch_processor.process_features_batch(data, batch_generators, **kwargs)
            return result
            
        except Exception as e:
            warnings.warn(f"VectorBT batch processing failed: {e}, using sequential processing")
            return _process_regime_features_sequential(data, generators, **kwargs)
    else:
        return _process_regime_features_sequential(data, generators, **kwargs)


def _process_regime_features_sequential(data: pd.DataFrame, 
                                      generators: List[FeatureGenerator],
                                      **kwargs) -> pd.DataFrame:
    """Process regime features sequentially (fallback)."""
    results = []
    
    for generator in generators:
        try:
            feature_result = generator._generate_feature(data, **kwargs)
            if not feature_result.empty:
                results.append(feature_result)
        except Exception as e:
            warnings.warn(f"Generator {generator.__class__.__name__} failed: {e}")
            continue
    
    if results:
        return pd.concat(results, axis=1)
    else:
        return pd.DataFrame(index=data.index)


__all__ = [
    'RegimeEntropyGenerator',
    'RegimeComplexityGenerator', 
    'RegimeFractalDimensionGenerator',
    'RegimeHurstExponentGenerator',
    'RegimeMemoryStrengthGenerator',
    'create_advanced_regime_generators',
    'create_vectorbt_optimized_regime_generators',
    'process_regime_features_batch'
]

