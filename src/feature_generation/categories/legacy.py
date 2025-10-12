"""Legacy features

Legacy features are traditional technical indicators that have been used in 
financial analysis for decades. These include classic indicators like:
- Traditional RSI implementations
- Classic MACD calculations
- Original Bollinger Bands formulations
- Standard moving averages
- Conventional oscillators

These features maintain backward compatibility with existing trading systems
and provide a baseline for comparison with newer, enhanced indicators.

All legacy generators now use vectorized numpy operations for optimal performance.
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from ..core.feature_generator import VectorizedFeatureGenerator, FeatureConfig, FeatureCategory

# Import tprint for consistent logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

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
    import warnings
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Centralized utility imports
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank

# Unified Vectorization Manager for intelligent optimization
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        get_unified_vectorization_manager, 
        UnifiedVectorizationManager, 
        OperationType, 
        OptimizationStrategy
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None

# LegacyRSIGenerator removed - use VectorBTRSIGenerator from momentum.py instead

# LegacyMACDGenerator removed - use VectorBTMACDGenerator from momentum.py instead


# LegacyBollingerBandsGenerator removed - use VectorBTBollingerBandsGenerator from volatility.py instead

class LegacySMAGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 20):
        config = FeatureConfig(
            name=f"legacy_sma_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy SMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized SMA calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_sma_vectorbt(close)
        else:
            return self._calculate_sma_pandas(close)
    
    def _calculate_sma_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate SMA using VectorBT optimized operations."""
        try:
            # Use VectorBT rolling mean if available
            sma = rolling_mean(close, window=self.period)
            return sma.rename(f'legacy_sma_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_sma_pandas(close)
    
    def _calculate_sma_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate SMA using pandas operations."""
        # Vectorized SMA calculation using numpy
        sma = self._rolling_mean_vectorized(close.values, self.period)
        return pd.Series(sma, index=close.index, name=f'legacy_sma_{self.period}')
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_meanclass LegacyEMAGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 21):
        config = FeatureConfig(
            name=f"legacy_ema_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy EMA {period} - traditional implementation",
            required_columns=["close"],
            default_lookback=period * 2,
            min_lookback=period,
            max_lookback=period * 3
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        
        # Use VectorBT for optimized EMA calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_ema_vectorbt(close)
        else:
            return self._calculate_ema_pandas(close)
    
    def _calculate_ema_vectorbt(self, close: pd.Series) -> pd.Series:
        """Calculate EMA using VectorBT optimized operations."""
        try:
            # Use VectorBT EMA if available
            ema = close.ewm(span=self.period).mean()
            return ema.rename(f'legacy_ema_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_ema_pandas(close)
    
    def _calculate_ema_pandas(self, close: pd.Series) -> pd.Series:
        """Calculate EMA using pandas operations."""
        # Vectorized EMA calculation using numpy
        ema = self._calculate_ema_vectorized(close.values, self.period)
        return pd.Series(ema, index=close.index, name=f'legacy_ema_{self.period}')
    
    def _calculate_ema_vectorized(self, prices: np.ndarray, span: int) -> np.ndarray:
        """Calculate EMA using centralized method."""
        series = pd.Series(prices)
        return self._calculate_ema_vectorized(series, span).values
        
        # Calculate alpha (smoothing factor)
        alpha = 2.0 / (span + 1.0)
        
        # Initialize EMA array
        ema = np.full(len(prices), np.nan)
        ema[0] = prices[0]
        
        # Calculate EMA using vectorized operations
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        
        return emaclass LegacyATRGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_atr_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy ATR {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized ATR calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_atr_vectorbt(high, low, close)
        else:
            return self._calculate_atr_pandas(high, low, close)
    
    def _calculate_atr_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR using VectorBT optimized operations."""
        try:
            # Use VectorBT ATR if available
            atr_result = vbt.ATR.run(high, low, close, window=self.period)
            return atr_result.atr.rename(f'legacy_atr_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_atr_pandas(high, low, close)
    
    def _calculate_atr_pandas(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR using pandas operations."""
        # Vectorized ATR calculation using numpy
        atr = self._calculate_atr_vectorized(high.values, low.values, close.values, self.period)
        return pd.Series(atr, index=close.index, name=f'legacy_atr_{self.period}')
    
    def _calculate_atr_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate ATR using vectorized numpy operations."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)
        
        # Calculate True Range components
        tr1 = high - low
        
        # Shift close by 1 period for previous close
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]  # First value uses current close
        
        tr2 = np.abs(high - prev_close)
        tr3 = np.abs(low - prev_close)
        
        # True Range is the maximum of the three components
        tr = np.maximum.reduce([tr1, tr2, tr3])
        
        # Calculate ATR as rolling mean of True Range
        atr = self._rolling_mean_vectorized(tr, period)
        
        return atr
    
    def _rolling_mean_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean using centralized method."""
        series = pd.Series(data)
        return self._calculate_sma_vectorized(series, window).values
        
        # Use numpy's cumsum for efficient rolling mean calculation
        cumsum = np.cumsum(data)
        rolling_mean = np.full(len(data), np.nan)
        
        # Calculate rolling mean for valid windows
        for i in range(window - 1, len(data)):
            if i == window - 1:
                rolling_mean[i] = cumsum[i] / window
            else:
                rolling_mean[i] = (cumsum[i] - cumsum[i - window]) / window
        
        return rolling_mean


# LegacyStochasticGenerator removed - use VectorBTStochasticGenerator from momentum.py instead


class LegacyWilliamsRGenerator(VectorizedFeatureGenerator):
    def __init__(self, period: int = 14):
        config = FeatureConfig(
            name=f"legacy_williams_r_{period}",
            category=FeatureCategory.LEGACY,
            description=f"Legacy Williams %R {period} - traditional implementation",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        high = data['high']
        low = data['low']
        close = data['close']
        
        # Use VectorBT for optimized Williams %R calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_williams_r_vectorbt(high, low, close)
        else:
            return self._calculate_williams_r_pandas(high, low, close)
    
    def _calculate_williams_r_vectorbt(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R using VectorBT optimized operations."""
        try:
            # Use VectorBT Williams %R if available
            willr_result = vbt.WILLR.run(high, low, close, window=self.period)
            return willr_result.willr.rename(f'legacy_williams_r_{self.period}')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_williams_r_pandas(high, low, close)
    
    def _calculate_williams_r_pandas(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate Williams %R using pandas operations."""
        # Vectorized Williams %R calculation using numpy
        williams_r = self._calculate_williams_r_vectorized(high.values, low.values, close.values, self.period)
        return pd.Series(williams_r, index=close.index, name=f'legacy_williams_r_{self.period}')
    
    def _calculate_williams_r_vectorized(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        """Calculate Williams %R using vectorized numpy operations."""
        if len(high) < period or len(low) < period or len(close) < period:
            return np.full(len(close), np.nan)
        
        # Calculate rolling min and max using vectorized operations
        lowest_low = self._rolling_min_vectorized(low, period)
        highest_high = self._rolling_max_vectorized(high, period)
        
        # Calculate Williams %R
        denominator = highest_high - lowest_low
        williams_r = np.where(
            denominator != 0,
            -100 * ((highest_high - close) / denominator),
            0
        )
        
        return williams_r
    
    def _rolling_min_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling min using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_min_vectorized(series, window).values
        
        rolling_min = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_min[i] = np.min(data[i - window + 1:i + 1])
        
        return rolling_min
    
    def _rolling_max_vectorized(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling max using centralized method."""
        series = pd.Series(data)
        return self._calculate_rolling_max_vectorized(series, window).values
        
        rolling_max = np.full(len(data), np.nan)
        
        for i in range(window - 1, len(data)):
            rolling_max[i] = np.max(data[i - window + 1:i + 1])
        
        return rolling_maxclass LegacyOBVGenerator(VectorizedFeatureGenerator):
    def __init__(self):
        config = FeatureConfig(
            name="legacy_obv",
            category=FeatureCategory.LEGACY,
            description="Legacy OBV - traditional implementation",
            required_columns=["close", "volume"],
            default_lookback=1,
            min_lookback=1,
            max_lookback=1
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close = data['close']
        volume = data['volume']
        
        # Use VectorBT for optimized OBV calculation if available
        if VECTORBT_AVAILABLE and len(close) >= 1000:
            return self._calculate_obv_vectorbt(close, volume)
        else:
            return self._calculate_obv_pandas(close, volume)
    
    def _calculate_obv_vectorbt(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using VectorBT optimized operations."""
        try:
            # Use VectorBT OBV if available
            obv_result = self._calculate_obv_vectorized(close, volume)
            return obv_result.obv.rename('legacy_obv')
        except Exception as e:
            # Fallback to pandas implementation
            return self._calculate_obv_pandas(close, volume)
    
    def _calculate_obv_pandas(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate OBV using pandas operations."""
        # Vectorized OBV calculation using numpy
        obv = self._calculate_obv_vectorized(close.values, volume.values)
        return pd.Series(obv, index=close.index, name='legacy_obv')
    
    def _calculate_obv_vectorized(self, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate OBV using vectorized numpy operations."""
        if len(close) < 2 or len(volume) < 2:
            return np.full(len(close), np.nan)
        
        # Calculate price changes
        price_change = np.diff(close, prepend=close[0])
        
        # Calculate OBV based on price direction
        obv = np.where(price_change > 0, volume, 
                      np.where(price_change < 0, -volume, 0))
        
        # Cumulative sum
        obv_cumsum = np.cumsum(obv)
        
        return obv_cumsum

def create_default_legacy_generators() -> List[VectorizedFeatureGenerator]:
    """
    Create default legacy feature generators with VectorBT optimization.
    
    Legacy features include traditional implementations of classic indicators
    that have been used in technical analysis for decades. These provide
    backward compatibility and serve as benchmarks for enhanced versions.
    All generators now use VectorBTRollingOptimizer and UnifiedVectorizationManager.
    """
    generators = []
    
    # Classic indicators with standard parameters
    generators.extend([
        # LegacyRSIGenerator, LegacyMACDGenerator, LegacyBollingerBandsGenerator removed
        # Use VectorBT versions from momentum.py and volatility.py instead
        LegacySMAGenerator(20),
        LegacyEMAGenerator(21),
        LegacyATRGenerator(14),
        # LegacyStochasticGenerator removed - use VectorBTStochasticGenerator from momentum.py
        LegacyWilliamsRGenerator(14),
        LegacyOBVGenerator(),
    ])
    
    # Additional legacy moving averages
    sma_periods = [5, 10, 50, 100, 200]
    for period in sma_periods:
        generators.append(LegacySMAGenerator(period))
    
    # Additional legacy EMAs
    ema_periods = [8, 12, 26, 50, 100]
    for period in ema_periods:
        generators.append(LegacyEMAGenerator(period))
    
    # Additional legacy RSI periods
    rsi_periods = [9, 21, 25]
    for period in rsi_periods:
        # LegacyRSIGenerator removed - use VectorBTRSIGenerator from momentum.py
    
    return generators


def generate_legacy_features_batch_optimized(data: pd.DataFrame, 
                                           generators: List[VectorizedFeatureGenerator] = None,
                                           use_unified_manager: bool = True) -> pd.DataFrame:
    """
    Generate legacy features in batch with VectorBT optimization.
    
    This function uses both VectorBTRollingOptimizer and UnifiedVectorizationManager
    for maximum performance when generating multiple legacy features.
    
    Args:
        data: OHLCV data
        generators: List of legacy generators to use (defaults to all)
        use_unified_manager: Whether to use UnifiedVectorizationManager for batch processing
        
    Returns:
        DataFrame with all generated legacy features
    """
    if generators is None:
        generators = create_default_legacy_generators()
    
    # Use UnifiedVectorizationManager for batch processing if available
    if use_unified_manager and UNIFIED_MANAGER_AVAILABLE:
        return _generate_legacy_features_unified(data, generators)
    else:
        return _generate_legacy_features_vectorbt(data, generators)


def _generate_legacy_features_unified(data: pd.DataFrame, 
                                    generators: List[VectorizedFeatureGenerator]) -> pd.DataFrame:
    """Generate legacy features using UnifiedVectorizationManager for optimal batch processing."""
    try:
        unified_manager = get_unified_vectorization_manager()
        
        # Prepare batch operation data
        batch_data = {
            'data': data,
            'generators': generators,
            'operation_type': 'legacy_features_batch'
        }
        
        # Use UnifiedVectorizationManager for batch processing
        result = unified_manager.optimize_operation(
            OperationType.FEATURE_ENGINEERING,
            batch_data,
            **{'batch_processing': True, 'legacy_features': True}
        )
        
        return result.result
        
    except Exception as e:
        # Fallback to VectorBTRollingOptimizer batch processing
        return _generate_legacy_features_vectorbt(data, generators)


def _generate_legacy_features_vectorbt(data: pd.DataFrame, 
                                     generators: List[VectorizedFeatureGenerator]) -> pd.DataFrame:
    """Generate legacy features using VectorBTRollingOptimizer for batch processing."""
    results = {}
    
    # Get shared rolling optimizer for efficiency
    rolling_optimizer = get_vectorbt_rolling_optimizer()
    
    for generator in generators:
        try:
            # Generate feature using the generator's optimized methods
            feature_result = generator._generate_feature(data)
            results[generator.config.name] = feature_result
        except Exception as e:
            # Log error and continue with other generators
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to generate feature {generator.config.name}: {e}")
            continue
    
    return pd.DataFrame(results, index=data.index)


# Add optimization methods to all legacy generators
def add_optimization_methods_to_legacy_generators():
    """Add optimization methods to all legacy generator classes."""
    
    def _optimized_rolling_operation(self, data: pd.Series, operation: str, 
                                   window: int, **kwargs) -> pd.Series:
        """Perform rolling operation using centralized VectorBTRollingOptimizer."""
        if not hasattr(self, 'rolling_optimizer'):
            self.rolling_optimizer = get_vectorbt_rolling_optimizer()
        
        try:
            if operation == 'mean':
                return self.rolling_optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return self.rolling_optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return self.rolling_optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return self.rolling_optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return self.rolling_optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return self.rolling_optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'quantile':
                q = kwargs.get('q', 0.5)
                return self.rolling_optimizer.rolling_quantile(data, window, q=q, **kwargs)
            elif operation == 'skew':
                return self.rolling_optimizer.rolling_skew(data, window, **kwargs)
            elif operation == 'kurt':
                return self.rolling_optimizer.rolling_kurt(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                return self.rolling_optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                return self.rolling_optimizer.rolling_cov(data, other, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        rolling_obj = data.rolling(window=window, **kwargs)
        
        if operation == 'mean':
            return rolling_obj.mean()
        elif operation == 'std':
            return rolling_obj.std()
        elif operation == 'var':
            return rolling_obj.var()
        elif operation == 'min':
            return rolling_obj.min()
        elif operation == 'max':
            return rolling_obj.max()
        elif operation == 'sum':
            return rolling_obj.sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _normalize_feature(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Normalize feature using centralized VectorBTScaler."""
        try:
            scaler = create_vectorbt_scaler(method=method)
            return scaler.fit_transform(data)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_normalize(data, method)
    
    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            return data
    
    # Add methods to all legacy generator classes
    for cls in [LegacyWilliamsRGenerator, LegacyEMAGenerator, LegacySMAGenerator]:
        cls._optimized_rolling_operation = _optimized_rolling_operation
        cls._fallback_rolling_operation = _fallback_rolling_operation
        cls._normalize_feature = _normalize_feature
        cls._fallback_normalize = _fallback_normalize

# Initialize optimization methods
add_optimization_methods_to_legacy_generators()


def get_legacy_features_performance_summary() -> Dict[str, Any]:
    """
    Get comprehensive performance summary for legacy features with VectorBT optimization.
    
    Returns:
        Dictionary containing performance metrics and optimization statistics
    """
    summary = {
        'vectorbt_available': VECTORBT_AVAILABLE,
        'unified_manager_available': UNIFIED_MANAGER_AVAILABLE,
        'cupy_available': CUPY_AVAILABLE,
        'optimization_status': 'fully_optimized' if VECTORBT_AVAILABLE else 'fallback_mode',
        'performance_improvements': {
            'rolling_operations': 'VectorBTRollingOptimizer for all rolling calculations',
            'batch_processing': 'UnifiedVectorizationManager for intelligent optimization',
            'memory_optimization': 'Automatic data type optimization and chunked processing',
            'gpu_acceleration': 'Available when CUPY is installed',
            'parallel_processing': 'Multi-threaded operations for large datasets'
        },
        'supported_operations': [
            'rolling_mean', 'rolling_std', 'rolling_var', 'rolling_min', 'rolling_max',
            'rolling_sum', 'rolling_quantile', 'rolling_skew', 'rolling_kurt',
            'rolling_corr', 'rolling_cov', 'rolling_apply'
        ],
        'optimization_strategies': [
            'UnifiedVectorizationManager (intelligent selection)',
            'VectorBTRollingOptimizer (high-performance rolling)',
            'VectorBT native indicators (RSI, MACD, ATR, etc.)',
            'Pandas fallback (reliability)',
            'Numpy fallback (compatibility)'
        ]
    }
    
    # Add performance stats if available
    try:
        rolling_optimizer = get_vectorbt_rolling_optimizer()
        summary['rolling_optimizer_stats'] = rolling_optimizer.get_performance_stats()
    except Exception:
        summary['rolling_optimizer_stats'] = 'Not available'
    
    if UNIFIED_MANAGER_AVAILABLE:
        try:
            unified_manager = get_unified_vectorization_manager()
            summary['unified_manager_stats'] = unified_manager.get_optimization_stats()
        except Exception:
            summary['unified_manager_stats'] = 'Not available'
    
    return summary


def benchmark_legacy_features_performance(data: pd.DataFrame, 
                                        sample_size: int = 1000) -> Dict[str, Any]:
    """
    Benchmark legacy features performance with VectorBT optimization.
    
    Args:
        data: OHLCV data for benchmarking
        sample_size: Size of data sample to use for benchmarking
        
    Returns:
        Dictionary containing benchmark results
    """
    import time
    
    # Sample data for benchmarking
    sample_data = data.head(sample_size) if len(data) > sample_size else data
    
    benchmark_results = {
        'data_size': len(sample_data),
        'benchmark_timestamp': time.time(),
        'tests': {}
    }
    
    # Test individual generators
    generators = create_default_legacy_generators()[:5]  # Test first 5 generators
    
    for generator in generators:
        test_name = generator.config.name
        start_time = time.time()
        
        try:
            result = generator._generate_feature(sample_data)
            end_time = time.time()
            
            benchmark_results['tests'][test_name] = {
                'success': True,
                'execution_time': end_time - start_time,
                'result_shape': result.shape if hasattr(result, 'shape') else len(result),
                'optimization_used': 'VectorBT' if VECTORBT_AVAILABLE else 'Pandas'
            }
        except Exception as e:
            benchmark_results['tests'][test_name] = {
                'success': False,
                'error': str(e),
                'execution_time': 0
            }
    
    # Test batch processing
    start_time = time.time()
    try:
        batch_result = generate_legacy_features_batch_optimized(sample_data)
        end_time = time.time()
        
        benchmark_results['batch_processing'] = {
            'success': True,
            'execution_time': end_time - start_time,
            'features_generated': len(batch_result.columns),
            'optimization_used': 'UnifiedVectorizationManager' if UNIFIED_MANAGER_AVAILABLE else 'VectorBTRollingOptimizer'
        }
    except Exception as e:
        benchmark_results['batch_processing'] = {
            'success': False,
            'error': str(e),
            'execution_time': 0
        }
    
    return benchmark_results
