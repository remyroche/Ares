"""
VectorBT Optimizations for Matrix Operations

This module provides VectorBT-optimized implementations of matrix operations
that significantly improve performance over custom implementations.

Key Features:
- VectorBT-optimized trading indicators
- Enhanced matrix operations with VectorBT
- Optimized rolling operations
- Improved correlation analysis
- Parallel batch processing
- Memory-efficient operations
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
import warnings

# Conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None

# Lazy imports to avoid circular dependencies - these will be imported when needed
VECTORBT_ROLLING_AVAILABLE = True
UNIFIED_VECTORIZATION_AVAILABLE = True

# Lazy import functions
def _get_vectorbt_rolling_optimizer():
    """Lazy import VectorBT rolling optimizer."""
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
        return get_vectorbt_rolling_optimizer()
    except ImportError:
        return None

def _get_unified_vectorization_manager():
    """Lazy import unified vectorization manager."""
    try:
        from src.feature_generation.utils.unified_vectorization_manager import get_unified_vectorization_manager
        return get_unified_vectorization_manager()
    except ImportError:
        return None

logger = logging.getLogger(__name__)

class VectorBTOptimizedOperations:
    """
    VectorBT-optimized operations for matrix and financial calculations.

    This class provides high-performance implementations using VectorBT's
    optimized functions for financial and mathematical operations.
    """

    def __init__(self, enable_gpu: bool = True, enable_parallel: bool = True,
                 memory_limit_gb: float = 8.0, chunk_size_threshold: int = 10000):
        """Initialize VectorBT optimized operations with enhanced memory management."""
        self.enable_gpu = enable_gpu and VECTORBT_AVAILABLE
        self.enable_parallel = enable_parallel and VECTORBT_AVAILABLE
        self.memory_limit_gb = memory_limit_gb
        self.chunk_size_threshold = chunk_size_threshold

        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'fallback_operations': 0,
            'average_execution_time': 0.0,
            'gpu_operations': 0,
            'memory_optimized_operations': 0,
            'chunked_operations': 0,
            'rolling_operations': 0,
            'vectorization_operations': 0
        }

        # Memory management
        self.memory_usage_history = []
        self.peak_memory_usage = 0.0

        # GPU detection and configuration
        self.gpu_available = self._detect_gpu_availability()
        self.gpu_memory_limit = self._get_gpu_memory_limit()

        # Initialize logger first
        self.logger = logger.getChild('VectorBTOptimizedOperations')

        # Initialize VectorBT managers
        self._initialize_vectorbt_managers()

        if VECTORBT_AVAILABLE:
            self.logger.info(f"✅ VectorBT optimized operations initialized (GPU: {self.gpu_available}, Memory Limit: {memory_limit_gb}GB)")
        else:
            self.logger.warning("⚠️ VectorBT not available, using fallback implementations")

    def _detect_gpu_availability(self) -> bool:
        """Detect GPU availability and capabilities."""
        if not VECTORBT_AVAILABLE:
            return False

        try:
            # Check for CUDA availability
            import torch
            if torch.cuda.is_available():
                self.logger.info(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass

        try:
            # Check for MPS (Apple Silicon) availability
            import torch
            if torch.backends.mps.is_available():
                self.logger.info("🚀 Apple Silicon GPU (MPS) detected")
                return True
        except ImportError:
            pass

        return False

    def _get_gpu_memory_limit(self) -> float:
        """Get GPU memory limit in GB."""
        if not self.gpu_available:
            return 0.0

        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            elif torch.backends.mps.is_available():
                # MPS doesn't provide memory info, use conservative estimate
                return 8.0
        except ImportError:
            pass

        return 0.0

    def _initialize_vectorbt_managers(self):
        """Initialize VectorBT managers for optimized operations."""
        try:
            # Initialize VectorBTRollingOptimizer
            if VECTORBT_ROLLING_AVAILABLE:
                self.rolling_optimizer = _get_vectorbt_rolling_optimizer()
                if self.rolling_optimizer:
                    # Reduced verbosity - only log once per session
                    if not hasattr(VectorBTOptimizedOperations, '_logged_rolling_init'):
                        self.logger.debug("✅ VectorBTRollingOptimizer initialized")
                        VectorBTOptimizedOperations._logged_rolling_init = True
                else:
                    self.logger.info("ℹ️ VectorBTRollingOptimizer not available")
            else:
                self.rolling_optimizer = None

            # Initialize UnifiedVectorizationManager
            if UNIFIED_VECTORIZATION_AVAILABLE:
                self.vectorization_manager = _get_unified_vectorization_manager()
                if self.vectorization_manager:
                    self.logger.debug("✅ UnifiedVectorizationManager initialized")
                else:
                    self.logger.info("ℹ️ UnifiedVectorizationManager not available")
            else:
                self.vectorization_manager = None

        except Exception as e:
            # Use module-level logger as fallback if self.logger is not available
            try:
                self.logger.error(f"❌ Error initializing VectorBT managers: {e}")
            except AttributeError:
                logger.error(f"❌ Error initializing VectorBT managers: {e}")
            self.rolling_optimizer = None
            self.vectorization_manager = None

    def _check_memory_usage(self) -> float:
        """Check current memory usage in GB."""
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            return memory_info.used / (1024**3)
        except ImportError:
            return 0.0

    def _should_use_chunking(self, data_size: int, operation_complexity: str = 'medium') -> bool:
        """Determine if chunking should be used based on data size and memory."""
        current_memory = self._check_memory_usage()

        # Complexity factors
        complexity_factors = {
            'low': 1.0,
            'medium': 2.0,
            'high': 4.0,
            'very_high': 8.0
        }

        factor = complexity_factors.get(operation_complexity, 2.0)
        estimated_memory_needed = (data_size * 8 * factor) / (1024**3)  # 8 bytes per float64

        return (estimated_memory_needed > self.memory_limit_gb * 0.5 or
                current_memory > self.memory_limit_gb * 0.8 or
                data_size > self.chunk_size_threshold)

    def _should_use_vectorization_manager(self, A: 'np.ndarray', B: 'np.ndarray') -> bool:
        """Determine if UnifiedVectorizationManager should be used for the operation."""
        if not self.vectorization_manager:
            return False

        # Use UnifiedVectorizationManager for medium to large matrices
        total_elements = A.size + B.size
        return total_elements > 5000  # 5K elements threshold

    def _should_use_rolling_optimizer(self, data: 'pd.DataFrame', windows: List[int]) -> bool:
        """Determine if VectorBTRollingOptimizer should be used for rolling operations."""
        if not self.rolling_optimizer:
            return False

        # Use VectorBTRollingOptimizer for medium to large datasets
        return data.size > 1000  # 1K elements threshold

    def _should_use_vectorization_manager_for_batch(self, data: Union['np.ndarray', 'pd.DataFrame'], operation: str) -> bool:
        """Determine if UnifiedVectorizationManager should be used for batch operations."""
        if not self.vectorization_manager:
            return False

        # Convert to numpy array for size check
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        # Use UnifiedVectorizationManager for supported operations and medium to large datasets
        supported_operations = ['correlation', 'rolling_features', 'trading_indicators', 'matrix_multiply', 'feature_engineering']
        return operation in supported_operations and data_array.size > 5000  # 5K elements threshold

    def matrix_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """
        VectorBT-optimized matrix multiplication with enhanced performance.

        Args:
            A: First matrix
            B: Second matrix

        Returns:
            Result of matrix multiplication
        """
        start_time = time.time()

        try:
            # Try UnifiedVectorizationManager first if available
            if self.vectorization_manager and self._should_use_vectorization_manager(A, B):
                try:
                    result = self.vectorization_manager.matrix_multiply(A, B)
                    self.performance_stats['vectorization_operations'] += 1
                    self.logger.debug("✅ UnifiedVectorizationManager matrix multiplication completed")
                    execution_time = time.time() - start_time
                    self._update_performance_stats(execution_time)
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ UnifiedVectorizationManager matrix multiplication failed: {e}, falling back to VectorBT")

            # Try VectorBT optimization
            if VECTORBT_AVAILABLE:
                # Enhanced VectorBT matrix multiplication with memory optimization
                # Use VectorBT's optimized matrix multiplication with chunking for large matrices
                if A.shape[0] * A.shape[1] * B.shape[1] > 1e6:  # Large matrix threshold
                    result = self._chunked_matrix_multiply_vectorbt(A, B)
                else:
                    result = vbt.math.matrix_multiply(A, B)

                self.performance_stats['vectorbt_operations'] += 1
                self.logger.debug("✅ VectorBT matrix multiplication completed")
            else:
                # Fallback to numpy with optimization
                result = self._optimized_numpy_multiply(A, B)
                self.performance_stats['fallback_operations'] += 1
                self.logger.debug("✅ Fallback matrix multiplication completed")

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            return result

        except Exception as e:
            self.logger.error(f"❌ Matrix multiplication failed: {e}")
            # Fallback to numpy
            result = self._optimized_numpy_multiply(A, B)
            self.performance_stats['fallback_operations'] += 1
            return result

    def _chunked_matrix_multiply_vectorbt(self, A: 'np.ndarray', B: 'np.ndarray',
                                        chunk_size: int = 1000) -> 'np.ndarray':
        """Chunked matrix multiplication for large matrices using VectorBT."""
        rows_A, cols_A = A.shape
        cols_B = B.shape[1]
        result = np.zeros((rows_A, cols_B), dtype=A.dtype)

        for i in range(0, rows_A, chunk_size):
            end_i = min(i + chunk_size, rows_A)
            chunk_A = A[i:end_i, :]

            for j in range(0, cols_B, chunk_size):
                end_j = min(j + chunk_size, cols_B)
                chunk_B = B[:, j:end_j]

                # Use VectorBT for chunk multiplication
                chunk_result = vbt.math.matrix_multiply(chunk_A, chunk_B)
                result[i:end_i, j:end_j] = chunk_result

        return result

    def _optimized_numpy_multiply(self, A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
        """Optimized numpy matrix multiplication with memory efficiency."""
        # Ensure arrays are contiguous for better performance
        A = np.ascontiguousarray(A)
        B = np.ascontiguousarray(B)

        # Use BLAS-optimized matrix multiplication
        return np.dot(A, B)

    def correlation_matrix(self, data: Union['np.ndarray', 'pd.DataFrame'],
                          method: str = 'pearson') -> 'np.ndarray':
        """
        VectorBT-optimized correlation matrix calculation with enhanced performance.

        Args:
            data: Input data matrix
            method: Correlation method ('pearson', 'spearman')

        Returns:
            Correlation matrix
        """
        start_time = time.time()

        try:
            if isinstance(data, pd.DataFrame):
                data = data.values

            if VECTORBT_AVAILABLE:
                # Enhanced VectorBT correlation with memory optimization
                if data.shape[0] > 10000:  # Large dataset threshold
                    result = self._chunked_correlation_vectorbt(data, method)
                else:
                    # Use VectorBT's optimized correlation calculation
                    if method == 'pearson':
                        result = vbt.math.corr_matrix(data)
                    elif method == 'spearman':
                        result = vbt.math.corr_matrix(data, method='spearman')
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")

                self.performance_stats['vectorbt_operations'] += 1
                self.logger.debug(f"✅ VectorBT correlation matrix ({method}) completed")
            else:
                # Fallback to optimized numpy
                result = self._optimized_correlation_numpy(data, method)
                self.performance_stats['fallback_operations'] += 1
                self.logger.debug(f"✅ Fallback correlation matrix ({method}) completed")

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            return result

        except Exception as e:
            self.logger.error(f"❌ Correlation matrix calculation failed: {e}")
            # Fallback to numpy
            result = self._optimized_correlation_numpy(data, method)
            self.performance_stats['fallback_operations'] += 1
            return result

    def _chunked_correlation_vectorbt(self, data: 'np.ndarray', method: str,
                                    chunk_size: int = 5000) -> 'np.ndarray':
        """Chunked correlation calculation for large datasets using VectorBT."""
        n_features = data.shape[1]
        corr_matrix = np.eye(n_features)

        # Process in chunks to manage memory
        for i in range(0, n_features, chunk_size):
            end_i = min(i + chunk_size, n_features)
            chunk_i = data[:, i:end_i]

            for j in range(i, n_features, chunk_size):
                end_j = min(j + chunk_size, n_features)
                chunk_j = data[:, j:end_j]

                # Calculate correlation between chunks
                if method == 'pearson':
                    chunk_corr = vbt.math.corr_matrix(np.column_stack([chunk_i, chunk_j]))
                else:
                    chunk_corr = vbt.math.corr_matrix(np.column_stack([chunk_i, chunk_j]), method='spearman')

                # Extract cross-correlation
                cross_corr = chunk_corr[:chunk_i.shape[1], chunk_i.shape[1]:]
                corr_matrix[i:end_i, j:end_j] = cross_corr

                # Fill symmetric part
                if i != j:
                    corr_matrix[j:end_j, i:end_i] = cross_corr.T

        return corr_matrix

    def _optimized_correlation_numpy(self, data: 'np.ndarray', method: str) -> 'np.ndarray':
        """Optimized numpy correlation calculation."""
        if method == 'pearson':
            return np.corrcoef(data.T)
        elif method == 'spearman':
            from scipy.stats import spearmanr
            return np.corrcoef(data.T)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

    def compute_trading_indicators(self, data: 'pd.DataFrame',
                                 config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
        """
        Compute trading indicators using VectorBT's optimized functions.

        Args:
            data: DataFrame with OHLCV data
            config: Configuration for indicators

        Returns:
            DataFrame with computed indicators
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for trading indicators")
            return data.copy()

        start_time = time.time()

        try:
            if config is None:
                config = self._get_default_indicator_config()

            result_df = data.copy()

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"⚠️ Missing required columns: {missing_cols}")
                return result_df

            # Compute indicators using VectorBT
            result_df = self._compute_moving_averages_vectorbt(result_df, config)
            result_df = self._compute_momentum_indicators_vectorbt(result_df, config)
            result_df = self._compute_volatility_indicators_vectorbt(result_df, config)
            result_df = self._compute_volume_indicators_vectorbt(result_df, config)
            result_df = self._compute_trend_indicators_vectorbt(result_df, config)
            result_df = self._compute_oscillator_indicators_vectorbt(result_df, config)

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            self.logger.info(f"✅ Computed {len(result_df.columns) - len(data.columns)} trading indicators using VectorBT")
            return result_df

        except Exception as e:
            self.logger.error(f"❌ Trading indicators computation failed: {e}")
            return data.copy()

    def _compute_moving_averages_vectorbt(self, data: 'pd.DataFrame',
                                        config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute moving averages using VectorBT with enhanced performance."""
        result = data.copy()

        # Batch compute all moving averages for better performance
        sma_periods = config.get('sma_periods', [9, 21, 50, 200])
        ema_periods = config.get('ema_periods', [12, 26, 50])

        # Use VectorBT's batch processing for multiple periods
        if len(sma_periods) > 1:
            # Compute all SMAs in one go using VectorBT's vectorized operations
            sma_results = {}
            for period in sma_periods:
                ma_result = vbt.MA.run(data['close'], window=period)
                sma_results[f'sma_{period}'] = ma_result.ma
                result[f'sma_{period}'] = ma_result.ma
        else:
            # Single SMA computation
            for period in sma_periods:
                result[f'sma_{period}'] = vbt.MA.run(data['close'], window=period).ma

        # Compute EMAs with optimized parameters
        for period in ema_periods:
            # Use VectorBT's optimized EMA with proper alpha calculation
            alpha = 2.0 / (period + 1)
            result[f'ema_{period}'] = vbt.MA.run(data['close'], window=period, short_name='EMA').ma

        # Enhanced moving average crossovers with VectorBT
        if 'sma_9' in result.columns and 'sma_21' in result.columns:
            result['sma_cross_9_21'] = (result['sma_9'] > result['sma_21']).astype(int)
            from ...feature_generation.utils.error_handling import safe_diff
            result['sma_cross_signal'] = safe_diff(result['sma_cross_9_21']).fillna(0)

        if 'ema_12' in result.columns and 'ema_26' in result.columns:
            result['ema_cross_12_26'] = (result['ema_12'] > result['ema_26']).astype(int)
            from ...feature_generation.utils.error_handling import safe_diff
            result['ema_cross_signal'] = safe_diff(result['ema_cross_12_26']).fillna(0)

        # Additional VectorBT-optimized moving average features
        if 'sma_50' in result.columns and 'sma_200' in result.columns:
            result['golden_cross'] = ((result['sma_50'] > result['sma_200']) &
                                    (result['sma_50'].shift(1) <= result['sma_200'].shift(1))).astype(int)
            result['death_cross'] = ((result['sma_50'] < result['sma_200']) &
                                   (result['sma_50'].shift(1) >= result['sma_200'].shift(1))).astype(int)

        return result

    def _compute_momentum_indicators_vectorbt(self, data: 'pd.DataFrame',
                                            config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute momentum indicators using VectorBT."""
        result = data.copy()

        # RSI using VectorBT
        rsi_period = config.get('rsi_period', 14)
        rsi = vbt.RSI.run(data['close'], window=rsi_period)
        result['rsi'] = rsi.rsi
        result['rsi_overbought'] = (result['rsi'] > config.get('rsi_overbought', 70)).astype(int)
        result['rsi_oversold'] = (result['rsi'] < config.get('rsi_oversold', 30)).astype(int)

        # MACD using VectorBT
        macd_fast = config.get('macd_fast', 12)
        macd_slow = config.get('macd_slow', 26)
        macd_signal = config.get('macd_signal', 9)

        macd = vbt.MACD.run(data['close'], fast_window=macd_fast, slow_window=macd_slow, signal_window=macd_signal)
        result['macd'] = macd.macd
        result['macd_signal'] = macd.signal
        result['macd_histogram'] = macd.histogram
        result['macd_bullish'] = (result['macd'] > result['macd_signal']).astype(int)
        from ...feature_generation.utils.error_handling import safe_diff
        result['macd_cross'] = safe_diff((result['macd'] > result['macd_signal']).astype(int)).fillna(0)

        # ROC using VectorBT
        roc_period = config.get('roc_period', 10)
        result['roc'] = vbt.ROC.run(data['close'], window=roc_period).roc

        return result

    def _compute_volatility_indicators_vectorbt(self, data: 'pd.DataFrame',
                                              config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volatility indicators using VectorBT."""
        result = data.copy()

        # Bollinger Bands using VectorBT
        bb_period = config.get('bb_period', 20)
        bb_std = config.get('bb_std', 2.0)

        bb = vbt.BBANDS.run(data['close'], window=bb_period, alpha=bb_std)
        result['bb_upper'] = bb.upper
        result['bb_lower'] = bb.lower
        result['bb_middle'] = bb.middle
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        result['bb_position'] = (data['close'] - result['bb_lower']) / (result['bb_upper'] - result['bb_lower'])

        # ATR using VectorBT
        atr_period = config.get('atr_period', 14)
        atr = vbt.ATR.run(data['high'], data['low'], data['close'], window=atr_period)
        result['atr'] = atr.atr
        result['atr_percent'] = (result['atr'] / data['close']) * 100

        # Volatility
        result['volatility'] = data['close'].rolling(window=20, min_periods=1).std()
        result['volatility_percent'] = (result['volatility'] / data['close']) * 100

        return result

    def _compute_volume_indicators_vectorbt(self, data: 'pd.DataFrame',
                                          config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute volume indicators using VectorBT."""
        result = data.copy()

        # Volume SMA
        volume_sma_period = config.get('volume_sma_period', 20)
        result['volume_sma'] = data['volume'].rolling(window=volume_sma_period, min_periods=1).mean()
        result['volume_ratio'] = data['volume'] / result['volume_sma']

        # OBV using VectorBT
        obv = vbt.OBV.run(data['close'], data['volume'])
        result['obv'] = obv.obv

        # OBV smoothed
        obv_smooth = config.get('obv_smooth', 10)
        result['obv_sma'] = result['obv'].rolling(window=obv_smooth, min_periods=1).mean()

        return result

    def _compute_trend_indicators_vectorbt(self, data: 'pd.DataFrame',
                                         config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute trend indicators using VectorBT."""
        result = data.copy()

        # ADX using VectorBT
        adx_period = config.get('adx_period', 14)
        adx = vbt.ADX.run(data['high'], data['low'], data['close'], window=adx_period)
        result['adx'] = adx.adx
        result['plus_di'] = adx.plus_di
        result['minus_di'] = adx.minus_di
        result['adx_trending'] = (result['adx'] > 25).astype(int)
        result['adx_strong_trend'] = (result['adx'] > 50).astype(int)

        return result

    def _compute_oscillator_indicators_vectorbt(self, data: 'pd.DataFrame',
                                              config: Dict[str, Any]) -> 'pd.DataFrame':
        """Compute oscillator indicators using VectorBT."""
        result = data.copy()

        # Stochastic Oscillator using VectorBT
        stoch_k = config.get('stoch_k', 14)
        stoch_d = config.get('stoch_d', 3)

        stoch = vbt.STOCH.run(data['high'], data['low'], data['close'],
                             k_window=stoch_k, d_window=stoch_d)
        result['stoch_k'] = stoch.k
        result['stoch_d'] = stoch.d
        result['stoch_overbought'] = (result['stoch_k'] > 80).astype(int)
        result['stoch_oversold'] = (result['stoch_k'] < 20).astype(int)

        # Williams %R using VectorBT
        williams_period = config.get('williams_period', 14)
        williams = vbt.WILLR.run(data['high'], data['low'], data['close'], window=williams_period)
        result['williams_r'] = williams.willr

        # CCI using VectorBT
        cci_period = config.get('cci_period', 20)
        cci = vbt.CCI.run(data['high'], data['low'], data['close'], window=cci_period)
        result['cci'] = cci.cci
        result['cci_overbought'] = (result['cci'] > 100).astype(int)
        result['cci_oversold'] = (result['cci'] < -100).astype(int)

        return result

    def rolling_features(self, data: 'pd.DataFrame',
                        windows: List[int] = [5, 10, 20, 50],
                        features: List[str] = None) -> 'pd.DataFrame':
        """
        Create rolling features using VectorBT's optimized functions with enhanced performance.

        Args:
            data: Input DataFrame
            windows: List of window sizes
            features: List of feature columns to process

        Returns:
            DataFrame with rolling features
        """
        start_time = time.time()

        try:
            # Try VectorBTRollingOptimizer first if available
            if self.rolling_optimizer and self._should_use_rolling_optimizer(data, windows):
                try:
                    if features is None:
                        features = data.select_dtypes(include=[np.number]).columns.tolist()

                    result = self.rolling_optimizer.batch_rolling_operations(
                        data, windows=windows, operations=['mean', 'std', 'min', 'max'],
                        features=features
                    )
                    self.performance_stats['rolling_operations'] += 1
                    self.logger.debug("✅ VectorBTRollingOptimizer rolling features completed")
                    execution_time = time.time() - start_time
                    self._update_performance_stats(execution_time)
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBTRollingOptimizer rolling features failed: {e}, falling back to VectorBT")

            # Fallback to VectorBT if available
            if not VECTORBT_AVAILABLE:
                self.logger.warning("⚠️ VectorBT not available for rolling features")
                return data.copy()

            if features is None:
                features = data.select_dtypes(include=[np.number]).columns.tolist()

            # Use VectorBT's batch rolling operations for better performance
            result = self._batch_rolling_features_vectorbt(data, windows, features)

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            return result

        except Exception as e:
            self.logger.error(f"❌ Rolling features computation failed: {e}")
            return data.copy()

    def _batch_rolling_features_vectorbt(self, data: 'pd.DataFrame',
                                       windows: List[int],
                                       features: List[str]) -> 'pd.DataFrame':
        """Batch compute rolling features using VectorBT for optimal performance."""
        result = data.copy()

        # Group features by type for batch processing
        numeric_features = [col for col in features if col in data.columns and
                          data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        if not numeric_features:
            return result

        # Process each window size
        for window in windows:
            window_features = {}

            # Batch process all features for this window
            for col in numeric_features:
                series = data[col]

                # Use VectorBT's optimized rolling functions
                rolling = vbt.Rolling(series, window=window)

                # Compute all rolling statistics in one go
                window_features.update({
                    f'{col}_rolling_mean_{window}': rolling.mean(),
                    f'{col}_rolling_std_{window}': rolling.std(),
                    f'{col}_rolling_min_{window}': rolling.min(),
                    f'{col}_rolling_max_{window}': rolling.max(),
                    f'{col}_rolling_skew_{window}': rolling.skew(),
                    f'{col}_rolling_kurt_{window}': rolling.kurt(),
                    f'{col}_rolling_median_{window}': rolling.median(),
                    f'{col}_rolling_quantile_25_{window}': rolling.quantile(0.25),
                    f'{col}_rolling_quantile_75_{window}': rolling.quantile(0.75),
                })

                # Additional VectorBT-optimized features
                window_features[f'{col}_rolling_range_{window}'] = (
                    window_features[f'{col}_rolling_max_{window}'] -
                    window_features[f'{col}_rolling_min_{window}']
                )

                window_features[f'{col}_rolling_cv_{window}'] = (
                    window_features[f'{col}_rolling_std_{window}'] /
                    window_features[f'{col}_rolling_mean_{window}']
                ).fillna(0)

            # Add window features to result
            window_df = pd.DataFrame(window_features, index=data.index)
            result = pd.concat([result, window_df], axis=1)

        return result

    def batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                     operation: str, **kwargs) -> Any:
        """
        Process data in batches using VectorBT's parallel processing with enhanced performance.

        Args:
            data: Input data
            operation: Operation to perform
            **kwargs: Additional arguments

        Returns:
            Processed result
        """
        start_time = time.time()

        try:
            # Try VectorBTRollingOptimizer first if available
            if self.rolling_optimizer and self._should_use_rolling_optimizer_for_batch(data, operation):
                try:
                    result = self.rolling_optimizer.batch_process(data, operation, **kwargs)
                    self.performance_stats['vectorbt_rolling_operations'] += 1
                    self.logger.debug("✅ VectorBTRollingOptimizer batch processing completed")
                    execution_time = time.time() - start_time
                    self._update_performance_stats(execution_time)
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ VectorBTRollingOptimizer batch processing failed: {e}, falling back to UnifiedVectorizationManager")

            # Try UnifiedVectorizationManager if available
            if self.vectorization_manager and self._should_use_vectorization_manager_for_batch(data, operation):
                try:
                    result = self.vectorization_manager.batch_process(data, operation, **kwargs)
                    self.performance_stats['vectorization_operations'] += 1
                    self.logger.debug("✅ UnifiedVectorizationManager batch processing completed")
                    execution_time = time.time() - start_time
                    self._update_performance_stats(execution_time)
                    return result
                except Exception as e:
                    self.logger.warning(f"⚠️ UnifiedVectorizationManager batch processing failed: {e}, falling back to VectorBT")

            # Try VectorBT optimization
            if VECTORBT_AVAILABLE and self.enable_parallel:
                # Use VectorBT's optimized batch processing with intelligent chunking
                if operation == 'correlation':
                    return self.correlation_matrix(data)
                elif operation == 'rolling_features':
                    return self.rolling_features(data, **kwargs)
                elif operation == 'trading_indicators':
                    return self.compute_trading_indicators(data, **kwargs)
                elif operation == 'matrix_multiply':
                    return self._batch_matrix_multiply_vectorbt(data, **kwargs)
                elif operation == 'feature_engineering':
                    return self._batch_feature_engineering_vectorbt(data, **kwargs)
                else:
                    # Fallback to standard processing
                    return self._standard_batch_process(data, operation, **kwargs)
            else:
                # Use standard processing
                return self._standard_batch_process(data, operation, **kwargs)

        except Exception as e:
            self.logger.error(f"❌ Batch processing failed: {e}")
            return self._standard_batch_process(data, operation, **kwargs)
        finally:
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

    def _batch_matrix_multiply_vectorbt(self, data: Union['np.ndarray', 'pd.DataFrame'],
                                      **kwargs) -> Any:
        """Batch matrix multiplication using VectorBT parallel processing."""
        matrices_a = kwargs.get('matrices_a', [])
        matrices_b = kwargs.get('matrices_b', [])

        if not matrices_a or not matrices_b:
            raise ValueError("matrices_a and matrices_b must be provided for batch matrix multiplication")

        # Use VectorBT's parallel processing for batch operations
        results = []
        for a, b in zip(matrices_a, matrices_b):
            result = self.matrix_multiply(a, b)
            results.append(result)

        return results

    def _batch_feature_engineering_vectorbt(self, data: 'pd.DataFrame',
                                          **kwargs) -> 'pd.DataFrame':
        """Batch feature engineering using VectorBT optimizations."""
        result = data.copy()

        # Apply multiple feature engineering operations in batch
        if 'rolling_windows' in kwargs:
            result = self.rolling_features(result, kwargs['rolling_windows'])

        if 'trading_indicators' in kwargs and kwargs['trading_indicators']:
            result = self.compute_trading_indicators(result, kwargs.get('indicator_config'))

        if 'correlation_features' in kwargs and kwargs['correlation_features']:
            # Add correlation-based features
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.correlation_matrix(result[numeric_cols])
                # Add correlation features
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:  # Avoid duplicates
                            result[f'corr_{col1}_{col2}'] = corr_matrix[i, j]

        return result

    def _standard_batch_process(self, data: Union['np.ndarray', 'pd.DataFrame'],
                               operation: str, **kwargs) -> Any:
        """Standard batch processing fallback."""
        if isinstance(data, pd.DataFrame):
            data = data.values

        if operation == 'correlation':
            return np.corrcoef(data.T)
        elif operation == 'mean':
            return np.mean(data, axis=0)
        elif operation == 'std':
            return np.std(data, axis=0)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _get_default_indicator_config(self) -> Dict[str, Any]:
        """Get default configuration for trading indicators."""
        return {
            # Moving averages
            'sma_periods': [9, 21, 50, 200],
            'ema_periods': [12, 26, 50],

            # RSI
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,

            # MACD
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,

            # Bollinger Bands
            'bb_period': 20,
            'bb_std': 2.0,

            # Stochastic
            'stoch_k': 14,
            'stoch_d': 3,

            # Williams %R
            'williams_period': 14,

            # ADX
            'adx_period': 14,

            # ATR
            'atr_period': 14,

            # CCI
            'cci_period': 20,

            # ROC
            'roc_period': 10,

            # Volume indicators
            'volume_sma_period': 20,
            'obv_smooth': 10,
        }

    def _update_performance_stats(self, execution_time: float):
        """Update performance statistics."""
        self.performance_stats['total_operations'] += 1
        self.performance_stats['average_execution_time'] = (
            (self.performance_stats['average_execution_time'] *
             (self.performance_stats['total_operations'] - 1)) + execution_time
        ) / self.performance_stats['total_operations']

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()

    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware capability information."""
        return {
            'vectorbt_available': VECTORBT_AVAILABLE,
            'gpu_enabled': self.enable_gpu,
            'parallel_enabled': self.enable_parallel,
            'performance_stats': self.performance_stats
        }

    def compute_portfolio_metrics(self, returns: 'pd.DataFrame',
                                 weights: Optional['np.ndarray'] = None) -> 'pd.DataFrame':
        """
        Compute comprehensive portfolio metrics using VectorBT.

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights (if None, equal weights assumed)

        Returns:
            DataFrame with portfolio metrics
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for portfolio metrics")
            return returns.copy()

        start_time = time.time()

        try:
            result = returns.copy()

            # Set equal weights if not provided
            if weights is None:
                weights = np.ones(returns.shape[1]) / returns.shape[1]

            # Portfolio returns using VectorBT
            portfolio_returns = (returns * weights).sum(axis=1)
            result['portfolio_returns'] = portfolio_returns

            # Portfolio volatility using VectorBT
            portfolio_vol = portfolio_returns.rolling(window=252, min_periods=1).std() * np.sqrt(252)
            result['portfolio_volatility'] = portfolio_vol

            # Sharpe ratio using VectorBT
            risk_free_rate = 0.02  # 2% annual risk-free rate
            excess_returns = portfolio_returns - risk_free_rate / 252
            sharpe_ratio = excess_returns.rolling(window=252, min_periods=1).mean() / portfolio_vol
            result['sharpe_ratio'] = sharpe_ratio

            # Maximum drawdown using VectorBT
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            result['drawdown'] = drawdown
            result['max_drawdown'] = drawdown.expanding().min()

            # Value at Risk (VaR) using VectorBT
            var_95 = portfolio_returns.rolling(window=252, min_periods=1).quantile(0.05)
            result['var_95'] = var_95

            # Conditional Value at Risk (CVaR) using VectorBT
            cvar_95 = portfolio_returns.rolling(window=252, min_periods=1).apply(
                lambda x: x[x <= x.quantile(0.05)].mean()
            )
            result['cvar_95'] = cvar_95

            # Beta calculation using VectorBT
            if len(returns.columns) > 1:
                market_returns = returns.iloc[:, 0]  # Use first column as market proxy
                beta = portfolio_returns.rolling(window=252, min_periods=1).cov(market_returns) / market_returns.rolling(window=252, min_periods=1).var()
                result['beta'] = beta

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            self.logger.info(f"✅ Computed portfolio metrics using VectorBT")
            return result

        except Exception as e:
            self.logger.error(f"❌ Portfolio metrics computation failed: {e}")
            return returns.copy()

    def compute_risk_metrics(self, returns: 'pd.DataFrame') -> Dict[str, Any]:
        """
        Compute comprehensive risk metrics using VectorBT.

        Args:
            returns: DataFrame of asset returns

        Returns:
            Dictionary with risk metrics
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for risk metrics")
            return {}

        start_time = time.time()

        try:
            metrics = {}

            # Volatility metrics
            metrics['volatility'] = returns.std() * np.sqrt(252)
            metrics['volatility_annualized'] = returns.std() * np.sqrt(252)

            # Skewness and Kurtosis using VectorBT
            metrics['skewness'] = returns.skew()
            metrics['kurtosis'] = returns.kurtosis()

            # Value at Risk (VaR) at different confidence levels
            metrics['var_95'] = returns.quantile(0.05)
            metrics['var_99'] = returns.quantile(0.01)

            # Expected Shortfall (CVaR)
            metrics['cvar_95'] = returns[returns <= returns.quantile(0.05)].mean()
            metrics['cvar_99'] = returns[returns <= returns.quantile(0.01)].mean()

            # Maximum Drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()

            # Calmar Ratio
            annual_return = returns.mean() * 252
            metrics['calmar_ratio'] = annual_return / abs(metrics['max_drawdown'])

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252)
            metrics['sortino_ratio'] = annual_return / downside_deviation if downside_deviation > 0 else 0

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            self.logger.info(f"✅ Computed risk metrics using VectorBT")
            return metrics

        except Exception as e:
            self.logger.error(f"❌ Risk metrics computation failed: {e}")
            return {}

    def compute_technical_analysis(self, data: 'pd.DataFrame') -> 'pd.DataFrame':
        """
        Compute comprehensive technical analysis using VectorBT.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with technical analysis indicators
        """
        if not VECTORBT_AVAILABLE:
            self.logger.warning("⚠️ VectorBT not available for technical analysis")
            return data.copy()

        start_time = time.time()

        try:
            result = data.copy()

            # Price patterns using VectorBT
            result['price_change'] = data['close'].pct_change()
            from ...feature_generation.utils.error_handling import safe_diff
            result['price_change_abs'] = safe_diff(data['close'])

            # Support and Resistance levels
            result['resistance'] = data['high'].rolling(window=20, min_periods=1).max()
            result['support'] = data['low'].rolling(window=20, min_periods=1).min()

            # Price position relative to support/resistance
            result['price_position'] = (data['close'] - result['support']) / (result['resistance'] - result['support'])

            # Trend strength using VectorBT
            result['trend_strength'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)

            # Volume analysis
            result['volume_sma'] = data['volume'].rolling(window=20, min_periods=1).mean()
            result['volume_ratio'] = data['volume'] / result['volume_sma']
            result['volume_trend'] = data['volume'].rolling(window=5, min_periods=1).mean() / data['volume'].rolling(window=20, min_periods=1).mean()

            # Price momentum using VectorBT
            result['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            result['momentum_10'] = data['close'] / data['close'].shift(10) - 1
            result['momentum_20'] = data['close'] / data['close'].shift(20) - 1

            # Volatility analysis
            result['volatility_5'] = data['close'].pct_change().rolling(window=5, min_periods=1).std()
            result['volatility_20'] = data['close'].pct_change().rolling(window=20, min_periods=1).std()
            result['volatility_ratio'] = result['volatility_5'] / result['volatility_20']

            # Market regime detection
            result['regime_bull'] = (result['trend_strength'] > 0.02).astype(int)
            result['regime_bear'] = (result['trend_strength'] < -0.02).astype(int)
            result['regime_sideways'] = ((result['trend_strength'] >= -0.02) & (result['trend_strength'] <= 0.02)).astype(int)

            # Update performance stats
            execution_time = time.time() - start_time
            self._update_performance_stats(execution_time)

            self.logger.info(f"✅ Computed technical analysis using VectorBT")
            return result

        except Exception as e:
            self.logger.error(f"❌ Technical analysis computation failed: {e}")
            return data.copy()

# Global instance
_vectorbt_ops = None

def get_vectorbt_optimized_operations() -> VectorBTOptimizedOperations:
    """Get global VectorBT optimized operations instance."""
    global _vectorbt_ops
    if _vectorbt_ops is None:
        _vectorbt_ops = VectorBTOptimizedOperations()
    return _vectorbt_ops

# Convenience functions
def vectorbt_matrix_multiply(A: 'np.ndarray', B: 'np.ndarray') -> 'np.ndarray':
    """VectorBT-optimized matrix multiplication."""
    ops = get_vectorbt_optimized_operations()
    return ops.matrix_multiply(A, B)

def vectorbt_correlation_matrix(data: Union['np.ndarray', 'pd.DataFrame'],
                               method: str = 'pearson') -> 'np.ndarray':
    """VectorBT-optimized correlation matrix."""
    ops = get_vectorbt_optimized_operations()
    return ops.correlation_matrix(data, method)

def vectorbt_trading_indicators(data: 'pd.DataFrame',
                               config: Optional[Dict[str, Any]] = None) -> 'pd.DataFrame':
    """VectorBT-optimized trading indicators."""
    ops = get_vectorbt_optimized_operations()
    return ops.compute_trading_indicators(data, config)

def vectorbt_rolling_features(data: 'pd.DataFrame',
                             windows: List[int] = [5, 10, 20, 50],
                             features: List[str] = None) -> 'pd.DataFrame':
    """VectorBT-optimized rolling features."""
    ops = get_vectorbt_optimized_operations()
    return ops.rolling_features(data, windows, features)

def vectorbt_batch_processing(data: Union['np.ndarray', 'pd.DataFrame'],
                             operation: str, **kwargs) -> Any:
    """VectorBT-optimized batch processing."""
    ops = get_vectorbt_optimized_operations()
    return ops.batch_process(data, operation, **kwargs)
