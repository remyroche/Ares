"""
VectorBT-Optimized Feature Generator

This module provides a high-performance feature generator base class that leverages
VectorBT's optimized C++ backend for maximum performance in feature generation.

Key Features:
- VectorBT indicators integration
- Optimized rolling operations
- Memory-efficient processing
- GPU acceleration support
- Batch processing capabilities
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod
import warnings

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.indicators import RSI, MACD, ATR, BBANDS, STOCH, WILLR, CCI, MFI, ADX, CMO, ROC, MOM, TRIX, ULTOSC, KAMA, TEMA, WMA, DEMA, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE, AROON, BOP, AD, OBV, ADOSC, AROONOSC, DX, MINUS_DI, MINUS_DM, PLUS_DI, PLUS_DM, PPO, TYPPRICE, WCLPRICE, WAPRICE, MEDPRICE, TRANGE, AVGPRICE
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.portfolio import Portfolio
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    # Define dummy classes for type hints
    class RSI: pass
    class MACD: pass
    class ATR: pass
    class BBANDS: pass
    class STOCH: pass
    class WILLR: pass
    class CCI: pass
    class MFI: pass
    class ADX: pass
    class CMO: pass
    class ROC: pass
    class MOM: pass
    class TRIX: pass
    class ULTOSC: pass
    class KAMA: pass
    class TEMA: pass
    class WMA: pass
    class DEMA: pass
    class HT_DCPERIOD: pass
    class HT_DCPHASE: pass
    class HT_PHASOR: pass
    class HT_SINE: pass
    class HT_TRENDMODE: pass
    class AROON: pass
    class BOP: pass
    class AD: pass
    class OBV: pass
    class ADOSC: pass
    class AROONOSC: pass
    class DX: pass
    class MINUS_DI: pass
    class MINUS_DM: pass
    class PLUS_DI: pass
    class PLUS_DM: pass
    class PPO: pass
    class TYPPRICE: pass
    class WCLPRICE: pass
    class WAPRICE: pass
    class MEDPRICE: pass
    class TRANGE: pass
    class AVGPRICE: pass

from .feature_generator import FeatureGenerator, FeatureConfig, FeatureResult, FeatureCategory
from ..utils.math_validation import safe_divide, validate_finite, safe_percentage_change

logger = logging.getLogger(__name__)

class VectorBTFeatureGenerator(FeatureGenerator):
    """
    High-performance feature generator using VectorBT's optimized backend.
    
    This class provides a foundation for creating feature generators that leverage
    VectorBT's C++ optimized implementations for maximum performance.
    """
    
    def __init__(self, config: FeatureConfig, enable_gpu: bool = False, enable_parallel: bool = True):
        """
        Initialize VectorBT feature generator.
        
        Args:
            config: Feature configuration
            enable_gpu: Whether to enable GPU acceleration
            enable_parallel: Whether to enable parallel processing
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required but not available. Install with: pip install vectorbt")
        
        super().__init__(config)
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        
        # Configure VectorBT settings
        self._configure_vectorbt()
        
        # Performance tracking
        self.vectorbt_stats = {
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0
        }
    
    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        if not VECTORBT_AVAILABLE:
            return
        
        # Configure VectorBT settings for optimal feature generation
        vbt.settings.setting('array_wrapper', 'pandas')
        vbt.settings.setting('caching', True)
        vbt.settings.setting('caching_dir', 'data_cache/vectorbt_cache')
        
        # Optimize for feature generation workloads
        vbt.settings.setting('array_wrapper', 'freq_precision', 0)
        vbt.settings.setting('array_wrapper', 'freq_rep', 'auto')
        vbt.settings.setting('array_wrapper', 'chunk_size', 50000)
        vbt.settings.setting('array_wrapper', 'memory_limit', 2 * 1024**3)  # 2GB limit
        
        if self.enable_gpu:
            try:
                vbt.settings.setting('use_gpu', True)
                vbt.settings.setting('gpu_memory_fraction', 0.7)  # Use 70% of GPU memory
                logger.info("✅ VectorBT GPU acceleration enabled")
            except Exception as e:
                logger.warning(f"⚠️ GPU acceleration not available: {e}")
                self.enable_gpu = False
        
        if self.enable_parallel:
            try:
                vbt.settings.setting('use_parallel', True)
                vbt.settings.setting('parallel', 'n_jobs', -1)  # Use all cores
                vbt.settings.setting('parallel', 'threading', True)
                vbt.settings.setting('parallel', 'multiprocessing', True)
                logger.info("✅ VectorBT parallel processing enabled")
            except Exception as e:
                logger.warning(f"⚠️ Parallel processing not available: {e}")
                self.enable_parallel = False
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """
        Perform VectorBT rolling operation.
        
        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters
            
        Returns:
            Result of rolling operation
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
        self.vectorbt_stats['vectorbt_operations'] += 1
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for correlation")
                return rolling_corr(data, other, window=window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for covariance")
                return rolling_cov(data, other, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        
        except Exception as e:
            logger.warning(f"VectorBT rolling operation failed: {e}, using fallback")
            return self._fallback_rolling_operation(data, operation, window, **kwargs)
    
    def _fallback_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        elif operation == 'corr':
            other = kwargs.get('other')
            return data.rolling(window=window).corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            return data.rolling(window=window).cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_scale(self, data: pd.Series, method: str = 'zscore', **kwargs) -> pd.Series:
        """
        Scale data using VectorBT scaling functions.
        
        Args:
            data: Input data series
            method: Scaling method ('zscore', 'minmax', 'robust', 'quantile', 'winsorize')
            **kwargs: Additional parameters
            
        Returns:
            Scaled data series
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
        self.vectorbt_stats['vectorbt_operations'] += 1
        
        try:
            if method == 'zscore':
                return zscore(data, **kwargs)
            elif method == 'minmax':
                return scale(data, method='minmax', **kwargs)
            elif method == 'robust':
                return scale(data, method='robust', **kwargs)
            elif method == 'quantile':
                return quantile(data, **kwargs)
            elif method == 'winsorize':
                return winsorize(data, **kwargs)
            elif method == 'rank':
                return rank(data, **kwargs)
            elif method == 'clip':
                return clip(data, **kwargs)
            else:
                raise ValueError(f"Unsupported scaling method: {method}")
        
        except Exception as e:
            logger.warning(f"VectorBT scaling failed: {e}, using fallback")
            return self._fallback_scale(data, method, **kwargs)
    
    def _fallback_scale(self, data: pd.Series, method: str, **kwargs) -> pd.Series:
        """Fallback scaling using pandas/numpy."""
        if method == 'zscore':
            return (data - data.mean()) / data.std()
        elif method == 'minmax':
            return (data - data.min()) / (data.max() - data.min())
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            return (data - median) / mad
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    def _vectorbt_technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """
        Calculate technical indicator using VectorBT.
        
        Args:
            data: OHLCV data
            indicator: Indicator name
            **kwargs: Indicator parameters
            
        Returns:
            Indicator values
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
        self.vectorbt_stats['vectorbt_operations'] += 1
        
        try:
            if indicator == 'rsi':
                return RSI.run(data['close'], **kwargs).rsi
            elif indicator == 'macd':
                macd_result = MACD.run(data['close'], **kwargs)
                return macd_result.macd
            elif indicator == 'macd_signal':
                macd_result = MACD.run(data['close'], **kwargs)
                return macd_result.signal
            elif indicator == 'macd_histogram':
                macd_result = MACD.run(data['close'], **kwargs)
                return macd_result.histogram
            elif indicator == 'atr':
                return ATR.run(data['high'], data['low'], data['close'], **kwargs).atr
            elif indicator == 'bbands_upper':
                bb_result = BBANDS.run(data['close'], **kwargs)
                return bb_result.upper
            elif indicator == 'bbands_middle':
                bb_result = BBANDS.run(data['close'], **kwargs)
                return bb_result.middle
            elif indicator == 'bbands_lower':
                bb_result = BBANDS.run(data['close'], **kwargs)
                return bb_result.lower
            elif indicator == 'bbands_width':
                bb_result = BBANDS.run(data['close'], **kwargs)
                return bb_result.width
            elif indicator == 'bbands_percent':
                bb_result = BBANDS.run(data['close'], **kwargs)
                return bb_result.percent
            elif indicator == 'stoch_k':
                stoch_result = STOCH.run(data['high'], data['low'], data['close'], **kwargs)
                return stoch_result.stoch_k
            elif indicator == 'stoch_d':
                stoch_result = STOCH.run(data['high'], data['low'], data['close'], **kwargs)
                return stoch_result.stoch_d
            elif indicator == 'willr':
                return WILLR.run(data['high'], data['low'], data['close'], **kwargs).willr
            elif indicator == 'cci':
                return CCI.run(data['high'], data['low'], data['close'], **kwargs).cci
            elif indicator == 'mfi':
                return MFI.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).mfi
            elif indicator == 'adx':
                return ADX.run(data['high'], data['low'], data['close'], **kwargs).adx
            elif indicator == 'cmo':
                return CMO.run(data['close'], **kwargs).cmo
            elif indicator == 'roc':
                return ROC.run(data['close'], **kwargs).roc
            elif indicator == 'mom':
                return MOM.run(data['close'], **kwargs).mom
            elif indicator == 'trix':
                return TRIX.run(data['close'], **kwargs).trix
            elif indicator == 'ultosc':
                return ULTOSC.run(data['high'], data['low'], data['close'], **kwargs).ultosc
            elif indicator == 'kama':
                return KAMA.run(data['close'], **kwargs).kama
            elif indicator == 'tema':
                return TEMA.run(data['close'], **kwargs).tema
            elif indicator == 'wma':
                return WMA.run(data['close'], **kwargs).wma
            elif indicator == 'dema':
                return DEMA.run(data['close'], **kwargs).dema
            elif indicator == 'aroon_up':
                aroon_result = AROON.run(data['high'], data['low'], **kwargs)
                return aroon_result.aroon_up
            elif indicator == 'aroon_down':
                aroon_result = AROON.run(data['high'], data['low'], **kwargs)
                return aroon_result.aroon_down
            elif indicator == 'aroon_oscillator':
                aroon_result = AROON.run(data['high'], data['low'], **kwargs)
                return aroon_result.aroon_oscillator
            elif indicator == 'bop':
                return BOP.run(data['open'], data['high'], data['low'], data['close'], **kwargs).bop
            elif indicator == 'ad':
                return AD.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).ad
            elif indicator == 'obv':
                return OBV.run(data['close'], data['volume'], **kwargs).obv
            elif indicator == 'adosc':
                return ADOSC.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).adosc
            elif indicator == 'dx':
                return DX.run(data['high'], data['low'], data['close'], **kwargs).dx
            elif indicator == 'minus_di':
                return MINUS_DI.run(data['high'], data['low'], data['close'], **kwargs).minus_di
            elif indicator == 'minus_dm':
                return MINUS_DM.run(data['high'], data['low'], **kwargs).minus_dm
            elif indicator == 'plus_di':
                return PLUS_DI.run(data['high'], data['low'], data['close'], **kwargs).plus_di
            elif indicator == 'plus_dm':
                return PLUS_DM.run(data['high'], data['low'], **kwargs).plus_dm
            elif indicator == 'ppo':
                return PPO.run(data['close'], **kwargs).ppo
            elif indicator == 'typprice':
                return TYPPRICE.run(data['high'], data['low'], data['close'], **kwargs).typprice
            elif indicator == 'wclprice':
                return WCLPRICE.run(data['high'], data['low'], data['close'], **kwargs).wclprice
            elif indicator == 'waprice':
                return WAPRICE.run(data['high'], data['low'], data['close'], data['volume'], **kwargs).waprice
            elif indicator == 'medprice':
                return MEDPRICE.run(data['high'], data['low'], **kwargs).medprice
            elif indicator == 'trange':
                return TRANGE.run(data['high'], data['low'], data['close'], **kwargs).trange
            elif indicator == 'avgprice':
                return AVGPRICE.run(data['open'], data['high'], data['low'], data['close'], **kwargs).avgprice
            else:
                raise ValueError(f"Unsupported indicator: {indicator}")
        
        except Exception as e:
            logger.warning(f"VectorBT indicator {indicator} failed: {e}, using fallback")
            return self._fallback_technical_indicator(data, indicator, **kwargs)
    
    def _fallback_technical_indicator(self, data: pd.DataFrame, indicator: str, **kwargs) -> pd.Series:
        """Fallback technical indicator calculation using pandas/numpy."""
        # This would contain fallback implementations
        # For now, return NaN series
        return pd.Series(np.nan, index=data.index)
    
    def _vectorbt_batch_operations(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Perform batch VectorBT operations for efficiency with parallel processing.
        
        Args:
            data: Input data
            operations: List of operation dictionaries
            
        Returns:
            DataFrame with results
        """
        if not VECTORBT_AVAILABLE:
            raise ImportError("VectorBT is required for this operation")
        
        self.vectorbt_stats['vectorbt_operations'] += len(operations)
        self.vectorbt_stats['parallel_operations'] += 1
        
        results = {}
        
        try:
            # Group operations by type for better parallel processing
            rolling_ops = [op for op in operations if op.get('type') == 'rolling']
            indicator_ops = [op for op in operations if op.get('type') == 'indicator']
            scale_ops = [op for op in operations if op.get('type') == 'scale']
            
            # Process rolling operations in parallel if enabled
            if self.enable_parallel and len(rolling_ops) > 1:
                results.update(self._process_rolling_operations_parallel(data, rolling_ops))
            else:
                for op in rolling_ops:
                    op_params = op.get('params', {})
                    op_name = op.get('name', f"rolling_{len(results)}")
                    operation = op_params.get('operation')
                    window = op_params.get('window')
                    column = op_params.get('column', 'close')
                    results[op_name] = self._vectorbt_rolling_operation(
                        data[column], operation, window, **op_params
                    )
            
            # Process indicator operations
            for op in indicator_ops:
                op_params = op.get('params', {})
                op_name = op.get('name', f"indicator_{len(results)}")
                indicator = op_params.get('indicator')
                results[op_name] = self._vectorbt_technical_indicator(
                    data, indicator, **op_params
                )
            
            # Process scaling operations
            for op in scale_ops:
                op_params = op.get('params', {})
                op_name = op.get('name', f"scale_{len(results)}")
                method = op_params.get('method', 'zscore')
                column = op_params.get('column', 'close')
                results[op_name] = self._vectorbt_scale(
                    data[column], method, **op_params
                )
        
        except Exception as e:
            logger.warning(f"VectorBT batch operations failed: {e}")
            # Return empty DataFrame on failure
            return pd.DataFrame(index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _process_rolling_operations_parallel(self, data: pd.DataFrame, operations: List[Dict[str, Any]]) -> Dict[str, pd.Series]:
        """Process rolling operations in parallel for better performance."""
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        results = {}
        results_lock = threading.Lock()
        
        def process_single_rolling_op(op):
            try:
                op_params = op.get('params', {})
                op_name = op.get('name', f"rolling_{len(results)}")
                operation = op_params.get('operation')
                window = op_params.get('window')
                column = op_params.get('column', 'close')
                
                result = self._vectorbt_rolling_operation(
                    data[column], operation, window, **op_params
                )
                
                with results_lock:
                    results[op_name] = result
                    
            except Exception as e:
                logger.warning(f"Parallel rolling operation failed: {e}")
        
        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(len(operations), 4)) as executor:
            futures = [executor.submit(process_single_rolling_op, op) for op in operations]
            
            # Wait for all operations to complete
            for future in futures:
                future.result()
        
        return results
    
    def get_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        return self.vectorbt_stats.copy()
    
    def reset_vectorbt_stats(self):
        """Reset VectorBT performance statistics."""
        self.vectorbt_stats = {
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'memory_optimizations': 0
        }


class VectorBTVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility feature generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized volatility measure over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility feature using VectorBT ATR."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_{self.period}')
        
        # Use VectorBT ATR for volatility calculation
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        return atr.rename(f'vectorbt_volatility_{self.period}')


class VectorBTMomentumGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized momentum feature generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rsi_{period}",
            category=FeatureCategory.MOMENTUM,
            description=f"VectorBT-optimized RSI over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate RSI feature using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rsi_{self.period}')
        
        # Use VectorBT RSI
        rsi = self._vectorbt_technical_indicator(data, 'rsi', window=self.period)
        return rsi.rename(f'vectorbt_rsi_{self.period}')


class VectorBTTrendGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized trend feature generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_sma_{period}",
            category=FeatureCategory.TREND,
            description=f"VectorBT-optimized SMA over {period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate SMA feature using VectorBT rolling mean."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_sma_{self.period}')
        
        # Use VectorBT rolling mean for SMA
        sma = self._vectorbt_rolling_operation(data['close'], 'mean', window=self.period)
        return sma.rename(f'vectorbt_sma_{self.period}')


def create_vectorbt_generators() -> List[VectorBTFeatureGenerator]:
    """Create a comprehensive set of VectorBT feature generators."""
    generators = []
    
    # Volatility generators
    for period in [10, 20, 50]:
        generators.append(VectorBTVolatilityGenerator(period))
    
    # Momentum generators
    for period in [14, 21, 30]:
        generators.append(VectorBTMomentumGenerator(period))
    
    # Trend generators
    for period in [10, 20, 50, 100]:
        generators.append(VectorBTTrendGenerator(period))
    
    return generators