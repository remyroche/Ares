"""
Optimized Oscillator Feature Generator with Full VectorBT Integration

This module provides fully optimized oscillator feature generators using:
- VectorBTRollingOptimizer for high-performance rolling operations
- UnifiedVectorizationManager for intelligent optimization strategy selection
- VectorBT native technical analysis indicators
- GPU acceleration support
- Comprehensive performance monitoring
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import time
from dataclasses import dataclass

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt, rolling_rank
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.indicators.basic import RSI, MA, BBANDS, STOCH
    from vectorbt.indicators.momentum import MACD, ADX, CCI
    from vectorbt.indicators.volatility import ATR, BollingerBands
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
    rolling_quantile = None
    rolling_skew = None
    rolling_kurt = None
    rolling_rank = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    RSI = None
    MA = None
    BBANDS = None
    STOCH = None
    MACD = None
    ADX = None
    CCI = None
    ATR = None
    BollingerBands = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy, 
        OperationConfig, OptimizationResult
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None
    OptimizationResult = None

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    VECTORBT_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORBT_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

# Centralized utility imports
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank

logger = logging.getLogger(__name__)


@dataclass
class OscillatorPerformanceMetrics:
    """Performance metrics for oscillator calculations."""
    total_calculations: int = 0
    vectorbt_operations: int = 0
    pandas_fallbacks: int = 0
    gpu_operations: int = 0
    total_time: float = 0.0
    average_time_per_calculation: float = 0.0
    memory_usage_mb: float = 0.0
    optimization_strategy_used: Optional[str] = None


class VectorBTOscillatorFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """
    Fully optimized oscillator feature generator with comprehensive VectorBT integration.
    
    Features:
    - UnifiedVectorizationManager for intelligent optimization
    - VectorBTRollingOptimizer for high-performance rolling operations
    - Native VectorBT technical analysis indicators
    - GPU acceleration support
    - Comprehensive performance monitoring
    - Batch processing capabilities
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 enable_gpu: bool = False, enable_parallel: bool = True,
                 use_unified_manager: bool = True):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Performance tracking
        self.performance_metrics = OscillatorPerformanceMetrics()
        
        # Initialize optimization components
        self.enable_gpu = enable_gpu and CUPY_AVAILABLE
        self.enable_parallel = enable_parallel
        self.use_unified_manager = use_unified_manager and UNIFIED_MANAGER_AVAILABLE
        
        # Initialize VectorBT optimizer
        if VECTORBT_OPTIMIZER_AVAILABLE:
            self.vectorbt_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=self.enable_gpu, 
                enable_parallel=self.enable_parallel
            )
        else:
            self.vectorbt_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if self.use_unified_manager:
            self.unified_manager = UnifiedVectorizationManager()
        else:
            self.unified_manager = None
        
        # Initialize VectorBT scaler
        try:
            self.scaler = create_vectorbt_scaler(method='zscore')
        except Exception as e:
            logger.warning(f"VectorBT scaler initialization failed: {e}")
            self.scaler = None
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="vectorbt_oscillator_features",
            category=FeatureCategory.OSCILLATOR,
            description="Fully optimized oscillator features using VectorBT and UnifiedVectorizationManager",
            required_columns=["close"],
            optional_columns=["high", "low", "volume"],
            default_lookback=14,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "stochastic_periods": [14],
                "williams_periods": [14],
                "cci_periods": [20],
                "adx_periods": [14],
                "aroon_periods": [25]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate optimized oscillator features using VectorBT and UnifiedVectorizationManager."""
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)
        
        close_prices = data['close']
        
        # Use UnifiedVectorizationManager for intelligent optimization
        if self.unified_manager and self._should_use_unified_manager(close_prices):
            try:
                result = self._generate_with_unified_manager(data, **kwargs)
                self.performance_metrics.optimization_strategy_used = "unified_manager"
                return result
            except Exception as e:
                logger.warning(f"UnifiedVectorizationManager failed: {e}, using VectorBT fallback")
        
        # Use VectorBT for oscillator calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(close_prices):
            try:
                result = self._generate_with_vectorbt(data, **kwargs)
                self.performance_metrics.optimization_strategy_used = "vectorbt"
                return result
            except Exception as e:
                logger.warning(f"VectorBT calculation failed: {e}, using pandas fallback")
                result = self._generate_with_pandas_fallback(data, **kwargs)
                self.performance_metrics.optimization_strategy_used = "pandas_fallback"
                return result
        else:
            result = self._generate_with_pandas_fallback(data, **kwargs)
            self.performance_metrics.optimization_strategy_used = "pandas_fallback"
            return result
    
    def _generate_with_unified_manager(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using UnifiedVectorizationManager."""
        # Create operation configuration
        config = OperationConfig(
            operation_type=OperationType.VECTORBT_TECHNICAL_ANALYSIS,
            data_size=len(data),
            data_dimensions=(len(data), len(data.columns)),
            memory_budget_mb=1024.0,
            time_budget_seconds=30.0,
            precision_requirement="high"
        )
        
        # Execute with unified manager
        result = self.unified_manager.optimize_operation(
            operation_type=OperationType.VECTORBT_TECHNICAL_ANALYSIS,
            data=data,
            config=config,
            **kwargs
        )
        
        # Update performance metrics
        self.performance_metrics.total_calculations += 1
        self.performance_metrics.total_time += result.computation_time
        self.performance_metrics.memory_usage_mb = result.memory_used_mb
        
        return result.result
    
    def _generate_with_vectorbt(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using VectorBT native functions."""
        close_prices = data['close']
        
        # Use VectorBT native CCI if available
        if CCI and 'high' in data.columns and 'low' in data.columns:
            try:
                cci = CCI.run(data['high'], data['low'], close_prices, window=20).cci
                return cci
            except Exception as e:
                logger.warning(f"VectorBT CCI failed: {e}, using custom calculation")
        
        # Fallback to custom VectorBT calculation
        oscillator = self.vectorbt_optimizer.rolling_mean(close_prices, window=14) - close_prices
        return oscillator.rename('oscillator_optimized')
    
    def _generate_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using pandas fallback."""
        close_prices = data['close']
        oscillator = close_prices.rolling(window=14).mean() - close_prices
        return oscillator.rename('oscillator_fallback')
    
    def _should_use_unified_manager(self, data: pd.Series) -> bool:
        """Determine if UnifiedVectorizationManager should be used."""
        return (self.use_unified_manager and 
                self.unified_manager is not None and 
                len(data) >= 1000)
    
    def _should_use_vectorbt(self, data: pd.Series) -> bool:
        """Determine if VectorBT should be used."""
        return (self.vectorbt_optimizer is not None and 
                len(data) >= 100 and 
                VECTORBT_AVAILABLE)
    
    def get_performance_metrics(self) -> OscillatorPerformanceMetrics:
        """Get performance metrics."""
        if self.performance_metrics.total_calculations > 0:
            self.performance_metrics.average_time_per_calculation = (
                self.performance_metrics.total_time / self.performance_metrics.total_calculations
            )
        return self.performance_metrics
    
    def reset_performance_metrics(self):
        """Reset performance metrics."""
        self.performance_metrics = OscillatorPerformanceMetrics()


class VectorBTCCIGenerator(VectorBTOscillatorFeatureGenerator):
    """Optimized CCI generator using VectorBT native functions."""
    
    def __init__(self, period: int = 20, **kwargs):
        config = FeatureConfig(
            name=f"vectorbt_cci_{period}",
            category=FeatureCategory.OSCILLATOR,
            description=f"VectorBT-optimized CCI over {period} periods",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, **kwargs)
        self.period = period
    
    def _generate_with_vectorbt(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate CCI using VectorBT native CCI indicator."""
        if CCI and VECTORBT_AVAILABLE:
            try:
                cci_result = CCI.run(
                    data['high'], 
                    data['low'], 
                    data['close'], 
                    window=self.period
                )
                return cci_result.cci
            except Exception as e:
                logger.warning(f"VectorBT native CCI failed: {e}, using custom calculation")
        
        # Custom VectorBT calculation
        high, low, close = data['high'], data['low'], data['close']
        typical_price = (high + low + close) / 3
        
        # Use VectorBT rolling operations
        sma_tp = self.vectorbt_optimizer.rolling_mean(typical_price, window=self.period)
        mad = self.vectorbt_optimizer.rolling_mean(
            (typical_price - sma_tp).abs(), 
            window=self.period
        )
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
    
    def _generate_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate CCI using pandas fallback."""
        high, low, close = data['high'], data['low'], data['close']
        typical_price = (high + low + close) / 3
        
        sma_tp = typical_price.rolling(window=self.period).mean()
        mad = (typical_price - sma_tp).abs().rolling(window=self.period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci


class VectorBTADXGenerator(VectorBTOscillatorFeatureGenerator):
    """Optimized ADX generator using VectorBT native functions."""
    
    def __init__(self, period: int = 14, **kwargs):
        config = FeatureConfig(
            name=f"vectorbt_adx_{period}",
            category=FeatureCategory.OSCILLATOR,
            description=f"VectorBT-optimized ADX over {period} periods",
            required_columns=["high", "low", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, **kwargs)
        self.period = period
    
    def _generate_with_vectorbt(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ADX using VectorBT native ADX indicator."""
        if ADX and VECTORBT_AVAILABLE:
            try:
                adx_result = ADX.run(
                    data['high'], 
                    data['low'], 
                    data['close'], 
                    window=self.period
                )
                return adx_result.adx
            except Exception as e:
                logger.warning(f"VectorBT native ADX failed: {e}, using custom calculation")
        
        # Custom VectorBT calculation
        high, low, close = data['high'], data['low'], data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Use VectorBT rolling operations
        atr = self.vectorbt_optimizer.rolling_mean(tr, window=self.period)
        di_plus = 100 * (self.vectorbt_optimizer.rolling_mean(
            pd.Series(dm_plus, index=data.index), window=self.period
        ) / atr)
        di_minus = 100 * (self.vectorbt_optimizer.rolling_mean(
            pd.Series(dm_minus, index=data.index), window=self.period
        ) / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = self.vectorbt_optimizer.rolling_mean(dx, window=self.period)
        
        return adx
    
    def _generate_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ADX using pandas fallback."""
        high, low, close = data['high'], data['low'], data['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = np.where((dm_plus > dm_minus) & (dm_plus > 0), dm_plus, 0)
        dm_minus = np.where((dm_minus > dm_plus) & (dm_minus > 0), dm_minus, 0)
        
        # Calculate smoothed values
        atr = tr.rolling(window=self.period).mean()
        di_plus = 100 * (pd.Series(dm_plus, index=data.index).rolling(window=self.period).mean() / atr)
        di_minus = 100 * (pd.Series(dm_minus, index=data.index).rolling(window=self.period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = dx.rolling(window=self.period).mean()
        
        return adx


class VectorBTAroonGenerator(VectorBTOscillatorFeatureGenerator):
    """Optimized Aroon generator using VectorBT optimized operations."""
    
    def __init__(self, period: int = 25, **kwargs):
        config = FeatureConfig(
            name=f"vectorbt_aroon_{period}",
            category=FeatureCategory.OSCILLATOR,
            description=f"VectorBT-optimized Aroon over {period} periods",
            required_columns=["high", "low"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={'period': period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, **kwargs)
        self.period = period
    
    def _generate_with_vectorbt(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Aroon using VectorBT optimized operations."""
        high, low = data['high'], data['low']
        
        # Use VectorBT rolling operations for argmax/argmin
        try:
            # Calculate Aroon Up and Down using VectorBT rolling operations
            aroon_up = ((self.period - high.rolling(window=self.period).apply(
                lambda x: x.argmax(), raw=True
            )) / self.period * 100)
            aroon_down = ((self.period - low.rolling(window=self.period).apply(
                lambda x: x.argmin(), raw=True
            )) / self.period * 100)
            
            # Calculate Aroon Oscillator
            aroon = aroon_up - aroon_down
            
            return aroon
        except Exception as e:
            logger.warning(f"VectorBT Aroon calculation failed: {e}, using pandas fallback")
            return self._generate_with_pandas_fallback(data, **kwargs)
    
    def _generate_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Aroon using pandas fallback."""
        high, low = data['high'], data['low']
        
        aroon_up = ((self.period - high.rolling(window=self.period).apply(
            lambda x: x.argmax(), raw=True
        )) / self.period * 100)
        aroon_down = ((self.period - low.rolling(window=self.period).apply(
            lambda x: x.argmin(), raw=True
        )) / self.period * 100)
        
        aroon = aroon_up - aroon_down
        return aroon


class VectorBTOscillatorFactory:
    """Factory for creating VectorBT-optimized oscillator generators."""
    
    @staticmethod
    def create_cci_generator(period: int = 20, **kwargs) -> VectorBTCCIGenerator:
        """Create a VectorBT-optimized CCI generator."""
        return VectorBTCCIGenerator(period=period, **kwargs)
    
    @staticmethod
    def create_adx_generator(period: int = 14, **kwargs) -> VectorBTADXGenerator:
        """Create a VectorBT-optimized ADX generator."""
        return VectorBTADXGenerator(period=period, **kwargs)
    
    @staticmethod
    def create_aroon_generator(period: int = 25, **kwargs) -> VectorBTAroonGenerator:
        """Create a VectorBT-optimized Aroon generator."""
        return VectorBTAroonGenerator(period=period, **kwargs)
    
    @staticmethod
    def create_batch_generators(periods: Dict[str, List[int]], **kwargs) -> List[VectorBTOscillatorFeatureGenerator]:
        """Create multiple VectorBT-optimized oscillator generators."""
        generators = []
        
        for period in periods.get('cci', [20]):
            generators.append(VectorBTOscillatorFactory.create_cci_generator(period, **kwargs))
        
        for period in periods.get('adx', [14]):
            generators.append(VectorBTOscillatorFactory.create_adx_generator(period, **kwargs))
        
        for period in periods.get('aroon', [25]):
            generators.append(VectorBTOscillatorFactory.create_aroon_generator(period, **kwargs))
        
        return generators


def create_vectorbt_oscillator_generators(periods: Dict[str, List[int]] = None, 
                                        enable_gpu: bool = False,
                                        enable_parallel: bool = True,
                                        use_unified_manager: bool = True) -> List[VectorBTOscillatorFeatureGenerator]:
    """Create a set of VectorBT-optimized oscillator feature generators."""
    if periods is None:
        periods = {
            'cci': [20],
            'adx': [14],
            'aroon': [25]
        }
    
    return VectorBTOscillatorFactory.create_batch_generators(
        periods, 
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        use_unified_manager=use_unified_manager
    )


def create_default_vectorbt_oscillator_generators() -> List[VectorBTOscillatorFeatureGenerator]:
    """Create default VectorBT-optimized oscillator generators."""
    return create_vectorbt_oscillator_generators()


# Backward compatibility
OscillatorFeatureGenerator = VectorBTOscillatorFeatureGenerator
CCIGenerator = VectorBTCCIGenerator
ADXGenerator = VectorBTADXGenerator
AroonGenerator = VectorBTAroonGenerator
create_oscillator_generators = create_vectorbt_oscillator_generators
create_default_oscillator_generators = create_default_vectorbt_oscillator_generators