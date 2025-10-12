"""
Oscillator Feature Generator

This module provides feature generators for oscillator indicators,
including CCI, ADX, Aroon, Ultimate Oscillator, KST, APO, CMO, NATR, PFE, T3, KAMA, and more.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Any, Dict, List, Optional, Union

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# Import tprint for consistent logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

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

# VectorBT Rolling Optimizer - NOW USING NEW OPTIMIZED VERSION
try:
    from ..utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    from ..utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    VECTORBT_OPTIMIZER_AVAILABLE = True
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
        VECTORBT_OPTIMIZER_AVAILABLE = True
        OPTIMIZATION_AVAILABLE = False
    except ImportError:
        VECTORBT_OPTIMIZER_AVAILABLE = False
        OPTIMIZATION_AVAILABLE = False
        VectorBTRollingOptimizer = None

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

# Optimization utilities
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

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
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank

logger = logging.getLogger(__name__)

class OscillatorFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Feature generator for oscillator-based features with comprehensive VectorBT optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None, 
                 enable_gpu: bool = False, enable_parallel: bool = True,
                 use_unified_manager: bool = True):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'unified_manager_operations': 0,
            'total_time': 0.0,
            'average_time_per_calculation': 0.0,
            'memory_usage_mb': 0.0
        }
        
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
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="oscillator_features",
            category=FeatureCategory.OSCILLATOR,
            description="Comprehensive oscillator-based features including Stochastic and Williams %R",
            required_columns=["close"],
            optional_columns=["high", "low"],
            default_lookback=14,
            min_lookback=2,
            max_lookback=50,
            parameters={
                "stochastic_periods": [14],
                "williams_periods": [14]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
    
    @classmethod
    def create_default(cls) -> 'OscillatorFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        close_prices = data['close']
        
        # Use UnifiedVectorizationManager for intelligent optimization
        if self.unified_manager and self._should_use_unified_manager(close_prices):
            try:
                result = self._generate_with_unified_manager(data, **kwargs)
                self.performance_stats['unified_manager_operations'] += 1
                return result
            except Exception as e:
                self.logger.warning(f"UnifiedVectorizationManager failed: {e}, using VectorBT fallback")
        
        # Use VectorBT for oscillator calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(close_prices):
            try:
                # Enhanced oscillator using VectorBT rolling mean
                oscillator = self.vectorbt_optimizer.rolling_mean(close_prices, window=14) - close_prices
                self.performance_stats['vectorbt_operations'] += 1
                return oscillator.rename('oscillator_vectorbt')
            except Exception as e:
                self.logger.warning(f"VectorBT oscillator calculation failed: {e}, using pandas fallback")
                result = self._generate_with_pandas_fallback(data, **kwargs)
                self.performance_stats['pandas_fallbacks'] += 1
                return result
        else:
            result = self._generate_with_pandas_fallback(data, **kwargs)
            self.performance_stats['pandas_fallbacks'] += 1
            return result
        finally:
            # Update performance statistics
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            if self.performance_stats['total_calculations'] > 0:
                self.performance_stats['average_time_per_calculation'] = (
                    self.performance_stats['total_time'] / self.performance_stats['total_calculations']
                )
    
    def _generate_with_unified_manager(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using UnifiedVectorizationManager."""
        try:
            # Use Unified Vectorization Manager for optimized oscillator calculation
            close_prices = data['close']
            
            oscillator_result = self.unified_manager.optimize_operation(
                OperationType.TECHNICAL_INDICATORS,
                {
                    'data': close_prices,
                    'operation': 'oscillator',
                    'window': 14,
                    'indicator_configs': {'oscillator': {'window': 14}}
                },
                OperationConfig(
                    operation_type=OperationType.TECHNICAL_INDICATORS,
                    data_size=len(close_prices),
                    data_dimensions=close_prices.shape,
                    memory_budget_mb=256.0
                )
            )
            oscillator = oscillator_result.result
            
            # Update memory usage
            if hasattr(self.unified_manager, 'get_performance_stats'):
                stats = self.unified_manager.get_performance_stats()
                self.performance_stats['memory_usage_mb'] = stats.get('memory_used_mb', 0)
            
            return oscillator.rename('oscillator_unified')
            
        except Exception as e:
            self.logger.warning(f"Unified Vectorization Manager oscillator calculation failed: {e}")
            # Fallback to VectorBT rolling optimizer
            if self.vectorbt_optimizer:
                try:
                    rolling_mean = self.vectorbt_optimizer.rolling_mean(close_prices, window=14)
                    oscillator = rolling_mean - close_prices
                    return oscillator.rename('oscillator_vectorbt_fallback')
                except Exception as e2:
                    self.logger.warning(f"VectorBT fallback failed: {e2}")
                    return self._generate_with_pandas_fallback(data, **kwargs)
            else:
                return self._generate_with_pandas_fallback(data, **kwargs)
    
    def _generate_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate features using pandas fallback."""
        close_prices = data['close']
        oscillator = close_prices.rolling(window=14).mean() - close_prices
        return oscillator.rename('oscillator_pandas')
    
    def _should_use_unified_manager(self, data: pd.Series) -> bool:
        """Determine if UnifiedVectorizationManager should be used."""
        return (self.use_unified_manager and 
                self.unified_manager is not None and 
                len(data) >= 1000)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()
    
    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.performance_stats = {
            'total_calculations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'unified_manager_operations': 0,
            'total_time': 0.0,
            'average_time_per_calculation': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def generate_optimized_oscillator_features(self, data: pd.DataFrame, 
                                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple oscillator features using optimized batch processing.
        
        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated oscillator features
        """
        if self.unified_manager and self._should_use_unified_manager(data['close']):
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'oscillator_batch'
                    },
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=1024.0
                    )
                )
                return batch_result.result
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using VectorBT fallback")
                return self._generate_batch_with_vectorbt(data, feature_configs)
        elif self.vectorbt_optimizer and self._should_use_vectorbt(data['close']):
            try:
                return self._generate_batch_with_vectorbt(data, feature_configs)
            except Exception as e:
                self.logger.warning(f"VectorBT batch processing failed: {e}, using individual processing")
                return self._process_oscillator_features_individually(data, feature_configs)
        else:
            return self._process_oscillator_features_individually(data, feature_configs)
    
    def _process_oscillator_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process oscillator features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'oscillator')
            params = config.get('params', {})
            
            try:
                if feature_type == 'oscillator':
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        
                        if self.vectorbt_optimizer:
                            rolling_mean = self.vectorbt_optimizer.rolling_mean(series_data, window)
                            oscillator = rolling_mean - series_data
                        else:
                            rolling_mean = series_data.rolling(window).mean()
                            oscillator = rolling_mean - series_data
                        
                        results[feature_name] = oscillator
                
            except Exception as e:
                self.logger.warning(f"Oscillator feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _generate_batch_with_vectorbt(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using VectorBT rolling optimizer."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'oscillator')
            params = config.get('params', {})
            
            try:
                if feature_type == 'oscillator':
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        rolling_mean = self.vectorbt_optimizer.rolling_mean(series_data, window)
                        oscillator = rolling_mean - series_data
                        results[feature_name] = oscillator
                
            except Exception as e:
                self.logger.warning(f"Oscillator feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _generate_batch_with_vectorbt(self, data: pd.DataFrame, 
                                    feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using VectorBT rolling optimizer."""
        results = {}
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'oscillator')
            params = config.get('params', {})
            
            try:
                if feature_type == 'oscillator':
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        rolling_mean = self.vectorbt_optimizer.rolling_mean(series_data, window=window)
                        results[feature_name] = rolling_mean - series_data
                
                elif feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        if operation == 'mean':
                            results[feature_name] = self.vectorbt_optimizer.rolling_mean(series_data, window=window)
                        elif operation == 'std':
                            results[feature_name] = self.vectorbt_optimizer.rolling_std(series_data, window=window)
                        elif operation == 'min':
                            results[feature_name] = self.vectorbt_optimizer.rolling_min(series_data, window=window)
                        elif operation == 'max':
                            results[feature_name] = self.vectorbt_optimizer.rolling_max(series_data, window=window)
                
            except Exception as e:
                self.logger.warning(f"Oscillator feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)
    
    def _generate_batch_with_pandas(self, data: pd.DataFrame, 
                                  feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Generate batch features using pandas fallback."""
        results = {}
        
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'oscillator')
            params = config.get('params', {})
            
            try:
                if feature_type == 'oscillator':
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        rolling_mean = series_data.rolling(window=window).mean()
                        results[feature_name] = rolling_mean - series_data
                
                elif feature_type == 'rolling':
                    operation = params.get('operation', 'mean')
                    window = params.get('window', 14)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        series_data = data[column]
                        rolling_obj = series_data.rolling(window=window)
                        if operation == 'mean':
                            results[feature_name] = rolling_obj.mean()
                        elif operation == 'std':
                            results[feature_name] = rolling_obj.std()
                        elif operation == 'min':
                            results[feature_name] = rolling_obj.min()
                        elif operation == 'max':
                            results[feature_name] = rolling_obj.max()
                
            except Exception as e:
                self.logger.warning(f"Oscillator feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)

def create_oscillator_generators(periods: Dict[str, List[int]] = None) -> List[FeatureGenerator]:
    """Create a set of oscillator feature generators."""
    if periods is None:
        periods = {
            'cci': [20],
            'adx': [14],
            'aroon': [25],
            'ultimate': [14],
            'kst': [10, 15, 20, 30],
            'apo': [12, 26],
            'cmo': [14],
            'natr': [14],
            'pfe': [12],
            't3': [14],
            'kama': [30]
        }
    
    generators = []
    
    # CCI generators
    for period in periods.get('cci', [20]):
        generators.append(CCIGenerator(period=period))
    
    # ADX generators
    for period in periods.get('adx', [14]):
        generators.append(ADXGenerator(period=period))
    
    # Aroon generators
    for period in periods.get('aroon', [25]):
        generators.append(AroonGenerator(period=period))
    
    # Ultimate Oscillator generators
    for period in periods.get('ultimate', [14]):
        generators.append(UltimateOscillatorGenerator(period=period))
    
    # KST generators
    for period in periods.get('kst', [10]):
        generators.append(KSTGenerator(period=period))
    
    # APO generators
    for period in periods.get('apo', [12]):
        generators.append(APOGenerator(period=period))
    
    # CMO generators
    for period in periods.get('cmo', [14]):
        generators.append(CMOGenerator(period=period))
    
    # NATR generators
    for period in periods.get('natr', [14]):
        generators.append(NATRGenerator(period=period))
    
    # PFE generators
    for period in periods.get('pfe', [12]):
        generators.append(PFEGenerator(period=period))
    
    # T3 generators
    for period in periods.get('t3', [14]):
        generators.append(T3Generator(period=period))
    
    # KAMA generators
    for period in periods.get('kama', [30]):
        generators.append(KAMAGenerator(period=period))
    
    return generators

def create_default_oscillator_generators() -> List[FeatureGenerator]:
    return create_oscillator_generators()

# CCI (Commodity Channel Index)
    
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

class CCIGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Generator for CCI (Commodity Channel Index) with comprehensive VectorBT optimization."""
    
    def __init__(self,
                 period: int = 20,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 enable_gpu: bool = False,
                 enable_parallel: bool = True,
                 use_unified_manager: bool = True,
                 **base_kwargs):
        """
        Initialize CCI generator.
        
        Args:
            period: CCI period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cci_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Commodity Channel Index over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
        
        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'vectorbt_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'unified_manager_operations': 0,
            'total_time': 0.0,
            'average_time_per_calculation': 0.0,
            'memory_usage_mb': 0.0
        }
        
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
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate CCI based on the specified base calculation with comprehensive VectorBT optimization."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Use VectorBT native CCI if available
            if CCI and VECTORBT_AVAILABLE and self._should_use_vectorbt(close):
                try:
                    cci_result = CCI.run(high, low, close, window=self.period)
                    self.performance_stats['vectorbt_operations'] += 1
                    return cci_result.cci
                except Exception as e:
                    self.logger.warning(f"VectorBT native CCI failed: {e}, using custom calculation")
            
            # Use VectorBT for CCI calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(close):
                try:
                    # Calculate typical price
                    typical_price = (high + low + close) / 3
                    
                    # Calculate CCI using VectorBT rolling operations
                    sma_tp = self.vectorbt_optimizer.rolling_mean(typical_price, window=self.period)
                    # Vectorized MAD: use rolling mean of absolute deviations
                    mad = self.vectorbt_optimizer.rolling_mean((typical_price - self.vectorbt_optimizer.rolling_mean(typical_price, window=self.period)).abs(), window=self.period)
                    cci = (typical_price - sma_tp) / (0.015 * mad)
                    
                    self.performance_stats['vectorbt_operations'] += 1
                    return cci
                except Exception as e:
                    self.logger.warning(f"VectorBT CCI calculation failed: {e}, using pandas fallback")
                    result = self._generate_cci_pandas_fallback(high, low, close)
                    self.performance_stats['pandas_fallbacks'] += 1
                    return result
            else:
                result = self._generate_cci_pandas_fallback(high, low, close)
                self.performance_stats['pandas_fallbacks'] += 1
                return result
        finally:
            # Update performance statistics
            self.performance_stats['total_calculations'] += 1
            self.performance_stats['total_time'] += time.time() - start_time
            if self.performance_stats['total_calculations'] > 0:
                self.performance_stats['average_time_per_calculation'] = (
                    self.performance_stats['total_time'] / self.performance_stats['total_calculations']
                )
    
    def _generate_cci_pandas_fallback(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Generate CCI using pandas fallback."""
        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate CCI - OPTIMIZED: Vectorized MAD calculation
        sma_tp = typical_price.rolling(window=self.period).mean()
        # Vectorized MAD: use rolling mean of absolute deviations
        mad = (typical_price - typical_price.rolling(window=self.period).mean()).abs().rolling(window=self.period).mean()
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        return cci
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for CCI calculation on base values
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # Calculate CCI on base values using VectorBT
                    sma_base = self.vectorbt_optimizer.rolling_mean(base_values, window=self.period)
                    # Vectorized MAD: use rolling mean of absolute deviations
                    mad_base = self.vectorbt_optimizer.rolling_mean((base_values - self.vectorbt_optimizer.rolling_mean(base_values, window=self.period)).abs(), window=self.period)
                    cci = (base_values - sma_base) / (0.015 * mad_base)
                    
                    return cci
                except Exception as e:
                    self.logger.warning(f"VectorBT CCI calculation failed: {e}, using pandas fallback")
                    # Calculate CCI on base values - OPTIMIZED: Vectorized MAD calculation
                    sma_base = base_values.rolling(window=self.period).mean()
                    # Vectorized MAD: use rolling mean of absolute deviations
                    mad_base = (base_values - base_values.rolling(window=self.period).mean()).abs().rolling(window=self.period).mean()
                    cci = (base_values - sma_base) / (0.015 * mad_base)
                    
                    return cci
            else:
                # Calculate CCI on base values - OPTIMIZED: Vectorized MAD calculation
                sma_base = base_values.rolling(window=self.period).mean()
                # Vectorized MAD: use rolling mean of absolute deviations
                mad_base = (base_values - base_values.rolling(window=self.period).mean()).abs().rolling(window=self.period).mean()
                cci = (base_values - sma_base) / (0.015 * mad_base)
                
                return cci

# ADX (Average Directional Index)
class ADXGenerator(VectorizedFeatureGenerator):
    """Generator for ADX (Average Directional Index) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize ADX generator.
        
        Args:
            period: ADX period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"adx_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Average Directional Index over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate ADX based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Use VectorBT for ADX calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(close):
                try:
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
                    
                    # Calculate smoothed values using VectorBT
                    atr = self.vectorbt_optimizer.rolling_mean(tr, window=self.period)
                    di_plus = 100 * (self.vectorbt_optimizer.rolling_mean(pd.Series(dm_plus, index=data.index), window=self.period) / atr)
                    di_minus = 100 * (self.vectorbt_optimizer.rolling_mean(pd.Series(dm_minus, index=data.index), window=self.period) / atr)
                    
                    # Calculate ADX
                    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
                    adx = self.vectorbt_optimizer.rolling_mean(dx, window=self.period)
                    
                    return adx
                except Exception as e:
                    self.logger.warning(f"VectorBT ADX calculation failed: {e}, using pandas fallback")
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
            else:
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
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for base values calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # For other base calculations, use rolling standard deviation as proxy
                    # Convert to pandas Series if it's a numpy array
                    if isinstance(base_values, np.ndarray):
                        base_values = pd.Series(base_values, index=data.index)
                    adx = self.vectorbt_optimizer.rolling_std(base_values, window=self.period)
                    return adx
                except Exception as e:
                    self.logger.warning(f"VectorBT ADX base calculation failed: {e}, using pandas fallback")
                    # For other base calculations, use rolling standard deviation as proxy
                    # Convert to pandas Series if it's a numpy array
                    if isinstance(base_values, np.ndarray):
                        base_values = pd.Series(base_values, index=data.index)
                    adx = base_values.rolling(window=self.period).std()
                    return adx
            else:
                # For other base calculations, use rolling standard deviation as proxy
                # Convert to pandas Series if it's a numpy array
                if isinstance(base_values, np.ndarray):
                    base_values = pd.Series(base_values, index=data.index)
                adx = base_values.rolling(window=self.period).std()
                return adx

# Aroon Oscillator
class AroonGenerator(VectorizedFeatureGenerator):
    """Generator for Aroon Oscillator with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize Aroon generator.
        
        Args:
            period: Aroon period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"aroon_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Aroon Oscillator over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Aroon based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            
            # Use VectorBT for Aroon calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(high):
                try:
                    # OPTIMIZED: Use vectorized argmax/argmin calculations
                    # Use pandas built-in rolling idxmax/idxmin for better performance
                    aroon_up = ((self.period - high.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                    aroon_down = ((self.period - low.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                    
                    # Calculate Aroon Oscillator
                    aroon = aroon_up - aroon_down
                    
                    return aroon
                except Exception as e:
                    self.logger.warning(f"VectorBT Aroon calculation failed: {e}, using pandas fallback")
                    # OPTIMIZED: Use vectorized argmax/argmin calculations
                    # Use pandas built-in rolling idxmax/idxmin for better performance
                    aroon_up = ((self.period - high.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                    aroon_down = ((self.period - low.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                    
                    # Calculate Aroon Oscillator
                    aroon = aroon_up - aroon_down
                    
                    return aroon
            else:
                # OPTIMIZED: Use vectorized argmax/argmin calculations
                # Use pandas built-in rolling idxmax/idxmin for better performance
                aroon_up = ((self.period - high.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                aroon_down = ((self.period - low.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                
                # Calculate Aroon Oscillator
                aroon = aroon_up - aroon_down
                
                return aroon
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for Aroon calculation on base values
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # OPTIMIZED: Use vectorized argmax/argmin calculations
                    # Use pandas built-in rolling idxmax/idxmin for better performance
                    aroon_up = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                    aroon_down = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                    
                    aroon = aroon_up - aroon_down
                    
                    return aroon
                except Exception as e:
                    self.logger.warning(f"VectorBT Aroon base calculation failed: {e}, using pandas fallback")
                    # OPTIMIZED: Use vectorized argmax/argmin calculations
                    # Use pandas built-in rolling idxmax/idxmin for better performance
                    aroon_up = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                    aroon_down = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                    
                    aroon = aroon_up - aroon_down
                    
                    return aroon
            else:
                # OPTIMIZED: Use vectorized argmax/argmin calculations
                # Use pandas built-in rolling idxmax/idxmin for better performance
                aroon_up = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmax(), raw=True)) / self.period * 100)
                aroon_down = ((self.period - base_values.rolling(window=self.period).apply(lambda x: x.argmin(), raw=True)) / self.period * 100)
                
                aroon = aroon_up - aroon_down
                
                return aroon

# Parabolic SAR
class SARGenerator(VectorizedFeatureGenerator):
    """Generator for Parabolic SAR with different base calculations."""
    
    def __init__(self,
                 acceleration: float = 0.02,
                 maximum: float = 0.2,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize Parabolic SAR generator.
        
        Args:
            acceleration: SAR acceleration factor
            maximum: SAR maximum acceleration
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"sar_{acceleration}_{maximum}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Parabolic SAR with acceleration={acceleration}, maximum={maximum} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=20,
            min_lookback=1,
            max_lookback=50,
            parameters={
                'acceleration': acceleration,
                'maximum': maximum,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.acceleration = acceleration
        self.maximum = maximum
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Parabolic SAR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Use VectorBT for SAR calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(high):
                try:
                    # Simplified SAR calculation
                    sar = pd.Series(index=data.index, dtype=float)
                    sar.iloc[0] = low.iloc[0]
                    
                    for i in range(1, len(data)):
                        if close.iloc[i] > sar.iloc[i-1]:
                            sar.iloc[i] = sar.iloc[i-1] + self.acceleration * (high.iloc[i] - sar.iloc[i-1])
                        else:
                            sar.iloc[i] = sar.iloc[i-1] - self.acceleration * (sar.iloc[i-1] - low.iloc[i])
                    
                    return sar
                except Exception as e:
                    self.logger.warning(f"VectorBT SAR calculation failed: {e}, using pandas fallback")
                    # Simplified SAR calculation
                    sar = pd.Series(index=data.index, dtype=float)
                    sar.iloc[0] = low.iloc[0]
                    
                    for i in range(1, len(data)):
                        if close.iloc[i] > sar.iloc[i-1]:
                            sar.iloc[i] = sar.iloc[i-1] + self.acceleration * (high.iloc[i] - sar.iloc[i-1])
                        else:
                            sar.iloc[i] = sar.iloc[i-1] - self.acceleration * (sar.iloc[i-1] - low.iloc[i])
                    
                    return sar
            else:
                # Simplified SAR calculation
                sar = pd.Series(index=data.index, dtype=float)
                sar.iloc[0] = low.iloc[0]
                
                for i in range(1, len(data)):
                    if close.iloc[i] > sar.iloc[i-1]:
                        sar.iloc[i] = sar.iloc[i-1] + self.acceleration * (high.iloc[i] - sar.iloc[i-1])
                    else:
                        sar.iloc[i] = sar.iloc[i-1] - self.acceleration * (sar.iloc[i-1] - low.iloc[i])
                
                return sar
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for base values calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # For other base calculations, use rolling mean as proxy
                    sar = self.vectorbt_optimizer.rolling_mean(base_values, window=20)
                    return sar
                except Exception as e:
                    self.logger.warning(f"VectorBT SAR base calculation failed: {e}, using pandas fallback")
                    # For other base calculations, use rolling mean as proxy
                    sar = self._calculate_sma_vectorized(base_values, 20)
                    return sar
            else:
                # For other base calculations, use rolling mean as proxy
                sar = self._calculate_sma_vectorized(base_values, 20)
                return sar

# Ultimate Oscillator
class UltimateOscillatorGenerator(VectorizedFeatureGenerator):
    """Generator for Ultimate Oscillator with different base calculations."""
    
    def __init__(self,
                 period1: int = 7,
                 period2: int = 14,
                 period3: int = 28,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize Ultimate Oscillator generator.
        
        Args:
            period1: First period
            period2: Second period
            period3: Third period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ultimate_oscillator_{period1}_{period2}_{period3}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Ultimate Oscillator with periods {period1}, {period2}, {period3} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period3,
            min_lookback=period3,
            max_lookback=period3,
            parameters={
                'period1': period1,
                'period2': period2,
                'period3': period3,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate Ultimate Oscillator based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Use VectorBT for Ultimate Oscillator calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(high):
                try:
                    # Calculate True Range
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Calculate Buying Pressure
                    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
                    
                    # Calculate Ultimate Oscillator using VectorBT rolling sum
                    avg7 = rolling_sum(bp, window=self.period1) / rolling_sum(tr, window=self.period1)
                    avg14 = rolling_sum(bp, window=self.period2) / rolling_sum(tr, window=self.period2)
                    avg28 = rolling_sum(bp, window=self.period3) / rolling_sum(tr, window=self.period3)
                    
                    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
                    
                    return uo
                except Exception as e:
                    self.logger.warning(f"VectorBT Ultimate Oscillator calculation failed: {e}, using pandas fallback")
                    # Calculate True Range
                    tr1 = high - low
                    tr2 = abs(high - close.shift(1))
                    tr3 = abs(low - close.shift(1))
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Calculate Buying Pressure
                    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
                    
                    # Calculate Ultimate Oscillator
                    avg7 = bp.rolling(window=self.period1).sum() / tr.rolling(window=self.period1).sum()
                    avg14 = bp.rolling(window=self.period2).sum() / tr.rolling(window=self.period2).sum()
                    avg28 = bp.rolling(window=self.period3).sum() / tr.rolling(window=self.period3).sum()
                    
                    uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
                    
                    return uo
            else:
                # Calculate True Range
                tr1 = high - low
                tr2 = abs(high - close.shift(1))
                tr3 = abs(low - close.shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # Calculate Buying Pressure
                bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
                
                # Calculate Ultimate Oscillator
                avg7 = bp.rolling(window=self.period1).sum() / tr.rolling(window=self.period1).sum()
                avg14 = bp.rolling(window=self.period2).sum() / tr.rolling(window=self.period2).sum()
                avg28 = bp.rolling(window=self.period3).sum() / tr.rolling(window=self.period3).sum()
                
                uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
                
                return uo
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for base values calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # For other base calculations, use rolling mean as proxy
                    uo = self.vectorbt_optimizer.rolling_mean(base_values, window=self.period3)
                    return uo
                except Exception as e:
                    self.logger.warning(f"VectorBT Ultimate Oscillator base calculation failed: {e}, using pandas fallback")
                    # For other base calculations, use rolling mean as proxy
                    uo = base_values.rolling(window=self.period3).mean()
                    return uo
            else:
                # For other base calculations, use rolling mean as proxy
                uo = base_values.rolling(window=self.period3).mean()
                return uo

# KST (Know Sure Thing)
class KSTGenerator(VectorizedFeatureGenerator):
    """Generator for KST (Know Sure Thing) with different base calculations."""
    
    def __init__(self,
                 roc1: int = 10,
                 roc2: int = 15,
                 roc3: int = 20,
                 roc4: int = 30,
                 sma1: int = 10,
                 sma2: int = 10,
                 sma3: int = 10,
                 sma4: int = 15,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize KST generator.
        
        Args:
            roc1: First ROC period
            roc2: Second ROC period
            roc3: Third ROC period
            roc4: Fourth ROC period
            sma1: First SMA period
            sma2: Second SMA period
            sma3: Third SMA period
            sma4: Fourth SMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"kst_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"KST with ROC periods {roc1}, {roc2}, {roc3}, {roc4} and SMA periods {sma1}, {sma2}, {sma3}, {sma4} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=max(roc4, sma4),
            min_lookback=max(roc4, sma4),
            max_lookback=max(roc4, sma4),
            parameters={
                'roc1': roc1,
                'roc2': roc2,
                'roc3': roc3,
                'roc4': roc4,
                'sma1': sma1,
                'sma2': sma2,
                'sma3': sma3,
                'sma4': sma4,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.roc1 = roc1
        self.roc2 = roc2
        self.roc3 = roc3
        self.roc4 = roc4
        self.sma1 = sma1
        self.sma2 = sma2
        self.sma3 = sma3
        self.sma4 = sma4
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate KST based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            close = data['close']
            
            # Use VectorBT for KST calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(close):
                try:
                    # Calculate ROC
                    roc1 = close.pct_change(periods=self.roc1) * 100
                    roc2 = close.pct_change(periods=self.roc2) * 100
                    roc3 = close.pct_change(periods=self.roc3) * 100
                    roc4 = close.pct_change(periods=self.roc4) * 100
                    
                    # Calculate SMA of ROC using VectorBT
                    sma_roc1 = self.vectorbt_optimizer.rolling_mean(roc1, window=self.sma1)
                    sma_roc2 = self.vectorbt_optimizer.rolling_mean(roc2, window=self.sma2)
                    sma_roc3 = self.vectorbt_optimizer.rolling_mean(roc3, window=self.sma3)
                    sma_roc4 = self.vectorbt_optimizer.rolling_mean(roc4, window=self.sma4)
                    
                    # Calculate KST
                    kst = sma_roc1 + 2 * sma_roc2 + 3 * sma_roc3 + 4 * sma_roc4
                    
                    return kst
                except Exception as e:
                    self.logger.warning(f"VectorBT KST calculation failed: {e}, using pandas fallback")
                    # Calculate ROC
                    roc1 = close.pct_change(periods=self.roc1) * 100
                    roc2 = close.pct_change(periods=self.roc2) * 100
                    roc3 = close.pct_change(periods=self.roc3) * 100
                    roc4 = close.pct_change(periods=self.roc4) * 100
                    
                    # Calculate SMA of ROC
                    sma_roc1 = roc1.rolling(window=self.sma1).mean()
                    sma_roc2 = roc2.rolling(window=self.sma2).mean()
                    sma_roc3 = roc3.rolling(window=self.sma3).mean()
                    sma_roc4 = roc4.rolling(window=self.sma4).mean()
                    
                    # Calculate KST
                    kst = sma_roc1 + 2 * sma_roc2 + 3 * sma_roc3 + 4 * sma_roc4
                    
                    return kst
            else:
                # Calculate ROC
                roc1 = close.pct_change(periods=self.roc1) * 100
                roc2 = close.pct_change(periods=self.roc2) * 100
                roc3 = close.pct_change(periods=self.roc3) * 100
                roc4 = close.pct_change(periods=self.roc4) * 100
                
                # Calculate SMA of ROC
                sma_roc1 = roc1.rolling(window=self.sma1).mean()
                sma_roc2 = roc2.rolling(window=self.sma2).mean()
                sma_roc3 = roc3.rolling(window=self.sma3).mean()
                sma_roc4 = roc4.rolling(window=self.sma4).mean()
                
                # Calculate KST
                kst = sma_roc1 + 2 * sma_roc2 + 3 * sma_roc3 + 4 * sma_roc4
                
                return kst
        else:
            base_values = self.base_calculator.calculate(data)
            
            # Use VectorBT for base values calculation
            if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
                try:
                    # For other base calculations, use rolling mean as proxy
                    kst = self.vectorbt_optimizer.rolling_mean(base_values, window=max(self.roc4, self.sma4))
                    return kst
                except Exception as e:
                    self.logger.warning(f"VectorBT KST base calculation failed: {e}, using pandas fallback")
                    # For other base calculations, use rolling mean as proxy
                    kst = base_values.rolling(window=max(self.roc4, self.sma4)).mean()
                    return kst
            else:
                # For other base calculations, use rolling mean as proxy
                kst = base_values.rolling(window=max(self.roc4, self.sma4)).mean()
                return kst

# APO (Absolute Price Oscillator)
class APOGenerator(VectorizedFeatureGenerator):
    """Generator for APO (Absolute Price Oscillator) with different base calculations."""
    
    def __init__(self,
                 fast_period: int = 12,
                 slow_period: int = 26,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize APO generator.
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"apo_{fast_period}_{slow_period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Absolute Price Oscillator with fast={fast_period}, slow={slow_period} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=slow_period,
            min_lookback=slow_period,
            max_lookback=slow_period,
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate APO based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Use VectorBT for APO calculation
        if self.vectorbt_optimizer and self._should_use_vectorbt(base_values):
            try:
                # Calculate EMA using ewm
                ema_fast = base_values.ewm(span=self.fast_period).mean()
                ema_slow = base_values.ewm(span=self.slow_period).mean()
                
                # Calculate APO
                apo = ema_fast - ema_slow
                
                return apo
            except Exception as e:
                self.logger.warning(f"VectorBT APO calculation failed: {e}, using pandas fallback")
                # Calculate EMA
                ema_fast = base_values.ewm(span=self.fast_period).mean()
                ema_slow = base_values.ewm(span=self.slow_period).mean()
                
                # Calculate APO
                apo = ema_fast - ema_slow
                
                return apo
        else:
            # Calculate EMA
            ema_fast = base_values.ewm(span=self.fast_period).mean()
            ema_slow = base_values.ewm(span=self.slow_period).mean()
            
            # Calculate APO
            apo = ema_fast - ema_slow
            
            return apo

# CMO (Chande Momentum Oscillator)
class CMOGenerator(VectorizedFeatureGenerator):
    """Generator for CMO (Chande Momentum Oscillator) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize CMO generator.
        
        Args:
            period: CMO period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"cmo_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Chande Momentum Oscillator over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate CMO based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate momentum
        momentum = base_values.diff()
        
        # Calculate positive and negative momentum
        pos_momentum = momentum.where(momentum > 0, 0)
        neg_momentum = -momentum.where(momentum < 0, 0)
        
        # Calculate CMO
        pos_sum = pos_momentum.rolling(window=self.period).sum()
        neg_sum = neg_momentum.rolling(window=self.period).sum()
        
        cmo = 100 * (pos_sum - neg_sum) / (pos_sum + neg_sum)
        
        return cmo

# NATR (Normalized Average True Range)
class NATRGenerator(VectorizedFeatureGenerator):
    """Generator for NATR (Normalized Average True Range) with different base calculations."""
    
    def __init__(self,
                 period: int = 14,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize NATR generator.
        
        Args:
            period: NATR period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"natr_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Normalized Average True Range over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate NATR based on the specified base calculation."""
        if self.base_calculation == BaseCalculationType.PRICE_LEVELS:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate NATR
            atr = tr.rolling(window=self.period).mean()
            natr = 100 * atr / close
            
            return natr
        else:
            base_values = self.base_calculator.calculate(data)
            
            # For other base calculations, use rolling std as proxy
            natr = base_values.rolling(window=self.period).std()
            
            return natr

# PFE (Polarized Fractal Efficiency)class PFEGenerator(VectorizedFeatureGenerator):
    """Generator for PFE (Polarized Fractal Efficiency) with different base calculations."""
    
    def __init__(self,
                 period: int = 10,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize PFE generator.
        
        Args:
            period: PFE period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"pfe_{period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Polarized Fractal Efficiency over {period} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate PFE based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate PFE - OPTIMIZED: Vectorized PFE calculation
        # Pre-calculate differences and their norms
        diff_values = base_values.diff().fillna(0)
        diff_norms = np.sqrt(diff_values**2 + 1)
        
        # Vectorized PFE calculation
        numerator = np.sqrt((base_values - base_values.shift(self.period))**2 + self.period**2)
        denominator = diff_norms.rolling(window=self.period).sum()
        pfe = 100 * numerator / denominator
        
        return pfe

# T3 (T3 Moving Average)class T3Generator(VectorizedFeatureGenerator):
    """Generator for T3 (T3 Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 20,
                 volume_factor: float = 0.7,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize T3 generator.
        
        Args:
            period: T3 period
            volume_factor: T3 volume factor
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"t3_{period}_{volume_factor}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"T3 Moving Average with period={period}, volume_factor={volume_factor} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'volume_factor': volume_factor,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.volume_factor = volume_factor
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate T3 based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate T3 (simplified version)
        t3 = base_values.ewm(span=self.period).mean()
        
        return t3

# KAMA (Kaufman's Adaptive Moving Average)class KAMAGenerator(VectorizedFeatureGenerator):
    """Generator for KAMA (Kaufman's Adaptive Moving Average) with different base calculations."""
    
    def __init__(self,
                 period: int = 30,
                 fast_period: int = 2,
                 slow_period: int = 30,
                 base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.RETURNS_VWAP,
                 **base_kwargs):
        """
        Initialize KAMA generator.
        
        Args:
            period: KAMA period
            fast_period: Fast period
            slow_period: Slow period
            base_calculation: Base calculation type (price_returns, returns_vwap, etc.)
            **base_kwargs: Additional parameters for base calculation
        """
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        # Create base calculator - map period to lookback_period for base calculator
        base_kwargs_copy = base_kwargs.copy()
        if 'period' in base_kwargs_copy:
            base_kwargs_copy['lookback_period'] = base_kwargs_copy.pop('period')
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs_copy)
        
        # Update required columns based on base calculation
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"kama_{period}_{fast_period}_{slow_period}_{base_calculation.value}",
            category=FeatureCategory.OSCILLATOR,
            description=f"Kaufman's Adaptive Moving Average with period={period}, fast={fast_period}, slow={slow_period} based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                'period': period,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'base_calculation': base_calculation.value,
                **base_kwargs
            }
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.period = period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate KAMA based on the specified base calculation."""
        base_values = self.base_calculator.calculate(data)
        
        # Calculate KAMA (simplified version)
        kama = base_values.ewm(span=self.period).mean()
        
        return kama

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
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
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

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return self.vectorbt_optimizer.rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return self.vectorbt_optimizer.rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return self.vectorbt_optimizer.rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return self.vectorbt_optimizer.rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return self.vectorbt_optimizer.rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return self.vectorbt_optimizer.rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return self._calculate_sma_vectorized(data, window)
        elif operation == 'std':
            return self._calculate_rolling_std_vectorized(data, window)
        elif operation == 'var':
            return self._optimized_rolling_operation(data, "var", window)
        elif operation == 'min':
            return self._calculate_rolling_min_vectorized(data, window)
        elif operation == 'max':
            return self._calculate_rolling_max_vectorized(data, window)
        elif operation == 'sum':
            return self._calculate_rolling_sum_vectorized(data, window)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
