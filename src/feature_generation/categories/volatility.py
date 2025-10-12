"""
Advanced Volatility Feature Generator

This module provides feature generators for advanced volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Fully optimized with VectorBT for maximum performance.

Key Features:
- VectorBT-optimized volatility calculations
- Advanced volatility indicators
- Memory-efficient processing
- GPU acceleration support
- Comprehensive volatility analysis
"""

import numpy as np
import pandas as pd
import warnings
import logging
import time
from typing import Any, Dict, List, Optional, Union

# Import tprint for consistent logging
try:
    from tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

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
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
        ROLLING_OPTIMIZER_AVAILABLE = True
    except ImportError:
        ROLLING_OPTIMIZER_AVAILABLE = False
        get_vectorbt_rolling_optimizer = None
        VectorBTRollingOptimizer = None

# Optimization utilities - NOW USING NEW OPTIMIZED VERSION
try:
    from ..utils.unified_optimization_wrapper import (
        UnifiedOptimizationWrapper,
        UnifiedOptimizationConfig,
        OptimizationMode,
        create_unified_optimizer
    )
    from ..utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
    UNIFIED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from ..utils.vectorization_optimizer import get_vectorization_optimizer
        from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
        OPTIMIZATION_AVAILABLE = True
        UNIFIED_OPTIMIZATION_AVAILABLE = False
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        UNIFIED_OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (
    BaseCalculator,
    BaseCalculationType,
    BaseCalculationConfig,
    create_base_calculator
)

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

# Centralized utility imports
from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
from ...features_common.transforms.vectorbt_scaler import VectorBTScaler, create_vectorbt_scaler
from ..core.feature_bank import get_global_feature_bank

# Enhanced VectorBT integration
try:
    from .enhanced_vectorbt_volatility import (
        EnhancedVectorBTVolatilityGenerator, 
        VolatilityConfig, 
        create_enhanced_volatility_generators,
        create_default_enhanced_volatility_generators
    )
    ENHANCED_VECTORBT_AVAILABLE = True
except ImportError:
    ENHANCED_VECTORBT_AVAILABLE = False
    EnhancedVectorBTVolatilityGenerator = None
    VolatilityConfig = None
    create_enhanced_volatility_generators = None
    create_default_enhanced_volatility_generators = None

# Unified Vectorization Manager
try:
    from ...utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy, OperationConfig
    )
    UNIFIED_MANAGER_AVAILABLE = True
except ImportError:
    UNIFIED_MANAGER_AVAILABLE = False
    UnifiedVectorizationManager = None
    OperationType = None
    OptimizationStrategy = None
    OperationConfig = None

logger = logging.getLogger(__name__)

class VolatilityFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Advanced feature generator for volatility-based features with comprehensive VectorBT optimization."""

    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None, 
                 base_calculation: Optional[BaseCalculationType] = None,
                 enable_gpu: bool = True, 
                 enable_parallel: bool = True):
        self.period = period
        self.base_calculation = base_calculation
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optimization components (now using new optimized versions with same names)
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=enable_gpu, enable_parallel=enable_parallel)
        else:
            self.rolling_optimizer = None
        
        # Initialize unified optimizer if available
        if UNIFIED_OPTIMIZATION_AVAILABLE:
            self.optimization_config = UnifiedOptimizationConfig(
                mode=OptimizationMode.AUTO,
                enable_gpu=enable_gpu,
                enable_parallel=enable_parallel,
                performance_threshold=1000,
                enable_performance_monitoring=True
            )
            self.unified_optimizer = create_unified_optimizer(self.optimization_config)
        else:
            self.unified_optimizer = None
        
        # Performance tracking
        self.performance_stats = {
            'total_features_generated': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'total_generation_time': 0.0,
            'average_time_per_feature': 0.0,
            'memory_usage_mb': 0.0
        }
            
        # Initialize Unified Vectorization Manager
        if UNIFIED_MANAGER_AVAILABLE:
            self.unified_manager = UnifiedVectorizationManager()
        else:
            self.unified_manager = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"advanced_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Advanced volatility measure over {period} periods with VectorBT optimization",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={
                "period": period
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    @classmethod
    def create_default(cls) -> 'VolatilityFeatureGenerator':
        return cls()
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate volatility feature using comprehensive VectorBT optimization."""
        start_time = time.time()
        
        # Optimize DataFrame for processing using Unified Vectorization Manager
        if self.unified_manager:
            try:
                # Use Unified Vectorization Manager for data optimization
                optimized_data = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    data,
                    OperationConfig(
                        operation_type=OperationType.FEATURE_ENGINEERING,
                        data_size=len(data),
                        data_dimensions=data.shape,
                        memory_budget_mb=512.0
                    )
                )
                data = optimized_data.result
                self.performance_stats['optimized_operations'] += 1
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager data optimization failed: {e}, using original data")
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'volatility_{self.period}')

        close_prices = data['close'].astype(float)
        
        # Calculate returns
        returns = close_prices.pct_change().dropna()
        
        if len(returns) < self.period:
            return pd.Series(np.nan, index=data.index, name=f'volatility_{self.period}')
        
        # Use Unified Vectorization Manager for optimized rolling operations
        if self.unified_manager:
            try:
                # Use VectorBT rolling operations through Unified Vectorization Manager
                volatility_result = self.unified_manager.optimize_operation(
                    OperationType.TECHNICAL_INDICATORS,
                    {
                        'data': returns,
                        'operation': 'rolling_std',
                        'window': self.period,
                        'indicator_configs': {'rolling_std': {'window': self.period}}
                    },
                    OperationConfig(
                        operation_type=OperationType.TECHNICAL_INDICATORS,
                        data_size=len(returns),
                        data_dimensions=returns.shape,
                        memory_budget_mb=256.0
                    )
                )
                volatility = volatility_result.result
                self.performance_stats['optimized_operations'] += 1
                self.performance_stats['total_features_generated'] += 1
                
                # Align with original data index
                volatility = volatility.reindex(data.index)
                return volatility
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager volatility calculation failed: {e}, using VectorBT fallback")
                # Fallback to VectorBT rolling optimizer
                if self.rolling_optimizer:
                    try:
                        volatility = self.rolling_optimizer.rolling_std(returns, window=self.period)
                        self.performance_stats['optimized_operations'] += 1
                        volatility = volatility.reindex(data.index)
                        return volatility
                    except Exception as e2:
                        self.logger.warning(f"VectorBT rolling optimizer failed: {e2}, using direct VectorBT fallback")
                        self.performance_stats['fallback_operations'] += 1
                else:
                    self.performance_stats['fallback_operations'] += 1
        elif self.rolling_optimizer:
            try:
                # Check if it's the new consolidated optimizer
                if hasattr(self.rolling_optimizer, 'single_rolling_operation'):
                    # New consolidated optimizer
                    config = RollingOperationConfig(
                        operation=RollingOperationType.STD,
                        window=self.period
                    )
                    volatility = self.rolling_optimizer.single_rolling_operation(returns, config)
                else:
                    # Legacy optimizer
                    volatility = self.rolling_optimizer.rolling_std(returns, window=self.period)
                
                self.performance_stats['optimized_operations'] += 1
                self.performance_stats['total_features_generated'] += 1
                
                # Align with original data index
                volatility = volatility.reindex(data.index)
                return volatility
            except Exception as e:
                self.logger.warning(f"Rolling optimizer failed: {e}, using fallback")
                self.performance_stats['fallback_operations'] += 1
        
        # Fallback to VectorBT direct operations
        elif VECTORBT_AVAILABLE:
            try:
                volatility = rolling_std(returns, window=self.period)
                self.performance_stats['optimized_operations'] += 1
                # Align with original data index
                volatility = volatility.reindex(data.index)
                return volatility
            except Exception as e:
                self.logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
                self.performance_stats['fallback_operations'] += 1
        
        # Final fallback to pandas
        volatility = returns.rolling(window=self.period).std()
        return volatility.reindex(data.index)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {
            'generator_stats': self.performance_stats.copy(),
            'optimization_available': ROLLING_OPTIMIZER_AVAILABLE,
            'unified_optimization_available': UNIFIED_OPTIMIZATION_AVAILABLE
        }
        
        # Add rolling optimizer stats if available
        if self.rolling_optimizer and hasattr(self.rolling_optimizer, 'get_performance_stats'):
            report['rolling_optimizer_stats'] = self.rolling_optimizer.get_performance_stats()
        
        # Add unified optimizer stats if available
        if self.unified_optimizer and hasattr(self.unified_optimizer, 'get_performance_report'):
            report['unified_optimizer_stats'] = self.unified_optimizer.get_performance_report()
        
        return report
    
    def reset_performance_stats(self):
        """Reset all performance statistics."""
        self.performance_stats = {
            'total_features_generated': 0,
            'optimized_operations': 0,
            'fallback_operations': 0,
            'gpu_operations': 0,
            'batch_operations': 0,
            'total_generation_time': 0.0,
            'average_time_per_feature': 0.0,
            'memory_usage_mb': 0.0
        }
        
        if self.rolling_optimizer and hasattr(self.rolling_optimizer, 'reset_performance_stats'):
            self.rolling_optimizer.reset_performance_stats()
        
        if self.unified_optimizer and hasattr(self.unified_optimizer, 'reset_performance_stats'):
            self.unified_optimizer.reset_performance_stats()

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        if not data.empty:
            closes = data['close'].astype(float)
            history_window = max(self.period, 1)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'close_history': close_history
            }
            self.update_state(state_update)
    
    def generate_optimized_volatility_features(self, data: pd.DataFrame, 
                                             feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate multiple volatility features using optimized batch processing.
        
        Args:
            data: OHLCV data
            feature_configs: List of feature configuration dictionaries
            
        Returns:
            DataFrame with generated volatility features
        """
        if self.unified_manager:
            try:
                # Use Unified Vectorization Manager for batch processing
                batch_result = self.unified_manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {
                        'data': data,
                        'feature_configs': feature_configs,
                        'operation_type': 'volatility_batch'
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
                self.logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                # Fallback to individual processing
                return self._process_features_individually(data, feature_configs)
        elif self.rolling_optimizer:
            # Fallback to individual VectorBT operations
            results = {}
            for config in feature_configs:
                feature_name = config['name']
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})
                
                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'std')
                        window = params.get('window', self.period)
                        column = params.get('column', 'close')
                        
                        if column in data.columns:
                            # Calculate returns for volatility features
                            if column == 'close':
                                series_data = data[column].pct_change().dropna()
                            else:
                                series_data = data[column]
                            
                            if operation == 'std':
                                results[feature_name] = self.rolling_optimizer.rolling_std(series_data, window)
                            elif operation == 'var':
                                results[feature_name] = self.rolling_optimizer.rolling_var(series_data, window)
                            elif operation == 'mean':
                                results[feature_name] = self.rolling_optimizer.rolling_mean(series_data, window)
                            elif operation == 'min':
                                results[feature_name] = self.rolling_optimizer.rolling_min(series_data, window)
                            elif operation == 'max':
                                results[feature_name] = self.rolling_optimizer.rolling_max(series_data, window)
                            elif operation == 'sum':
                                results[feature_name] = self.rolling_optimizer.rolling_sum(series_data, window)
                    
                    elif feature_type == 'scaling':
                        method = params.get('method', 'zscore')
                        column = params.get('column', 'close')
                        
                        if column in data.columns:
                            series_data = data[column]
                            if method == 'zscore':
                                results[feature_name] = self.rolling_optimizer.rolling_apply(
                                    series_data, lambda x: (x - x.mean()) / x.std(), window=20
                                )
                            elif method == 'minmax':
                                results[feature_name] = self.rolling_optimizer.rolling_apply(
                                    series_data, lambda x: (x - x.min()) / (x.max() - x.min()), window=20
                                )
                    
                except Exception as e:
                    self.logger.warning(f"Volatility feature {feature_name} failed: {e}")
                    results[feature_name] = pd.Series(np.nan, index=data.index)
            
            return pd.DataFrame(results, index=data.index)
        else:
            # Fallback to pandas operations
            results = {}
            for config in feature_configs:
                feature_name = config['name']
                feature_type = config.get('type', 'rolling')
                params = config.get('params', {})
                
                try:
                    if feature_type == 'rolling':
                        operation = params.get('operation', 'std')
                        window = params.get('window', self.period)
                        column = params.get('column', 'close')
                        
                        if column in data.columns:
                            # Calculate returns for volatility features
                            if column == 'close':
                                series_data = data[column].pct_change().dropna()
                            else:
                                series_data = data[column]
                            
                            rolling_obj = series_data.rolling(window=window)
                            if operation == 'std':
                                results[feature_name] = rolling_obj.std()
                            elif operation == 'var':
                                results[feature_name] = rolling_obj.var()
                            elif operation == 'mean':
                                results[feature_name] = rolling_obj.mean()
                            elif operation == 'min':
                                results[feature_name] = rolling_obj.min()
                            elif operation == 'max':
                                results[feature_name] = rolling_obj.max()
                            elif operation == 'sum':
                                results[feature_name] = rolling_obj.sum()
                    
                except Exception as e:
                    self.logger.warning(f"Volatility feature {feature_name} failed: {e}")
                    results[feature_name] = pd.Series(np.nan, index=data.index)
            
            return pd.DataFrame(results, index=data.index)
    
    def _process_features_individually(self, data: pd.DataFrame, feature_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Process features individually as fallback when batch processing fails."""
        results = {}
        for config in feature_configs:
            feature_name = config['name']
            feature_type = config.get('type', 'rolling')
            params = config.get('params', {})
            
            try:
                if feature_type == 'rolling':
                    operation = params.get('operation', 'std')
                    window = params.get('window', self.period)
                    column = params.get('column', 'close')
                    
                    if column in data.columns:
                        # Calculate returns for volatility features
                        if column == 'close':
                            series_data = data[column].pct_change().dropna()
                        else:
                            series_data = data[column]
                        
                        if self.rolling_optimizer:
                            if operation == 'std':
                                results[feature_name] = self.rolling_optimizer.rolling_std(series_data, window)
                            elif operation == 'var':
                                results[feature_name] = self.rolling_optimizer.rolling_var(series_data, window)
                            elif operation == 'mean':
                                results[feature_name] = self.rolling_optimizer.rolling_mean(series_data, window)
                            elif operation == 'min':
                                results[feature_name] = self.rolling_optimizer.rolling_min(series_data, window)
                            elif operation == 'max':
                                results[feature_name] = self.rolling_optimizer.rolling_max(series_data, window)
                            elif operation == 'sum':
                                results[feature_name] = self.rolling_optimizer.rolling_sum(series_data, window)
                        else:
                            # Fallback to pandas
                            rolling_obj = series_data.rolling(window=window)
                            if operation == 'std':
                                results[feature_name] = rolling_obj.std()
                            elif operation == 'var':
                                results[feature_name] = rolling_obj.var()
                            elif operation == 'mean':
                                results[feature_name] = rolling_obj.mean()
                            elif operation == 'min':
                                results[feature_name] = rolling_obj.min()
                            elif operation == 'max':
                                results[feature_name] = rolling_obj.max()
                            elif operation == 'sum':
                                results[feature_name] = rolling_obj.sum()
                
            except Exception as e:
                self.logger.warning(f"Volatility feature {feature_name} failed: {e}")
                results[feature_name] = pd.Series(np.nan, index=data.index)
        
        return pd.DataFrame(results, index=data.index)


class VectorBTVolatilityFeatureGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized volatility feature generator with comprehensive indicators."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_volatility_comprehensive_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized comprehensive volatility features over {period} periods",
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
        """Generate comprehensive volatility features using VectorBT."""
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_volatility_{self.period}')
        
        try:
            # Calculate returns for volatility
            returns = data['close'].pct_change().dropna()
            
            if len(returns) < self.period:
                return pd.Series(np.nan, index=data.index, name=f'vectorbt_volatility_{self.period}')
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    # Calculate multiple volatility measures
                    volatility_std = self.rolling_optimizer.rolling_std(returns, window=self.period)
                    volatility_var = self.rolling_optimizer.rolling_var(returns, window=self.period)
                    
                    # Combine volatility measures
                    volatility = (volatility_std + volatility_var) / 2
                    
                    # Align with original data index
                    volatility = volatility.reindex(data.index)
                    return volatility
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility_std = rolling_std(returns, window=self.period)
                    volatility_var = rolling_var(returns, window=self.period)
                    
                    # Combine volatility measures
                    volatility = (volatility_std + volatility_var) / 2
                    
                    # Align with original data index
                    volatility = volatility.reindex(data.index)
                    return volatility
                except Exception as e:
                    self.logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = returns.rolling(window=self.period).std()
            return volatility.reindex(data.index)
            
        except Exception as e:
            self.logger.error(f"Error generating volatility features: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_volatility_{self.period}')


class VectorBTBollingerBandsGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Bollinger Bands generator."""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period, std_dev)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        self.std_dev = std_dev
    
    @classmethod
    def _create_default_config(cls, period: int = 20, std_dev: float = 2.0) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_bbands_{period}_{std_dev}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Bollinger Bands over {period} periods with {std_dev} std dev",
            required_columns=["close"],
            optional_columns=["high", "low", "open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period, "std_dev": std_dev},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Bollinger Bands features using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_bbands_{self.period}')
        
        # Generate Bollinger Bands using VectorBT
        bb_result = self._vectorbt_technical_indicator(data, 'bbands_percent', 
                                                     window=self.period, 
                                                     alpha=self.std_dev)
        
        return bb_result.rename(f'vectorbt_bbands_{self.period}')


class VectorBTAverageTrueRangeGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Average True Range generator."""
    
    def __init__(self, period: int = 14, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
    
    @classmethod
    def _create_default_config(cls, period: int = 14) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_atr_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Average True Range over {period} periods",
            required_columns=["high", "low", "close"],
            optional_columns=["open"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate ATR using VectorBT."""
        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_atr_{self.period}')
        
        # Generate ATR using VectorBT
        atr = self._vectorbt_technical_indicator(data, 'atr', window=self.period)
        
        return atr.rename(f'vectorbt_atr_{self.period}')


class VectorBTGarmanKlassVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Garman-Klass Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_garman_klass_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Garman-Klass Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Garman-Klass Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_garman_klass_volatility_{self.period}')
        
        try:
            # Calculate Garman-Klass volatility components
            log_hl = np.log(data['high'] / data['low'])
            log_co = np.log(data['close'] / data['open'])
            
            # Garman-Klass formula: 0.5 * (log(high/low))^2 - (2*log(2)-1) * (log(close/open))^2
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(gk_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(gk_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Garman-Klass calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = gk_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_garman_klass_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Garman-Klass volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_garman_klass_volatility_{self.period}')


class VectorBTParkinsonVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Parkinson Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_parkinson_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Parkinson Volatility over {period} periods",
            required_columns=["high", "low"],
            optional_columns=["open", "close"],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Parkinson Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['high', 'low']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_parkinson_volatility_{self.period}')
        
        try:
            # Calculate Parkinson volatility: (1/(4*ln(2))) * ln(high/low)^2
            log_hl = np.log(data['high'] / data['low'])
            parkinson_volatility = (1 / (4 * np.log(2))) * log_hl**2
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(parkinson_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(parkinson_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Parkinson calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = parkinson_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_parkinson_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Parkinson volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_parkinson_volatility_{self.period}')


class VectorBTRogersSatchellVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Rogers-Satchell Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_rogers_satchell_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Rogers-Satchell Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Rogers-Satchell Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rogers_satchell_volatility_{self.period}')
        
        try:
            # Calculate Rogers-Satchell volatility components
            log_ho = np.log(data['high'] / data['open'])
            log_hc = np.log(data['high'] / data['close'])
            log_lo = np.log(data['low'] / data['open'])
            log_lc = np.log(data['low'] / data['close'])
            
            # Rogers-Satchell formula
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(rs_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(rs_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Rogers-Satchell calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = rs_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_rogers_satchell_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Rogers-Satchell volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_rogers_satchell_volatility_{self.period}')


class VectorBTYangZhangVolatilityGenerator(VectorBTFeatureGenerator):
    """VectorBT-optimized Yang-Zhang Volatility generator."""
    
    def __init__(self, period: int = 20, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config(period)
        super().__init__(config, enable_gpu=True, enable_parallel=True)
        self.period = period
        
        # Initialize VectorBT rolling optimizer
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(enable_gpu=True, enable_parallel=True)
        else:
            self.rolling_optimizer = None
    
    @classmethod
    def _create_default_config(cls, period: int = 20) -> FeatureConfig:
        return FeatureConfig(
            name=f"vectorbt_yang_zhang_volatility_{period}",
            category=FeatureCategory.VOLATILITY,
            description=f"VectorBT-optimized Yang-Zhang Volatility over {period} periods",
            required_columns=["open", "high", "low", "close"],
            optional_columns=[],
            default_lookback=period,
            min_lookback=period,
            max_lookback=period,
            parameters={"period": period},
            matrix_optimized=True,
            gpu_accelerated=True
        )
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate Yang-Zhang Volatility using VectorBT."""
        if data.empty or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_yang_zhang_volatility_{self.period}')
        
        try:
            # Calculate Yang-Zhang volatility components
            # Overnight volatility
            log_co = np.log(data['close'] / data['open'])
            overnight_vol = log_co**2
            
            # Rogers-Satchell volatility (already calculated above)
            log_ho = np.log(data['high'] / data['open'])
            log_hc = np.log(data['high'] / data['close'])
            log_lo = np.log(data['low'] / data['open'])
            log_lc = np.log(data['low'] / data['close'])
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Garman-Klass volatility
            log_hl = np.log(data['high'] / data['low'])
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Yang-Zhang formula: overnight + Rogers-Satchell + Garman-Klass
            yz_volatility = overnight_vol + rs_volatility + gk_volatility
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_mean(yz_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_mean(yz_volatility, window=self.period)
                    volatility = np.sqrt(volatility)  # Convert variance to volatility
                    return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
                except Exception as e:
                    self.logger.warning(f"VectorBT Yang-Zhang calculation failed: {e}, using pandas fallback")
            
            # Final fallback to pandas
            volatility = yz_volatility.rolling(window=self.period).mean()
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            return volatility.rename(f'vectorbt_yang_zhang_volatility_{self.period}')
            
        except Exception as e:
            self.logger.error(f"Error generating Yang-Zhang volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'vectorbt_yang_zhang_volatility_{self.period}')



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

def create_default_volatility_generators() -> List[FeatureGenerator]:
    """Create default volatility feature generators with VectorBT optimization."""
    generators = []
    
    # Use enhanced VectorBT generators if available
    if ENHANCED_VECTORBT_AVAILABLE:
        return create_default_enhanced_volatility_generators()
    
    if VECTORBT_AVAILABLE:
        # VectorBT-optimized generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VectorBTVolatilityFeatureGenerator(period))
            generators.append(VectorBTAverageTrueRangeGenerator(period))
            
        # Bollinger Bands with different parameters
        for period in [20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                generators.append(VectorBTBollingerBandsGenerator(period, std_dev))
        
        # Advanced volatility indicators
        for period in [10, 14, 20, 30]:
            generators.append(VectorBTGarmanKlassVolatilityGenerator(period))
            generators.append(VectorBTParkinsonVolatilityGenerator(period))
            generators.append(VectorBTRogersSatchellVolatilityGenerator(period))
            generators.append(VectorBTYangZhangVolatilityGenerator(period))
    else:
        # Fallback to original generators
        for period in [10, 14, 20, 30, 50]:
            generators.append(VolatilityFeatureGenerator(period))
    
    return generators


def create_comprehensive_vectorbt_volatility_generators(
    periods: List[int] = [10, 14, 20, 30, 50],
    std_devs: List[float] = [1.5, 2.0, 2.5],
    enable_gpu: bool = False,
    enable_parallel: bool = True,
    use_unified_manager: bool = True
) -> List[FeatureGenerator]:
    """
    Create comprehensive volatility generators with full VectorBT optimization.
    
    This function provides the most advanced volatility feature generation
    with intelligent strategy selection and comprehensive VectorBT integration.
    
    Args:
        periods: List of periods for volatility calculations
        std_devs: List of standard deviations for Bollinger Bands
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        use_unified_manager: Use UnifiedVectorizationManager for optimization
        
    Returns:
        List of optimized volatility feature generators
    """
    generators = []
    
    # Use enhanced VectorBT generators if available
    if ENHANCED_VECTORBT_AVAILABLE:
        return create_enhanced_volatility_generators(
            periods=periods,
            std_devs=std_devs,
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel
        )
    
    # Fallback to standard VectorBT generators
    if VECTORBT_AVAILABLE:
        # Basic volatility generators
        for period in periods:
            generators.append(VectorBTVolatilityFeatureGenerator(period))
            generators.append(VectorBTAverageTrueRangeGenerator(period))
            
        # Bollinger Bands with different parameters
        for period in [20, 30] if 20 in periods or 30 in periods else periods:
            for std_dev in std_devs:
                generators.append(VectorBTBollingerBandsGenerator(period, std_dev))
        
        # Advanced volatility indicators
        for period in periods:
            generators.append(VectorBTGarmanKlassVolatilityGenerator(period))
            generators.append(VectorBTParkinsonVolatilityGenerator(period))
            generators.append(VectorBTRogersSatchellVolatilityGenerator(period))
            generators.append(VectorBTYangZhangVolatilityGenerator(period))
    else:
        # Fallback to original generators
        for period in periods:
            generators.append(VolatilityFeatureGenerator(period))
    
    return generators


def create_optimized_volatility_pipeline(
    data: pd.DataFrame,
    periods: List[int] = [10, 14, 20, 30, 50],
    std_devs: List[float] = [1.5, 2.0, 2.5],
    enable_gpu: bool = False,
    enable_parallel: bool = True,
    use_unified_manager: bool = True
) -> pd.DataFrame:
    """
    Create an optimized volatility feature pipeline using VectorBT.
    
    This function provides a high-level interface for generating comprehensive
    volatility features with automatic optimization strategy selection.
    
    Args:
        data: Input DataFrame with OHLCV data
        periods: List of periods for volatility calculations
        std_devs: List of standard deviations for Bollinger Bands
        enable_gpu: Enable GPU acceleration
        enable_parallel: Enable parallel processing
        use_unified_manager: Use UnifiedVectorizationManager for optimization
        
    Returns:
        DataFrame with comprehensive volatility features
    """
    if ENHANCED_VECTORBT_AVAILABLE:
        # Use enhanced VectorBT pipeline
        config = VolatilityConfig(
            period=max(periods),
            std_dev=max(std_devs),
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel,
            use_unified_manager=use_unified_manager
        )
        
        generator = EnhancedVectorBTVolatilityGenerator(config)
        return generator.generate_comprehensive_volatility_features(data)
    
    # Fallback to individual generators
    generators = create_comprehensive_vectorbt_volatility_generators(
        periods=periods,
        std_devs=std_devs,
        enable_gpu=enable_gpu,
        enable_parallel=enable_parallel,
        use_unified_manager=use_unified_manager
    )
    
    # Generate features using all generators
    features = {}
    for generator in generators:
        try:
            feature_result = generator._generate_feature(data)
            if isinstance(feature_result, pd.Series):
                features[feature_result.name] = feature_result
        except Exception as e:
            logger.warning(f"Generator {generator.__class__.__name__} failed: {e}")
    
    return pd.DataFrame(features, index=data.index)


def benchmark_volatility_optimizations(
    data: pd.DataFrame,
    periods: List[int] = [10, 20, 30],
    trials: int = 3
) -> Dict[str, Any]:
    """
    Benchmark different volatility optimization approaches.
    
    Args:
        data: Input DataFrame with OHLCV data
        periods: List of periods to test
        trials: Number of trials for each approach
        
    Returns:
        Dictionary with benchmarking results
    """
    results = {
        'enhanced_vectorbt': {},
        'standard_vectorbt': {},
        'pandas_fallback': {},
        'unified_manager': {}
    }
    
    # Test enhanced VectorBT if available
    if ENHANCED_VECTORBT_AVAILABLE:
        try:
            config = VolatilityConfig(period=20, enable_gpu=False, enable_parallel=True)
            generator = EnhancedVectorBTVolatilityGenerator(config)
            
            times = []
            for _ in range(trials):
                start_time = time.time()
                _ = generator._generate_feature(data)
                times.append(time.time() - start_time)
            
            results['enhanced_vectorbt'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        except Exception as e:
            logger.warning(f"Enhanced VectorBT benchmark failed: {e}")
    
    # Test standard VectorBT
    if VECTORBT_AVAILABLE:
        try:
            generator = VectorBTVolatilityFeatureGenerator(period=20)
            
            times = []
            for _ in range(trials):
                start_time = time.time()
                _ = generator._generate_feature(data)
                times.append(time.time() - start_time)
            
            results['standard_vectorbt'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        except Exception as e:
            logger.warning(f"Standard VectorBT benchmark failed: {e}")
    
    # Test pandas fallback
    try:
        generator = VolatilityFeatureGenerator(period=20)
        
        times = []
        for _ in range(trials):
            start_time = time.time()
            _ = generator._generate_feature(data)
            times.append(time.time() - start_time)
        
        results['pandas_fallback'] = {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    except Exception as e:
        logger.warning(f"Pandas fallback benchmark failed: {e}")
    
    # Test unified manager if available
    if UNIFIED_MANAGER_AVAILABLE:
        try:
            manager = UnifiedVectorizationManager()
            config = OperationConfig(
                operation_type=OperationType.FEATURE_ENGINEERING,
                data_size=len(data),
                data_dimensions=data.shape
            )
            
            times = []
            for _ in range(trials):
                start_time = time.time()
                _ = manager.optimize_operation(
                    OperationType.FEATURE_ENGINEERING,
                    {'close': data['close'], 'period': 20},
                    config
                )
                times.append(time.time() - start_time)
            
            results['unified_manager'] = {
                'avg_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }
        except Exception as e:
            logger.warning(f"Unified manager benchmark failed: {e}")
    
    return results