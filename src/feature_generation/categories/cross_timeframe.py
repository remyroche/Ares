"""
Advanced Cross-Timeframe Feature Generators

This module provides feature generators for advanced cross-timeframe analysis,
capturing relationships and patterns across different time horizons.
Fully optimized with VectorBT for maximum performance.

Key Features:
- VectorBT-optimized cross-timeframe calculations
- Advanced multi-timeframe analysis
- Memory-efficient processing
- GPU acceleration support
- Comprehensive cross-timeframe indicators
- UnifiedVectorizationManager integration
- Advanced performance monitoring
"""

import numpy as np
import pandas as pd
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
from scipy import stats

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for optimization
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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Unified Vectorization Manager
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer, VectorizationOptimizer, VectorizationConfig
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    get_vectorization_optimizer = None
    VectorizationOptimizer = None
    VectorizationConfig = None

# Optimization utilities
try:
    from ..utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

from ..base_calculations import (
    BaseCalculationType,
    create_base_calculator
)

logger = logging.getLogger(__name__)

class CrossTimeframeFeatureGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """Advanced feature generator for cross-timeframe features with VectorBT optimization."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize VectorBT rolling optimizer with enhanced settings
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=True, 
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=5000
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            vectorization_config = VectorizationConfig(
                chunk_size=10000,
                enable_gpu_acceleration=True,
                enable_parallel=True,
                vectorization_strategy="aggressive",
                enable_memory_pooling=True
            )
            self.vectorization_manager = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_manager = None
        
        # Initialize Unified Vectorization Manager from ml_common
        try:
            from ...utils.ml_common.unified_vectorization_manager import (
                get_unified_vectorization_manager, UnifiedVectorizationManager, 
                OperationType, OptimizationStrategy, OperationConfig
            )
            self.unified_manager = get_unified_vectorization_manager()
            self.UNIFIED_MANAGER_AVAILABLE = True
        except ImportError:
            self.unified_manager = None
            self.UNIFIED_MANAGER_AVAILABLE = False
        
        # Enhanced performance tracking
        self.performance_stats = {
            'vectorbt_operations': 0,
            'unified_vectorization_operations': 0,
            'chunked_operations': 0,
            'gpu_operations': 0,
            'memory_optimizations': 0,
            'total_operations': 0,
            'total_execution_time': 0.0,
            'cross_timeframe_features_generated': 0
        }
    
    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="advanced_cross_timeframe_features",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description="Advanced cross-timeframe features across multiple time horizons with VectorBT optimization",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=30,
            min_lookback=5,
            max_lookback=100,
            parameters={
                "timeframes": [1, 5, 15, 30, 60],
                "feature_types": ["momentum", "volatility", "volume", "trend", "range"],
                "lag_handling": True,
                "fractional_changes": True,
                "learned_projections": True,
                "regime_aware": True,
                "alignment_methods": ["lag", "resample", "interpolate"],
                "projection_methods": ["pca", "autoencoder", "patchtst"]
            },
            matrix_optimized=True,
            gpu_accelerated=True
        )

    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing using Unified Vectorization Manager."""
        if self.vectorization_manager:
            return self.vectorization_manager.optimize_dataframe_processing(data)
        elif hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        # Fallback to parent class method
        return super().optimize_dataframe_processing(data)
    
    def vectorized_rolling_operations(self, data: pd.DataFrame, operations: List[str], 
                                    windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform vectorized rolling operations with hardware optimization."""
        if self.vectorization_manager:
            return self.vectorization_manager.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        elif hasattr(self, 'vectorization_optimizer') and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        # Fallback to parent class method
        return super().vectorized_rolling_operations(data, operations, windows, columns)
    
    def generate_enhanced_cross_timeframe_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate comprehensive cross-timeframe features using VectorBT optimization."""
        import time
        start_time = time.time()
        
        features = {}
        
        try:
            # Optimize data for processing
            optimized_data = self.optimize_dataframe_processing(data)
            
            # Use Unified Vectorization Manager for batch processing if available
            if hasattr(self, 'unified_manager') and self.unified_manager and len(data) > 1000:
                try:
                    batch_result = self.unified_manager.optimize_operation(
                        OperationType.FEATURE_ENGINEERING,
                        {
                            'data': optimized_data,
                            'operation_type': 'cross_timeframe_batch',
                            'timeframes': self.config.parameters.get("timeframes", [1, 5, 15, 30, 60]),
                            'feature_types': self.config.parameters.get("feature_types", ["momentum", "volatility", "volume", "trend", "range"])
                        },
                        OperationConfig(
                            operation_type=OperationType.FEATURE_ENGINEERING,
                            data_size=len(optimized_data),
                            data_dimensions=optimized_data.shape,
                            memory_budget_mb=2048.0
                        )
                    )
                    features.update(batch_result.result)
                    self.performance_stats['unified_manager_operations'] = self.performance_stats.get('unified_manager_operations', 0) + 1
                except Exception as e:
                    logger.warning(f"Unified Vectorization Manager batch processing failed: {e}, using fallback")
                    # Fallback to individual processing
                    features.update(self._generate_vectorbt_optimized_features(optimized_data, **kwargs))
                    features.update(self._generate_unified_vectorization_features(optimized_data, **kwargs))
            else:
                # Generate features using VectorBT rolling optimizer
                if self.rolling_optimizer:
                    features.update(self._generate_vectorbt_optimized_features(optimized_data, **kwargs))
                    self.performance_stats['vectorbt_operations'] += 1
                
                # Generate features using unified vectorization manager
                if self.vectorization_manager:
                    features.update(self._generate_unified_vectorization_features(optimized_data, **kwargs))
                    self.performance_stats['unified_vectorization_operations'] += 1
            
            # Update performance stats
            self.performance_stats['total_execution_time'] += time.time() - start_time
            self.performance_stats['total_operations'] += 1
            self.performance_stats['cross_timeframe_features_generated'] += len(features)
            
            logger.info(f"Generated {len(features)} cross-timeframe features in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error generating enhanced cross-timeframe features: {e}")
        
        return features
    
    def _generate_vectorbt_optimized_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features using VectorBT rolling optimizer."""
        features = {}
        
        try:
            timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
            
            for tf in timeframes:
                # Momentum features
                if 'close' in data.columns:
                    momentum = self.rolling_optimizer.rolling_apply(
                        data['close'], 
                        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
                        window=tf
                    )
                    features[f'vectorbt_momentum_{tf}'] = momentum
                
                # Volatility features
                if 'close' in data.columns:
                    returns = data['close'].pct_change().fillna(0)
                    volatility = self.rolling_optimizer.rolling_std(returns, window=tf)
                    features[f'vectorbt_volatility_{tf}'] = volatility
                
                # Volume features
                if 'volume' in data.columns:
                    volume_ma = self.rolling_optimizer.rolling_mean(data['volume'], window=tf)
                    features[f'vectorbt_volume_ma_{tf}'] = volume_ma
                
        except Exception as e:
            logger.warning(f"VectorBT optimized features generation failed: {e}")
        
        return features
    
    def _generate_unified_vectorization_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, pd.Series]:
        """Generate features using unified vectorization manager."""
        features = {}
        
        try:
            # Use unified vectorization for complex multi-timeframe operations
            operations = ['mean', 'std', 'var', 'min', 'max']
            windows = [5, 15, 30, 60]
            
            vectorized_result = self.vectorization_manager.vectorized_rolling_operations(
                data, operations, windows, ['close'] if 'close' in data.columns else None
            )
            
            # Extract features from vectorized result
            for col in vectorized_result.columns:
                if col != 'close':  # Skip original column
                    features[f'unified_{col}'] = vectorized_result[col]
                
        except Exception as e:
            logger.warning(f"Unified vectorization features generation failed: {e}")
        
        return features
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if self.vectorization_manager:
            vectorization_report = self.vectorization_manager.get_performance_report()
        else:
            vectorization_report = {}
        
        if self.rolling_optimizer:
            rolling_stats = self.rolling_optimizer.get_performance_stats()
        else:
            rolling_stats = {}
        
        return {
            'cross_timeframe_performance': self.performance_stats.copy(),
            'vectorization_report': vectorization_report,
            'rolling_optimizer_stats': rolling_stats,
            'optimization_availability': {
                'vectorbt_available': VECTORBT_AVAILABLE,
                'rolling_optimizer_available': ROLLING_OPTIMIZER_AVAILABLE,
                'unified_vectorization_available': UNIFIED_VECTORIZATION_AVAILABLE
            }
        }

# Cross-Timeframe Momentum Generator

class CrossTimeframeMomentumGenerator(FeatureGenerator, VectorBTOptimizationMixin):
    """Generator for cross-timeframe momentum features with VectorBT optimization."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_momentum_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe momentum over {timeframe} periods based on {base_calculation.value} with VectorBT optimization",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
        
        # Initialize VectorBT rolling optimizer with enhanced settings
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=True, 
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=2000
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            vectorization_config = VectorizationConfig(
                chunk_size=5000,
                enable_gpu_acceleration=True,
                enable_parallel=True,
                vectorization_strategy="aggressive"
            )
            self.vectorization_manager = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe momentum using VectorBT optimization."""
        import time
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe_processing(data)
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'ctf_momentum_{self.timeframe}')

        try:
            base_values = self.base_calculator.calculate(data)
            
            if base_values.empty:
                return pd.Series(dtype=float, index=data.index, name=f'ctf_momentum_{self.timeframe}')
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    def momentum_func(x):
                        if len(x) < 2 or x.iloc[0] == 0:
                            return 0.0
                        return (x.iloc[-1] - x.iloc[0]) / x.iloc[0]
                    
                    momentum = self.rolling_optimizer.rolling_apply(base_values, window=self.timeframe, func=momentum_func)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return momentum
                except Exception as e:
                    logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    momentum = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.timeframe)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return momentum
                except Exception as e:
                    logger.warning(f"VectorBT momentum calculation failed: {e}, using pandas fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Final fallback to pandas
            momentum = base_values.rolling(window=self.timeframe).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0)
            return momentum
            
        except Exception as e:
            logger.error(f"Error generating cross-timeframe momentum: {e}")
            return pd.Series(np.nan, index=data.index, name=f'ctf_momentum_{self.timeframe}')
        finally:
            if hasattr(self, 'performance_stats'):
                self.performance_stats['total_execution_time'] += time.time() - start_time

# Cross-Timeframe Volatility Generator
    

class CrossTimeframeVolatilityGenerator(FeatureGenerator, VectorBTOptimizationMixin):
    """Generator for cross-timeframe volatility features with VectorBT optimization."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_volatility_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe volatility over {timeframe} periods based on {base_calculation.value} with VectorBT optimization",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs},
            matrix_optimized=True,
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
        
        # Initialize VectorBT rolling optimizer with enhanced settings
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=True, 
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=2000
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            vectorization_config = VectorizationConfig(
                chunk_size=5000,
                enable_gpu_acceleration=True,
                enable_parallel=True,
                vectorization_strategy="aggressive"
            )
            self.vectorization_manager = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe volatility using VectorBT optimization."""
        import time
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe_processing(data)
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'ctf_volatility_{self.timeframe}')

        try:
            base_values = self.base_calculator.calculate(data)
            
            if base_values.empty:
                return pd.Series(dtype=float, index=data.index, name=f'ctf_volatility_{self.timeframe}')
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volatility = self.rolling_optimizer.rolling_std(base_values, window=self.timeframe)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return volatility
                except Exception as e:
                    logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volatility = rolling_std(base_values, window=self.timeframe)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return volatility
                except Exception as e:
                    logger.warning(f"VectorBT volatility calculation failed: {e}, using pandas fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Final fallback to pandas
            volatility = base_values.rolling(window=self.timeframe).std()
            return volatility
            
        except Exception as e:
            logger.error(f"Error generating cross-timeframe volatility: {e}")
            return pd.Series(np.nan, index=data.index, name=f'ctf_volatility_{self.timeframe}')
        finally:
            if hasattr(self, 'performance_stats'):
                self.performance_stats['total_execution_time'] += time.time() - start_time

# Cross-Timeframe Volume Generator
    

class CrossTimeframeVolumeGenerator(FeatureGenerator, VectorBTOptimizationMixin):
    """Generator for cross-timeframe volume features with VectorBT optimization."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.VOLUME_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_volume_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe volume over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs},
            gpu_accelerated=True
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
        
        # Initialize VectorBT rolling optimizer with enhanced settings
        if ROLLING_OPTIMIZER_AVAILABLE:
            self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                enable_gpu=True, 
                enable_parallel=True,
                memory_efficient=True,
                chunk_size=2000
            )
        else:
            self.rolling_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE:
            vectorization_config = VectorizationConfig(
                chunk_size=5000,
                enable_gpu_acceleration=True,
                enable_parallel=True,
                vectorization_strategy="aggressive"
            )
            self.vectorization_manager = get_vectorization_optimizer(vectorization_config)
        else:
            self.vectorization_manager = None
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate cross-timeframe volume using VectorBT optimization."""
        import time
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if self.vectorization_manager:
            data = self.vectorization_manager.optimize_dataframe_processing(data)
        elif hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty:
            return pd.Series(dtype=float, index=data.index, name=f'ctf_volume_{self.timeframe}')

        try:
            base_values = self.base_calculator.calculate(data)
            
            if base_values.empty:
                return pd.Series(dtype=float, index=data.index, name=f'ctf_volume_{self.timeframe}')
            
            # Use VectorBT rolling optimizer if available
            if self.rolling_optimizer:
                try:
                    volume_ma = self.rolling_optimizer.rolling_mean(base_values, window=self.timeframe)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return volume_ma
                except Exception as e:
                    logger.warning(f"VectorBT rolling optimizer failed: {e}, using fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Fallback to VectorBT direct operations
            if VECTORBT_AVAILABLE:
                try:
                    volume_ma = rolling_mean(base_values, window=self.timeframe)
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['vectorbt_operations'] += 1
                    return volume_ma
                except Exception as e:
                    logger.warning(f"VectorBT volume calculation failed: {e}, using pandas fallback")
                    if hasattr(self, 'performance_stats'):
                        self.performance_stats['pandas_fallbacks'] += 1
            
            # Final fallback to pandas
            volume_ma = base_values.rolling(window=self.timeframe).mean()
            return volume_ma
            
        except Exception as e:
            logger.error(f"Error generating cross-timeframe volume: {e}")
            return pd.Series(np.nan, index=data.index, name=f'ctf_volume_{self.timeframe}')
        finally:
            if hasattr(self, 'performance_stats'):
                self.performance_stats['total_execution_time'] += time.time() - start_time

# Cross-Timeframe Trend Generator
    

class CrossTimeframeTrendGenerator(FeatureGenerator):
    """Generator for cross-timeframe trend features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_trend_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe trend over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe trend."""
        base_values = self.base_calculator.calculate(data)
        
        def calculate_trend_strength(series):
            if len(series) < 2:
                return 0.0
            try:
                # Calculate linear regression slope
                slope = np.polyfit(range(len(series)), series, 1)[0]
                return slope
            except:
                return 0.0
        
        trend = rolling_apply(base_values, calculate_trend_strength, window=self.timeframe)
        return trend

# Cross-Timeframe High-Low Generator
    

class CrossTimeframeHighLowGenerator(FeatureGenerator):
    """Generator for cross-timeframe high-low range features."""
    
    def __init__(self, timeframe: int = 5, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_LEVELS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = ["high", "low"]
        
        config = FeatureConfig(
            name=f"ctf_{timeframe}m_hl_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe high-low range over {timeframe} periods based on {base_calculation.value}",
            required_columns=required_columns,
            default_lookback=timeframe,
            min_lookback=timeframe,
            max_lookback=timeframe,
            parameters={'timeframe': timeframe, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe = timeframe
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe high-low range using VectorBT."""
        hl_range = rolling_mean(data['high'] - data['low'], window=self.timeframe)
        return hl_range

# Cross-Timeframe Ratio Generator
    

class CrossTimeframeRatioGenerator(FeatureGenerator):
    """Generator for cross-timeframe ratio features."""
    
    def __init__(self, short_timeframe: int = 5, long_timeframe: int = 20, feature_type: str = "momentum", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_ratio_{feature_type}_{short_timeframe}_{long_timeframe}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe {feature_type} ratio between {short_timeframe} and {long_timeframe} periods",
            required_columns=required_columns,
            default_lookback=long_timeframe,
            min_lookback=long_timeframe,
            max_lookback=long_timeframe,
            parameters={'short_timeframe': short_timeframe, 'long_timeframe': long_timeframe, 'feature_type': feature_type, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe ratio using VectorBT."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            short_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.short_timeframe)
            long_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.long_timeframe)
        elif self.feature_type == "volatility":
            short_feature = rolling_std(base_values, window=self.short_timeframe)
            long_feature = rolling_std(base_values, window=self.long_timeframe)
        elif self.feature_type == "sma":
            short_feature = rolling_mean(base_values, window=self.short_timeframe)
            long_feature = rolling_mean(base_values, window=self.long_timeframe)
        else:  # Default to momentum
            short_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.short_timeframe)
            long_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.long_timeframe)
        
        # Calculate ratio with safe division
        ratio = short_feature / (long_feature + 1e-8)  # Add small epsilon to prevent division by zero
        return ratio

# Cross-Timeframe Correlation Generator
    

class CrossTimeframeCorrelationGenerator(FeatureGenerator):
    """Generator for cross-timeframe correlation features."""
    
    def __init__(self, timeframe1: int = 5, timeframe2: int = 15, feature_type: str = "momentum", correlation_window: int = 20, base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_corr_{feature_type}_{timeframe1}_{timeframe2}_{correlation_window}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe correlation of {feature_type} between {timeframe1} and {timeframe2} periods over {correlation_window} window",
            required_columns=required_columns,
            default_lookback=max(timeframe1, timeframe2, correlation_window),
            min_lookback=max(timeframe1, timeframe2, correlation_window),
            max_lookback=max(timeframe1, timeframe2, correlation_window),
            parameters={'timeframe1': timeframe1, 'timeframe2': timeframe2, 'feature_type': feature_type, 'correlation_window': correlation_window, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframe1 = timeframe1
        self.timeframe2 = timeframe2
        self.feature_type = feature_type
        self.correlation_window = correlation_window
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe correlation using VectorBT."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            feature1 = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.timeframe1)
            feature2 = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.timeframe2)
        elif self.feature_type == "volatility":
            feature1 = rolling_std(base_values, window=self.timeframe1)
            feature2 = rolling_std(base_values, window=self.timeframe2)
        else:  # Default to momentum
            feature1 = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.timeframe1)
            feature2 = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.timeframe2)
        
        # Calculate rolling correlation using VectorBT
        correlation = rolling_corr(feature1, feature2, window=self.correlation_window)
        return correlation

# Cross-Timeframe Divergence Generator
    

class CrossTimeframeDivergenceGenerator(FeatureGenerator):
    """Generator for cross-timeframe divergence features."""
    
    def __init__(self, short_timeframe: int = 5, long_timeframe: int = 20, feature_type: str = "momentum", base_calculation: Union[str, BaseCalculationType] = BaseCalculationType.PRICE_RETURNS, **base_kwargs):
        if isinstance(base_calculation, str):
            base_calculation = BaseCalculationType(base_calculation)
        
        self.base_calculator = create_base_calculator(base_calculation, **base_kwargs)
        required_columns = self.base_calculator.get_required_columns()
        
        config = FeatureConfig(
            name=f"ctf_divergence_{feature_type}_{short_timeframe}_{long_timeframe}_{base_calculation.value}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Cross-timeframe {feature_type} divergence between {short_timeframe} and {long_timeframe} periods",
            required_columns=required_columns,
            default_lookback=long_timeframe,
            min_lookback=long_timeframe,
            max_lookback=long_timeframe,
            parameters={'short_timeframe': short_timeframe, 'long_timeframe': long_timeframe, 'feature_type': feature_type, 'base_calculation': base_calculation.value, **base_kwargs}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.short_timeframe = short_timeframe
        self.long_timeframe = long_timeframe
        self.feature_type = feature_type
        self.base_calculation = base_calculation
    
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate cross-timeframe divergence using VectorBT."""
        base_values = self.base_calculator.calculate(data)
        
        if self.feature_type == "momentum":
            short_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.short_timeframe)
            long_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.long_timeframe)
        elif self.feature_type == "volatility":
            short_feature = rolling_std(base_values, window=self.short_timeframe)
            long_feature = rolling_std(base_values, window=self.long_timeframe)
        elif self.feature_type == "sma":
            short_feature = rolling_mean(base_values, window=self.short_timeframe)
            long_feature = rolling_mean(base_values, window=self.long_timeframe)
        else:  # Default to momentum
            short_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.short_timeframe)
            long_feature = rolling_apply(base_values, lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.long_timeframe)
        
        # Calculate divergence (difference)
        divergence = short_feature - long_feature
        return divergence

def create_cross_timeframe_generators() -> List[FeatureGenerator]:
    """Create all cross-timeframe feature generators."""
    generators = []
    
    # Cross-timeframe momentum for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeMomentumGenerator(timeframe=timeframe))
    
    # Cross-timeframe volatility for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeVolatilityGenerator(timeframe=timeframe))
    
    # Cross-timeframe volume for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeVolumeGenerator(timeframe=timeframe))
    
    # Cross-timeframe trend for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeTrendGenerator(timeframe=timeframe))
    
    # Cross-timeframe high-low for different timeframes
    for timeframe in [5, 15, 30]:
        generators.append(CrossTimeframeHighLowGenerator(timeframe=timeframe))
    
    # Cross-timeframe ratios
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=5, long_timeframe=20, feature_type="momentum"))
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=5, long_timeframe=20, feature_type="volatility"))
    generators.append(CrossTimeframeRatioGenerator(short_timeframe=10, long_timeframe=50, feature_type="sma"))
    
    # Cross-timeframe correlations
    generators.append(CrossTimeframeCorrelationGenerator(timeframe1=5, timeframe2=15, feature_type="momentum", correlation_window=20))
    generators.append(CrossTimeframeCorrelationGenerator(timeframe1=15, timeframe2=30, feature_type="volatility", correlation_window=20))
    
    # Cross-timeframe divergences
    generators.append(CrossTimeframeDivergenceGenerator(short_timeframe=5, long_timeframe=20, feature_type="momentum"))
    generators.append(CrossTimeframeDivergenceGenerator(short_timeframe=5, long_timeframe=20, feature_type="volatility"))
    
    return generators

def create_default_cross_timeframe_generators() -> List[FeatureGenerator]:
    """Create default set of cross-timeframe generators."""
    return create_cross_timeframe_generators()

# Enhanced Cross-Timeframe Generators for Better Aggregation

    

class CrossTimeframeFractionalChangeGenerator(FeatureGenerator):
    """Generator for fractional change features across timeframes."""

    def __init__(self, fast_tf: int = 5, slow_tf: int = 15, feature_type: str = "volatility"):
        config = FeatureConfig(
            name=f"ctf_fractional_{feature_type}_{fast_tf}m_{slow_tf}m",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Fractional change of {feature_type} from {fast_tf}m to {slow_tf}m timeframe",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=max(fast_tf, slow_tf),
            min_lookback=max(fast_tf, slow_tf),
            max_lookback=max(fast_tf, slow_tf),
            parameters={"fast_tf": fast_tf, "slow_tf": slow_tf, "feature_type": feature_type}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.fast_tf = fast_tf
        self.slow_tf = slow_tf
        self.feature_type = feature_type

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate fractional change feature across timeframes using VectorBT."""
        if self.feature_type == "volatility":
            returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
            fast_vol = rolling_std(returns, window=self.fast_tf)
            slow_vol = rolling_std(returns, window=self.slow_tf)
            fractional_change = fast_vol / (slow_vol + 1e-8)
        elif self.feature_type == "momentum":
            fast_momentum = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.fast_tf)
            slow_momentum = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=self.slow_tf)
            fractional_change = fast_momentum / (slow_momentum + 1e-8)
        elif self.feature_type == "volume":
            if "volume" in data.columns:
                fast_volume = rolling_mean(data["volume"], window=self.fast_tf)
                slow_volume = rolling_mean(data["volume"], window=self.slow_tf)
                fractional_change = fast_volume / (slow_volume + 1e-8)
            else:
                fractional_change = pd.Series(np.zeros(len(data)), index=data.index)
        else:
            fractional_change = pd.Series(np.zeros(len(data)), index=data.index)

        return fractional_change.fillna(0)


    

class CrossTimeframeAlignmentGenerator(FeatureGenerator):
    """Generator for properly aligned cross-timeframe features."""

    def __init__(self, source_tf: int = 1, target_tf: int = 5, alignment_method: str = "lag"):
        config = FeatureConfig(
            name=f"ctf_aligned_{source_tf}m_to_{target_tf}m",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Align {source_tf}m features to {target_tf}m timeframe using {alignment_method}",
            required_columns=["close"],
            default_lookback=target_tf,
            min_lookback=target_tf,
            max_lookback=target_tf,
            parameters={"source_tf": source_tf, "target_tf": target_tf, "alignment_method": alignment_method}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.source_tf = source_tf
        self.target_tf = target_tf
        self.alignment_method = alignment_method

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate properly aligned cross-timeframe feature."""
        # Calculate lag needed for alignment
        lag_bars = self.target_tf // self.source_tf - 1

        if self.alignment_method == "lag":
            # Lag fast timeframe features by appropriate number of bars using VectorBT
            returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
            aligned_returns = rolling_apply(returns, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)
            return aligned_returns.fillna(0)
        elif self.alignment_method == "resample":
            # Resample to target timeframe using VectorBT
            resampled = data["close"].resample(f'{self.target_tf}min').last()
            # Forward fill to original frequency
            aligned = resampled.reindex(data.index, method='ffill')
            return rolling_apply(aligned, lambda x: (x.iloc[-1] / x.iloc[0] - 1) if x.iloc[0] != 0 else 0, window=2).fillna(0)
        else:
            return pd.Series(np.zeros(len(data)), index=data.index)


    

class CrossTimeframeLearnedProjectionGenerator(FeatureGenerator):
    """Generator for learned projections across timeframes using PCA/dimensionality reduction."""

    def __init__(self, timeframes: List[int] = [1, 5, 15], n_components: int = 3):
        config = FeatureConfig(
            name=f"ctf_learned_projection_{'_'.join(map(str, timeframes))}_{n_components}",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description=f"Learned projection across {timeframes} timeframes using {n_components} components",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=max(timeframes) * 10,
            min_lookback=max(timeframes) * 5,
            max_lookback=max(timeframes) * 20,
            parameters={"timeframes": timeframes, "n_components": n_components}
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.timeframes = timeframes
        self.n_components = n_components

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate learned projection features across timeframes."""
        try:
            from sklearn.decomposition import PCA

            # Create features for each timeframe
            tf_features = []
            for tf in self.timeframes:
                # Calculate returns for this timeframe using VectorBT
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=tf)

                # Calculate volatility for this timeframe using VectorBT
                volatility = rolling_std(rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1), window=tf)

                # Calculate momentum for this timeframe using VectorBT
                momentum = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=tf * 5)

                tf_features.append(pd.concat([returns, volatility, momentum], axis=1))

            # Combine features from all timeframes
            feature_matrix = pd.concat(tf_features, axis=1).fillna(0)

            # Apply PCA for dimensionality reduction
            if len(feature_matrix.columns) >= self.n_components:
                pca = PCA(n_components=self.n_components)
                pca_result = pca.fit_transform(feature_matrix)

                # Return first principal component as representative feature
                return pd.Series(pca_result[:, 0], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.warning(f"Error in learned projection: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)


# Enhanced Cross-Timeframe Features

    

class EnhancedCrossTimeframeFeatureGenerator(VectorizedFeatureGenerator):
    """Enhanced cross-timeframe feature generator with proper lag handling and fractional changes."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        if config is None:
            config = self._create_default_config()
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)

    @classmethod
    def _create_default_config(cls) -> FeatureConfig:
        return FeatureConfig(
            name="enhanced_cross_timeframe_features",
            category=FeatureCategory.CROSS_TIMEFRAME,
            description="Enhanced cross-timeframe features with proper lag handling and learned projections",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=100,
            min_lookback=50,
            max_lookback=500,
            parameters={
                "timeframes": [1, 5, 15, 30, 60],
                "feature_types": ["momentum", "volatility", "volume", "trend", "range"],
                "lag_handling": True,
                "fractional_changes": True,
                "learned_projections": True,
                "regime_aware": True,
                "alignment_methods": ["lag", "resample", "interpolate"],
                "projection_methods": ["pca", "autoencoder", "patchtst"]
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        """Generate enhanced cross-timeframe features."""
        try:
            # Generate all enhanced cross-timeframe features
            features_dict = self.generate_enhanced_cross_timeframe_features(data, **kwargs)

            # Return first feature as representative for base class
            if features_dict:
                first_feature_name = list(features_dict.keys())[0]
                return pd.Series(features_dict[first_feature_name], index=data.index)
            else:
                return pd.Series(np.zeros(len(data)), index=data.index)

        except Exception as e:
            logger.error(f"Error generating enhanced cross-timeframe features: {e}")
            return pd.Series(np.zeros(len(data)), index=data.index)

    def generate_enhanced_cross_timeframe_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """Generate comprehensive enhanced cross-timeframe features."""
        features = {}

        try:
            # Fractional change features with proper lag handling
            features.update(self._generate_fractional_change_features(data))

            # Cross-timeframe alignment features
            features.update(self._generate_alignment_features(data))

            # Learned projection features
            features.update(self._generate_learned_projection_features(data))

            # Regime-aware cross-timeframe features
            features.update(self._generate_regime_aware_cross_timeframe_features(data))

            # Multi-scale correlation features
            features.update(self._generate_multi_scale_correlation_features(data))

            logger.info(f"Generated {len(features)} enhanced cross-timeframe features")
            return features

        except Exception as e:
            logger.error(f"Error in generate_enhanced_cross_timeframe_features: {e}")
            return {}

    def _generate_fractional_change_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate fractional change features across timeframes with proper lag handling."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        feature_types = self.config.parameters.get("feature_types", ["momentum", "volatility", "volume", "trend"])

        for fast_tf in timeframes:
            for slow_tf in timeframes:
                if fast_tf >= slow_tf:
                    continue

                for feature_type in feature_types:
                    # Calculate features with proper lag handling
                    fast_feature = self._calculate_feature_with_lag(data, fast_tf, feature_type)
                    slow_feature = self._calculate_feature_with_lag(data, slow_tf, feature_type)

                    if fast_feature is not None and slow_feature is not None:
                        # Fractional change
                        fractional_change = fast_feature / (slow_feature + 1e-8)
                        features[f"frac_change_{feature_type}_{fast_tf}m_{slow_tf}m"] = fractional_change.fillna(0).values

                        # Relative change
                        relative_change = (fast_feature - slow_feature) / (slow_feature + 1e-8)
                        features[f"rel_change_{feature_type}_{fast_tf}m_{slow_tf}m"] = relative_change.fillna(0).values

                        # Momentum divergence
                        momentum_div = fast_feature - slow_feature
                        features[f"momentum_div_{feature_type}_{fast_tf}m_{slow_tf}m"] = momentum_div.fillna(0).values

        return features

    def _calculate_feature_with_lag(self, data: pd.DataFrame, timeframe: int, feature_type: str) -> Optional[pd.Series]:
        """Calculate feature with proper lag handling to avoid lookahead bias."""
        try:
            if feature_type == "momentum":
                # Calculate momentum with lag using VectorBT
                lag_bars = max(1, timeframe // 5)  # Lag by 20% of timeframe
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=timeframe)
                return rolling_apply(returns, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)

            elif feature_type == "volatility":
                # Calculate volatility with lag using VectorBT
                lag_bars = max(1, timeframe // 5)
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
                vol = rolling_std(returns, window=timeframe)
                return rolling_apply(vol, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)

            elif feature_type == "volume":
                if "volume" in data.columns:
                    lag_bars = max(1, timeframe // 5)
                    vol_ma = rolling_mean(data["volume"], window=timeframe)
                    return rolling_apply(vol_ma, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)
                else:
                    return None

            elif feature_type == "trend":
                # Calculate trend strength with lag using VectorBT
                lag_bars = max(1, timeframe // 5)
                trend = self._calculate_trend_strength(data["close"], timeframe)
                return rolling_apply(trend, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)

            elif feature_type == "range":
                # Calculate high-low range with lag using VectorBT
                lag_bars = max(1, timeframe // 5)
                if "high" in data.columns and "low" in data.columns:
                    hl_range = rolling_mean(data["high"] - data["low"], window=timeframe)
                    return rolling_apply(hl_range, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)
                else:
                    return None

            else:
                return None

        except Exception as e:
            logger.warning(f"Error calculating {feature_type} for timeframe {timeframe}: {e}")
            return None

    def _calculate_trend_strength(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend strength using linear regression slope."""
        def calc_slope(x):
            if len(x) < 2:
                return 0.0
            try:
                return np.polyfit(range(len(x)), x, 1)[0]
            except:
                return 0.0

        return rolling_apply(series, calc_slope, window=window)

    def _generate_alignment_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate cross-timeframe alignment features."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        alignment_methods = self.config.parameters.get("alignment_methods", ["lag", "resample", "interpolate"])

        for source_tf in timeframes:
            for target_tf in timeframes:
                if source_tf >= target_tf:
                    continue

                for method in alignment_methods:
                    aligned_feature = self._align_timeframes(data, source_tf, target_tf, method)
                    if aligned_feature is not None:
                        features[f"aligned_{source_tf}m_to_{target_tf}m_{method}"] = aligned_feature.fillna(0).values

        return features

    def _align_timeframes(self, data: pd.DataFrame, source_tf: int, target_tf: int, method: str) -> Optional[pd.Series]:
        """Align features from source timeframe to target timeframe."""
        try:
            if method == "lag":
                # Lag fast timeframe features by appropriate number of bars using VectorBT
                lag_bars = target_tf // source_tf - 1
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
                return rolling_apply(returns, lambda x: x.iloc[0] if len(x) > lag_bars else 0, window=lag_bars+1)

            elif method == "resample":
                # Resample to target timeframe using VectorBT
                resampled = data["close"].resample(f'{target_tf}min').last()
                # Forward fill to original frequency
                aligned = resampled.reindex(data.index, method='ffill')
                return rolling_apply(aligned, lambda x: (x.iloc[-1] / x.iloc[0] - 1) if x.iloc[0] != 0 else 0, window=2).fillna(0)

            elif method == "interpolate":
                # Interpolate between timeframes using VectorBT
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1)
                # Simple interpolation (in practice, would use more sophisticated methods)
                return rolling_mean(returns, window=target_tf//source_tf)

            else:
                return None

        except Exception as e:
            logger.warning(f"Error aligning timeframes {source_tf} to {target_tf} with method {method}: {e}")
            return None

    def _generate_learned_projection_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate learned projection features across timeframes."""
        features = {}
        timeframes = self.config.parameters.get("timeframes", [1, 5, 15, 30, 60])
        projection_methods = self.config.parameters.get("projection_methods", ["pca", "autoencoder", "patchtst"])

        for method in projection_methods:
            if method == "pca":
                features.update(self._generate_pca_projection_features(data, timeframes))
            elif method == "autoencoder":
                features.update(self._generate_autoencoder_projection_features(data, timeframes))
            elif method == "patchtst":
                features.update(self._generate_patchtst_projection_features(data, timeframes))

        return features

    def _generate_pca_projection_features(self, data: pd.DataFrame, timeframes: List[int]) -> Dict[str, np.ndarray]:
        """Generate PCA projection features across timeframes."""
        features = {}

        try:
            from sklearn.decomposition import PCA

            # Create features for each timeframe
            tf_features = []
            for tf in timeframes:
                # Calculate returns for this timeframe using VectorBT
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=tf).fillna(0)

                # Calculate volatility for this timeframe using VectorBT
                vol = rolling_std(rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1), window=tf).fillna(0)

                # Calculate momentum for this timeframe using VectorBT
                momentum = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=tf * 2).fillna(0)

                # Calculate trend for this timeframe
                trend = self._calculate_trend_strength(data["close"], tf).fillna(0)

                tf_features.append(pd.concat([returns, vol, momentum, trend], axis=1))

            # Combine features from all timeframes
            feature_matrix = pd.concat(tf_features, axis=1).fillna(0)

            # Apply PCA for dimensionality reduction
            if len(feature_matrix.columns) >= 3:
                pca = PCA(n_components=min(3, len(feature_matrix.columns)))
                pca_result = pca.fit_transform(feature_matrix)

                for i in range(pca_result.shape[1]):
                    features[f"pca_component_{i+1}"] = pca_result[:, i]

                # Explained variance ratio
                for i, ratio in enumerate(pca.explained_variance_ratio_):
                    features[f"pca_explained_var_{i+1}"] = np.full(len(data), ratio)

        except Exception as e:
            logger.warning(f"Error in PCA projection: {e}")

        return features

    def _generate_autoencoder_projection_features(self, data: pd.DataFrame, timeframes: List[int]) -> Dict[str, np.ndarray]:
        """Generate autoencoder projection features across timeframes."""
        features = {}

        try:
            # Create input features
            input_features = []
            for tf in timeframes:
                returns = rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=tf).fillna(0)
                vol = rolling_std(rolling_apply(data["close"], lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, window=1), window=tf).fillna(0)
                input_features.extend([returns, vol])

            feature_matrix = pd.concat(input_features, axis=1).fillna(0)

            # Simple autoencoder using PCA as proxy
            try:
                if len(feature_matrix.columns) >= 2:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.n_components)
                    projections = pca.fit_transform(feature_matrix)
                    return pd.Series(projections[:, 0], index=data.index, name=self.config.name)
                else:
                    return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)
            except Exception as e:
                return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)
        
        except Exception as e:
            return pd.Series(np.zeros(len(data)), index=data.index, name=self.config.name)

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
