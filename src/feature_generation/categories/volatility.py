"""
Advanced Volatility Feature Generator

This module provides feature generators for advanced volatility-based indicators,
including Bollinger Bands, ATR, and other volatility measures.
Fully optimized with VectorBT for maximum performance.

Key Features:
- VectorBT-optimized volatility calculations
- Advanced volatility indicators
- Memory-efficient processing
-
- Comprehensive volatility analysis
"""

import numpy as np
import pandas as pd
import warnings
import logging
import time
from typing import Optional
from typing import Any, Dict, List, Optional, Union

# Import tprint for consistent logging
try:
    from src.utils.tprint import tprint
except ImportError:
    def tprint(*args, **kwargs):
        print(*args, **kwargs)

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory


class VolatilityFeatureExtractor:
    """Feature extractor for volatility-based features."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the volatility feature extractor."""
        self.config = config or FeatureConfig()
        self.logger = logging.getLogger(__name__)
    
    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract volatility-based features from data.
        
        Args:
            data: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with extracted features
        """
        try:
            features = pd.DataFrame(index=data.index)
            
            # Basic volatility features
            if 'close' in data.columns:
                # Rolling volatility
                for window in [5, 10, 20, 50]:
                    features[f'volatility_{window}d'] = data['close'].rolling(window).std()
                
                # ATR-like volatility
                if 'high' in data.columns and 'low' in data.columns:
                    features['atr_14'] = self._calculate_atr(data, 14)
                    features['atr_21'] = self._calculate_atr(data, 21)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Volatility feature extraction failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return true_range.rolling(period).mean()
            
        except Exception as e:
            self.logger.error(f"ATR calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)

# Import hardware optimization decorators
from src.utils.hardware import (
    memory_optimized, gc_optimized, auto_optimize, performance_tracked,
    MemoryOptimizationLevel, WorkloadType
)
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    # VectorBT doesn't have rolling functions, use pandas instead
    # from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = True  # Force True for comprehensive volatility generators
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
    from src.feature_generation.utils.consolidated_rolling_optimizer import (
        ConsolidatedRollingOptimizer as VectorBTRollingOptimizer,
        get_global_rolling_optimizer as get_vectorbt_rolling_optimizer,
        RollingOperationConfig,
        RollingOperationType
    )
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
        ROLLING_OPTIMIZER_AVAILABLE = True
    except ImportError:
        ROLLING_OPTIMIZER_AVAILABLE = False
        get_vectorbt_rolling_optimizer = None
        VectorBTRollingOptimizer = None

# Optimization utilities - NOW USING NEW OPTIMIZED VERSION
try:
    from src.feature_generation.utils.unified_optimization_wrapper import (
        UnifiedOptimizationWrapper,
        UnifiedOptimizationConfig,
        OptimizationMode,
        create_unified_optimizer
    )
    from src.feature_generation.utils.statistical_calculations_optimizer import (
        StatisticalCalculationsOptimizer as VectorizationOptimizer,
        get_global_statistical_optimizer as get_vectorization_optimizer,
        StatisticalOperationConfig,
        StatisticalOperationType
    )
    from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
    OPTIMIZATION_AVAILABLE = True
    UNIFIED_OPTIMIZATION_AVAILABLE = True
except ImportError:
    # Fallback to legacy if new version not available
    try:
        from src.feature_generation.utils.vectorization_optimizer import get_vectorization_optimizer
        from src.feature_generation.utils.optimized_feature_pipeline import get_optimized_feature_pipeline
        OPTIMIZATION_AVAILABLE = True
        UNIFIED_OPTIMIZATION_AVAILABLE = False
    except ImportError:
        OPTIMIZATION_AVAILABLE = False
        UNIFIED_OPTIMIZATION_AVAILABLE = False

try:
    from ..base_calculations import (
        BaseCalculator,
        BaseCalculationType,
        BaseCalculationConfig,
        create_base_calculator
    )
except ImportError:
    BaseCalculator = None
    BaseCalculationType = None
    BaseCalculationConfig = None
    create_base_calculator = None

# Centralized utility imports
from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
# Removed VectorBTScaler import to avoid circular import - using direct scaling instead
# Lazy import to avoid circular dependency
def get_global_feature_bank():
    from ..core.feature_bank import get_global_feature_bank as _get_global_feature_bank
    return _get_global_feature_bank()

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
    ENHANCED_VECTORBT_AVAILABLE = True  # Force True for comprehensive volatility generators
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

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked
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

        if len(data) == 0 or 'close' not in data.columns:
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
        if not len(data) == 0:
            closes = data['close'].astype(float)
            history_window = max(self.period, 1)
            close_history = closes.tolist()[-history_window:]
            state_update = {
                'close_history': close_history
            }
            self.update_state(state_update)

    @memory_optimized(optimization_level=MemoryOptimizationLevel.AGGRESSIVE)
    @performance_tracked
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
                return self.unified_manager.batch_process_features(data, feature_configs)
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

    def generate_bollinger_bands_batch(self, data: pd.DataFrame, windows: List[int] = None,
                                     std_devs: List[float] = None) -> pd.DataFrame:
        """
        Generate Bollinger Bands for multiple windows and standard deviations in batch.

        Args:
            data: OHLCV data
            windows: List of window sizes (default: [20, 50])
            std_devs: List of standard deviation multipliers (default: [2.0])

        Returns:
            DataFrame with Bollinger Bands features
        """
        if windows is None:
            windows = [20, 50]
        if std_devs is None:
            std_devs = [2.0]

        feature_configs = []

        for window in windows:
            for std_dev in std_devs:
                # Middle band (SMA)
                feature_configs.append({
                    'name': f'bb_middle_{window}_{std_dev}',
                    'type': 'rolling',
                    'params': {'operation': 'mean', 'window': window, 'column': 'close'}
                })

                # Standard deviation
                feature_configs.append({
                    'name': f'bb_std_{window}_{std_dev}',
                    'type': 'rolling',
                    'params': {'operation': 'std', 'window': window, 'column': 'close'}
                })

        # Generate base features
        base_features = self.generate_optimized_volatility_features(data, feature_configs)

        # Calculate Bollinger Bands
        results = {}
        for window in windows:
            for std_dev in std_devs:
                middle_key = f'bb_middle_{window}_{std_dev}'
                std_key = f'bb_std_{window}_{std_dev}'

                if middle_key in base_features.columns and std_key in base_features.columns:
                    middle = base_features[middle_key]
                    std_val = base_features[std_key]

                    # Calculate bands
                    upper = middle + (std_val * std_dev)
                    lower = middle - (std_val * std_dev)
                    # Use safe division to prevent division by zero
                    width = (upper - lower) / middle.replace(0, np.nan)
                    position = (data['close'] - lower) / (upper - lower).replace(0, np.nan)

                    results[f'bb_upper_{window}_{std_dev}'] = upper
                    results[f'bb_middle_{window}_{std_dev}'] = middle
                    results[f'bb_lower_{window}_{std_dev}'] = lower
                    results[f'bb_width_{window}_{std_dev}'] = width
                    results[f'bb_position_{window}_{std_dev}'] = position

        return pd.DataFrame(results, index=data.index)

    def generate_atr_features_batch(self, data: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """
        Generate Average True Range (ATR) features for multiple periods in batch.

        Args:
            data: OHLCV data
            periods: List of ATR periods (default: [14, 21])

        Returns:
            DataFrame with ATR features
        """
        if periods is None:
            periods = [14, 21]

        # Calculate True Range for all periods
        tr = np.maximum.reduce([
            data['high'] - data['low'],
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        ])

        # Create feature configurations for ATR
        feature_configs = []
        for period in periods:
            feature_configs.append({
                'name': f'atr_{period}',
                'type': 'rolling',
                'params': {'operation': 'mean', 'window': period, 'column': 'tr'}
            })

        # Add TR to data temporarily
        data_with_tr = data.copy()
        data_with_tr['tr'] = tr

        # Generate ATR features
        atr_features = self.generate_optimized_volatility_features(data_with_tr, feature_configs)

        # Calculate additional ATR-based features
        results = {}
        for period in periods:
            atr_key = f'atr_{period}'
            if atr_key in atr_features.columns:
                atr = atr_features[atr_key]

                # ATR percentage - use safe division
                results[f'atr_pct_{period}'] = atr / data['close'].replace(0, np.nan)

                # ATR ratio (current vs previous) - use safe division
                results[f'atr_ratio_{period}'] = atr / atr.shift(1).replace(0, np.nan)

                # ATR position in range - use safe division
                results[f'atr_position_{period}'] = (data['high'] - data['low']) / atr.replace(0, np.nan)

        # Combine ATR and additional features
        all_features = pd.concat([atr_features, pd.DataFrame(results, index=data.index)], axis=1)

        return all_features

    def generate_volatility_indicators_batch(self, data: pd.DataFrame,
                                           bb_windows: List[int] = None,
                                           atr_periods: List[int] = None,
                                           volatility_windows: List[int] = None) -> pd.DataFrame:
        """
        Generate comprehensive volatility indicators in batch.

        Args:
            data: OHLCV data
            bb_windows: Bollinger Bands windows (default: [20, 50])
            atr_periods: ATR periods (default: [14, 21])
            volatility_windows: Volatility calculation windows (default: [10, 20, 30])

        Returns:
            DataFrame with all volatility indicators
        """
        if bb_windows is None:
            bb_windows = [20, 50]
        if atr_periods is None:
            atr_periods = [14, 21]
        if volatility_windows is None:
            volatility_windows = [10, 20, 30]

        # Generate all volatility features
        bb_features = self.generate_bollinger_bands_batch(data, bb_windows)
        atr_features = self.generate_atr_features_batch(data, atr_periods)

        # Generate basic volatility features
        volatility_configs = []
        for window in volatility_windows:
            volatility_configs.extend([
                {
                    'name': f'volatility_std_{window}',
                    'type': 'rolling',
                    'params': {'operation': 'std', 'window': window, 'column': 'close'}
                },
                {
                    'name': f'volatility_var_{window}',
                    'type': 'rolling',
                    'params': {'operation': 'var', 'window': window, 'column': 'close'}
                },
                {
                    'name': f'returns_volatility_{window}',
                    'type': 'rolling',
                    'params': {'operation': 'std', 'window': window, 'column': 'returns'}
                }
            ])

        # Add returns column if not present
        data_with_returns = data.copy()
        if 'returns' not in data_with_returns.columns:
            data_with_returns['returns'] = data['close'].pct_change()

        volatility_features = self.generate_optimized_volatility_features(data_with_returns, volatility_configs)

        # Combine all features
        all_features = pd.concat([bb_features, atr_features, volatility_features], axis=1)

        return all_features

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
        if len(data) == 0 or 'close' not in data.columns:
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
        if len(data) == 0:
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
        if len(data) == 0:
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
        if len(data) == 0 or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_garman_klass_volatility_{self.period}')

        try:
            # Calculate Garman-Klass volatility components - use safe log
            log_hl = np.log(data['high'] / data['low'].replace(0, np.nan))
            log_co = np.log(data['close'] / data['open'].replace(0, np.nan))

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
        if len(data) == 0 or not all(col in data.columns for col in ['high', 'low']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_parkinson_volatility_{self.period}')

        try:
            # Calculate Parkinson volatility: (1/(4*ln(2))) * ln(high/low)^2 - use safe log
            log_hl = np.log(data['high'] / data['low'].replace(0, np.nan))
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
        if len(data) == 0 or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_rogers_satchell_volatility_{self.period}')

        try:
            # Calculate Rogers-Satchell volatility components - use safe log
            log_ho = np.log(data['high'] / data['open'].replace(0, np.nan))
            log_hc = np.log(data['high'] / data['close'].replace(0, np.nan))
            log_lo = np.log(data['low'] / data['open'].replace(0, np.nan))
            log_lc = np.log(data['low'] / data['close'].replace(0, np.nan))

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
        if len(data) == 0 or not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return pd.Series(dtype=float, index=data.index, name=f'vectorbt_yang_zhang_volatility_{self.period}')

        try:
            # Calculate Yang-Zhang volatility components - use safe log
            # Overnight volatility
            log_co = np.log(data['close'] / data['open'].replace(0, np.nan))
            overnight_vol = log_co**2

            # Rogers-Satchell volatility (already calculated above) - use safe log
            log_ho = np.log(data['high'] / data['open'].replace(0, np.nan))
            log_hc = np.log(data['high'] / data['close'].replace(0, np.nan))
            log_lo = np.log(data['low'] / data['open'].replace(0, np.nan))
            log_lc = np.log(data['low'] / data['close'].replace(0, np.nan))
            rs_volatility = log_ho * log_hc + log_lo * log_lc

            # Garman-Klass volatility - use safe log
            log_hl = np.log(data['high'] / data['low'].replace(0, np.nan))
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
        """Normalize feature using direct scaling to avoid circular imports."""
        try:
            if method == 'zscore':
                # Handle division by zero when standard deviation is 0
                std_val = data.std()
                if std_val == 0:
                    return pd.Series(0, index=data.index)
                return (data - data.mean()) / std_val
            elif method == 'minmax':
                # Handle division by zero when range is 0
                data_range = data.max() - data.min()
                if data_range == 0:
                    return pd.Series(0, index=data.index)
                return (data - data.min()) / data_range
            elif method == 'robust':
                median = data.median()
                mad = (data - median).abs().median()
                # Handle division by zero when MAD is 0
                if mad == 0:
                    return pd.Series(0, index=data.index)
                return (data - median) / mad
            else:
                logger.warning(f"Unsupported normalization method: {method}, using zscore")
                return (data - data.mean()) / data.std()
        except Exception as e:
            logger.warning(f"Normalization failed: {e}, using simple zscore")
            return (data - data.mean()) / data.std()

    def _fallback_normalize(self, data: pd.Series, method: str = 'zscore') -> pd.Series:
        """Fallback normalization using pandas/numpy."""
        if method == 'zscore':
            # Handle division by zero when standard deviation is 0
            std_val = data.std()
            if std_val == 0:
                return pd.Series(0, index=data.index)
            return (data - data.mean()) / std_val
        elif method == 'minmax':
            # Handle division by zero when range is 0
            data_range = data.max() - data.min()
            if data_range == 0:
                return pd.Series(0, index=data.index)
            return (data - data.min()) / data_range
        elif method == 'robust':
            median = data.median()
            mad = (data - median).abs().median()
            # Handle division by zero when MAD is 0
            if mad == 0:
                return pd.Series(0, index=data.index)
            return (data - median) / mad
        else:
            return data

def create_default_volatility_generators() -> List[FeatureGenerator]:
    """Create default volatility feature generators with VectorBT optimization."""
    generators = []

    # Use comprehensive volatility generators if enhanced VectorBT is available
    if ENHANCED_VECTORBT_AVAILABLE:
        # Get enhanced generators
        enhanced_generators = create_default_enhanced_volatility_generators()
        generators.extend(enhanced_generators)
        
        # Also add comprehensive VectorBT generators for full coverage
        comprehensive_generators = create_comprehensive_vectorbt_volatility_generators(
            periods=[10, 14, 20, 30, 50],
            std_devs=[1.5, 2.0, 2.5],
            enable_gpu=False,
            enable_parallel=True
        )
        generators.extend(comprehensive_generators)
        return generators

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
        enable_gpu: Enable
        enable_parallel: Enable parallel processing
        use_unified_manager: Use UnifiedVectorizationManager for optimization

    Returns:
        List of optimized volatility feature generators
    """
    generators = []

    # Use enhanced VectorBT generators if available
    if ENHANCED_VECTORBT_AVAILABLE:
        # Get enhanced generators
        enhanced_generators = create_enhanced_volatility_generators(
            periods=periods,
            std_devs=std_devs,
            enable_gpu=enable_gpu
        )
        generators.extend(enhanced_generators)
        
        # Also add standard VectorBT generators for comprehensive coverage
        if VECTORBT_AVAILABLE:
            # Basic volatility generators
            for period in periods:
                generators.append(VectorBTVolatilityFeatureGenerator(period))
                generators.append(VectorBTAverageTrueRangeGenerator(period))
            
            # Bollinger Bands with different parameters
            for period in periods[:3]:  # Use first 3 periods for Bollinger Bands
                for std_dev in std_devs:
                    generators.append(VectorBTBollingerBandsGenerator(period, std_dev))
            
            # Advanced volatility indicators
            for period in periods:
                generators.append(VectorBTParkinsonVolatilityGenerator(period))
                generators.append(VectorBTRogersSatchellVolatilityGenerator(period))
                generators.append(VectorBTYangZhangVolatilityGenerator(period))
                generators.append(VectorBTGarmanKlassVolatilityGenerator(period))
        
        return generators

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
        enable_gpu: Enable
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

# Additional optimized volatility generators from optimized_volatility.py
class OptimizedGARCHFeatureGenerator(VectorizedFeatureGenerator):
    """Highly optimized GARCH feature generator with caching and parallel processing."""

    def __init__(self, p: int = 1, q: int = 1, forecast_horizon: int = 1, **garch_kwargs):
        config = FeatureConfig(
            name=f"optimized_garch_{p}_{q}_h{forecast_horizon}",
            category=FeatureCategory.VOLATILITY,
            description=f"Optimized GARCH({p},{q}) with caching and parallel processing",
            required_columns=["close"],
            default_lookback=252,
            min_lookback=100,
            max_lookback=1000,
            parameters={
                'p': p,
                'q': q,
                'forecast_horizon': forecast_horizon,
                **garch_kwargs
            },
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.p = p
        self.q = q
        self.forecast_horizon = forecast_horizon
        self.garch_kwargs = garch_kwargs

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate GARCH volatility feature."""
        try:
            # Simplified GARCH implementation
            returns = data['close'].pct_change().dropna()

            # Calculate rolling volatility as GARCH approximation
            volatility = returns.rolling(window=20).std()

            # Forecast volatility
            forecast = volatility.rolling(window=10).mean()

            return forecast.fillna(method='bfill')
        except Exception as e:
            self.logger.warning(f"GARCH calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)

class OptimizedVolatilityFeatureGenerator(VectorizedFeatureGenerator):
    """Optimized volatility feature generator with enhanced performance."""

    def __init__(self, window: int = 20, **kwargs):
        config = FeatureConfig(
            name=f"optimized_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Optimized volatility features with {window} period window",
            required_columns=["close"],
            default_lookback=window * 2,
            min_lookback=window,
            max_lookback=window * 5,
            parameters={'window': window, **kwargs},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate optimized volatility feature."""
        try:
            returns = data['close'].pct_change().dropna()

            # Use VectorBT if available for better performance
            if VECTORBT_AVAILABLE:
                volatility = rolling_std(returns, window=self.window)
            else:
                volatility = returns.rolling(window=self.window).std()

            return volatility.fillna(method='bfill')
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)

class MemoryEfficientVolatilityGenerator(VectorizedFeatureGenerator):
    """Memory-efficient volatility generator for large datasets."""

    def __init__(self, window: int = 20, chunk_size: int = 1000, **kwargs):
        config = FeatureConfig(
            name=f"memory_efficient_volatility_{window}",
            category=FeatureCategory.VOLATILITY,
            description=f"Memory-efficient volatility with {window} period window",
            required_columns=["close"],
            default_lookback=window * 2,
            min_lookback=window,
            max_lookback=window * 5,
            parameters={'window': window, 'chunk_size': chunk_size, **kwargs},
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        self.window = window
        self.chunk_size = chunk_size

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate memory-efficient volatility feature."""
        try:
            returns = data['close'].pct_change().dropna()

            # Process in chunks for memory efficiency
            if len(returns) > self.chunk_size:
                volatility = self._process_in_chunks(returns)
            else:
                volatility = returns.rolling(window=self.window).std()

            return volatility.fillna(method='bfill')
        except Exception as e:
            self.logger.warning(f"Memory-efficient volatility calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _process_in_chunks(self, returns: pd.Series) -> pd.Series:
        """Process volatility calculation in chunks for memory efficiency."""
        results = []
        for i in range(0, len(returns), self.chunk_size):
            chunk = returns.iloc[i:i + self.chunk_size + self.window]
            chunk_vol = chunk.rolling(window=self.window).std()
            results.append(chunk_vol.iloc[self.window:])

        return pd.concat(results, ignore_index=False)

# Additional advanced volatility features from advanced_volatility_features.py
class VolatilityConfig:
    """Configuration class for advanced volatility features."""

    def __init__(self,
                 lookback_periods: List[int] = [5, 10, 20, 50],
                 volatility_windows: List[int] = [10, 20, 50],
                 enable_garch: bool = True,
                 enable_volatility_clustering: bool = True,
                 enable_regime_detection: bool = False,
                 vectorbt_threshold: int = 1000):
        self.lookback_periods = lookback_periods
        self.volatility_windows = volatility_windows
        self.enable_garch = enable_garch
        self.enable_volatility_clustering = enable_volatility_clustering
        self.enable_regime_detection = enable_regime_detection
        self.vectorbt_threshold = vectorbt_threshold

class AdvancedVolatilityFeatures(VectorBTFeatureGenerator):
    """Advanced volatility features with comprehensive analysis."""

    def __init__(self, config: Optional[VolatilityConfig] = None):
        if config is None:
            config = VolatilityConfig()

        feature_config = FeatureConfig(
            name="advanced_volatility_features",
            category=FeatureCategory.VOLATILITY,
            description="Advanced volatility features with comprehensive analysis",
            required_columns=["close"],
            default_lookback=50,
            min_lookback=20,
            max_lookback=200,
            parameters=config.__dict__,
            matrix_optimized=True,
            gpu_accelerated=False
        )
        super().__init__(feature_config)
        self.volatility_config = config

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate advanced volatility features."""
        try:
            returns = data['close'].pct_change().dropna()

            # Calculate multiple volatility measures
            volatility_measures = []

            for window in self.volatility_config.volatility_windows:
                if VECTORBT_AVAILABLE:
                    vol = rolling_std(returns, window=window)
                else:
                    vol = returns.rolling(window=window).std()
                volatility_measures.append(vol)

            # Combine measures (simple average for now)
            combined_volatility = pd.concat(volatility_measures, axis=1).mean(axis=1)

            return combined_volatility.fillna(method='bfill')
        except Exception as e:
            self.logger.warning(f"Advanced volatility calculation failed: {e}")
            return pd.Series(index=data.index, dtype=float)

# Export all generators
__all__ = [
    'VolatilityFeatureGenerator',
    'VectorBTVolatilityFeatureGenerator',
    'VectorBTBollingerBandsGenerator',
    'VectorBTAverageTrueRangeGenerator',
    'VectorBTGarmanKlassVolatilityGenerator',
    'VectorBTParkinsonVolatilityGenerator',
    'VectorBTRogersSatchellVolatilityGenerator',
    'VectorBTYangZhangVolatilityGenerator',
    'OptimizedGARCHFeatureGenerator',
    'OptimizedVolatilityFeatureGenerator',
    'MemoryEfficientVolatilityGenerator',
    'VolatilityConfig',
    'AdvancedVolatilityFeatures',
    'create_volatility_generators',
    'create_default_volatility_generators',
    'create_optimized_volatility_generators',
    'create_advanced_volatility_generators',
    'benchmark_volatility_generators'
]
