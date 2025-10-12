"""
Enhanced VectorBT Volatility Feature Generator

This module provides comprehensive volatility feature generation with full VectorBT optimization,
including UnifiedVectorizationManager integration, advanced rolling operations, and intelligent
strategy selection based on data characteristics.

Key Features:
- Full VectorBT integration with UnifiedVectorizationManager
- Intelligent optimization strategy selection
- Advanced volatility indicators with VectorBT acceleration
- Memory-efficient processing for large datasets
- Comprehensive performance monitoring
- GPU acceleration support
- Parallel processing capabilities
"""

import numpy as np
import pandas as pd
import warnings
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import time

from ..core.feature_generator import FeatureGenerator, FeatureResult, VectorizedFeatureGenerator, FeatureConfig, FeatureCategory
from ..core.vectorbt_feature_generator import VectorBTFeatureGenerator, VECTORBT_AVAILABLE
from ..core.vectorbt_optimization_mixin import VectorBTOptimizationMixin

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import (
        rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, 
        rolling_sum, rolling_apply, rolling_corr, rolling_cov,
        rolling_quantile, rolling_skew, rolling_kurt
    )
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    from vectorbt.portfolio import Portfolio
    from vectorbt.records import Drawdowns
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
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    Portfolio = None
    Drawdowns = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

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

# VectorBT Rolling Optimizer
try:
    from ..utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer, VectorBTRollingOptimizer
    ROLLING_OPTIMIZER_AVAILABLE = True
except ImportError:
    ROLLING_OPTIMIZER_AVAILABLE = False
    get_vectorbt_rolling_optimizer = None
    VectorBTRollingOptimizer = None

# Vectorization Optimizer
try:
    from ..utils.vectorization_optimizer import get_vectorization_optimizer, VectorizationConfig
    VECTORIZATION_OPTIMIZER_AVAILABLE = True
except ImportError:
    VECTORIZATION_OPTIMIZER_AVAILABLE = False
    get_vectorization_optimizer = None
    VectorizationConfig = None

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

logger = logging.getLogger(__name__)


@dataclass
class VolatilityConfig:
    """Configuration for volatility feature generation."""
    # Basic parameters
    period: int = 20
    std_dev: float = 2.0
    
    # VectorBT optimization settings
    use_unified_manager: bool = True
    use_rolling_optimizer: bool = True
    enable_gpu: bool = False
    enable_parallel: bool = True
    memory_efficient: bool = True
    
    # Advanced settings
    chunk_size: int = 1000
    vectorization_threshold: int = 1000
    precision_requirement: str = "medium"  # "low", "medium", "high"
    
    # Performance monitoring
    enable_profiling: bool = False
    track_performance: bool = True


class EnhancedVectorBTVolatilityGenerator(VectorizedFeatureGenerator, VectorBTOptimizationMixin):
    """
    Enhanced volatility feature generator with comprehensive VectorBT optimization.
    
    This generator provides:
    - Full VectorBT integration with UnifiedVectorizationManager
    - Intelligent strategy selection based on data characteristics
    - Advanced volatility indicators with VectorBT acceleration
    - Memory-efficient processing for large datasets
    - Comprehensive performance monitoring
    """

    def __init__(self, config: Optional[VolatilityConfig] = None, base_calculation: Optional[BaseCalculationType] = None):
        self.config = config or VolatilityConfig()
        self.base_calculation = base_calculation
        
        # Initialize feature configuration
        feature_config = self._create_feature_config()
        super().__init__(feature_config, enable_matrix_ops=True, enable_vectorization_optimization=True)
        
        # Initialize optimization components
        self._initialize_optimization_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_operations': 0,
            'vectorbt_operations': 0,
            'unified_manager_operations': 0,
            'rolling_optimizer_operations': 0,
            'pandas_fallbacks': 0,
            'gpu_operations': 0,
            'parallel_operations': 0,
            'total_time': 0.0,
            'memory_savings': 0.0
        }

    def _create_feature_config(self) -> FeatureConfig:
        """Create feature configuration."""
        return FeatureConfig(
            name=f"enhanced_vectorbt_volatility_{self.config.period}",
            category=FeatureCategory.VOLATILITY,
            description=f"Enhanced VectorBT-optimized volatility features over {self.config.period} periods",
            required_columns=["close"],
            optional_columns=["high", "low", "open", "volume"],
            default_lookback=self.config.period,
            min_lookback=self.config.period,
            max_lookback=self.config.period,
            parameters={
                "period": self.config.period,
                "std_dev": self.config.std_dev,
                "use_unified_manager": self.config.use_unified_manager,
                "enable_gpu": self.config.enable_gpu,
                "enable_parallel": self.config.enable_parallel
            },
            matrix_optimized=True,
            gpu_accelerated=self.config.enable_gpu
        )

    def _initialize_optimization_components(self):
        """Initialize optimization components."""
        # Unified Vectorization Manager
        if self.config.use_unified_manager and UNIFIED_MANAGER_AVAILABLE:
            try:
                self.unified_manager = UnifiedVectorizationManager()
                logger.info("✅ Unified Vectorization Manager initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Unified Vectorization Manager: {e}")
                self.unified_manager = None
        else:
            self.unified_manager = None

        # VectorBT Rolling Optimizer
        if self.config.use_rolling_optimizer and ROLLING_OPTIMIZER_AVAILABLE:
            try:
                self.rolling_optimizer = get_vectorbt_rolling_optimizer(
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel
                )
                logger.info("✅ VectorBT Rolling Optimizer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize VectorBT Rolling Optimizer: {e}")
                self.rolling_optimizer = None
        else:
            self.rolling_optimizer = None

        # Vectorization Optimizer
        if VECTORIZATION_OPTIMIZER_AVAILABLE:
            try:
                vectorization_config = VectorizationConfig(
                    chunk_size=self.config.chunk_size,
                    enable_gpu_acceleration=self.config.enable_gpu,
                    enable_parallel_processing=self.config.enable_parallel,
                    vectorization_threshold=self.config.vectorization_threshold
                )
                self.vectorization_optimizer = get_vectorization_optimizer(vectorization_config)
                logger.info("✅ Vectorization Optimizer initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Vectorization Optimizer: {e}")
                self.vectorization_optimizer = None
        else:
            self.vectorization_optimizer = None

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Generate enhanced volatility features using VectorBT optimization."""
        start_time = time.time()
        
        # Optimize DataFrame for processing
        if hasattr(self, 'optimize_dataframe_processing'):
            data = self.optimize_dataframe_processing(data)

        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=float, index=data.index, name=f'enhanced_volatility_{self.config.period}')

        # Select optimal strategy based on data characteristics
        strategy = self._select_optimal_strategy(data)
        
        # Execute with selected strategy
        result = self._execute_with_strategy(strategy, data, **kwargs)
        
        # Update performance stats
        execution_time = time.time() - start_time
        self.performance_stats['total_operations'] += 1
        self.performance_stats['total_time'] += execution_time
        
        return result

    def _select_optimal_strategy(self, data: pd.DataFrame) -> str:
        """Select optimal VectorBT strategy based on data characteristics."""
        data_size = len(data)
        
        # Use Unified Vectorization Manager for complex operations
        if (self.unified_manager and 
            data_size >= self.config.vectorization_threshold and
            self.config.use_unified_manager):
            return 'unified_manager'
        
        # Use VectorBT Rolling Optimizer for medium-sized datasets
        elif (self.rolling_optimizer and 
              data_size >= 100 and 
              data_size < self.config.vectorization_threshold):
            return 'rolling_optimizer'
        
        # Use direct VectorBT operations for smaller datasets
        elif VECTORBT_AVAILABLE and data_size >= 50:
            return 'direct_vectorbt'
        
        # Fallback to pandas
        else:
            return 'pandas_fallback'

    def _execute_with_strategy(self, strategy: str, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute volatility calculation with selected strategy."""
        if strategy == 'unified_manager':
            return self._execute_with_unified_manager(data, **kwargs)
        elif strategy == 'rolling_optimizer':
            return self._execute_with_rolling_optimizer(data, **kwargs)
        elif strategy == 'direct_vectorbt':
            return self._execute_with_direct_vectorbt(data, **kwargs)
        else:
            return self._execute_with_pandas_fallback(data, **kwargs)

    def _execute_with_unified_manager(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute using Unified Vectorization Manager."""
        try:
            # Prepare data for unified manager
            close_prices = data['close'].dropna()
            returns = close_prices.pct_change().dropna()
            
            if len(returns) < self.config.period:
                return pd.Series(np.nan, index=data.index, name=f'unified_volatility_{self.config.period}')
            
            # Create operation configuration
            config = OperationConfig(
                operation_type=OperationType.FEATURE_ENGINEERING,
                data_size=len(returns),
                data_dimensions=(len(returns),),
                memory_budget_mb=1024.0,
                time_budget_seconds=300.0,
                precision_requirement=self.config.precision_requirement
            )
            
            # Execute with unified manager
            result = self.unified_manager.optimize_operation(
                OperationType.FEATURE_ENGINEERING,
                {'returns': returns, 'period': self.config.period},
                config,
                operation='volatility_calculation'
            )
            
            # Extract volatility from result
            if hasattr(result, 'result'):
                volatility = result.result
            else:
                volatility = result
            
            # Ensure it's a pandas Series
            if not isinstance(volatility, pd.Series):
                volatility = pd.Series(volatility, index=returns.index)
            
            # Align with original data index
            volatility = volatility.reindex(data.index)
            self.performance_stats['unified_manager_operations'] += 1
            
            return volatility.rename(f'unified_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Unified manager execution failed: {e}, using fallback")
            return self._execute_with_rolling_optimizer(data, **kwargs)

    def _execute_with_rolling_optimizer(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute using VectorBT Rolling Optimizer."""
        try:
            close_prices = data['close'].dropna()
            returns = close_prices.pct_change().dropna()
            
            if len(returns) < self.config.period:
                return pd.Series(np.nan, index=data.index, name=f'rolling_volatility_{self.config.period}')
            
            # Use rolling optimizer for volatility calculation
            volatility = self.rolling_optimizer.rolling_std(returns, window=self.config.period)
            
            # Align with original data index
            volatility = volatility.reindex(data.index)
            self.performance_stats['rolling_optimizer_operations'] += 1
            
            return volatility.rename(f'rolling_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Rolling optimizer execution failed: {e}, using fallback")
            return self._execute_with_direct_vectorbt(data, **kwargs)

    def _execute_with_direct_vectorbt(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute using direct VectorBT operations."""
        try:
            close_prices = data['close'].dropna()
            returns = close_prices.pct_change().dropna()
            
            if len(returns) < self.config.period:
                return pd.Series(np.nan, index=data.index, name=f'direct_volatility_{self.config.period}')
            
            # Use direct VectorBT rolling operations
            volatility = rolling_std(returns, window=self.config.period)
            
            # Align with original data index
            volatility = volatility.reindex(data.index)
            self.performance_stats['vectorbt_operations'] += 1
            
            return volatility.rename(f'direct_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Direct VectorBT execution failed: {e}, using pandas fallback")
            return self._execute_with_pandas_fallback(data, **kwargs)

    def _execute_with_pandas_fallback(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Execute using pandas fallback."""
        close_prices = data['close'].dropna()
        returns = close_prices.pct_change().dropna()
        
        if len(returns) < self.config.period:
            return pd.Series(np.nan, index=data.index, name=f'pandas_volatility_{self.config.period}')
        
        # Use pandas rolling operations
        volatility = returns.rolling(window=self.config.period).std()
        
        # Align with original data index
        volatility = volatility.reindex(data.index)
        self.performance_stats['pandas_fallbacks'] += 1
        
        return volatility.rename(f'pandas_volatility_{self.config.period}')

    def generate_comprehensive_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive volatility features using VectorBT."""
        features = {}
        
        # Basic volatility
        features['volatility'] = self._generate_feature(data)
        
        # Advanced volatility indicators
        if 'high' in data.columns and 'low' in data.columns and 'close' in data.columns:
            features['atr'] = self._calculate_atr(data)
            features['bbands_upper'] = self._calculate_bollinger_bands_upper(data)
            features['bbands_lower'] = self._calculate_bollinger_bands_lower(data)
            features['bbands_width'] = self._calculate_bollinger_bands_width(data)
        
        # Garman-Klass volatility
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            features['garman_klass_volatility'] = self._calculate_garman_klass_volatility(data)
            features['parkinson_volatility'] = self._calculate_parkinson_volatility(data)
            features['rogers_satchell_volatility'] = self._calculate_rogers_satchell_volatility(data)
            features['yang_zhang_volatility'] = self._calculate_yang_zhang_volatility(data)
        
        # Volatility of volatility
        if len(features) > 0:
            volatility_series = features['volatility']
            features['volatility_of_volatility'] = self._calculate_volatility_of_volatility(volatility_series)
        
        return pd.DataFrame(features, index=data.index)

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range using VectorBT."""
        if not self.rolling_optimizer:
            return self._calculate_atr_pandas(data)
        
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR using rolling optimizer
            atr = self.rolling_optimizer.rolling_mean(true_range, window=self.config.period)
            return atr.rename(f'atr_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"ATR calculation failed: {e}, using pandas fallback")
            return self._calculate_atr_pandas(data)

    def _calculate_atr_pandas(self, data: pd.DataFrame) -> pd.Series:
        """Calculate ATR using pandas fallback."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = true_range.rolling(window=self.config.period).mean()
        return atr.rename(f'atr_{self.config.period}')

    def _calculate_bollinger_bands_upper(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands upper band."""
        close = data['close']
        
        if self.rolling_optimizer:
            try:
                sma = self.rolling_optimizer.rolling_mean(close, window=self.config.period)
                std = self.rolling_optimizer.rolling_std(close, window=self.config.period)
                return (sma + (std * self.config.std_dev)).rename(f'bbands_upper_{self.config.period}')
            except Exception:
                pass
        
        # Fallback to pandas
        sma = close.rolling(window=self.config.period).mean()
        std = close.rolling(window=self.config.period).std()
        return (sma + (std * self.config.std_dev)).rename(f'bbands_upper_{self.config.period}')

    def _calculate_bollinger_bands_lower(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands lower band."""
        close = data['close']
        
        if self.rolling_optimizer:
            try:
                sma = self.rolling_optimizer.rolling_mean(close, window=self.config.period)
                std = self.rolling_optimizer.rolling_std(close, window=self.config.period)
                return (sma - (std * self.config.std_dev)).rename(f'bbands_lower_{self.config.period}')
            except Exception:
                pass
        
        # Fallback to pandas
        sma = close.rolling(window=self.config.period).mean()
        std = close.rolling(window=self.config.period).std()
        return (sma - (std * self.config.std_dev)).rename(f'bbands_lower_{self.config.period}')

    def _calculate_bollinger_bands_width(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Bollinger Bands width."""
        upper = self._calculate_bollinger_bands_upper(data)
        lower = self._calculate_bollinger_bands_lower(data)
        return ((upper - lower) / ((upper + lower) / 2)).rename(f'bbands_width_{self.config.period}')

    def _calculate_garman_klass_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility using VectorBT."""
        if not self.rolling_optimizer:
            return self._calculate_garman_klass_volatility_pandas(data)
        
        try:
            open_price = data['open']
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Garman-Klass formula
            log_hl = np.log(high / low)
            log_co = np.log(close / open_price)
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Calculate rolling mean using VectorBT
            volatility = self.rolling_optimizer.rolling_mean(gk_volatility, window=self.config.period)
            volatility = np.sqrt(volatility)  # Convert variance to volatility
            
            return volatility.rename(f'garman_klass_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Garman-Klass volatility calculation failed: {e}, using pandas fallback")
            return self._calculate_garman_klass_volatility_pandas(data)

    def _calculate_garman_klass_volatility_pandas(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility using pandas fallback."""
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_price)
        gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        
        volatility = gk_volatility.rolling(window=self.config.period).mean()
        volatility = np.sqrt(volatility)
        
        return volatility.rename(f'garman_klass_volatility_{self.config.period}')

    def _calculate_parkinson_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility using VectorBT."""
        if not self.rolling_optimizer:
            return self._calculate_parkinson_volatility_pandas(data)
        
        try:
            high = data['high']
            low = data['low']
            
            # Parkinson formula
            log_hl = np.log(high / low)
            parkinson_volatility = (1 / (4 * np.log(2))) * log_hl**2
            
            # Calculate rolling mean using VectorBT
            volatility = self.rolling_optimizer.rolling_mean(parkinson_volatility, window=self.config.period)
            volatility = np.sqrt(volatility)
            
            return volatility.rename(f'parkinson_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Parkinson volatility calculation failed: {e}, using pandas fallback")
            return self._calculate_parkinson_volatility_pandas(data)

    def _calculate_parkinson_volatility_pandas(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility using pandas fallback."""
        high = data['high']
        low = data['low']
        
        log_hl = np.log(high / low)
        parkinson_volatility = (1 / (4 * np.log(2))) * log_hl**2
        
        volatility = parkinson_volatility.rolling(window=self.config.period).mean()
        volatility = np.sqrt(volatility)
        
        return volatility.rename(f'parkinson_volatility_{self.config.period}')

    def _calculate_rogers_satchell_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Rogers-Satchell volatility using VectorBT."""
        if not self.rolling_optimizer:
            return self._calculate_rogers_satchell_volatility_pandas(data)
        
        try:
            open_price = data['open']
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Rogers-Satchell formula
            log_ho = np.log(high / open_price)
            log_hc = np.log(high / close)
            log_lo = np.log(low / open_price)
            log_lc = np.log(low / close)
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Calculate rolling mean using VectorBT
            volatility = self.rolling_optimizer.rolling_mean(rs_volatility, window=self.config.period)
            volatility = np.sqrt(volatility)
            
            return volatility.rename(f'rogers_satchell_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Rogers-Satchell volatility calculation failed: {e}, using pandas fallback")
            return self._calculate_rogers_satchell_volatility_pandas(data)

    def _calculate_rogers_satchell_volatility_pandas(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Rogers-Satchell volatility using pandas fallback."""
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        log_ho = np.log(high / open_price)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_price)
        log_lc = np.log(low / close)
        rs_volatility = log_ho * log_hc + log_lo * log_lc
        
        volatility = rs_volatility.rolling(window=self.config.period).mean()
        volatility = np.sqrt(volatility)
        
        return volatility.rename(f'rogers_satchell_volatility_{self.config.period}')

    def _calculate_yang_zhang_volatility(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Yang-Zhang volatility using VectorBT."""
        if not self.rolling_optimizer:
            return self._calculate_yang_zhang_volatility_pandas(data)
        
        try:
            open_price = data['open']
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Yang-Zhang components
            log_co = np.log(close / open_price)
            overnight_vol = log_co**2
            
            # Rogers-Satchell component
            log_ho = np.log(high / open_price)
            log_hc = np.log(high / close)
            log_lo = np.log(low / open_price)
            log_lc = np.log(low / close)
            rs_volatility = log_ho * log_hc + log_lo * log_lc
            
            # Garman-Klass component
            log_hl = np.log(high / low)
            gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            
            # Yang-Zhang formula
            yz_volatility = overnight_vol + rs_volatility + gk_volatility
            
            # Calculate rolling mean using VectorBT
            volatility = self.rolling_optimizer.rolling_mean(yz_volatility, window=self.config.period)
            volatility = np.sqrt(volatility)
            
            return volatility.rename(f'yang_zhang_volatility_{self.config.period}')
            
        except Exception as e:
            logger.warning(f"Yang-Zhang volatility calculation failed: {e}, using pandas fallback")
            return self._calculate_yang_zhang_volatility_pandas(data)

    def _calculate_yang_zhang_volatility_pandas(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Yang-Zhang volatility using pandas fallback."""
        open_price = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        log_co = np.log(close / open_price)
        overnight_vol = log_co**2
        
        log_ho = np.log(high / open_price)
        log_hc = np.log(high / close)
        log_lo = np.log(low / open_price)
        log_lc = np.log(low / close)
        rs_volatility = log_ho * log_hc + log_lo * log_lc
        
        log_hl = np.log(high / low)
        gk_volatility = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        
        yz_volatility = overnight_vol + rs_volatility + gk_volatility
        
        volatility = yz_volatility.rolling(window=self.config.period).mean()
        volatility = np.sqrt(volatility)
        
        return volatility.rename(f'yang_zhang_volatility_{self.config.period}')

    def _calculate_volatility_of_volatility(self, volatility_series: pd.Series) -> pd.Series:
        """Calculate volatility of volatility."""
        if self.rolling_optimizer:
            try:
                return self.rolling_optimizer.rolling_std(volatility_series, window=self.config.period).rename(f'volatility_of_volatility_{self.config.period}')
            except Exception:
                pass
        
        # Fallback to pandas
        return volatility_series.rolling(window=self.config.period).std().rename(f'volatility_of_volatility_{self.config.period}')

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_operations'] > 0:
            stats['avg_time_per_operation'] = stats['total_time'] / stats['total_operations']
            stats['vectorbt_usage_rate'] = stats['vectorbt_operations'] / stats['total_operations']
            stats['unified_manager_usage_rate'] = stats['unified_manager_operations'] / stats['total_operations']
            stats['rolling_optimizer_usage_rate'] = stats['rolling_optimizer_operations'] / stats['total_operations']
            stats['pandas_fallback_rate'] = stats['pandas_fallbacks'] / stats['total_operations']
        
        return stats

    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for vectorized processing."""
        if self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        return data


def create_enhanced_volatility_generators(periods: List[int] = [10, 14, 20, 30, 50],
                                        std_devs: List[float] = [1.5, 2.0, 2.5],
                                        enable_gpu: bool = False,
                                        enable_parallel: bool = True) -> List[FeatureGenerator]:
    """Create enhanced volatility generators with comprehensive VectorBT optimization."""
    generators = []
    
    for period in periods:
        for std_dev in std_devs:
            config = VolatilityConfig(
                period=period,
                std_dev=std_dev,
                enable_gpu=enable_gpu,
                enable_parallel=enable_parallel,
                use_unified_manager=True,
                use_rolling_optimizer=True
            )
            generators.append(EnhancedVectorBTVolatilityGenerator(config))
    
    return generators


def create_default_enhanced_volatility_generators() -> List[FeatureGenerator]:
    """Create default enhanced volatility generators."""
    return create_enhanced_volatility_generators()


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=5000, freq='1min')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(5000) * 0.01) + np.random.rand(5000) * 2,
        'low': 100 + np.cumsum(np.random.randn(5000) * 0.01) - np.random.rand(5000) * 2,
        'close': 100 + np.cumsum(np.random.randn(5000) * 0.01),
        'volume': np.random.lognormal(10, 1, 5000)
    }, index=dates)
    
    # Test enhanced generator
    config = VolatilityConfig(period=20, enable_gpu=False, enable_parallel=True)
    generator = EnhancedVectorBTVolatilityGenerator(config)
    
    print("Testing Enhanced VectorBT Volatility Generator...")
    
    # Test basic volatility
    volatility = generator._generate_feature(data)
    print(f"Basic volatility shape: {volatility.shape}")
    
    # Test comprehensive features
    comprehensive_features = generator.generate_comprehensive_volatility_features(data)
    print(f"Comprehensive features shape: {comprehensive_features.shape}")
    print(f"Features: {list(comprehensive_features.columns)}")
    
    # Performance stats
    stats = generator.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("✅ Enhanced VectorBT Volatility Generator test completed!")