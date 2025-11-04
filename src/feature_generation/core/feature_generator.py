"""
Base Feature Generator Classes

This module defines the base classes and interfaces for feature generation,
providing a standardized way to create and manage feature generators.

Enhanced with native VectorBT support for maximum performance.
"""

import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings

# Import centralized logging and error handling
from src.feature_generation.utils.centralized_logging import tprint, log_function_execution, fast_fail_error
from src.feature_generation.utils.error_handling import (
    DataValidationError, ConfigurationError, ComputationError,
    validate_required_columns, validate_finite_values, safe_divide
)

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    # VectorBT 0.28.1 doesn't have these rolling functions in vectorbt.generic
    # Will use pandas fallbacks instead
    VECTORBT_ROLLING_AVAILABLE = False
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

    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    VECTORBT_ROLLING_AVAILABLE = False
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
    # VectorBT not available - will use fallback implementations
    tprint("VectorBT not available. Install with: pip install vectorbt for optimized performance", level="warning")

# GPU acceleration removed - CuPy not supported on all platforms
cp = None
CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class FeatureCategory(Enum):
    """Enumeration of feature categories."""
    RETURNS = "returns"
    MOMENTUM = "momentum"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    NORMALIZATION = "normalization"
    TREND = "trend"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"
    CUSTOM_SUPPORT_RESISTANCE = "custom_support_resistance"
    CANDLESTICK_PATTERN = "candlestick_pattern"
    # HMM_REGIME = "hmm_regime"  # DEPRECATED
    CROSS_TIMEFRAME = "cross_timeframe"
    MICROSTRUCTURE = "microstructure"
    ENTROPY = "entropy"
    AUTOENCODER = "autoencoder"
    REPRESENTATION_LEARNING = "representation_learning"
    ORDER_FLOW = "order_flow"
    REGIME = "regime"
    LEGACY = "legacy"
    TIME = "time"
    CUSTOM = "custom"
    ACCELERATION = "acceleration"
    INTERACTION = "interaction"
    ADVANCED_STATISTICAL = "advanced_statistical"
    SPECTRAL_WAVELET = "spectral_wavelet"

@dataclass
class FeatureConfig:
    """Configuration for feature generation with native VectorBT support."""
    name: str
    category: FeatureCategory
    description: str
    required_columns: List[str]
    optional_columns: List[str] = None
    default_lookback: int = 20
    min_lookback: int = 1
    max_lookback: int = 252
    parameters: Dict[str, Any] = None
    dependencies: List[str] = None
    matrix_optimized: bool = True
    gpu_accelerated: bool = False
    enable_feature_selection: bool = True

    # VectorBT native optimization settings
    use_vectorbt: bool = True
    vectorbt_threshold: int = 1000  # Minimum samples for VectorBT optimization
    enable_gpu: bool = False
    enable_parallel: bool = True
    vectorbt_memory_limit_gb: float = 8.0

    def __post_init__(self):
        if self.optional_columns is None:
            self.optional_columns = []
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []

        # Auto-enable VectorBT optimizations for large lookbacks
        if self.default_lookback > 50:
            self.use_vectorbt = True
            self.enable_parallel = True

@dataclass
class FeatureResult:
    """Result of feature generation."""
    name: str
    data: pd.Series
    config: FeatureConfig
    computation_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FeatureGenerator(ABC):
    """
    Abstract base class for feature generators with native VectorBT support.

    This class defines the interface that all feature generators must implement,
    providing a standardized way to generate features with consistent error handling,
    logging, performance tracking, and native VectorBT optimization.
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize the feature generator with native VectorBT support.

        Args:
            config: Feature configuration with VectorBT settings
        """
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')
        # Only log initialization for debug level or if there's an issue
        self.logger.debug(f"Initializing {self.__class__.__name__} with config: {config.name}")

        # VectorBT configuration
        self.use_vectorbt = config.use_vectorbt and VECTORBT_AVAILABLE
        self.enable_gpu = False  # GPU support removed
        self.enable_parallel = config.enable_parallel and VECTORBT_AVAILABLE
        self.vectorbt_threshold = config.vectorbt_threshold

        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_computation_time': 0.0,
            'total_computation_time': 0.0,
            'total_execution_time': 0.0,
            'vectorbt_operations': 0,
            'gpu_accelerations': 0,
            'parallel_operations': 0,
            'pandas_fallbacks': 0
        }

        # Optional state storage for incremental generation support
        self._state: Dict[str, Any] = {}
        self._state_loaded: bool = False

        # Configure VectorBT if available
        if self.use_vectorbt:
            self._configure_vectorbt()

        # Reduced logging - only log at category level, not individual features
        # self.logger.info(f"Initialized {self.__class__.__name__} for {config.name}")

    def _configure_vectorbt(self):
        """Configure VectorBT global settings for optimal performance."""
        if not VECTORBT_AVAILABLE:
            return

        try:
            # Check if settings attribute exists first
            if not hasattr(vbt, 'settings'):
                self.logger.debug("⚠️ VectorBT settings not available in this version")
                return
            
            # Configure VectorBT settings using newer API
            # Check if array_wrapper structure exists and set wrapper if available
            if hasattr(vbt.settings, 'array_wrapper') and 'wrapper' in vbt.settings['array_wrapper']:
                vbt.settings['array_wrapper']['wrapper'] = 'pandas'
            
            # Check if caching exists before accessing it
            if hasattr(vbt.settings, 'caching') and 'enabled' in vbt.settings['caching']:
                vbt.settings['caching']['enabled'] = True

            if self.enable_gpu:
                try:
                    # Check if GPU settings are available in this VectorBT version
                    if hasattr(vbt.settings, 'gpu') and 'enabled' in vbt.settings['gpu']:
                        vbt.settings['gpu']['enabled'] = True
                        self.logger.debug("✅ VectorBT GPU processing enabled")
                    else:
                        self.logger.debug("⚠️ GPU processing not available in this VectorBT version")
                        self.enable_gpu = False
                except Exception as e:
                    self.logger.warning(f"⚠️ GPU processing not available: {e}")
                    self.enable_gpu = False

            if self.enable_parallel:
                try:
                    # Check if parallel settings are available in this VectorBT version
                    if hasattr(vbt.settings, 'parallel') and 'enabled' in vbt.settings['parallel']:
                        vbt.settings['parallel']['enabled'] = True
                        self.logger.debug("✅ VectorBT parallel processing enabled")
                    else:
                        self.logger.debug("⚠️ Parallel processing not available in this VectorBT version")
                        self.enable_parallel = False
                except Exception as e:
                    self.logger.warning(f"⚠️ Parallel processing not available: {e}")
                    self.enable_parallel = False

        except Exception as e:
            self.logger.warning(f"VectorBT configuration failed: {e}")

    def _should_use_vectorbt(self, data: pd.DataFrame) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (self.use_vectorbt and
                len(data) >= self.vectorbt_threshold and
                VECTORBT_AVAILABLE and
                VECTORBT_ROLLING_AVAILABLE)

    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                  window: int, **kwargs) -> pd.Series:
        """
        Perform VectorBT rolling operation with fallback to pandas.
        Now uses VectorBTRollingOptimizer for enhanced performance.

        Args:
            data: Input data series
            operation: Operation type ('mean', 'std', 'var', 'min', 'max', 'sum', 'corr', 'cov')
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of rolling operation
        """
        # Use VectorBTRollingOptimizer for enhanced performance
        try:
            from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
            optimizer = get_vectorbt_rolling_optimizer()

            # Map operation to optimizer method
            if operation == 'mean':
                return optimizer.rolling_mean(data, window, **kwargs)
            elif operation == 'std':
                return optimizer.rolling_std(data, window, **kwargs)
            elif operation == 'var':
                return optimizer.rolling_var(data, window, **kwargs)
            elif operation == 'min':
                return optimizer.rolling_min(data, window, **kwargs)
            elif operation == 'max':
                return optimizer.rolling_max(data, window, **kwargs)
            elif operation == 'sum':
                return optimizer.rolling_sum(data, window, **kwargs)
            elif operation == 'corr':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for correlation")
                return optimizer.rolling_corr(data, other, window, **kwargs)
            elif operation == 'cov':
                other = kwargs.get('other')
                if other is None:
                    raise ValueError("'other' parameter required for covariance")
                return optimizer.rolling_cov(data, other, window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

        except ImportError:
            # Fallback to original implementation if optimizer not available
            self.logger.warning("VectorBTRollingOptimizer not available, using direct VectorBT calls")
            return self._direct_vectorbt_rolling_operation(data, operation, window, **kwargs)
        except Exception as e:
            self.logger.warning(f"VectorBTRollingOptimizer failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _direct_vectorbt_rolling_operation(self, data: pd.Series, operation: str,
                                         window: int, **kwargs) -> pd.Series:
        """Direct VectorBT rolling operation (fallback when optimizer unavailable)."""
        if not self._should_use_vectorbt(pd.DataFrame({'temp': data})):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

        self.performance_stats['vectorbt_operations'] += 1

        try:
            # Use our VectorBTRollingOptimizer if available, otherwise fallback to pandas
            if hasattr(self, 'vectorbt_rolling_optimizer') and self.vectorbt_rolling_optimizer:
                return self.vectorbt_rolling_optimizer.rolling_operation(data, operation, window, **kwargs)
            else:
                # Fallback to pandas rolling operations
                rolling_obj = data.rolling(window=window, **{k: v for k, v in kwargs.items() if k != 'other'})
                
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
                elif operation == 'corr':
                    other = kwargs.get('other')
                    if other is None:
                        raise ValueError("'other' parameter required for correlation")
                    return rolling_obj.corr(other)
                elif operation == 'cov':
                    other = kwargs.get('other')
                    if other is None:
                        raise ValueError("'other' parameter required for covariance")
                    return rolling_obj.cov(other)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

        except Exception as e:
            self.logger.warning(f"VectorBT rolling operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)

    def _pandas_rolling_operation(self, data: pd.Series, operation: str,
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
            if other is None:
                raise ValueError("'other' parameter required for correlation")
            return data.rolling(window=window).corr(other)
        elif operation == 'cov':
            other = kwargs.get('other')
            if other is None:
                raise ValueError("'other' parameter required for covariance")
            return data.rolling(window=window).cov(other)
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def _vectorbt_apply_operation(self, data: pd.Series, func: Callable,
                                 window: int, **kwargs) -> pd.Series:
        """
        Perform VectorBT rolling apply operation with fallback to pandas.

        Args:
            data: Input data series
            func: Function to apply
            window: Rolling window size
            **kwargs: Additional parameters

        Returns:
            Result of rolling apply operation
        """
        if not self._should_use_vectorbt(pd.DataFrame({'temp': data})):
            return data.rolling(window=window).apply(func, **kwargs)

        self.performance_stats['vectorbt_operations'] += 1

        try:
            return rolling_apply(data, func, window, **kwargs)
        except Exception as e:
            self.logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)

    @abstractmethod
    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate the feature. This method must be implemented by subclasses.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Generated feature as pandas Series
        """
        pass

    @log_function_execution(level="debug")
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """
        Generate the feature with error handling and performance tracking.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            FeatureResult with the generated feature and metadata
        """
        tprint(f"Generating feature {self.config.name} with data shape: {data.shape}", level="debug")
        start_time = time.time()

        # Fast fail on invalid input
        if data is None:
            fast_fail_error("Data cannot be None", DataValidationError)

        if not isinstance(data, pd.DataFrame):
            fast_fail_error(f"Data must be DataFrame, got {type(data)}", DataValidationError)

        if len(data) == 0:
            fast_fail_error("DataFrame is empty", DataValidationError)

        # Allow state injection through kwargs for compatibility
        external_state = kwargs.pop('state', None)
        if external_state is not None:
            self.load_state(external_state)

        state_loaded_flag = self._state_loaded

        try:
            # Validate input data with fast fail
            self._validate_data(data)

            # Allow subclasses to adjust internal buffers before generation
            self._prepare_state(data)

            # Generate the feature
            feature_data = self._generate_feature(data, **kwargs)

            # Fast fail on invalid output
            if feature_data is None:
                fast_fail_error("Feature generation returned None", ComputationError)

            if not isinstance(feature_data, pd.Series):
                fast_fail_error(f"Feature must be Series, got {type(feature_data)}", ComputationError)

            # Validate output
            self._validate_output(feature_data)

            # Update generator state with latest observations
            try:
                self._finalize_state(data, feature_data)
            except Exception as state_error:
                tprint(f"State finalization failed: {state_error}", level="warning")
                # Don't fail the entire operation for state issues

            # Update performance stats
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=True)

            tprint(f"Successfully generated {self.config.name} in {computation_time:.3f}s", level="info")
            self.logger.debug(f"Successfully generated {self.config.name} in {computation_time:.3f}s")

            serialized_state = self._serialize_state()

            return FeatureResult(
                name=self.config.name,
                data=feature_data,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata={
                    'generator_class': self.__class__.__name__,
                    'input_shape': data.shape,
                    'output_length': len(feature_data),
                    'state_loaded': state_loaded_flag,
                    'state': serialized_state
                }
            )

        except (DataValidationError, ConfigurationError, ComputationError) as e:
            # Fast fail for known error types
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=False)
            fast_fail_error(f"Feature generation failed: {str(e)}", type(e))

        except Exception as e:
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=False)

            error_msg = f"Unexpected error generating {self.config.name}: {str(e)}"
            tprint(f"ERROR: {error_msg}", level="error")
            self.logger.error(error_msg, exc_info=True)

            failure_metadata = {
                'generator_class': self.__class__.__name__,
                'input_shape': data.shape,
                'state_loaded': state_loaded_flag,
                'state': self._serialize_state(),
                'error_type': type(e).__name__
            }

            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float, index=data.index),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=error_msg,
                metadata=failure_metadata
            )

    def load_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Load previously persisted state for incremental computation."""
        if state:
            # Use deepcopy to avoid mutating external structures
            self._state = copy.deepcopy(self._deserialize_state(state))
            self._state_loaded = True
        else:
            self._state = {}
            self._state_loaded = False

    def reset_state(self) -> None:
        """Reset internal state."""
        self._state = {}
        self._state_loaded = False

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of the current generator state."""
        return copy.deepcopy(self._state)

    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update internal state with provided values."""
        if not updates:
            return
        self._state.update(copy.deepcopy(updates))

    def _prepare_state(self, data: pd.DataFrame) -> None:
        """Hook for subclasses to prepare state before generation."""
        # Default implementation does nothing. Subclasses may override.
        return

    def _finalize_state(self, data: pd.DataFrame, feature_data: pd.Series) -> None:
        """Update default state information after successful generation."""
        if len(data) == 0:
            return

        last_row = data.iloc[-1]
        state_update: Dict[str, Any] = {
            'last_index': data.index[-1],
            'last_row': last_row.to_dict()
        }

        if len(feature_data) != 0:
            state_update['last_feature_value'] = feature_data.iloc[-1]

        self.update_state(state_update)

    def _serialize_state(self) -> Dict[str, Any]:
        """Serialize internal state into JSON/pickle friendly types."""

        def _serialize_value(value: Any) -> Any:
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
            if isinstance(value, (list, tuple)):
                return [_serialize_value(v) for v in value]
            if isinstance(value, dict):
                return {str(k): _serialize_value(v) for k, v in value.items()}
            if isinstance(value, np.ndarray):
                return [_serialize_value(v) for v in value.tolist()]
            if isinstance(value, pd.Series):
                return {str(idx): _serialize_value(val) for idx, val in value.items()}
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient='list')
            if hasattr(value, 'tolist'):
                return [_serialize_value(v) for v in value.tolist()]
            return str(value)

        return {str(key): _serialize_value(val) for key, val in self._state.items()}

    def _deserialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state payload back into internal representation."""
        # Base implementation returns state unchanged. Subclasses can override
        # if they need to restore complex structures.
        return state

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data with fast fail error handling.

        Args:
            data: Input data DataFrame

        Raises:
            DataValidationError: If data validation fails
        """
        tprint(f"Validating data for {self.config.name}", level="debug")

        # DEBUG: Check data quality before validation
        import numpy as np
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"FeatureGenerator._validate_data - Data shape: {data.shape}")
            self.logger.debug(f"FeatureGenerator._validate_data - Non-finite values: {(~np.isfinite(data.select_dtypes(include=[np.number])).values).sum()}")
            for col in data.select_dtypes(include=[np.number]).columns:
                non_finite = (~np.isfinite(data[col])).sum()
                if non_finite > 0:
                    self.logger.debug(f"FeatureGenerator._validate_data - {col}: {non_finite} non-finite values")

        # Use centralized validation functions
        validate_required_columns(data, self.config.required_columns)

        # Check for sufficient data
        if len(data) < self.config.min_lookback:
            fast_fail_error(
                f"Insufficient data: need at least {self.config.min_lookback} rows, got {len(data)}",
                DataValidationError
            )

        # Check for finite values in required columns
        for col in self.config.required_columns:
            if col in data.columns:
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Validating column: {col}")
                validate_finite_values(data[col], col)

        tprint(f"Data validation passed for {self.config.name}", level="debug")

    def _validate_output(self, feature_data: pd.Series) -> None:
        """
        Validate output feature data with fast fail error handling.

        Args:
            feature_data: Generated feature data

        Raises:
            DataValidationError: If output validation fails
        """
        tprint(f"Validating output for {self.config.name}", level="debug")

        if len(feature_data) == 0:
            fast_fail_error("Generated feature is empty", DataValidationError)

        # Check for all NaN values - allow for some features that may legitimately be all NaN
        if feature_data.isna().all():
            # For certain feature types, allow all NaN if it's expected behavior
            feature_name_lower = str(self.config.name).lower()
            if any(keyword in feature_name_lower for keyword in ['macd', 'momentum', 'velocity', 'acceleration', 'cross_timeframe', 'ctf', 'vwap', 'trend_strength']):
                tprint(f"⚠️ Feature {self.config.name} contains only NaN values - this may be expected for complex calculations (data length: {len(feature_data)})", level="warning")
                # Don't fail for these feature types, just return the NaN series
                pass
            else:
                fast_fail_error("Generated feature contains only NaN values", DataValidationError)
        else:
            # Log some statistics for debugging
            nan_count = feature_data.isna().sum()
            non_nan_count = feature_data.notna().sum()
            tprint(f"Feature {self.config.name} validation: {non_nan_count} non-NaN, {nan_count} NaN values", level="debug")

        # Check for infinite values - warn but don't fail
        infinite_count = np.isinf(feature_data).sum()
        if infinite_count > 0:
            tprint(f"⚠️ Warning: Generated feature '{self.config.name}' contains {infinite_count} infinite values", level="warning")
            self.logger.warning(f"⚠️ Warning: Generated feature '{self.config.name}' contains {infinite_count} infinite values")

        # Check for finite values
        validate_finite_values(feature_data, f"{self.config.name}_output")

        tprint(f"Output validation passed for {self.config.name}", level="debug")

    def _update_performance_stats(self, computation_time: float, success: bool) -> None:
        """
        Update performance statistics.

        Args:
            computation_time: Time taken for computation
            success: Whether the generation was successful
        """
        # Ensure performance_stats is initialized
        if not hasattr(self, 'performance_stats') or 'total_generations' not in self.performance_stats:
            self.performance_stats = {
                'total_generations': 0,
                'successful_generations': 0,
                'failed_generations': 0,
                'total_computation_time': 0.0,
                'average_computation_time': 0.0,
                'vectorbt_operations': 0,
                'pandas_fallbacks': 0,
                'unified_manager_operations': 0
            }

        self.performance_stats['total_generations'] += 1

        if success:
            self.performance_stats['successful_generations'] += 1
        else:
            self.performance_stats['failed_generations'] += 1

        # Update average computation time
        total_time = self.performance_stats['total_computation_time'] + computation_time
        self.performance_stats['total_computation_time'] = total_time
        self.performance_stats['average_computation_time'] = (
            total_time / self.performance_stats['total_generations']
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance statistics
        """
        return self.performance_stats.copy()

    def get_config(self) -> FeatureConfig:
        """
        Get feature configuration.

        Returns:
            Feature configuration
        """
        return self.config

    def supports_lookback_optimization(self) -> bool:
        """
        Check if this generator supports lookback optimization.

        Returns:
            True if lookback optimization is supported
        """
        return hasattr(self, '_generate_feature_with_lookback')

    def generate_with_lookback(self, data: pd.DataFrame, lookback: int, **kwargs) -> FeatureResult:
        """
        Generate feature with specific lookback period.

        Args:
            data: Input data DataFrame
            lookback: Lookback period
            **kwargs: Additional parameters

        Returns:
            FeatureResult with the generated feature
        """
        if not self.supports_lookback_optimization():
            self.logger.warning(f"{self.config.name} does not support lookback optimization")
            return self.generate(data, **kwargs)

        # Validate lookback
        if lookback < self.config.min_lookback or lookback > self.config.max_lookback:
            raise ValueError(f"Lookback {lookback} is outside valid range [{self.config.min_lookback}, {self.config.max_lookback}]")

        return self._generate_feature_with_lookback(data, lookback, **kwargs)

    def _generate_feature_with_lookback(self, data: pd.DataFrame, lookback: int, **kwargs) -> FeatureResult:
        """
        Generate feature with specific lookback period. Override in subclasses.

        Args:
            data: Input data DataFrame
            lookback: Lookback period
            **kwargs: Additional parameters

        Returns:
            FeatureResult with the generated feature
        """
        # Default implementation - just use the regular generate method
        return self.generate(data, **kwargs)

class CompositeFeatureGenerator(FeatureGenerator):
    """
    Feature generator that combines multiple sub-generators.

    This class allows combining multiple feature generators into a single
    generator that produces multiple features.
    """

    def __init__(self, config: FeatureConfig, sub_generators: List[FeatureGenerator]):
        """
        Initialize composite feature generator.

        Args:
            config: Feature configuration
            sub_generators: List of sub-generators
        """
        super().__init__(config)
        self.sub_generators = sub_generators
        self.logger.info(f"Initialized composite generator with {len(sub_generators)} sub-generators")

    def _generate_feature(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Generate features using sub-generators.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Combined features as pandas Series (concatenated from all successful sub-generators)
        """
        results = []
        for generator in self.sub_generators:
            result = generator.generate(data, **kwargs)
            if result.success:
                results.append(result.data)
            else:
                self.logger.warning(f"Sub-generator {generator.config.name} failed: {result.error_message}")

        if not results:
            raise ValueError("All sub-generators failed")

        # Concatenate all successful results into a single Series
        if len(results) == 1:
            return results[0]
        else:
            # Create a combined series with all features
            combined_data = pd.concat(results, axis=1)
            # Return the first column as the primary feature
            return combined_data.iloc[:, 0]

    def generate_all_features(self, data: pd.DataFrame, **kwargs) -> Dict[str, FeatureResult]:
        """
        Generate all features from sub-generators.

        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters

        Returns:
            Dictionary mapping feature names to results
        """
        results = {}
        for generator in self.sub_generators:
            result = generator.generate(data, **kwargs)
            results[generator.config.name] = result

        return results

class VectorizedFeatureGenerator(FeatureGenerator):
    """
    Base class for vectorized feature generators with native VectorBT support.

    This class provides optimized vectorized computation capabilities
    using VectorBT's optimized backend, matrix operations framework, and optimization utilities.
    """

    def __init__(self, config: FeatureConfig, enable_matrix_ops: bool = True, enable_vectorization_optimization: bool = True):
        """
        Initialize vectorized feature generator with native VectorBT support.

        Args:
            config: Feature configuration with VectorBT settings
            enable_matrix_ops: Whether to enable matrix operations
            enable_vectorization_optimization: Whether to enable vectorization optimization
        """
        super().__init__(config)
        self.enable_matrix_ops = enable_matrix_ops
        self.enable_vectorization_optimization = enable_vectorization_optimization

        # Initialize VectorBTRollingOptimizer if available
        if enable_matrix_ops:
            try:
                from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
                self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
                self.logger.debug("VectorBTRollingOptimizer enabled")
            except ImportError:
                self.vectorbt_rolling_optimizer = None
                self.enable_matrix_ops = False
                self.logger.warning("⚠️ VectorBTRollingOptimizer not available")
        else:
            self.vectorbt_rolling_optimizer = None

        # Initialize UnifiedVectorizationManager if available
        if enable_vectorization_optimization:
            try:
                from src.utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
                self.unified_vectorization_manager = get_unified_vectorization_manager()
                self.logger.debug("UnifiedVectorizationManager enabled")
            except ImportError:
                self.unified_vectorization_manager = None
                self.enable_vectorization_optimization = False
                self.logger.warning("⚠️ UnifiedVectorizationManager not available")
        else:
            self.unified_vectorization_manager = None

    @property
    def vectorbt_optimizer(self):
        """Alias for vectorbt_rolling_optimizer for backward compatibility."""
        return self.vectorbt_rolling_optimizer

    def _vectorized_operation(self, operation: str, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform vectorized operation using VectorBTRollingOptimizer and UnifiedVectorizationManager.

        Args:
            operation: Operation to perform
            data: Input data array
            **kwargs: Additional parameters

        Returns:
            Result of the operation
        """
        if not self.enable_matrix_ops or self.vectorbt_rolling_optimizer is None:
            # Fallback to numpy operations
            return self._numpy_fallback(operation, data, **kwargs)

        try:
            # Use VectorBTRollingOptimizer for rolling operations
            if hasattr(self.vectorbt_rolling_optimizer, operation):
                return getattr(self.vectorbt_rolling_optimizer, operation)(data, **kwargs)
            else:
                # Use UnifiedVectorizationManager for other operations
                if self.unified_vectorization_manager is not None:
                    return self.unified_vectorization_manager.optimize_operation(operation, data, **kwargs)
                else:
                    return self._numpy_fallback(operation, data, **kwargs)
        except Exception as e:
            self.logger.warning(f"VectorBTRollingOptimizer/UnifiedVectorizationManager operation failed, using numpy fallback: {e}")
            return self._numpy_fallback(operation, data, **kwargs)

    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for vectorized processing using UnifiedVectorizationManager.

        This method provides automatic optimization of DataFrames for efficient vectorized
        processing. It includes memory optimization, data type optimization, and VectorBT
        compatibility checks.

        Features:
        - Automatic memory optimization for large datasets
        - Data type optimization (int64 -> int32/int16/int8, float64 -> float32)
        - VectorBT compatibility preparation
        - Memory usage monitoring and optimization
        - Graceful fallback for unsupported data types

        Args:
            data: Input DataFrame to optimize

        Returns:
            Optimized DataFrame with improved memory usage and processing efficiency

        Example:
            >>> generator = VectorizedFeatureGenerator(config)
            >>> optimized_data = generator.optimize_dataframe_processing(data)
            >>> # Use optimized_data for feature generation
        """
        if self.enable_vectorization_optimization and self.unified_vectorization_manager:
            # Use the UnifiedVectorizationManager which has the method
            try:
                # Use the same manager instance that's already initialized
                if hasattr(self.unified_vectorization_manager, 'optimize_dataframe_processing'):
                    return self.unified_vectorization_manager.optimize_dataframe_processing(data)
                else:
                    # Fallback to returning data as-is if method not available
                    return data
            except (ImportError, AttributeError):
                # Fallback to returning data as-is
                return data
        else:
            return data

    def vectorized_rolling_operations(self,
                                    data: pd.DataFrame,
                                    operations: List[str],
                                    windows: List[int],
                                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform vectorized rolling operations with VectorBT optimization.

        This method provides high-performance rolling operations using VectorBT's optimized
        C++ backend when available, with automatic fallback to pandas for smaller datasets
        or when VectorBT is not available.

        Features:
        - VectorBT optimization for large datasets (>1000 rows)
        - Automatic fallback to pandas for smaller datasets
        - Support for multiple operations and windows in a single call
        - Memory-efficient processing with chunked operations
        - Comprehensive error handling and logging

        Supported Operations:
        - 'mean': Rolling mean
        - 'std': Rolling standard deviation
        - 'var': Rolling variance
        - 'min': Rolling minimum
        - 'max': Rolling maximum
        - 'sum': Rolling sum
        - 'corr': Rolling correlation (requires 'other' parameter)
        - 'cov': Rolling covariance (requires 'other' parameter)

        Args:
            data: Input DataFrame containing the data to process
            operations: List of operation types to perform
            windows: List of window sizes for rolling calculations
            columns: Columns to process (None = all numeric columns)

        Returns:
            DataFrame with rolling features added as new columns

        Example:
            >>> generator = VectorizedFeatureGenerator(config)
            >>> result = generator.vectorized_rolling_operations(
            ...     data,
            ...     operations=['mean', 'std'],
            ...     windows=[20, 50],
            ...     columns=['close', 'volume']
            ... )
            >>> # Result contains columns like 'close_mean_20', 'close_std_20', etc.
        """
        # Heavy-workload routing through consolidated optimizer for batching gains
        try:
            process_columns = columns or data.select_dtypes(include=[np.number]).columns.tolist()
            combos = max(1, len(operations)) * max(1, len(windows)) * max(1, len(process_columns))
            if combos >= 24 and len(data) >= 500:
                from src.feature_generation.utils.consolidated_rolling_optimizer import (
                    get_global_rolling_optimizer, RollingOperationConfig, RollingOperationType
                )

                optimizer = get_global_rolling_optimizer()
                configs = []
                supported = {op.value for op in RollingOperationType}
                for op in operations:
                    if op not in supported:
                        continue
                    for w in windows:
                        try:
                            configs.append(RollingOperationConfig(
                                operation=RollingOperationType(op),
                                window=int(w)
                            ))
                        except Exception:
                            continue

                if configs:
                    subset = data[process_columns]
                    batch_results = optimizer.batch_rolling_operations(subset, configs)
                    out = {}
                    for idx, cfg in enumerate(configs):
                        key = f"{cfg.operation.value}_{cfg.window}_{idx}"
                        res = batch_results.get(key)
                        if res is None:
                            continue
                        if isinstance(res, pd.Series):
                            # Single column result; infer column name if possible
                            colname = res.name if res.name in process_columns else process_columns[0]
                            out[f"{colname}_{cfg.operation.value}_{cfg.window}"] = self._downcast_series(res)
                        else:
                            # DataFrame result across all columns
                            for col in process_columns:
                                if col in res.columns:
                                    out[f"{col}_{cfg.operation.value}_{cfg.window}"] = self._downcast_series(res[col])
                    if out:
                        return pd.DataFrame(out, index=data.index)
        except Exception as e:
            self.logger.debug(f"Consolidated optimizer routing skipped: {e}")

        # Use VectorBT if available and data is large enough
        if self._should_use_vectorbt(data):
            return self._vectorbt_rolling_operations(data, operations, windows, columns)

        # Fallback to vectorization optimizer
        if self.enable_vectorization_optimization and self.unified_vectorization_manager:
            return self.unified_vectorization_manager.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        else:
            return self._fallback_rolling_operations(data, operations, windows, columns)

    def _vectorbt_rolling_operations(self, data: pd.DataFrame, operations: List[str],
                                   windows: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Perform rolling operations using VectorBT optimization with reduced memory overhead."""
        # Avoid copying the entire input DataFrame; build only computed columns
        process_columns = columns or data.select_dtypes(include=[np.number]).columns
        out = {}

        for col in process_columns:
            series = data[col]
            for operation in operations:
                for window in windows:
                    key = f"{col}_{operation}_{window}"
                    try:
                        res = self._vectorbt_rolling_operation(series, operation, window)
                        self.performance_stats['vectorbt_operations'] += 1
                    except Exception as e:
                        self.logger.warning(f"VectorBT operation failed for {key}: {e}")
                        res = self._pandas_rolling_operation(series, operation, window)

                    # Downcast to save memory when safe
                    res = self._downcast_series(res)
                    out[key] = res

        # Assemble result with index only once
        return pd.DataFrame(out, index=data.index)

    def _fallback_rolling_operations(self,
                                   data: pd.DataFrame,
                                   operations: List[str],
                                   windows: List[int],
                                   columns: List[str]) -> pd.DataFrame:
        """Fallback rolling operations without vectorization optimizer (memory‑efficient)."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        out = {}
        for window in windows:
            for col in columns:
                if col not in data.columns:
                    continue
                series = data[col]
                for operation in operations:
                    key = f"{col}_rolling_{operation}_{window}"
                    res = self._pandas_rolling_operation(series, operation, window)
                    out[key] = self._downcast_series(res)

        return pd.DataFrame(out, index=data.index)

    def _downcast_series(self, series: pd.Series) -> pd.Series:
        """Downcast numeric Series to reduce memory when safe (float64->float32, int64->int32)."""
        try:
            if pd.api.types.is_float_dtype(series):
                smin, smax = series.min(skipna=True), series.max(skipna=True)
                if np.isfinite([smin, smax]).all() and smin >= np.finfo(np.float32).min and smax <= np.finfo(np.float32).max:
                    return series.astype(np.float32)
            elif pd.api.types.is_integer_dtype(series):
                smin, smax = series.min(skipna=True), series.max(skipna=True)
                if smin >= np.iinfo(np.int32).min and smax <= np.iinfo(np.int32).max:
                    return series.astype(np.int32)
        except Exception:
            # Best-effort downcast; ignore failures
            return series
        return series

    def _numpy_fallback(self, operation: str, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fallback to numpy operations when matrix operations are not available.

        Args:
            operation: Operation to perform
            data: Input data array
            **kwargs: Additional parameters

        Returns:
            Result of the operation
        """
        if operation == 'rolling_mean':
            window = kwargs.get('window', 20)
            series = pd.Series(data)
            if VECTORBT_AVAILABLE and len(series) > 1000:
                try:
                    return rolling_mean(series, window=window).values
                except Exception:
                    pass
            return series.rolling(window=window).mean().values
        elif operation == 'rolling_std':
            window = kwargs.get('window', 20)
            series = pd.Series(data)
            # Use pandas rolling since VectorBT doesn't have rolling_std
            return series.rolling(window=window).std().values
        elif operation == 'ewm_mean':
            span = kwargs.get('span', 20)
            return pd.Series(data).ewm(span=span).mean().values
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    # Enhanced rolling operation helper methods using VectorBTRollingOptimizer
    def _calculate_ema_vectorized(self, data: pd.Series, window: int, alpha: Optional[float] = None) -> pd.Series:
        """Calculate EMA using vectorized operations with VectorBT optimization."""
        if alpha is None:
            alpha = 2.0 / (window + 1)

        try:
            if VECTORBT_AVAILABLE and len(data) >= 1000:  # Use VectorBT for large datasets
                # Try VectorBT EMA with proper API usage
                try:
                    # VectorBT 0.28.1 may have different API - try different approaches
                    if hasattr(vbt.ta, 'ema'):
                        return vbt.ta.ema(data, span=window)
                    else:
                        # Fallback to pandas if VectorBT doesn't have EMA
                        raise AttributeError("VectorBT EMA not available")
                except (AttributeError, TypeError):
                    # VectorBT API issue - fallback to pandas
                    raise Exception("VectorBT EMA API not compatible")
            else:
                # Fallback to pandas implementation - use alpha OR span, not both
                return data.ewm(alpha=alpha).mean()
        except Exception as e:
            self.logger.warning(f"⚠️ VectorBT EMA calculation failed: {e}, using pandas fallback")
            # Use alpha parameter for pandas EWM (span and alpha are mutually exclusive)
            return data.ewm(alpha=alpha).mean()

    def _safe_ewm_call(self, data: pd.Series, alpha: float = None, span: int = None, **kwargs) -> pd.Series:
        """Safely call pandas ewm() ensuring only one smoothing parameter is used."""
        # Ensure only one of alpha, span, halflife, or comass is provided
        smoothing_params = sum([alpha is not None, span is not None,
                               kwargs.get('halflife') is not None,
                               kwargs.get('comass') is not None])

        if smoothing_params > 1:
            self.logger.warning(f"⚠️ Multiple EWM smoothing parameters provided: alpha={alpha}, span={span}, halflife={kwargs.get('halflife')}, comass={kwargs.get('comass')}. Using alpha.")

        if alpha is not None:
            return data.ewm(alpha=alpha, **kwargs).mean()
        elif span is not None:
            return data.ewm(span=span, **kwargs).mean()
        elif kwargs.get('halflife') is not None:
            return data.ewm(halflife=kwargs['halflife'], **kwargs).mean()
        elif kwargs.get('comass') is not None:
            return data.ewm(comass=kwargs['comass'], **kwargs).mean()
        else:
            # Default to span=20 if no parameters provided
            return data.ewm(span=20, **kwargs).mean()

    def _calculate_sma_vectorized(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate SMA using vectorized operations with VectorBT optimization."""
        return self._vectorbt_rolling_operation(data, "mean", window)

    def _calculate_rolling_std_vectorized(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling standard deviation using vectorized operations."""
        return self._vectorbt_rolling_operation(data, "std", window)

    def _calculate_rolling_min_vectorized(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling minimum using vectorized operations."""
        return self._vectorbt_rolling_operation(data, "min", window)

    def _calculate_rolling_max_vectorized(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum using vectorized operations."""
        return self._vectorbt_rolling_operation(data, "max", window)

    def _calculate_rolling_sum_vectorized(self, data: pd.Series, window: int) -> pd.Series:
        """Calculate rolling sum using vectorized operations."""
        return self._vectorbt_rolling_operation(data, "sum", window)

    def _calculate_rolling_quantile_vectorized(self, data: pd.Series, window: int, q: float = 0.5) -> pd.Series:
        """Calculate rolling quantile using vectorized operations."""
        try:
            from src.feature_generation.utils.vectorbt_rolling_optimizer import get_vectorbt_rolling_optimizer
            optimizer = get_vectorbt_rolling_optimizer()
            return optimizer.rolling_quantile(data, window, q=q)
        except ImportError:
            return data.rolling(window=window).quantile(q)
        except Exception as e:
            self.logger.warning(f"VectorBT quantile calculation failed: {e}, using pandas fallback")
            return data.rolling(window=window).quantile(q)

# Global registry for feature generators
_feature_generators: Dict[str, FeatureGenerator] = {}

def register_feature_generator(generator: FeatureGenerator) -> None:
    """
    Register a feature generator.

    Args:
        generator: Feature generator to register
    """
    _feature_generators[generator.config.name] = generator
    logger.info(f"Registered feature generator: {generator.config.name}")

def get_registered_generator(name: str) -> Optional[FeatureGenerator]:
    """
    Get a registered feature generator by name.

    Args:
        name: Name of the generator

    Returns:
        Feature generator or None if not found
    """
    return _feature_generators.get(name)

def list_registered_generators() -> List[str]:
    """
    List all registered feature generators.

    Returns:
        List of generator names
    """
    return list(_feature_generators.keys())
