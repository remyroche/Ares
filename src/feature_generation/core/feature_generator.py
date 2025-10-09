"""
Base Feature Generator Classes

This module defines the base classes and interfaces for feature generation,
providing a standardized way to create and manage feature generators.
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
    """Configuration for feature generation."""
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
    
    def __post_init__(self):
        if self.optional_columns is None:
            self.optional_columns = []
        if self.parameters is None:
            self.parameters = {}
        if self.dependencies is None:
            self.dependencies = []

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
    Abstract base class for feature generators.
    
    This class defines the interface that all feature generators must implement,
    providing a standardized way to generate features with consistent error handling,
    logging, and performance tracking.
    """
    
    def __init__(self, config: FeatureConfig):
        """
        Initialize the feature generator.
        
        Args:
            config: Feature configuration
        """
        self.config = config
        self.logger = logger.getChild(f'{self.__class__.__name__}')
        self.performance_stats = {
            'total_generations': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_computation_time': 0.0,
            'total_computation_time': 0.0
        }

        # Optional state storage for incremental generation support
        self._state: Dict[str, Any] = {}
        self._state_loaded: bool = False

        # Reduced logging - only log at category level, not individual features
        # self.logger.info(f"Initialized {self.__class__.__name__} for {config.name}")
    
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
    
    def generate(self, data: pd.DataFrame, **kwargs) -> FeatureResult:
        """
        Generate the feature with error handling and performance tracking.
        
        Args:
            data: Input data DataFrame
            **kwargs: Additional parameters
            
        Returns:
            FeatureResult with the generated feature and metadata
        """
        start_time = time.time()

        # Allow state injection through kwargs for compatibility
        external_state = kwargs.pop('state', None)
        if external_state is not None:
            self.load_state(external_state)

        state_loaded_flag = self._state_loaded

        try:
            # Validate input data
            self._validate_data(data)

            # Allow subclasses to adjust internal buffers before generation
            self._prepare_state(data)

            # Generate the feature
            feature_data = self._generate_feature(data, **kwargs)

            # Validate output
            self._validate_output(feature_data)

            # Update generator state with latest observations
            try:
                self._finalize_state(data, feature_data)
            except Exception as state_error:
                self.logger.debug(f"State finalization failed: {state_error}")

            # Update performance stats
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=True)

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

        except Exception as e:
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=False)

            error_msg = f"Failed to generate {self.config.name}: {str(e)}"
            self.logger.error(error_msg)

            failure_metadata = {
                'generator_class': self.__class__.__name__,
                'input_shape': data.shape,
                'state_loaded': state_loaded_flag,
                'state': self._serialize_state()
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
        if data.empty:
            return

        last_row = data.iloc[-1]
        state_update: Dict[str, Any] = {
            'last_index': data.index[-1],
            'last_row': last_row.to_dict()
        }

        if not feature_data.empty:
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
        Validate input data.
        
        Args:
            data: Input data DataFrame
            
        Raises:
            ValueError: If data validation fails
        """
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check required columns
        missing_columns = set(self.config.required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for sufficient data
        if len(data) < self.config.min_lookback:
            raise ValueError(f"Insufficient data: need at least {self.config.min_lookback} rows, got {len(data)}")
    
    def _validate_output(self, feature_data: pd.Series) -> None:
        """
        Validate output feature data.
        
        Args:
            feature_data: Generated feature data
            
        Raises:
            ValueError: If output validation fails
        """
        if feature_data.empty:
            raise ValueError("Generated feature is empty")
        
        # Check for all NaN values
        if feature_data.isna().all():
            raise ValueError("Generated feature contains only NaN values")
        
        # Check for infinite values
        if np.isinf(feature_data).any():
            self.logger.warning("Generated feature contains infinite values")
    
    def _update_performance_stats(self, computation_time: float, success: bool) -> None:
        """
        Update performance statistics.
        
        Args:
            computation_time: Time taken for computation
            success: Whether the generation was successful
        """
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
            Combined features as pandas Series (this is a placeholder - 
            composite generators typically return multiple features)
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
        
        # For now, return the first successful result
        # In practice, composite generators should handle multiple outputs differently
        return results[0]
    
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
    Base class for vectorized feature generators that leverage matrix operations.
    
    This class provides optimized vectorized computation capabilities
    using the matrix operations framework and new optimization utilities.
    """
    
    def __init__(self, config: FeatureConfig, enable_matrix_ops: bool = True, enable_vectorization_optimization: bool = True):
        """
        Initialize vectorized feature generator.
        
        Args:
            config: Feature configuration
            enable_matrix_ops: Whether to enable matrix operations
            enable_vectorization_optimization: Whether to enable vectorization optimization
        """
        super().__init__(config)
        self.enable_matrix_ops = enable_matrix_ops
        self.enable_vectorization_optimization = enable_vectorization_optimization
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from ...utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.debug("Matrix operations enabled")
            except ImportError:
                self.matrix_ops = None
                self.enable_matrix_ops = False
                self.logger.warning("Matrix operations not available")
        else:
            self.matrix_ops = None
        
        # Initialize vectorization optimizer if available
        if enable_vectorization_optimization:
            try:
                from ..utils.vectorization_optimizer import get_vectorization_optimizer
                self.vectorization_optimizer = get_vectorization_optimizer()
                self.logger.debug("Vectorization optimizer enabled")
            except ImportError:
                self.vectorization_optimizer = None
                self.enable_vectorization_optimization = False
                self.logger.warning("Vectorization optimizer not available")
        else:
            self.vectorization_optimizer = None
    
    def _vectorized_operation(self, operation: str, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform vectorized operation using matrix operations framework.
        
        Args:
            operation: Operation to perform
            data: Input data array
            **kwargs: Additional parameters
            
        Returns:
            Result of the operation
        """
        if not self.enable_matrix_ops or self.matrix_ops is None:
            # Fallback to numpy operations
            return self._numpy_fallback(operation, data, **kwargs)
        
        try:
            return self.matrix_ops.batch_process(data, operation, **kwargs)
        except Exception as e:
            self.logger.warning(f"Matrix operation failed, using numpy fallback: {e}")
            return self._numpy_fallback(operation, data, **kwargs)
    
    def optimize_dataframe_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame for vectorized processing using the vectorization optimizer.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Optimized DataFrame
        """
        if self.enable_vectorization_optimization and self.vectorization_optimizer:
            return self.vectorization_optimizer.optimize_dataframe_processing(data)
        else:
            return data
    
    def vectorized_rolling_operations(self, 
                                    data: pd.DataFrame,
                                    operations: List[str],
                                    windows: List[int],
                                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Perform vectorized rolling operations with hardware optimization.
        
        Args:
            data: Input DataFrame
            operations: List of operations ('mean', 'std', 'var', 'min', 'max', 'sum')
            windows: List of window sizes
            columns: Columns to process (None = all numeric columns)
            
        Returns:
            DataFrame with rolling features
        """
        if self.enable_vectorization_optimization and self.vectorization_optimizer:
            return self.vectorization_optimizer.vectorized_rolling_operations(
                data, operations, windows, columns
            )
        else:
            return self._fallback_rolling_operations(data, operations, windows, columns)
    
    def _fallback_rolling_operations(self, 
                                   data: pd.DataFrame,
                                   operations: List[str],
                                   windows: List[int],
                                   columns: List[str]) -> pd.DataFrame:
        """Fallback rolling operations without vectorization optimizer."""
        result = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        for window in windows:
            for col in columns:
                if col in data.columns:
                    series = data[col]
                    
                    for operation in operations:
                        if operation == 'mean':
                            result[f'{col}_rolling_mean_{window}'] = series.rolling(window=window).mean()
                        elif operation == 'std':
                            result[f'{col}_rolling_std_{window}'] = series.rolling(window=window).std()
                        elif operation == 'var':
                            result[f'{col}_rolling_var_{window}'] = series.rolling(window=window).var()
                        elif operation == 'min':
                            result[f'{col}_rolling_min_{window}'] = series.rolling(window=window).min()
                        elif operation == 'max':
                            result[f'{col}_rolling_max_{window}'] = series.rolling(window=window).max()
                        elif operation == 'sum':
                            result[f'{col}_rolling_sum_{window}'] = series.rolling(window=window).sum()
        
        return result
    
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
            return pd.Series(data).rolling(window=window).mean().values
        elif operation == 'rolling_std':
            window = kwargs.get('window', 20)
            return pd.Series(data).rolling(window=window).std().values
        elif operation == 'ewm_mean':
            span = kwargs.get('span', 20)
            return pd.Series(data).ewm(span=span).mean().values
        else:
            raise ValueError(f"Unsupported operation: {operation}")

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