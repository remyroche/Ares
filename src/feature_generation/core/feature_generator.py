"""
Base Feature Generator Classes

This module defines the base classes and interfaces for feature generation,
providing a standardized way to create and manage feature generators.
"""

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
    TREND = "trend"
    OSCILLATOR = "oscillator"
    SUPPORT_RESISTANCE = "support_resistance"
    CANDLESTICK_PATTERN = "candlestick_pattern"
    # HMM_REGIME = "hmm_regime"  # DEPRECATED
    CROSS_TIMEFRAME = "cross_timeframe"
    MICROSTRUCTURE = "microstructure"
    ENTROPY = "entropy"
    AUTOENCODER = "autoencoder"
    ORDER_FLOW = "order_flow"
    REGIME = "regime"
    LEGACY = "legacy"
    TIME = "time"
    CUSTOM = "custom"
    ACCELERATION = "acceleration"
    INTERACTION = "interaction"

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
        
        try:
            # Validate input data
            self._validate_data(data)
            
            # Generate the feature
            feature_data = self._generate_feature(data, **kwargs)
            
            # Validate output
            self._validate_output(feature_data)
            
            # Update performance stats
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=True)
            
            self.logger.debug(f"Successfully generated {self.config.name} in {computation_time:.3f}s")
            
            return FeatureResult(
                name=self.config.name,
                data=feature_data,
                config=self.config,
                computation_time=computation_time,
                success=True,
                metadata={
                    'generator_class': self.__class__.__name__,
                    'input_shape': data.shape,
                    'output_length': len(feature_data)
                }
            )
            
        except Exception as e:
            computation_time = time.time() - start_time
            self._update_performance_stats(computation_time, success=False)
            
            error_msg = f"Failed to generate {self.config.name}: {str(e)}"
            self.logger.error(error_msg)
            
            return FeatureResult(
                name=self.config.name,
                data=pd.Series(dtype=float, index=data.index),
                config=self.config,
                computation_time=computation_time,
                success=False,
                error_message=error_msg
            )
    
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
    using the matrix operations framework.
    """
    
    def __init__(self, config: FeatureConfig, enable_matrix_ops: bool = True):
        """
        Initialize vectorized feature generator.
        
        Args:
            config: Feature configuration
            enable_matrix_ops: Whether to enable matrix operations
        """
        super().__init__(config)
        self.enable_matrix_ops = enable_matrix_ops
        
        # Initialize matrix operations if available
        if enable_matrix_ops:
            try:
                from ...utils.matrix_operations import get_unified_matrix_operations
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("Matrix operations enabled")
            except ImportError:
                self.matrix_ops = None
                self.enable_matrix_ops = False
                self.logger.warning("Matrix operations not available")
        else:
            self.matrix_ops = None
    
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