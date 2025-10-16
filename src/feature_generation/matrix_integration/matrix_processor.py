"""
Matrix Feature Processor

This module provides integration with the matrix operations framework
for optimized feature computation.
"""

import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
import warnings

from ..core.feature_generator import FeatureGenerator, FeatureResult

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

except ImportError:

    cp = None

logger = logging.getLogger(__name__)

class MatrixFeatureProcessor:
    """
    Processor for matrix-optimized feature generation.

    This class integrates with the matrix operations framework to provide
    optimized feature computation using vectorized operations and
    """

    def __init__(self, enable_gpu: bool = True, enable_parallel: bool = True):
        """
        Initialize the matrix feature processor.

        Args:
            enable_gpu: Whether to enable
            enable_parallel: Whether to enable parallel processing
        """
        self.logger = logger.getChild('MatrixFeatureProcessor')
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel

        # Initialize VectorBTRollingOptimizer and UnifiedVectorizationManager
        try:
            from ...utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer, get_vectorbt_rolling_optimizer
            self.vectorbt_rolling_optimizer = get_vectorbt_rolling_optimizer()
            self.logger.info("✅ VectorBTRollingOptimizer initialized")
        except ImportError:
            self.vectorbt_rolling_optimizer = None
            self.logger.warning("⚠️ VectorBTRollingOptimizer not available")

        try:
            from ...utils.ml_common.unified_vectorization_manager import UnifiedVectorizationManager, get_unified_vectorization_manager
            self.unified_vectorization_manager = get_unified_vectorization_manager()
            self.logger.info("✅ UnifiedVectorizationManager initialized")
        except ImportError:
            self.unified_vectorization_manager = None
            self.logger.warning("⚠️ UnifiedVectorizationManager not available")

        self.matrix_available = self.vectorbt_rolling_optimizer is not None or self.unified_vectorization_manager is not None

    def process_features(self,
                        generators: List[FeatureGenerator],
                        data: pd.DataFrame,
                        **kwargs) -> List[FeatureResult]:
        """
        Process features using matrix operations optimization.

        Args:
            generators: List of feature generators
            data: Input data
            **kwargs: Additional parameters

        Returns:
            List of feature results
        """
        if not self.matrix_available:
            self.logger.warning("VectorBTRollingOptimizer not available, using fallback")
            return self._fallback_processing(generators, data, **kwargs)

        self.logger.info(f"Processing {len(generators)} features with matrix optimization")

        results = []

        # Group generators by type for batch processing
        grouped_generators = self._group_generators_by_type(generators)

        for group_type, group_generators in grouped_generators.items():
            try:
                group_results = self._process_generator_group(
                    group_type, group_generators, data, **kwargs
                )
                results.extend(group_results)
            except Exception as e:
                self.logger.error(f"Error processing {group_type} group: {e}")
                # Fallback to individual processing
                for generator in group_generators:
                    try:
                        result = generator.generate(data, **kwargs)
                        results.append(result)
                    except Exception as e2:
                        self.logger.error(f"Error processing {generator.config.name}: {e2}")

        return results

    def _group_generators_by_type(self, generators: List[FeatureGenerator]) -> Dict[str, List[FeatureGenerator]]:
        """Group generators by type for batch processing."""
        groups = {
            'vectorized': [],
            'rolling': [],
            'other': []
        }

        for generator in generators:
            if hasattr(generator, 'enable_matrix_ops') and generator.enable_matrix_ops:
                groups['vectorized'].append(generator)
            elif 'rolling' in generator.config.name.lower() or 'ma' in generator.config.name.lower():
                groups['rolling'].append(generator)
            else:
                groups['other'].append(generator)

        return {k: v for k, v in groups.items() if v}

    def _process_generator_group(self,
                               group_type: str,
                               generators: List[FeatureGenerator],
                               data: pd.DataFrame,
                               **kwargs) -> List[FeatureResult]:
        """Process a group of generators with matrix optimization."""
        if group_type == 'vectorized':
            return self._process_vectorized_generators(generators, data, **kwargs)
        elif group_type == 'rolling':
            return self._process_rolling_generators(generators, data, **kwargs)
        else:
            return self._process_other_generators(generators, data, **kwargs)

    def _process_vectorized_generators(self,
                                     generators: List[FeatureGenerator],
                                     data: pd.DataFrame,
                                     **kwargs) -> List[FeatureResult]:
        """Process vectorized generators with matrix operations."""
        results = []

        # Extract common data arrays
        data_arrays = self._extract_data_arrays(data, generators)

        for generator in generators:
            try:
                # Use matrix operations for computation
                if hasattr(generator, '_vectorized_operation'):
                    feature_data = generator._vectorized_operation(
                        'rolling_mean', data_arrays['close'], window=generator.config.default_lookback
                    )
                else:
                    # Fallback to regular generation
                    result = generator.generate(data, **kwargs)
                    results.append(result)
                    continue

                # Create result
                result = FeatureResult(
                    name=generator.config.name,
                    data=pd.Series(feature_data, index=data.index),
                    config=generator.config,
                    computation_time=0.0,  # Would be measured in practice
                    success=True
                )
                results.append(result)

            except Exception as e:
                self.logger.error(f"Error in vectorized processing for {generator.config.name}: {e}")
                # Fallback to regular generation
                result = generator.generate(data, **kwargs)
                results.append(result)

        return results

    def _process_rolling_generators(self,
                                  generators: List[FeatureGenerator],
                                  data: pd.DataFrame,
                                  **kwargs) -> List[FeatureResult]:
        """Process rolling generators with batch operations."""
        results = []

        # Group by window size for batch processing
        window_groups = {}
        for generator in generators:
            window = generator.config.default_lookback
            if window not in window_groups:
                window_groups[window] = []
            window_groups[window].append(generator)

        for window, window_generators in window_groups.items():
            try:
                # Batch process rolling operations
                window_results = self._batch_rolling_operations(
                    window_generators, data, window, **kwargs
                )
                results.extend(window_results)
            except Exception as e:
                self.logger.error(f"Error in batch rolling processing for window {window}: {e}")
                # Fallback to individual processing
                for generator in window_generators:
                    result = generator.generate(data, **kwargs)
                    results.append(result)

        return results

    def _process_other_generators(self,
                                generators: List[FeatureGenerator],
                                data: pd.DataFrame,
                                **kwargs) -> List[FeatureResult]:
        """Process other generators individually."""
        results = []

        for generator in generators:
            try:
                result = generator.generate(data, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing {generator.config.name}: {e}")

        return results

    def _extract_data_arrays(self, data: pd.DataFrame, generators: List[FeatureGenerator]) -> Dict[str, np.ndarray]:
        """Extract data arrays needed by generators."""
        arrays = {}

        # Determine required columns
        required_columns = set()
        for generator in generators:
            required_columns.update(generator.config.required_columns)

        # Extract arrays
        for column in required_columns:
            if column in data.columns:
                arrays[column] = data[column].values

        return arrays

    def _batch_rolling_operations(self,
                                generators: List[FeatureGenerator],
                                data: pd.DataFrame,
                                window: int,
                                **kwargs) -> List[FeatureResult]:
        """Batch process rolling operations."""
        results = []

        # Extract common data
        close_prices = data['close'].values

        # Use VectorBTRollingOptimizer for batch rolling
        if self.vectorbt_rolling_optimizer:
            try:
                # Batch rolling mean using VectorBTRollingOptimizer
                rolling_mean = self.vectorbt_rolling_optimizer.rolling_mean(
                    close_prices, window=window
                )

                # Create results for each generator
                for generator in generators:
                    if 'ma' in generator.config.name.lower() or 'mean' in generator.config.name.lower():
                        feature_data = rolling_mean.flatten()
                    else:
                        # Fallback to individual generation
                        result = generator.generate(data, **kwargs)
                        results.append(result)
                        continue

                    result = FeatureResult(
                        name=generator.config.name,
                        data=pd.Series(feature_data, index=data.index),
                        config=generator.config,
                        computation_time=0.0,
                        success=True
                    )
                    results.append(result)

            except Exception as e:
                self.logger.error(f"Error in batch rolling operations: {e}")
                # Fallback to individual processing
                for generator in generators:
                    result = generator.generate(data, **kwargs)
                    results.append(result)
        else:
            # Fallback to individual processing
            for generator in generators:
                result = generator.generate(data, **kwargs)
                results.append(result)

        return results

    def _fallback_processing(self,
                           generators: List[FeatureGenerator],
                           data: pd.DataFrame,
                           **kwargs) -> List[FeatureResult]:
        """Fallback processing without matrix operations."""
        results = []

        for generator in generators:
            try:
                result = generator.generate(data, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in fallback processing for {generator.config.name}: {e}")

        return results

class VectorizedFeatureGenerator:
    """
    Base class for vectorized feature generators.

    This class provides optimized vectorized computation capabilities
    using the matrix operations framework.
    """

    def __init__(self, enable_matrix_ops: bool = True):
        """
        Initialize vectorized feature generator.

        Args:
            enable_matrix_ops: Whether to enable matrix operations
        """
        self.enable_matrix_ops = enable_matrix_ops
        self.logger = logger.getChild('VectorizedFeatureGenerator')

        # Initialize matrix operations if enabled
        if enable_matrix_ops:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.debug("Matrix operations enabled")
            except ImportError:
                self.matrix_ops = None
                self.enable_matrix_ops = False
                self.logger.warning("Matrix operations not available")
        else:
            self.matrix_ops = None

    def vectorized_operation(self, operation: str, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform vectorized operation using matrix operations framework.

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
            # Use VectorBTRollingOptimizer for batch operations
            if hasattr(self.vectorbt_rolling_optimizer, operation):
                return getattr(self.vectorbt_rolling_optimizer, operation)(data, **kwargs)
            else:
                # Use UnifiedVectorizationManager for other operations
                if self.unified_vectorization_manager:
                    return self.unified_vectorization_manager.optimize_operation(operation, data, **kwargs)
                else:
                    return self._numpy_fallback(operation, data, **kwargs)
        except Exception as e:
            self.logger.warning(f"VectorBTRollingOptimizer operation failed, using numpy fallback: {e}")
            return self._numpy_fallback(operation, data, **kwargs)

    def _numpy_fallback(self, operation: str, data: np.ndarray, **kwargs) -> np.ndarray:
        """Fallback to numpy operations when matrix operations are not available."""
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

# Global matrix processor instance
_global_matrix_processor: Optional[MatrixFeatureProcessor] = None

def get_matrix_processor(enable_gpu: bool = True, enable_parallel: bool = True) -> MatrixFeatureProcessor:
    """
    Get the global matrix processor instance.

    Args:
        enable_gpu: Whether to enable
        enable_parallel: Whether to enable parallel processing

    Returns:
        Matrix processor instance
    """
    global _global_matrix_processor
    if _global_matrix_processor is None:
        _global_matrix_processor = MatrixFeatureProcessor(enable_gpu, enable_parallel)
    return _global_matrix_processor

def enable_matrix_acceleration(enable: bool = True) -> None:
    """
    Enable or disable matrix acceleration globally.

    Args:
        enable: Whether to enable matrix acceleration
    """
    global _global_matrix_processor
    if enable:
        _global_matrix_processor = MatrixFeatureProcessor(enable_gpu=True, enable_parallel=True)
    else:
        _global_matrix_processor = MatrixFeatureProcessor(enable_gpu=False, enable_parallel=False)
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
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
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
        else:
            raise ValueError(f"Unsupported operation: {operation}")
