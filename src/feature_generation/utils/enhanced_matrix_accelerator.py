"""
Enhanced Matrix Accelerator for Feature Generation

This module provides hardware-accelerated matrix operations specifically
optimized for feature generation tasks, integrating with the existing
matrix operations framework.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedMatrixAccelerator:
    """Enhanced matrix accelerator with hardware optimization for feature generation."""

    def __init__(self, enable_gpu: bool = True, enable_parallel: bool = True, enable_simd: bool = True):
        self.enable_gpu = enable_gpu
        self.enable_parallel = enable_parallel
        self.enable_simd = enable_simd

        # Initialize matrix operations
        self.matrix_ops = None
        self._initialize_matrix_operations()

        # Performance tracking
        self.performance_stats = {
            'matrix_multiplications': 0,
            'batch_operations': 0,
            'gpu_operations': 0,
            'simd_operations': 0,
            'total_acceleration_time': 0.0
        }

        logger.info(f"🚀 Enhanced Matrix Accelerator initialized: GPU={enable_gpu}, Parallel={enable_parallel}, SIMD={enable_simd}")

    def _initialize_matrix_operations(self):
        """Initialize matrix operations with hardware acceleration."""
        try:
            from ...utils.matrix_operations import get_unified_matrix_operations
            self.matrix_ops = get_unified_matrix_operations(
                enable_gpu=self.enable_gpu,
                enable_parallel=self.enable_parallel
            )
            logger.info("✅ Matrix operations initialized successfully")
        except ImportError as e:
            logger.warning(f"⚠️ Matrix operations not available: {e}")
            self.matrix_ops = None

    def batch_rolling_calculations(self, data: pd.DataFrame, windows: List[int],
                                   operations: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Perform batch rolling calculations with hardware acceleration.

        Args:
            data: Input DataFrame with numeric columns
            windows: List of rolling window sizes
            operations: List of operations ('mean', 'std', 'var', 'min', 'max', 'sum')

        Returns:
            Dictionary mapping (column, window, operation) -> result DataFrame
        """
        if not self.matrix_ops:
            return self._fallback_batch_rolling(data, windows, operations)

        results = {}
        start_time = pd.Timestamp.now()

        try:
            numeric_data = data.select_dtypes(include=[np.number]).values

            for window in windows:
                for operation in operations:
                    for col_idx, col_name in enumerate(data.select_dtypes(include=[np.number]).columns):
                        try:
                            # Use matrix operations for rolling calculations
                            if operation == 'mean':
                                result = self._accelerated_rolling_mean(numeric_data[:, col_idx], window)
                            elif operation == 'std':
                                result = self._accelerated_rolling_std(numeric_data[:, col_idx], window)
                            elif operation == 'var':
                                result = self._accelerated_rolling_var(numeric_data[:, col_idx], window)
                            elif operation == 'min':
                                result = self._accelerated_rolling_min(numeric_data[:, col_idx], window)
                            elif operation == 'max':
                                result = self._accelerated_rolling_max(numeric_data[:, col_idx], window)
                            elif operation == 'sum':
                                result = self._accelerated_rolling_sum(numeric_data[:, col_idx], window)
                            else:
                                continue

                            results[f"{col_name}_{operation}_{window}"] = pd.Series(result, index=data.index)

                        except Exception as e:
                            logger.warning(f"Error in {operation} for {col_name} window {window}: {e}")

            self.performance_stats['batch_operations'] += 1
            self.performance_stats['total_acceleration_time'] += (pd.Timestamp.now() - start_time).total_seconds()

            logger.info(f"✅ Batch rolling calculations completed in {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error in batch rolling calculations: {e}")
            return self._fallback_batch_rolling(data, windows, operations)

    def _accelerated_rolling_mean(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling mean calculation."""
        if self.matrix_ops:
            # Use matrix operations for optimized calculation
            return self.matrix_ops.rolling_mean(data, window)
        else:
            # Fallback to pandas rolling
            return pd.Series(data).rolling(window=window).mean().fillna(0).values

    def _accelerated_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling standard deviation calculation."""
        if self.matrix_ops:
            return self.matrix_ops.rolling_std(data, window)
        else:
            return pd.Series(data).rolling(window=window).std().fillna(0).values

    def _accelerated_rolling_var(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling variance calculation."""
        if self.matrix_ops:
            return self.matrix_ops.rolling_var(data, window)
        else:
            return pd.Series(data).rolling(window=window).var().fillna(0).values

    def _accelerated_rolling_min(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling minimum calculation."""
        if self.matrix_ops:
            return self.matrix_ops.rolling_min(data, window)
        else:
            return pd.Series(data).rolling(window=window).min().fillna(0).values

    def _accelerated_rolling_max(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling maximum calculation."""
        if self.matrix_ops:
            return self.matrix_ops.rolling_max(data, window)
        else:
            return pd.Series(data).rolling(window=window).max().fillna(0).values

    def _accelerated_rolling_sum(self, data: np.ndarray, window: int) -> np.ndarray:
        """Accelerated rolling sum calculation."""
        if self.matrix_ops:
            return self.matrix_ops.rolling_sum(data, window)
        else:
            return pd.Series(data).rolling(window=window).sum().fillna(0).values

    def _fallback_batch_rolling(self, data: pd.DataFrame, windows: List[int],
                               operations: List[str]) -> Dict[str, pd.DataFrame]:
        """Fallback batch rolling calculations without acceleration."""
        results = {}

        for window in windows:
            for operation in operations:
                for col_name in data.select_dtypes(include=[np.number]).columns:
                    try:
                        series = data[col_name]
                        if operation == 'mean':
                            result = series.rolling(window=window).mean()
                        elif operation == 'std':
                            result = series.rolling(window=window).std()
                        elif operation == 'var':
                            result = series.rolling(window=window).var()
                        elif operation == 'min':
                            result = series.rolling(window=window).min()
                        elif operation == 'max':
                            result = series.rolling(window=window).max()
                        elif operation == 'sum':
                            result = series.rolling(window=window).sum()
                        else:
                            continue

                        results[f"{col_name}_{operation}_{window}"] = result.fillna(0)

                    except Exception as e:
                        logger.warning(f"Fallback error in {operation} for {col_name}: {e}")

        return results

    def vectorized_feature_transformations(self, data: pd.DataFrame,
                                          transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply vectorized feature transformations with hardware acceleration.

        Args:
            data: Input DataFrame
            transformations: List of transformation configs

        Returns:
            DataFrame with transformed features
        """
        if not self.matrix_ops:
            return self._fallback_vectorized_transformations(data, transformations)

        results = data.copy()
        start_time = pd.Timestamp.now()

        try:
            for transform in transformations:
                transform_type = transform.get('type', '')
                params = transform.get('params', {})

                if transform_type == 'zscore':
                    results = self._accelerated_zscore_transform(results, **params)
                elif transform_type == 'minmax':
                    results = self._accelerated_minmax_transform(results, **params)
                elif transform_type == 'robust':
                    results = self._accelerated_robust_transform(results, **params)
                elif transform_type == 'log':
                    results = self._accelerated_log_transform(results, **params)
                elif transform_type == 'sqrt':
                    results = self._accelerated_sqrt_transform(results, **params)

            self.performance_stats['total_acceleration_time'] += (pd.Timestamp.now() - start_time).total_seconds()

            logger.info(f"✅ Vectorized transformations completed in {(pd.Timestamp.now() - start_time).total_seconds():.3f}s")
            return results

        except Exception as e:
            logger.error(f"Error in vectorized transformations: {e}")
            return self._fallback_vectorized_transformations(data, transformations)

    def _accelerated_zscore_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Accelerated z-score transformation."""
        if not self.matrix_ops:
            return self._pandas_zscore_transform(data, columns)

        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                values = data[col].values
                zscore_values = self.matrix_ops.zscore_normalize(values)
                result[col] = zscore_values

        return result

    def _accelerated_minmax_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Accelerated min-max transformation."""
        if not self.matrix_ops:
            return self._pandas_minmax_transform(data, columns)

        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                values = data[col].values
                minmax_values = self.matrix_ops.minmax_normalize(values)
                result[col] = minmax_values

        return result

    def _accelerated_robust_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Accelerated robust scaling transformation."""
        if not self.matrix_ops:
            return self._pandas_robust_transform(data, columns)

        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                values = data[col].values
                robust_values = self.matrix_ops.robust_scale(values)
                result[col] = robust_values

        return result

    def _accelerated_log_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Accelerated logarithmic transformation."""
        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                values = data[col].values
                # Handle zeros and negative values
                log_values = np.log(np.where(values > 0, values, 1e-8))
                result[col] = log_values

        return result

    def _accelerated_sqrt_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None, **kwargs) -> pd.DataFrame:
        """Accelerated square root transformation."""
        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                values = data[col].values
                # Handle negative values
                sqrt_values = np.sqrt(np.abs(values))
                result[col] = sqrt_values

        return result

    def _pandas_zscore_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Pandas-based z-score transformation (fallback)."""
        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                mean_val = data[col].mean()
                std_val = data[col].std()
                if std_val > 0:
                    result[col] = (data[col] - mean_val) / std_val
                else:
                    result[col] = 0.0

        return result

    def _pandas_minmax_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Pandas-based min-max transformation (fallback)."""
        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                if max_val > min_val:
                    result[col] = (data[col] - min_val) / (max_val - min_val)
                else:
                    result[col] = 0.0

        return result

    def _pandas_robust_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Pandas-based robust scaling transformation (fallback)."""
        result = data.copy()
        cols_to_transform = columns or data.select_dtypes(include=[np.number]).columns

        for col in cols_to_transform:
            if col in data.columns:
                median_val = data[col].median()
                mad_val = (data[col] - median_val).abs().median()
                if mad_val > 0:
                    result[col] = (data[col] - median_val) / mad_val
                else:
                    result[col] = 0.0

        return result

    def _fallback_vectorized_transformations(self, data: pd.DataFrame,
                                           transformations: List[Dict[str, Any]]) -> pd.DataFrame:
        """Fallback vectorized transformations without acceleration."""
        results = data.copy()

        for transform in transformations:
            transform_type = transform.get('type', '')
            params = transform.get('params', {})

            if transform_type == 'zscore':
                results = self._pandas_zscore_transform(results, **params)
            elif transform_type == 'minmax':
                results = self._pandas_minmax_transform(results, **params)
            elif transform_type == 'robust':
                results = self._pandas_robust_transform(results, **params)
            elif transform_type == 'log':
                results = self._accelerated_log_transform(results, **params)
            elif transform_type == 'sqrt':
                results = self._accelerated_sqrt_transform(results, **params)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.performance_stats,
            'matrix_ops_available': self.matrix_ops is not None,
            'avg_acceleration_time': self.performance_stats['total_acceleration_time'] / max(1, self.performance_stats['batch_operations'])
        }


# Global instance for easy access
_enhanced_matrix_accelerator = None

def get_enhanced_matrix_accelerator(enable_gpu: bool = True, enable_parallel: bool = True,
                                   enable_simd: bool = True) -> EnhancedMatrixAccelerator:
    """Get or create global enhanced matrix accelerator instance."""
    global _enhanced_matrix_accelerator

    if _enhanced_matrix_accelerator is None:
        _enhanced_matrix_accelerator = EnhancedMatrixAccelerator(
            enable_gpu=enable_gpu,
            enable_parallel=enable_parallel,
            enable_simd=enable_simd
        )

    return _enhanced_matrix_accelerator