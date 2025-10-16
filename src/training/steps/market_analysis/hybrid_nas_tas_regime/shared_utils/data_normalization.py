"""
Data Normalization Utilities for Hybrid NAS-TAS Regime Detection.

Provides comprehensive data preprocessing and normalization utilities
using existing utils for robust data handling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
import time
from datetime import datetime
from enum import Enum
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

# Import existing utilities
try:
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power,
        validate_finite, validate_positive, validate_range
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False

try:
    from src.utils.common_operations import (
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer
    )
    HARDWARE_UTILS_AVAILABLE = True
except ImportError:
    HARDWARE_UTILS_AVAILABLE = False

try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations,
        get_vectorized_processing_core,
        get_enhanced_matrix_operations,
        get_batch_matrix_processor
    )
    MATRIX_OPERATIONS_AVAILABLE = True
except ImportError:
    MATRIX_OPERATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

class NormalizationMethod(Enum):
    """Normalization methods available."""
    Z_SCORE = "z_score"
    MIN_MAX = "min_max"
    ROBUST = "robust"
    QUANTILE = "quantile"
    POWER = "power"
    LOG = "log"
    SQRT = "sqrt"

@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    method: NormalizationMethod = NormalizationMethod.Z_SCORE
    feature_range: Tuple[float, float] = (0.0, 1.0)
    quantile_range: Tuple[float, float] = (0.25, 0.75)
    power_exponent: float = 0.5
    handle_outliers: bool = True
    outlier_threshold: float = 3.0
    handle_missing: bool = True
    missing_strategy: str = "median"  # "mean", "median", "mode", "drop", "forward_fill"
    use_hardware_acceleration: bool = True
    use_matrix_operations: bool = True
    batch_size: int = 1000
    memory_limit_gb: float = 8.0

@dataclass
class NormalizationResult:
    """Result from data normalization."""
    normalized_data: pd.DataFrame
    normalization_params: Dict[str, Any]
    outlier_info: Dict[str, Any]
    missing_info: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    hardware_optimization_applied: bool = False
    matrix_operations_used: bool = False

class DataNormalizer:
    """Advanced data normalizer with hardware acceleration and robust preprocessing."""

    def __init__(self, config: NormalizationConfig):
        """Initialize the data normalizer.

        Args:
            config: Normalization configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize hardware acceleration if available
        self.hardware_accelerator = None
        self.memory_optimizer = None
        self.cpu_optimizer = None

        if HARDWARE_UTILS_AVAILABLE and config.use_hardware_acceleration:
            try:
                self.hardware_accelerator = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()
                self.logger.info("✅ Hardware acceleration initialized for data normalization")
            except Exception as e:
                self.logger.warning(f"⚠️ Hardware acceleration not available: {e}")

        # Initialize matrix operations if available
        self.matrix_ops = None
        self.vectorized_core = None
        self.enhanced_ops = None
        self.batch_processor = None

        if MATRIX_OPERATIONS_AVAILABLE and config.use_matrix_operations:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.enhanced_ops = get_enhanced_matrix_operations()
                self.batch_processor = get_batch_matrix_processor()
                self.logger.info("✅ Matrix operations initialized for data normalization")
            except Exception as e:
                self.logger.warning(f"⚠️ Matrix operations not available: {e}")

        self.logger.info("✅ Data Normalizer initialized")
        self.logger.info(f"   Method: {config.method.value}")
        self.logger.info(f"   Handle outliers: {config.handle_outliers}")
        self.logger.info(f"   Handle missing: {config.handle_missing}")

    def normalize_data(self, data: pd.DataFrame,
                      target_columns: Optional[List[str]] = None) -> NormalizationResult:
        """Normalize data using the specified method.

        Args:
            data: Input DataFrame
            target_columns: Optional list of columns to normalize (None = all numeric columns)

        Returns:
            NormalizationResult with normalized data and metadata
        """
        start_time = time.time()

        try:
            self.logger.info("🔄 Starting data normalization")
            self.logger.info(f"   Input shape: {data.shape}")
            self.logger.info(f"   Method: {self.config.method.value}")

            # Validate input data
            if data.empty:
                raise ValueError("Input data is empty")

            # Select target columns
            if target_columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            if not target_columns:
                raise ValueError("No numeric columns found for normalization")

            self.logger.info(f"   Target columns: {len(target_columns)}")

            # Create working copy
            normalized_data = data.copy()

            # Handle missing values first
            missing_info = self._handle_missing_values(normalized_data, target_columns)

            # Handle outliers if configured
            outlier_info = {}
            if self.config.handle_outliers:
                outlier_info = self._handle_outliers(normalized_data, target_columns)

            # Apply normalization
            normalization_params = self._apply_normalization(
                normalized_data, target_columns
            )

            processing_time = time.time() - start_time

            self.logger.info(f"✅ Data normalization completed in {processing_time:.2f}s")
            self.logger.info(f"   Normalized columns: {len(target_columns)}")

            return NormalizationResult(
                normalized_data=normalized_data,
                normalization_params=normalization_params,
                outlier_info=outlier_info,
                missing_info=missing_info,
                processing_time=processing_time,
                success=True,
                hardware_optimization_applied=self.hardware_accelerator is not None,
                matrix_operations_used=self.matrix_ops is not None
            )

        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Data normalization failed: {e}")

            return NormalizationResult(
                normalized_data=pd.DataFrame(),
                normalization_params={},
                outlier_info={},
                missing_info={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )

    def _handle_missing_values(self, data: pd.DataFrame,
                             target_columns: List[str]) -> Dict[str, Any]:
        """Handle missing values in the data.

        Args:
            data: DataFrame to process
            target_columns: Columns to process

        Returns:
            Missing value information
        """
        try:
            missing_info = {
                'strategy': self.config.missing_strategy,
                'missing_counts': {},
                'missing_percentages': {},
                'handled': True
            }

            if not self.config.handle_missing:
                missing_info['handled'] = False
                return missing_info

            for col in target_columns:
                if col in data.columns:
                    missing_count = data[col].isnull().sum()
                    missing_info['missing_counts'][col] = int(missing_count)
                    missing_info['missing_percentages'][col] = float(missing_count / len(data))

                    if missing_count > 0:
                        if self.config.missing_strategy == "mean":
                            data[col].fillna(data[col].mean(), inplace=True)
                        elif self.config.missing_strategy == "median":
                            data[col].fillna(data[col].median(), inplace=True)
                        elif self.config.missing_strategy == "mode":
                            mode_value = data[col].mode()
                            if not mode_value.empty:
                                data[col].fillna(mode_value[0], inplace=True)
                        elif self.config.missing_strategy == "forward_fill":
                            data[col].fillna(method='ffill', inplace=True)
                        elif self.config.missing_strategy == "drop":
                            data.dropna(subset=[col], inplace=True)

            return missing_info

        except Exception as e:
            self.logger.warning(f"⚠️ Missing value handling failed: {e}")
            return {'handled': False, 'error': str(e)}

    def _handle_outliers(self, data: pd.DataFrame,
                        target_columns: List[str]) -> Dict[str, Any]:
        """Handle outliers in the data.

        Args:
            data: DataFrame to process
            target_columns: Columns to process

        Returns:
            Outlier information
        """
        try:
            outlier_info = {
                'threshold': self.config.outlier_threshold,
                'outlier_counts': {},
                'outlier_percentages': {},
                'handled': True
            }

            for col in target_columns:
                if col in data.columns:
                    # Calculate outlier bounds using IQR method
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - self.config.outlier_threshold * IQR
                    upper_bound = Q3 + self.config.outlier_threshold * IQR

                    # Identify outliers
                    outlier_mask = (data[col] < lower_bound) | (data[col] > upper_bound)
                    outlier_count = outlier_mask.sum()

                    outlier_info['outlier_counts'][col] = int(outlier_count)
                    outlier_info['outlier_percentages'][col] = float(outlier_count / len(data))

                    # Cap outliers to bounds
                    if outlier_count > 0:
                        data[col] = np.clip(data[col], lower_bound, upper_bound)

            return outlier_info

        except Exception as e:
            self.logger.warning(f"⚠️ Outlier handling failed: {e}")
            return {'handled': False, 'error': str(e)}

    def _apply_normalization(self, data: pd.DataFrame,
                           target_columns: List[str]) -> Dict[str, Any]:
        """Apply normalization to the specified columns.

        Args:
            data: DataFrame to normalize
            target_columns: Columns to normalize

        Returns:
            Normalization parameters used
        """
        try:
            normalization_params = {
                'method': self.config.method.value,
                'columns': target_columns,
                'parameters': {}
            }

            for col in target_columns:
                if col in data.columns:
                    col_params = self._normalize_column(data, col)
                    normalization_params['parameters'][col] = col_params

            return normalization_params

        except Exception as e:
            self.logger.error(f"❌ Normalization application failed: {e}")
            return {}

    def _normalize_column(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Normalize a single column.

        Args:
            data: DataFrame containing the column
            column: Column name to normalize

        Returns:
            Normalization parameters for the column
        """
        try:
            col_data = data[column].copy()
            params = {'column': column, 'method': self.config.method.value}

            if self.config.method == NormalizationMethod.Z_SCORE:
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    data[column] = (col_data - mean_val) / std_val
                    params.update({'mean': float(mean_val), 'std': float(std_val)})
                else:
                    data[column] = 0
                    params.update({'mean': float(mean_val), 'std': 0.0})

            elif self.config.method == NormalizationMethod.MIN_MAX:
                min_val = col_data.min()
                max_val = col_data.max()
                if max_val > min_val:
                    data[column] = (col_data - min_val) / (max_val - min_val)
                    # Scale to feature range
                    data[column] = data[column] * (self.config.feature_range[1] - self.config.feature_range[0]) + self.config.feature_range[0]
                    params.update({
                        'min': float(min_val),
                        'max': float(max_val),
                        'feature_range': self.config.feature_range
                    })
                else:
                    data[column] = self.config.feature_range[0]
                    params.update({'min': float(min_val), 'max': float(max_val)})

            elif self.config.method == NormalizationMethod.ROBUST:
                median_val = col_data.median()
                q75 = col_data.quantile(0.75)
                q25 = col_data.quantile(0.25)
                iqr = q75 - q25
                if iqr > 0:
                    data[column] = (col_data - median_val) / iqr
                    params.update({
                        'median': float(median_val),
                        'q25': float(q25),
                        'q75': float(q75),
                        'iqr': float(iqr)
                    })
                else:
                    data[column] = 0
                    params.update({'median': float(median_val), 'iqr': 0.0})

            elif self.config.method == NormalizationMethod.QUANTILE:
                q_low = col_data.quantile(self.config.quantile_range[0])
                q_high = col_data.quantile(self.config.quantile_range[1])
                if q_high > q_low:
                    data[column] = (col_data - q_low) / (q_high - q_low)
                    params.update({
                        'q_low': float(q_low),
                        'q_high': float(q_high),
                        'quantile_range': self.config.quantile_range
                    })
                else:
                    data[column] = 0.5
                    params.update({'q_low': float(q_low), 'q_high': float(q_high)})

            elif self.config.method == NormalizationMethod.POWER:
                if MATH_VALIDATION_AVAILABLE:
                    # Apply power transformation with safe operations
                    data[column] = col_data.apply(
                        lambda x: safe_power(x, self.config.power_exponent, 0.0)
                    )
                else:
                    data[column] = np.power(np.abs(col_data), self.config.power_exponent) * np.sign(col_data)
                params.update({'exponent': self.config.power_exponent})

            elif self.config.method == NormalizationMethod.LOG:
                if MATH_VALIDATION_AVAILABLE:
                    # Apply log transformation with safe operations
                    data[column] = col_data.apply(
                        lambda x: safe_log(x, 0.0)
                    )
                else:
                    data[column] = np.log(np.maximum(col_data, 1e-10))
                params.update({'transformation': 'log'})

            elif self.config.method == NormalizationMethod.SQRT:
                if MATH_VALIDATION_AVAILABLE:
                    # Apply sqrt transformation with safe operations
                    data[column] = col_data.apply(
                        lambda x: safe_sqrt(x, 0.0)
                    )
                else:
                    data[column] = np.sqrt(np.maximum(col_data, 0))
                params.update({'transformation': 'sqrt'})

            return params

        except Exception as e:
            self.logger.warning(f"⚠️ Column normalization failed for {column}: {e}")
            return {'column': column, 'error': str(e)}

    def inverse_normalize(self, normalized_data: pd.DataFrame,
                        normalization_params: Dict[str, Any]) -> pd.DataFrame:
        """Inverse normalize data using stored parameters.

        Args:
            normalized_data: Normalized data
            normalization_params: Parameters used for normalization

        Returns:
            Inverse normalized data
        """
        try:
            inverse_data = normalized_data.copy()

            for col, params in normalization_params.get('parameters', {}).items():
                if col in inverse_data.columns:
                    self._inverse_normalize_column(inverse_data, col, params)

            return inverse_data

        except Exception as e:
            self.logger.error(f"❌ Inverse normalization failed: {e}")
            return normalized_data

    def _inverse_normalize_column(self, data: pd.DataFrame, column: str, params: Dict[str, Any]):
        """Inverse normalize a single column.

        Args:
            data: DataFrame containing the column
            column: Column name to inverse normalize
            params: Normalization parameters
        """
        try:
            method = params.get('method', self.config.method.value)

            if method == NormalizationMethod.Z_SCORE.value:
                mean_val = params.get('mean', 0.0)
                std_val = params.get('std', 1.0)
                data[column] = data[column] * std_val + mean_val

            elif method == NormalizationMethod.MIN_MAX.value:
                min_val = params.get('min', 0.0)
                max_val = params.get('max', 1.0)
                feature_range = params.get('feature_range', (0.0, 1.0))
                # First scale back from feature range
                data[column] = (data[column] - feature_range[0]) / (feature_range[1] - feature_range[0])
                # Then scale to original range
                data[column] = data[column] * (max_val - min_val) + min_val

            elif method == NormalizationMethod.ROBUST.value:
                median_val = params.get('median', 0.0)
                iqr = params.get('iqr', 1.0)
                data[column] = data[column] * iqr + median_val

            elif method == NormalizationMethod.QUANTILE.value:
                q_low = params.get('q_low', 0.0)
                q_high = params.get('q_high', 1.0)
                data[column] = data[column] * (q_high - q_low) + q_low

            # For power, log, and sqrt transformations, inverse is more complex
            # and may not be exact due to information loss

        except Exception as e:
            self.logger.warning(f"⚠️ Inverse normalization failed for {column}: {e}")

    def get_normalization_statistics(self, data: pd.DataFrame,
                                   target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get statistics about the data before normalization.

        Args:
            data: Input DataFrame
            target_columns: Columns to analyze

        Returns:
            Data statistics
        """
        try:
            if target_columns is None:
                target_columns = data.select_dtypes(include=[np.number]).columns.tolist()

            stats = {
                'total_columns': len(target_columns),
                'total_samples': len(data),
                'column_statistics': {},
                'overall_statistics': {}
            }

            for col in target_columns:
                if col in data.columns:
                    col_data = data[col]
                    stats['column_statistics'][col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q25': float(col_data.quantile(0.25)),
                        'q75': float(col_data.quantile(0.75)),
                        'missing_count': int(col_data.isnull().sum()),
                        'missing_percentage': float(col_data.isnull().sum() / len(data))
                    }

            # Overall statistics
            all_numeric = data[target_columns].select_dtypes(include=[np.number])
            if not all_numeric.empty:
                stats['overall_statistics'] = {
                    'mean_of_means': float(all_numeric.mean().mean()),
                    'std_of_stds': float(all_numeric.std().std()),
                    'total_missing': int(all_numeric.isnull().sum().sum()),
                    'missing_percentage': float(all_numeric.isnull().sum().sum() / all_numeric.size)
                }

            return stats

        except Exception as e:
            self.logger.warning(f"⚠️ Statistics calculation failed: {e}")
            return {'error': str(e)}

def create_data_normalizer(config: Optional[NormalizationConfig] = None) -> DataNormalizer:
    """Create a data normalizer instance.

    Args:
        config: Optional normalization configuration

    Returns:
        DataNormalizer instance
    """
    if config is None:
        config = NormalizationConfig()
    return DataNormalizer(config)

def quick_normalize(data: pd.DataFrame,
                   method: NormalizationMethod = NormalizationMethod.Z_SCORE,
                   target_columns: Optional[List[str]] = None) -> NormalizationResult:
    """Quick data normalization with default settings.

    Args:
        data: Input DataFrame
        method: Normalization method
        target_columns: Optional target columns

    Returns:
        Normalization result
    """
    config = NormalizationConfig(method=method)
    normalizer = DataNormalizer(config)
    return normalizer.normalize_data(data, target_columns)
