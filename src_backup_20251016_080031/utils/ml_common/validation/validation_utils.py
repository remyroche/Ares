from src.utils.tprint import tprint

"""
ML Validation Utilities

This module provides comprehensive validation utilities with fast fail mechanisms
for configuration, data, resources, execution, and results. These utilities are
designed to be reusable across all ML steps and components.

Key Features:
- Configuration validation with fast fail
- Data integrity and quality validation
- Resource availability checking
- Execution timeout protection
- Result quality validation
- Comprehensive error handling and logging

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Leverages common_operations.py for robust error handling
- Integrates with logger for comprehensive logging
"""

import asyncio
import time
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from datetime import datetime
import logging
import psutil
import pandas as pd
import numpy as np

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ValidationUtils")
    tprint("✅ Custom logger available for MLCommon.ValidationUtils")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ValidationUtils")
    _LOGGER.setLevel(logging.INFO)

from ..math_validation import safe_divide, MathValidationError
from ...common_operations import create_fallback_logger

logger = logging.getLogger(__name__)

# Export list for the module
__all__ = [
    'ValidationError',
    'ConfigurationValidator',
    'DataValidator',
    'ResourceValidator',
    'ExecutionValidator',
    'ResultValidator',
    'MLValidationSuite',
    'ValidationUtils',
    'create_validation_suite',
    'validate_ml_step',
    'validate_data_quality',
    'validate_feature_matrix',
    'validate_input_data',
    'validate_model_config',
    'validate_training_data'
]


class ValidationError(Exception):
    """Custom exception for validation failures."""


## Removed legacy ValidationFramework in favor of unified framework


class ConfigurationValidator:
    """Configuration validation with fast fail mechanisms."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.ConfigurationValidator")
        _LOGGER.info("🚀 Initializing ConfigurationValidator...")
        _LOGGER.info("✅ ConfigurationValidator initialized successfully")

    def validate_ml_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ML configuration with fast fail.

        Args:
            config: Configuration dictionary

        Returns:
            Validation result with errors and warnings

        Raises:
            ValidationError: For critical configuration issues
        """
        result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'validated_config': {}
        }

        start_time = time.time()
        _LOGGER.info(f"🔍 Starting ML configuration validation...")
        _LOGGER.debug(f"📊 Configuration keys: {list(config.keys()) if config else 'None'}")
        
        try:
            # Required keys validation
            required_keys = ['symbol', 'exchange', 'timeframe']
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                error_msg = f"❌ FAST FAIL: Missing required configuration keys: {missing_keys}"
                _LOGGER.error(error_msg)
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "configuration", {"missing_keys": missing_keys})
            
            _LOGGER.info(f"✅ Required keys validation passed")

            # Symbol validation
            symbol = config.get('symbol', '')
            if not isinstance(symbol, str) or len(symbol) < 2:
                error_msg = f"❌ FAST FAIL: Invalid symbol format: {symbol}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "configuration", {"invalid_symbol": symbol})

            # Exchange validation
            exchange = config.get('exchange', '')
            if not isinstance(exchange, str) or len(exchange) < 2:
                error_msg = f"❌ FAST FAIL: Invalid exchange format: {exchange}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "configuration", {"invalid_exchange": exchange})

            # Timeframe validation
            timeframe = config.get('timeframe', '')
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            if timeframe not in valid_timeframes:
                error_msg = f"❌ FAST FAIL: Invalid timeframe '{timeframe}'. Must be one of: {valid_timeframes}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "configuration", {"invalid_timeframe": timeframe, "valid_options": valid_timeframes})

            # Data directory validation
            data_dir = config.get('data_dir', 'data')
            if not isinstance(data_dir, str):
                error_msg = f"❌ FAST FAIL: Invalid data directory: {data_dir}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "configuration", {"invalid_data_dir": data_dir})

            result['validated_config'] = config
            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Configuration validation passed in {execution_time:.3f}s")
            self.logger.info("✅ Configuration validation passed")

        except ValidationError:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Configuration validation failed after {execution_time:.3f}s")
            result['passed'] = False
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            result['passed'] = False
            error_msg = f"❌ FAST FAIL: Configuration validation failed: {e}"
            _LOGGER.error(f"❌ Configuration validation failed after {execution_time:.3f}s: {e}")
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "configuration", {"validation_error": str(e)}) from e

        return result


class DataValidator:
    """Comprehensive data validation with fast fail mechanisms."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.DataValidator")
        _LOGGER.info("🚀 Initializing DataValidator...")
        _LOGGER.info("✅ DataValidator initialized successfully")

    def validate_dataframe(self, data: pd.DataFrame, validation_level: str = "comprehensive") -> Dict[str, Any]:
        """
        Comprehensive data validation with fast fail.

        Args:
            data: DataFrame to validate
            validation_level: 'basic', 'standard', or 'comprehensive'

        Returns:
            Validation result dictionary

        Raises:
            ValidationError: For critical data issues
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting DataFrame validation...")
        _LOGGER.info(f"📊 Validation level: {validation_level}")
        
        if data is None:
            error_msg = "❌ FAST FAIL: Input data is None"
            _LOGGER.error(error_msg)
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "data_integrity", {"data_is_none": True})

        _LOGGER.info(f"📊 Data shape: {data.shape}")
        
        if len(data) == 0:
            error_msg = "❌ FAST FAIL: Input data is empty"
            _LOGGER.error(error_msg)
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "data_integrity", {"data_is_empty": True})

        if len(data) < 100:
            _LOGGER.warning("⚠️ Very small dataset detected - results may not be reliable")
            self.logger.warning("⚠️ Very small dataset detected - results may not be reliable")

        result = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'data_characteristics': {
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
        }

        try:
            # Schema validation
            if validation_level in ['standard', 'comprehensive']:
                schema_result = self._validate_schema(data)
                result['errors'].extend(schema_result['errors'])
                result['warnings'].extend(schema_result['warnings'])

            # Data type validation
            if validation_level == 'comprehensive':
                dtype_result = self._validate_data_types(data)
                result['errors'].extend(dtype_result['errors'])
                result['warnings'].extend(dtype_result['warnings'])

            # Statistical validation
            if validation_level == 'comprehensive':
                stat_result = self._validate_statistics(data)
                result['errors'].extend(stat_result['errors'])
                result['warnings'].extend(stat_result['warnings'])

            # Set final pass/fail
            if result['errors']:
                result['passed'] = False
                error_msg = f"❌ FAST FAIL: Data validation failed: {result['errors']}"
                _LOGGER.error(error_msg)
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "data_quality", {
                    "validation_errors": result['errors'],
                    "validation_warnings": result['warnings']
                })

            if result['warnings']:
                for warning in result['warnings']:
                    _LOGGER.warning(f"⚠️ Data validation warning: {warning}")
                    self.logger.warning(f"⚠️ Data validation warning: {warning}")

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ DataFrame validation passed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Errors: {len(result['errors'])}, Warnings: {len(result['warnings'])}")

        except ValidationError:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ DataFrame validation failed after {execution_time:.3f}s")
            raise
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"❌ FAST FAIL: Data validation process failed: {e}"
            _LOGGER.error(f"❌ DataFrame validation failed after {execution_time:.3f}s: {e}")
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "data_validation", {"process_error": str(e)}) from e

        return result

    def _validate_schema(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data schema and required columns."""
        errors = []
        warnings = []

        # Required columns
        required_columns = ['timestamp']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")

        # Recommended columns
        recommended_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_recommended = [col for col in recommended_columns if col not in data.columns]
        if missing_recommended:
            warnings.append(f"Missing recommended columns: {missing_recommended}")

        return {'errors': errors, 'warnings': warnings}

    def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data types for each column."""
        errors = []
        warnings = []

        expected_types = {
            'timestamp': ['datetime64[ns]', 'object'],
            'open': ['float64', 'float32'],
            'high': ['float64', 'float32'],
            'low': ['float64', 'float32'],
            'close': ['float64', 'float32'],
            'volume': ['float64', 'float32', 'int64', 'int32']
        }

        for col in data.columns:
            if col in expected_types:
                actual_type = str(data[col].dtype)
                if actual_type not in expected_types[col]:
                    warnings.append(f"Column '{col}' has unexpected type: {actual_type} (expected: {expected_types[col]})")

        return {'errors': errors, 'warnings': warnings}

    def _validate_statistics(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate statistical properties of the data."""
        errors = []
        warnings = []

        price_columns = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_columns if col in data.columns]

        if available_price_cols:
            for col in available_price_cols:
                try:
                    series = data[col]
                    if series.isnull().all():
                        errors.append(f"Column '{col}' contains only null values")
                        continue

                    if (series < 0).any():
                        errors.append(f"Column '{col}' contains negative values")

                    mean_val = series.mean()
                    std_val = series.std()
                    if std_val > 0:
                        z_scores = abs((series - mean_val) / std_val)
                        extreme_count = (z_scores > 10).sum()
                        if extreme_count > len(series) * 0.01:
                            warnings.append(f"Column '{col}' has {extreme_count} extreme values (>10 std)")

                except Exception as e:
                    warnings.append(f"Statistical validation failed for column '{col}': {e}")

        return {'errors': errors, 'warnings': warnings}


class ResourceValidator:
    """Resource availability validation with fast fail."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.ResourceValidator")

    def validate_system_resources(self, data: Optional[pd.DataFrame] = None,
                                memory_threshold: float = 0.8) -> Dict[str, Any]:
        """
        Validate system resource availability.

        Args:
            data: Optional DataFrame to estimate memory requirements for
            memory_threshold: Maximum memory usage threshold (0-1)

        Returns:
            Resource validation result

        Raises:
            ValidationError: For insufficient resources
        """
        result = {
            'available': True,
            'memory_mb': {'available': 0, 'estimated': 0},
            'cpu_percent': 0,
            'warnings': []
        }

        try:
            # Check memory availability
            memory_info = psutil.virtual_memory()
            available_memory_mb = memory_info.available / (1024 * 1024)

            if data is not None:
                # Estimate memory requirements for the data
                estimated_memory_mb = (data.memory_usage(deep=True).sum() / (1024 * 1024)) * 3  # 3x overhead
                result['memory_mb']['estimated'] = estimated_memory_mb

                if estimated_memory_mb > available_memory_mb * memory_threshold:
                    error_msg = f"❌ FAST FAIL: Insufficient memory for data processing. Estimated: {estimated_memory_mb:.1f}MB, Available: {available_memory_mb:.1f}MB"
                    self.logger.error(error_msg)
                    raise ValidationError(error_msg, "resource_exhaustion", {
                        "estimated_memory_mb": estimated_memory_mb,
                        "available_memory_mb": available_memory_mb,
                        "memory_threshold": memory_threshold
                    })

            result['memory_mb']['available'] = available_memory_mb

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            result['cpu_percent'] = cpu_percent

            if cpu_percent > 95:
                result['warnings'].append(f"High CPU usage detected: {cpu_percent}%")

            self.logger.info(f"✅ Resource validation passed - Memory: {available_memory_mb:.1f}MB available, CPU: {cpu_percent}%")

        except ValidationError:
            result['available'] = False
            raise
        except Exception as e:
            error_msg = f"❌ FAST FAIL: Resource validation failed: {e}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "resource_validation", {"validation_error": str(e)}) from e

        return result


class ExecutionValidator:
    """Execution validation with timeout and monitoring."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.ExecutionValidator")

    async def execute_with_timeout(self, coro: Callable, timeout_seconds: int,
                                 operation_name: str = "operation") -> Any:
        """
        Execute coroutine with timeout protection.

        Args:
            coro: Coroutine to execute
            timeout_seconds: Timeout in seconds
            operation_name: Name for logging

        Returns:
            Execution result

        Raises:
            ValidationError: For timeout or execution failures
        """
        try:
            self.logger.info(f"🚀 Starting {operation_name} with {timeout_seconds}s timeout")

            result = await asyncio.wait_for(coro, timeout=timeout_seconds)

            self.logger.info(f"✅ {operation_name} completed successfully")
            return result

        except asyncio.TimeoutError:
            error_msg = f"❌ FAST FAIL: {operation_name} timed out after {timeout_seconds} seconds"
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "timeout", {
                "operation": operation_name,
                "timeout_seconds": timeout_seconds
            })
        except Exception as e:
            error_msg = f"❌ FAST FAIL: {operation_name} failed: {e}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "execution_failure", {
                "operation": operation_name,
                "error": str(e)
            }) from e

    def validate_execution_result(self, result: Any, expected_type: Optional[type] = None,
                                success_criteria: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Validate execution result.

        Args:
            result: Execution result to validate
            expected_type: Expected type of result
            success_criteria: Custom success criteria function

        Returns:
            Validation result

        Raises:
            ValidationError: For invalid results
        """
        validation_result = {
            'passed': True,
            'errors': [],
            'warnings': []
        }

        try:
            # Type validation
            if expected_type and not isinstance(result, expected_type):
                error_msg = f"❌ FAST FAIL: Invalid result type. Expected: {expected_type}, Got: {type(result)}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "result_validation", {
                    "expected_type": str(expected_type),
                    "actual_type": str(type(result))
                })

            # Custom success criteria
            if success_criteria:
                try:
                    success_check = success_criteria(result)
                    if not success_check:
                        error_msg = "❌ FAST FAIL: Result failed custom success criteria"
                        self.logger.error(error_msg)
                        raise ValidationError(error_msg, "result_validation", {
                            "custom_criteria_failed": True
                        })
                except Exception as e:
                    error_msg = f"❌ FAST FAIL: Custom success criteria validation failed: {e}"
                    self.logger.error(error_msg)
                    raise ValidationError(error_msg, "result_validation", {
                        "custom_criteria_error": str(e)
                    })

            self.logger.info("✅ Result validation passed")

        except ValidationError:
            validation_result['passed'] = False
            raise

        return validation_result


class ResultValidator:
    """Result quality validation and statistical checks."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.ResultValidator")

    def validate_labeling_results(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate labeling results quality and consistency.

        Args:
            labeled_data: Labeled DataFrame to validate

        Returns:
            Validation result

        Raises:
            ValidationError: For critical result issues
        """
        result = {
            'passed': True,
            'warnings': [],
            'metrics': {},
            'quality_score': 0.0
        }

        try:
            # Check for required columns
            required_cols = ['meta_label', 'confidence']
            missing_cols = [col for col in required_cols if col not in labeled_data.columns]
            if missing_cols:
                error_msg = f"❌ FAST FAIL: Missing required result columns: {missing_cols}"
                self.logger.error(error_msg)
                raise ValidationError(error_msg, "result_integrity", {
                    "missing_columns": missing_cols
                })

            # Validate label distribution
            if 'meta_label' in labeled_data.columns:
                label_counts = labeled_data['meta_label'].value_counts()
                total_labels = len(labeled_data)

                # Check for single-class problem
                if len(label_counts) == 1:
                    result['warnings'].append("Single-class labeling detected - may indicate issues")

                # Check for class imbalance
                if len(label_counts) > 1:
                    max_class_ratio = label_counts.max() / total_labels
                    if max_class_ratio > 0.95:
                        result['warnings'].append(f"Severe class imbalance detected: {max_class_ratio:.1f}")

                # Store metrics
                result['metrics'] = {
                    'total_samples': total_labels,
                    'unique_labels': len(label_counts),
                    'label_distribution': label_counts.to_dict(),
                    'max_class_ratio': max_class_ratio if len(label_counts) > 1 else 1.0
                }

            # Validate confidence scores
            if 'confidence' in labeled_data.columns:
                confidence_stats = labeled_data['confidence'].describe()

                # Check for unrealistic confidence values
                if confidence_stats['min'] < 0 or confidence_stats['max'] > 1:
                    result['warnings'].append("Confidence scores outside [0,1] range detected")

                # Check for uniform confidence (might indicate poor calibration)
                if confidence_stats['std'] < 0.01:
                    result['warnings'].append("Very low confidence score variance detected")

                result['metrics']['confidence_stats'] = confidence_stats.to_dict()

            # Calculate quality score
            result['quality_score'] = self._calculate_quality_score(result)

            if result['warnings']:
                for warning in result['warnings']:
                    self.logger.warning(f"⚠️ Result validation warning: {warning}")

            self.logger.info(f"✅ Result validation completed - Quality score: {result['quality_score']:.2f}")

        except ValidationError:
            result['passed'] = False
            raise
        except Exception as e:
            error_msg = f"❌ FAST FAIL: Result validation failed: {e}"
            self.logger.error(error_msg)
            raise ValidationError(error_msg, "result_validation", {"validation_error": str(e)}) from e

        return result

    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate overall quality score for results."""
        score = 1.0  # Start with perfect score

        # Penalize warnings
        warning_penalty = len(validation_result['warnings']) * 0.1
        score -= min(warning_penalty, 0.5)  # Max penalty of 0.5 for warnings

        # Penalize class imbalance
        metrics = validation_result.get('metrics', {})
        max_class_ratio = metrics.get('max_class_ratio', 1.0)
        if max_class_ratio > 0.8:
            imbalance_penalty = (max_class_ratio - 0.8) * 2  # Scale penalty
            score -= min(imbalance_penalty, 0.3)

        # Penalize confidence issues
        confidence_stats = metrics.get('confidence_stats', {})
        if confidence_stats:
            std_conf = confidence_stats.get('std', 1.0)
            if std_conf < 0.05:  # Very low variance
                score -= 0.2

        return max(0.0, score)  # Ensure non-negative score


class MLValidationSuite:
    """Comprehensive ML validation suite combining all validators."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(f"{__name__}.ValidationSuite")

        # Initialize all validators
        self.config_validator = ConfigurationValidator(self.logger)
        self.data_validator = DataValidator(self.logger)
        self.resource_validator = ResourceValidator(self.logger)
        self.execution_validator = ExecutionValidator(self.logger)
        self.result_validator = ResultValidator(self.logger)

    async def validate_step_execution(self, config: Dict[str, Any],
                                    data: Optional[pd.DataFrame] = None,
                                    timeout_seconds: int = 1800) -> Dict[str, Any]:
        """
        Complete validation suite for ML step execution.

        Args:
            config: Step configuration
            data: Optional input data
            timeout_seconds: Execution timeout

        Returns:
            Comprehensive validation result

        Raises:
            ValidationError: For any validation failure
        """
        validation_result = {
            'step_name': config.get('step_name', 'unknown'),
            'passed': True,
            'validation_stages': {},
            'errors': [],
            'warnings': [],
            'start_time': datetime.now().isoformat()
        }

        try:
            # 1. Configuration validation
            self.logger.info("🔧 Stage 1: Configuration validation")
            config_result = self.config_validator.validate_ml_config(config)
            validation_result['validation_stages']['configuration'] = config_result

            # 2. Data validation (if provided)
            if data is not None:
                self.logger.info("📊 Stage 2: Data validation")
                data_result = self.data_validator.validate_dataframe(data, "comprehensive")
                validation_result['validation_stages']['data'] = data_result

            # 3. Resource validation
            self.logger.info("💾 Stage 3: Resource validation")
            resource_result = self.resource_validator.validate_system_resources(data)
            validation_result['validation_stages']['resources'] = resource_result

            # 4. Pre-execution validation complete
            validation_result['pre_execution_validated'] = True
            self.logger.info("✅ All pre-execution validations passed")

        except ValidationError as e:
            validation_result['passed'] = False
            validation_result['errors'].append({
                'stage': 'pre_execution',
                'error_type': e.error_type,
                'message': str(e),
                'details': e.details
            })
            raise
        except Exception as e:
            validation_result['passed'] = False
            validation_result['errors'].append({
                'stage': 'validation_suite',
                'error_type': 'unexpected_error',
                'message': f"Validation suite failed: {e}",
                'details': {'error': str(e)}
            })
            raise ValidationError(f"ML validation suite failed: {e}", "validation_suite", validation_result) from e

        return validation_result


# Convenience functions for easy access
def create_validation_suite(logger: Optional[logging.Logger] = None) -> MLValidationSuite:
    """Create a complete ML validation suite."""
    return MLValidationSuite(logger)


async def validate_ml_step(config: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Convenience function for complete ML step validation."""
    suite = create_validation_suite()
    return await suite.validate_step_execution(config, data)


class ValidationUtils:
    """Utility class for common validation operations."""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration parameters."""
        if not isinstance(config, dict):
            return False
        if not config:
            return False
        return True
    
    @staticmethod
    def validate_data_shapes(X, y, regime_labels) -> bool:
        """Validate data shapes for ML training."""
        if X is None or y is None or regime_labels is None:
            return False
        if len(X) != len(y) or len(X) != len(regime_labels):
            return False
        return True
    
    @staticmethod
    def validate_data_quality(X, y, regime_labels) -> bool:
        """Validate data quality for ML training."""
        if X is None or y is None or regime_labels is None:
            return False
        # Check for empty data
        if len(X) == 0 or len(y) == 0 or len(regime_labels) == 0:
            return False
        return True
    
    @staticmethod
    def validate_regime_distribution(regime_labels, min_samples_per_regime: int = 10) -> bool:
        """Validate regime distribution for ML training."""
        if regime_labels is None or len(regime_labels) == 0:
            return False
        
        # Count samples per regime
        unique_regimes, counts = np.unique(regime_labels, return_counts=True)
        
        # Check if all regimes have enough samples
        for count in counts:
            if count < min_samples_per_regime:
                return False
        
        return True
    
    @staticmethod
    def validate_feature_matrix(X, min_samples: int = 10, max_nan_ratio: float = 0.1) -> bool:
        """Validate feature matrix for ML training."""
        if X is None:
            return False
        
        # Check for empty data
        if len(X) == 0:
            return False
        
        # Check minimum samples
        if len(X) < min_samples:
            return False
        
        # Check for excessive NaN values
        if hasattr(X, 'isna'):  # pandas DataFrame
            nan_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
        else:  # numpy array
            nan_ratio = np.isnan(X).sum() / X.size
        
        if nan_ratio > max_nan_ratio:
            return False
        
        return True


def validate_input_data(data, required_columns=None, min_samples=10) -> bool:
    """
    Validate input data for ML processing.
    
    Args:
        data: Input data (pandas DataFrame or numpy array)
        required_columns: List of required column names (for DataFrames)
        min_samples: Minimum number of samples required
        
    Returns:
        bool: True if data is valid, False otherwise
    """
    if data is None:
        return False
    
    # Check for empty data
    if len(data) == 0:
        return False
    
    # Check minimum samples
    if len(data) < min_samples:
        return False
    
    # Check required columns for DataFrames
    if hasattr(data, 'columns') and required_columns:
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False
    
    # Check for excessive NaN values
    if hasattr(data, 'isna'):  # pandas DataFrame
        nan_ratio = data.isna().sum().sum() / (data.shape[0] * data.shape[1])
    elif hasattr(data, 'size'):  # numpy array
        nan_ratio = np.isnan(data).sum() / data.size
    else:
        nan_ratio = 0.0
    
    # Allow up to 10% NaN values
    if nan_ratio > 0.1:
        return False
    
    return True


# Standalone functions for direct import compatibility
def validate_data_quality(X, y, regime_labels) -> bool:
    """Validate data quality for ML training - standalone function."""
    return ValidationUtils.validate_data_quality(X, y, regime_labels)

def validate_feature_matrix(X, min_samples: int = 10, max_nan_ratio: float = 0.1) -> bool:
    """Validate feature matrix for ML training - standalone function."""
    return ValidationUtils.validate_feature_matrix(X, min_samples, max_nan_ratio)

def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration for ML training.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    if not isinstance(config, dict):
        return False
    
    # Check for required configuration keys
    required_keys = ['model_type', 'parameters']
    for key in required_keys:
        if key not in config:
            return False
    
    # Validate model type
    model_type = config.get('model_type')
    if not isinstance(model_type, str) or len(model_type.strip()) == 0:
        return False
    
    # Validate parameters
    parameters = config.get('parameters')
    if not isinstance(parameters, dict):
        return False
    
    return True


def validate_training_data(X, y=None, regime_labels=None, **kwargs) -> Dict[str, Any]:
    """
    Validate training data for ML training with comprehensive checks.
    
    Args:
        X: Input features (DataFrame or numpy array)
        y: Target values (optional, for supervised learning)
        regime_labels: Regime/cluster labels (optional)
        **kwargs: Additional validation parameters
        
    Returns:
        Dict containing validation result with 'valid' key and additional info
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'data_info': {}
    }
    
    try:
        # Check if X is provided
        if X is None:
            result['valid'] = False
            result['errors'].append("Input features X is None")
            return result
        
        # Check if X is empty
        if len(X) == 0:
            result['valid'] = False
            result['errors'].append("Input features X is empty")
            return result
        
        # Basic data info
        if hasattr(X, 'shape'):
            result['data_info']['shape'] = X.shape
        else:
            result['data_info']['length'] = len(X)
        
        # Check minimum samples
        min_samples = kwargs.get('min_samples', 10)
        if len(X) < min_samples:
            result['warnings'].append(f"Dataset has only {len(X)} samples (minimum recommended: {min_samples})")
        
        # Check for excessive NaN values
        nan_ratio = 0.0
        if hasattr(X, 'isna'):  # pandas DataFrame
            nan_ratio = X.isna().sum().sum() / (X.shape[0] * X.shape[1])
        elif hasattr(X, 'size'):  # numpy array
            nan_ratio = np.isnan(X).sum() / X.size
        
        max_nan_ratio = kwargs.get('max_nan_ratio', 0.1)
        if nan_ratio > max_nan_ratio:
            result['warnings'].append(f"High NaN ratio: {nan_ratio:.2%} (threshold: {max_nan_ratio:.2%})")
        
        # Validate target values if provided
        if y is not None:
            if len(y) != len(X):
                result['valid'] = False
                result['errors'].append(f"Length mismatch: X has {len(X)} samples, y has {len(y)} samples")
                return result
            
            # Check for constant target (classification issue)
            if hasattr(y, 'nunique'):
                unique_values = y.nunique()
                if unique_values == 1:
                    result['warnings'].append("Target variable has only one unique value")
                elif unique_values < 3 and hasattr(y, 'dtype') and y.dtype == 'object':
                    result['warnings'].append(f"Target variable has only {unique_values} classes")
        
        # Validate regime labels if provided
        if regime_labels is not None:
            if len(regime_labels) != len(X):
                result['valid'] = False
                result['errors'].append(f"Length mismatch: X has {len(X)} samples, regime_labels has {len(regime_labels)} samples")
                return result
            
            # Check regime distribution
            unique_regimes, counts = np.unique(regime_labels, return_counts=True)
            min_samples_per_regime = kwargs.get('min_samples_per_regime', 10)
            
            for regime, count in zip(unique_regimes, counts):
                if count < min_samples_per_regime:
                    result['warnings'].append(f"Regime {regime} has only {count} samples (minimum: {min_samples_per_regime})")
            
            result['data_info']['n_regimes'] = len(unique_regimes)
            result['data_info']['regime_distribution'] = dict(zip(unique_regimes, counts))
        
        # Add data characteristics
        result['data_info']['nan_ratio'] = nan_ratio
        result['data_info']['n_samples'] = len(X)
        
        # Check for near-constant features (warning only)
        if hasattr(X, 'var'):  # pandas DataFrame
            near_constant_features = X.var() < 1e-10
            if near_constant_features.any():
                constant_feature_names = X.columns[near_constant_features].tolist()
                result['warnings'].append(f"⚠️ Near-constant features detected: {constant_feature_names[:5]}")
        
        return result
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Validation error: {str(e)}")
        return result