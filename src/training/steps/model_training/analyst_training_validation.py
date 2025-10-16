"""
Validation Manager for Analyst Models Training

Handles all validation logic with comprehensive error reporting.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.tprint import tprint_info, tprint_debug, tprint_success, tprint_error, tprint_warning
from src.utils.math_validation import validate_positive, validate_finite, MathValidationError
from src.utils.common_operations import (
    safe_divide, validate_dataframe, validate_dataframe_columns,
    check_disk_space, get_memory_usage, sanitize_string, ensure_directory
)
from src.utils.ml_common.config import PerRegimeTrainingConfig

from .analyst_training_constants import (
    NAN_CRITICAL_PERCENT, NAN_WARNING_PERCENT, INF_CRITICAL_PERCENT, INF_WARNING_PERCENT,
    TARGET_NAN_CRITICAL_PERCENT, TARGET_INF_CRITICAL_PERCENT,
    REGIME_IMBALANCE_RATIO_THRESHOLD, REGIME_ENTROPY_LOW_THRESHOLD,
    DATA_SIZE_MEMORY_RATIO, VALID_TIMEFRAMES, VALID_MODEL_TYPES, VALID_METRICS,
    HPO_TRIALS_MIN, HPO_TRIALS_WARNING_MAX, HPO_TIMEOUT_MIN_SECONDS, HPO_TIMEOUT_WARNING_MAX_SECONDS,
    MIN_SAMPLES_LOW_THRESHOLD, MIN_SAMPLES_HIGH_THRESHOLD, REQUIRED_DISK_SPACE_GB
)

class ValidationManager:
    """Manages all validation operations for analyst training."""

    def __init__(self, config: PerRegimeTrainingConfig):
        """
        Initialize validation manager.

        Args:
            config: Training configuration to validate against
        """
        self.config = config
        tprint_info("🔧 ValidationManager initialized")

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate training configuration with comprehensive checks.

        Returns:
            Dictionary with validation results including errors and warnings
        """
        tprint_info("🔍 Starting configuration validation")

        errors = []
        warnings = []

        try:
            # Model name validation
            if not self.config.model_name or not isinstance(self.config.model_name, str):
                errors.append("model_name must be a non-empty string")
            else:
                self.config.model_name = sanitize_string(self.config.model_name, 50)
                tprint_debug(f"✅ Model name validated: {self.config.model_name}")

            # Timeframe validation
            if not self.config.timeframe or not isinstance(self.config.timeframe, str):
                errors.append("timeframe must be a non-empty string")
            elif self.config.timeframe not in VALID_TIMEFRAMES:
                warnings.append(f"Unusual timeframe: {self.config.timeframe}. Valid: {VALID_TIMEFRAMES}")

            # Model types validation
            if not self.config.model_types or not isinstance(self.config.model_types, list):
                errors.append("model_types must be a non-empty list")
            elif len(self.config.model_types) == 0:
                errors.append("model_types list cannot be empty")
            else:
                invalid_types = [mt for mt in self.config.model_types if mt not in VALID_MODEL_TYPES]
                if invalid_types:
                    warnings.append(f"Unknown model types: {invalid_types}")

                if len(self.config.model_types) < 2:
                    warnings.append("Consider using multiple model types for better ensemble performance")

            # HPO validation
            try:
                hpo_trials = validate_positive(self.config.hpo_n_trials, "hpo_n_trials")
                if hpo_trials > HPO_TRIALS_WARNING_MAX:
                    warnings.append(f"hpo_n_trials > {HPO_TRIALS_WARNING_MAX} may cause long training times")
                elif hpo_trials < HPO_TRIALS_MIN:
                    warnings.append(f"hpo_n_trials < {HPO_TRIALS_MIN} may not provide sufficient optimization")
            except (ValueError, MathValidationError) as e:
                errors.append(f"Invalid hpo_n_trials: {e}")

            try:
                hpo_timeout = validate_positive(self.config.hpo_timeout_seconds, "hpo_timeout_seconds")
                if hpo_timeout < HPO_TIMEOUT_MIN_SECONDS:
                    warnings.append(f"hpo_timeout_seconds < {HPO_TIMEOUT_MIN_SECONDS} may be too short")
                elif hpo_timeout > HPO_TIMEOUT_WARNING_MAX_SECONDS:
                    warnings.append(f"hpo_timeout_seconds > {HPO_TIMEOUT_WARNING_MAX_SECONDS} may cause very long training times")
            except (ValueError, MathValidationError) as e:
                errors.append(f"Invalid hpo_timeout_seconds: {e}")

            # Min samples validation
            try:
                min_samples = validate_positive(self.config.min_samples_per_regime, "min_samples_per_regime")
                if min_samples < MIN_SAMPLES_LOW_THRESHOLD:
                    warnings.append(f"min_samples_per_regime < {MIN_SAMPLES_LOW_THRESHOLD} may cause poor model performance")
                elif min_samples > MIN_SAMPLES_HIGH_THRESHOLD:
                    warnings.append(f"min_samples_per_regime > {MIN_SAMPLES_HIGH_THRESHOLD} may cause memory issues")
            except (ValueError, MathValidationError) as e:
                errors.append(f"Invalid min_samples_per_regime: {e}")

            # Path validation
            if not self.config.model_save_path:
                errors.append("model_save_path cannot be empty")
            else:
                try:
                    ensure_directory(self.config.model_save_path)
                    disk_check = check_disk_space(self.config.model_save_path, REQUIRED_DISK_SPACE_GB)
                    if not disk_check['sufficient']:
                        warnings.append(f"Insufficient disk space: {disk_check['free_gb']:.1f}GB available, {REQUIRED_DISK_SPACE_GB}GB required")
                except Exception as e:
                    errors.append(f"Invalid model_save_path: {e}")

            # Metrics validation
            if not self.config.evaluation_metrics or not isinstance(self.config.evaluation_metrics, list):
                errors.append("evaluation_metrics must be a non-empty list")
            else:
                invalid_metrics = [m for m in self.config.evaluation_metrics if m not in VALID_METRICS]
                if invalid_metrics:
                    warnings.append(f"Unknown evaluation metrics: {invalid_metrics}")

                if len(self.config.evaluation_metrics) < 2:
                    warnings.append("Consider using multiple evaluation metrics for comprehensive assessment")

            # Memory check
            available_memory = get_memory_usage() / 1024 / 1024  # MB
            estimated_memory_need = len(self.config.model_types) * 1000
            if available_memory < estimated_memory_need:
                warnings.append(f"Low available memory: {available_memory:.1f}MB, estimated need: {estimated_memory_need}MB")

            validation_result = {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'available_memory_mb': available_memory
            }

            if validation_result['valid']:
                tprint_success("✅ Configuration validation passed")
            else:
                tprint_error(f"❌ Configuration validation failed: {len(errors)} errors")

            if warnings:
                tprint_warning(f"⚠️ Configuration warnings: {len(warnings)}")
                for warning in warnings:
                    tprint_warning(f"  - {warning}")

            return validation_result

        except Exception as e:
            tprint_error(f"❌ Configuration validation exception: {e}")
            return {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'exception': str(e)
            }

    def validate_input_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Validate input data with comprehensive checks.

        Args:
            X: Input features
            y: Target values
            regime_labels: Regime labels

        Returns:
            Dictionary with validation results
        """
        tprint_info("🔍 Starting input data validation")

        errors = []
        warnings = []

        try:
            # Null checks
            if X is None or y is None or regime_labels is None:
                errors.append("Input data cannot be None")
                return {'valid': False, 'errors': errors, 'warnings': warnings}

            # Shape validation
            try:
                x_len = validate_positive(len(X), "X length")
                y_len = validate_positive(len(y), "y length")
                regime_len = validate_positive(len(regime_labels), "regime_labels length")

                if x_len != y_len or x_len != regime_len:
                    errors.append(f"Length mismatch: X={x_len}, y={y_len}, regime_labels={regime_len}")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}

                if x_len == 0:
                    errors.append("Input data is empty")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}

                tprint_debug(f"✅ Data length validation passed: {x_len} samples")

            except (ValueError, MathValidationError) as e:
                errors.append(f"Data length validation failed: {e}")
                return {'valid': False, 'errors': errors, 'warnings': warnings}

            # Data quality checks
            nan_count_X = np.isnan(X).sum()
            inf_count_X = np.isinf(X).sum()
            nan_count_y = np.isnan(y).sum()
            inf_count_y = np.isinf(y).sum()

            # Validate counts are finite
            try:
                validate_finite(nan_count_X, "NaN count in X")
                validate_finite(inf_count_X, "Inf count in X")
                validate_finite(nan_count_y, "NaN count in y")
                validate_finite(inf_count_y, "Inf count in y")
            except (ValueError, MathValidationError) as e:
                errors.append(f"Data quality validation failed: {e}")

            # Check NaN/Inf percentages
            if nan_count_X > 0:
                nan_percentage = safe_divide(nan_count_X, X.size, 0) * 100
                if nan_percentage > NAN_CRITICAL_PERCENT:
                    errors.append(f"X contains {nan_count_X} NaN values ({nan_percentage:.1f}%)")
                elif nan_percentage > NAN_WARNING_PERCENT:
                    warnings.append(f"X contains {nan_count_X} NaN values ({nan_percentage:.1f}%)")

            if inf_count_X > 0:
                inf_percentage = safe_divide(inf_count_X, X.size, 0) * 100
                if inf_percentage > INF_CRITICAL_PERCENT:
                    errors.append(f"X contains {inf_count_X} infinite values ({inf_percentage:.1f}%)")
                elif inf_percentage > INF_WARNING_PERCENT:
                    warnings.append(f"X contains {inf_count_X} infinite values ({inf_percentage:.1f}%)")

            if nan_count_y > 0:
                nan_percentage = safe_divide(nan_count_y, len(y), 0) * 100
                if nan_percentage > TARGET_NAN_CRITICAL_PERCENT:
                    errors.append(f"y contains {nan_count_y} NaN values ({nan_percentage:.1f}%)")
                else:
                    warnings.append(f"y contains {nan_count_y} NaN values ({nan_percentage:.1f}%)")

            if inf_count_y > 0:
                inf_percentage = safe_divide(inf_count_y, len(y), 0) * 100
                if inf_percentage > TARGET_INF_CRITICAL_PERCENT:
                    errors.append(f"y contains {inf_count_y} infinite values ({inf_percentage:.1f}%)")
                else:
                    warnings.append(f"y contains {inf_count_y} infinite values ({inf_percentage:.1f}%)")

            # Regime distribution checks
            unique_regimes = np.unique(regime_labels)
            regime_counts = np.bincount(regime_labels.astype(int))

            if len(regime_counts) > 0:
                min_regime_size = validate_positive(regime_counts.min(), "min regime size")
                max_regime_size = validate_positive(regime_counts.max(), "max regime size")

                if min_regime_size < self.config.min_samples_per_regime:
                    warnings.append(f"Some regimes have < {self.config.min_samples_per_regime} samples (min: {min_regime_size})")

                regime_ratio = safe_divide(max_regime_size, min_regime_size, 1)
                if regime_ratio > REGIME_IMBALANCE_RATIO_THRESHOLD:
                    warnings.append(f"High regime imbalance: ratio {regime_ratio:.1f}x")

                regime_entropy = self._calculate_entropy(regime_counts)
                if regime_entropy < REGIME_ENTROPY_LOW_THRESHOLD:
                    warnings.append(f"Low regime diversity (entropy: {regime_entropy:.3f})")

            # Feature statistics
            feature_stats = {
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 1,
                'n_regimes': len(unique_regimes),
                'regime_distribution': dict(zip(unique_regimes, regime_counts)),
            }

            # Memory check
            data_size_mb = (X.nbytes + y.nbytes + regime_labels.nbytes) / (1024 * 1024)
            available_memory_mb = get_memory_usage() / 1024 / 1024

            if data_size_mb > available_memory_mb * DATA_SIZE_MEMORY_RATIO:
                warnings.append(f"Large dataset: {data_size_mb:.1f}MB ({DATA_SIZE_MEMORY_RATIO*100:.0f}% of available memory)")

            validation_result = {
                'valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings,
                'statistics': feature_stats
            }

            if validation_result['valid']:
                tprint_success("✅ Input data validation passed")
            else:
                tprint_error(f"❌ Input data validation failed: {len(errors)} errors")

            if warnings:
                tprint_warning(f"⚠️ Input data warnings: {len(warnings)}")

            return validation_result

        except Exception as e:
            tprint_error(f"❌ Input data validation exception: {e}")
            return {
                'valid': False,
                'errors': [f"Validation exception: {str(e)}"],
                'warnings': [],
                'exception': str(e)
            }

    def _calculate_entropy(self, counts: np.ndarray) -> float:
        """Calculate Shannon entropy of distribution."""
        try:
            total = np.sum(counts)
            if total == 0:
                return 0.0

            probabilities = counts / total
            probabilities = probabilities[probabilities > 0]

            if len(probabilities) == 0:
                return 0.0

            entropy = -np.sum(probabilities * np.log2(probabilities))
            return validate_finite(entropy, "entropy")

        except Exception as e:
            tprint_warning(f"⚠️ Failed to calculate entropy: {e}")
            return 0.0
