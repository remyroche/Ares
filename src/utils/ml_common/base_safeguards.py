from src.utils.tprint import tprint

"""
ML Training Safeguards Utility

This module provides comprehensive safeguards and utilities to prevent common ML training issues
identified in step02_5_sr_optimization.py and ensure robust ML training across all steps.

Key Features:
- Parquet schema harmonization
- Class imbalance detection and handling
- Single-class chunk detection
- Proper error classification and fast-fail logic
- Cross-validation and evaluation metrics implementation
- Preflight validation for ML methods
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from collections import Counter
import traceback
import time

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.BaseSafeguards")
    tprint("✅ Custom logger available for MLCommon.BaseSafeguards")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.BaseSafeguards")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

class MLTrainingError(Exception):
    """Base exception for ML training errors."""
    pass

class ClassImbalanceError(MLTrainingError):
    """Raised when class imbalance is too extreme."""
    pass

class SingleClassError(MLTrainingError):
    """Raised when only one class is present."""
    pass

class DataQualityError(MLTrainingError):
    """Raised when data quality issues prevent training."""
    pass

class MLTrainingSafeguards:
    """Comprehensive safeguards for ML training."""

    def __init__(self):
        self.logger = logger.getChild('MLTrainingSafeguards')
        _LOGGER.info("🚀 Initializing MLTrainingSafeguards...")
        _LOGGER.info("✅ MLTrainingSafeguards initialized successfully")

    @staticmethod
    def harmonize_parquet_schema(df: pd.DataFrame, schema_reference: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Harmonize parquet schema by ensuring consistent dtypes across all columns.

        Args:
            df: DataFrame to harmonize
            schema_reference: Optional reference schema with column -> dtype mappings

        Returns:
            DataFrame with harmonized schema
        """
        start_time = time.time()
        _LOGGER.info(f"🔄 Starting parquet schema harmonization...")
        _LOGGER.debug(f"📊 Input DataFrame shape: {df.shape}, Columns: {len(df.columns)}")
        
        try:
            # Common dtype harmonization rules
            harmonized_df = df.copy()

            # Handle year column specifically (common issue)
            if 'year' in harmonized_df.columns:
                _LOGGER.debug("📅 Harmonizing year column to int32")
                harmonized_df['year'] = harmonized_df['year'].astype('int32')

            # Handle other categorical/dictionary columns
            categorical_cols = [col for col in harmonized_df.columns if col in ['symbol', 'ticker', 'month', 'exchange']]
            if len(categorical_cols):
                _LOGGER.debug(f"📝 Harmonizing categorical columns: {categorical_cols}")
                harmonized_df[categorical_cols] = harmonized_df[categorical_cols].astype('string')

            # Handle timestamp columns
            timestamp_cols = [col for col in harmonized_df.columns if 'timestamp' in col.lower() or 'time' in col.lower()]
            if timestamp_cols:
                _LOGGER.debug(f"⏰ Harmonizing timestamp columns: {timestamp_cols}")
                for col in timestamp_cols:
                    if col in harmonized_df.columns:
                        # Ensure consistent datetime format
                        if not pd.api.types.is_datetime64_any_dtype(harmonized_df[col]):
                            try:
                                harmonized_df[col] = pd.to_datetime(harmonized_df[col])
                            except Exception:
                                # If conversion fails, convert to string for consistency
                                harmonized_df[col] = harmonized_df[col].astype('string')

            # ------------------------------------------------------------
            # Vectorised numeric optimisation – much faster than per-col loops
            # ------------------------------------------------------------
            float_cols = harmonized_df.select_dtypes(include=["float64"]).columns
            if len(float_cols):
                _LOGGER.debug(f"🔢 Optimizing {len(float_cols)} float64 columns")
                harmonized_df[float_cols] = harmonized_df[float_cols].apply(pd.to_numeric, downcast="float")

            int_cols = harmonized_df.select_dtypes(include=["int64"]).columns
            if len(int_cols):
                _LOGGER.debug(f"🔢 Optimizing {len(int_cols)} int64 columns")
                harmonized_df[int_cols] = harmonized_df[int_cols].apply(pd.to_numeric, downcast="integer")

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Parquet schema harmonized for {len(harmonized_df.columns)} columns in {execution_time:.3f}s")
            return harmonized_df

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Schema harmonization failed after {execution_time:.3f}s: {e}")
            return df

    @staticmethod
    def check_class_distribution(y: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
        """
        Check class distribution for imbalance issues.

        Args:
            y: Target array
            threshold: Threshold for extreme imbalance detection

        Returns:
            Dictionary with distribution analysis
        """
        start_time = time.time()
        _LOGGER.info(f"🔍 Starting class distribution analysis...")
        _LOGGER.debug(f"📊 Input array length: {len(y)}, Threshold: {threshold}")
        
        try:
            unique_classes, counts = np.unique(y, return_counts=True)
            total_samples = len(y)
            class_ratios = counts / total_samples

            # Check for extreme imbalance
            max_ratio = np.max(class_ratios)
            min_ratio = np.min(class_ratios)

            analysis = {
                'n_classes': len(unique_classes),
                'class_counts': dict(zip(unique_classes, counts)),
                'class_ratios': dict(zip(unique_classes, class_ratios)),
                'max_class_ratio': max_ratio,
                'min_class_ratio': min_ratio,
                'is_extreme_imbalance': max_ratio >= threshold,
                'is_single_class': len(unique_classes) <= 1,
                'dominant_class': unique_classes[np.argmax(counts)] if len(unique_classes) > 0 else None,
                'dominant_ratio': max_ratio
            }

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Class distribution analysis completed in {execution_time:.3f}s")
            _LOGGER.info(f"📊 Results - Classes: {len(unique_classes)}, Total samples: {total_samples}")
            _LOGGER.info(f"📊 Class distribution: {dict(zip(unique_classes, counts))}")

            if analysis['is_extreme_imbalance']:
                _LOGGER.warning(f"⚠️ Extreme class imbalance detected!")
                _LOGGER.warning(f"⚠️ Dominant class {analysis['dominant_class']}: {analysis['dominant_ratio']:.2%}")
            else:
                _LOGGER.info(f"✅ Class distribution appears balanced (max ratio: {max_ratio:.2%})")

            if analysis['is_single_class']:
                _LOGGER.error(f"❌ Single class detected: {unique_classes[0] if len(unique_classes) > 0 else 'No classes'}")
            else:
                _LOGGER.info(f"✅ Multiple classes detected: {len(unique_classes)}")

            return analysis

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Class distribution check failed after {execution_time:.3f}s: {e}")
            return {
                'error': str(e),
                'n_classes': 0,
                'is_extreme_imbalance': False,
                'is_single_class': True
            }

    @staticmethod
    def validate_chunk_for_training(X: np.ndarray, y: np.ndarray,
                                  min_samples_per_class: int = 10) -> Dict[str, Any]:
        """
        Validate if a data chunk is suitable for training.

        Args:
            X: Feature matrix
            y: Target array
            min_samples_per_class: Minimum samples required per class

        Returns:
            Validation results
        """
        try:
            if len(X) == 0 or len(y) == 0:
                return {
                    'is_valid': False,
                    'reason': 'Empty data',
                    'n_samples': 0,
                    'n_features': 0
                }

            # Check class distribution
            class_analysis = MLTrainingSafeguards.check_class_distribution(y)

            if class_analysis['is_single_class']:
                # Raise explicit error for downstream fast-fail handlers
                raise SingleClassError("Single class detected in training chunk")

            # Check minimum samples per class
            min_class_samples = min(class_analysis['class_counts'].values())
            if min_class_samples < min_samples_per_class:
                # Too few samples in at least one class – classify as imbalance
                raise ClassImbalanceError(
                    f"Insufficient samples per class (min: {min_class_samples} < {min_samples_per_class})")

            # Check for extreme imbalance
            if class_analysis['is_extreme_imbalance']:
                raise ClassImbalanceError(
                    f"Extreme class imbalance (max ratio: {class_analysis['max_class_ratio']:.2%})")

            return {
                'is_valid': True,
                'reason': 'Valid for training',
                'n_samples': len(X),
                'n_features': X.shape[1] if len(X.shape) > 1 else 0,
                'class_analysis': class_analysis
            }

        except Exception as e:
            logger.error(f"❌ Chunk validation failed: {e}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {e}',
                'n_samples': len(X) if 'X' in locals() else 0,
                'n_features': X.shape[1] if 'X' in locals() and len(X.shape) > 1 else 0
            }

    @staticmethod
    def create_balanced_sample_weights(y: np.ndarray, strategy: str = 'balanced') -> np.ndarray:
        """
        Create sample weights for balanced training.

        Args:
            y: Target array
            strategy: Weighting strategy ('balanced', 'balanced_subsample')

        Returns:
            Array of sample weights
        """
        try:
            if len(y) == 0:
                return np.array([])

            from sklearn.utils.class_weight import compute_sample_weight

            if strategy == 'balanced':
                weights = compute_sample_weight('balanced', y)
            elif strategy == 'balanced_subsample':
                # For subsample balancing (useful for RandomForest)
                class_counts = Counter(y)
                total_samples = len(y)
                weights = np.array([total_samples / (len(class_counts) * class_counts[cls]) for cls in y])
            else:
                # Uniform weights
                weights = np.ones(len(y))

            return weights

        except Exception as e:
            logger.warning(f"⚠️ Sample weight calculation failed: {e}")
            return np.ones(len(y))

    @staticmethod
    def perform_robust_cross_validation(X: np.ndarray, y: np.ndarray,
                                      model_class: Any, model_params: Dict[str, Any],
                                      n_splits: int = 5, min_samples_per_fold: int = 50) -> Dict[str, Any]:
        """
        Perform robust cross-validation with temporal integrity.

        Args:
            X: Feature matrix
            y: Target array
            model_class: ML model class
            model_params: Model parameters
            n_splits: Number of CV splits
            min_samples_per_fold: Minimum samples per fold

        Returns:
            Cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

            # Validate data
            if len(X) < min_samples_per_fold * (n_splits + 1):
                return {
                    'error': f'Insufficient data for CV: {len(X)} samples, need at least {min_samples_per_fold * (n_splits + 1)}',
                    'direction_accuracy_mean': 0.5,
                    'direction_accuracy_std': 0.0,
                    'balanced_accuracy_mean': 0.5,
                    'f1_macro_mean': 0.5
                }

            # Create model
            model = model_class(**model_params)

            # Use TimeSeriesSplit for temporal integrity
            test_size = max(min_samples_per_fold, len(X) // (n_splits + 1))
            n_splits = min(n_splits, max(2, len(X) // test_size - 1))

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

            # Perform cross-validation with multiple metrics
            accuracy_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            balanced_accuracy_scores = cross_val_score(model, X, y, cv=tscv, scoring='balanced_accuracy')
            f1_scores = cross_val_score(model, X, y, cv=tscv, scoring='f1_macro')

            results = {
                'direction_accuracy_scores': accuracy_scores.tolist(),
                'direction_accuracy_mean': accuracy_scores.mean(),
                'direction_accuracy_std': accuracy_scores.std(),
                'balanced_accuracy_scores': balanced_accuracy_scores.tolist(),
                'balanced_accuracy_mean': balanced_accuracy_scores.mean(),
                'balanced_accuracy_std': balanced_accuracy_scores.std(),
                'f1_scores': f1_scores.tolist(),
                'f1_mean': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'n_splits': n_splits,
                'test_size': test_size
            }

            acc_mean, acc_std = results.get('direction_accuracy_mean'), results.get('direction_accuracy_std')
            bal_mean, bal_std = results.get('balanced_accuracy_mean'), results.get('balanced_accuracy_std')
            f1_mean, f1_std = results.get('f1_mean'), results.get('f1_std')
            acc_mean_s = f"{acc_mean:.4f}" if isinstance(acc_mean, (int, float, np.floating)) else str(acc_mean)
            acc_std_s  = f"{acc_std:.4f}" if isinstance(acc_std,  (int, float, np.floating)) else str(acc_std)
            bal_mean_s = f"{bal_mean:.4f}" if isinstance(bal_mean, (int, float, np.floating)) else str(bal_mean)
            bal_std_s  = f"{bal_std:.4f}" if isinstance(bal_std,  (int, float, np.floating)) else str(bal_std)
            f1_mean_s  = f"{f1_mean:.4f}" if isinstance(f1_mean,  (int, float, np.floating)) else str(f1_mean)
            f1_std_s   = f"{f1_std:.4f}" if isinstance(f1_std,   (int, float, np.floating)) else str(f1_std)
            logger.info(f"✅ CV Results - Accuracy: {acc_mean_s} ± {acc_std_s}")
            logger.info(f"✅ CV Results - Balanced Accuracy: {bal_mean_s} ± {bal_std_s}")
            logger.info(f"✅ CV Results - F1 Macro: {f1_mean_s} ± {f1_std_s}")

            return results

        except Exception as e:
            logger.error(f"❌ Cross-validation failed: {e}")
            return {
                'error': str(e),
                'direction_accuracy_scores': [0.5] * n_splits,
                'direction_accuracy_mean': 0.5,
                'direction_accuracy_std': 0.0,
                'balanced_accuracy_mean': 0.5,
                'f1_mean': 0.5
            }

    @staticmethod
    def calculate_comprehensive_metrics(models_results: Dict[str, Any],
                                      cv_results: Dict[str, Any],
                                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.

        Args:
            models_results: Results from trained models
            cv_results: Cross-validation results
            X_test: Test features
            y_test: Test targets

        Returns:
            Comprehensive evaluation metrics
        """
        try:
            metrics = {
                'cv_results': cv_results,
                'model_performance': {},
                'best_models': {},
                'feature_importance': {},
                'data_quality': {}
            }

            # Analyze individual model performance
            for model_name, model_result in models_results.items():
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    metrics['model_performance'][model_name] = {
                        'direction_accuracy': accuracy,
                        'is_best_direction': accuracy == max([m.get('direction', {}).get('accuracy', 0)
                                                            for m in models_results.values()])
                    }

                if 'volatility' in model_result and 'mae' in model_result['volatility']:
                    mae = model_result['volatility']['mae']
                    metrics['model_performance'][model_name] = metrics['model_performance'].get(model_name, {})
                    metrics['model_performance'][model_name].update({
                        'volatility_mae': mae,
                        'is_best_volatility': mae == min([m.get('volatility', {}).get('mae', float('inf'))
                                                        for m in models_results.values()])
                    })

                # Aggregate feature importance
                if 'feature_importance' in model_result.get('direction', {}):
                    for feature, importance in model_result['direction']['feature_importance'].items():
                        if feature not in metrics['feature_importance']:
                            metrics['feature_importance'][feature] = []
                        metrics['feature_importance'][feature].append(importance)

            # Calculate average feature importance
            for feature in metrics['feature_importance']:
                importances = metrics['feature_importance'][feature]
                metrics['feature_importance'][feature] = {
                    'mean': np.mean(importances),
                    'std': np.std(importances),
                    'count': len(importances)
                }

            # Sort features by importance
            sorted_features = sorted(metrics['feature_importance'].items(),
                                   key=lambda x: x[1]['mean'], reverse=True)
            metrics['top_features'] = dict(sorted_features[:20])

            # Data quality assessment
            if len(y_test) > 0:
                unique_vals, counts = np.unique(y_test, return_counts=True)
                class_dist = dict(zip(unique_vals.tolist(), counts.tolist()))
                is_balanced = False
                try:
                    if len(unique_vals) > 1:
                        ratio_std = np.std(counts / counts.sum())
                        is_balanced = ratio_std < 0.1
                except Exception:
                    is_balanced = False
                metrics['data_quality'] = {
                    'n_test_samples': len(X_test),
                    'n_classes': len(unique_vals),
                    'class_distribution': class_dist,
                    'is_balanced': is_balanced
                }

            return metrics

        except Exception as e:
            logger.error(f"❌ Metrics calculation failed: {e}")
            return {
                'error': str(e),
                'cv_results': cv_results,
                'model_performance': {},
                'data_quality': {}
            }

    @staticmethod
    def classify_ml_error(error: Exception, context: str = "") -> str:
        """
        Classify ML errors for proper handling.

        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred

        Returns:
            Error classification string
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__

        # Classify based on error type and message
        if isinstance(error, SingleClassError) or 'single class' in error_msg:
            return 'SINGLE_CLASS_ERROR'
        elif isinstance(error, ClassImbalanceError) or 'imbalance' in error_msg:
            return 'CLASS_IMBALANCE_ERROR'
        elif isinstance(error, DataQualityError) or 'data quality' in error_msg:
            return 'DATA_QUALITY_ERROR'
        elif isinstance(error, AttributeError) or 'attributeerror' in error_type.lower():
            return 'METHOD_VALIDATION_ERROR'
        elif 'optuna' in error_msg or 'study' in error_msg:
            return 'OPTUNA_ERROR'
        elif 'cross' in error_msg and 'validation' in error_msg:
            return 'CV_ERROR'
        elif 'fit' in error_msg or 'training' in error_msg:
            return 'MODEL_FIT_ERROR'
        elif 'memory' in error_msg or 'out of memory' in error_msg:
            return 'MEMORY_ERROR'
        elif 'timeout' in error_msg:
            return 'TIMEOUT_ERROR'
        else:
            return 'ML_TRAINING_ERROR'

    @staticmethod
    def create_smart_fast_fail_handler(max_failures: int = 5,
                                     critical_threshold: int = 2,
                                     recoverable_threshold: int = 3) -> 'SmartFastFailHandler':
        """
        Create a smart fast-fail handler for ML training.

        Args:
            max_failures: Maximum number of failures before fast fail
            critical_threshold: Threshold for critical errors
            recoverable_threshold: Threshold for recoverable errors

        Returns:
            SmartFastFailHandler instance
        """
        return SmartFastFailHandler(max_failures, critical_threshold, recoverable_threshold)


class SmartFastFailHandler:
    """Smart fast-fail handler for ML training with proper error classification."""

    def __init__(self, max_failures: int = 5, critical_threshold: int = 2, recoverable_threshold: int = 3):
        self.max_failures = max_failures
        self.critical_threshold = critical_threshold
        self.recoverable_threshold = recoverable_threshold
        self.failure_count = 0
        self.critical_failure_count = 0
        self.recoverable_failure_count = 0
        self.failures = []
        self.fast_fail_engaged = False
        self.logger = logger.getChild('SmartFastFailHandler')

    def handle_failure(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle an ML training failure.

        Args:
            error: The exception that occurred
            context: Context where the failure occurred

        Returns:
            Failure handling result
        """
        self.failure_count += 1

        # Classify the error
        error_type = MLTrainingSafeguards.classify_ml_error(error, context)

        # Track failure types
        if error_type in ['SINGLE_CLASS_ERROR', 'DATA_QUALITY_ERROR']:
            self.critical_failure_count += 1
        elif error_type in ['OPTUNA_ERROR', 'CV_ERROR', 'MODEL_FIT_ERROR', 'ML_TRAINING_ERROR']:
            self.recoverable_failure_count += 1

        # Record failure
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'failure_count': self.failure_count
        }
        self.failures.append(failure_record)

        # Determine if fast fail should be triggered
        should_fast_fail = self._should_fast_fail(error_type)

        if should_fast_fail and not self.fast_fail_engaged:
            self.fast_fail_engaged = True
            self.logger.critical(f"🚨 FAST FAIL triggered after {self.failure_count} failures")
            self.logger.critical(f"🚨 Critical: {self.critical_failure_count}, Recoverable: {self.recoverable_failure_count}")
            raise RuntimeError(f"Fast fail triggered after {self.failure_count} ML training failures")

        # Log failure
        self._log_failure(failure_record, should_fast_fail)

        # Return fallback result
        return self._create_fallback_result(failure_record)

    def _should_fast_fail(self, error_type: str) -> bool:
        """Determine if fast fail should be triggered."""
        if self.critical_failure_count >= self.critical_threshold:
            return True
        elif self.recoverable_failure_count >= self.recoverable_threshold:
            return True
        elif self.failure_count >= self.max_failures:
            return True
        return False

    def _log_failure(self, failure_record: Dict[str, Any], will_fast_fail: bool):
        """Log the failure appropriately."""
        error_type = failure_record['error_type']
        message = failure_record['error_message']

        if error_type in ['SINGLE_CLASS_ERROR', 'DATA_QUALITY_ERROR']:
            level = 'error' if will_fast_fail else 'warning'
            emoji = "❌" if will_fast_fail else "⚠️"
        elif error_type in ['OPTUNA_ERROR', 'CV_ERROR', 'MODEL_FIT_ERROR']:
            level = 'warning'
            emoji = "⚠️"
        else:
            level = 'info'
            emoji = "ℹ️"

        log_msg = f"{emoji} ML Failure #{self.failure_count} ({error_type}): {message[:100]}..."

        if level == 'error':
            self.logger.error(log_msg)
        elif level == 'warning':
            self.logger.warning(log_msg)
        else:
            self.logger.info(log_msg)

    def _create_fallback_result(self, failure_record: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback result for failed training."""
        return {
            'direction_accuracy': 0.5,
            'volatility_mae': 0.1,
            'model_type': 'fallback_due_to_failure',
            'training_samples': 0,
            'failure_reason': failure_record['error_message'],
            'failure_type': failure_record['error_type'],
            'failure_count': self.failure_count,
            'fast_fail_engaged': self.fast_fail_engaged
        }

    def get_failure_summary(self) -> Dict[str, Any]:
        """Get a summary of all failures."""
        return {
            'total_failures': self.failure_count,
            'critical_failures': self.critical_failure_count,
            'recoverable_failures': self.recoverable_failure_count,
            'fast_fail_engaged': self.fast_fail_engaged,
            'failure_types': [f['error_type'] for f in self.failures[-10:]],  # Last 10 failures
            'recent_failures': self.failures[-5:]  # Last 5 failure details
        }


# Convenience functions for easy access
def harmonize_parquet_schema(df: pd.DataFrame, schema_reference: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Convenience function for parquet schema harmonization."""
    return MLTrainingSafeguards.harmonize_parquet_schema(df, schema_reference)

def check_class_distribution(y: np.ndarray, threshold: float = 0.95) -> Dict[str, Any]:
    """Convenience function for class distribution checking."""
    return MLTrainingSafeguards.check_class_distribution(y, threshold)

def validate_chunk_for_training(X: np.ndarray, y: np.ndarray, min_samples_per_class: int = 10) -> Dict[str, Any]:
    """Convenience function for chunk validation."""
    return MLTrainingSafeguards.validate_chunk_for_training(X, y, min_samples_per_class)

def create_balanced_sample_weights(y: np.ndarray, strategy: str = 'balanced') -> np.ndarray:
    """Convenience function for sample weight creation."""
    return MLTrainingSafeguards.create_balanced_sample_weights(y, strategy)

def perform_robust_cross_validation(X: np.ndarray, y: np.ndarray, model_class: Any,
                                 model_params: Dict[str, Any], n_splits: int = 5,
                                 min_samples_per_fold: int = 50) -> Dict[str, Any]:
    """Convenience function for robust cross-validation."""
    return MLTrainingSafeguards.perform_robust_cross_validation(X, y, model_class, model_params, n_splits, min_samples_per_fold)

def calculate_comprehensive_metrics(models_results: Dict[str, Any], cv_results: Dict[str, Any],
                                  X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Convenience function for comprehensive metrics calculation."""
    return MLTrainingSafeguards.calculate_comprehensive_metrics(models_results, cv_results, X_test, y_test)

def classify_ml_error(error: Exception, context: str = "") -> str:
    """Convenience function for error classification."""
    return MLTrainingSafeguards.classify_ml_error(error, context)

def create_smart_fast_fail_handler(max_failures: int = 5, critical_threshold: int = 2,
                                 recoverable_threshold: int = 3) -> SmartFastFailHandler:
    """Convenience function for creating smart fast-fail handler."""
    return MLTrainingSafeguards.create_smart_fast_fail_handler(max_failures, critical_threshold, recoverable_threshold)
