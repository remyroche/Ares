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
from datetime import datetime, timedelta
from collections import Counter, defaultdict, deque
from enum import Enum
from dataclasses import dataclass, field
import threading
import json
from pathlib import Path
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

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.error_type = "ML_TRAINING_ERROR"
        self.severity = "HIGH"
        self.suggested_actions = ["Review training pipeline configuration", "Check data preprocessing steps"]

    def __str__(self):
        if self.context:
            return f"{self.error_type} ({self.severity}): {super().__str__()} | Context: {self.context}"
        return f"{self.error_type} ({self.severity}): {super().__str__()}"

class ClassImbalanceError(MLTrainingError):
    """Raised when class imbalance is too extreme."""

    def __init__(self, message: str, imbalance_ratio: Optional[float] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "CLASS_IMBALANCE_ERROR"
        self.severity = "CRITICAL"
        self.imbalance_ratio = imbalance_ratio
        self.suggested_actions = [
            "Apply class balancing techniques (SMOTE, undersampling, oversampling)",
            "Adjust class weights in model configuration",
            "Consider using ensemble methods",
            "Review data collection strategy",
            "Implement stratified sampling"
        ]

class SingleClassError(MLTrainingError):
    """Raised when only one class is present."""

    def __init__(self, message: str, dominant_class: Optional[Any] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "SINGLE_CLASS_ERROR"
        self.severity = "CRITICAL"
        self.dominant_class = dominant_class
        self.suggested_actions = [
            "Review data splitting strategy",
            "Check for data leakage issues",
            "Verify temporal splits are not creating single-class chunks",
            "Consider alternative labeling approaches",
            "Implement robust data validation checks"
        ]

class DataQualityError(MLTrainingError):
    """Raised when data quality issues prevent training."""

    def __init__(self, message: str, data_issues: Optional[List[str]] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_type = "DATA_QUALITY_ERROR"
        self.severity = "HIGH"
        self.data_issues = data_issues or []
        self.suggested_actions = [
            "Clean and preprocess data thoroughly",
            "Handle missing values appropriately",
            "Check for data type consistency",
            "Validate data schema and ranges",
            "Implement data quality monitoring"
        ]

class ErrorSeverity(Enum):
    """Error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_QUALITY = "data_quality"
    MODEL_TRAINING = "model_training"
    HPO_OPTIMIZATION = "hpo_optimization"
    VALIDATION = "validation"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    CONVERGENCE = "convergence"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """Context information for an error."""
    timestamp: datetime
    component: str
    function: str
    line_number: int
    error_type: str
    error_message: str
    stack_trace: str
    input_data_shape: Optional[Tuple] = None
    input_data_dtypes: Optional[Dict] = None
    memory_usage: Optional[float] = None
    execution_time: Optional[float] = None
    model_type: Optional[str] = None
    hyperparameters: Optional[Dict] = None
    data_characteristics: Optional[Dict] = None

@dataclass
class ErrorRecord:
    """Complete error record with classification and context."""
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    classification_confidence: float
    suggested_actions: List[str]
    related_errors: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_notes: Optional[str] = None

class MLTrainingSafeguards:
    """Comprehensive safeguards for ML training."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logger.getChild('MLTrainingSafeguards')
        self.config = config or {}

        # Enhanced error tracking
        self.error_history: deque = deque(maxlen=1000)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.component_errors: Dict[str, List[ErrorRecord]] = defaultdict(list)

        # Monitoring configuration
        self.enable_real_time_monitoring = self.config.get('enable_real_time_monitoring', True)
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'critical_errors_per_hour': 5,
            'high_errors_per_hour': 20,
            'same_error_repetition': 10,
            'component_failure_rate': 0.3
        })

        # Monitoring state
        self.lock = threading.Lock()

        # Error classification rules
        self.classification_rules = self._initialize_classification_rules()

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
            from sklearn.model_selection import TimeSeriesSplit
            from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv

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

            # Perform cross-validation with multiple metrics via unified API
            acc = unified_perform_cv(model, X, y, strategy='temporal', cv_folds=n_splits, scoring='accuracy')
            bal = unified_perform_cv(model, X, y, strategy='temporal', cv_folds=n_splits, scoring='balanced_accuracy')
            f1r = unified_perform_cv(model, X, y, strategy='temporal', cv_folds=n_splits, scoring='f1_macro')

            results = {
                'direction_accuracy_scores': acc.get('scores', []) or [],
                'direction_accuracy_mean': float(acc.get('mean', 0.0)),
                'direction_accuracy_std': float(acc.get('std', 0.0)),
                'balanced_accuracy_scores': bal.get('scores', []) or [],
                'balanced_accuracy_mean': float(bal.get('mean', 0.0)),
                'balanced_accuracy_std': float(bal.get('std', 0.0)),
                'f1_scores': f1r.get('scores', []) or [],
                'f1_mean': float(f1r.get('mean', 0.0)),
                'f1_std': float(f1r.get('std', 0.0)),
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

    # ---------------------------------------------------------------------
    # Enhanced monitoring and error detection (ported from standalone module)
    # ---------------------------------------------------------------------
    def _initialize_classification_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error classification rules."""
        return {
            'data_quality': {
                'patterns': [
                    r'single class',
                    r'class imbalance',
                    r'empty data',
                    r'nan values',
                    r'infinite values',
                    r'data type mismatch',
                    r'schema validation'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.DATA_QUALITY,
                'confidence_threshold': 0.8
            },
            'model_training': {
                'patterns': [
                    r'fit failed',
                    r'training error',
                    r'convergence failed',
                    r'gradient explosion',
                    r'gradient vanishing',
                    r'loss nan',
                    r'loss infinite'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.MODEL_TRAINING,
                'confidence_threshold': 0.8
            },
            'memory': {
                'patterns': [
                    r'out of memory',
                    r'memory error',
                    r'oom',
                    r'memory allocation',
                    r'cuda out of memory'
                ],
                'severity': ErrorSeverity.HIGH,
                'category': ErrorCategory.MEMORY,
                'confidence_threshold': 0.9
            },
            'convergence': {
                'patterns': [
                    r'not converged',
                    r'convergence warning',
                    r'max iterations',
                    r'early stopping'
                ],
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.CONVERGENCE,
                'confidence_threshold': 0.7
            }
        }

    def detect_and_classify_error(self, error: Exception, context: Dict[str, Any]) -> ErrorRecord:
        """Detect and classify an error with comprehensive analysis."""
        try:
            # Create error context
            error_context = self._create_error_context(error, context)

            # Classify the error
            classification = self._classify_error(error, error_context)

            # Generate error ID
            error_id = self._generate_error_id(error_context)

            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                severity=classification['severity'],
                category=classification['category'],
                context=error_context,
                classification_confidence=classification['confidence'],
                suggested_actions=classification['suggested_actions']
            )

            # Store error record
            with self.lock:
                self.error_history.append(error_record)
                self.component_errors[error_context.component].append(error_record)
                self.error_patterns[error_record.error_id] += 1

            # Check for alert conditions
            self._check_alert_conditions(error_record)

            # Log error
            self._log_error(error_record)

            return error_record

        except Exception as e:
            self.logger.error(f"❌ Error detection failed: {e}")
            # Return a fallback error record
            return self._create_fallback_error_record(error, context)

    def _create_error_context(self, error: Exception, context: Dict[str, Any]) -> ErrorContext:
        """Create comprehensive error context."""
        try:
            # Extract stack trace information
            tb = traceback.extract_tb(error.__traceback__)
            frame = tb[-1] if tb else None

            return ErrorContext(
                timestamp=datetime.now(),
                component=context.get('component', 'unknown'),
                function=frame.name if frame else 'unknown',
                line_number=frame.lineno if frame else 0,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                input_data_shape=context.get('input_data_shape'),
                input_data_dtypes=context.get('input_data_dtypes'),
                memory_usage=context.get('memory_usage'),
                execution_time=context.get('execution_time'),
                model_type=context.get('model_type'),
                hyperparameters=context.get('hyperparameters'),
                data_characteristics=context.get('data_characteristics')
            )
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create error context: {e}")
            return ErrorContext(
                timestamp=datetime.now(),
                component='unknown',
                function='unknown',
                line_number=0,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc()
            )

    def _classify_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Classify error based on patterns and context."""
        import re

        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        best_match = None
        best_confidence = 0.0

        # Check against classification rules
        for _, rule_config in self.classification_rules.items():
            confidence = self._calculate_classification_confidence(
                error_message, error_type, context, rule_config
            )

            if confidence > best_confidence and confidence >= rule_config['confidence_threshold']:
                best_confidence = confidence
                best_match = rule_config

        # Default classification if no match found
        if best_match is None:
            best_match = {
                'severity': ErrorSeverity.MEDIUM,
                'category': ErrorCategory.UNKNOWN,
                'confidence_threshold': 0.5
            }
            best_confidence = 0.5

        # Generate suggested actions
        suggested_actions = self._generate_suggested_actions(
            best_match['category'], context, best_confidence
        )

        return {
            'severity': best_match['severity'],
            'category': best_match['category'],
            'confidence': best_confidence,
            'suggested_actions': suggested_actions
        }

    def _calculate_classification_confidence(self,
                                             error_message: str,
                                             error_type: str,
                                             context: ErrorContext,
                                             rule_config: Dict[str, Any]) -> float:
        """Calculate confidence score for error classification."""

        confidence = 0.0
        patterns = rule_config['patterns']

        # Pattern matching
        for pattern in patterns:
            if re.search(pattern, error_message, re.IGNORECASE):
                confidence += 0.3

        # Error type matching
        if rule_config['category'].value.replace('_', '') in error_type:
            confidence += 0.2

        # Context-based scoring
        if context.component and rule_config['category'].value in context.component.lower():
            confidence += 0.2

        if context.model_type and rule_config['category'].value in context.model_type.lower():
            confidence += 0.1

        # Data characteristics scoring
        if context.data_characteristics:
            if rule_config['category'] == ErrorCategory.DATA_QUALITY:
                if context.data_characteristics.get('has_nan', False):
                    confidence += 0.1
                if context.data_characteristics.get('has_inf', False):
                    confidence += 0.1
                if context.data_characteristics.get('single_class', False):
                    confidence += 0.2

        return min(confidence, 1.0)

    def _generate_suggested_actions(self,
                                    category: ErrorCategory,
                                    context: ErrorContext,
                                    confidence: float) -> List[str]:
        """Generate suggested actions based on error category."""
        actions = []

        if category == ErrorCategory.DATA_QUALITY:
            actions.extend([
                "Check data preprocessing pipeline",
                "Validate input data schema",
                "Handle missing values appropriately",
                "Check for class imbalance",
                "Verify data types and ranges"
            ])

        elif category == ErrorCategory.MODEL_TRAINING:
            actions.extend([
                "Check model hyperparameters",
                "Verify training data quality",
                "Consider reducing model complexity",
                "Check for gradient issues",
                "Validate loss function"
            ])

        elif category == ErrorCategory.MEMORY:
            actions.extend([
                "Reduce batch size",
                "Use data streaming",
                "Clear unused variables",
                "Consider model quantization",
                "Check for memory leaks"
            ])

        elif category == ErrorCategory.CONVERGENCE:
            actions.extend([
                "Adjust learning rate",
                "Increase maximum iterations",
                "Check for numerical stability",
                "Consider different optimizer",
                "Validate convergence criteria"
            ])

        else:
            actions.extend([
                "Review error logs",
                "Check system resources",
                "Validate configuration",
                "Contact support if persistent"
            ])

        # Add confidence-based actions
        if confidence < 0.7:
            actions.append("Manual review recommended - low classification confidence")

        return actions[:5]

    def _generate_error_id(self, context: ErrorContext) -> str:
        """Generate unique error ID based on context."""
        import hashlib

        # Create hash from key context elements
        key_elements = [
            context.component,
            context.function,
            context.error_type,
            context.error_message[:100]
        ]

        key_string = "|".join(str(elem) for elem in key_elements)
        error_hash = hashlib.md5(key_string.encode()).hexdigest()[:8]

        return f"{context.component}_{context.error_type}_{error_hash}"

    def _check_alert_conditions(self, error_record: ErrorRecord):
        """Check if error conditions warrant alerts."""
        try:
            current_time = datetime.now()
            one_hour_ago = current_time - timedelta(hours=1)

            # Count recent errors by severity
            recent_errors = [
                err for err in self.error_history
                if err.context.timestamp > one_hour_ago
            ]

            critical_count = sum(1 for err in recent_errors if err.severity == ErrorSeverity.CRITICAL)
            high_count = sum(1 for err in recent_errors if err.severity == ErrorSeverity.HIGH)

            # Check thresholds
            if critical_count >= self.alert_thresholds['critical_errors_per_hour']:
                self._trigger_alert("CRITICAL", f"Too many critical errors: {critical_count}")

            if high_count >= self.alert_thresholds['high_errors_per_hour']:
                self._trigger_alert("HIGH", f"Too many high severity errors: {high_count}")

            # Check for repeated errors
            if self.error_patterns[error_record.error_id] >= self.alert_thresholds['same_error_repetition']:
                self._trigger_alert("REPETITION", f"Error repeated {self.error_patterns[error_record.error_id]} times: {error_record.error_id}")

        except Exception as e:
            self.logger.error(f"❌ Alert condition check failed: {e}")

    def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert for critical conditions."""
        alert_message = f"🚨 ALERT [{alert_type}]: {message}"

        # Log alert
        self.logger.critical(alert_message)

        # Save to a file for traceability
        try:
            alert_file = Path("alerts") / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            alert_file.parent.mkdir(exist_ok=True)

            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': alert_type,
                'message': message,
                'error_count': len(self.error_history),
                'component_errors': {k: [er.error_id for er in v] for k, v in self.component_errors.items()}
            }

            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"❌ Failed to save alert: {e}")

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level."""
        log_message = (
            f"Error [{error_record.severity.value.upper()}] "
            f"[{error_record.category.value}] "
            f"[{error_record.context.component}] "
            f"{error_record.context.error_message[:100]}..."
        )

        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _create_fallback_error_record(self, error: Exception, context: Dict[str, Any]) -> ErrorRecord:
        """Create a fallback error record when detection fails."""
        fallback_context = ErrorContext(
            timestamp=datetime.now(),
            component=context.get('component', 'unknown'),
            function='unknown',
            line_number=0,
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc()
        )

        return ErrorRecord(
            error_id=f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.UNKNOWN,
            context=fallback_context,
            classification_confidence=0.0,
            suggested_actions=["Manual investigation required"]
        )

    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary."""
        with self.lock:
            current_time = datetime.now()
            # Compute time windows
            one_hour_ago = current_time - timedelta(hours=1)
            one_day_ago = current_time - timedelta(days=1)

            recent_errors = [err for err in self.error_history if err.context.timestamp > one_hour_ago]
            daily_errors = [err for err in self.error_history if err.context.timestamp > one_day_ago]

            # Count by severity
            severity_counts: Dict[str, int] = defaultdict(int)
            for err in self.error_history:
                severity_counts[err.severity.value] += 1

            # Count by category
            category_counts: Dict[str, int] = defaultdict(int)
            for err in self.error_history:
                category_counts[err.category.value] += 1

            # Count by component
            component_counts: Dict[str, int] = defaultdict(int)
            for err in self.error_history:
                component_counts[err.context.component] += 1

            # Most frequent errors
            most_frequent = sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]

            return {
                'total_errors': len(self.error_history),
                'recent_errors_1h': len(recent_errors),
                'daily_errors': len(daily_errors),
                'severity_distribution': dict(severity_counts),
                'category_distribution': dict(category_counts),
                'component_distribution': dict(component_counts),
                'most_frequent_errors': most_frequent,
                'unresolved_errors': sum(1 for err in self.error_history if not err.resolved)
            }

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
