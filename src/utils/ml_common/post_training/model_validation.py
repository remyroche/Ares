"""
Model Validation Component

This module provides comprehensive model validation capabilities including
cross-validation, holdout validation, and validation reporting.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path

# Common utilities - use lazy imports to avoid circular dependencies
try:
    from src.utils.common_operations import (
        safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
        safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
        safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
        format_datetime, get_file_size, check_disk_space
    )
    from src.utils.lazy_imports import validate_file_path
except ImportError as e:
    # Fallback implementations
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Using fallback implementations for common operations: {e}")
    
    def safe_json_dump(*args, **kwargs):
        import json
        return json.dump(*args, **kwargs)
    
    def safe_json_load(*args, **kwargs):
        import json
        return json.load(*args, **kwargs)
    
    def safe_file_exists(*args, **kwargs):
        import os
        return os.path.exists(*args, **kwargs)
    
    def ensure_directory(*args, **kwargs):
        import os
        return os.makedirs(*args, exist_ok=True)
    
    def safe_mean(*args, **kwargs):
        import numpy as np
        return np.mean(*args, **kwargs)
    
    def safe_std(*args, **kwargs):
        import numpy as np
        return np.std(*args, **kwargs)
    
    def safe_float(*args, **kwargs):
        return float(*args, **kwargs)
    
    def safe_int(*args, **kwargs):
        return int(*args, **kwargs)
    
    def get_current_datetime(*args, **kwargs):
        from datetime import datetime
        return datetime.now()
    
    def safe_append(*args, **kwargs):
        return args[0].append(*args[1:], **kwargs)
    
    def safe_extend(*args, **kwargs):
        return args[0].extend(*args[1:], **kwargs)
    
    def safe_dict_get(*args, **kwargs):
        return args[0].get(*args[1:], **kwargs)
    
    def safe_lower(*args, **kwargs):
        return str(*args).lower()
    
    def safe_upper(*args, **kwargs):
        return str(*args).upper()
    
    def format_datetime(*args, **kwargs):
        return str(*args)
    
    def get_file_size(*args, **kwargs):
        import os
        return os.path.getsize(*args, **kwargs)
    
    def check_disk_space(*args, **kwargs):
        import shutil
        return shutil.disk_usage(*args, **kwargs)
    
    from src.utils.lazy_imports import validate_file_path
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials,
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger

# ML validation
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

@dataclass
class ValidationMetrics:
    """Container for validation metrics."""

    # Cross-validation metrics
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_min: Optional[float] = None
    cv_max: Optional[float] = None

    # Holdout validation metrics
    holdout_score: Optional[float] = None

    # Validation stability
    score_variance: Optional[float] = None
    score_range: Optional[float] = None

    # Metadata
    validation_time: Optional[float] = None
    cv_folds: Optional[int] = None
    sample_count: Optional[int] = None

@dataclass
class ValidationConfig:
    """Configuration for model validation."""

    # Validation settings
    enable_cross_validation: bool = True
    enable_holdout_validation: bool = True
    enable_purged_cv: bool = False
    enable_data_leakage_detection: bool = True
    enable_time_series_validation: bool = True
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # "stratified", "kfold", "time_series"

    # Holdout settings
    holdout_ratio: float = 0.2
    random_state: Optional[int] = None

    # Validation criteria
    min_cv_score: float = 0.5
    max_cv_std: float = 0.1
    min_holdout_score: float = 0.5

    # Output settings
    save_validation_results: bool = True
    generate_validation_report: bool = True
    validation_report_path: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of model validation."""

    # Validation metrics
    cv_metrics: Optional[ValidationMetrics] = None
    holdout_metrics: Optional[ValidationMetrics] = None

    # Overall assessment
    validation_passed: bool = False
    validation_grade: str = "F"  # A, B, C, D, F

    # Stability assessment
    is_stable: bool = False
    stability_grade: str = "F"  # A, B, C, D, F

    # Metadata
    validation_time: float = 0.0
    model_name: str = ""
    validation_timestamp: str = ""

    def __post_init__(self):
        """Calculate validation assessment."""
        self._assess_validation()
        self._assess_stability()

    def _assess_validation(self):
        """Assess overall validation performance."""
        if not self.cv_metrics and not self.holdout_metrics:
            return

        score = 0
        total_metrics = 0

        # Cross-validation assessment
        if self.cv_metrics and self.cv_metrics.cv_mean is not None:
            score += min(self.cv_metrics.cv_mean * 100, 100)
            total_metrics += 1

        # Holdout validation assessment
        if self.holdout_metrics and self.holdout_metrics.holdout_score is not None:
            score += min(self.holdout_metrics.holdout_score * 100, 100)
            total_metrics += 1

        if total_metrics > 0:
            avg_score = score / total_metrics

            if avg_score >= 90:
                self.validation_grade = "A"
            elif avg_score >= 80:
                self.validation_grade = "B"
            elif avg_score >= 70:
                self.validation_grade = "C"
            elif avg_score >= 60:
                self.validation_grade = "D"
            else:
                self.validation_grade = "F"

    def _assess_stability(self):
        """Assess model stability."""
        if not self.cv_metrics:
            return

        # Check if model is stable based on CV variance
        if self.cv_metrics.cv_std is not None:
            if self.cv_metrics.cv_std <= 0.02:
                self.stability_grade = "A"
                self.is_stable = True
            elif self.cv_metrics.cv_std <= 0.05:
                self.stability_grade = "B"
                self.is_stable = True
            elif self.cv_metrics.cv_std <= 0.1:
                self.stability_grade = "C"
                self.is_stable = True
            elif self.cv_metrics.cv_std <= 0.15:
                self.stability_grade = "D"
                self.is_stable = False
            else:
                self.stability_grade = "F"
                self.is_stable = False

class ModelValidator:
    """Comprehensive model validator with cross-validation and holdout testing."""

    def __init__(self, config: ValidationConfig):
        """Initialize the model validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.logger = system_logger.getChild('ModelValidator')

        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to validation config")

    def _apply_intensity_scaling(self, intensity_pct: float) -> ValidationConfig:
        """Apply intensity scaling to the configuration."""
        return ValidationConfig(
            enable_cross_validation=self.config.enable_cross_validation,
            enable_holdout_validation=self.config.enable_holdout_validation,
            cv_folds=max(3, int(self.config.cv_folds * intensity_pct)),
            cv_strategy=self.config.cv_strategy,
            holdout_ratio=self.config.holdout_ratio,
            random_state=self.config.random_state,
            min_cv_score=self.config.min_cv_score,
            max_cv_std=self.config.max_cv_std,
            min_holdout_score=self.config.min_holdout_score,
            save_validation_results=self.config.save_validation_results,
            generate_validation_report=self.config.generate_validation_report,
            validation_report_path=self.config.validation_report_path
        )

    @handles_errors(default_return=None, context='Model validation')
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           model_name: str = "", scoring: str = "accuracy") -> ValidationResult:
        """Validate a trained model.

        Args:
            model: Trained model
            X: Features
            y: Targets
            model_name: Name of the model
            scoring: Scoring metric for validation

        Returns:
            ValidationResult with comprehensive validation metrics
        """
        try:
            self.logger.info(f"🔍 Validating model: {model_name}")
            start_time = time.time()

            # Perform cross-validation
            cv_metrics = None
            if self.config.enable_cross_validation:
                cv_metrics = await self._perform_cross_validation(model, X, y, scoring)

            # Perform holdout validation
            holdout_metrics = None
            if self.config.enable_holdout_validation:
                holdout_metrics = await self._perform_holdout_validation(model, X, y, scoring)

            # Create validation result
            result = ValidationResult(
                cv_metrics=cv_metrics,
                holdout_metrics=holdout_metrics,
                validation_time=time.time() - start_time,
                model_name=model_name,
                validation_timestamp=get_current_datetime()
            )

            # Check if validation passes thresholds
            result.validation_passed = self._check_validation_thresholds(result)

            # Save results if configured
            if self.config.save_validation_results:
                await self._save_validation_results(result)

            # Generate report if configured
            if self.config.generate_validation_report:
                await self._generate_validation_report(result)

            self.logger.info(f"✅ Model validation completed: {result.validation_grade} grade, Stability: {result.stability_grade}")
            return result

        except Exception as e:
            self.logger.exception(f"💥 Error validating model: {e}")
            return ValidationResult(
                validation_passed=False,
                model_name=model_name,
                validation_timestamp=get_current_datetime()
            )

    @handles_errors(default_return=None, context='Cross-validation')
    async def _perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray, scoring: str) -> ValidationMetrics:
        """Perform cross-validation."""
        try:
            self.logger.info(f"🔄 Performing {self.config.cv_folds}-fold cross-validation...")

            from src.utils.ml_common.validation.unified_cv import perform_cross_validation as unified_perform_cv

            # Map strategy
            stratified = True if self.config.cv_strategy == "stratified" else False
            result = unified_perform_cv(
                model,
                X,
                y,
                strategy="standard",
                cv_folds=self.config.cv_folds,
                scoring=scoring,
                random_state=self.config.random_state,
                stratified=stratified,
            )

            scores = result.get('scores', []) or []
            metrics = ValidationMetrics(
                cv_scores=scores,
                cv_mean=float(result.get('mean', np.mean(scores) if len(scores) else 0.0)),
                cv_std=float(result.get('std', np.std(scores) if len(scores) else 0.0)),
                cv_min=float(result.get('min', np.min(scores))) if len(scores) else 0.0,
                cv_max=float(result.get('max', np.max(scores))) if len(scores) else 0.0,
                score_variance=float(np.var(scores)) if len(scores) else 0.0,
                score_range=(float(np.max(scores) - np.min(scores)) if len(scores) else 0.0),
                cv_folds=self.config.cv_folds,
                sample_count=len(X)
            )

            self.logger.info(f"📊 CV Results: Mean={metrics.cv_mean:.4f}, Std={metrics.cv_std:.4f}")
            return metrics

        except Exception as e:
            self.logger.exception(f"💥 Error in cross-validation: {e}")
            return ValidationMetrics()

    @handles_errors(default_return=None, context='Holdout validation')
    async def _perform_holdout_validation(self, model: Any, X: np.ndarray, y: np.ndarray, scoring: str) -> ValidationMetrics:
        """Perform holdout validation."""
        try:
            self.logger.info(f"🔄 Performing holdout validation (holdout ratio: {self.config.holdout_ratio})...")

            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.holdout_ratio,
                random_state=self.config.random_state, stratify=y if self._is_classification_task(y) else None
            )

            # Train model on training set
            model.fit(X_train, y_train)

            # Evaluate on test set
            if scoring == "accuracy":
                score = accuracy_score(y_test, model.predict(X_test))
            elif scoring == "f1":
                score = f1_score(y_test, model.predict(X_test), average='weighted')
            elif scoring == "neg_mean_squared_error":
                score = -mean_squared_error(y_test, model.predict(X_test))
            elif scoring == "r2":
                score = r2_score(y_test, model.predict(X_test))
            else:
                # Default to accuracy
                score = accuracy_score(y_test, model.predict(X_test))

            # Calculate metrics
            metrics = ValidationMetrics(
                holdout_score=score,
                sample_count=len(X_test)
            )

            self.logger.info(f"📊 Holdout Results: Score={score:.4f}")
            return metrics

        except Exception as e:
            self.logger.exception(f"💥 Error in holdout validation: {e}")
            return ValidationMetrics()

    def _is_classification_task(self, y: np.ndarray) -> bool:
        """Determine if this is a classification task."""
        try:
            unique_values = len(np.unique(y))
            return unique_values <= 10  # Assume classification if <= 10 unique values
        except:
            return False

    def _check_validation_thresholds(self, result: ValidationResult) -> bool:
        """Check if validation passes configured thresholds."""
        try:
            # Check cross-validation thresholds
            if result.cv_metrics:
                if result.cv_metrics.cv_mean is not None and result.cv_metrics.cv_mean < self.config.min_cv_score:
                    return False

                if result.cv_metrics.cv_std is not None and result.cv_metrics.cv_std > self.config.max_cv_std:
                    return False

            # Check holdout validation thresholds
            if result.holdout_metrics:
                if result.holdout_metrics.holdout_score is not None and result.holdout_metrics.holdout_score < self.config.min_holdout_score:
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"⚠️ Error checking validation thresholds: {e}")
            return False

    @handles_errors(default_return=None, context='Validation results saving')
    async def _save_validation_results(self, result: ValidationResult):
        """Save validation results to file."""
        try:
            results_data = {
                'model_name': result.model_name,
                'validation_timestamp': result.validation_timestamp,
                'validation_time': result.validation_time,
                'validation_passed': result.validation_passed,
                'validation_grade': result.validation_grade,
                'is_stable': result.is_stable,
                'stability_grade': result.stability_grade,
                'cv_metrics': result.cv_metrics.__dict__ if result.cv_metrics else None,
                'holdout_metrics': result.holdout_metrics.__dict__ if result.holdout_metrics else None
            }

            # Save to file
            results_path = f"data_cache/validation_results_{result.model_name}_{get_current_datetime()}.json"
            ensure_directory(Path(results_path).parent)
            safe_json_dump(results_data, results_path)

            self.logger.info(f"💾 Validation results saved to {results_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error saving validation results: {e}")

    @handles_errors(default_return=None, context='Validation report generation')
    async def _generate_validation_report(self, result: ValidationResult):
        """Generate comprehensive validation report."""
        try:
            report_path = self.config.validation_report_path or f"data_cache/validation_report_{result.model_name}_{get_current_datetime()}.txt"
            ensure_directory(Path(report_path).parent)

            with open(report_path, 'w') as f:
                f.write(f"Model Validation Report\n")
                f.write(f"======================\n\n")
                f.write(f"Model Name: {result.model_name}\n")
                f.write(f"Validation Timestamp: {result.validation_timestamp}\n")
                f.write(f"Validation Time: {result.validation_time:.2f}s\n")
                f.write(f"Validation Grade: {result.validation_grade}\n")
                f.write(f"Validation Passed: {result.validation_passed}\n")
                f.write(f"Model Stability: {result.is_stable}\n")
                f.write(f"Stability Grade: {result.stability_grade}\n\n")

                if result.cv_metrics:
                    f.write(f"Cross-Validation Results:\n")
                    f.write(f"------------------------\n")
                    f.write(f"CV Folds: {result.cv_metrics.cv_folds}\n")
                    f.write(f"CV Mean: {result.cv_metrics.cv_mean:.4f}\n")
                    f.write(f"CV Std: {result.cv_metrics.cv_std:.4f}\n")
                    f.write(f"CV Min: {result.cv_metrics.cv_min:.4f}\n")
                    f.write(f"CV Max: {result.cv_metrics.cv_max:.4f}\n")
                    f.write(f"Score Variance: {result.cv_metrics.score_variance:.4f}\n")
                    f.write(f"Score Range: {result.cv_metrics.score_range:.4f}\n")
                    f.write(f"Sample Count: {result.cv_metrics.sample_count}\n\n")

                if result.holdout_metrics:
                    f.write(f"Holdout Validation Results:\n")
                    f.write(f"---------------------------\n")
                    f.write(f"Holdout Score: {result.holdout_metrics.holdout_score:.4f}\n")
                    f.write(f"Sample Count: {result.holdout_metrics.sample_count}\n\n")

            self.logger.info(f"📊 Validation report generated: {report_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error generating validation report: {e}")
