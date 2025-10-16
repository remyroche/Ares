"""
Universal ML Validation System

Comprehensive validation system that integrates overfitting detection,
temporal validation, and timeframe configuration for all ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json

# Import universal validation components
from .enhanced_overfitting_detection import (
    UniversalOverfittingDetector,
    OverfittingConfig,
    OverfittingReport,
    get_overfitting_detector
)
from .universal_temporal_validation import (
    UniversalTemporalValidator,
    UniversalTemporalCrossValidator,
    TemporalValidationConfig,
    TemporalValidationReport,
    get_temporal_validator,
    get_temporal_cv
)
from ..config.universal_timeframe_config import (
    UniversalTimeframeManager,
    UniversalTimeframeConfig,
    get_timeframe_manager,
    get_timeframe_config,
    validate_timeframe_consistency
)

logger = logging.getLogger(__name__)

@dataclass
class UniversalMLValidationConfig:
    """Comprehensive configuration for universal ML validation."""

    # Overfitting detection
    overfitting_config: OverfittingConfig = None

    # Temporal validation
    temporal_config: TemporalValidationConfig = None

    # Timeframe configuration
    timeframe_config: UniversalTimeframeConfig = None

    # Integration settings
    enable_overfitting_detection: bool = True
    enable_temporal_validation: bool = True
    enable_timeframe_validation: bool = True

    # Reporting settings
    save_comprehensive_reports: bool = True
    report_directory: str = "reports/universal_ml_validation"
    enable_visualization: bool = True
    detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.overfitting_config is None:
            self.overfitting_config = OverfittingConfig()

        if self.temporal_config is None:
            self.temporal_config = TemporalValidationConfig()

        if self.timeframe_config is None:
            self.timeframe_config = UniversalTimeframeConfig()

@dataclass
class UniversalMLValidationReport:
    """Comprehensive validation report for any ML model."""

    # Model metadata
    model_name: str
    model_type: str
    fold_number: Optional[int] = None
    validation_timestamp: str = None

    # Validation results
    timeframe_validation: Dict[str, Any] = None
    temporal_validation: TemporalValidationReport = None
    overfitting_analysis: OverfittingReport = None

    # Overall validation status
    overall_validation_passed: bool = False
    validation_score: float = 0.0

    # Summary
    warnings: List[str] = None
    recommendations: List[str] = None
    critical_issues: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now().isoformat()

        if self.warnings is None:
            self.warnings = []

        if self.recommendations is None:
            self.recommendations = []

        if self.critical_issues is None:
            self.critical_issues = []

class UniversalMLValidator:
    """Universal ML validator for all models."""

    def __init__(self, config: Optional[UniversalMLValidationConfig] = None):
        """
        Initialize universal ML validator.

        Args:
            config: Comprehensive validation configuration
        """
        self.config = config or UniversalMLValidationConfig()

        # Initialize components
        self.overfitting_detector = get_overfitting_detector(self.config.overfitting_config)
        self.temporal_validator = get_temporal_validator(self.config.temporal_config)
        self.temporal_cv = get_temporal_cv(self.config.temporal_config)
        self.timeframe_manager = get_timeframe_manager()

        # Create report directory
        if self.config.save_comprehensive_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        # Track validation history
        self.validation_history = []

    def validate_model(self,
                      model,
                      X_train: np.ndarray,
                      X_val: np.ndarray,
                      y_train: np.ndarray,
                      y_val: np.ndarray,
                      timestamps: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None,
                      model_name: str = "unknown",
                      model_type: str = "unknown",
                      fold_number: Optional[int] = None) -> UniversalMLValidationReport:
        """
        Comprehensive validation for any ML model.

        Args:
            model: Trained ML model
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            timestamps: Optional timestamps for temporal validation
            feature_names: Optional feature names
            model_name: Name of the model
            model_type: Type of model
            fold_number: Fold number for cross-validation

        Returns:
            UniversalMLValidationReport: Comprehensive validation results
        """
        logger.info(f"Starting comprehensive validation for {model_name} ({model_type})")

        # Initialize report
        report = UniversalMLValidationReport(
            model_name=model_name,
            model_type=model_type,
            fold_number=fold_number
        )

        try:
            # 1. Timeframe validation
            if self.config.enable_timeframe_validation:
                timeframe_valid = self._validate_timeframe(model_type, model_name)
                report.timeframe_validation = {
                    'valid': timeframe_valid,
                    'primary_timeframe': self.timeframe_manager.config.primary_timeframe,
                    'model_timeframe': self.timeframe_manager.get_timeframe_for_model(model_type)
                }
            else:
                timeframe_valid = True
                report.timeframe_validation = {'valid': True, 'skipped': True}

            # 2. Temporal validation
            if self.config.enable_temporal_validation:
                temporal_report = self.temporal_validator.validate_temporal_split(
                    X_train, X_val, y_train, y_val, timestamps, model_name, model_type
                )
                report.temporal_validation = temporal_report
                temporal_valid = temporal_report.temporal_order_valid and not temporal_report.leakage_detected
            else:
                temporal_valid = True
                report.temporal_validation = None

            # 3. Overfitting detection
            if self.config.enable_overfitting_detection:
                overfitting_report = self._detect_overfitting(
                    model, X_train, X_val, y_train, y_val, model_name, model_type, fold_number
                )
                report.overfitting_analysis = overfitting_report
                overfitting_ok = not overfitting_report.is_overfitting or overfitting_report.severity == 'none'
            else:
                overfitting_ok = True
                report.overfitting_analysis = None

            # 4. Calculate overall validation status
            validation_components = [timeframe_valid, temporal_valid, overfitting_ok]
            report.overall_validation_passed = all(validation_components)
            report.validation_score = sum(validation_components) / len(validation_components)

            # 5. Generate warnings and recommendations
            self._generate_warnings_and_recommendations(report)

            # 6. Save comprehensive report
            if self.config.save_comprehensive_reports:
                self._save_comprehensive_report(report)

            # 7. Track validation history
            self.validation_history.append(report)

            logger.info(f"Validation completed for {model_name}: {'PASSED' if report.overall_validation_passed else 'FAILED'}")

            return report

        except Exception as e:
            logger.error(f"❌ Validation failed for {model_name}: {e}")
            logger.warning("⚠️ Universal ML validation failed - returning failure report")
            report.overall_validation_passed = False
            report.validation_score = 0.0
            report.critical_issues.append(f"Validation failed: {str(e)}")
            report.critical_issues.append(f"Validation failed: {str(e)}")
            return report

    def _validate_timeframe(self, model_type: str, model_name: str) -> bool:
        """Validate timeframe for model."""
        try:
            primary_timeframe = self.timeframe_manager.config.primary_timeframe
            return self.timeframe_manager.validate_timeframe_consistency(
                primary_timeframe, model_type, model_name
            )
        except Exception as e:
            logger.error(f"❌ Timeframe validation failed: {e}")
            logger.warning("⚠️ Timeframe validation failed - returning False")
            return False

    def _detect_overfitting(self,
                           model,
                           X_train: np.ndarray,
                           X_val: np.ndarray,
                           y_train: np.ndarray,
                           y_val: np.ndarray,
                           model_name: str,
                           model_type: str,
                           fold_number: Optional[int]) -> OverfittingReport:
        """Detect overfitting for model."""
        try:
            # Get predictions
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)

            # Get probabilities if available
            train_probabilities = None
            val_probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    train_probabilities = model.predict_proba(X_train)
                    val_probabilities = model.predict_proba(X_val)
                except Exception as e:
                    logger.error(f"❌ Could not get probabilities from model: {e}")
                    logger.warning("⚠️ Could not get probabilities from model - continuing without probability-based metrics")
                    # Continue without probabilities - they're optional

            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_).flatten()

            # Detect overfitting
            return self.overfitting_detector.detect_overfitting(
                train_predictions=train_predictions,
                val_predictions=val_predictions,
                train_labels=y_train,
                val_labels=y_val,
                train_probabilities=train_probabilities,
                val_probabilities=val_probabilities,
                feature_importance=feature_importance,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number
            )

        except Exception as e:
            logger.error(f"❌ Overfitting detection failed: {e}")
            logger.warning("⚠️ Overfitting detection failed - returning error report")
            return self.overfitting_detector._create_error_report(
                str(e), model_name, model_type, fold_number
            )

    def _generate_warnings_and_recommendations(self, report: UniversalMLValidationReport):
        """Generate warnings and recommendations based on validation results."""
        warnings = []
        recommendations = []
        critical_issues = []

        # Timeframe validation warnings
        if report.timeframe_validation and not report.timeframe_validation.get('valid', True):
            warnings.append("⚠️ Timeframe validation failed")
            recommendations.append("Fix timeframe configuration")

        # Temporal validation warnings
        if report.temporal_validation:
            if not report.temporal_validation.temporal_order_valid:
                warnings.append("⚠️ Temporal order violation detected")
                critical_issues.append("Temporal order violation - potential lookahead bias")
                recommendations.append("Fix temporal order in train/test split")

            if report.temporal_validation.leakage_detected:
                warnings.append("🚨 Data leakage detected")
                critical_issues.append("Data leakage detected - model may be invalid")
                recommendations.append("Investigate and fix data leakage")

        # Overfitting warnings
        if report.overfitting_analysis:
            if report.overfitting_analysis.is_overfitting:
                severity = report.overfitting_analysis.severity
                if severity == 'severe':
                    warnings.append("🚨 SEVERE overfitting detected")
                    critical_issues.append("Severe overfitting - model likely to fail in production")
                    recommendations.extend(report.overfitting_analysis.recommendations)
                elif severity == 'high':
                    warnings.append("⚠️ HIGH overfitting detected")
                    recommendations.extend(report.overfitting_analysis.recommendations)
                else:
                    warnings.append("📊 Moderate overfitting detected")
                    recommendations.extend(report.overfitting_analysis.recommendations)

        # Overall validation warnings
        if not report.overall_validation_passed:
            warnings.append("❌ Overall validation failed")
            recommendations.append("Address all validation issues before deployment")

        report.warnings = warnings
        report.recommendations = recommendations
        report.critical_issues = critical_issues

    def _save_comprehensive_report(self, report: UniversalMLValidationReport):
        """Save comprehensive validation report."""
        try:
            report_dict = asdict(report)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"universal_ml_validation_{report.model_name}_{timestamp}.json"
            filepath = Path(self.config.report_directory) / filename

            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            logger.info(f"Comprehensive validation report saved: {filepath}")

        except Exception as e:
            logger.error(f"❌ Failed to save comprehensive report: {e}")
            logger.warning("⚠️ Comprehensive report save failed - validation results may not be persisted")

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations."""
        if not self.validation_history:
            return {'message': 'No validations performed'}

        total_validations = len(self.validation_history)
        passed_validations = sum(1 for r in self.validation_history if r.overall_validation_passed)
        success_rate = passed_validations / total_validations

        # Model type distribution
        model_type_counts = {}
        for report in self.validation_history:
            model_type = report.model_type
            model_type_counts[model_type] = model_type_counts.get(model_type, 0) + 1

        # Severity distribution
        severity_counts = {}
        for report in self.validation_history:
            if report.overfitting_analysis:
                severity = report.overfitting_analysis.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'success_rate': success_rate,
            'model_type_distribution': model_type_counts,
            'overfitting_severity_distribution': severity_counts,
            'average_validation_score': np.mean([r.validation_score for r in self.validation_history])
        }

# Global validator instance
DEFAULT_ML_VALIDATOR = UniversalMLValidator()

def get_ml_validator(config: Optional[UniversalMLValidationConfig] = None) -> UniversalMLValidator:
    """Get universal ML validator."""
    if config is None:
        return DEFAULT_ML_VALIDATOR
    return UniversalMLValidator(config)

def validate_ml_model(model,
                     X_train: np.ndarray,
                     X_val: np.ndarray,
                     y_train: np.ndarray,
                     y_val: np.ndarray,
                     timestamps: Optional[np.ndarray] = None,
                     feature_names: Optional[List[str]] = None,
                     model_name: str = "unknown",
                     model_type: str = "unknown",
                     fold_number: Optional[int] = None,
                     config: Optional[UniversalMLValidationConfig] = None) -> UniversalMLValidationReport:
    """
    Convenience function to validate any ML model.

    Args:
        model: Trained ML model
        X_train: Training features
        X_val: Validation features
        y_train: Training labels
        y_val: Validation labels
        timestamps: Optional timestamps for temporal validation
        feature_names: Optional feature names
        model_name: Name of the model
        model_type: Type of model
        fold_number: Fold number for cross-validation
        config: Validation configuration

    Returns:
        UniversalMLValidationReport: Comprehensive validation results
    """
    validator = get_ml_validator(config)
    return validator.validate_model(
        model, X_train, X_val, y_train, y_val, timestamps, feature_names,
        model_name, model_type, fold_number
    )
