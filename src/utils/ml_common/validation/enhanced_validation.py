"""
Enhanced Validation for ML Common

Comprehensive model validation procedures with multiple validation strategies,
bootstrap confidence intervals, and robustness testing.

This module is now consolidated with unified_cv.py for all cross-validation functionality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

# Import unified cross-validation for all CV operations
from .unified_cv import UnifiedCrossValidator, UnifiedCVResult, perform_cross_validation

@dataclass
class EnhancedValidationConfig:
    """Configuration for enhanced validation."""

    # Multiple validation strategies
    enable_bootstrap_validation: bool = True
    enable_cross_validation: bool = True
    enable_robustness_testing: bool = True
    enable_confidence_intervals: bool = True

    # Cross-validation settings
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, regular, time_series
    cv_random_state: int = 42

    # Bootstrap settings
    bootstrap_samples: int = 1000
    bootstrap_confidence_level: float = 0.95
    bootstrap_method: str = "percentile"  # percentile, normal, bca

    # Robustness testing
    enable_noise_injection: bool = True
    noise_levels: List[float] = None  # [0.01, 0.05, 0.1]
    enable_feature_perturbation: bool = True
    perturbation_magnitude: float = 0.1

    # Statistical tests
    enable_statistical_tests: bool = True
    significance_level: float = 0.05
    test_for_overfitting: bool = True

    # Reporting
    save_validation_reports: bool = True
    report_directory: str = "reports/enhanced_validation"
    enable_detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.05, 0.1]

@dataclass
class ValidationReport:
    """Comprehensive enhanced validation report."""

    # Basic information
    model_name: str = "unknown"
    model_type: str = "unknown"
    dataset_name: str = "unknown"
    validation_timestamp: str = None

    # Performance metrics
    primary_metric: str = "accuracy"
    metric_value: float = 0.0
    metric_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Cross-validation results
    cv_scores: List[float] = None
    cv_mean: float = 0.0
    cv_std: float = 0.0
    cv_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Bootstrap results
    bootstrap_scores: List[float] = None
    bootstrap_mean: float = 0.0
    bootstrap_std: float = 0.0
    bootstrap_confidence_interval: Tuple[float, float] = (0.0, 0.0)

    # Robustness metrics
    robustness_score: float = 0.0
    noise_sensitivity: float = 0.0
    feature_sensitivity: float = 0.0

    # Statistical test results
    statistical_tests: Dict[str, Any] = None
    overfitting_detected: bool = False
    performance_stability: str = "stable"

    # Validation quality
    validation_quality_score: float = 0.0
    validation_reliability: str = "unknown"
    data_sufficiency: str = "unknown"

    # Recommendations
    recommendations: List[str] = None
    warnings: List[str] = None
    critical_issues: List[str] = None

    # Configuration used
    config_used: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default collections."""
        if self.cv_scores is None:
            self.cv_scores = []
        if self.bootstrap_scores is None:
            self.bootstrap_scores = []
        if self.statistical_tests is None:
            self.statistical_tests = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.warnings is None:
            self.warnings = []
        if self.critical_issues is None:
            self.critical_issues = []
        if self.validation_timestamp is None:
            self.validation_timestamp = datetime.now().isoformat()
        if self.config_used is None:
            self.config_used = {}

class EnhancedValidator:
    """Comprehensive enhanced validation system."""

    def __init__(self, config: Optional[EnhancedValidationConfig] = None):
        """
        Initialize enhanced validation system.

        Args:
            config: Configuration for validation
        """
        self.config = config or EnhancedValidationConfig()
        self.validation_history = []

        # Create report directory
        if self.config.save_validation_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Enhanced Validation initialized")

    def validate_model(self,
                      model: Any,
                      X: np.ndarray,
                      y: np.ndarray,
                      model_name: str = "unknown",
                      model_type: str = "unknown",
                      dataset_name: str = "dataset",
                      is_classification: bool = True,
                      cv_folds: Optional[int] = None,
                      random_state: int = 42) -> ValidationReport:
        """
        Perform comprehensive validation of a model.

        Args:
            model: Trained model to validate
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            model_type: Type of the model
            dataset_name: Name of the dataset
            is_classification: Whether it's a classification problem
            cv_folds: Number of CV folds (overrides config)
            random_state: Random state for reproducibility

        Returns:
            ValidationReport with comprehensive analysis
        """
        report = ValidationReport(
            model_name=model_name,
            model_type=model_type,
            dataset_name=dataset_name
        )

        try:
            # Determine primary metric
            report.primary_metric = "accuracy" if is_classification else "r2_score"

            # Cross-validation
            if self.config.enable_cross_validation:
                cv_results = self._perform_cross_validation(
                    model, X, y, is_classification, cv_folds or self.config.cv_folds, random_state
                )
                report.cv_scores = cv_results['scores']
                report.cv_mean = cv_results['mean']
                report.cv_std = cv_results['std']
                report.cv_confidence_interval = cv_results['confidence_interval']

            # Bootstrap validation
            if self.config.enable_bootstrap_validation:
                bootstrap_results = self._perform_bootstrap_validation(
                    model, X, y, is_classification, random_state
                )
                report.bootstrap_scores = bootstrap_results['scores']
                report.bootstrap_mean = bootstrap_results['mean']
                report.bootstrap_std = bootstrap_results['std']
                report.bootstrap_confidence_interval = bootstrap_results['confidence_interval']

            # Robustness testing
            if self.config.enable_robustness_testing:
                robustness_results = self._perform_robustness_testing(
                    model, X, y, is_classification, random_state
                )
                report.robustness_score = robustness_results['robustness_score']
                report.noise_sensitivity = robustness_results['noise_sensitivity']
                report.feature_sensitivity = robustness_results['feature_sensitivity']

            # Statistical tests
            if self.config.enable_statistical_tests:
                statistical_results = self._perform_statistical_tests(model, X, y, is_classification)
                report.statistical_tests = statistical_results

            # Calculate final metrics
            report = self._calculate_final_metrics(report)

            # Assess validation quality
            report = self._assess_validation_quality(report)

            # Generate recommendations
            report = self._generate_validation_recommendations(report)

            # Store report
            self.validation_history.append(report)

            # Log results
            self._log_validation_report(report)

            return report

        except Exception as e:
            logger.error(f"Enhanced validation failed: {e}")
            report.critical_issues.append(f"Validation failed: {str(e)}")
            return report

    def _perform_cross_validation(self,
                                 model: Any,
                                 X: np.ndarray,
                                 y: np.ndarray,
                                 is_classification: bool,
                                 cv_folds: int,
                                 random_state: int) -> Dict[str, Any]:
        """Perform cross-validation using unified CV system."""
        try:
            # Use unified cross-validation
            strategy = "standard"
            if self.config.cv_strategy == "stratified":
                strategy = "stratified"
            elif self.config.cv_strategy == "temporal":
                strategy = "temporal"

            result = perform_cross_validation(
                model,
                X,
                y,
                strategy=strategy,
                cv_folds=cv_folds,
                scoring="accuracy" if is_classification else "r2",
                random_state=random_state,
                stratified=is_classification if strategy == "stratified" else None,
                temporal_gap=getattr(self.config, 'temporal_gap', 0),
                temporal_test_size=getattr(self.config, 'temporal_test_size', None)
            )

            # Extract results from unified format
            scores = result.get('scores', []) or []
            mean_score = float(result.get('mean', np.mean(scores) if len(scores) else 0.0))
            std_score = float(result.get('std', np.std(scores) if len(scores) else 0.0))

            # Calculate confidence interval
            n = len(scores) if len(scores) else cv_folds
            confidence_interval = (
                mean_score - 1.96 * std_score / np.sqrt(max(1, n)),
                mean_score + 1.96 * std_score / np.sqrt(max(1, n))
            )

            return {
                'scores': scores,
                'mean': mean_score,
                'std': std_score,
                'confidence_interval': confidence_interval
            }

        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {
                'scores': [],
                'mean': 0.0,
                'std': 0.0,
                'confidence_interval': (0.0, 0.0)
            }

    def _perform_bootstrap_validation(self,
                                     model: Any,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     is_classification: bool,
                                     random_state: int) -> Dict[str, Any]:
        """Perform bootstrap validation."""
        try:
            n_samples = len(X)
            bootstrap_scores = []

            # Set random state for reproducibility
            np.random.seed(random_state)

            for _ in range(self.config.bootstrap_samples):
                # Bootstrap sample
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X[indices]
                y_boot = y[indices]

                # Train and evaluate
                model_boot = type(model)()
                model_boot.fit(X_boot, y_boot)

                if is_classification:
                    y_pred = model_boot.predict(X)
                    score = accuracy_score(y, y_pred)
                else:
                    y_pred = model_boot.predict(X)
                    score = 1 - np.mean((y - y_pred) ** 2) / np.var(y)

                bootstrap_scores.append(score)

            bootstrap_scores = np.array(bootstrap_scores)
            mean_score = np.mean(bootstrap_scores)
            std_score = np.std(bootstrap_scores)

            # Calculate confidence interval
            alpha = 1 - self.config.bootstrap_confidence_level
            if self.config.bootstrap_method == "percentile":
                lower = np.percentile(bootstrap_scores, alpha/2 * 100)
                upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
            else:
                lower = mean_score - 1.96 * std_score
                upper = mean_score + 1.96 * std_score

            return {
                'scores': bootstrap_scores.tolist(),
                'mean': mean_score,
                'std': std_score,
                'confidence_interval': (lower, upper)
            }

        except Exception as e:
            logger.error(f"Bootstrap validation failed: {e}")
            return {
                'scores': [],
                'mean': 0.0,
                'std': 0.0,
                'confidence_interval': (0.0, 0.0)
            }

    def _perform_robustness_testing(self,
                                   model: Any,
                                   X: np.ndarray,
                                   y: np.ndarray,
                                   is_classification: bool,
                                   random_state: int) -> Dict[str, Any]:
        """Perform robustness testing."""
        try:
            base_score = self._evaluate_model(model, X, y, is_classification)
            robustness_score = 0.0
            noise_sensitivity = 0.0
            feature_sensitivity = 0.0

            # Noise injection testing
            if self.config.enable_noise_injection:
                noise_scores = []
                for noise_level in self.config.noise_levels:
                    X_noisy = X + np.random.normal(0, noise_level, X.shape)
                    noisy_score = self._evaluate_model(model, X_noisy, y, is_classification)
                    noise_scores.append(noisy_score)

                if noise_scores:
                    noise_sensitivity = np.mean([abs(base_score - score) for score in noise_scores])
                    robustness_score += (1 - noise_sensitivity)

            # Feature perturbation testing
            if self.config.enable_feature_perturbation:
                feature_scores = []
                for i in range(min(5, X.shape[1])):  # Test first 5 features
                    X_perturbed = X.copy()
                    X_perturbed[:, i] *= (1 + self.config.perturbation_magnitude)
                    perturbed_score = self._evaluate_model(model, X_perturbed, y, is_classification)
                    feature_scores.append(perturbed_score)

                if feature_scores:
                    feature_sensitivity = np.mean([abs(base_score - score) for score in feature_scores])
                    robustness_score += (1 - feature_sensitivity)

            # Normalize robustness score
            robustness_score = max(0, min(1, robustness_score / 2))

            return {
                'robustness_score': robustness_score,
                'noise_sensitivity': noise_sensitivity,
                'feature_sensitivity': feature_sensitivity
            }

        except Exception as e:
            logger.error(f"Robustness testing failed: {e}")
            return {
                'robustness_score': 0.0,
                'noise_sensitivity': 0.0,
                'feature_sensitivity': 0.0
            }

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, is_classification: bool) -> float:
        """Evaluate model performance."""
        try:
            if is_classification:
                y_pred = model.predict(X)
                return accuracy_score(y, y_pred)
            else:
                y_pred = model.predict(X)
                return 1 - np.mean((y - y_pred) ** 2) / np.var(y)
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0

    def _perform_statistical_tests(self,
                                  model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  is_classification: bool) -> Dict[str, Any]:
        """Perform statistical tests for model validation."""
        results = {}

        try:
            # Basic performance test
            y_pred = model.predict(X)
            if is_classification:
                accuracy = accuracy_score(y, y_pred)
                results['accuracy'] = accuracy

                # Check if accuracy is significantly better than random
                random_accuracy = 1.0 / len(np.unique(y))
                z_score = (accuracy - random_accuracy) / np.sqrt((accuracy * (1 - accuracy)) / len(y))
                results['significantly_better_than_random'] = z_score > 1.96  # 95% confidence
            else:
                r2_score = 1 - np.mean((y - y_pred) ** 2) / np.var(y)
                results['r2_score'] = r2_score

            # Overfitting test using cross-validation
            if self.config.test_for_overfitting:
                try:
                    cv_res = perform_cross_validation(model, X, y, strategy='standard', cv_folds=5, scoring='accuracy' if is_classification else 'r2')
                    cv_scores = np.array(cv_res.get('scores', []) or [])
                    cv_mean = float(cv_res.get('mean', np.mean(cv_scores) if cv_scores.size else 0.0))
                    cv_std = float(cv_res.get('std', np.std(cv_scores) if cv_scores.size else 0.0))
                except Exception:
                    cv_scores = np.array([])
                    cv_mean = 0.0
                    cv_std = 0.0

                if is_classification:
                    train_score = accuracy_score(y, model.predict(X))
                else:
                    train_score = 1 - np.mean((y - model.predict(X)) ** 2) / np.var(y)

                overfitting_gap = train_score - cv_mean
                results['overfitting_gap'] = overfitting_gap
                results['overfitting_detected'] = overfitting_gap > 0.1  # 10% gap suggests overfitting

        except Exception as e:
            logger.error(f"Statistical tests failed: {e}")
            results['error'] = str(e)

        return results

    def _calculate_final_metrics(self, report: ValidationReport) -> ValidationReport:
        """Calculate final aggregated metrics."""
        try:
            # Use bootstrap mean as primary metric if available
            if report.bootstrap_mean > 0:
                report.metric_value = report.bootstrap_mean
                report.metric_confidence_interval = report.bootstrap_confidence_interval
            elif report.cv_mean > 0:
                report.metric_value = report.cv_mean
                report.metric_confidence_interval = report.cv_confidence_interval

            # Calculate validation quality score
            quality_components = []
            if report.cv_std > 0:
                quality_components.append(1 - min(1, report.cv_std / 0.1))  # Lower std = higher quality
            if report.robustness_score > 0:
                quality_components.append(report.robustness_score)
            if report.statistical_tests and 'significantly_better_than_random' in report.statistical_tests:
                quality_components.append(1.0 if report.statistical_tests['significantly_better_than_random'] else 0.5)

            report.validation_quality_score = np.mean(quality_components) if quality_components else 0.5

        except Exception as e:
            logger.error(f"Final metrics calculation failed: {e}")

        return report

    def _assess_validation_quality(self, report: ValidationReport) -> ValidationReport:
        """Assess overall validation quality."""
        try:
            # Validation reliability
            if report.validation_quality_score > 0.8:
                report.validation_reliability = "high"
            elif report.validation_quality_score > 0.6:
                report.validation_reliability = "medium"
            else:
                report.validation_reliability = "low"

            # Data sufficiency
            if len(report.cv_scores) >= 10:
                report.data_sufficiency = "sufficient"
            elif len(report.cv_scores) >= 5:
                report.data_sufficiency = "marginal"
            else:
                report.data_sufficiency = "insufficient"

            # Performance stability
            if report.cv_std < 0.05:
                report.performance_stability = "very_stable"
            elif report.cv_std < 0.1:
                report.performance_stability = "stable"
            elif report.cv_std < 0.2:
                report.performance_stability = "unstable"
            else:
                report.performance_stability = "very_unstable"

        except Exception as e:
            logger.error(f"Validation quality assessment failed: {e}")

        return report

    def _generate_validation_recommendations(self, report: ValidationReport) -> ValidationReport:
        """Generate validation recommendations."""
        try:
            # Data sufficiency recommendations
            if report.data_sufficiency == "insufficient":
                report.recommendations.append("Consider collecting more data for reliable validation")
                report.critical_issues.append("Insufficient data for robust validation")

            # Performance stability recommendations
            if report.performance_stability in ["unstable", "very_unstable"]:
                report.recommendations.append("Model performance is unstable across folds")
                report.recommendations.append("Consider regularization or different model architecture")

            # Overfitting recommendations
            if report.statistical_tests and report.statistical_tests.get('overfitting_detected'):
                report.recommendations.append("Overfitting detected - implement regularization techniques")
                report.recommendations.append("Consider collecting more diverse training data")

            # Robustness recommendations
            if report.robustness_score < 0.7:
                report.recommendations.append("Model shows low robustness to noise and perturbations")
                report.recommendations.append("Consider robust training techniques")

            # Quality-based recommendations
            if report.validation_reliability == "low":
                report.warnings.append("Validation reliability is low - results may not be trustworthy")

            # Positive feedback
            if (report.validation_quality_score > 0.7 and
                report.performance_stability in ["stable", "very_stable"]):
                report.recommendations.append("Validation results are reliable and consistent")

        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")

        return report

    def _log_validation_report(self, report: ValidationReport):
        """Log validation results."""
        if not self.config.enable_detailed_logging:
            return

        logger.info(f"Enhanced Validation Report for {report.model_name}:")
        logger.info(f"  Primary Metric: {report.primary_metric} = {report.metric_value:.4f}")
        logger.info(f"  CV Mean ± Std: {report.cv_mean:.4f} ± {report.cv_std:.4f}")
        logger.info(f"  Bootstrap Mean: {report.bootstrap_mean:.4f}")
        logger.info(f"  Robustness Score: {report.robustness_score:.4f}")
        logger.info(f"  Validation Quality: {report.validation_quality_score:.4f}")
        logger.info(f"  Performance Stability: {report.performance_stability}")

        if report.critical_issues:
            for issue in report.critical_issues:
                logger.error(f"  Critical: {issue}")

        if report.warnings:
            for warning in report.warnings:
                logger.warning(f"  Warning: {warning}")

        if report.recommendations:
            logger.info(f"  Recommendations: {len(report.recommendations)}")
            for rec in report.recommendations[:3]:  # Show first 3
                logger.info(f"    - {rec}")

    def save_validation_report(self, report: ValidationReport, filename: Optional[str] = None):
        """Save validation report to file."""
        if not self.config.save_validation_reports:
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_report_{report.model_name}_{timestamp}.json"

        filepath = Path(self.config.report_directory) / filename

        try:
            report_dict = asdict(report)
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            logger.info(f"Validation report saved: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save validation report: {e}")

    def get_validation_history(self) -> List[ValidationReport]:
        """Get validation history."""
        return self.validation_history.copy()

# Global instance
DEFAULT_ENHANCED_VALIDATOR = EnhancedValidator()

def get_enhanced_validator(config: Optional[EnhancedValidationConfig] = None) -> EnhancedValidator:
    """Get enhanced validator instance."""
    if config is None:
        return DEFAULT_ENHANCED_VALIDATOR
    return EnhancedValidator(config)

def validate_model_comprehensively(model: Any,
                                  X: np.ndarray,
                                  y: np.ndarray,
                                  model_name: str = "unknown",
                                  model_type: str = "unknown",
                                  dataset_name: str = "dataset",
                                  is_classification: bool = True,
                                  cv_folds: Optional[int] = None,
                                  random_state: int = 42) -> ValidationReport:
    """Convenience function to perform comprehensive model validation."""
    validator = get_enhanced_validator()
    return validator.validate_model(
        model, X, y, model_name, model_type, dataset_name,
        is_classification, cv_folds, random_state
    )
