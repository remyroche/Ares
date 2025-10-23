"""
Underfitting Detection System for ML Common

Comprehensive underfitting detection and enhancement recommendations system
that identifies when models are not learning enough from the data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error
from sklearn.model_selection import learning_curve, validation_curve

logger = logging.getLogger(__name__)

@dataclass
class UnderfittingConfig:
    """Configuration for underfitting detection across all ML models."""

    # Performance thresholds
    min_acceptable_score: float = 0.6  # Minimum acceptable performance
    low_performance_threshold: float = 0.4  # Performance below this indicates underfitting
    learning_curve_flat_threshold: float = 0.02  # 2% improvement threshold
    validation_curve_flat_threshold: float = 0.01  # 1% improvement threshold

    # Model complexity analysis
    low_complexity_threshold: float = 0.3  # Low model complexity
    feature_utilization_threshold: float = 0.5  # 50% of features should be utilized
    parameter_utilization_threshold: float = 0.4  # 40% of parameters should be active

    # Learning analysis
    learning_curve_samples: List[int] = None  # Sample sizes for learning curve
    learning_curve_cv_folds: int = 5
    validation_curve_params: Dict[str, List[Any]] = None

    # Enhancement recommendations
    enable_complexity_analysis: bool = True
    enable_feature_analysis: bool = True
    enable_hyperparameter_analysis: bool = True
    enable_ensemble_analysis: bool = True

    # Reporting
    save_reports: bool = True
    report_directory: str = "reports/underfitting"
    enable_visualization: bool = True
    detailed_logging: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.learning_curve_samples is None:
            self.learning_curve_samples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if self.validation_curve_params is None:
            self.validation_curve_params = {
                'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'n_estimators': [10, 20, 50, 100, 200, 300, 400, 500],
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
            }

@dataclass
class UnderfittingReport:
    """Comprehensive underfitting detection report."""

    # Basic metrics
    train_score: float
    val_score: float
    test_score: float
    score_gap: float  # Difference between train and val

    # Underfitting status
    is_underfitting: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    confidence_level: float  # 0.0 to 1.0

    # Detailed analysis
    indicators: List[str]
    warnings: List[str]
    recommendations: List[str]

    # Performance analysis
    learning_curve_analysis: Dict[str, Any] = None
    validation_curve_analysis: Dict[str, Any] = None
    complexity_analysis: Dict[str, Any] = None
    feature_analysis: Dict[str, Any] = None

    # Enhancement suggestions
    complexity_enhancements: List[str] = None
    feature_enhancements: List[str] = None
    hyperparameter_enhancements: List[str] = None
    ensemble_enhancements: List[str] = None

    # Model metadata
    model_name: str = "unknown"
    model_type: str = "unknown"
    fold_number: Optional[int] = None
    detection_timestamp: str = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.detection_timestamp is None:
            self.detection_timestamp = datetime.now().isoformat()

class UnderfittingDetector:
    """Comprehensive underfitting detector for all ML models."""

    def __init__(self, config: Optional[UnderfittingConfig] = None):
        """
        Initialize underfitting detector.

        Args:
            config: Configuration for underfitting detection
        """
        self.config = config or UnderfittingConfig()

        # Create report directory
        if self.config.save_reports:
            Path(self.config.report_directory).mkdir(parents=True, exist_ok=True)

        logger.info("✅ Underfitting Detector initialized")

    def detect_underfitting(self,
                           model: Any,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_val: np.ndarray,
                           y_val: np.ndarray,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None,
                           model_name: str = "model",
                           model_type: str = "unknown",
                           fold_number: Optional[int] = None) -> UnderfittingReport:
        """
        Detect underfitting in a trained model.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            model_name: Name of the model
            model_type: Type of model
            fold_number: Fold number for cross-validation

        Returns:
            UnderfittingReport with comprehensive analysis
        """
        try:
            # Calculate basic scores
            train_score = self._calculate_score(model, X_train, y_train, model_type)
            val_score = self._calculate_score(model, X_val, y_val, model_type)
            test_score = self._calculate_score(model, X_test, y_test, model_type) if X_test is not None else None

            score_gap = train_score - val_score

            # Initialize report
            report = UnderfittingReport(
                train_score=train_score,
                val_score=val_score,
                test_score=test_score or 0.0,
                score_gap=score_gap,
                model_name=model_name,
                model_type=model_type,
                fold_number=fold_number
            )

            # Detect underfitting indicators
            indicators = []
            warnings = []
            recommendations = []

            # 1. Low performance detection
            if val_score < self.config.min_acceptable_score:
                indicators.append("Low validation performance")
                if val_score < self.config.low_performance_threshold:
                    indicators.append("Very low validation performance")
                    warnings.append("Model performance is critically low")
                    recommendations.append("Consider increasing model complexity")
                    recommendations.append("Try different algorithms")
                    recommendations.append("Check data quality and preprocessing")

            # 2. Learning curve analysis
            if self.config.enable_complexity_analysis:
                learning_analysis = self._analyze_learning_curve(
                    model, X_train, y_train, X_val, y_val, model_type
                )
                report.learning_curve_analysis = learning_analysis

                if learning_analysis.get('is_flat', False):
                    indicators.append("Flat learning curve")
                    recommendations.append("Increase model complexity")
                    recommendations.append("Try ensemble methods")

                if learning_analysis.get('low_learning_rate', False):
                    indicators.append("Low learning rate")
                    recommendations.append("Increase learning rate")
                    recommendations.append("Try adaptive learning rates")

            # 3. Validation curve analysis
            if self.config.enable_hyperparameter_analysis:
                validation_analysis = self._analyze_validation_curve(
                    model, X_train, y_train, X_val, y_val, model_type
                )
                report.validation_curve_analysis = validation_analysis

                if validation_analysis.get('underfitting_params', []):
                    indicators.append("Suboptimal hyperparameters")
                    recommendations.extend([
                        "Increase model complexity parameters",
                        "Try different hyperparameter ranges",
                        "Use more sophisticated HPO"
                    ])

            # 4. Complexity analysis
            if self.config.enable_complexity_analysis:
                complexity_analysis = self._analyze_model_complexity(
                    model, X_train, y_train, model_type
                )
                report.complexity_analysis = complexity_analysis

                if complexity_analysis.get('low_complexity', False):
                    indicators.append("Low model complexity")
                    recommendations.append("Increase model complexity")
                    recommendations.append("Add more layers/estimators")
                    recommendations.append("Try ensemble methods")

            # 5. Feature analysis
            if self.config.enable_feature_analysis:
                feature_analysis = self._analyze_feature_utilization(
                    model, X_train, y_train, model_type
                )
                report.feature_analysis = feature_analysis

                if feature_analysis.get('low_utilization', False):
                    indicators.append("Low feature utilization")
                    recommendations.append("Feature engineering")
                    recommendations.append("Try different feature selection")
                    recommendations.append("Add domain-specific features")

            # 6. Ensemble analysis
            if self.config.enable_ensemble_analysis:
                ensemble_enhancements = self._suggest_ensemble_enhancements(
                    model, X_train, y_train, X_val, y_val, model_type
                )
                report.ensemble_enhancements = ensemble_enhancements

                if ensemble_enhancements:
                    recommendations.extend(ensemble_enhancements)

            # Determine underfitting severity
            severity, confidence = self._determine_severity(
                indicators, train_score, val_score, score_gap
            )

            # Update report
            report.is_underfitting = len(indicators) > 0
            report.severity = severity
            report.confidence_level = confidence
            report.indicators = indicators
            report.warnings = warnings
            report.recommendations = recommendations

            # Generate enhancement suggestions
            report.complexity_enhancements = self._suggest_complexity_enhancements(model_type)
            report.feature_enhancements = self._suggest_feature_enhancements(model_type)
            report.hyperparameter_enhancements = self._suggest_hyperparameter_enhancements(model_type)

            logger.info(f"✅ Underfitting detection completed for {model_name}")
            logger.info(f"📊 Underfitting: {report.is_underfitting}, Severity: {severity}")

            return report

        except Exception as e:
            logger.error(f"❌ Underfitting detection failed: {e}")
            return UnderfittingReport(
                train_score=0.0, val_score=0.0, test_score=0.0, score_gap=0.0,
                is_underfitting=False, severity="unknown", confidence_level=0.0,
                indicators=[], warnings=[f"Detection failed: {str(e)}"], recommendations=[]
            )

    def _calculate_score(self, model: Any, X: np.ndarray, y: np.ndarray, model_type: str) -> float:
        """Calculate appropriate score for model type."""
        try:
            if hasattr(model, 'predict_proba'):
                # Classification with probability
                y_pred = model.predict_proba(X)[:, 1]
                return roc_auc_score(y, y_pred)
            elif hasattr(model, 'predict'):
                # Regression or classification
                y_pred = model.predict(X)
                if model_type in ['classification', 'binary', 'multiclass']:
                    return accuracy_score(y, y_pred)
                else:
                    return r2_score(y, y_pred)
            else:
                return 0.0
        except Exception:
            return 0.0

    def _analyze_learning_curve(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Analyze learning curve for underfitting indicators."""
        try:
            # Calculate learning curve
            train_sizes = np.array(self.config.learning_curve_samples) * len(X_train)
            train_sizes = train_sizes.astype(int)

            train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=self.config.learning_curve_cv_folds,
                scoring='accuracy' if 'classification' in model_type else 'r2',
                n_jobs=-1
            )

            # Analyze curve
            train_mean = np.mean(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)

            # Check if curve is flat (low learning rate)
            train_improvement = train_mean[-1] - train_mean[0]
            val_improvement = val_mean[-1] - val_mean[0]

            is_flat = (train_improvement < self.config.learning_curve_flat_threshold and
                      val_improvement < self.config.learning_curve_flat_threshold)

            low_learning_rate = (train_improvement < self.config.learning_curve_flat_threshold)

            return {
                'train_scores': train_scores.tolist(),
                'val_scores': val_scores.tolist(),
                'train_mean': train_mean.tolist(),
                'val_mean': val_mean.tolist(),
                'is_flat': is_flat,
                'low_learning_rate': low_learning_rate,
                'train_improvement': float(train_improvement),
                'val_improvement': float(val_improvement)
            }

        except Exception as e:
            logger.warning(f"Learning curve analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_validation_curve(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Analyze validation curve for underfitting indicators."""
        try:
            underfitting_params = []

            # Test different parameter ranges
            for param_name, param_values in self.config.validation_curve_params.items():
                if hasattr(model, param_name):
                    try:
                        train_scores, val_scores = validation_curve(
                            model, X_train, y_train,
                            param_name=param_name,
                            param_range=param_values,
                            cv=3,
                            scoring='accuracy' if 'classification' in model_type else 'r2',
                            n_jobs=-1
                        )

                        # Check if increasing parameter improves performance
                        val_mean = np.mean(val_scores, axis=1)
                        max_val_score = np.max(val_mean)
                        current_val_score = val_mean[-1]  # Highest parameter value

                        if max_val_score - current_val_score > self.config.validation_curve_flat_threshold:
                            underfitting_params.append(param_name)

                    except Exception as e:
                        logger.warning(f"Validation curve failed for {param_name}: {e}")

            return {
                'underfitting_params': underfitting_params,
                'total_params_tested': len(self.config.validation_curve_params),
                'underfitting_ratio': len(underfitting_params) / len(self.config.validation_curve_params)
            }

        except Exception as e:
            logger.warning(f"Validation curve analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_model_complexity(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Analyze model complexity for underfitting indicators."""
        try:
            complexity_score = 0.0
            low_complexity = False

            # Tree-based models
            if hasattr(model, 'n_estimators'):
                n_estimators = getattr(model, 'n_estimators', 0)
                complexity_score += min(n_estimators / 1000, 1.0)

            if hasattr(model, 'max_depth'):
                max_depth = getattr(model, 'max_depth', 0)
                complexity_score += min(max_depth / 10, 1.0)

            # Neural networks
            if hasattr(model, 'n_layers_'):
                n_layers = getattr(model, 'n_layers_', 0)
                complexity_score += min(n_layers / 10, 1.0)

            if hasattr(model, 'n_hidden_'):
                n_hidden = getattr(model, 'n_hidden_', 0)
                complexity_score += min(n_hidden / 100, 1.0)

            # Linear models
            if hasattr(model, 'coef_'):
                coef = getattr(model, 'coef_', np.array([]))
                if coef.size > 0:
                    non_zero_coef = np.count_nonzero(coef)
                    complexity_score += min(non_zero_coef / len(coef), 1.0)

            # Determine if complexity is low
            low_complexity = complexity_score < self.config.low_complexity_threshold

            return {
                'complexity_score': float(complexity_score),
                'low_complexity': low_complexity,
                'model_attributes': {
                    'n_estimators': getattr(model, 'n_estimators', None),
                    'max_depth': getattr(model, 'max_depth', None),
                    'n_layers': getattr(model, 'n_layers_', None),
                    'n_hidden': getattr(model, 'n_hidden_', None)
                }
            }

        except Exception as e:
            logger.warning(f"Complexity analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_feature_utilization(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Analyze feature utilization for underfitting indicators."""
        try:
            n_features = X_train.shape[1]
            utilized_features = 0
            low_utilization = False

            # Tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                utilized_features = np.count_nonzero(importances > 0.01)  # Features with >1% importance
                utilization_ratio = utilized_features / n_features
                low_utilization = utilization_ratio < self.config.feature_utilization_threshold

            # Linear models
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                if coef.ndim > 1:
                    coef = coef[0]  # Take first class for multi-class
                utilized_features = np.count_nonzero(np.abs(coef) > 0.01)
                utilization_ratio = utilized_features / n_features
                low_utilization = utilization_ratio < self.config.feature_utilization_threshold

            return {
                'n_features': n_features,
                'utilized_features': int(utilized_features),
                'utilization_ratio': float(utilized_features / n_features),
                'low_utilization': low_utilization
            }

        except Exception as e:
            logger.warning(f"Feature utilization analysis failed: {e}")
            return {'error': str(e)}

    def _suggest_ensemble_enhancements(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray, model_type: str) -> List[str]:
        """Suggest ensemble enhancements for underfitting models."""
        enhancements = []

        # Check if model is already an ensemble
        is_ensemble = hasattr(model, 'estimators_') or hasattr(model, 'base_estimator')

        if not is_ensemble:
            enhancements.append("Try ensemble methods (Random Forest, XGBoost, LightGBM)")
            enhancements.append("Implement stacking with diverse base models")
            enhancements.append("Use voting classifiers/regressors")

        # Suggest specific ensemble types based on model type
        if 'classification' in model_type:
            enhancements.append("Try AdaBoost for classification")
            enhancements.append("Use Gradient Boosting for classification")
        else:
            enhancements.append("Try Gradient Boosting for regression")
            enhancements.append("Use Random Forest for regression")

        return enhancements

    def _suggest_complexity_enhancements(self, model_type: str) -> List[str]:
        """Suggest complexity enhancements based on model type."""
        enhancements = []

        if 'tree' in model_type or 'forest' in model_type:
            enhancements.extend([
                "Increase n_estimators (100-1000)",
                "Increase max_depth (5-20)",
                "Decrease min_samples_split (2-10)",
                "Decrease min_samples_leaf (1-5)"
            ])
        elif 'neural' in model_type or 'mlp' in model_type:
            enhancements.extend([
                "Increase hidden layer size (50-500)",
                "Add more hidden layers (2-5)",
                "Decrease regularization (alpha, l1_ratio)",
                "Increase learning rate (0.01-0.1)"
            ])
        elif 'linear' in model_type or 'logistic' in model_type:
            enhancements.extend([
                "Decrease regularization (C=1-100)",
                "Try polynomial features",
                "Use kernel methods (SVM)",
                "Add interaction terms"
            ])

        return enhancements

    def _suggest_feature_enhancements(self, model_type: str) -> List[str]:
        """Suggest feature enhancements based on model type."""
        enhancements = [
            "Engineer polynomial features",
            "Add interaction terms",
            "Create domain-specific features",
            "Try feature selection methods",
            "Use dimensionality reduction (PCA, ICA)",
            "Add temporal features for time series",
            "Create lag features",
            "Add statistical features (mean, std, skew, kurtosis)"
        ]

        return enhancements

    def _suggest_hyperparameter_enhancements(self, model_type: str) -> List[str]:
        """Suggest hyperparameter enhancements based on model type."""
        enhancements = [
            "Use more sophisticated HPO (Bayesian optimization)",
            "Increase HPO trials (100-1000)",
            "Try different parameter ranges",
            "Use multi-objective optimization",
            "Implement automated hyperparameter tuning",
            "Use early stopping with patience",
            "Try different optimization algorithms"
        ]

        return enhancements

    def _determine_severity(self, indicators: List[str], train_score: float, val_score: float, score_gap: float) -> Tuple[str, float]:
        """Determine underfitting severity and confidence."""
        severity_score = 0
        confidence = 0.0

        # Count indicators
        severity_score += len(indicators)

        # Low performance penalty
        if val_score < self.config.low_performance_threshold:
            severity_score += 3
        elif val_score < self.config.min_acceptable_score:
            severity_score += 2

        # Large score gap penalty (indicates overfitting, not underfitting)
        if score_gap > 0.1:
            severity_score -= 1  # Reduce severity if overfitting

        # Determine severity
        if severity_score >= 5:
            severity = "severe"
            confidence = 0.9
        elif severity_score >= 3:
            severity = "moderate"
            confidence = 0.7
        elif severity_score >= 1:
            severity = "mild"
            confidence = 0.5
        else:
            severity = "none"
            confidence = 0.3

        return severity, confidence

# Global instance
DEFAULT_UNDERFITTING_DETECTOR = UnderfittingDetector()

def get_underfitting_detector(config: Optional[UnderfittingConfig] = None) -> UnderfittingDetector:
    """Get underfitting detector instance."""
    if config is None:
        return DEFAULT_UNDERFITTING_DETECTOR
    return UnderfittingDetector(config)

def detect_underfitting_for_model(model: Any,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                X_val: np.ndarray,
                                y_val: np.ndarray,
                                X_test: Optional[np.ndarray] = None,
                                y_test: Optional[np.ndarray] = None,
                                model_name: str = "model",
                                model_type: str = "unknown",
                                fold_number: Optional[int] = None,
                                config: Optional[UnderfittingConfig] = None) -> UnderfittingReport:
    """Convenience function to detect underfitting for a model."""
    detector = get_underfitting_detector(config)
    return detector.detect_underfitting(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        model_name, model_type, fold_number
    )
