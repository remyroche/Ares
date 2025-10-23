"""
Enhanced Learning Curve Analysis for ML Common

Provides comprehensive learning curve analysis capabilities integrated with
existing ml_common evaluation infrastructure and enhanced with adaptive regularization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import logging
from dataclasses import dataclass, field

# Import existing ml_common utilities
try:
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    from src.training.steps.market_analysis.hmm_models_training.shared_utilities.learning_curve_analysis import LearningCurveAnalyzer
    HMM_LEARNING_CURVE_AVAILABLE = True
except ImportError:
    HMM_LEARNING_CURVE_AVAILABLE = False
    LearningCurveAnalyzer = None

logger = logging.getLogger(__name__)

@dataclass
class LearningCurveAnalysisResult:
    """Structured learning curve analysis results."""
    learning_rate: str
    convergence_stability: str
    overfitting_risk: str
    training_efficiency: str
    max_score_gap: float
    final_score_gap: float
    early_learning_slope: float
    convergence_stability_score: float
    train_sizes: List[float]
    train_scores_mean: List[float]
    train_scores_std: List[float]
    val_scores_mean: List[float]
    val_scores_std: List[float]
    score_gaps: List[float]
    final_train_score: Optional[float] = None
    final_validation_score: Optional[float] = None
    test_score: Optional[float] = None
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class EnhancedLearningCurveAnalyzer:
    """Enhanced learning curve analyzer integrated with ml_common infrastructure."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize enhanced learning curve analyzer.

        Args:
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize evaluation utilities
        self.evaluation_utils = EvaluationUtils()

        # Initialize HMM learning curve analyzer if available
        self.hmm_analyzer = None
        if HMM_LEARNING_CURVE_AVAILABLE:
            self.hmm_analyzer = LearningCurveAnalyzer(random_state=random_state)

        logger.info("✅ Enhanced Learning Curve Analyzer initialized")

    def analyze_learning_curve(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        scoring: str = 'accuracy'
    ) -> LearningCurveAnalysisResult:
        """
        Perform comprehensive learning curve analysis.

        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            train_sizes: Training sizes to test
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric to use

        Returns:
            LearningCurveAnalysisResult with comprehensive analysis
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        try:
            # Generate learning curves using sklearn
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            )

            # Calculate learning curve metrics
            analysis = self._analyze_learning_dynamics(
                train_sizes_abs, train_scores, val_scores, X_test, y_test, model
            )

            return LearningCurveAnalysisResult(**analysis)

        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")
            return LearningCurveAnalysisResult(
                learning_rate='unknown',
                convergence_stability='unknown',
                overfitting_risk='high',
                training_efficiency='unknown',
                max_score_gap=0.0,
                final_score_gap=0.0,
                early_learning_slope=0.0,
                convergence_stability_score=0.0,
                train_sizes=[],
                train_scores_mean=[],
                train_scores_std=[],
                val_scores_mean=[],
                val_scores_std=[],
                score_gaps=[],
                anomalies=[{'type': 'analysis_error', 'description': str(e)}],
                recommendations=['Learning curve analysis failed - check data quality and model compatibility']
            )

    def _analyze_learning_dynamics(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model: Any
    ) -> Dict[str, Any]:
        """Analyze the dynamics of the learning curve."""

        # Calculate mean and std for train and validation scores
        train_means = np.mean(train_scores, axis=1)
        train_stds = np.std(train_scores, axis=1)
        val_means = np.mean(val_scores, axis=1)
        val_stds = np.std(val_scores, axis=1)

        # Calculate score gaps
        score_gaps = train_means - val_means

        # Determine learning rate (slope of validation curve in early phase)
        if len(val_means) > 3:
            early_slope = np.polyfit(train_sizes[:4], val_means[:4], 1)[0]
        else:
            early_slope = np.polyfit(train_sizes, val_means, 1)[0]

        # Determine convergence stability (stability of validation scores in late phase)
        if len(val_means) > 3:
            late_scores = val_means[-3:]
            convergence_stability_score = 1.0 - np.std(late_scores) / (np.mean(late_scores) + 1e-8)
        else:
            convergence_stability_score = 1.0 - np.std(val_means) / (np.mean(val_means) + 1e-8)

        # Assess overfitting risk based on score gaps and stability
        max_gap = np.max(score_gaps)
        if max_gap > 0.3:
            overfitting_risk = 'high'
        elif max_gap > 0.15:
            overfitting_risk = 'medium'
        else:
            overfitting_risk = 'low'

        # Assess training efficiency
        if early_slope > 0.1:
            training_efficiency = 'fast'
        elif early_slope > 0.05:
            training_efficiency = 'moderate'
        else:
            training_efficiency = 'slow'

        # Check for underfitting (low final performance)
        final_performance = val_means[-1]
        if final_performance < 0.6:
            training_efficiency = 'underfitting'

        # Determine learning rate category
        if early_slope > 0.1:
            learning_rate = 'fast'
        elif early_slope > 0.05:
            learning_rate = 'moderate'
        else:
            learning_rate = 'slow'

        # Determine convergence stability category
        if convergence_stability_score > 0.8:
            convergence_stability = 'high'
        elif convergence_stability_score > 0.6:
            convergence_stability = 'medium'
        else:
            convergence_stability = 'low'

        # Calculate final scores
        test_predictions = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            # For classification
            test_score = accuracy_score(y_test, test_predictions)
        else:
            # For regression
            test_score = r2_score(y_test, test_predictions)

        # Detect anomalies
        anomalies = self._detect_learning_anomalies(train_scores, val_scores)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overfitting_risk, convergence_stability, training_efficiency, anomalies
        )

        return {
            'learning_rate': learning_rate,
            'convergence_stability': convergence_stability,
            'overfitting_risk': overfitting_risk,
            'training_efficiency': training_efficiency,
            'max_score_gap': float(max_gap),
            'final_score_gap': float(score_gaps[-1]),
            'early_learning_slope': float(early_slope),
            'convergence_stability_score': float(convergence_stability_score),
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_means.tolist(),
            'train_scores_std': train_stds.tolist(),
            'val_scores_mean': val_means.tolist(),
            'val_scores_std': val_stds.tolist(),
            'score_gaps': score_gaps.tolist(),
            'final_train_score': float(train_means[-1]),
            'final_validation_score': float(val_means[-1]),
            'test_score': float(test_score),
            'anomalies': anomalies,
            'recommendations': recommendations
        }

    def _detect_learning_anomalies(
        self,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in learning curves that may indicate issues."""
        anomalies = []

        # Check for training score instability
        train_std = np.std(train_scores)
        if train_std > 0.2:
            anomalies.append({
                'type': 'training_instability',
                'severity': 'high',
                'description': f'Training scores show high variance (std={train_std:.3f})',
                'recommendation': 'Consider increasing regularization or reducing model complexity'
            })

        # Check for validation score plateauing
        val_trend = np.polyfit(range(len(val_scores)), val_scores, 1)[0]
        if abs(val_trend) < 0.01 and val_scores[-1] < 0.7:
            anomalies.append({
                'type': 'validation_plateau',
                'severity': 'medium',
                'description': 'Validation scores have plateaued at suboptimal level',
                'recommendation': 'Consider increasing model capacity or adjusting hyperparameters'
            })

        # Check for sudden drops in validation scores
        val_diffs = np.diff(val_scores)
        significant_drops = np.where(val_diffs < -0.1)[0]
        if len(significant_drops) > 0:
            anomalies.append({
                'type': 'validation_collapse',
                'severity': 'high',
                'description': f'Sudden validation score drops detected at indices: {significant_drops.tolist()}',
                'recommendation': 'Check for data quality issues or reduce learning rate'
            })

        # Check for overfitting indicators
        if len(train_scores) > 0 and len(val_scores) > 0:
            final_gap = train_scores[-1] - val_scores[-1]
            if final_gap > 0.2:
                anomalies.append({
                    'type': 'overfitting_indication',
                    'severity': 'medium',
                    'description': f'Large training-validation gap detected: {final_gap:.3f}',
                    'recommendation': 'Increase regularization or use early stopping'
                })

        return anomalies

    def _generate_recommendations(
        self,
        overfitting_risk: str,
        convergence_stability: str,
        training_efficiency: str,
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Overfitting risk recommendations
        if overfitting_risk == 'high':
            recommendations.append("High overfitting risk detected - increase regularization significantly")
        elif overfitting_risk == 'medium':
            recommendations.append("Moderate overfitting risk detected - consider regularization adjustment")

        # Convergence stability recommendations
        if convergence_stability == 'low':
            recommendations.append("Poor convergence stability - consider adjusting learning rate or model architecture")

        # Training efficiency recommendations
        if training_efficiency == 'underfitting':
            recommendations.append("Underfitting detected - consider increasing model capacity or adjusting hyperparameters")
        elif training_efficiency == 'slow':
            recommendations.append("Slow learning detected - consider increasing learning rate or using faster algorithms")

        # Anomaly-specific recommendations
        for anomaly in anomalies:
            recommendations.append(anomaly.get('recommendation', ''))

        return recommendations

    def analyze_validation_curve(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_name: str,
        param_range: List[Any],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze validation curve for hyperparameter sensitivity.

        Args:
            model_class: Model class to analyze
            X_train: Training features
            y_train: Training labels
            param_name: Name of parameter to vary
            param_range: Range of parameter values to test
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with validation curve analysis
        """
        try:
            train_scores, val_scores = validation_curve(
                model_class(), X_train, y_train,
                param_name=param_name,
                param_range=param_range,
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=self.n_jobs
            )

            # Calculate means and stds
            train_means = np.mean(train_scores, axis=1)
            train_stds = np.std(train_scores, axis=1)
            val_means = np.mean(val_scores, axis=1)
            val_stds = np.std(val_scores, axis=1)

            # Find optimal parameter value
            optimal_idx = np.argmax(val_means)
            optimal_value = param_range[optimal_idx]
            optimal_score = val_means[optimal_idx]

            # Assess parameter sensitivity
            score_range = np.max(val_means) - np.min(val_means)
            if score_range > 0.1:
                sensitivity = 'high'
            elif score_range > 0.05:
                sensitivity = 'medium'
            else:
                sensitivity = 'low'

            return {
                'optimal_value': optimal_value,
                'optimal_score': float(optimal_score),
                'score_range': float(score_range),
                'parameter_sensitivity': sensitivity,
                'param_name': param_name,
                'param_range': param_range,
                'train_scores_mean': train_means.tolist(),
                'train_scores_std': train_stds.tolist(),
                'val_scores_mean': val_means.tolist(),
                'val_scores_std': val_stds.tolist()
            }

        except Exception as e:
            logger.error(f"Validation curve analysis failed: {e}")
            return {
                'error': str(e),
                'param_name': param_name,
                'param_range': param_range
            }
