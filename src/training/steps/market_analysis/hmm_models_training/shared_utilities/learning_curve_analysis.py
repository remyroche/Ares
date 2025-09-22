"""
Learning Curve Analysis for HMM Training

Provides learning curve analysis to detect training dynamics and potential overfitting.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)


class LearningCurveAnalyzer:
    """Analyze learning curves to detect training dynamics and overfitting."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the learning curve analyzer.

        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state

    def analyze_learning_curve(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_sizes: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive learning curve analysis.

        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            train_sizes: Training sizes to test (default: np.linspace(0.1, 1.0, 10))

        Returns:
            Dictionary with learning curve analysis results
        """
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        try:
            # Generate learning curves
            train_sizes_abs, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                random_state=self.random_state
            )

            # Calculate learning curve metrics
            analysis = self._analyze_learning_dynamics(
                train_sizes_abs, train_scores, val_scores
            )

            # Add validation performance
            analysis['final_train_accuracy'] = float(np.mean(train_scores[-1]))
            analysis['final_validation_accuracy'] = float(np.mean(val_scores[-1]))
            analysis['test_accuracy'] = accuracy_score(y_test, model.predict(X_test))

            return analysis

        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")
            return {
                'error': str(e),
                'learning_rate': 'unknown',
                'convergence_stability': 'unknown',
                'overfitting_risk': 'high',
                'training_efficiency': 'unknown'
            }

    def _analyze_learning_dynamics(
        self,
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze the dynamics of the learning curve.

        Args:
            train_sizes: Training sizes used
            train_scores: Training scores for each size
            val_scores: Validation scores for each size

        Returns:
            Dictionary with learning dynamics analysis
        """
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
            convergence_stability = 1.0 - np.std(late_scores) / (np.mean(late_scores) + 1e-8)
        else:
            convergence_stability = 1.0 - np.std(val_means) / (np.mean(val_means) + 1e-8)

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

        return {
            'learning_rate': 'fast' if early_slope > 0.1 else 'moderate' if early_slope > 0.05 else 'slow',
            'convergence_stability': 'high' if convergence_stability > 0.8 else 'medium' if convergence_stability > 0.6 else 'low',
            'overfitting_risk': overfitting_risk,
            'training_efficiency': training_efficiency,
            'max_score_gap': float(max_gap),
            'final_score_gap': float(score_gaps[-1]),
            'early_learning_slope': float(early_slope),
            'convergence_stability_score': float(convergence_stability),
            'train_sizes': train_sizes.tolist(),
            'train_scores_mean': train_means.tolist(),
            'train_scores_std': train_stds.tolist(),
            'val_scores_mean': val_means.tolist(),
            'val_scores_std': val_stds.tolist(),
            'score_gaps': score_gaps.tolist()
        }

    def analyze_validation_curve(
        self,
        model_class: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_name: str,
        param_range: List[Any]
    ) -> Dict[str, Any]:
        """
        Analyze validation curve for hyperparameter sensitivity.

        Args:
            model_class: Model class to analyze
            X_train: Training features
            y_train: Training labels
            param_name: Name of parameter to vary
            param_range: Range of parameter values to test

        Returns:
            Dictionary with validation curve analysis
        """
        try:
            train_scores, val_scores = validation_curve(
                model_class(), X_train, y_train,
                param_name=param_name,
                param_range=param_range,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
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

    def detect_learning_anomalies(
        self,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect anomalies in learning curves that may indicate issues.

        Args:
            train_scores: Training scores array
            val_scores: Validation scores array

        Returns:
            Dictionary with anomaly detection results
        """
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

        return {
            'anomalies': anomalies,
            'total_anomalies': len(anomalies),
            'max_severity': max([a['severity'] for a in anomalies], default='none')
        }