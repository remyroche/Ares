"""
Comprehensive Overfitting Monitoring System for ML Training

This module provides comprehensive overfitting monitoring and detection
capabilities to ensure robust model training across all models in the pipeline.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success

logger = system_logger.getChild('OverfittingMonitoring')

@dataclass
class OverfittingMonitoringConfig:
    """Configuration for overfitting monitoring strategies."""

    # Detection thresholds
    overfitting_threshold: float = 0.15  # Max allowed train/val performance gap
    severe_overfitting_threshold: float = 0.3  # Threshold for severe overfitting
    underfitting_threshold: float = 0.05  # Min performance gap for underfitting detection

    # Performance monitoring
    enable_learning_curve_analysis: bool = True
    learning_curve_min_points: int = 5
    enable_performance_drift_detection: bool = True
    performance_drift_window: int = 10

    # Complexity analysis
    enable_model_complexity_analysis: bool = True
    max_model_complexity_score: float = 0.8
    enable_feature_importance_analysis: bool = True

    # Validation settings
    enable_cross_validation_monitoring: bool = True
    cv_folds: int = 10
    enable_bootstrap_validation: bool = True
    bootstrap_samples: int = 100

    # Early stopping
    enable_early_stopping_monitoring: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4

    # Regularization monitoring
    enable_regularization_monitoring: bool = True
    min_regularization_strength: float = 1e-6
    max_regularization_strength: float = 1.0

    # Ensemble monitoring
    enable_ensemble_diversity_monitoring: bool = True
    min_ensemble_diversity: float = 0.6
    enable_ensemble_stability_analysis: bool = True

class OverfittingMonitoring:
    """
    Comprehensive overfitting monitoring system for ML training.

    This class provides various strategies to detect and prevent overfitting:
    1. Performance gap analysis between train/val/test sets
    2. Learning curve analysis and monitoring
    3. Model complexity assessment
    4. Cross-validation stability monitoring
    5. Regularization effectiveness tracking
    6. Ensemble diversity and stability analysis
    """

    def __init__(self, config: Optional[OverfittingMonitoringConfig] = None):
        """Initialize overfitting monitoring system."""
        self.config = config or OverfittingMonitoringConfig()
        self.logger = logger.getChild('OverfittingMonitoring')

        # Initialize monitoring state
        self.overfitting_detected = False
        self.underfitting_detected = False
        self.model_complexity_score = 0.0

        # Performance tracking
        self.performance_history = []
        self.learning_curves = {}
        self.complexity_metrics = {}
        self.diversity_metrics = {}

        # Monitoring results
        self.monitoring_results = {
            'overfitting_detected': False,
            'underfitting_detected': False,
            'performance_gaps': {},
            'complexity_analysis': {},
            'recommendations': []
        }

        self.logger.info("✅ Overfitting Monitoring system initialized")

    def monitor_model_performance(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        model_name: str = "unknown_model",
        epoch: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Monitor model performance for overfitting detection.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            X_test: Optional test features
            y_test: Optional test targets
            model_name: Name of the model for tracking
            epoch: Optional training epoch for learning curve tracking

        Returns:
            Dictionary containing performance monitoring results
        """
        self.logger.debug(f"🔍 Monitoring performance for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'overfitting_detected': False,
            'underfitting_detected': False,
            'performance_metrics': {},
            'performance_gaps': {},
            'recommendations': []
        }

        try:
            # Calculate performance metrics
            train_metrics = self._calculate_performance_metrics(model, X_train, y_train, "train")
            val_metrics = self._calculate_performance_metrics(model, X_val, y_val, "val")

            if X_test is not None and y_test is not None:
                test_metrics = self._calculate_performance_metrics(model, X_test, y_test, "test")
                results['performance_metrics']['test'] = test_metrics

            results['performance_metrics']['train'] = train_metrics
            results['performance_metrics']['val'] = val_metrics

            # Analyze performance gaps
            performance_gaps = self._analyze_performance_gaps(train_metrics, val_metrics)
            results['performance_gaps'] = performance_gaps

            # Detect overfitting/underfitting
            overfitting_analysis = self._detect_overfitting(train_metrics, val_metrics, test_metrics if X_test is not None else None)
            results.update(overfitting_analysis)

            # Update monitoring state
            if results['overfitting_detected']:
                self.overfitting_detected = True
            if results['underfitting_detected']:
                self.underfitting_detected = True

            # Track learning curve if epoch provided
            if epoch is not None and self.config.enable_learning_curve_analysis:
                self._track_learning_curve(model_name, epoch, train_metrics, val_metrics)

            # Generate recommendations
            results['recommendations'] = self._generate_performance_recommendations(results)

            # Store performance record
            performance_record = {
                'model_name': model_name,
                'timestamp': results['timestamp'],
                'epoch': epoch,
                'train_performance': train_metrics,
                'val_performance': val_metrics,
                'test_performance': test_metrics if X_test is not None else None,
                'overfitting_detected': results['overfitting_detected'],
                'underfitting_detected': results['underfitting_detected']
            }
            self.performance_history.append(performance_record)

            self.logger.debug(f"✅ Performance monitoring completed for {model_name}")

        except Exception as e:
            error_msg = f"Performance monitoring failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review model performance calculation")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _calculate_performance_metrics(
        self,
        model: Any,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        dataset_name: str
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        try:
            # Get predictions
            y_pred = model.predict(X)

            # Handle different prediction formats
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            elif hasattr(y_pred, 'flatten'):
                y_pred = y_pred.flatten()

            # Convert y to numpy array
            if hasattr(y, 'values'):
                y_true = y.values
            else:
                y_true = np.array(y)

            # Calculate regression metrics
            if len(np.unique(y_true)) > 10:  # Likely regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)

                metrics = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
            else:  # Likely classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                # Get unique classes
                classes = np.unique(y_true)

                # Binary classification
                if len(classes) == 2:
                    y_pred_binary = (y_pred > 0.5).astype(int) if y_pred.ndim > 1 else (y_pred > np.median(y_pred)).astype(int)

                    accuracy = accuracy_score(y_true, y_pred_binary)
                    precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
                    recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
                    f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)

                    metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }
                else:  # Multi-class classification
                    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

                    accuracy = accuracy_score(y_true, y_pred_classes)
                    precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred_classes, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred_classes, average='weighted', zero_division=0)

                    metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1': float(f1)
                    }

        except Exception as e:
            self.logger.warning(f"Failed to calculate metrics for {dataset_name}: {e}")
            # Return placeholder metrics
            metrics = {
                'error': str(e),
                'mse': float('inf') if len(np.unique(y)) > 10 else 0.0,
                'accuracy': 0.0
            }

        return metrics

    def _analyze_performance_gaps(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Analyze performance gaps between train and validation sets."""
        gaps = {}

        try:
            # Common metrics to compare
            common_metrics = set(train_metrics.keys()) & set(val_metrics.keys())

            for metric in common_metrics:
                if metric == 'error':
                    continue

                train_val = train_metrics[metric]
                val_val = val_metrics[metric]

                # Calculate relative gap
                if val_val != 0:
                    gap = (train_val - val_val) / abs(val_val)
                else:
                    gap = train_val - val_val

                gaps[f"{metric}_gap"] = float(gap)
                gaps[f"{metric}_train"] = float(train_val)
                gaps[f"{metric}_val"] = float(val_val)

        except Exception as e:
            self.logger.warning(f"Performance gap analysis failed: {e}")
            gaps['error'] = str(e)

        return gaps

    def _detect_overfitting(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        test_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Detect overfitting/underfitting based on performance gaps."""
        results = {
            'overfitting_detected': False,
            'underfitting_detected': False,
            'overfitting_severity': 'none',
            'underfitting_severity': 'none'
        }

        try:
            # Analyze common metrics
            common_metrics = set(train_metrics.keys()) & set(val_metrics.keys())

            if 'error' in common_metrics:
                common_metrics.remove('error')

            if not common_metrics:
                return results

            # Calculate average performance gaps
            gaps = []
            for metric in common_metrics:
                if metric.endswith('_gap'):
                    gaps.append(abs(train_metrics[metric]))

            if not gaps:
                return results

            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)

            # Detect overfitting
            if avg_gap > self.config.overfitting_threshold:
                results['overfitting_detected'] = True

                if max_gap > self.config.severe_overfitting_threshold:
                    results['overfitting_severity'] = 'severe'
                elif avg_gap > self.config.overfitting_threshold * 1.5:
                    results['overfitting_severity'] = 'moderate'
                else:
                    results['overfitting_severity'] = 'mild'

            # Detect underfitting
            if avg_gap < self.config.underfitting_threshold:
                results['underfitting_detected'] = True

                if avg_gap < self.config.underfitting_threshold * 0.5:
                    results['underfitting_severity'] = 'severe'
                else:
                    results['underfitting_severity'] = 'mild'

        except Exception as e:
            self.logger.warning(f"Overfitting detection failed: {e}")
            results['error'] = str(e)

        return results

    def _track_learning_curve(
        self,
        model_name: str,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """Track learning curve for overfitting monitoring."""
        if model_name not in self.learning_curves:
            self.learning_curves[model_name] = {
                'epochs': [],
                'train_metrics': [],
                'val_metrics': []
            }

        # Store learning curve data
        self.learning_curves[model_name]['epochs'].append(epoch)
        self.learning_curves[model_name]['train_metrics'].append(train_metrics)
        self.learning_curves[model_name]['val_metrics'].append(val_metrics)

        # Analyze learning curve for overfitting
        self._analyze_learning_curve(model_name)

    def _analyze_learning_curve(self, model_name: str):
        """Analyze learning curve for overfitting patterns."""
        if model_name not in self.learning_curves:
            return

        curve_data = self.learning_curves[model_name]
        epochs = curve_data['epochs']
        train_metrics = curve_data['train_metrics']
        val_metrics = curve_data['val_metrics']

        if len(epochs) < self.config.learning_curve_min_points:
            return

        try:
            # Extract performance metrics (assuming first numeric metric)
            def extract_metric(metrics_list, metric_name):
                return [m.get(metric_name, 0) for m in metrics_list if metric_name in m]

            # Try common metrics
            for metric in ['accuracy', 'f1', 'r2']:
                train_vals = extract_metric(train_metrics, metric)
                val_vals = extract_metric(val_metrics, metric)

                if len(train_vals) >= self.config.learning_curve_min_points:
                    # Check for overfitting pattern: train continues improving while val plateaus/declines
                    if len(train_vals) >= 3:
                        train_trend = np.polyfit(range(len(train_vals)), train_vals, 1)[0]
                        val_trend = np.polyfit(range(len(val_vals)), val_vals, 1)[0]

                        # Overfitting pattern: train improving, val declining
                        if train_trend > 0 and val_trend < -abs(train_trend) * 0.1:
                            self.logger.warning(f"⚠️ Learning curve suggests overfitting in {model_name}")
                            break

        except Exception as e:
            self.logger.debug(f"Learning curve analysis failed for {model_name}: {e}")

    def _generate_performance_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        try:
            if results.get('overfitting_detected', False):
                severity = results.get('overfitting_severity', 'unknown')

                if severity == 'severe':
                    recommendations.extend([
                        "Implement strong regularization (L1/L2)",
                        "Reduce model complexity (fewer layers/nodes)",
                        "Increase training data",
                        "Use early stopping with strict patience",
                        "Consider ensemble methods",
                        "Implement data augmentation"
                    ])
                elif severity == 'moderate':
                    recommendations.extend([
                        "Increase regularization strength",
                        "Reduce learning rate",
                        "Add dropout layers",
                        "Monitor validation loss more closely"
                    ])
                else:  # mild
                    recommendations.extend([
                        "Monitor model performance regularly",
                        "Consider adding validation-based early stopping",
                        "Review feature importance for pruning"
                    ])

            if results.get('underfitting_detected', False):
                severity = results.get('underfitting_severity', 'unknown')

                if severity == 'severe':
                    recommendations.extend([
                        "Increase model capacity (more layers/nodes)",
                        "Train for more epochs",
                        "Reduce regularization",
                        "Review feature engineering",
                        "Consider more complex model architectures"
                    ])
                else:  # mild
                    recommendations.extend([
                        "Slightly increase model complexity",
                        "Reduce early stopping patience",
                        "Review learning rate schedule"
                    ])

            # Performance drift detection
            if self.config.enable_performance_drift_detection:
                drift_detected = self._detect_performance_drift(results)
                if drift_detected:
                    recommendations.append("Performance drift detected - investigate data distribution changes")

        except Exception as e:
            self.logger.warning(f"Recommendation generation failed: {e}")
            recommendations.append("Review model performance monitoring setup")

        return recommendations

    def _detect_performance_drift(self, results: Dict[str, Any]) -> bool:
        """Detect performance drift over time."""
        # Simple implementation - compare recent performance to historical average
        if len(self.performance_history) < 5:
            return False

        recent_performances = self.performance_history[-5:]
        model_name = results.get('model_name', 'unknown')

        # Filter performances for this model
        model_performances = [p for p in recent_performances if p.get('model_name') == model_name]

        if len(model_performances) < 3:
            return False

        # Extract validation performance
        val_performances = []
        for perf in model_performances:
            val_metrics = perf.get('val_performance', {})
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                    val_performances.append(metric_value)
                    break

        if not val_performances:
            return False

        # Check for significant change in recent performance
        recent_avg = np.mean(val_performances[-2:])
        historical_avg = np.mean(val_performances[:-2])

        if len(val_performances) >= 4:
            relative_change = abs(recent_avg - historical_avg) / abs(historical_avg)
            return relative_change > 0.1  # 10% change threshold

        return False

    def analyze_model_complexity(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        model_name: str = "unknown_model"
    ) -> Dict[str, Any]:
        """
        Analyze model complexity for overfitting risk assessment.

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_name: Name of the model

        Returns:
            Dictionary containing complexity analysis results
        """
        self.logger.debug(f"🔍 Analyzing model complexity for {model_name}")

        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'complexity_score': 0.0,
            'complexity_factors': {},
            'overfitting_risk': 'low',
            'recommendations': []
        }

        try:
            # Calculate complexity metrics
            complexity_factors = self._calculate_model_complexity(model, X_train, y_train)
            results['complexity_factors'] = complexity_factors

            # Calculate overall complexity score
            complexity_score = self._calculate_complexity_score(complexity_factors)
            results['complexity_score'] = complexity_score

            # Assess overfitting risk
            risk_level = self._assess_complexity_risk(complexity_score)
            results['overfitting_risk'] = risk_level

            # Generate recommendations
            results['recommendations'] = self._generate_complexity_recommendations(complexity_factors, risk_level)

            # Store complexity metrics
            self.complexity_metrics[model_name] = {
                'timestamp': results['timestamp'],
                'complexity_score': complexity_score,
                'complexity_factors': complexity_factors,
                'overfitting_risk': risk_level
            }

        except Exception as e:
            error_msg = f"Model complexity analysis failed for {model_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review model complexity analysis setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def _calculate_model_complexity(
        self,
        model: Any,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate various model complexity metrics."""
        complexity_factors = {}

        try:
            # Basic model characteristics
            n_samples, n_features = X_train.shape
            n_classes = len(np.unique(y_train))

            complexity_factors['n_samples'] = n_samples
            complexity_factors['n_features'] = n_features
            complexity_factors['n_classes'] = n_classes
            complexity_factors['feature_to_sample_ratio'] = n_features / n_samples

            # Model-specific complexity
            model_type = str(type(model).__name__).lower()

            if 'randomforest' in model_type:
                complexity_factors.update(self._analyze_random_forest_complexity(model))
            elif 'xgboost' in model_type or 'xgb' in model_type:
                complexity_factors.update(self._analyze_xgboost_complexity(model))
            elif 'lightgbm' in model_type or 'lgb' in model_type:
                complexity_factors.update(self._analyze_lightgbm_complexity(model))
            elif 'neural' in model_type or 'nn' in model_type or 'keras' in model_type:
                complexity_factors.update(self._analyze_neural_network_complexity(model))

            # General complexity indicators
            complexity_factors['model_type_complexity'] = self._get_model_type_complexity(model_type)

        except Exception as e:
            self.logger.warning(f"Model complexity calculation failed: {e}")
            complexity_factors['error'] = str(e)

        return complexity_factors

    def _analyze_random_forest_complexity(self, model: Any) -> Dict[str, float]:
        """Analyze Random Forest model complexity."""
        complexity = {}

        try:
            # Get model parameters
            params = model.get_params()

            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            min_samples_split = params.get('min_samples_split', 2)
            min_samples_leaf = params.get('min_samples_leaf', 1)

            complexity['n_estimators'] = n_estimators
            complexity['max_depth'] = max_depth if max_depth else 20  # Default assumption
            complexity['min_samples_split'] = min_samples_split
            complexity['min_samples_leaf'] = min_samples_leaf

            # Complexity score components
            complexity['tree_complexity'] = (n_estimators * (max_depth or 10)) / 1000
            complexity['regularization_strength'] = min(1.0, min_samples_leaf / 10)

        except Exception as e:
            self.logger.debug(f"Random Forest complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_xgboost_complexity(self, model: Any) -> Dict[str, float]:
        """Analyze XGBoost model complexity."""
        complexity = {}

        try:
            # Get model parameters
            params = model.get_params()

            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', 6)
            learning_rate = params.get('learning_rate', 0.3)
            reg_alpha = params.get('reg_alpha', 0)
            reg_lambda = params.get('reg_lambda', 1)

            complexity['n_estimators'] = n_estimators
            complexity['max_depth'] = max_depth
            complexity['learning_rate'] = learning_rate
            complexity['reg_alpha'] = reg_alpha
            complexity['reg_lambda'] = reg_lambda

            # Complexity score components
            complexity['tree_complexity'] = (n_estimators * max_depth) / 500
            complexity['regularization_strength'] = min(1.0, (reg_alpha + reg_lambda) / 2)

        except Exception as e:
            self.logger.debug(f"XGBoost complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_lightgbm_complexity(self, model: Any) -> Dict[str, float]:
        """Analyze LightGBM model complexity."""
        complexity = {}

        try:
            # Get model parameters
            params = model.get_params()

            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', -1)
            num_leaves = params.get('num_leaves', 31)
            learning_rate = params.get('learning_rate', 0.1)
            reg_alpha = params.get('reg_alpha', 0)
            reg_lambda = params.get('reg_lambda', 0)

            complexity['n_estimators'] = n_estimators
            complexity['max_depth'] = max_depth if max_depth > 0 else 10
            complexity['num_leaves'] = num_leaves
            complexity['learning_rate'] = learning_rate
            complexity['reg_alpha'] = reg_alpha
            complexity['reg_lambda'] = reg_lambda

            # Complexity score components
            complexity['tree_complexity'] = (n_estimators * num_leaves) / 1000
            complexity['regularization_strength'] = min(1.0, (reg_alpha + reg_lambda) / 1)

        except Exception as e:
            self.logger.debug(f"LightGBM complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _analyze_neural_network_complexity(self, model: Any) -> Dict[str, float]:
        """Analyze neural network model complexity."""
        complexity = {}

        try:
            # Get model architecture
            if hasattr(model, 'layers'):
                n_layers = len(model.layers)
                total_params = model.count_params()

                complexity['n_layers'] = n_layers
                complexity['total_params'] = total_params
                complexity['params_per_layer'] = total_params / n_layers if n_layers > 0 else 0

                # Complexity score components
                complexity['architecture_complexity'] = min(1.0, n_layers / 10)
                complexity['parameter_complexity'] = min(1.0, total_params / 100000)

            # Get regularization if available
            if hasattr(model, 'get_config'):
                config = model.get_config()
                if 'dropout' in str(config).lower():
                    complexity['has_dropout'] = 1.0
                else:
                    complexity['has_dropout'] = 0.0

        except Exception as e:
            self.logger.debug(f"Neural network complexity analysis failed: {e}")
            complexity['error'] = str(e)

        return complexity

    def _get_model_type_complexity(self, model_type: str) -> float:
        """Get base complexity score for model type."""
        complexity_scores = {
            'randomforest': 0.7,
            'xgboost': 0.8,
            'lightgbm': 0.8,
            'neural': 0.9,
            'lstm': 0.9,
            'tcn': 0.8,
            'linear': 0.3,
            'ridge': 0.4,
            'lasso': 0.4
        }

        return complexity_scores.get(model_type, 0.5)

    def _calculate_complexity_score(self, complexity_factors: Dict[str, float]) -> float:
        """Calculate overall model complexity score."""
        try:
            # Base complexity from model type
            base_complexity = complexity_factors.get('model_type_complexity', 0.5)

            # Add feature ratio complexity
            feature_ratio = complexity_factors.get('feature_to_sample_ratio', 0)
            ratio_complexity = min(1.0, feature_ratio * 2)

            # Add tree/parameter complexity
            tree_complexity = complexity_factors.get('tree_complexity', 0)
            param_complexity = complexity_factors.get('parameter_complexity', 0)

            # Calculate regularization factor (higher regularization = lower complexity)
            reg_strength = complexity_factors.get('regularization_strength', 0.5)
            regularization_factor = 1 - reg_strength

            # Combine factors
            overall_complexity = (
                base_complexity * 0.4 +
                ratio_complexity * 0.2 +
                max(tree_complexity, param_complexity) * 0.3 +
                regularization_factor * 0.1
            )

            return min(1.0, overall_complexity)

        except Exception as e:
            self.logger.warning(f"Complexity score calculation failed: {e}")
            return 0.5  # Default medium complexity

    def _assess_complexity_risk(self, complexity_score: float) -> str:
        """Assess overfitting risk based on complexity score."""
        if complexity_score > self.config.max_model_complexity_score:
            return 'high'
        elif complexity_score > self.config.max_model_complexity_score * 0.7:
            return 'medium'
        else:
            return 'low'

    def _generate_complexity_recommendations(
        self,
        complexity_factors: Dict[str, float],
        risk_level: str
    ) -> List[str]:
        """Generate recommendations based on complexity analysis."""
        recommendations = []

        try:
            if risk_level == 'high':
                recommendations.extend([
                    "High complexity model detected - consider regularization",
                    "Reduce model depth/complexity to prevent overfitting",
                    "Implement strong L1/L2 regularization",
                    "Consider early stopping with validation monitoring",
                    "Increase training data or use data augmentation"
                ])
            elif risk_level == 'medium':
                recommendations.extend([
                    "Medium complexity model - monitor for overfitting",
                    "Consider adding dropout or batch normalization",
                    "Implement validation-based early stopping",
                    "Monitor validation performance closely"
                ])
            else:
                recommendations.extend([
                    "Low complexity model - consider increasing capacity if underfitting",
                    "Monitor for underfitting and adjust model size accordingly"
                ])

            # Feature-specific recommendations
            feature_ratio = complexity_factors.get('feature_to_sample_ratio', 0)
            if feature_ratio > 0.1:
                recommendations.append("High feature-to-sample ratio - consider feature selection")

            # Regularization recommendations
            reg_strength = complexity_factors.get('regularization_strength', 0.5)
            if reg_strength < 0.3:
                recommendations.append("Low regularization - consider increasing L1/L2 penalties")

        except Exception as e:
            self.logger.warning(f"Complexity recommendations failed: {e}")
            recommendations.append("Review model complexity and regularization settings")

        return recommendations

    def analyze_ensemble_diversity(
        self,
        models: Dict[str, Any],
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        ensemble_name: str = "ensemble"
    ) -> Dict[str, Any]:
        """
        Analyze ensemble diversity for overfitting prevention.

        Args:
            models: Dictionary of base models
            X_val: Validation features
            y_val: Validation targets
            ensemble_name: Name of the ensemble

        Returns:
            Dictionary containing diversity analysis results
        """
        self.logger.debug(f"🔍 Analyzing ensemble diversity for {ensemble_name}")

        results = {
            'ensemble_name': ensemble_name,
            'timestamp': datetime.now().isoformat(),
            'diversity_score': 0.0,
            'correlation_matrix': None,
            'diverse_models': [],
            'overfitting_risk': 'low',
            'recommendations': []
        }

        try:
            if len(models) < 2:
                results['recommendations'].append("Need at least 2 models for diversity analysis")
                return results

            # Get predictions from all models
            predictions = {}
            model_names = []

            for model_name, model in models.items():
                try:
                    pred = model.predict(X_val)
                    if pred.ndim > 1:
                        pred = pred.flatten()
                    predictions[model_name] = pred
                    model_names.append(model_name)
                except Exception as e:
                    self.logger.warning(f"Failed to get predictions from {model_name}: {e}")
                    continue

            if len(predictions) < 2:
                results['recommendations'].append("Insufficient valid model predictions for diversity analysis")
                return results

            # Calculate pairwise correlations
            n_models = len(model_names)
            correlation_matrix = np.zeros((n_models, n_models))

            for i in range(n_models):
                for j in range(n_models):
                    if i != j:
                        pred_i = predictions[model_names[i]]
                        pred_j = predictions[model_names[j]]

                        # Handle NaN values
                        valid_mask = ~(np.isnan(pred_i) | np.isnan(pred_j))
                        if valid_mask.sum() > 10:
                            correlation = np.corrcoef(pred_i[valid_mask], pred_j[valid_mask])[0, 1]
                            correlation_matrix[i, j] = correlation if not np.isnan(correlation) else 0.0

            results['correlation_matrix'] = correlation_matrix.tolist()

            # Calculate diversity score (1 - average correlation)
            avg_correlation = np.mean(correlation_matrix[correlation_matrix != 0])
            diversity_score = 1 - avg_correlation
            results['diversity_score'] = float(diversity_score)

            # Identify diverse models (low correlation with others)
            diverse_models = []
            for i, model_name in enumerate(model_names):
                model_correlations = correlation_matrix[i, correlation_matrix[i] != 0]
                avg_model_correlation = np.mean(model_correlations) if len(model_correlations) > 0 else 0

                if avg_model_correlation < self.config.min_ensemble_diversity:
                    diverse_models.append(model_name)

            results['diverse_models'] = diverse_models

            # Assess ensemble overfitting risk
            if diversity_score < 0.3:
                results['overfitting_risk'] = 'high'
                results['recommendations'].append("Low ensemble diversity - high overfitting risk")
            elif diversity_score < 0.5:
                results['overfitting_risk'] = 'medium'
                results['recommendations'].append("Moderate ensemble diversity - monitor for overfitting")

            # Generate recommendations
            if results['overfitting_risk'] == 'high':
                results['recommendations'].extend([
                    "Add more diverse models to ensemble",
                    "Use different algorithms or feature sets",
                    "Implement model stacking with diverse base learners",
                    "Consider bagging or boosting variants"
                ])
            elif results['overfitting_risk'] == 'medium':
                results['recommendations'].extend([
                    "Monitor ensemble performance on validation set",
                    "Consider adding dropout or regularization",
                    "Implement ensemble pruning based on diversity"
                ])

            # Store diversity metrics
            self.diversity_metrics[ensemble_name] = {
                'timestamp': results['timestamp'],
                'diversity_score': diversity_score,
                'n_models': len(models),
                'diverse_models': diverse_models,
                'overfitting_risk': results['overfitting_risk']
            }

        except Exception as e:
            error_msg = f"Ensemble diversity analysis failed for {ensemble_name}: {e}"
            results['error'] = error_msg
            results['recommendations'].append("Review ensemble diversity analysis setup")
            self.logger.error(f"❌ {error_msg}")

        return results

    def generate_overfitting_report(self) -> Dict[str, Any]:
        """Generate comprehensive overfitting monitoring report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'overfitting_detected': self.overfitting_detected,
                'underfitting_detected': self.underfitting_detected,
                'models_monitored': len(set(p['model_name'] for p in self.performance_history)),
                'total_performance_records': len(self.performance_history),
                'avg_complexity_score': np.mean([m.get('complexity_score', 0) for m in self.complexity_metrics.values()]) if self.complexity_metrics else 0,
                'ensemble_diversity_score': np.mean([d.get('diversity_score', 0) for d in self.diversity_metrics.values()]) if self.diversity_metrics else 0
            },
            'performance_analysis': {
                'performance_history': self.performance_history[-10:],  # Last 10 records
                'learning_curves': self.learning_curves
            },
            'complexity_analysis': {
                'complexity_metrics': self.complexity_metrics,
                'high_risk_models': [
                    name for name, metrics in self.complexity_metrics.items()
                    if metrics.get('overfitting_risk') == 'high'
                ]
            },
            'diversity_analysis': {
                'diversity_metrics': self.diversity_metrics,
                'low_diversity_ensembles': [
                    name for name, metrics in self.diversity_metrics.items()
                    if metrics.get('overfitting_risk') == 'high'
                ]
            },
            'recommendations': self._generate_overall_recommendations()
        }

        return report

    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations based on all monitoring data."""
        recommendations = []

        if self.overfitting_detected:
            recommendations.extend([
                "Implement comprehensive regularization across all models",
                "Reduce model complexity where possible",
                "Increase training data and validation rigor",
                "Monitor validation performance more frequently",
                "Consider ensemble methods with diverse base learners"
            ])

        if self.underfitting_detected:
            recommendations.extend([
                "Increase model capacity for underfitting models",
                "Review feature engineering quality",
                "Reduce regularization for underfitting models",
                "Consider more complex model architectures"
            ])

        # Complexity-based recommendations
        high_risk_models = [
            name for name, metrics in self.complexity_metrics.items()
            if metrics.get('overfitting_risk') == 'high'
        ]

        if high_risk_models:
            recommendations.append(f"Focus regularization efforts on high-risk models: {high_risk_models[:3]}")

        # Diversity-based recommendations
        low_diversity_ensembles = [
            name for name, metrics in self.diversity_metrics.items()
            if metrics.get('overfitting_risk') == 'high'
        ]

        if low_diversity_ensembles:
            recommendations.append(f"Improve ensemble diversity for: {low_diversity_ensembles[:3]}")

        return recommendations

# Convenience functions
def create_overfitting_monitor(config: Optional[OverfittingMonitoringConfig] = None) -> OverfittingMonitoring:
    """Create overfitting monitoring instance."""
    return OverfittingMonitoring(config)

def monitor_model_performance(
    model: Any,
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    X_val: Union[pd.DataFrame, np.ndarray],
    y_val: Union[pd.Series, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    y_test: Optional[Union[pd.Series, np.ndarray]] = None,
    model_name: str = "unknown_model",
    config: Optional[OverfittingMonitoringConfig] = None
) -> Dict[str, Any]:
    """
    Convenience function to monitor model performance for overfitting.

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        X_test: Optional test features
        y_test: Optional test targets
        model_name: Name of the model
        config: Optional configuration

    Returns:
        Dictionary containing performance monitoring results
    """
    monitor = OverfittingMonitoring(config)
    return monitor.monitor_model_performance(
        model, X_train, y_train, X_val, y_val, X_test, y_test, model_name
    )