from src.utils.tprint import tprint

"""
Comprehensive Model Evaluation Utilities

This module provides comprehensive model evaluation utilities with multi-metric assessment,
class imbalance awareness, fairness analysis, and performance stability evaluation.

Key Features:
- Multi-metric evaluation with class imbalance awareness
- Temporal performance stability assessment
- Prediction confidence analysis
- Fairness and bias analysis
- Model calibration assessment
- Robustness to noise evaluation
- Comprehensive evaluation reporting

Built on existing utilities:
- Uses math_validation.py for safe mathematical operations
- Integrates with m1_gpu_utils.py for GPU acceleration
- Leverages common_operations.py for robust error handling
- Builds on existing evaluation patterns from step08
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime
import logging
from collections import defaultdict
import warnings
import time

from ...math_validation import safe_divide, safe_log
from ...common_operations import create_fallback_logger
from src.utils.hardware.m1_gpu_utils import M1GPUManager

# Enhanced dependency management with fast fail
try:
    from ..logger import get_logger
    _LOGGER = get_logger("MLCommon.ModelEvaluation")
    tprint("✅ Custom logger available for MLCommon.ModelEvaluation")
except Exception as e:
    tprint(f"⚠️ Custom logger not available: {e}. Using standard logging.")
    _LOGGER = logging.getLogger("MLCommon.ModelEvaluation")
    _LOGGER.setLevel(logging.INFO)

logger = _LOGGER

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, average_precision_score, brier_score_loss,
        confusion_matrix, classification_report, roc_curve, precision_recall_curve,
        mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    )
    from sklearn.calibration import calibration_curve
    from sklearn.utils.class_weight import compute_sample_weight
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - limited evaluation functionality")


class ModelEvaluationUtilities:
    """Comprehensive model evaluation utilities with advanced metrics and analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model evaluation utilities with configuration."""
        self.config = config or {}
        self.logger = logger.getChild('ModelEvaluation')
        
        _LOGGER.info("🚀 Initializing ModelEvaluationUtilities...")

        # Configuration defaults
        self.enable_gpu = self.config.get('enable_gpu', True)
        self.enable_detailed_metrics = self.config.get('enable_detailed_metrics', True)
        self.confidence_thresholds = self.config.get('confidence_thresholds', [0.5, 0.7, 0.9])
        self.performance_stability_window = self.config.get('performance_stability_window', 30)
        self.fairness_attributes = self.config.get('fairness_attributes', [])

        _LOGGER.info(f"⚙️ Configuration - GPU enabled: {self.enable_gpu}")
        _LOGGER.info(f"⚙️ Configuration - Detailed metrics: {self.enable_detailed_metrics}")
        _LOGGER.info(f"⚙️ Configuration - Confidence thresholds: {self.confidence_thresholds}")
        _LOGGER.info(f"⚙️ Configuration - Stability window: {self.performance_stability_window}")
        _LOGGER.info(f"⚙️ Configuration - Fairness attributes: {len(self.fairness_attributes)}")

        # Initialize utilities
        self.gpu_manager = M1GPUManager() if self.enable_gpu else None
        
        _LOGGER.info("✅ ModelEvaluationUtilities initialized successfully")

    def multi_metric_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: Optional[np.ndarray] = None,
                              task_type: str = 'classification',
                              class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive multi-metric evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities (for classification)
            task_type: 'classification' or 'regression'
            class_names: Class names for classification

        Returns:
            Dictionary with comprehensive evaluation metrics
        """
        start_time = time.time()
        _LOGGER.info(f"📊 Starting comprehensive evaluation for {task_type} task...")
        _LOGGER.info(f"📊 Data shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}")
        if y_prob is not None:
            _LOGGER.info(f"📊 Probability shape: {y_prob.shape}")
        
        try:
            evaluation_results = {
                'task_type': task_type,
                'basic_metrics': {},
                'detailed_metrics': {},
                'confidence_analysis': {},
                'stability_metrics': {},
                'data_characteristics': {},
                'evaluation_metadata': {
                    'n_samples': len(y_true),
                    'evaluation_timestamp': datetime.now().isoformat()
                }
            }

            # Data characteristics
            evaluation_results['data_characteristics'] = self._analyze_data_characteristics(
                y_true, y_pred, y_prob, task_type
            )

            if task_type == 'classification':
                evaluation_results.update(self._evaluate_classification_metrics(
                    y_true, y_pred, y_prob, class_names
                ))
            elif task_type == 'regression':
                evaluation_results.update(self._evaluate_regression_metrics(
                    y_true, y_pred
                ))
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # Confidence analysis (if probabilities available)
            if y_prob is not None and task_type == 'classification':
                evaluation_results['confidence_analysis'] = self._analyze_prediction_confidence(
                    y_true, y_pred, y_prob
                )

            # Stability metrics
            _LOGGER.debug('📊 Calculating stability metrics...')
            evaluation_results['stability_metrics'] = self._calculate_stability_metrics(
                y_true, y_pred
            )

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Comprehensive evaluation completed in {execution_time:.3f}s for {len(y_true)} samples")
            
            # Log key metrics summary
            if 'basic_metrics' in evaluation_results:
                basic_metrics = evaluation_results['basic_metrics']
                if task_type == 'classification' and 'accuracy' in basic_metrics:
                    _LOGGER.info(f"📊 Key metrics - Accuracy: {basic_metrics['accuracy']:.4f}")
                elif task_type == 'regression' and 'mse' in basic_metrics:
                    _LOGGER.info(f"📊 Key metrics - MSE: {basic_metrics['mse']:.4f}")
            
            return evaluation_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Multi-metric evaluation failed after {execution_time:.3f}s: {e}")
            return {'error': str(e), 'task_type': task_type}

    def class_imbalance_aware_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    imbalance_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Calculate metrics with class imbalance awareness.

        Args:
            y_true: True values
            y_pred: Predicted values
            imbalance_threshold: Threshold for extreme imbalance detection

        Returns:
            Class imbalance aware evaluation results
        """
        start_time = time.time()
        _LOGGER.info("⚖️ Starting class imbalance aware metrics calculation...")
        _LOGGER.debug(f"📊 Parameters - Threshold: {imbalance_threshold}, Samples: {len(y_true)}")
        
        try:
            imbalance_results = {
                'imbalance_analysis': {},
                'balanced_metrics': {},
                'class_specific_metrics': {},
                'recommendations': []
            }

            # Analyze class distribution
            unique_classes, class_counts = np.unique(y_true, return_counts=True)
            total_samples = len(y_true)
            class_ratios = class_counts / total_samples

            imbalance_results['imbalance_analysis'] = {
                'n_classes': len(unique_classes),
                'class_distribution': dict(zip(unique_classes, class_counts)),
                'class_ratios': dict(zip(unique_classes, class_ratios)),
                'max_class_ratio': float(np.max(class_ratios)),
                'min_class_ratio': float(np.min(class_ratios)),
                'is_extreme_imbalance': np.max(class_ratios) >= imbalance_threshold
            }
            
            _LOGGER.info(f"📊 Class distribution - Classes: {len(unique_classes)}, "
                        f"Max ratio: {np.max(class_ratios):.3f}, "
                        f"Min ratio: {np.min(class_ratios):.3f}")
            
            if np.max(class_ratios) >= imbalance_threshold:
                _LOGGER.warning(f"⚠️ Extreme class imbalance detected (max ratio: {np.max(class_ratios):.3f})")
            else:
                _LOGGER.info("✅ Class distribution appears balanced")

            # Calculate balanced metrics
            if SKLEARN_AVAILABLE:
                imbalance_results['balanced_metrics'] = {
                    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                    'f1_macro': f1_score(y_true, y_pred, average='macro'),
                    'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
                    'precision_macro': precision_score(y_true, y_pred, average='macro'),
                    'recall_macro': recall_score(y_true, y_pred, average='macro')
                }

            # Class-specific metrics
            class_specific = {}
            for class_label in unique_classes:
                mask = y_true == class_label
                if np.sum(mask) > 0:
                    y_true_class = y_true[mask]
                    y_pred_class = y_pred[mask]

                    class_specific[str(class_label)] = {
                        'precision': precision_score(y_true_class, y_pred_class, pos_label=class_label),
                        'recall': recall_score(y_true_class, y_pred_class, pos_label=class_label),
                        'f1': f1_score(y_true_class, y_pred_class, pos_label=class_label),
                        'support': int(np.sum(mask))
                    }

            imbalance_results['class_specific_metrics'] = class_specific

            # Generate recommendations
            if imbalance_results['imbalance_analysis']['is_extreme_imbalance']:
                imbalance_results['recommendations'].extend([
                    "Consider using class balancing techniques (SMOTE, undersampling, etc.)",
                    "Use balanced accuracy or F1-macro for model evaluation",
                    "Consider class-weighted loss functions during training"
                ])
                _LOGGER.warning("⚠️ Recommendations added for extreme class imbalance")

            execution_time = time.time() - start_time
            _LOGGER.info(f"✅ Class imbalance analysis completed in {execution_time:.3f}s - "
                        f"{len(unique_classes)} classes detected")
            return imbalance_results

        except Exception as e:
            execution_time = time.time() - start_time
            _LOGGER.error(f"❌ Class imbalance aware metrics failed after {execution_time:.3f}s: {e}")
            return {'error': str(e)}

    def temporal_performance_stability(self, predictions_over_time: List[Dict[str, Any]],
                                     time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Assess temporal performance stability.

        Args:
            predictions_over_time: List of prediction results with timestamps
            time_window: Time window for stability calculation

        Returns:
            Temporal stability analysis results
        """
        try:
            self.logger.info("⏰ Analyzing temporal performance stability")

            if time_window is None:
                time_window = self.performance_stability_window

            stability_results = {
                'temporal_windows': [],
                'performance_trends': {},
                'stability_metrics': {},
                'drift_detection': {},
                'recommendations': []
            }

            if not predictions_over_time:
                return stability_results

            # Group predictions by time windows
            windowed_predictions = self._group_predictions_by_time(
                predictions_over_time, time_window
            )

            stability_results['temporal_windows'] = windowed_predictions

            # Calculate performance trends
            if len(windowed_predictions) > 1:
                stability_results['performance_trends'] = self._calculate_performance_trends(
                    windowed_predictions
                )

                stability_results['stability_metrics'] = self._calculate_temporal_stability_metrics(
                    windowed_predictions
                )

                stability_results['drift_detection'] = self._detect_performance_drift(
                    windowed_predictions
                )

            # Generate recommendations
            stability_results['recommendations'] = self._generate_stability_recommendations(
                stability_results
            )

            self.logger.info(f"✅ Temporal stability analysis completed - "
                           f"{len(windowed_predictions)} time windows analyzed")
            return stability_results

        except Exception as e:
            self.logger.error(f"❌ Temporal performance stability analysis failed: {e}")
            return {'error': str(e)}

    def prediction_confidence_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: np.ndarray,
                                     confidence_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Analyze prediction confidence and reliability.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_prob: Prediction probabilities
            confidence_thresholds: Confidence thresholds to analyze

        Returns:
            Prediction confidence analysis results
        """
        try:
            self.logger.info("🎯 Analyzing prediction confidence")

            if confidence_thresholds is None:
                confidence_thresholds = self.confidence_thresholds

            confidence_results = {
                'confidence_distribution': {},
                'threshold_analysis': {},
                'calibration_analysis': {},
                'confidence_metrics': {}
            }

            # Handle multi-class vs binary classification
            if y_prob.shape[1] == 2:
                # Binary classification
                confidence_scores = np.max(y_prob, axis=1)
                predicted_classes = np.argmax(y_prob, axis=1)
            else:
                # Multi-class classification
                confidence_scores = np.max(y_prob, axis=1)
                predicted_classes = np.argmax(y_prob, axis=1)

            confidence_results['confidence_distribution'] = {
                'mean_confidence': float(np.mean(confidence_scores)),
                'std_confidence': float(np.std(confidence_scores)),
                'min_confidence': float(np.min(confidence_scores)),
                'max_confidence': float(np.max(confidence_scores)),
                'median_confidence': float(np.median(confidence_scores))
            }

            # Threshold analysis
            threshold_results = {}
            for threshold in confidence_thresholds:
                high_conf_mask = confidence_scores >= threshold
                high_conf_predictions = y_pred[high_conf_mask]
                high_conf_true = y_true[high_conf_mask]

                if len(high_conf_predictions) > 0:
                    accuracy = accuracy_score(high_conf_true, high_conf_predictions)
                    coverage = np.mean(high_conf_mask)
                else:
                    accuracy = 0.0
                    coverage = 0.0

                threshold_results[str(threshold)] = {
                    'accuracy': accuracy,
                    'coverage': coverage,
                    'n_predictions': int(np.sum(high_conf_mask))
                }

            confidence_results['threshold_analysis'] = threshold_results

            # Calibration analysis
            if SKLEARN_AVAILABLE:
                try:
                    prob_true, prob_pred = calibration_curve(
                        y_true, confidence_scores, n_bins=10, strategy='uniform'
                    )

                    confidence_results['calibration_analysis'] = {
                        'prob_true': prob_true.tolist(),
                        'prob_pred': prob_pred.tolist(),
                        'calibration_error': float(np.mean(np.abs(prob_true - prob_pred)))
                    }
                except Exception as cal_e:
                    self.logger.warning(f"Calibration analysis failed: {cal_e}")

            # Confidence-rejection curve
            confidence_results['confidence_metrics'] = self._calculate_confidence_metrics(
                y_true, y_pred, confidence_scores
            )

            self.logger.info(f"✅ Prediction confidence analysis completed - "
                           f"Mean confidence: {confidence_results['confidence_distribution']['mean_confidence']:.3f}")
            return confidence_results

        except Exception as e:
            self.logger.error(f"❌ Prediction confidence analysis failed: {e}")
            return {'error': str(e)}

    def fairness_and_bias_analysis(self, predictions: np.ndarray,
                                 protected_attributes: List[np.ndarray],
                                 true_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Analyze fairness and bias in model predictions.

        Args:
            predictions: Model predictions
            protected_attributes: List of protected attribute arrays
            true_values: True values for performance fairness analysis

        Returns:
            Fairness and bias analysis results
        """
        try:
            self.logger.info("⚖️ Analyzing fairness and bias")

            fairness_results = {
                'attribute_analysis': {},
                'disparity_metrics': {},
                'fairness_assessment': {},
                'recommendations': []
            }

            # Analyze each protected attribute
            for idx, protected_attr in enumerate(protected_attributes):
                attr_name = f"attribute_{idx}"
                attr_analysis = self._analyze_protected_attribute(
                    predictions, protected_attr, true_values, attr_name
                )
                fairness_results['attribute_analysis'][attr_name] = attr_analysis

            # Calculate overall disparity metrics
            fairness_results['disparity_metrics'] = self._calculate_disparity_metrics(
                fairness_results['attribute_analysis']
            )

            # Overall fairness assessment
            fairness_results['fairness_assessment'] = self._assess_overall_fairness(
                fairness_results['disparity_metrics']
            )

            # Generate recommendations
            fairness_results['recommendations'] = self._generate_fairness_recommendations(
                fairness_results
            )

            self.logger.info(f"✅ Fairness and bias analysis completed - "
                           f"{len(protected_attributes)} attributes analyzed")
            return fairness_results

        except Exception as e:
            self.logger.error(f"❌ Fairness and bias analysis failed: {e}")
            return {'error': str(e)}

    def model_calibration_assessment(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   n_bins: int = 10) -> Dict[str, Any]:
        """
        Assess model calibration quality.

        Args:
            y_true: True values
            y_prob: Prediction probabilities
            n_bins: Number of bins for calibration curve

        Returns:
            Model calibration assessment results
        """
        try:
            self.logger.info("📏 Assessing model calibration")

            calibration_results = {
                'calibration_curve': {},
                'calibration_metrics': {},
                'reliability_diagram': {},
                'assessment': {}
            }

            if not SKLEARN_AVAILABLE:
                return {'error': 'Scikit-learn required for calibration assessment'}

            # Handle binary vs multi-class
            if y_prob.shape[1] == 2:
                # Binary classification
                prob_pos = y_prob[:, 1]

                prob_true, prob_pred = calibration_curve(
                    y_true, prob_pos, n_bins=n_bins, strategy='uniform'
                )

                calibration_results['calibration_curve'] = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist()
                }

                # Calculate calibration metrics
                calibration_error = np.mean(np.abs(prob_true - prob_pred))
                brier_score = brier_score_loss(y_true, prob_pos)

                calibration_results['calibration_metrics'] = {
                    'expected_calibration_error': float(calibration_error),
                    'brier_score': float(brier_score),
                    'max_calibration_error': float(np.max(np.abs(prob_true - prob_pred)))
                }

            else:
                # Multi-class calibration (simplified)
                calibration_results['calibration_metrics'] = {
                    'note': 'Multi-class calibration assessment requires specialized methods'
                }

            # Calibration assessment
            calibration_results['assessment'] = self._assess_calibration_quality(
                calibration_results['calibration_metrics']
            )

            ece = calibration_results.get('calibration_metrics', {}).get('expected_calibration_error')
            ece_str = f"{ece:.4f}" if isinstance(ece, (int, float, np.floating)) else (str(ece) if ece is not None else 'N/A')
            self.logger.info(f"✅ Model calibration assessment completed - ECE: {ece_str}")
            return calibration_results

        except Exception as e:
            self.logger.error(f"❌ Model calibration assessment failed: {e}")
            return {'error': str(e)}

    def robustness_to_noise(self, model: Any, X: np.ndarray, y: np.ndarray,
                          noise_levels: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate model robustness to input noise.

        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            noise_levels: List of noise levels to test

        Returns:
            Robustness analysis results
        """
        try:
            self.logger.info("🔊 Evaluating robustness to noise")

            if noise_levels is None:
                noise_levels = [0.01, 0.05, 0.1, 0.2]

            robustness_results = {
                'noise_analysis': {},
                'baseline_performance': {},
                'robustness_metrics': {},
                'recommendations': []
            }

            # Get baseline performance
            baseline_pred = model.predict(X)
            baseline_metrics = self._calculate_basic_metrics(y, baseline_pred)
            robustness_results['baseline_performance'] = baseline_metrics

            # Test different noise levels
            noise_analysis = {}
            for noise_level in noise_levels:
                try:
                    # Add noise to features
                    noise = np.random.normal(0, noise_level, X.shape)
                    X_noisy = X + noise * X.std(axis=0)

                    # Make predictions on noisy data
                    noisy_pred = model.predict(X_noisy)
                    noisy_metrics = self._calculate_basic_metrics(y, noisy_pred)

                    # Calculate robustness metrics
                    robustness = {}
                    for metric_name, baseline_value in baseline_metrics.items():
                        if metric_name in noisy_metrics:
                            noisy_value = noisy_metrics[metric_name]
                            robustness[metric_name] = {
                                'baseline': baseline_value,
                                'noisy': noisy_value,
                                'degradation': baseline_value - noisy_value,
                                'relative_degradation': safe_divide(baseline_value - noisy_value, baseline_value)
                            }

                    noise_analysis[str(noise_level)] = {
                        'noise_level': noise_level,
                        'metrics': noisy_metrics,
                        'robustness': robustness
                    }

                except Exception as noise_e:
                    self.logger.warning(f"⚠️ Noise level {noise_level} analysis failed: {noise_e}")
                    continue

            robustness_results['noise_analysis'] = noise_analysis

            # Calculate overall robustness metrics
            robustness_results['robustness_metrics'] = self._calculate_overall_robustness(
                robustness_results['baseline_performance'], noise_analysis
            )

            # Generate recommendations
            robustness_results['recommendations'] = self._generate_robustness_recommendations(
                robustness_results
            )

            self.logger.info(f"✅ Robustness analysis completed - "
                           f"{len(noise_analysis)} noise levels tested")
            return robustness_results

        except Exception as e:
            self.logger.error(f"❌ Robustness analysis failed: {e}")
            return {'error': str(e)}

    def _evaluate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray],
                                       class_names: Optional[List[str]]) -> Dict[str, Any]:
        """Evaluate comprehensive classification metrics."""
        try:
            metrics = {'basic_metrics': {}, 'detailed_metrics': {}}

            if not SKLEARN_AVAILABLE:
                return metrics

            # Basic metrics
            metrics['basic_metrics'] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
                'precision_macro': precision_score(y_true, y_pred, average='macro'),
                'recall_macro': recall_score(y_true, y_pred, average='macro')
            }

            # Detailed metrics
            if self.enable_detailed_metrics:
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                metrics['detailed_metrics']['confusion_matrix'] = cm.tolist()

                # Classification report
                report = classification_report(y_true, y_pred, output_dict=True)
                metrics['detailed_metrics']['classification_report'] = report

                # ROC AUC if probabilities available
                if y_prob is not None:
                    if y_prob.shape[1] == 2:
                        # Binary classification
                        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                        metrics['detailed_metrics']['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': roc_auc_score(y_true, y_prob[:, 1])
                        }

                        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
                        metrics['detailed_metrics']['pr_curve'] = {
                            'precision': precision_curve.tolist(),
                            'recall': recall_curve.tolist(),
                            'average_precision': average_precision_score(y_true, y_prob[:, 1])
                        }
                    else:
                        # Multi-class
                        metrics['detailed_metrics']['roc_auc_ovr'] = roc_auc_score(
                            y_true, y_prob, multi_class='ovr'
                        )

            return metrics

        except Exception as e:
            self.logger.warning(f"Classification metrics evaluation failed: {e}")
            return {'basic_metrics': {}, 'detailed_metrics': {}}

    def _evaluate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Evaluate comprehensive regression metrics."""
        try:
            metrics = {'basic_metrics': {}, 'detailed_metrics': {}}

            if not SKLEARN_AVAILABLE:
                return metrics

            # Basic metrics
            metrics['basic_metrics'] = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }

            # Additional regression metrics
            if self.enable_detailed_metrics:
                try:
                    metrics['detailed_metrics']['mape'] = mean_absolute_percentage_error(y_true, y_pred)
                except:
                    metrics['detailed_metrics']['mape'] = None

                # Distribution analysis
                residuals = y_true - y_pred
                metrics['detailed_metrics']['residual_analysis'] = {
                    'mean_residual': float(np.mean(residuals)),
                    'std_residual': float(np.std(residuals)),
                    'skewness': float(self._calculate_skewness(residuals)),
                    'kurtosis': float(self._calculate_kurtosis(residuals))
                }

            return metrics

        except Exception as e:
            self.logger.warning(f"Regression metrics evaluation failed: {e}")
            return {'basic_metrics': {}, 'detailed_metrics': {}}

    def _analyze_data_characteristics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    y_prob: Optional[np.ndarray], task_type: str) -> Dict[str, Any]:
        """Analyze data characteristics for evaluation context."""
        try:
            characteristics = {
                'n_samples': len(y_true),
                'n_features': y_pred.shape[1] if len(y_pred.shape) > 1 else 1,
                'task_type': task_type
            }

            if task_type == 'classification':
                unique_classes, class_counts = np.unique(y_true, return_counts=True)
                characteristics.update({
                    'n_classes': len(unique_classes),
                    'class_distribution': dict(zip(unique_classes.tolist(), class_counts.tolist())),
                    'is_binary': len(unique_classes) == 2,
                    'min_class_samples': int(np.min(class_counts)),
                    'max_class_samples': int(np.max(class_counts))
                })

                if y_prob is not None:
                    characteristics['probability_shape'] = y_prob.shape

            return characteristics

        except Exception as e:
            return {'error': str(e)}

    def _analyze_prediction_confidence(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction confidence patterns."""
        try:
            confidence_scores = np.max(y_prob, axis=1)
            correct_predictions = y_pred == y_true

            # Confidence vs accuracy analysis
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2

            bin_accuracies = []
            bin_confidences = []

            for i in range(len(confidence_bins) - 1):
                mask = (confidence_scores >= confidence_bins[i]) & (confidence_scores < confidence_bins[i + 1])
                if np.sum(mask) > 0:
                    bin_accuracy = np.mean(correct_predictions[mask])
                    bin_confidence = np.mean(confidence_scores[mask])
                    bin_accuracies.append(bin_accuracy)
                    bin_confidences.append(bin_confidence)
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(bin_centers[i])

            return {
                'confidence_accuracy_correlation': float(np.corrcoef(confidence_scores, correct_predictions.astype(int))[0, 1]),
                'confidence_bins': {
                    'centers': bin_centers.tolist(),
                    'accuracies': bin_accuracies,
                    'mean_confidences': bin_confidences
                },
                'confidence_calibration_assessment': self._assess_confidence_calibration(
                    bin_centers, bin_accuracies
                )
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_stability_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate prediction stability metrics."""
        try:
            stability = {}

            # Prediction variance
            if len(y_pred) > 1:
                if y_pred.dtype in [np.float32, np.float64]:
                    stability['prediction_variance'] = float(np.var(y_pred))
                    stability['prediction_std'] = float(np.std(y_pred))
                else:
                    # For classification, calculate prediction distribution stability
                    unique_preds, counts = np.unique(y_pred, return_counts=True)
                    stability['prediction_entropy'] = -sum((count/len(y_pred)) * safe_log(count/len(y_pred))
                                                          for count in counts if count > 0)

            # Error stability
            if y_true.dtype in [np.float32, np.float64]:
                errors = y_true - y_pred
                stability['error_stability'] = {
                    'mean_error': float(np.mean(errors)),
                    'error_std': float(np.std(errors)),
                    'error_range': float(np.max(errors) - np.min(errors))
                }

            return stability

        except Exception as e:
            return {'error': str(e)}

    def _group_predictions_by_time(self, predictions_over_time: List[Dict[str, Any]],
                                 time_window: int) -> List[Dict[str, Any]]:
        """Group predictions by time windows."""
        try:
            if not predictions_over_time:
                return []

            # Sort by timestamp
            sorted_predictions = sorted(predictions_over_time,
                                      key=lambda x: x.get('timestamp', datetime.now()))

            # Group into time windows
            windows = []
            current_window = []
            window_start_time = None

            for prediction in sorted_predictions:
                timestamp = prediction.get('timestamp', datetime.now())

                if window_start_time is None:
                    window_start_time = timestamp
                    current_window = [prediction]
                elif (timestamp - window_start_time).days >= time_window:
                    # Create new window
                    if current_window:
                        windows.append({
                            'window_start': window_start_time,
                            'window_end': current_window[-1].get('timestamp', timestamp),
                            'predictions': current_window,
                            'n_predictions': len(current_window)
                        })

                    window_start_time = timestamp
                    current_window = [prediction]
                else:
                    current_window.append(prediction)

            # Add final window
            if current_window:
                windows.append({
                    'window_start': window_start_time,
                    'window_end': current_window[-1].get('timestamp', datetime.now()),
                    'predictions': current_window,
                    'n_predictions': len(current_window)
                })

            return windows

        except Exception as e:
            self.logger.warning(f"Time window grouping failed: {e}")
            return []

    def _calculate_performance_trends(self, windowed_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance trends across time windows."""
        try:
            trends = {}

            if len(windowed_predictions) < 2:
                return trends

            # Extract performance metrics from each window
            window_metrics = []
            for window in windowed_predictions:
                if 'predictions' in window and window['predictions']:
                    # Aggregate metrics across predictions in window
                    accuracies = []
                    for pred in window['predictions']:
                        if 'metrics' in pred and 'accuracy' in pred['metrics']:
                            accuracies.append(pred['metrics']['accuracy'])

                    if accuracies:
                        window_metrics.append({
                            'window_idx': len(window_metrics),
                            'mean_accuracy': np.mean(accuracies),
                            'std_accuracy': np.std(accuracies),
                            'n_predictions': len(accuracies)
                        })

            if len(window_metrics) >= 2:
                accuracies = [w['mean_accuracy'] for w in window_metrics]

                # Calculate trend metrics
                trends = {
                    'accuracy_trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining',
                    'accuracy_change': float(accuracies[-1] - accuracies[0]),
                    'relative_accuracy_change': safe_divide(accuracies[-1] - accuracies[0], accuracies[0]),
                    'accuracy_volatility': float(np.std(accuracies)),
                    'best_window': int(np.argmax(accuracies)),
                    'worst_window': int(np.argmin(accuracies))
                }

            return trends

        except Exception as e:
            return {'error': str(e)}

    def _calculate_temporal_stability_metrics(self, windowed_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate temporal stability metrics."""
        try:
            stability = {}

            if len(windowed_predictions) < 2:
                return stability

            # Extract metrics from windows
            window_performances = []
            for window in windowed_predictions:
                if 'predictions' in window:
                    accuracies = []
                    for pred in window['predictions']:
                        if 'metrics' in pred and 'accuracy' in pred['metrics']:
                            accuracies.append(pred['metrics']['accuracy'])

                    if accuracies:
                        window_performances.append(np.mean(accuracies))

            if len(window_performances) >= 2:
                stability = {
                    'performance_stability': 1.0 - safe_divide(np.std(window_performances),
                                                              np.mean(window_performances)),
                    'performance_range': float(np.max(window_performances) - np.min(window_performances)),
                    'consistency_score': float(1.0 - np.var(window_performances)),
                    'trend_stability': self._assess_trend_stability(window_performances)
                }

            return stability

        except Exception as e:
            return {'error': str(e)}

    def _detect_performance_drift(self, windowed_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect performance drift across time windows."""
        try:
            drift_detection = {'drift_detected': False, 'drift_severity': 'none'}

            if len(windowed_predictions) < 3:
                return drift_detection

            # Simple drift detection based on performance changes
            window_performances = []
            for window in windowed_predictions:
                if 'predictions' in window:
                    accuracies = []
                    for pred in window['predictions']:
                        if 'metrics' in pred and 'accuracy' in pred['metrics']:
                            accuracies.append(pred['metrics']['accuracy'])

                    if accuracies:
                        window_performances.append(np.mean(accuracies))

            if len(window_performances) >= 3:
                # Calculate performance differences
                diffs = np.diff(window_performances)
                mean_diff = np.mean(np.abs(diffs))

                # Detect significant drift
                threshold = np.std(window_performances) * 2
                if mean_diff > threshold:
                    drift_detection['drift_detected'] = True
                    drift_detection['drift_severity'] = 'severe' if mean_diff > threshold * 2 else 'moderate'

                drift_detection.update({
                    'mean_performance_change': float(mean_diff),
                    'max_performance_change': float(np.max(np.abs(diffs))),
                    'drift_threshold': float(threshold)
                })

            return drift_detection

        except Exception as e:
            return {'error': str(e)}

    def _analyze_protected_attribute(self, predictions: np.ndarray, protected_attr: np.ndarray,
                                   true_values: Optional[np.ndarray], attr_name: str) -> Dict[str, Any]:
        """Analyze fairness for a single protected attribute."""
        try:
            analysis = {'attribute_name': attr_name}

            unique_groups = np.unique(protected_attr)

            if len(unique_groups) < 2:
                analysis['error'] = 'Insufficient groups for fairness analysis'
                return analysis

            group_metrics = {}
            for group in unique_groups:
                mask = protected_attr == group
                group_pred = predictions[mask]

                if true_values is not None:
                    group_true = true_values[mask]
                    accuracy = accuracy_score(group_true, group_pred)
                    group_metrics[str(group)] = {
                        'accuracy': accuracy,
                        'sample_size': int(np.sum(mask))
                    }
                else:
                    group_metrics[str(group)] = {
                        'sample_size': int(np.sum(mask)),
                        'prediction_distribution': dict(zip(*np.unique(group_pred, return_counts=True)))
                    }

            analysis['group_metrics'] = group_metrics

            # Calculate disparity metrics
            if true_values is not None:
                accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
                analysis['disparity'] = {
                    'accuracy_range': float(np.max(accuracies) - np.min(accuracies)),
                    'accuracy_ratio': safe_divide(np.min(accuracies), np.max(accuracies)),
                    'is_fair': np.max(accuracies) - np.min(accuracies) < 0.1  # 10% threshold
                }

            return analysis

        except Exception as e:
            return {'error': str(e)}

    def _calculate_disparity_metrics(self, attribute_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall disparity metrics across protected attributes."""
        try:
            disparities = {}

            for attr_name, analysis in attribute_analysis.items():
                if 'disparity' in analysis:
                    disparities[attr_name] = analysis['disparity']

            if disparities:
                all_ranges = [d['accuracy_range'] for d in disparities.values()]
                all_ratios = [d['accuracy_ratio'] for d in disparities.values()]

                disparities['overall'] = {
                    'max_disparity': float(np.max(all_ranges)),
                    'mean_disparity': float(np.mean(all_ranges)),
                    'min_fairness_ratio': float(np.min(all_ratios)),
                    'n_unfair_attributes': sum(1 for d in disparities.values() if not d.get('is_fair', True))
                }

            return disparities

        except Exception as e:
            return {'error': str(e)}

    def _assess_overall_fairness(self, disparity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall fairness based on disparity metrics."""
        try:
            if 'overall' not in disparity_metrics:
                return {'assessment': 'unable_to_assess'}

            overall = disparity_metrics['overall']

            if overall['max_disparity'] < 0.05:  # Less than 5% disparity
                fairness_level = 'excellent'
            elif overall['max_disparity'] < 0.1:  # Less than 10% disparity
                fairness_level = 'good'
            elif overall['max_disparity'] < 0.2:  # Less than 20% disparity
                fairness_level = 'fair'
            else:
                fairness_level = 'poor'

            return {
                'fairness_level': fairness_level,
                'max_disparity': overall['max_disparity'],
                'n_unfair_attributes': overall['n_unfair_attributes'],
                'recommendations_needed': fairness_level in ['fair', 'poor']
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate basic classification/regression metrics."""
        try:
            if len(np.unique(y_true)) <= 10 and not np.issubdtype(y_true.dtype, np.floating):
                # Classification
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'f1_macro': f1_score(y_true, y_pred, average='macro')
                }
            else:
                # Regression
                return {
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                }
        except Exception:
            return {'error': 'Metrics calculation failed'}

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        try:
            if len(data) < 3:
                return 0.0
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 3)
        except:
            return 0.0

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        try:
            if len(data) < 4:
                return 0.0
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                return 0.0
            return np.mean(((data - mean_val) / std_val) ** 4) - 3
        except:
            return 0.0

    def _assess_confidence_calibration(self, bin_centers: np.ndarray,
                                     bin_accuracies: List[float]) -> Dict[str, Any]:
        """Assess confidence calibration quality."""
        try:
            calibration_errors = np.abs(bin_centers - np.array(bin_accuracies))
            mean_calibration_error = np.mean(calibration_errors)

            if mean_calibration_error < 0.05:
                calibration_quality = 'excellent'
            elif mean_calibration_error < 0.1:
                calibration_quality = 'good'
            elif mean_calibration_error < 0.2:
                calibration_quality = 'fair'
            else:
                calibration_quality = 'poor'

            return {
                'calibration_quality': calibration_quality,
                'mean_calibration_error': float(mean_calibration_error),
                'max_calibration_error': float(np.max(calibration_errors))
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_confidence_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    confidence_scores: np.ndarray) -> Dict[str, Any]:
        """Calculate confidence-related metrics."""
        try:
            correct_predictions = y_pred == y_true

            return {
                'confidence_correct': float(np.mean(confidence_scores[correct_predictions])),
                'confidence_incorrect': float(np.mean(confidence_scores[~correct_predictions])),
                'confidence_separation': float(np.mean(confidence_scores[correct_predictions]) -
                                             np.mean(confidence_scores[~correct_predictions]))
            }

        except Exception as e:
            return {'error': str(e)}

    def _assess_calibration_quality(self, calibration_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess calibration quality based on metrics."""
        try:
            ece = calibration_metrics.get('expected_calibration_error', 1.0)

            if ece < 0.05:
                quality = 'well_calibrated'
                recommendation = 'Model is well calibrated'
            elif ece < 0.1:
                quality = 'moderately_calibrated'
                recommendation = 'Consider calibration techniques like Platt scaling'
            elif ece < 0.2:
                quality = 'poorly_calibrated'
                recommendation = 'Strong calibration needed - consider isotonic regression'
            else:
                quality = 'severely_miscalibrated'
                recommendation = 'Major calibration issues - retraining may be necessary'

            return {
                'calibration_quality': quality,
                'recommendation': recommendation,
                'expected_calibration_error': ece
            }

        except Exception as e:
            return {'error': str(e)}

    def _calculate_overall_robustness(self, baseline_metrics: Dict[str, Any],
                                    noise_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall robustness metrics."""
        try:
            robustness_scores = {}

            for metric_name, baseline_value in baseline_metrics.items():
                if isinstance(baseline_value, (int, float)):
                    noise_degradations = []

                    for noise_result in noise_analysis.values():
                        if 'robustness' in noise_result and metric_name in noise_result['robustness']:
                            degradation = noise_result['robustness'][metric_name]['relative_degradation']
                            if not np.isnan(degradation):
                                noise_degradations.append(abs(degradation))

                    if noise_degradations:
                        robustness_scores[metric_name] = {
                            'mean_degradation': float(np.mean(noise_degradations)),
                            'max_degradation': float(np.max(noise_degradations)),
                            'robustness_score': 1.0 - float(np.mean(noise_degradations))
                        }

            return robustness_scores

        except Exception as e:
            return {'error': str(e)}

    def _generate_stability_recommendations(self, stability_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stability analysis."""
        recommendations = []

        trends = stability_results.get('performance_trends', {})
        drift = stability_results.get('drift_detection', {})

        if trends.get('accuracy_trend') == 'declining':
            recommendations.append("Performance declining over time - investigate concept drift or data changes")

        if drift.get('drift_detected', False):
            severity = drift.get('drift_severity', 'moderate')
            recommendations.append(f"Performance drift detected ({severity}) - consider model retraining")

        stability_metrics = stability_results.get('stability_metrics', {})
        if stability_metrics.get('performance_stability', 1.0) < 0.8:
            recommendations.append("Low performance stability - consider ensemble methods or regularization")

        if not recommendations:
            recommendations.append("✅ Temporal performance stability is acceptable")

        return recommendations

    def _generate_fairness_recommendations(self, fairness_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on fairness analysis."""
        recommendations = []

        assessment = fairness_results.get('fairness_assessment', {})
        disparity = fairness_results.get('disparity_metrics', {})

        if assessment.get('fairness_level') in ['poor', 'fair']:
            recommendations.append("Significant fairness issues detected - consider fairness-aware algorithms")

        if disparity.get('overall', {}).get('n_unfair_attributes', 0) > 0:
            recommendations.append("Multiple protected attributes show bias - comprehensive fairness analysis needed")

        if assessment.get('recommendations_needed', False):
            recommendations.extend([
                "Implement fairness constraints during model training",
                "Consider bias mitigation techniques (reweighting, adversarial debiasing)",
                "Regular fairness audits and monitoring"
            ])

        if not recommendations:
            recommendations.append("✅ Fairness analysis shows acceptable bias levels")

        return recommendations

    def _generate_robustness_recommendations(self, robustness_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on robustness analysis."""
        recommendations = []

        robustness_metrics = robustness_results.get('robustness_metrics', {})

        for metric_name, metrics in robustness_metrics.items():
            if metrics.get('robustness_score', 1.0) < 0.8:
                recommendations.append(f"Low robustness in {metric_name} - consider data augmentation or regularization")

        noise_analysis = robustness_results.get('noise_analysis', {})
        if len(noise_analysis) > 1:
            # Check if performance degrades significantly with noise
            high_noise_levels = [k for k in noise_analysis.keys() if float(k) > 0.1]
            if high_noise_levels:
                recommendations.append("Model sensitive to high noise levels - consider denoising techniques")

        if not recommendations:
            recommendations.append("✅ Model shows good robustness to input noise")

        return recommendations
