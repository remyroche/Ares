"""
HMM-Specific Validation Pipeline

This module provides comprehensive validation for HMM models using ml_commons tools,
reducing code duplication in market_analysis/ training files.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

from .enhanced_overfitting_detection import (
    get_overfitting_detector,
    UniversalOverfittingDetector,
    OverfittingConfig,
    detect_overfitting_for_model
)
from ..utils.lookahead_protection import LookaheadProtection
from ..evaluation.unified_evaluator import evaluate_multiple_datasets
from ..config.base_training_config import HMMTrainingConfig

logger = logging.getLogger(__name__)

class HMMValidationPipeline:
    """
    Comprehensive validation pipeline specifically designed for HMM model training.
    Integrates overfitting detection, lookahead protection, and temporal validation.
    """

    def __init__(self, config: Optional[HMMTrainingConfig] = None):
        """
        Initialize HMM validation pipeline.

        Args:
            config: HMM training configuration
        """
        self.config = config or HMMTrainingConfig()
        self.overfitting_detector = get_overfitting_detector()
        self.lookahead_protection = LookaheadProtection()
        # Deprecated evaluator removed; using unified evaluator where needed
        self.logger = logger.getChild('HMMValidationPipeline')

    def validate_hmm_training_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
        current_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of HMM training data.

        Args:
            X: Input features
            y: Target values (HMM states)
            regime_labels: Regime labels for stratification
            feature_names: Names of input features
            timestamps: Optional timestamps for temporal validation
            current_timestamp: Current timestamp for lookahead protection

        Returns:
            Comprehensive validation results
        """
        self.logger.info("🔍 Starting comprehensive HMM training data validation")

        validation_results = {
            'valid': True,
            'data_quality': {},
            'temporal_integrity': {},
            'regime_analysis': {},
            'feature_analysis': {},
            'lookahead_bias': {},
            'recommendations': []
        }

        try:
            # 1. Basic data quality validation
            validation_results['data_quality'] = self._validate_data_quality(
                X, y, regime_labels, feature_names
            )

            # 2. Temporal integrity validation
            if timestamps is not None:
                validation_results['temporal_integrity'] = self._validate_temporal_integrity(
                    timestamps, current_timestamp
                )

                # Set current timestamp for lookahead protection
                if current_timestamp is not None:
                    self.lookahead_protection.set_current_timestamp(current_timestamp)

            # 3. Regime analysis
            validation_results['regime_analysis'] = self._analyze_regime_structure(
                regime_labels, X, y
            )

            # 4. Feature analysis
            if feature_names is not None:
                validation_results['feature_analysis'] = self._analyze_feature_quality(
                    X, feature_names, y
                )

            # 5. Lookahead bias detection
            if timestamps is not None:
                validation_results['lookahead_bias'] = self._detect_lookahead_bias(
                    X, y, timestamps, feature_names
                )

            # 6. Aggregate validation status
            validation_results['valid'] = self._aggregate_validation_status(validation_results)

            # 7. Generate recommendations
            validation_results['recommendations'] = self._generate_validation_recommendations(
                validation_results
            )

            self.logger.info(f"✅ HMM training data validation completed - {'Valid' if validation_results['valid'] else 'Invalid'}")

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ HMM training data validation failed: {e}")
            validation_results['valid'] = False
            validation_results['error'] = str(e)
            return validation_results

    def validate_hmm_model_performance(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime_labels: Optional[np.ndarray] = None,
        feature_importance: Optional[np.ndarray] = None,
        fold_number: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of HMM model performance including overfitting detection.

        Args:
            model: Trained HMM model
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'xgboost')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            regime_labels: Optional regime labels
            feature_importance: Optional feature importance scores
            fold_number: Optional fold number for cross-validation

        Returns:
            Comprehensive model performance validation
        """
        self.logger.info(f"🔍 Validating HMM model performance: {model_name} ({model_type})")

        validation_results = {
            'model_name': model_name,
            'model_type': model_type,
            'performance_metrics': {},
            'overfitting_analysis': {},
            'regime_performance': {},
            'feature_analysis': {},
            'recommendations': []
        }

        try:
            # 1. Basic performance metrics
            validation_results['performance_metrics'] = self._calculate_comprehensive_metrics(
                model, X_train, y_train, X_val, y_val
            )

            # 2. Overfitting detection using ml_commons
            validation_results['overfitting_analysis'] = self._detect_overfitting_comprehensive(
                model, X_train, y_train, X_val, y_val, feature_importance, model_name, model_type, fold_number
            )

            # 3. Regime-specific performance analysis
            if regime_labels is not None:
                validation_results['regime_performance'] = self._analyze_regime_performance(
                    model, X_train, y_train, X_val, y_val, regime_labels
                )

            # 4. Feature importance analysis
            if feature_importance is not None:
                validation_results['feature_analysis'] = self._analyze_feature_importance(
                    feature_importance, model_name
                )

            # 5. Generate recommendations
            validation_results['recommendations'] = self._generate_model_recommendations(
                validation_results
            )

            self.logger.info(f"✅ HMM model performance validation completed: {model_name}")

            return validation_results

        except Exception as e:
            self.logger.error(f"❌ HMM model performance validation failed: {e}")
            return {
                'model_name': model_name,
                'model_type': model_type,
                'error': str(e),
                'recommendations': ["Model validation failed - investigate error"]
            }

    def validate_hmm_temporal_integrity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        prediction_timestamp: Optional[datetime] = None,
        lookback_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Validate temporal integrity for HMM predictions.

        Args:
            X: Feature matrix
            y: Target values
            timestamps: Timestamps for data points
            prediction_timestamp: Timestamp when prediction is made
            lookback_window: Maximum allowed lookback window

        Returns:
            Temporal integrity validation results
        """
        self.logger.info("⏰ Validating HMM temporal integrity")

        # Use lookahead protection for temporal validation
        if prediction_timestamp is None:
            prediction_timestamp = datetime.now()

        # Create temporary DataFrame for validation
        temp_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        temp_df['target'] = y
        temp_df['timestamp'] = pd.to_datetime(timestamps)

        # Perform temporal feature validation
        temporal_results = self.lookahead_protection.temporal_feature_validation(
            feature_data=temp_df,
            prediction_timestamp=prediction_timestamp,
            lookback_window=lookback_window
        )

        # Add additional HMM-specific temporal checks
        hmm_temporal_checks = self._perform_hmm_temporal_checks(temp_df, prediction_timestamp)

        return {
            'temporal_validation': temporal_results,
            'hmm_specific_checks': hmm_temporal_checks,
            'overall_temporal_integrity': (
                temporal_results.get('is_valid', False) and
                hmm_temporal_checks.get('is_valid', False)
            )
        }

    def _validate_data_quality(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Validate basic data quality for HMM training."""
        quality = {
            'shape': {'X': X.shape, 'y': y.shape, 'regime_labels': regime_labels.shape},
            'missing_values': {
                'X': np.isnan(X).sum(),
                'y': np.isnan(y).sum(),
                'regime_labels': np.isnan(regime_labels).sum()
            },
            'data_types': {
                'X': X.dtype,
                'y': y.dtype,
                'regime_labels': regime_labels.dtype
            },
            'class_distribution': None,
            'feature_statistics': {}
        }

        # Class distribution
        if len(y) > 0:
            unique_classes, counts = np.unique(y, return_counts=True)
            quality['class_distribution'] = dict(zip(unique_classes, counts))

        # Feature statistics
        if feature_names is not None and len(feature_names) > 0:
            for i, feature_name in enumerate(feature_names):
                if i < X.shape[1]:
                    quality['feature_statistics'][feature_name] = {
                        'mean': float(np.mean(X[:, i])),
                        'std': float(np.std(X[:, i])),
                        'min': float(np.min(X[:, i])),
                        'max': float(np.max(X[:, i])),
                        'missing': int(np.isnan(X[:, i]).sum())
                    }

        return quality

    def _validate_temporal_integrity(
        self,
        timestamps: np.ndarray,
        current_timestamp: Optional[datetime]
    ) -> Dict[str, Any]:
        """Validate temporal integrity of data."""
        integrity = {
            'timestamp_range': {
                'min': pd.to_datetime(timestamps).min(),
                'max': pd.to_datetime(timestamps).max()
            },
            'timestamp_ordering': np.all(timestamps[:-1] <= timestamps[1:]),
            'future_data_present': False,
            'temporal_gaps': []
        }

        if current_timestamp is not None:
            future_mask = pd.to_datetime(timestamps) > current_timestamp
            integrity['future_data_present'] = future_mask.any()

        # Detect temporal gaps
        if len(timestamps) > 1:
            time_diffs = np.diff(pd.to_datetime(timestamps))
            large_gaps = time_diffs > np.median(time_diffs) * 3
            if large_gaps.any():
                integrity['temporal_gaps'] = time_diffs[large_gaps].tolist()

        return integrity

    def _analyze_regime_structure(
        self,
        regime_labels: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze regime structure for HMM training."""
        regime_analysis = {
            'n_regimes': len(np.unique(regime_labels)),
            'regime_sizes': {},
            'regime_quality': {},
            'regime_separation': {}
        }

        unique_regimes = np.unique(regime_labels)

        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id
            regime_size = regime_mask.sum()

            regime_analysis['regime_sizes'][int(regime_id)] = int(regime_size)

            # Calculate regime quality metrics
            if regime_size > 0:
                regime_X = X[regime_mask]
                regime_y = y[regime_mask]

                # Class distribution within regime
                unique_classes, counts = np.unique(regime_y, return_counts=True)
                regime_analysis['regime_quality'][int(regime_id)] = {
                    'size': int(regime_size),
                    'class_distribution': dict(zip(unique_classes, counts)),
                    'feature_variance': float(np.mean(np.var(regime_X, axis=0))),
                    'min_samples_per_class': int(np.min(counts)) if len(counts) > 0 else 0
                }

        return regime_analysis

    def _analyze_feature_quality(
        self,
        X: np.ndarray,
        feature_names: List[str],
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze feature quality for HMM training."""
        feature_analysis = {
            'feature_importance': {},
            'correlation_analysis': {},
            'redundancy_check': {},
            'temporal_stability': {}
        }

        # Calculate basic statistics for each feature
        for i, feature_name in enumerate(feature_names):
            if i < X.shape[1]:
                feature_data = X[:, i]
                feature_analysis['feature_importance'][feature_name] = {
                    'variance': float(np.var(feature_data)),
                    'missing_percentage': float(np.isnan(feature_data).sum() / len(feature_data)),
                    'correlation_with_target': float(np.corrcoef(feature_data, y)[0, 1])
                }

        return feature_analysis

    def _detect_lookahead_bias(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Detect lookahead bias using ml_commons tools."""
        # Create temporary DataFrame
        temp_df = pd.DataFrame(X, columns=feature_names or [f'feature_{i}' for i in range(X.shape[1])])
        temp_df['target'] = y
        temp_df['timestamp'] = pd.to_datetime(timestamps)

        # Use lookahead protection
        bias_results = self.lookahead_protection.detect_data_leakage(
            features_df=temp_df,
            target_df=temp_df,  # Use same data for features and targets
            timestamp_col='timestamp'
        )

        return bias_results

    def _calculate_comprehensive_metrics(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

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
                except:
                    pass

            # Calculate metrics using sklearn
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                roc_auc_score, log_loss, confusion_matrix
            )

            # Basic classification metrics
            metrics['accuracy'] = {
                'train': float(accuracy_score(y_train, train_predictions)),
                'val': float(accuracy_score(y_val, val_predictions))
            }

            # F1 scores
            metrics['f1_score'] = {
                'train': float(f1_score(y_train, train_predictions, average='weighted')),
                'val': float(f1_score(y_val, val_predictions, average='weighted'))
            }

            # Precision and recall
            metrics['precision'] = {
                'train': float(precision_score(y_train, train_predictions, average='weighted')),
                'val': float(precision_score(y_val, val_predictions, average='weighted'))
            }

            metrics['recall'] = {
                'train': float(recall_score(y_train, train_predictions, average='weighted')),
                'val': float(recall_score(y_val, val_predictions, average='weighted'))
            }

            # ROC AUC if probabilities available
            if train_probabilities is not None and val_probabilities is not None:
                try:
                    if len(np.unique(y_train)) == 2:  # Binary classification
                        metrics['roc_auc'] = {
                            'train': float(roc_auc_score(y_train, train_probabilities[:, 1])),
                            'val': float(roc_auc_score(y_val, val_probabilities[:, 1]))
                        }
                    else:  # Multi-class
                        metrics['roc_auc'] = {
                            'train': float(roc_auc_score(y_train, train_probabilities, multi_class='ovr')),
                            'val': float(roc_auc_score(y_val, val_probabilities, multi_class='ovr'))
                        }
                except:
                    metrics['roc_auc'] = {'train': None, 'val': None}

            # Log loss if probabilities available
            if train_probabilities is not None and val_probabilities is not None:
                try:
                    metrics['log_loss'] = {
                        'train': float(log_loss(y_train, train_probabilities)),
                        'val': float(log_loss(y_val, val_probabilities))
                    }
                except:
                    metrics['log_loss'] = {'train': None, 'val': None}

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate comprehensive metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def _detect_overfitting_comprehensive(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_importance: Optional[np.ndarray],
        model_name: str,
        model_type: str,
        fold_number: Optional[int]
    ) -> Dict[str, Any]:
        """Comprehensive overfitting detection using ml_commons tools."""
        # Use the ml_commons overfitting detector
        overfitting_report = detect_overfitting_for_model(
            model=model,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val,
            model_name=model_name,
            model_type=model_type,
            fold_number=fold_number
        )

        return {
            'overfitting_detected': overfitting_report.is_overfitting,
            'severity': overfitting_report.severity,
            'confidence_level': overfitting_report.confidence_level,
            'indicators': overfitting_report.indicators,
            'warnings': overfitting_report.warnings,
            'recommendations': overfitting_report.recommendations,
            'detailed_report': {
                'train_accuracy': overfitting_report.train_accuracy,
                'val_accuracy': overfitting_report.val_accuracy,
                'accuracy_gap': overfitting_report.accuracy_gap,
                'train_f1': overfitting_report.train_f1,
                'val_f1': overfitting_report.val_f1,
                'f1_gap': overfitting_report.f1_gap
            }
        }

    def _analyze_regime_performance(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze model performance across different regimes."""
        regime_performance = {}
        unique_regimes = np.unique(regime_labels)

        for regime_id in unique_regimes:
            regime_mask = regime_labels == regime_id

            # Skip if no samples in this regime
            if not regime_mask.any():
                continue

            # Get regime-specific data
            X_regime = X_val[regime_mask]
            y_regime = y_val[regime_mask]

            if len(X_regime) == 0:
                continue

            # Calculate regime-specific metrics
            y_pred_regime = model.predict(X_regime)

            from sklearn.metrics import accuracy_score, f1_score
            regime_performance[int(regime_id)] = {
                'accuracy': float(accuracy_score(y_regime, y_pred_regime)),
                'f1_score': float(f1_score(y_regime, y_pred_regime, average='weighted')),
                'sample_count': int(len(X_regime))
            }

        return regime_performance

    def _analyze_feature_importance(
        self,
        feature_importance: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Analyze feature importance for model interpretability."""
        feature_analysis = {
            'top_features': {},
            'concentration_analysis': {},
            'stability_metrics': {}
        }

        # Get top features
        if len(feature_importance) > 0:
            sorted_indices = np.argsort(feature_importance)[::-1]
            top_n = min(10, len(feature_importance))

            feature_analysis['top_features'] = {
                'indices': sorted_indices[:top_n].tolist(),
                'importances': feature_importance[sorted_indices[:top_n]].tolist()
            }

            # Feature concentration analysis
            total_importance = np.sum(feature_importance)
            if total_importance > 0:
                concentration = np.sum(feature_importance[sorted_indices[:top_n]]) / total_importance
                feature_analysis['concentration_analysis'] = {
                    'top_10_percent_concentration': float(concentration),
                    'gini_coefficient': self._calculate_gini_coefficient(feature_importance)
                }

        return feature_analysis

    def _perform_hmm_temporal_checks(
        self,
        temp_df: pd.DataFrame,
        prediction_timestamp: datetime
    ) -> Dict[str, Any]:
        """Perform HMM-specific temporal checks."""
        hmm_checks = {
            'is_valid': True,
            'checks': [],
            'recommendations': []
        }

        # Check for reasonable HMM state transitions
        if 'target' in temp_df.columns:
            # Calculate state transition matrix
            states = temp_df['target'].values
            if len(states) > 1:
                transition_matrix = self._calculate_transition_matrix(states)
                hmm_checks['checks'].append({
                    'name': 'state_transition_analysis',
                    'result': 'valid' if np.sum(transition_matrix) > 0 else 'invalid'
                })

        # Check temporal consistency
        timestamps = temp_df['timestamp']
        if len(timestamps) > 1:
            time_diffs = timestamps.diff().dropna()
            if len(time_diffs) > 0:
                mean_diff = time_diffs.mean()
                std_diff = time_diffs.std()

                hmm_checks['checks'].append({
                    'name': 'temporal_consistency',
                    'result': 'valid' if std_diff < mean_diff * 2 else 'warning',
                    'details': {
                        'mean_time_diff': str(mean_diff),
                        'std_time_diff': str(std_diff)
                    }
                })

        # Aggregate results
        invalid_checks = [check for check in hmm_checks['checks'] if check['result'] == 'invalid']
        hmm_checks['is_valid'] = len(invalid_checks) == 0

        return hmm_checks

    def _aggregate_validation_status(self, validation_results: Dict[str, Any]) -> bool:
        """Aggregate individual validation checks into overall status."""
        # Data quality checks
        if validation_results['data_quality'].get('missing_values', {}).get('X', 0) > 0:
            return False

        if validation_results['data_quality'].get('missing_values', {}).get('y', 0) > 0:
            return False

        # Temporal integrity checks
        if validation_results['temporal_integrity'].get('future_data_present', False):
            return False

        if not validation_results['temporal_integrity'].get('timestamp_ordering', True):
            return False

        # Regime analysis checks
        regime_quality = validation_results['regime_analysis'].get('regime_quality', {})
        for regime_id, quality in regime_quality.items():
            if quality.get('min_samples_per_class', 0) < 10:  # Minimum samples per class
                return False

        # Lookahead bias checks
        if validation_results['lookahead_bias'].get('leakage_detected', False):
            return False

        return True

    def _generate_validation_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Data quality recommendations
        data_quality = validation_results['data_quality']
        if data_quality.get('missing_values', {}).get('X', 0) > 0:
            recommendations.append("Handle missing values in features before training")

        if data_quality.get('missing_values', {}).get('y', 0) > 0:
            recommendations.append("Handle missing values in target variable")

        # Temporal recommendations
        temporal_integrity = validation_results['temporal_integrity']
        if temporal_integrity.get('future_data_present', False):
            recommendations.append("Remove or handle future data points in training set")

        # Regime recommendations
        regime_analysis = validation_results['regime_analysis']
        min_samples_per_class = min([
            quality.get('min_samples_per_class', 0)
            for quality in regime_analysis.get('regime_quality', {}).values()
        ])

        if min_samples_per_class < 50:
            recommendations.append(f"Ensure minimum 50 samples per class per regime (currently {min_samples_per_class})")

        # Lookahead bias recommendations
        lookahead_bias = validation_results['lookahead_bias']
        if lookahead_bias.get('leakage_detected', False):
            recommendations.append("Address detected lookahead bias issues")
            recommendations.extend(lookahead_bias.get('recommendations', []))

        return recommendations

    def _generate_model_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate model-specific recommendations."""
        recommendations = []

        # Overfitting recommendations
        overfitting_analysis = validation_results['overfitting_analysis']
        if overfitting_analysis.get('overfitting_detected', False):
            recommendations.extend(overfitting_analysis.get('recommendations', []))

        # Performance recommendations
        performance_metrics = validation_results['performance_metrics']
        if 'accuracy' in performance_metrics:
            train_acc = performance_metrics['accuracy'].get('train', 0)
            val_acc = performance_metrics['accuracy'].get('val', 0)
            accuracy_gap = abs(train_acc - val_acc)

            if accuracy_gap > 0.1:  # 10% gap
                recommendations.append("Large accuracy gap detected - consider regularization")

        # Regime performance recommendations
        regime_performance = validation_results['regime_performance']
        if regime_performance:
            regime_accuracies = [regime['accuracy'] for regime in regime_performance.values()]
            if len(regime_accuracies) > 1:
                accuracy_variance = np.var(regime_accuracies)
                if accuracy_variance > 0.05:  # High variance across regimes
                    recommendations.append("High performance variance across regimes - consider regime-specific models")

        return recommendations

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for feature importance distribution."""
        values = np.abs(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)

        if n == 0 or cumsum[-1] == 0:
            return 0.0

        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

    def _calculate_transition_matrix(self, states: np.ndarray) -> np.ndarray:
        """Calculate state transition matrix for HMM analysis."""
        unique_states = np.unique(states)
        n_states = len(unique_states)

        # Create transition matrix
        transition_matrix = np.zeros((n_states, n_states))

        for i in range(len(states) - 1):
            current_state = np.where(unique_states == states[i])[0][0]
            next_state = np.where(unique_states == states[i + 1])[0][0]
            transition_matrix[current_state, next_state] += 1

        # Normalize
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums

        return transition_matrix

# Global instance
_hmm_validation_instance = None

def get_hmm_validation_pipeline(config: Optional[HMMTrainingConfig] = None) -> HMMValidationPipeline:
    """Get global HMM validation pipeline instance."""
    global _hmm_validation_instance
    if _hmm_validation_instance is None:
        _hmm_validation_instance = HMMValidationPipeline(config)
    return _hmm_validation_instance

# Export key classes and functions
__all__ = ['HMMValidationPipeline', 'get_hmm_validation_pipeline']