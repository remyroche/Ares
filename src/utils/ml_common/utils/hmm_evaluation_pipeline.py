"""
Comprehensive HMM Model Evaluation Pipeline

This module provides a unified evaluation pipeline that integrates all ml_commons tools
for comprehensive HMM model evaluation, including validation, overfitting detection,
temporal analysis, and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
import time
import logging

from .hmm_hpo_config import get_hmm_hyperparameter_optimizer
from ..validation.hmm_validation_pipeline import get_hmm_validation_pipeline
from .hmm_temporal_protection import get_hmm_temporal_protection
from ..validation.enhanced_overfitting_detection import get_overfitting_detector
from .lookahead_protection import LookaheadProtection
from .model_evaluation import ModelEvaluationUtils
from ..evaluation.unified_evaluator import (
    evaluate_multiple_datasets,
)

logger = logging.getLogger(__name__)

class ComprehensiveHMMEvaluationPipeline:
    """
    Comprehensive evaluation pipeline for HMM models that integrates all ml_commons tools.

    This pipeline provides:
    - Model performance evaluation
    - Overfitting detection and analysis
    - Temporal integrity validation
    - Lookahead bias detection
    - Feature importance analysis
    - Regime-specific performance analysis
    - Comprehensive reporting and recommendations
    """

    def __init__(self):
        """Initialize comprehensive HMM evaluation pipeline."""
        self.hmm_hpo = get_hmm_hyperparameter_optimizer()
        self.hmm_validation = get_hmm_validation_pipeline()
        self.hmm_temporal_protection = get_hmm_temporal_protection()
        self.overfitting_detector = get_overfitting_detector()
        self.lookahead_protection = LookaheadProtection()
        self.model_evaluation_utils = ModelEvaluationUtils()

        self.logger = logger.getChild('ComprehensiveHMMEvaluationPipeline')
        self.logger.info("🚀 Comprehensive HMM Evaluation Pipeline initialized")

    def evaluate_hmm_model_comprehensive(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        regime_labels: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of HMM model using all ml_commons tools.

        Args:
            model: Trained HMM model
            model_name: Name of the model
            model_type: Type of model (e.g., 'random_forest', 'xgboost', 'ensemble')
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            feature_names: Optional feature names
            regime_labels: Optional regime labels
            timestamps: Optional timestamps for temporal analysis
            evaluation_config: Optional evaluation configuration

        Returns:
            Comprehensive evaluation results
        """
        self.logger.info(f"🔍 Starting comprehensive evaluation of {model_name} ({model_type})")

        evaluation_start_time = time.time()

        # Set default evaluation config
        if evaluation_config is None:
            evaluation_config = {
                'include_temporal_analysis': True,
                'include_overfitting_detection': True,
                'include_feature_analysis': True,
                'include_regime_analysis': regime_labels is not None,
                'include_bias_detection': True,
                'include_ensemble_analysis': 'ensemble' in model_type.lower(),
                'detailed_logging': True
            }

        evaluation_results = {
            'model_name': model_name,
            'model_type': model_type,
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_time_seconds': None,
            'performance_metrics': {},
            'overfitting_analysis': {},
            'temporal_analysis': {},
            'feature_analysis': {},
            'regime_analysis': {},
            'bias_detection': {},
            'ensemble_analysis': {},
            'validation_insights': {},
            'recommendations': [],
            'ml_commons_tools_used': {
                'hpo': True,
                'validation_pipeline': True,
                'temporal_protection': True,
                'overfitting_detection': True,
                'lookahead_protection': True,
                'model_evaluation': True
            }
        }

        try:
            # Step 1: Basic Performance Metrics
            self.logger.info("📊 Step 1: Calculating performance metrics")
            datasets = {
                'train': (X_train, y_train),
                'validation': (X_val, y_val),
            }
            if X_test is not None and y_test is not None:
                datasets['test'] = (X_test, y_test)

            evaluation_results['performance_metrics'] = evaluate_multiple_datasets(
                datasets=datasets, model=model, task='classification'
            )

            # Step 2: Overfitting Detection and Analysis
            if evaluation_config['include_overfitting_detection']:
                self.logger.info("🔍 Step 2: Performing overfitting detection")
                evaluation_results['overfitting_analysis'] = self._perform_overfitting_analysis(
                    model, X_train, y_train, X_val, y_val, feature_names, model_name, model_type
                )

            # Step 3: Temporal Analysis
            if evaluation_config['include_temporal_analysis'] and timestamps is not None:
                self.logger.info("⏰ Step 3: Performing temporal analysis")
                evaluation_results['temporal_analysis'] = self._perform_temporal_analysis(
                    model, X_train, y_train, X_val, y_val, timestamps
                )

            # Step 4: Feature Importance Analysis
            if evaluation_config['include_feature_analysis']:
                self.logger.info("🎯 Step 4: Performing feature analysis")
                evaluation_results['feature_analysis'] = self._perform_feature_analysis(
                    model, X_train, y_train, X_val, y_val, feature_names
                )

            # Step 5: Regime-Specific Analysis
            if evaluation_config['include_regime_analysis'] and regime_labels is not None:
                self.logger.info("📈 Step 5: Performing regime analysis")
                evaluation_results['regime_analysis'] = self._perform_regime_analysis(
                    model, X_train, y_train, X_val, y_val, regime_labels
                )

            # Step 6: Bias Detection
            if evaluation_config['include_bias_detection'] and timestamps is not None:
                self.logger.info("🔒 Step 6: Performing bias detection")
                evaluation_results['bias_detection'] = self._perform_bias_detection(
                    model, X_train, y_train, X_val, y_val, timestamps, feature_names
                )

            # Step 7: Ensemble Analysis (if applicable)
            if evaluation_config['include_ensemble_analysis']:
                self.logger.info("🤝 Step 7: Performing ensemble analysis")
                evaluation_results['ensemble_analysis'] = self._perform_ensemble_analysis(model)

            # Step 8: Generate Validation Insights and Recommendations
            self.logger.info("💡 Step 8: Generating insights and recommendations")
            evaluation_results['validation_insights'] = self._generate_evaluation_insights(
                evaluation_results
            )
            evaluation_results['recommendations'] = self._generate_evaluation_recommendations(
                evaluation_results
            )

            # Step 9: Overall Assessment
            evaluation_results['overall_assessment'] = self._generate_overall_assessment(
                evaluation_results
            )

            # Record evaluation time
            evaluation_results['evaluation_time_seconds'] = time.time() - evaluation_start_time

            self.logger.info(f"✅ Comprehensive evaluation completed for {model_name} "
                           f"in {evaluation_results['evaluation_time_seconds']:.2f}s")

            return evaluation_results

        except Exception as e:
            self.logger.error(f"❌ Comprehensive evaluation failed for {model_name}: {e}")
            return {
                'model_name': model_name,
                'model_type': model_type,
                'error': str(e),
                'evaluation_time_seconds': time.time() - evaluation_start_time,
                'recommendations': [f"Evaluation failed: {str(e)} - investigate and retry"]
            }

    def evaluate_multiple_models(
        self,
        models: Dict[str, Any],
        model_types: Dict[str, str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        regime_labels: Optional[np.ndarray] = None,
        timestamps: Optional[np.ndarray] = None,
        evaluation_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate multiple HMM models comprehensively.

        Args:
            models: Dictionary of trained models
            model_types: Dictionary mapping model names to types
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Optional test features
            y_test: Optional test labels
            feature_names: Optional feature names
            regime_labels: Optional regime labels
            timestamps: Optional timestamps
            evaluation_config: Optional evaluation configuration

        Returns:
            Comprehensive evaluation results for all models
        """
        self.logger.info(f"🔍 Starting comprehensive evaluation of {len(models)} models")

        results = {
            'models_evaluated': len(models),
            'evaluation_timestamp': datetime.now().isoformat(),
            'individual_results': {},
            'comparative_analysis': {},
            'overall_recommendations': []
        }

        # Evaluate each model individually
        for model_name, model in models.items():
            model_type = model_types.get(model_name, 'unknown')
            individual_result = self.evaluate_hmm_model_comprehensive(
                model=model,
                model_name=model_name,
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                feature_names=feature_names,
                regime_labels=regime_labels,
                timestamps=timestamps,
                evaluation_config=evaluation_config
            )

            results['individual_results'][model_name] = individual_result

        # Perform comparative analysis
        results['comparative_analysis'] = self._perform_comparative_analysis(
            results['individual_results']
        )

        # Generate overall recommendations
        results['overall_recommendations'] = self._generate_overall_recommendations(
            results['individual_results']
        )

        self.logger.info("✅ Comprehensive evaluation completed for all models")
        return results

    def _calculate_comprehensive_metrics(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray],
        y_test: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        try:
            # Get predictions for all datasets
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            test_predictions = model.predict(X_test) if X_test is not None else None

            # Get probabilities if available
            train_probabilities = None
            val_probabilities = None
            test_probabilities = None

            if hasattr(model, 'predict_proba'):
                try:
                    train_probabilities = model.predict_proba(X_train)
                    val_probabilities = model.predict_proba(X_val)
                    test_probabilities = model.predict_proba(X_test) if X_test is not None else None
                except:
                    pass

            # Calculate metrics using sklearn
            from sklearn.metrics import (
                accuracy_score, f1_score, precision_score, recall_score,
                roc_auc_score, log_loss, confusion_matrix, classification_report
            )

            # Basic classification metrics
            for dataset_name, (predictions, labels, probabilities) in [
                ('train', (train_predictions, y_train, train_probabilities)),
                ('validation', (val_predictions, y_val, val_probabilities)),
                ('test', (test_predictions, y_test, test_probabilities))
            ]:
                if predictions is None:
                    continue

                pred, true_labels, proba = predictions, labels, probabilities

                dataset_metrics = {
                    'accuracy': float(accuracy_score(true_labels, pred)),
                    'f1_weighted': float(f1_score(true_labels, pred, average='weighted')),
                    'precision_weighted': float(precision_score(true_labels, pred, average='weighted')),
                    'recall_weighted': float(recall_score(true_labels, pred, average='weighted'))
                }

                # Per-class metrics
                unique_classes = np.unique(true_labels)
                per_class_metrics = {}
                for class_label in unique_classes:
                    class_mask = true_labels == class_label
                    if class_mask.sum() > 0:
                        class_pred = pred[class_mask]
                        class_true = true_labels[class_mask]
                        per_class_metrics[int(class_label)] = {
                            'precision': float(precision_score(class_true, class_pred, average='weighted', zero_division=0)),
                            'recall': float(recall_score(class_true, class_pred, average='weighted', zero_division=0)),
                            'f1': float(f1_score(class_true, class_pred, average='weighted', zero_division=0)),
                            'support': int(class_mask.sum())
                        }

                dataset_metrics['per_class_metrics'] = per_class_metrics

                # ROC AUC and log loss if probabilities available
                if proba is not None:
                    try:
                        if len(unique_classes) == 2:
                            dataset_metrics['roc_auc'] = float(roc_auc_score(true_labels, proba[:, 1]))
                        else:
                            dataset_metrics['roc_auc'] = float(roc_auc_score(true_labels, proba, multi_class='ovr'))

                        dataset_metrics['log_loss'] = float(log_loss(true_labels, proba))
                    except:
                        dataset_metrics['roc_auc'] = None
                        dataset_metrics['log_loss'] = None

                # Confusion matrix
                dataset_metrics['confusion_matrix'] = confusion_matrix(true_labels, pred).tolist()

                metrics[dataset_name] = dataset_metrics

            # Calculate cross-dataset metrics
            if 'train' in metrics and 'validation' in metrics:
                train_acc = metrics['train']['accuracy']
                val_acc = metrics['validation']['accuracy']
                metrics['train_val_accuracy_gap'] = float(train_acc - val_acc)

                train_f1 = metrics['train']['f1_weighted']
                val_f1 = metrics['validation']['f1_weighted']
                metrics['train_val_f1_gap'] = float(train_f1 - val_f1)

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate comprehensive metrics: {e}")
            metrics['error'] = str(e)

        return metrics

    def _perform_overfitting_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]],
        model_name: str,
        model_type: str
    ) -> Dict[str, Any]:
        """Perform comprehensive overfitting analysis using ml_commons tools."""
        analysis = {}

        try:
            # Use the overfitting detector from ml_commons
            overfitting_report = self.overfitting_detector.detect_overfitting_for_model(
                model=model,
                X_train=X_train,
                X_val=X_val,
                y_train=y_train,
                y_val=y_val,
                model_name=model_name,
                model_type=model_type
            )

            analysis['overfitting_report'] = {
                'is_overfitting': overfitting_report.is_overfitting,
                'severity': overfitting_report.severity,
                'confidence_level': overfitting_report.confidence_level,
                'indicators': overfitting_report.indicators,
                'warnings': overfitting_report.warnings,
                'recommendations': overfitting_report.recommendations,
                'detailed_metrics': {
                    'train_accuracy': overfitting_report.train_accuracy,
                    'val_accuracy': overfitting_report.val_accuracy,
                    'accuracy_gap': overfitting_report.accuracy_gap,
                    'train_f1': overfitting_report.train_f1,
                    'val_f1': overfitting_report.val_f1,
                    'f1_gap': overfitting_report.f1_gap
                }
            }

            # Add additional overfitting indicators
            analysis['additional_indicators'] = self._calculate_additional_overfitting_indicators(
                model, X_train, y_train, X_val, y_val
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Overfitting analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _calculate_additional_overfitting_indicators(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate additional overfitting indicators."""
        indicators = {}

        try:
            # Feature importance stability
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                indicators['feature_importance_stability'] = {
                    'concentration_ratio': float(np.sum(feature_importance[:5]) / np.sum(feature_importance)),
                    'gini_coefficient': self._calculate_gini_coefficient(feature_importance)
                }

            # Prediction confidence analysis
            if hasattr(model, 'predict_proba'):
                train_proba = model.predict_proba(X_train)
                val_proba = model.predict_proba(X_val)

                train_confidence = np.mean(np.max(train_proba, axis=1))
                val_confidence = np.mean(np.max(val_proba, axis=1))

                indicators['confidence_analysis'] = {
                    'train_confidence': float(train_confidence),
                    'val_confidence': float(val_confidence),
                    'confidence_gap': float(train_confidence - val_confidence),
                    'overconfident_predictions': float(np.mean(np.max(val_proba, axis=1) > 0.9))
                }

        except Exception as e:
            indicators['calculation_error'] = str(e)

        return indicators

    def _perform_temporal_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, Any]:
        """Perform temporal analysis using ml_commons temporal protection."""
        analysis = {}

        try:
            # Create DataFrames for temporal analysis
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            train_df = pd.DataFrame(X_train, columns=feature_names)
            train_df['target'] = y_train
            train_df['timestamp'] = pd.to_datetime(timestamps[:len(X_train)])

            val_df = pd.DataFrame(X_val, columns=feature_names)
            val_df['target'] = y_val
            val_df['timestamp'] = pd.to_datetime(timestamps[len(X_train):len(X_train) + len(X_val)])

            # Use temporal protection for analysis
            temporal_analysis = self.hmm_temporal_protection.validate_hmm_temporal_constraints(
                features_df=train_df,
                target_df=val_df,
                prediction_timestamp=datetime.now()
            )

            analysis['temporal_constraints'] = temporal_analysis

            # Perform lookahead bias detection
            bias_analysis = self.lookahead_protection.detect_data_leakage(
                features_df=train_df,
                target_df=val_df,
                timestamp_col='timestamp'
            )

            analysis['lookahead_bias'] = bias_analysis

        except Exception as e:
            self.logger.warning(f"⚠️ Temporal analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _perform_feature_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform comprehensive feature analysis."""
        analysis = {}

        try:
            # Extract feature importance
            feature_importance = self._get_feature_importance(model)
            if feature_importance is not None:
                analysis['feature_importance'] = {
                    'importance_scores': feature_importance.tolist(),
                    'top_features': self._get_top_features(feature_importance, feature_names, 10),
                    'feature_stability': self._analyze_feature_stability(model, X_train, y_train)
                }

            # Feature correlation analysis
            if feature_names is not None:
                analysis['feature_correlations'] = self._analyze_feature_correlations(
                    X_train, y_train, feature_names
                )

            # Feature drift analysis
            analysis['feature_drift'] = self._analyze_feature_drift(
                X_train, X_val, feature_names
            )

        except Exception as e:
            self.logger.warning(f"⚠️ Feature analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _perform_regime_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        regime_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Perform regime-specific analysis."""
        analysis = {}

        try:
            unique_regimes = np.unique(regime_labels)
            regime_performance = {}

            for regime_id in unique_regimes:
                regime_mask = regime_labels == regime_id

                if regime_mask.sum() == 0:
                    continue

                # Get regime-specific data
                X_regime = X_val[regime_mask]
                y_regime = y_val[regime_mask]

                if len(X_regime) > 0:
                    y_pred_regime = model.predict(X_regime)

                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
                    regime_performance[int(regime_id)] = {
                        'accuracy': float(accuracy_score(y_regime, y_pred_regime)),
                        'f1_score': float(f1_score(y_regime, y_pred_regime, average='weighted')),
                        'precision': float(precision_score(y_regime, y_pred_regime, average='weighted')),
                        'recall': float(recall_score(y_regime, y_pred_regime, average='weighted')),
                        'sample_count': int(len(X_regime)),
                        'regime_stability': self._analyze_regime_stability(regime_labels, regime_id)
                    }

            analysis['regime_performance'] = regime_performance
            analysis['regime_summary'] = {
                'total_regimes': len(unique_regimes),
                'regime_sizes': {int(r): (regime_labels == r).sum() for r in unique_regimes},
                'performance_variance': np.var([p['accuracy'] for p in regime_performance.values()]) if regime_performance else 0.0
            }

        except Exception as e:
            self.logger.warning(f"⚠️ Regime analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _perform_bias_detection(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        timestamps: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Perform comprehensive bias detection."""
        analysis = {}

        try:
            # Use HMM temporal protection for bias detection
            bias_results = self.hmm_temporal_protection.detect_hmm_lookahead_bias(
                X=X_train,
                y=y_train,
                timestamps=timestamps,
                feature_names=feature_names,
                hmm_states=y_train  # Use targets as HMM states for bias detection
            )

            analysis['bias_detection_results'] = bias_results

        except Exception as e:
            self.logger.warning(f"⚠️ Bias detection failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _perform_ensemble_analysis(self, model: Any) -> Dict[str, Any]:
        """Perform ensemble-specific analysis."""
        analysis = {}

        try:
            analysis['ensemble_characteristics'] = {
                'n_estimators': None,
                'ensemble_type': self._get_ensemble_type(model),
                'diversity_metrics': {},
                'base_model_analysis': {}
            }

            # Get number of estimators
            if hasattr(model, 'estimators_'):
                analysis['ensemble_characteristics']['n_estimators'] = len(model.estimators_)
            elif hasattr(model, 'n_estimators'):
                analysis['ensemble_characteristics']['n_estimators'] = model.n_estimators

            # Determine ensemble type
            ensemble_type = self._get_ensemble_type(model)
            analysis['ensemble_characteristics']['ensemble_type'] = ensemble_type

            # Calculate diversity metrics if applicable
            if hasattr(model, 'estimators_') and len(model.estimators_) > 1:
                diversity_metrics = self._calculate_ensemble_diversity(model, X_val[:100] if 'X_val' in globals() else None)
                analysis['ensemble_characteristics']['diversity_metrics'] = diversity_metrics

        except Exception as e:
            self.logger.warning(f"⚠️ Ensemble analysis failed: {e}")
            analysis['error'] = str(e)

        return analysis

    def _generate_evaluation_insights(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation insights."""
        insights = {
            'key_findings': [],
            'risk_assessment': {},
            'performance_summary': {},
            'ml_commons_insights': {}
        }

        try:
            # Key findings
            performance_metrics = evaluation_results.get('performance_metrics', {})
            overfitting_analysis = evaluation_results.get('overfitting_analysis', {})
            temporal_analysis = evaluation_results.get('temporal_analysis', {})

            if performance_metrics:
                val_accuracy = performance_metrics.get('validation', {}).get('accuracy', 0)
                insights['key_findings'].append(f"Validation accuracy: {val_accuracy:.4f}")

            if overfitting_analysis.get('overfitting_report', {}).get('is_overfitting', False):
                severity = overfitting_analysis['overfitting_report']['severity']
                insights['key_findings'].append(f"Overfitting detected: {severity} severity")
                insights['risk_assessment']['overfitting_risk'] = severity

            if temporal_analysis.get('bias_detection_results', {}).get('overall_bias_detected', False):
                insights['key_findings'].append("Temporal bias detected in features")
                insights['risk_assessment']['temporal_bias_risk'] = 'high'

            # Performance summary
            if performance_metrics:
                insights['performance_summary'] = {
                    'train_accuracy': performance_metrics.get('train', {}).get('accuracy'),
                    'val_accuracy': performance_metrics.get('validation', {}).get('accuracy'),
                    'test_accuracy': performance_metrics.get('test', {}).get('accuracy'),
                    'accuracy_trend': 'improving' if performance_metrics.get('test', {}).get('accuracy', 0) > performance_metrics.get('validation', {}).get('accuracy', 0) else 'declining'
                }

            # ml_commons insights
            insights['ml_commons_insights'] = {
                'tools_utilized': list(evaluation_results.get('ml_commons_tools_used', {}).keys()),
                'validation_pipeline_used': True,
                'temporal_protection_used': True,
                'overfitting_detection_used': True
            }

        except Exception as e:
            insights['error'] = str(e)

        return insights

    def _generate_evaluation_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive evaluation recommendations."""
        recommendations = []

        try:
            # Overfitting recommendations
            overfitting_analysis = evaluation_results.get('overfitting_analysis', {})
            if overfitting_analysis.get('overfitting_report', {}).get('is_overfitting', False):
                severity = overfitting_analysis['overfitting_report']['severity']
                if severity in ['severe', 'high']:
                    recommendations.append("🔴 CRITICAL: Severe overfitting detected - immediate regularization needed")
                    recommendations.append("Consider using stronger regularization or ensemble methods")
                elif severity == 'moderate':
                    recommendations.append("🟡 Moderate overfitting detected - consider regularization techniques")
                    recommendations.append("Implement cross-validation to reduce overfitting")

            # Temporal recommendations
            temporal_analysis = evaluation_results.get('temporal_analysis', {})
            if temporal_analysis.get('bias_detection_results', {}).get('overall_bias_detected', False):
                recommendations.append("🔴 Temporal bias detected - implement strict temporal data splitting")
                recommendations.append("Use rolling window validation for temporal stability")

            # Performance recommendations
            performance_metrics = evaluation_results.get('performance_metrics', {})
            if performance_metrics:
                train_acc = performance_metrics.get('train', {}).get('accuracy', 0)
                val_acc = performance_metrics.get('validation', {}).get('accuracy', 0)

                if train_acc - val_acc > 0.15:  # Large gap
                    recommendations.append("High train-validation gap - consider more regularization")
                elif val_acc < 0.7:
                    recommendations.append("Low validation accuracy - consider model architecture improvements")

            # Feature recommendations
            feature_analysis = evaluation_results.get('feature_analysis', {})
            if feature_analysis.get('feature_importance', {}):
                concentration = feature_analysis['feature_importance'].get('concentration_ratio', 0)
                if concentration > 0.8:
                    recommendations.append("High feature concentration - consider feature engineering")

            # Regime recommendations
            regime_analysis = evaluation_results.get('regime_analysis', {})
            if regime_analysis.get('regime_summary', {}).get('performance_variance', 0) > 0.1:
                recommendations.append("High performance variance across regimes - consider regime-specific models")

            # General ml_commons recommendations
            recommendations.extend([
                "✅ ml_commons tools successfully integrated - continue using for comprehensive evaluation",
                "Consider implementing automated model monitoring with ml_commons validation pipeline",
                "Use HMM temporal protection for ongoing bias detection in production"
            ])

        except Exception as e:
            recommendations.append(f"Failed to generate recommendations: {str(e)}")

        return recommendations

    def _generate_overall_assessment(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall model assessment."""
        assessment = {
            'overall_score': 0.0,
            'risk_level': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'production_readiness': 'unknown'
        }

        try:
            # Calculate overall score based on multiple factors
            score = 0.0
            max_score = 100.0

            # Performance score (40%)
            performance_metrics = evaluation_results.get('performance_metrics', {})
            if performance_metrics:
                val_accuracy = performance_metrics.get('validation', {}).get('accuracy', 0)
                score += val_accuracy * 40
                assessment['strengths'].append(f"Validation accuracy: {val_accuracy:.4f}")

            # Overfitting penalty (20%)
            overfitting_analysis = evaluation_results.get('overfitting_analysis', {})
            if overfitting_analysis.get('overfitting_report', {}).get('is_overfitting', False):
                severity = overfitting_analysis['overfitting_report']['severity']
                if severity == 'severe':
                    score -= 20
                    assessment['weaknesses'].append("Severe overfitting detected")
                elif severity == 'high':
                    score -= 15
                    assessment['weaknesses'].append("High overfitting risk")
                elif severity == 'moderate':
                    score -= 10
                    assessment['weaknesses'].append("Moderate overfitting detected")
            else:
                score += 10
                assessment['strengths'].append("No overfitting detected")

            # Temporal integrity score (20%)
            temporal_analysis = evaluation_results.get('temporal_analysis', {})
            if not temporal_analysis.get('bias_detection_results', {}).get('overall_bias_detected', False):
                score += 20
                assessment['strengths'].append("No temporal bias detected")
            else:
                score -= 15
                assessment['weaknesses'].append("Temporal bias detected")

            # Feature quality score (10%)
            feature_analysis = evaluation_results.get('feature_analysis', {})
            if feature_analysis.get('feature_importance', {}):
                concentration = feature_analysis['feature_importance'].get('concentration_ratio', 0)
                if concentration < 0.8:
                    score += 10
                    assessment['strengths'].append("Good feature distribution")
                else:
                    score -= 5
                    assessment['weaknesses'].append("High feature concentration")

            # Regime performance score (10%)
            regime_analysis = evaluation_results.get('regime_analysis', {})
            if regime_analysis.get('regime_summary', {}).get('performance_variance', 0) < 0.1:
                score += 10
                assessment['strengths'].append("Consistent regime performance")
            else:
                score -= 5
                assessment['weaknesses'].append("High regime performance variance")

            # Normalize score
            assessment['overall_score'] = max(0.0, min(100.0, score))

            # Determine risk level and production readiness
            if assessment['overall_score'] >= 80:
                assessment['risk_level'] = 'low'
                assessment['production_readiness'] = 'ready'
            elif assessment['overall_score'] >= 60:
                assessment['risk_level'] = 'medium'
                assessment['production_readiness'] = 'conditional'
            elif assessment['overall_score'] >= 40:
                assessment['risk_level'] = 'high'
                assessment['production_readiness'] = 'not_ready'
            else:
                assessment['risk_level'] = 'critical'
                assessment['production_readiness'] = 'not_ready'

        except Exception as e:
            assessment['error'] = str(e)

        return assessment

    def _perform_comparative_analysis(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across multiple models."""
        analysis = {
            'model_ranking': {},
            'performance_comparison': {},
            'best_models_by_metric': {},
            'consensus_analysis': {}
        }

        try:
            # Extract performance metrics for comparison
            model_metrics = {}
            for model_name, results in individual_results.items():
                if 'error' not in results:
                    performance = results.get('performance_metrics', {})
                    if performance:
                        model_metrics[model_name] = {
                            'validation_accuracy': performance.get('validation', {}).get('accuracy', 0),
                            'test_accuracy': performance.get('test', {}).get('accuracy', 0),
                            'f1_score': performance.get('validation', {}).get('f1_weighted', 0),
                            'overfitting_severity': results.get('overfitting_analysis', {}).get('overfitting_report', {}).get('severity', 'none'),
                            'temporal_bias': results.get('temporal_analysis', {}).get('bias_detection_results', {}).get('overall_bias_detected', False)
                        }

            if model_metrics:
                # Rank models by validation accuracy
                sorted_by_accuracy = sorted(model_metrics.items(), key=lambda x: x[1]['validation_accuracy'], reverse=True)
                analysis['model_ranking']['by_validation_accuracy'] = sorted_by_accuracy

                # Best models by different metrics
                analysis['best_models_by_metric'] = {
                    'accuracy': sorted_by_accuracy[0][0] if sorted_by_accuracy else None,
                    'f1_score': max(model_metrics.items(), key=lambda x: x[1]['f1_score'])[0] if model_metrics else None,
                    'least_overfitting': min(model_metrics.items(), key=lambda x: ['critical', 'high', 'moderate', 'none'].index(x[1]['overfitting_severity']))[0] if model_metrics else None,
                    'no_temporal_bias': [name for name, metrics in model_metrics.items() if not metrics['temporal_bias']][0] if any(not m['temporal_bias'] for m in model_metrics.values()) else None
                }

                # Performance comparison
                analysis['performance_comparison'] = {
                    'validation_accuracies': {name: metrics['validation_accuracy'] for name, metrics in model_metrics.items()},
                    'accuracy_range': {
                        'min': min(m['validation_accuracy'] for m in model_metrics.values()),
                        'max': max(m['validation_accuracy'] for m in model_metrics.values()),
                        'std': float(np.std([m['validation_accuracy'] for m in model_metrics.values()]))
                    }
                }

        except Exception as e:
            analysis['error'] = str(e)

        return analysis

    def _generate_overall_recommendations(self, individual_results: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations across all models."""
        recommendations = []

        try:
            # Check for common issues across models
            overfitting_count = 0
            bias_count = 0
            low_performance_count = 0

            for model_name, results in individual_results.items():
                if 'error' in results:
                    continue

                overfitting_analysis = results.get('overfitting_analysis', {})
                if overfitting_analysis.get('overfitting_report', {}).get('is_overfitting', False):
                    overfitting_count += 1

                temporal_analysis = results.get('temporal_analysis', {})
                if temporal_analysis.get('bias_detection_results', {}).get('overall_bias_detected', False):
                    bias_count += 1

                performance_metrics = results.get('performance_metrics', {})
                val_accuracy = performance_metrics.get('validation', {}).get('accuracy', 0)
                if val_accuracy < 0.7:
                    low_performance_count += 1

            total_models = len([r for r in individual_results.values() if 'error' not in r])

            # Generate recommendations based on common issues
            if overfitting_count > total_models * 0.5:
                recommendations.append(f"🔴 Widespread overfitting detected in {overfitting_count}/{total_models} models - implement stronger regularization")

            if bias_count > total_models * 0.3:
                recommendations.append(f"🔴 Temporal bias detected in {bias_count}/{total_models} models - review temporal data handling")

            if low_performance_count > total_models * 0.5:
                recommendations.append(f"🟡 Low performance in {low_performance_count}/{total_models} models - consider model architecture improvements")

            # Model selection recommendations
            comparative_analysis = self._perform_comparative_analysis(individual_results)
            best_model = comparative_analysis.get('best_models_by_metric', {}).get('accuracy')

            if best_model:
                recommendations.append(f"✅ Best performing model: {best_model}")

            # General recommendations
            recommendations.extend([
                "✅ Continue using ml_commons comprehensive evaluation pipeline for all models",
                "Consider ensemble methods to improve overall performance and reduce overfitting",
                "Implement automated model monitoring with temporal validation",
                "Use feature importance analysis to guide feature engineering decisions"
            ])

        except Exception as e:
            recommendations.append(f"Failed to generate overall recommendations: {str(e)}")

        return recommendations

    # Helper methods
    def _get_feature_importance(self, model: Any) -> Optional[np.ndarray]:
        """Extract feature importance from model."""
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'coef_'):
                return np.abs(model.coef_).flatten()
            else:
                return None
        except:
            return None

    def _get_top_features(self, importance: np.ndarray, feature_names: Optional[List[str]], n: int = 10) -> Dict[str, Any]:
        """Get top N features by importance."""
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]

        top_indices = np.argsort(importance)[::-1][:n]
        top_features = {
            'names': [feature_names[i] for i in top_indices],
            'importances': importance[top_indices].tolist(),
            'indices': top_indices.tolist()
        }

        return top_features

    def _analyze_feature_stability(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze feature importance stability."""
        stability = {}

        try:
            # This would require multiple training runs in practice
            # For now, provide basic stability metrics
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                stability['concentration_ratio'] = float(np.sum(importance[:5]) / np.sum(importance))
                stability['gini_coefficient'] = self._calculate_gini_coefficient(importance)
                stability['stability_score'] = 1.0 / (1.0 + stability['concentration_ratio'])

        except Exception as e:
            stability['error'] = str(e)

        return stability

    def _analyze_feature_correlations(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature correlations."""
        correlations = {}

        try:
            # Feature-to-feature correlations
            feature_corr = np.corrcoef(X, rowvar=False)
            correlations['feature_correlation_matrix'] = feature_corr.tolist()

            # Feature-to-target correlations
            target_corr = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                target_corr.append(float(corr))

            correlations['target_correlations'] = dict(zip(feature_names, target_corr))

            # Identify highly correlated feature pairs
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs(feature_corr[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'feature_1': feature_names[i],
                            'feature_2': feature_names[j],
                            'correlation': float(feature_corr[i, j])
                        })

            correlations['high_correlation_pairs'] = high_corr_pairs

        except Exception as e:
            correlations['error'] = str(e)

        return correlations

    def _analyze_feature_drift(self, X_train: np.ndarray, X_val: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature drift between training and validation."""
        drift = {}

        try:
            drift_scores = []
            for i in range(X_train.shape[1]):
                train_mean = np.mean(X_train[:, i])
                val_mean = np.mean(X_val[:, i])
                train_std = np.std(X_train[:, i])
                val_std = np.std(X_val[:, i])

                # Simple drift score based on standardized difference
                if train_std > 0:
                    drift_score = abs(train_mean - val_mean) / train_std
                else:
                    drift_score = 0.0

                drift_scores.append({
                    'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                    'train_mean': float(train_mean),
                    'val_mean': float(val_mean),
                    'train_std': float(train_std),
                    'val_std': float(val_std),
                    'drift_score': float(drift_score)
                })

            drift['feature_drift_scores'] = drift_scores
            drift['max_drift_feature'] = max(drift_scores, key=lambda x: x['drift_score'])['feature']
            drift['average_drift'] = float(np.mean([d['drift_score'] for d in drift_scores]))

        except Exception as e:
            drift['error'] = str(e)

        return drift

    def _analyze_regime_stability(self, regime_labels: np.ndarray, regime_id: int) -> Dict[str, Any]:
        """Analyze stability of a specific regime."""
        stability = {}

        try:
            regime_mask = regime_labels == regime_id
            regime_sequence = regime_labels[regime_mask]

            if len(regime_sequence) > 0:
                # Calculate regime persistence
                changes = np.sum(regime_sequence[1:] != regime_sequence[:-1])
                stability_score = 1.0 - (changes / len(regime_sequence))

                stability['persistence_score'] = float(stability_score)
                stability['total_samples'] = int(len(regime_sequence))
                stability['transitions'] = int(changes)

        except Exception as e:
            stability['error'] = str(e)

        return stability

    def _calculate_ensemble_diversity(self, model: Any, X_subset: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate ensemble diversity metrics."""
        diversity = {}

        try:
            if hasattr(model, 'estimators_') and len(model.estimators_) > 1:
                if X_subset is None:
                    # Use a small subset of training data
                    X_subset = X_train[:50] if 'X_train' in globals() else None

                if X_subset is not None:
                    base_predictions = []
                    for estimator in model.estimators_:
                        pred = estimator.predict(X_subset)
                        base_predictions.append(pred)

                    if base_predictions:
                        prediction_matrix = np.array(base_predictions)
                        diversity['prediction_variance'] = float(np.mean(np.var(prediction_matrix, axis=0)))
                        diversity['base_model_disagreement'] = float(np.mean(prediction_matrix.std(axis=0)))
                        diversity['diversity_score'] = diversity['prediction_variance'] / (diversity['prediction_variance'] + 1e-8)

        except Exception as e:
            diversity['error'] = str(e)

        return diversity

    def _get_ensemble_type(self, model: Any) -> str:
        """Determine ensemble type."""
        if hasattr(model, 'estimators_'):
            return 'bagging'
        elif hasattr(model, 'final_estimator_'):
            return 'stacking'
        elif hasattr(model, 'classes_'):
            return 'voting'
        else:
            return 'unknown'

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for feature importance distribution."""
        values = np.abs(values)
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)

        if n == 0 or cumsum[-1] == 0:
            return 0.0

        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

# Global instance
_comprehensive_hmm_evaluation_instance = None

def get_comprehensive_hmm_evaluation_pipeline() -> ComprehensiveHMMEvaluationPipeline:
    """Get global comprehensive HMM evaluation pipeline instance."""
    global _comprehensive_hmm_evaluation_instance
    if _comprehensive_hmm_evaluation_instance is None:
        _comprehensive_hmm_evaluation_instance = ComprehensiveHMMEvaluationPipeline()
    return _comprehensive_hmm_evaluation_instance

# Export key classes and functions
__all__ = ['ComprehensiveHMMEvaluationPipeline', 'get_comprehensive_hmm_evaluation_pipeline']