"""
Evaluation Utilities

Common evaluation patterns shared across all training modules.
Uses existing math validation utilities for safe calculations.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score
)

# Use existing utilities
from src.utils.math_validation import safe_divide, safe_log, validate_finite
from src.utils.logger import system_logger
from src.utils.ml_common.evaluation.unified_evaluator import (
    compute_classification_metrics,
    compute_regression_metrics,
    evaluate_model as unified_evaluate_model,
)

# Enhanced analysis imports
try:
    from src.utils.ml_common.evaluation.enhanced_learning_curve_analysis import EnhancedLearningCurveAnalyzer, LearningCurveAnalysisResult
    ENHANCED_LEARNING_CURVE_AVAILABLE = True
except ImportError:
    ENHANCED_LEARNING_CURVE_AVAILABLE = False
    EnhancedLearningCurveAnalyzer = None
    LearningCurveAnalysisResult = None

try:
    from src.utils.ml_common.evaluation.enhanced_bootstrap_confidence_intervals import EnhancedBootstrapConfidenceIntervalAnalyzer, BootstrapAnalysisResult
    ENHANCED_BOOTSTRAP_AVAILABLE = True
except ImportError:
    ENHANCED_BOOTSTRAP_AVAILABLE = False
    EnhancedBootstrapConfidenceIntervalAnalyzer = None
    BootstrapAnalysisResult = None

logger = system_logger.getChild('EvaluationUtils')
logger.info("EvaluationUtils delegating core metric computation to unified_evaluator")

class EvaluationUtils:
    """Common evaluation utilities."""

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Calculate common metrics for model evaluation.

        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities (for classification)
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing calculated metrics
        """
        if metrics is None:
            metrics = (
                ['accuracy', 'precision', 'recall', 'f1_score']
                if is_classification
                else ['mse', 'mae', 'r2', 'mape', 'smape']
            )

        if is_classification:
            return compute_classification_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_pred_proba,
                include=metrics,
            )
        else:
            return compute_regression_metrics(
                y_true=y_true,
                y_pred=y_pred,
                include=metrics,
            )

    @staticmethod
    def evaluate_model_performance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model performance with common metrics.

        Args:
            model: Trained model
            X: Input features
            y: True target values
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing calculated metrics
        """
        task = 'classification' if is_classification else 'regression'
        return unified_evaluate_model(model=model, X=X, y=y, task=task, include=metrics)

    @staticmethod
    def evaluate_ensemble_performance(
        ensemble: Any,
        X: np.ndarray,
        y: np.ndarray,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance with common metrics.

        Args:
            ensemble: Trained ensemble model
            X: Input features
            y: True target values
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing calculated metrics
        """
        task = 'classification' if is_classification else 'regression'
        return unified_evaluate_model(model=ensemble, X=X, y=y, task=task, include=metrics)

    @staticmethod
    def evaluate_regime_performance(
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        metrics: List[str] = None,
        is_classification: bool = True
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Evaluate model performance per regime.

        Args:
            models: Dictionary of trained models
            X: Input features
            y: True target values
            regime_labels: Array of regime labels
            metrics: List of metrics to calculate
            is_classification: Whether this is a classification task

        Returns:
            Dictionary containing evaluation results per regime per model
        """
        evaluation_results = {}

        for regime in np.unique(regime_labels):
            regime_mask = regime_labels == regime
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]

            regime_evaluation = {}

            for model_name, model in models.items():
                try:
                    metrics_dict = EvaluationUtils.evaluate_model_performance(
                        model, regime_X, regime_y, metrics, is_classification
                    )
                    regime_evaluation[model_name] = metrics_dict
                except Exception as e:
                    logger.warning(f"⚠️ Failed to evaluate {model_name} for regime {regime}: {e}")
                    regime_evaluation[model_name] = {'error': str(e)}

            evaluation_results[regime] = regime_evaluation

        return evaluation_results

    @staticmethod
    def analyze_regime_distribution(
        y_train: np.ndarray,
        y_test: np.ndarray,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze regime distribution and model performance per regime.

        Args:
            y_train: Training regime labels
            y_test: Test regime labels
            results: Training results containing model performance

        Returns:
            Dictionary containing regime analysis results
        """
        # Overall regime distribution
        unique_regimes, train_counts = np.unique(y_train, return_counts=True)
        _, test_counts = np.unique(y_test, return_counts=True)

        regime_analysis = {
            'n_regimes': len(unique_regimes),
            'regime_distribution_train': dict(zip(unique_regimes, train_counts)),
            'regime_distribution_test': dict(zip(unique_regimes, test_counts)),
            'regime_balance_train': np.std(train_counts) / np.mean(train_counts) if len(train_counts) > 1 else 0.0,
            'regime_balance_test': np.std(test_counts) / np.mean(test_counts) if len(test_counts) > 1 else 0.0
        }

        # Model performance per regime
        for model_name, metrics in results.get('performance', {}).items():
            if 'confusion_matrix' in metrics:
                cm = np.array(metrics['confusion_matrix'])
                regime_precision = np.diag(cm) / np.sum(cm, axis=0)
                regime_recall = np.diag(cm) / np.sum(cm, axis=1)
                regime_f1 = 2 * (regime_precision * regime_recall) / (regime_precision + regime_recall)

                regime_analysis[f'{model_name}_regime_performance'] = {
                    'precision_per_regime': regime_precision.tolist(),
                    'recall_per_regime': regime_recall.tolist(),
                    'f1_per_regime': regime_f1.tolist()
                }

        return regime_analysis

    @staticmethod
    def get_best_model(results: Dict[str, Any], metric: str = 'accuracy') -> Optional[str]:
        """
        Get the best model based on a specific metric.

        Args:
            results: Training results containing model performance
            metric: Metric to use for comparison

        Returns:
            Name of the best model, or None if not found
        """
        best_model = None
        best_score = -np.inf

        for model_name, metrics in results.get('performance', {}).items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model = model_name

        return best_model

# Export functions for direct import compatibility
calculate_metrics = EvaluationUtils.calculate_metrics
evaluate_model_performance = EvaluationUtils.evaluate_model_performance
evaluate_ensemble_performance = EvaluationUtils.evaluate_ensemble_performance
evaluate_regime_performance = EvaluationUtils.evaluate_regime_performance
analyze_regime_distribution = EvaluationUtils.analyze_regime_distribution
get_best_model = EvaluationUtils.get_best_model

def create_evaluation_report(results: Dict[str, Any], output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive evaluation report.

    Args:
        results: Training results containing model performance
        output_path: Optional path to save the report

    Returns:
        Dictionary containing the evaluation report
    """
    report = {
        'summary': {},
        'detailed_metrics': {},
        'best_models': {},
        'recommendations': []
    }

    # Get performance metrics
    performance = results.get('performance', {})

    if performance:
        # Summary statistics
        report['summary'] = {
            'total_models': len(performance),
            'models_evaluated': list(performance.keys())
        }

        # Detailed metrics
        report['detailed_metrics'] = performance

        # Find best models for different metrics
        common_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'mse', 'mae', 'r2']
        for metric in common_metrics:
            best_model = get_best_model(results, metric)
            if best_model:
                report['best_models'][metric] = {
                    'model': best_model,
                    'score': performance[best_model].get(metric, 'N/A')
                }

        # Generate recommendations
        if report['best_models']:
            best_overall = max(report['best_models'].items(),
                             key=lambda x: x[1]['score'] if isinstance(x[1]['score'], (int, float)) else 0)
            report['recommendations'].append(f"Best overall model: {best_overall[1]['model']} ({best_overall[0]}: {best_overall[1]['score']:.4f})")

        # Check for potential issues
        for model_name, metrics in performance.items():
            if 'accuracy' in metrics and metrics['accuracy'] < 0.6:
                report['recommendations'].append(f"⚠️ {model_name} has low accuracy ({metrics['accuracy']:.4f}) - consider hyperparameter tuning")
            if 'f1_score' in metrics and metrics['f1_score'] < 0.5:
                report['recommendations'].append(f"⚠️ {model_name} has low F1 score ({metrics['f1_score']:.4f}) - may indicate class imbalance issues")

    # Save report if path provided
    if output_path:
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"✅ Evaluation report saved to {output_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to save evaluation report: {e}")

    return report

    def analyze_learning_curves(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        train_sizes: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        scoring: str = 'accuracy'
    ) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive learning curve analysis using enhanced analyzer.

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
            Dictionary with learning curve analysis results or None if not available
        """
        if not ENHANCED_LEARNING_CURVE_AVAILABLE:
            logger.warning("⚠️ Enhanced learning curve analysis not available")
            return None

        try:
            analyzer = EnhancedLearningCurveAnalyzer(random_state=42, n_jobs=-1)
            result = analyzer.analyze_learning_curve(
                model, X_train, y_train, X_test, y_test, train_sizes, cv_folds, scoring
            )

            # Convert dataclass to dictionary for compatibility
            if isinstance(result, LearningCurveAnalysisResult):
                return {
                    'learning_rate': result.learning_rate,
                    'convergence_stability': result.convergence_stability,
                    'overfitting_risk': result.overfitting_risk,
                    'training_efficiency': result.training_efficiency,
                    'max_score_gap': result.max_score_gap,
                    'final_score_gap': result.final_score_gap,
                    'early_learning_slope': result.early_learning_slope,
                    'convergence_stability_score': result.convergence_stability_score,
                    'train_sizes': result.train_sizes,
                    'train_scores_mean': result.train_scores_mean,
                    'train_scores_std': result.train_scores_std,
                    'val_scores_mean': result.val_scores_mean,
                    'val_scores_std': result.val_scores_std,
                    'score_gaps': result.score_gaps,
                    'final_train_score': result.final_train_score,
                    'final_validation_score': result.final_validation_score,
                    'test_score': result.test_score,
                    'anomalies': result.anomalies,
                    'recommendations': result.recommendations
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Learning curve analysis failed: {e}")
            return None

    def analyze_bootstrap_confidence_intervals(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        scoring_metrics: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform bootstrap confidence interval analysis using enhanced analyzer.

        Args:
            model: Trained model instance
            X: Feature matrix
            y: Target labels
            train_size: Fraction of data to use for training in each bootstrap
            scoring_metrics: List of metrics to evaluate

        Returns:
            Dictionary with bootstrap analysis results or None if not available
        """
        if not ENHANCED_BOOTSTRAP_AVAILABLE:
            logger.warning("⚠️ Enhanced bootstrap analysis not available")
            return None

        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'f1', 'precision', 'recall']

        try:
            analyzer = EnhancedBootstrapConfidenceIntervalAnalyzer(
                n_bootstrap=100,  # Reduced from 1000 to 100 for efficiency
                confidence_level=0.95,
                n_jobs=-1
            )

            result = analyzer.analyze_model_stability(model, X, y, train_size, scoring_metrics)

            # Convert dataclass to dictionary for compatibility
            if isinstance(result, BootstrapAnalysisResult):
                return {
                    'stability_score': result.stability_score,
                    'stability_level': result.stability_level,
                    'overfitting_probability': result.overfitting_probability,
                    'overfitting_risk': result.overfitting_risk,
                    'confidence_intervals': result.confidence_intervals,
                    'stability_scores': result.stability_scores,
                    'n_successful_bootstrap': result.n_successful_bootstrap,
                    'recommendations': result.recommendations
                }
            else:
                return result

        except Exception as e:
            logger.error(f"Bootstrap confidence interval analysis failed: {e}")
            return None

    def comprehensive_enhanced_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_full: Optional[np.ndarray] = None,
        y_full: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive enhanced analysis combining all available tools.

        Args:
            model: Trained model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            X_full: Full feature matrix (for bootstrap analysis)
            y_full: Full target labels (for bootstrap analysis)

        Returns:
            Dictionary with comprehensive enhanced analysis results
        """
        results = {
            'enhanced_analysis_available': ENHANCED_LEARNING_CURVE_AVAILABLE or ENHANCED_BOOTSTRAP_AVAILABLE,
            'learning_curve_analysis': None,
            'bootstrap_analysis': None,
            'combined_recommendations': []
        }

        # Perform learning curve analysis
        if ENHANCED_LEARNING_CURVE_AVAILABLE:
            learning_curve_results = self.analyze_learning_curves(
                model, X_train, y_train, X_test, y_test
            )
            if learning_curve_results:
                results['learning_curve_analysis'] = learning_curve_results
                results['combined_recommendations'].extend(learning_curve_results.get('recommendations', []))

        # Perform bootstrap confidence interval analysis
        if ENHANCED_BOOTSTRAP_AVAILABLE and X_full is not None and y_full is not None:
            bootstrap_results = self.analyze_bootstrap_confidence_intervals(
                model, X_full, y_full
            )
            if bootstrap_results:
                results['bootstrap_analysis'] = bootstrap_results
                results['combined_recommendations'].extend(bootstrap_results.get('recommendations', []))

        # Remove duplicate recommendations
        results['combined_recommendations'] = list(set(results['combined_recommendations']))

        return results
