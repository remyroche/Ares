"""Performance evaluator component for tactician specialist training."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger

class PerformanceEvaluator:
    """Handles performance evaluation for specialist models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the performance evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('performance_evaluation', {})
        self.logger = system_logger.getChild('performance_evaluator')
        self.metrics_to_calculate = self.config.get('metrics', ['accuracy', 'precision', 'recall', 'f1_score', 'confusion_matrix', 'classification_report'])
        self.evaluation_window = self.config.get('evaluation_window', 1000)
        self.sliding_window_step = self.config.get('sliding_window_step', 100)
        self.performance_thresholds = self.config.get('thresholds', {'min_precision': 0.85, 'min_recall': 0.1, 'min_f1': 0.2, 'min_accuracy': 0.6})

    @handles_errors(exceptions=(Exception,), default_return={}, context='specialist evaluation')
    async def evaluate_all_specialists(self, all_models: Dict[str, Dict[str, Any]], test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate all specialist models.
        
        Args:
            all_models: Dictionary of specialist models by regime
            test_data: Test data for evaluation
            
        Returns:
            Comprehensive evaluation results
        """
        self.logger.info('Evaluating all specialist models...')
        evaluation_results = {'regime_evaluations': {}, 'tactic_comparisons': {}, 'overall_metrics': {}, 'performance_summary': {}}
        for regime_id, regime_models in all_models.items():
            self.logger.info(f'Evaluating specialists for regime {regime_id}...')
            regime_eval = await self._evaluate_regime_specialists(regime_models, test_data, regime_id)
            evaluation_results['regime_evaluations'][regime_id] = regime_eval
        evaluation_results['tactic_comparisons'] = await self._compare_tactics_across_regimes(evaluation_results['regime_evaluations'])
        evaluation_results['overall_metrics'] = self._calculate_overall_metrics(evaluation_results['regime_evaluations'])
        evaluation_results['performance_summary'] = self._create_performance_summary(evaluation_results)
        return evaluation_results

    async def _evaluate_regime_specialists(self, regime_models: Dict[str, Any], test_data: pd.DataFrame, regime_id: str) -> Dict[str, Any]:
        """Evaluate specialists for a specific regime.
        
        Args:
            regime_models: Models for the regime
            test_data: Test data
            regime_id: Regime identifier
            
        Returns:
            Regime evaluation results
        """
        regime_evaluation = {'tactics': {}, 'best_tactic': None, 'regime_metrics': {}}
        X_test, y_test = self._prepare_test_data(test_data)
        if X_test.empty or len(y_test) == 0:
            self.logger.warning(f'No test data available for regime {regime_id}')
            return regime_evaluation
        best_score = -np.inf
        best_tactic = None
        for tactic_name, tactic_models in regime_models.items():
            tactic_eval = await self._evaluate_tactic(tactic_name, tactic_models, X_test, y_test)
            regime_evaluation['tactics'][tactic_name] = tactic_eval
            tactic_score = tactic_eval.get('average_metrics', {}).get('f1_score', 0)
            if tactic_score > best_score:
                best_score = tactic_score
                best_tactic = tactic_name
        regime_evaluation['best_tactic'] = best_tactic
        regime_evaluation['regime_metrics'] = self._aggregate_tactic_metrics(regime_evaluation['tactics'])
        return regime_evaluation

    async def _evaluate_tactic(self, tactic_name: str, models: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate a specific tactic.
        
        Args:
            tactic_name: Name of the tactic
            models: Tactic models (dict or single model)
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Tactic evaluation results
        """
        tactic_evaluation = {'model_metrics': {}, 'average_metrics': {}, 'stability_metrics': {}, 'meets_thresholds': False}
        if isinstance(models, dict):
            all_metrics = []
            for model_name, model in models.items():
                metrics = await self._evaluate_single_model(model, X_test, y_test, f'{tactic_name}_{model_name}')
                tactic_evaluation['model_metrics'][model_name] = metrics
                all_metrics.append(metrics)
            if all_metrics:
                tactic_evaluation['average_metrics'] = self._average_metrics(all_metrics)
        else:
            metrics = await self._evaluate_single_model(models, X_test, y_test, tactic_name)
            tactic_evaluation['model_metrics']['single'] = metrics
            tactic_evaluation['average_metrics'] = metrics
        tactic_evaluation['stability_metrics'] = await self._calculate_stability_metrics(models, X_test, y_test)
        avg_metrics = tactic_evaluation['average_metrics']
        tactic_evaluation['meets_thresholds'] = self._check_performance_thresholds(avg_metrics)
        return tactic_evaluation

    async def _evaluate_single_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> Dict[str, float]:
        """Evaluate a single model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Model metrics
        """
        metrics = {}
        try:
            y_pred = model.predict(X_test)
            if 'accuracy' in self.metrics_to_calculate:
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
            if 'precision' in self.metrics_to_calculate:
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            if 'recall' in self.metrics_to_calculate:
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            if 'f1_score' in self.metrics_to_calculate:
                metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['n_predictions'] = len(y_pred)
            metrics['positive_rate'] = (y_pred > 0).mean() if len(y_pred) > 0 else 0
            if 'confusion_matrix' in self.metrics_to_calculate and len(np.unique(y_test)) == 2:
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    metrics['true_positive_rate'] = tp / (tp + fn) if tp + fn > 0 else 0
                    metrics['false_positive_rate'] = fp / (fp + tn) if fp + tn > 0 else 0
                    metrics['specificity'] = tn / (tn + fp) if tn + fp > 0 else 0
        except Exception as e:
            self.logger.error(f'Failed to evaluate model {model_name}: {str(e)}')
            metrics['error'] = str(e)
        return metrics

    async def _calculate_stability_metrics(self, models: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calculate stability metrics for models.
        
        Args:
            models: Model(s) to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Stability metrics
        """
        stability_metrics = {'performance_variance': 0.0, 'prediction_consistency': 0.0, 'temporal_stability': 0.0}
        if len(X_test) < self.evaluation_window:
            return stability_metrics
        window_metrics = []
        for start_idx in range(0, len(X_test) - self.evaluation_window + 1, self.sliding_window_step):
            end_idx = start_idx + self.evaluation_window
            X_window = X_test.iloc[start_idx:end_idx]
            y_window = y_test.iloc[start_idx:end_idx]
            if isinstance(models, dict):
                model = list(models.values())[0]
            else:
                model = models
            try:
                y_pred = model.predict(X_window)
                window_accuracy = accuracy_score(y_window, y_pred)
                window_metrics.append(window_accuracy)
            except:
                continue
        if window_metrics:
            stability_metrics['performance_variance'] = np.var(window_metrics)
            if len(window_metrics) > 1:
                x = np.arange(len(window_metrics))
                slope = np.polyfit(x, window_metrics, 1)[0]
                stability_metrics['temporal_stability'] = 1.0 - abs(slope)
        return stability_metrics

    def _prepare_test_data(self, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare test data for evaluation."""
        label_col = 'label'
        if label_col not in test_data.columns:
            return (pd.DataFrame(), pd.Series())
        labeled_data = test_data[test_data[label_col] != 0]
        exclude_cols = [label_col, 'regime_id', 'timestamp', 'barrier_type', 'exit_time', 'potential_profit_pct', 'signal_strength']
        feature_cols = [col for col in labeled_data.columns if col not in exclude_cols]
        X = labeled_data[feature_cols]
        y = labeled_data[label_col]
        return (X, y)

    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple evaluations."""
        if not metrics_list:
            return {}
        averaged = {}
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        for metric_name in all_metrics:
            values = []
            for metrics in metrics_list:
                if metric_name in metrics and (not isinstance(metrics[metric_name], str)):
                    values.append(metrics[metric_name])
            if values:
                averaged[metric_name] = np.mean(values)
        return averaged

    def _aggregate_tactic_metrics(self, tactics_evaluation: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across tactics."""
        aggregated = {'avg_precision': [], 'avg_recall': [], 'avg_f1': [], 'n_tactics': len(tactics_evaluation), 'n_meeting_thresholds': 0}
        for tactic_eval in tactics_evaluation.values():
            avg_metrics = tactic_eval.get('average_metrics', {})
            if 'precision' in avg_metrics:
                aggregated['avg_precision'].append(avg_metrics['precision'])
            if 'recall' in avg_metrics:
                aggregated['avg_recall'].append(avg_metrics['recall'])
            if 'f1_score' in avg_metrics:
                aggregated['avg_f1'].append(avg_metrics['f1_score'])
            if tactic_eval.get('meets_thresholds', False):
                aggregated['n_meeting_thresholds'] += 1
        for key in ['avg_precision', 'avg_recall', 'avg_f1']:
            if aggregated[key]:
                aggregated[key] = np.mean(aggregated[key])
            else:
                aggregated[key] = 0.0
        return aggregated

    async def _compare_tactics_across_regimes(self, regime_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare tactics across different regimes."""
        tactic_comparison = {}
        all_tactics = set()
        for regime_eval in regime_evaluations.values():
            all_tactics.update(regime_eval.get('tactics', {}).keys())
        for tactic in all_tactics:
            tactic_data = {'regimes': [], 'metrics': {'precision': [], 'recall': [], 'f1_score': []}}
            for regime_id, regime_eval in regime_evaluations.items():
                if tactic in regime_eval.get('tactics', {}):
                    tactic_eval = regime_eval['tactics'][tactic]
                    avg_metrics = tactic_eval.get('average_metrics', {})
                    tactic_data['regimes'].append(regime_id)
                    for metric in ['precision', 'recall', 'f1_score']:
                        if metric in avg_metrics:
                            tactic_data['metrics'][metric].append(avg_metrics[metric])
            for metric, values in tactic_data['metrics'].items():
                if values:
                    tactic_data['metrics'][f'{metric}_mean'] = np.mean(values)
                    tactic_data['metrics'][f'{metric}_std'] = np.std(values)
                    tactic_data['metrics'][f'{metric}_min'] = np.min(values)
                    tactic_data['metrics'][f'{metric}_max'] = np.max(values)
            tactic_comparison[tactic] = tactic_data
        return tactic_comparison

    def _calculate_overall_metrics(self, regime_evaluations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall metrics across all regimes and tactics."""
        overall = {'total_regimes': len(regime_evaluations), 'total_tactics': 0, 'overall_precision': [], 'overall_recall': [], 'overall_f1': [], 'best_regime': None, 'best_tactic': None, 'best_score': 0.0}
        for regime_id, regime_eval in regime_evaluations.items():
            regime_metrics = regime_eval.get('regime_metrics', {})
            if 'avg_precision' in regime_metrics:
                overall['overall_precision'].append(regime_metrics['avg_precision'])
            if 'avg_recall' in regime_metrics:
                overall['overall_recall'].append(regime_metrics['avg_recall'])
            if 'avg_f1' in regime_metrics:
                overall['overall_f1'].append(regime_metrics['avg_f1'])
                if regime_metrics['avg_f1'] > overall['best_score']:
                    overall['best_score'] = regime_metrics['avg_f1']
                    overall['best_regime'] = regime_id
                    overall['best_tactic'] = regime_eval.get('best_tactic')
            overall['total_tactics'] += regime_metrics.get('n_tactics', 0)
        for metric in ['overall_precision', 'overall_recall', 'overall_f1']:
            if overall[metric]:
                overall[f'{metric}_mean'] = np.mean(overall[metric])
                overall[f'{metric}_std'] = np.std(overall[metric])
            else:
                overall[f'{metric}_mean'] = 0.0
                overall[f'{metric}_std'] = 0.0
        return overall

    def _create_performance_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a concise performance summary."""
        summary = {'highlights': [], 'warnings': [], 'recommendations': []}
        overall = evaluation_results.get('overall_metrics', {})
        if overall.get('best_regime') and overall.get('best_tactic'):
            summary['highlights'].append(f"Best performance: {overall['best_tactic']} in regime {overall['best_regime']} (F1: {overall['best_score']:.3f})")
        if overall.get('overall_precision_mean', 0) > self.performance_thresholds['min_precision']:
            summary['highlights'].append(f"Overall precision meets threshold: {overall['overall_precision_mean']:.3f}")
        if overall.get('overall_recall_mean', 0) < self.performance_thresholds['min_recall']:
            summary['warnings'].append(f"Low overall recall: {overall.get('overall_recall_mean', 0):.3f}")
        if overall.get('overall_f1_std', 0) > 0.1:
            summary['recommendations'].append('High variance in performance across regimes - consider regime-specific tuning')
        return summary

    def _check_performance_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if metrics meet performance thresholds."""
        for metric, threshold in self.performance_thresholds.items():
            metric_name = metric.replace('min_', '')
            if metric_name in metrics and metrics[metric_name] < threshold:
                return False
        return True