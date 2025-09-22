"""
Bootstrap Confidence Intervals for HMM Training

Provides bootstrap-based confidence intervals to assess model stability
and detect overfitting with statistical rigor.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class BootstrapConfidenceIntervalAnalyzer:
    """Analyze model performance using bootstrap confidence intervals."""

    def __init__(self, n_bootstrap: int = 100, confidence_level: float = 0.95, n_jobs: int = -1):
        """
        Initialize the bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals (0.95 = 95%)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs
        self._lock = threading.Lock()

    def analyze_model_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze model stability using bootstrap confidence intervals.

        Args:
            model: Trained model instance
            X: Feature matrix
            y: Target labels
            train_size: Fraction of data to use for training in each bootstrap

        Returns:
            Dictionary with stability analysis results
        """
        try:
            # Split data once for consistency
            n_samples = len(X)
            train_indices = np.random.choice(n_samples, size=int(n_samples * train_size), replace=False)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Perform bootstrap analysis
            bootstrap_results = self._perform_bootstrap_analysis(model, X_train, y_train, X_test, y_test)

            # Analyze stability
            stability_analysis = self._analyze_stability(bootstrap_results)

            return {
                **stability_analysis,
                'n_bootstrap': self.n_bootstrap,
                'confidence_level': self.confidence_level,
                'train_size': train_size,
                'bootstrap_results': bootstrap_results
            }

        except Exception as e:
            logger.error(f"Bootstrap stability analysis failed: {e}")
            return {
                'error': str(e),
                'stability_score': 0.0,
                'confidence_intervals': {},
                'overfitting_probability': 1.0
            }

    def _perform_bootstrap_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Perform bootstrap resampling analysis.

        Args:
            model: Model instance to analyze
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with bootstrap results for each metric
        """
        results = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_f1': [],
            'test_f1': [],
            'train_precision': [],
            'test_precision': [],
            'train_recall': [],
            'test_recall': []
        }

        def bootstrap_iteration(i: int) -> Dict[str, float]:
            """Single bootstrap iteration."""
            try:
                # Resample training data
                boot_indices = resample(np.arange(len(X_train)), random_state=i)
                X_boot = X_train[boot_indices]
                y_boot = y_train[boot_indices]

                # Clone and train model
                model_boot = self._clone_model(model)
                model_boot.fit(X_boot, y_boot)

                # Calculate predictions
                train_pred = model_boot.predict(X_boot)
                test_pred = model_boot.predict(X_test)
                train_prob = model_boot.predict_proba(X_boot) if hasattr(model_boot, 'predict_proba') else None
                test_prob = model_boot.predict_proba(X_test) if hasattr(model_boot, 'predict_proba') else None

                # Calculate metrics
                iteration_results = {
                    'train_accuracy': accuracy_score(y_boot, train_pred),
                    'test_accuracy': accuracy_score(y_test, test_pred),
                    'train_f1': f1_score(y_boot, train_pred, average='weighted'),
                    'test_f1': f1_score(y_test, test_pred, average='weighted'),
                    'train_precision': precision_score(y_boot, train_pred, average='weighted'),
                    'test_precision': precision_score(y_test, test_pred, average='weighted'),
                    'train_recall': recall_score(y_boot, train_pred, average='weighted'),
                    'test_recall': recall_score(y_test, test_pred, average='weighted')
                }

                return iteration_results

            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                return {key: 0.5 for key in results.keys()}  # Default values

        # Parallel execution
        if self.n_jobs == 1:
            # Sequential execution
            for i in range(self.n_bootstrap):
                iteration_results = bootstrap_iteration(i)
                for key, value in iteration_results.items():
                    results[key].append(value)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(bootstrap_iteration, i) for i in range(self.n_bootstrap)]

                for future in as_completed(futures):
                    iteration_results = future.result()
                    for key, value in iteration_results.items():
                        results[key].append(value)

        return results

    def _clone_model(self, model: Any) -> Any:
        """Clone a model instance safely."""
        try:
            # Try sklearn clone first
            from sklearn.base import clone
            return clone(model)
        except Exception:
            # Fallback: try to create new instance with same parameters
            if hasattr(model, 'get_params'):
                params = model.get_params()
                model_class = model.__class__
                return model_class(**params)
            else:
                # Last resort: return the same model (not ideal but better than failing)
                logger.warning("Could not clone model, using original instance")
                return model

    def _analyze_stability(self, bootstrap_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Analyze bootstrap results for stability assessment.

        Args:
            bootstrap_results: Results from bootstrap analysis

        Returns:
            Dictionary with stability analysis
        """
        # Calculate confidence intervals for each metric
        confidence_intervals = {}
        stability_scores = {}

        for metric, values in bootstrap_results.items():
            values_array = np.array(values)

            # Calculate confidence interval
            lower_bound = np.percentile(values_array, (1 - self.confidence_level) / 2 * 100)
            upper_bound = np.percentile(values_array, (1 + self.confidence_level) / 2 * 100)
            mean_value = np.mean(values_array)
            std_value = np.std(values_array)

            confidence_intervals[metric] = {
                'mean': float(mean_value),
                'std': float(std_value),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'confidence_level': self.confidence_level
            }

            # Calculate stability score (inverse of coefficient of variation)
            if mean_value > 0:
                stability_scores[metric] = 1.0 - min(std_value / mean_value, 1.0)
            else:
                stability_scores[metric] = 0.0

        # Calculate overfitting probability
        train_accs = np.array(bootstrap_results['train_accuracy'])
        test_accs = np.array(bootstrap_results['test_accuracy'])
        overfitting_cases = np.sum(train_accs - test_accs > 0.1)  # >10% gap
        overfitting_probability = overfitting_cases / len(train_accs)

        # Overall stability score (average of metric stability scores)
        overall_stability = np.mean(list(stability_scores.values()))

        # Determine stability level
        if overall_stability > 0.8:
            stability_level = 'high'
        elif overall_stability > 0.6:
            stability_level = 'medium'
        else:
            stability_level = 'low'

        # Assess overfitting risk based on bootstrap analysis
        if overfitting_probability > 0.7:
            overfitting_risk = 'high'
        elif overfitting_probability > 0.4:
            overfitting_risk = 'medium'
        else:
            overfitting_risk = 'low'

        return {
            'stability_score': float(overall_stability),
            'stability_level': stability_level,
            'overfitting_probability': float(overfitting_probability),
            'overfitting_risk': overfitting_risk,
            'confidence_intervals': confidence_intervals,
            'stability_scores': stability_scores,
            'n_successful_bootstrap': len(bootstrap_results['train_accuracy'])
        }

    def compare_models_bootstrap(
        self,
        models: List[Any],
        model_names: List[str],
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7
    ) -> Dict[str, Any]:
        """
        Compare multiple models using bootstrap analysis.

        Args:
            models: List of trained model instances
            model_names: Names for each model
            X: Feature matrix
            y: Target labels
            train_size: Fraction of data to use for training

        Returns:
            Dictionary with model comparison results
        """
        if len(models) != len(model_names):
            raise ValueError("Number of models must match number of model names")

        comparison_results = {}

        for model, name in zip(models, model_names):
            logger.info(f"Analyzing model: {name}")
            model_results = self.analyze_model_stability(model, X, y, train_size)
            comparison_results[name] = model_results

        # Determine best model based on stability and performance
        best_model = None
        best_score = -1

        for name, results in comparison_results.items():
            if 'error' not in results:
                # Score based on test accuracy and stability
                test_acc = results['confidence_intervals']['test_accuracy']['mean']
                stability = results['stability_score']
                composite_score = test_acc * stability

                if composite_score > best_score:
                    best_score = composite_score
                    best_model = name

        return {
            'comparison_results': comparison_results,
            'best_model': best_model,
            'best_score': float(best_score) if best_model else 0.0,
            'model_names': model_names
        }

    def detect_statistical_significance(
        self,
        model1_results: Dict[str, Any],
        model2_results: Dict[str, Any],
        metric: str = 'test_accuracy'
    ) -> Dict[str, Any]:
        """
        Test if one model is statistically significantly better than another.

        Args:
            model1_results: Bootstrap results for first model
            model2_results: Bootstrap results for second model
            metric: Metric to compare

        Returns:
            Dictionary with significance test results
        """
        if metric not in model1_results['bootstrap_results'] or metric not in model2_results['bootstrap_results']:
            return {
                'error': f'Metric {metric} not found in bootstrap results',
                'significant': False
            }

        values1 = np.array(model1_results['bootstrap_results'][metric])
        values2 = np.array(model2_results['bootstrap_results'][metric])

        # Perform bootstrap significance test
        diff_values = values1 - values2
        p_value = np.mean(diff_values <= 0)  # Proportion where model2 is better

        # Calculate confidence interval of difference
        diff_lower = np.percentile(diff_values, 2.5)
        diff_upper = np.percentile(diff_values, 97.5)

        significant = p_value < 0.05  # 5% significance level

        return {
            'significant': significant,
            'p_value': float(p_value),
            'confidence_interval': {
                'lower': float(diff_lower),
                'upper': float(diff_upper)
            },
            'mean_difference': float(np.mean(diff_values)),
            'model1_better': np.mean(diff_values > 0) > 0.5
        }