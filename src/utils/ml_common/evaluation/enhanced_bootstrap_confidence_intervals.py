"""
Enhanced Bootstrap Confidence Intervals for ML Common

Provides bootstrap-based confidence intervals integrated with existing ml_common
evaluation infrastructure and enhanced with statistical significance testing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
from sklearn.utils import resample
from sklearn.base import clone
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, field

# Import existing ml_common utilities
try:
    from src.utils.ml_common.evaluation.evaluation_utils import EvaluationUtils
    from src.training.steps.market_analysis.hmm_models_training.shared_utilities.bootstrap_confidence_intervals import BootstrapConfidenceIntervalAnalyzer
    HMM_BOOTSTRAP_AVAILABLE = True
except ImportError:
    HMM_BOOTSTRAP_AVAILABLE = False
    BootstrapConfidenceIntervalAnalyzer = None

logger = logging.getLogger(__name__)


@dataclass
class BootstrapAnalysisResult:
    """Structured bootstrap confidence interval analysis results."""
    stability_score: float
    stability_level: str
    overfitting_probability: float
    overfitting_risk: str
    confidence_intervals: Dict[str, Dict[str, Union[float, str]]]
    stability_scores: Dict[str, float]
    n_successful_bootstrap: int
    recommendations: List[str] = field(default_factory=list)


class EnhancedBootstrapConfidenceIntervalAnalyzer:
    """Enhanced bootstrap analyzer integrated with ml_common infrastructure."""

    def __init__(self, n_bootstrap: int = 100, confidence_level: float = 0.95, n_jobs: int = -1):
        """
        Initialize enhanced bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples (reduced from 1000 to 100 for efficiency)
            confidence_level: Confidence level for intervals (0.95 = 95%)
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.n_jobs = n_jobs
        self._lock = threading.Lock()

        # Initialize evaluation utilities
        self.evaluation_utils = EvaluationUtils()

        # Initialize HMM bootstrap analyzer if available
        self.hmm_analyzer = None
        if HMM_BOOTSTRAP_AVAILABLE:
            self.hmm_analyzer = BootstrapConfidenceIntervalAnalyzer(
                n_bootstrap=n_bootstrap,
                confidence_level=confidence_level,
                n_jobs=n_jobs
            )

        logger.info("✅ Enhanced Bootstrap Confidence Interval Analyzer initialized")

    def analyze_model_stability(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        scoring_metrics: List[str] = None
    ) -> BootstrapAnalysisResult:
        """
        Analyze model stability using bootstrap confidence intervals.

        Args:
            model: Trained model instance
            X: Feature matrix
            y: Target labels
            train_size: Fraction of data to use for training in each bootstrap
            scoring_metrics: List of metrics to evaluate

        Returns:
            BootstrapAnalysisResult with comprehensive stability analysis
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'f1', 'precision', 'recall']

        try:
            # Split data once for consistency
            n_samples = len(X)
            train_indices = np.random.choice(n_samples, size=int(n_samples * train_size), replace=False)
            test_indices = np.setdiff1d(np.arange(n_samples), train_indices)

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Perform bootstrap analysis
            bootstrap_results = self._perform_bootstrap_analysis(
                model, X_train, y_train, X_test, y_test, scoring_metrics
            )

            # Analyze stability
            stability_analysis = self._analyze_stability(bootstrap_results, scoring_metrics)

            # Generate recommendations
            recommendations = self._generate_bootstrap_recommendations(stability_analysis)

            return BootstrapAnalysisResult(
                stability_score=stability_analysis['stability_score'],
                stability_level=stability_analysis['stability_level'],
                overfitting_probability=stability_analysis['overfitting_probability'],
                overfitting_risk=stability_analysis['overfitting_risk'],
                confidence_intervals=stability_analysis['confidence_intervals'],
                stability_scores=stability_analysis['stability_scores'],
                n_successful_bootstrap=stability_analysis['n_successful_bootstrap'],
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Bootstrap stability analysis failed: {e}")
            return BootstrapAnalysisResult(
                stability_score=0.0,
                stability_level='unknown',
                overfitting_probability=1.0,
                overfitting_risk='high',
                confidence_intervals={},
                stability_scores={},
                n_successful_bootstrap=0,
                recommendations=[f'Analysis failed: {str(e)}']
            )

    def _perform_bootstrap_analysis(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scoring_metrics: List[str]
    ) -> Dict[str, List[float]]:
        """
        Perform bootstrap resampling analysis.

        Args:
            model: Model instance to analyze
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            scoring_metrics: Metrics to evaluate

        Returns:
            Dictionary with bootstrap results for each metric
        """
        results = {metric: [] for metric in scoring_metrics}

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

                # Calculate metrics based on model type
                iteration_results = {}
                for metric in scoring_metrics:
                    if metric == 'accuracy':
                        iteration_results[metric] = accuracy_score(y_boot, train_pred)
                        iteration_results[f'test_{metric}'] = accuracy_score(y_test, test_pred)
                    elif metric == 'f1':
                        iteration_results[metric] = f1_score(y_boot, train_pred, average='weighted')
                        iteration_results[f'test_{metric}'] = f1_score(y_test, test_pred, average='weighted')
                    elif metric == 'precision':
                        iteration_results[metric] = precision_score(y_boot, train_pred, average='weighted')
                        iteration_results[f'test_{metric}'] = precision_score(y_test, test_pred, average='weighted')
                    elif metric == 'recall':
                        iteration_results[metric] = recall_score(y_boot, train_pred, average='weighted')
                        iteration_results[f'test_{metric}'] = recall_score(y_test, test_pred, average='weighted')
                    else:
                        # Default to accuracy for unknown metrics
                        iteration_results[metric] = accuracy_score(y_boot, train_pred)
                        iteration_results[f'test_{metric}'] = accuracy_score(y_test, test_pred)

                return iteration_results

            except Exception as e:
                logger.warning(f"Bootstrap iteration {i} failed: {e}")
                # Return default values on failure
                return {metric: 0.5 for metric in scoring_metrics}

        # Parallel execution
        if self.n_jobs == 1:
            # Sequential execution
            for i in range(self.n_bootstrap):
                iteration_results = bootstrap_iteration(i)
                for key, value in iteration_results.items():
                    if key in results:
                        results[key].append(value)
        else:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(bootstrap_iteration, i) for i in range(self.n_bootstrap)]

                for future in as_completed(futures):
                    iteration_results = future.result()
                    for key, value in iteration_results.items():
                        if key in results:
                            results[key].append(value)

        return results

    def _clone_model(self, model: Any) -> Any:
        """Clone a model instance safely."""
        try:
            # Try sklearn clone first
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

    def _analyze_stability(
        self,
        bootstrap_results: Dict[str, List[float]],
        scoring_metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze bootstrap results for stability assessment.

        Args:
            bootstrap_results: Results from bootstrap analysis
            scoring_metrics: Metrics that were evaluated

        Returns:
            Dictionary with stability analysis
        """
        # Calculate confidence intervals for each metric
        confidence_intervals = {}
        stability_scores = {}

        # Focus on test metrics for stability analysis
        test_metrics = [f'test_{metric}' for metric in scoring_metrics if f'test_{metric}' in bootstrap_results]

        for metric in test_metrics:
            values = bootstrap_results.get(metric, [])
            if not values:
                continue

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

        # Calculate overfitting probability using accuracy metrics
        train_accuracy = bootstrap_results.get('accuracy', [])
        test_accuracy = bootstrap_results.get('test_accuracy', [])

        if train_accuracy and test_accuracy and len(train_accuracy) == len(test_accuracy):
            train_accs = np.array(train_accuracy)
            test_accs = np.array(test_accuracy)
            overfitting_cases = np.sum(train_accs - test_accs > 0.1)  # >10% gap
            overfitting_probability = overfitting_cases / len(train_accs)
        else:
            overfitting_probability = 0.5  # Default to moderate probability

        # Overall stability score (average of metric stability scores)
        overall_stability = np.mean(list(stability_scores.values())) if stability_scores else 0.0

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
            'n_successful_bootstrap': len(bootstrap_results.get('accuracy', [])),
            'bootstrap_values': bootstrap_results  # Store raw bootstrap values for statistical testing
        }

    def _generate_bootstrap_recommendations(self, stability_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on bootstrap analysis."""
        recommendations = []

        stability_score = stability_analysis.get('stability_score', 0.0)
        overfitting_prob = stability_analysis.get('overfitting_probability', 0.0)
        stability_level = stability_analysis.get('stability_level', 'unknown')

        # Stability recommendations
        if stability_score < 0.6:
            recommendations.append("Model stability is low - consider ensemble methods or more robust algorithms")
        elif stability_score < 0.8:
            recommendations.append("Model stability is moderate - consider hyperparameter tuning")

        # Overfitting recommendations
        if overfitting_prob > 0.7:
            recommendations.append(f"High overfitting probability ({overfitting_prob:.1%}) - implement stronger regularization")
        elif overfitting_prob > 0.4:
            recommendations.append(f"Moderate overfitting probability ({overfitting_prob:.1%}) - consider regularization adjustment")

        # Stability level specific recommendations
        if stability_level == 'low':
            recommendations.append("Low stability level detected - review model selection criteria")

        return recommendations

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
            # Score based on test accuracy and stability
            test_acc = results.confidence_intervals.get('test_accuracy', {}).get('mean', 0.0)
            stability = results.stability_score
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
        try:
            # Check if metric exists in both results
            if metric not in model1_results.get('confidence_intervals', {}):
                return {
                    'error': f'Metric {metric} not found in model1 results',
                    'significant': False
                }

            if metric not in model2_results.get('confidence_intervals', {}):
                return {
                    'error': f'Metric {metric} not found in model2 results',
                    'significant': False
                }

            # Extract bootstrap values if available
            values1 = model1_results.get('bootstrap_values', {}).get(metric, [])
            values2 = model2_results.get('bootstrap_values', {}).get(metric, [])

            # If raw bootstrap values are not available, use confidence interval data
            if not values1 or not values2:
                values1_mean = model1_results['confidence_intervals'][metric]['mean']
                values2_mean = model2_results['confidence_intervals'][metric]['mean']
                values1_std = model1_results['confidence_intervals'][metric].get('std', 0.01)
                values2_std = model2_results['confidence_intervals'][metric].get('std', 0.01)

                # Perform t-test using means and standard deviations
                try:
                    from scipy import stats
                    
                    # Calculate pooled standard error
                    n1 = model1_results.get('n_successful_bootstrap', 100)
                    n2 = model2_results.get('n_successful_bootstrap', 100)
                    
                    pooled_std = np.sqrt((values1_std**2 / n1) + (values2_std**2 / n2))
                    
                    # Calculate t-statistic
                    t_stat = (values1_mean - values2_mean) / pooled_std
                    
                    # Calculate degrees of freedom
                    df = n1 + n2 - 2
                    
                    # Calculate p-value (two-tailed test)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                    
                except ImportError:
                    # Fallback to simple comparison if scipy not available
                    significant = abs(values1_mean - values2_mean) > 2 * np.sqrt(values1_std**2 + values2_std**2)
                    p_value = 0.05 if significant else 0.5
                    t_stat = (values1_mean - values2_mean) / np.sqrt(values1_std**2 + values2_std**2)
                    df = n1 + n2 - 2
                else:
                    # Use raw bootstrap values for more accurate testing
                    try:
                        from scipy import stats
                        
                        # Perform paired t-test if same number of bootstrap samples
                        if len(values1) == len(values2):
                            t_stat, p_value = stats.ttest_rel(values1, values2)
                            df = len(values1) - 1
                        else:
                            # Perform independent t-test
                            t_stat, p_value = stats.ttest_ind(values1, values2)
                            df = len(values1) + len(values2) - 2
                        
                    except ImportError:
                        # Fallback to simple comparison
                        mean1, mean2 = np.mean(values1), np.mean(values2)
                        std1, std2 = np.std(values1), np.std(values2)
                        significant = abs(mean1 - mean2) > 2 * np.sqrt(std1**2 + std2**2)
                        p_value = 0.05 if significant else 0.5
                        t_stat = (mean1 - mean2) / np.sqrt(std1**2 + std2**2)
                        df = len(values1) + len(values2) - 2

            # Determine significance (using alpha = 0.05)
            significant = p_value < 0.05
            
            # Calculate confidence interval for the difference
            if 'values1' in locals() and 'values2' in locals():
                diff_values = np.array(values1) - np.array(values2)
                ci_lower = np.percentile(diff_values, 2.5)
                ci_upper = np.percentile(diff_values, 97.5)
                mean_diff = np.mean(diff_values)
            else:
                mean_diff = values1_mean - values2_mean
                margin_error = 1.96 * pooled_std  # 95% CI
                ci_lower = mean_diff - margin_error
                ci_upper = mean_diff + margin_error

            return {
                'significant': significant,
                'p_value': float(p_value),
                't_statistic': float(t_stat),
                'degrees_of_freedom': int(df),
                'confidence_interval': {
                    'lower': float(ci_lower),
                    'upper': float(ci_upper)
                },
                'mean_difference': float(mean_diff),
                'model1_better': mean_diff > 0,
                'effect_size': float(abs(mean_diff) / np.sqrt((values1_std**2 + values2_std**2) / 2)) if 'values1_std' in locals() else None,
                'test_type': 'paired_ttest' if len(values1) == len(values2) and 'values1' in locals() else 'independent_ttest'
            }

        except Exception as e:
            logger.error(f"Statistical significance test failed: {e}")
            return {
                'error': str(e),
                'significant': False,
                'p_value': 1.0
            }

    def comprehensive_model_comparison(
        self,
        models: List[Any],
        model_names: List[str],
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model comparison with statistical significance testing.
        
        Args:
            models: List of trained model instances
            model_names: Names for each model
            X: Feature matrix
            y: Target labels
            train_size: Fraction of data to use for training
            significance_level: Significance level for statistical tests
            
        Returns:
            Comprehensive comparison results with statistical significance
        """
        try:
            if len(models) != len(model_names):
                raise ValueError("Number of models must match number of model names")
            
            logger.info(f"🔬 Starting comprehensive model comparison with {len(models)} models")
            
            # Analyze each model
            model_results = {}
            for model, name in zip(models, model_names):
                logger.info(f"   → Analyzing model: {name}")
                model_results[name] = self.analyze_model_stability(model, X, y, train_size)
            
            # Perform pairwise statistical significance tests
            significance_matrix = {}
            pairwise_comparisons = []
            
            for i, name1 in enumerate(model_names):
                significance_matrix[name1] = {}
                for j, name2 in enumerate(model_names):
                    if i != j:
                        # Test significance for test_accuracy metric
                        sig_result = self.detect_statistical_significance(
                            model_results[name1], 
                            model_results[name2], 
                            metric='test_accuracy'
                        )
                        
                        significance_matrix[name1][name2] = sig_result
                        pairwise_comparisons.append({
                            'model1': name1,
                            'model2': name2,
                            'significant': sig_result.get('significant', False),
                            'p_value': sig_result.get('p_value', 1.0),
                            'mean_difference': sig_result.get('mean_difference', 0.0),
                            'effect_size': sig_result.get('effect_size', 0.0)
                        })
            
            # Rank models by performance
            model_scores = []
            for name in model_names:
                test_acc = model_results[name].confidence_intervals.get('test_accuracy', {}).get('mean', 0.0)
                stability = model_results[name].stability_score
                overfitting_prob = model_results[name].overfitting_probability
                
                # Composite score considering performance, stability, and overfitting
                composite_score = test_acc * stability * (1 - overfitting_prob)
                
                model_scores.append({
                    'model_name': name,
                    'test_accuracy': test_acc,
                    'stability_score': stability,
                    'overfitting_probability': overfitting_prob,
                    'composite_score': composite_score
                })
            
            # Sort by composite score
            model_scores.sort(key=lambda x: x['composite_score'], reverse=True)
            
            # Determine best model
            best_model = model_scores[0]['model_name'] if model_scores else None
            
            # Count significant wins for each model
            win_counts = {name: 0 for name in model_names}
            for comparison in pairwise_comparisons:
                if comparison['significant'] and comparison['mean_difference'] > 0:
                    win_counts[comparison['model1']] += 1
            
            # Generate summary statistics
            summary_stats = {
                'total_comparisons': len(pairwise_comparisons),
                'significant_comparisons': sum(1 for c in pairwise_comparisons if c['significant']),
                'significance_rate': sum(1 for c in pairwise_comparisons if c['significant']) / len(pairwise_comparisons) if pairwise_comparisons else 0,
                'best_model': best_model,
                'model_rankings': model_scores,
                'win_counts': win_counts
            }
            
            logger.info(f"✅ Comprehensive comparison completed")
            logger.info(f"   → Best model: {best_model}")
            logger.info(f"   → Significant comparisons: {summary_stats['significant_comparisons']}/{summary_stats['total_comparisons']}")
            
            return {
                'model_results': model_results,
                'significance_matrix': significance_matrix,
                'pairwise_comparisons': pairwise_comparisons,
                'summary_stats': summary_stats,
                'best_model': best_model,
                'significance_level': significance_level
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive model comparison failed: {e}")
            return {
                'error': str(e),
                'model_results': {},
                'significance_matrix': {},
                'pairwise_comparisons': [],
                'summary_stats': {},
                'best_model': None
        }