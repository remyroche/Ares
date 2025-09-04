"""Step 19: Monte Carlo Validation - Updated to use BaseStep pattern."""
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from src.core.decorators import handles_errors, log_execution_time
from .base_validation_step import BaseValidationStep
from copy import copy

class MonteCarloValidationStep(BaseValidationStep):
    """Step 19: Monte Carlo Validation with random sampling."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the Monte Carlo Validation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, '19', 'monte_carlo_validation')

    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.mc_config = {'n_iterations': self.config.get('monte_carlo_iterations', 100), 'test_size': self.config.get('monte_carlo_test_size', 0.2), 'bootstrap': self.config.get('monte_carlo_bootstrap', True), 'stratify': self.config.get('monte_carlo_stratify', True), 'confidence_level': self.config.get('confidence_level', 0.95), 'parallel_iterations': self.config.get('parallel_iterations', True)}
        self.simulation_results: List[Dict[str, Any]] = []
        self.model_distributions: Dict[str, Dict[str, List[float]]] = {}

    def _validate_step_specific_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> List[str]:
        """Validate step-specific inputs."""
        errors = []
        X, y = self._extract_validation_data(pipeline_state)
        if len(X) < 1000:
            errors.append(f'Insufficient data for Monte Carlo validation: {len(X)} samples (minimum 1000 recommended)')
        return errors

    @handles_errors(exceptions=(Exception,), default_return={'success': False}, context='monte carlo validation logic')
    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the Monte Carlo validation logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with validation results
        """
        self.logger.info('🎲 Starting Monte Carlo validation...')
        X, y = self._extract_validation_data(pipeline_state)
        if X.empty or len(y) == 0:
            self.logger.warning('No data available for Monte Carlo validation')
            return pipeline_state
        models = self._get_models_for_validation(pipeline_state)
        if not models:
            self.logger.warning('No models available for validation')
            return pipeline_state
        for iteration in range(self.mc_config['n_iterations']):
            if iteration % 10 == 0:
                self.logger.info(f"Running iteration {iteration + 1}/{self.mc_config['n_iterations']}...")
            iteration_results = await self._run_single_simulation(models, X, y, iteration)
            self.simulation_results.append(iteration_results)
            self._update_distributions(iteration_results)
        mc_statistics = self._calculate_mc_statistics()
        confidence_intervals = self._calculate_confidence_intervals()
        result = pipeline_state.copy()
        result[f'{self.full_step_name}_results'] = {'simulation_results': self.simulation_results, 'model_distributions': self.model_distributions, 'statistics': mc_statistics, 'confidence_intervals': confidence_intervals, 'configuration': self.mc_config}
        result[f'{self.full_step_name}_summary'] = self._create_validation_summary({'model_results': mc_statistics, 'overall_metrics': self._calculate_overall_mc_metrics(mc_statistics)})
        return result

    async def _run_single_simulation(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series, iteration: int) -> Dict[str, Any]:
        """Run a single Monte Carlo simulation.
        
        Args:
            models: Models to validate
            X: Features
            y: Labels
            iteration: Iteration number
            
        Returns:
            Simulation results
        """
        simulation_result = {'iteration': iteration, 'model_results': {}, 'sample_info': {}}
        if self.mc_config['bootstrap']:
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[indices]
            y_sample = y.iloc[indices]
            oob_indices = list(set(range(len(X))) - set(indices))
            if oob_indices:
                X_test = X.iloc[oob_indices]
                y_test = y.iloc[oob_indices]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=self.mc_config['test_size'], stratify=y_sample if self.mc_config['stratify'] else None, random_state=iteration)
                X_sample = X_train
                y_sample = y_train
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.mc_config['test_size'], stratify=y if self.mc_config['stratify'] else None, random_state=iteration)
            X_sample = X_train
            y_sample = y_train
        simulation_result['sample_info'] = {'train_size': len(X_sample), 'test_size': len(X_test)}
        for model_name, model in models.items():
            try:
                model.fit(X_sample, y_sample)
                y_pred = model.predict(X_test)
                metrics = {'accuracy': accuracy_score(y_test, y_pred), 'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0), 'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0), 'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)}
                simulation_result['model_results'][model_name] = metrics
            except Exception as e:
                self.logger.warning(f'Failed to validate {model_name} in iteration {iteration}: {str(e)}')
                simulation_result['model_results'][model_name] = {'error': str(e)}
        return simulation_result

    def _update_distributions(self, iteration_results: Dict[str, Any]) -> None:
        """Update metric distributions with iteration results."""
        for model_name, metrics in iteration_results['model_results'].items():
            if 'error' in metrics:
                continue
            if model_name not in self.model_distributions:
                self.model_distributions[model_name] = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
            for metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric_name in metrics:
                    self.model_distributions[model_name][metric_name].append(metrics[metric_name])

    def _calculate_mc_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculate statistics from Monte Carlo simulations."""
        statistics = {}
        for model_name, distributions in self.model_distributions.items():
            model_stats = {}
            for metric_name, values in distributions.items():
                if values:
                    model_stats[f'{metric_name}_mean'] = np.mean(values)
                    model_stats[f'{metric_name}_std'] = np.std(values)
                    model_stats[f'{metric_name}_median'] = np.median(values)
                    model_stats[f'{metric_name}_min'] = np.min(values)
                    model_stats[f'{metric_name}_max'] = np.max(values)
                    model_stats[f'{metric_name}_cv'] = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
            statistics[model_name] = model_stats
        return statistics

    def _calculate_confidence_intervals(self) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for each model and metric."""
        confidence_intervals = {}
        alpha = 1 - self.mc_config['confidence_level']
        for model_name, distributions in self.model_distributions.items():
            model_ci = {}
            for metric_name, values in distributions.items():
                if values:
                    lower_percentile = alpha / 2 * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    ci_lower = np.percentile(values, lower_percentile)
                    ci_upper = np.percentile(values, upper_percentile)
                    model_ci[f'{metric_name}_ci'] = (ci_lower, ci_upper)
            confidence_intervals[model_name] = model_ci
        return confidence_intervals

    def _calculate_overall_mc_metrics(self, mc_statistics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall Monte Carlo metrics."""
        metrics = {'n_models_validated': len(mc_statistics), 'n_simulations': self.mc_config['n_iterations'], 'avg_accuracy': [], 'avg_f1': [], 'avg_cv': [], 'model_stability': []}
        for model_stats in mc_statistics.values():
            if 'accuracy_mean' in model_stats:
                metrics['avg_accuracy'].append(model_stats['accuracy_mean'])
            if 'f1_score_mean' in model_stats:
                metrics['avg_f1'].append(model_stats['f1_score_mean'])
            if 'f1_score_cv' in model_stats:
                metrics['avg_cv'].append(model_stats['f1_score_cv'])
                metrics['model_stability'].append(1.0 / (1.0 + model_stats['f1_score_cv']))
        for key in ['avg_accuracy', 'avg_f1', 'avg_cv', 'model_stability']:
            if metrics[key]:
                metrics[key] = np.mean(metrics[key])
            else:
                metrics[key] = 0.0
        return metrics

    def _validate_step_specific_outputs(self, pipeline_state: Dict[str, Any]) -> List[str]:
        """Validate step-specific outputs."""
        errors = []
        results_key = f'{self.full_step_name}_results'
        if results_key in pipeline_state:
            results = pipeline_state[results_key]
            if 'simulation_results' not in results or len(results['simulation_results']) == 0:
                errors.append('No simulation results found in Monte Carlo validation')
            if 'confidence_intervals' not in results:
                errors.append('No confidence intervals calculated')
        return errors

    def _add_step_specific_summary(self, summary: Dict[str, Any], validation_results: Dict[str, Any]) -> None:
        """Add step-specific items to summary."""
        overall = validation_results.get('overall_metrics', {})
        if overall.get('avg_f1', 0) > 0:
            summary['key_findings'].append(f"Average F1 score across {overall.get('n_simulations', 0)} simulations: {overall['avg_f1']:.3f}")
        if overall.get('model_stability', 0) > 0.85:
            summary['key_findings'].append(f"High model stability across random samples: {overall['model_stability']:.3f}")
        if overall.get('avg_cv', 0) > 0.2:
            summary['warnings'].append(f"High coefficient of variation ({overall['avg_cv']:.3f}) indicates unstable performance")
        if overall.get('n_simulations', 0) < 100:
            summary['recommendations'].append('Consider increasing the number of Monte Carlo iterations for more robust estimates')
        if 'model_results' in validation_results:
            ci_info = []
            for model_name, stats in validation_results['model_results'].items():
                if f'f1_score_mean' in stats:
                    ci_info.append(f"{model_name}: F1={stats['f1_score_mean']:.3f}±{stats.get('f1_score_std', 0):.3f}")
            if ci_info:
                summary['key_findings'].append(f"Model performance: {', '.join(ci_info[:3])}")

    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return ['tactician_specialist_models', 'features', 'step15_tactician_specialist_training_completed']

    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [f'{self.full_step_name}_results', f'{self.full_step_name}_summary']

    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        return ['step15_tactician_specialist_training']