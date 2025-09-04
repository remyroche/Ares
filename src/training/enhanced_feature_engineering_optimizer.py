from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
'\nEnhanced Feature Engineering Optimizer\n\nThis module optimizes the period optimization process itself using:\n1. Random Forest + SHAP for meta-optimization\n2. Mutual Information for parameter space reduction\n3. Adaptive parameter sampling based on performance\n4. Multi-objective optimization considering multiple metrics\n'
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import optuna
import pandas as pd
import shap
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from src.utils.logger import system_logger
import asyncio

class EnhancedFeatureEngineeringOptimizer:
    """
    Enhanced feature engineering optimizer that optimizes the optimization process itself.

    Features:
    - Meta-optimization using Random Forest + SHAP
    - Mutual Information for parameter space reduction
    - Adaptive parameter sampling
    - Multi-objective optimization
    - Performance-based early stopping
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the enhanced feature engineering optimizer."""
        self.config = config
        self.logger = system_logger.getChild('EnhancedFeatureEngineeringOptimizer')
        self.meta_optimization_config = config.get('enhanced_feature_optimization', {'meta_optimization': {'enabled': True, 'n_trials': 200, 'cv_folds': 5, 'random_state': 42, 'early_stopping_patience': 20, 'performance_threshold': 0.8}, 'parameter_space_optimization': {'enabled': True, 'mi_threshold': 0.1, 'correlation_threshold': 0.8, 'adaptive_sampling': True, 'space_reduction_factor': 0.5}, 'multi_objective': {'enabled': True, 'objectives': ['importance', 'stability', 'diversity', 'efficiency'], 'weights': [0.4, 0.2, 0.2, 0.2]}, 'shap_analysis': {'n_samples': 1000, 'max_display': 20, 'interaction_analysis': True, 'feature_interactions': True}})
        self.base_feature_params = self._initialize_base_parameters()
        self.optimization_history = []
        self.performance_metrics = {}
        self.logger.info('🚀 Enhanced Feature Engineering Optimizer initialized')

    def _initialize_base_parameters(self) -> dict[str, Any]:
        """Initialize base parameter ranges for all features."""
        return {'RSI': {'lookback_period': list(range(5, 61, 5)), 'overbought_threshold': list(range(65, 91, 5)), 'oversold_threshold': list(range(10, 36, 5))}, 'MACD': {'fast_period': list(range(5, 26, 1)), 'slow_period': list(range(20, 41, 2)), 'signal_period': list(range(5, 16, 1))}, 'Bollinger_Bands': {'lookback_period': list(range(10, 61, 5)), 'std_dev': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 'squeeze_threshold': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}, 'SMA': {'short_period': list(range(3, 26, 1)), 'long_period': list(range(20, 121, 5))}, 'EMA': {'short_period': list(range(3, 26, 1)), 'long_period': list(range(20, 121, 5))}, 'ATR': {'lookback_period': list(range(5, 36, 1))}, 'Stochastic': {'k_period': list(range(5, 36, 1)), 'd_period': list(range(3, 11, 1)), 'overbought': list(range(70, 91, 5)), 'oversold': list(range(10, 31, 5))}, 'ADX': {'lookback_period': list(range(5, 36, 1)), 'threshold': list(range(15, 41, 5))}, 'CCI': {'lookback_period': list(range(5, 36, 1)), 'constant': [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]}}

    @handles_errors(fallback={})
    async def optimize_feature_parameters_enhanced(self, data: pd.DataFrame, target: pd.Series, regimes: pd.Series | None=None, symbol: str='UNKNOWN', exchange: str='UNKNOWN', timeframe: str='1m') -> dict[str, Any]:
        """
        Enhanced feature parameter optimization with meta-optimization.

        Args:
            data: Feature data
            target: Target variable
            regimes: HMM regime labels (optional)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Dictionary with enhanced optimization results
        """
        self.logger.info(f'🎯 Starting enhanced feature parameter optimization for {symbol} on {exchange}')
        results = {'optimization_timestamp': datetime.now().isoformat(), 'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'meta_optimization_results': {}, 'parameter_space_optimization': {}, 'multi_objective_results': {}, 'enhanced_optimizations': {}, 'performance_analysis': {}}
        self.logger.info('🔍 Optimizing parameter space...')
        optimized_param_space = await self._optimize_parameter_space(data, target)
        results['parameter_space_optimization'] = optimized_param_space
        self.logger.info('🧠 Performing meta-optimization...')
        meta_results = await self._perform_meta_optimization(data, target, optimized_param_space)
        results['meta_optimization_results'] = meta_results
        self.logger.info('🎯 Performing multi-objective optimization...')
        multi_obj_results = await self._perform_multi_objective_optimization(data, target, optimized_param_space, regimes)
        results['multi_objective_results'] = multi_obj_results
        self.logger.info('⚡ Performing enhanced feature optimization...')
        enhanced_results = await self._perform_enhanced_feature_optimization(data, target, optimized_param_space, regimes)
        results['enhanced_optimizations'] = enhanced_results
        self.logger.info('📊 Analyzing optimization performance...')
        performance_analysis = await self._analyze_optimization_performance(results)
        results['performance_analysis'] = performance_analysis
        await self._save_enhanced_optimization_results(results, symbol, exchange, timeframe)
        self.logger.info('✅ Enhanced feature parameter optimization completed successfully')
        return results

    async def _optimize_parameter_space(self, data: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
        """Optimize the parameter space using MI and correlation analysis."""
        optimized_space = {}
        for feature_name, base_params in self.base_feature_params.items():
            self.logger.info(f'🔍 Optimizing parameter space for {feature_name}...')
            sample_combinations = self._generate_sample_combinations(base_params, n_samples=100)
            performance_metrics = []
            for params in sample_combinations:
                feature_values = self._calculate_feature_with_params(data, feature_name, params)
                if feature_values is not None:
                    metrics = await self._calculate_performance_metrics(feature_values, target)
                    performance_metrics.append({'params': params, 'metrics': metrics})
            param_importance = await self._analyze_parameter_importance(sample_combinations, performance_metrics)
            reduced_params = self._reduce_parameter_space(base_params, param_importance)
            optimized_space[feature_name] = {'original_params': base_params, 'reduced_params': reduced_params, 'parameter_importance': param_importance, 'space_reduction_ratio': len(reduced_params) / len(base_params)}
        return optimized_space

    async def _perform_meta_optimization(self, data: pd.DataFrame, target: pd.Series, optimized_param_space: dict[str, Any]) -> dict[str, Any]:
        """Perform meta-optimization using Optuna."""
        meta_results = {}
        for feature_name, param_space in optimized_param_space.items():
            self.logger.info(f'🧠 Meta-optimizing {feature_name}...')
            study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=MedianPruner())

            def objective(trial: Any) -> float:
                params = self._sample_parameters_from_space(param_space['reduced_params'], trial)
                feature_values = self._calculate_feature_with_params(data, feature_name, params)
                if feature_values is None:
                    return 0.0
                return self._calculate_multi_objective_score(feature_values, target, params)
            study.optimize(objective, n_trials=self.meta_optimization_config['meta_optimization']['n_trials'], callbacks=[self._early_stopping_callback])
            meta_results[feature_name] = {'best_params': study.best_params, 'best_value': study.best_value, 'n_trials': len(study.trials), 'optimization_history': study.trials_dataframe().to_dict('records')}
        return meta_results

    async def _perform_multi_objective_optimization(self, data: pd.DataFrame, target: pd.Series, optimized_param_space: dict[str, Any], regimes: pd.Series | None=None) -> dict[str, Any]:
        """Perform multi-objective optimization considering multiple metrics."""
        multi_obj_results = {}
        for feature_name, param_space in optimized_param_space.items():
            self.logger.info(f'🎯 Multi-objective optimizing {feature_name}...')
            combinations = self._generate_param_combinations(param_space['reduced_params'])
            objective_scores = []
            for params in combinations:
                feature_values = self._calculate_feature_with_params(data, feature_name, params)
                if feature_values is not None:
                    scores = await self._calculate_all_objectives(feature_values, target, params, regimes)
                    objective_scores.append({'params': params, 'scores': scores, 'weighted_score': self._calculate_weighted_score(scores)})
            pareto_optimal = self._find_pareto_optimal_solutions(objective_scores)
            multi_obj_results[feature_name] = {'pareto_optimal_solutions': pareto_optimal, 'objective_weights': self.meta_optimization_config['multi_objective']['weights'], 'n_solutions': len(pareto_optimal)}
        return multi_obj_results

    async def _perform_enhanced_feature_optimization(self, data: pd.DataFrame, target: pd.Series, optimized_param_space: dict[str, Any], regimes: pd.Series | None=None) -> dict[str, Any]:
        """Perform enhanced feature optimization with optimized parameters."""
        enhanced_results = {}
        for feature_name, param_space in optimized_param_space.items():
            self.logger.info(f'⚡ Enhanced optimizing {feature_name}...')
            reduced_params = param_space['reduced_params']
            if regimes is not None and len(regimes.unique()) > 1:
                regime_results = {}
                for regime in regimes.unique():
                    regime_mask = regimes == regime
                    regime_data = data[regime_mask]
                    regime_target = target[regime_mask]
                    if len(regime_data) >= 100:
                        regime_opt = await self._optimize_feature_for_regime(regime_data, regime_target, feature_name, reduced_params)
                        regime_results[f'regime_{regime}'] = regime_opt
                enhanced_results[feature_name] = {'regime_optimizations': regime_results, 'global_optimization': await self._optimize_feature_globally(data, target, feature_name, reduced_params)}
            else:
                enhanced_results[feature_name] = await self._optimize_feature_globally(data, target, feature_name, reduced_params)
        return enhanced_results

    async def _analyze_parameter_importance(self, parameter_combinations: list[dict[str, Any]], performance_metrics: list[dict[str, Any]]) -> dict[str, float]:
        """Analyze parameter importance using Random Forest + SHAP."""
        if not parameter_combinations or not performance_metrics:
            return {}
        param_data = []
        performance_scores = []
        for combo, metrics in zip(parameter_combinations, performance_metrics, strict=False):
            flat_params = self._flatten_parameters(combo)
            param_data.append(flat_params)
            performance_scores.append(metrics['metrics']['overall_score'])
        param_df = pd.DataFrame(param_data)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(param_df, performance_scores)
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(param_df)
        importance_dict = {}
        for i, feature in enumerate(param_df.columns):
            importance_dict[feature] = np.mean(np.abs(shap_values[:, i]))
        return importance_dict

    def _reduce_parameter_space(self, base_params: dict[str, list], param_importance: dict[str, float]) -> dict[str, list]:
        """Reduce parameter space based on importance scores."""
        reduced_params = {}
        self.meta_optimization_config['parameter_space_optimization']['space_reduction_factor']
        for param_name, param_values in base_params.items():
            importance = param_importance.get(param_name, 0.0)
            if importance > 0.5:
                keep_ratio = 0.8
            elif importance > 0.2:
                keep_ratio = 0.6
            else:
                keep_ratio = 0.4
            n_keep = max(2, int(len(param_values) * keep_ratio))
            selected_values = self._select_representative_values(param_values, n_keep)
            reduced_params[param_name] = selected_values
        return reduced_params

    async def _calculate_all_objectives(self, feature_values: pd.Series, target: pd.Series, params: dict[str, Any], regimes: pd.Series | None=None) -> dict[str, float]:
        """Calculate all objective scores for multi-objective optimization."""
        objectives = {}
        objectives['importance'] = await self._calculate_importance_score(feature_values, target)
        objectives['stability'] = await self._calculate_stability_score(feature_values, target)
        objectives['diversity'] = await self._calculate_diversity_score(feature_values, target)
        objectives['efficiency'] = self._calculate_efficiency_score(params)
        return objectives

    def _calculate_weighted_score(self, scores: dict[str, float]) -> float:
        """Calculate weighted score from multiple objectives."""
        weights = self.meta_optimization_config['multi_objective']['weights']
        objectives = self.meta_optimization_config['multi_objective']['objectives']
        weighted_score = 0.0
        for obj, weight in zip(objectives, weights, strict=False):
            weighted_score += scores.get(obj, 0.0) * weight
        return weighted_score

    def _find_pareto_optimal_solutions(self, objective_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Find Pareto-optimal solutions."""
        pareto_optimal = []
        for i, solution in enumerate(objective_scores):
            is_pareto_optimal = True
            for j, other_solution in enumerate(objective_scores):
                if i != j:
                    dominates = True
                    for obj in self.meta_optimization_config['multi_objective']['objectives']:
                        if other_solution['scores'].get(obj, 0.0) < solution['scores'].get(obj, 0.0):
                            dominates = False
                            break
                    if dominates:
                        is_pareto_optimal = False
                        break
            if is_pareto_optimal:
                pareto_optimal.append(solution)
        return pareto_optimal

    def _early_stopping_callback(self, study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        """Early stopping callback for Optuna optimization."""
        patience = self.meta_optimization_config['meta_optimization']['early_stopping_patience']
        threshold = self.meta_optimization_config['meta_optimization']['performance_threshold']
        if study.best_value > threshold:
            study.stop()
        if len(study.trials) > patience:
            recent_trials = study.trials[-patience:]
            if all((trial.value <= study.best_value for trial in recent_trials)):
                study.stop()

    def _calculate_feature_with_params(self, data: pd.DataFrame, feature_name: str, params: dict[str, Any]) -> pd.Series | None:
        """Calculate feature with given parameters."""
        from src.training.feature_engineering_optimizer import FeatureEngineeringOptimizer
        base_optimizer = FeatureEngineeringOptimizer(self.config)
        return base_optimizer._generate_synthetic_feature(data, feature_name, params)

    def _generate_sample_combinations(self, params: dict[str, list], n_samples: int) -> list[dict[str, Any]]:
        """Generate sample parameter combinations."""
        import itertools
        param_names = list(params.keys())
        param_values = list(params.values())
        all_combinations = list(itertools.product(*param_values))
        if len(all_combinations) <= n_samples:
            return [dict(zip(param_names, combo, strict=False)) for combo in all_combinations]
        sampled_indices = np.random.choice(len(all_combinations), size=n_samples, replace=False)
        return [dict(zip(param_names, all_combinations[i], strict=False)) for i in sampled_indices]

    def _flatten_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested parameters for analysis."""
        flattened = {}
        for key, value in params.items():
            if isinstance(value, int | float):
                flattened[key] = value
            elif isinstance(value, list):
                flattened[f'{key}_count'] = len(value)
                flattened[f'{key}_min'] = min(value)
                flattened[f'{key}_max'] = max(value)
                flattened[f'{key}_mean'] = np.mean(value)
        return flattened

    def _select_representative_values(self, values: list, n_select: int) -> list:
        """Select representative values from a list."""
        if len(values) <= n_select:
            return values
        if isinstance(values[0], int | float):
            quantiles = np.linspace(0, 1, n_select)
            selected = [np.percentile(values, q * 100) for q in quantiles]
            selected = [min(values, key=lambda x: abs(x - s)) for s in selected]
            return list(set(selected))
        return list(np.random.choice(values, size=n_select, replace=False))

    async def _save_enhanced_optimization_results(self, results: dict[str, Any], symbol: str, exchange: str, timeframe: str) -> None:
        """Save enhanced optimization results to file."""
        output_dir = Path('data/enhanced_feature_engineering_optimization')
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f'{exchange}_{symbol}_{timeframe}_enhanced_feature_optimization.json'
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f'💾 Saved enhanced optimization results to {filepath}')

    async def _calculate_performance_metrics(self, feature_values: pd.Series, target: pd.Series) -> dict[str, float]:
        """Calculate performance metrics for a feature."""
        metrics = {'importance': 0.0, 'stability': 0.0, 'diversity': 0.0, 'efficiency': 0.0, 'overall_score': 0.0}
        try:
            metrics['importance'] = await self._calculate_importance_score(feature_values, target)
            metrics['stability'] = await self._calculate_stability_score(feature_values, target)
            metrics['diversity'] = await self._calculate_diversity_score(feature_values, target)
            metrics['efficiency'] = 1.0
            weights = self.meta_optimization_config['multi_objective']['weights']
            objectives = self.meta_optimization_config['multi_objective']['objectives']
            overall_score = 0.0
            for obj, weight in zip(objectives, weights, strict=False):
                overall_score += metrics.get(obj, 0.0) * weight
            metrics['overall_score'] = overall_score
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating performance metrics: {e}')
        return metrics

    async def _calculate_importance_score(self, feature_values: pd.Series, target: pd.Series) -> float:
        """Calculate importance score using SHAP."""
        try:
            X = feature_values.values.reshape(-1, 1)
            y = target.values
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
            importance = np.mean(np.abs(shap_values))
            return float(importance)
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating importance score: {e}')
            return 0.0

    async def _calculate_stability_score(self, feature_values: pd.Series, target: pd.Series) -> float:
        """Calculate stability score using cross-validation."""
        try:
            X = feature_values.values.reshape(-1, 1)
            y = target.values
            cv_scores = cross_val_score(RandomForestRegressor(n_estimators=50, random_state=42), X, y, cv=5, scoring='neg_mean_squared_error')
            stability = 1.0 - np.std(cv_scores) / np.mean(np.abs(cv_scores))
            return max(0.0, min(1.0, stability))
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating stability score: {e}')
            return 0.0

    async def _calculate_diversity_score(self, feature_values: pd.Series, target: pd.Series) -> float:
        """Calculate diversity score (inverse correlation with target)."""
        try:
            correlation = feature_values.corr(target)
            diversity = 1.0 - abs(correlation)
            return max(0.0, min(1.0, diversity))
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating diversity score: {e}')
            return 0.0

    def _calculate_efficiency_score(self, params: dict[str, Any]) -> float:
        """Calculate efficiency score based on parameter complexity."""
        try:
            efficiency = 1.0
            for param_name, param_value in params.items():
                if 'period' in param_name.lower() or 'lookback' in param_name.lower():
                    if isinstance(param_value, int | float):
                        efficiency *= 1.0 / (1.0 + param_value / 100.0)
            return max(0.0, min(1.0, efficiency))
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating efficiency score: {e}')
            return 0.5

    def _calculate_multi_objective_score(self, feature_values: pd.Series, target: pd.Series, params: dict[str, Any]) -> float:
        """Calculate multi-objective score."""
        try:
            objectives = {'importance': 0.0, 'stability': 0.0, 'diversity': 0.0, 'efficiency': 0.0}
            X = feature_values.values.reshape(-1, 1)
            y = target.values
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            objectives['importance'] = rf.feature_importances_[0]
            objectives['stability'] = 0.8
            objectives['diversity'] = 1.0 - abs(feature_values.corr(target))
            objectives['efficiency'] = self._calculate_efficiency_score(params)
            return self._calculate_weighted_score(objectives)
        except Exception as e:
            self.logger.warning(f'⚠️ Error calculating multi-objective score: {e}')
            return 0.0

    def _sample_parameters_from_space(self, param_space: dict[str, list], trial: optuna.Trial) -> dict[str, Any]:
        """Sample parameters from parameter space using Optuna trial."""
        sampled_params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values[0], int):
                sampled_params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
            elif isinstance(param_values[0], float):
                sampled_params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
            else:
                sampled_params[param_name] = trial.suggest_categorical(param_name, param_values)
        return sampled_params

    def _generate_param_combinations(self, params: dict[str, list]) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools
from src.core.decorators.errors import handles_errors
        param_names = list(params.keys())
        param_values = list(params.values())
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination, strict=False))
            combinations.append(param_dict)
        return combinations

    async def _optimize_feature_for_regime(self, data: pd.DataFrame, target: pd.Series, feature_name: str, reduced_params: dict[str, list]) -> dict[str, Any]:
        """Optimize feature for a specific regime."""
        combinations = self._generate_param_combinations(reduced_params)
        feature_scores = []
        for params in combinations:
            feature_values = self._calculate_feature_with_params(data, feature_name, params)
            if feature_values is not None:
                importance_score = await self._calculate_importance_score(feature_values, target)
                feature_scores.append({'params': params, 'importance': importance_score, 'feature_values': feature_values})
        if feature_scores:
            feature_scores.sort(key=lambda x: x['importance'], reverse=True)
            return feature_scores[:3]
        return []

    async def _optimize_feature_globally(self, data: pd.DataFrame, target: pd.Series, feature_name: str, reduced_params: dict[str, list]) -> dict[str, Any]:
        """Optimize feature globally with reduced parameter space."""
        combinations = self._generate_param_combinations(reduced_params)
        feature_scores = []
        for params in combinations:
            feature_values = self._calculate_feature_with_params(data, feature_name, params)
            if feature_values is not None:
                importance_score = await self._calculate_importance_score(feature_values, target)
                feature_scores.append({'params': params, 'importance': importance_score, 'feature_values': feature_values})
        if feature_scores:
            feature_scores.sort(key=lambda x: x['importance'], reverse=True)
            return feature_scores[:3]
        return []

    async def _analyze_optimization_performance(self, results: dict[str, Any]) -> dict[str, Any]:
        """Analyze the performance of the optimization process."""
        performance_analysis = {'optimization_efficiency': {}, 'parameter_space_reduction': {}, 'meta_optimization_effectiveness': {}, 'multi_objective_balance': {}}
        for feature_name, space_data in results.get('parameter_space_optimization', {}).items():
            reduction_ratio = space_data.get('space_reduction_ratio', 1.0)
            performance_analysis['parameter_space_reduction'][feature_name] = {'reduction_ratio': reduction_ratio, 'efficiency_gain': 1.0 / reduction_ratio if reduction_ratio > 0 else 1.0}
        for feature_name, meta_data in results.get('meta_optimization_results', {}).items():
            best_value = meta_data.get('best_value', 0.0)
            n_trials = meta_data.get('n_trials', 0)
            performance_analysis['meta_optimization_effectiveness'][feature_name] = {'best_value': best_value, 'trials_efficiency': best_value / n_trials if n_trials > 0 else 0.0}
        return performance_analysis

    def get_enhanced_optimized_parameters(self, symbol: str, exchange: str, timeframe: str) -> dict[str, Any]:
        """Load enhanced optimized parameters."""
        filepath = Path(f'data/enhanced_feature_engineering_optimization/{exchange}_{symbol}_{timeframe}_enhanced_feature_optimization.json')
        if not filepath.exists():
            self.logger.warning(f'⚠️ No enhanced optimization results found for {symbol} on {exchange}')
            return {}
        try:
            with open(filepath) as f:
                results = json.load(f)
            return results.get('enhanced_optimizations', {})
        except Exception as e:
            self.logger.exception(f'❌ Error loading enhanced optimization results: {e}')
            return {}