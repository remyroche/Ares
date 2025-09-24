from typing import Dict, List, Optional, Union, Any, Tuple
from ...utils.logger import system_logger
"""
S/R Relevance Weight Optimizer

Optimizes the weights for the 5 relevance scoring factors:
1. Return magnitude
2. Touch count
3. Recency
4. Volume confirmation
5. Success rate
"""
import optuna
from datetime import datetime
import numpy as np
import pandas as pd
import logging
import time

class SRRelevanceOptimizer:
    """
    Optimizes relevance scoring weights for S/R levels using historical performance.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = system_logger.getChild('SRRelevanceOptimizer')
        self.config = config
        self.default_weights = {'return_magnitude': 0.3, 'touch_count': 0.2, 'recency': 0.2, 'volume_confirmation': 0.15, 'success_rate': 0.15}
        self.weight_bounds = {'return_magnitude': (0.1, 0.5), 'touch_count': (0.05, 0.3), 'recency': (0.05, 0.3), 'volume_confirmation': (0.05, 0.25), 'success_rate': (0.1, 0.4)}
        self.optimization_method = config.get('optimization_method', 'optuna')
        self.n_trials = config.get('n_trials', 100)
        self.validation_metric = config.get('validation_metric', 'sharpe_ratio')
        self.optimization_history = []
        self.best_weights = self.default_weights.copy()

    def optimize_weights(self, historical_data: pd.DataFrame, detected_sr_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize relevance weights using historical data.
        
        Args:
            historical_data: Market data with prices and returns
            detected_sr_levels: List of detected S/R levels with their features
            actual_outcomes: Actual price movements after S/R tests
            
        Returns:
            Optimized weights dictionary
        """
        self.logger.info('Starting S/R relevance weight optimization...')
        if self.optimization_method == 'optuna':
            return self._optimize_with_optuna(historical_data, detected_sr_levels, actual_outcomes)
        elif self.optimization_method == 'scipy':
            return self._optimize_with_scipy(historical_data, detected_sr_levels, actual_outcomes)
        else:
            return self._optimize_with_grid_search(historical_data, detected_sr_levels, actual_outcomes)

    def _optimize_with_optuna(self, historical_data: pd.DataFrame, detected_sr_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> Dict[str, float]:
        """Optimize using Optuna (Bayesian optimization)."""

        def objective(trial: Any) -> None:
            weights_raw = {}
            for factor, (min_w, max_w) in self.weight_bounds.items():
                weights_raw[factor] = trial.suggest_float(factor, min_w, max_w)
            total = sum(weights_raw.values())
            weights = {k: v / total for k, v in weights_raw.items()}
            performance = self._evaluate_weights(weights, historical_data, detected_sr_levels, actual_outcomes)
            return -performance
        study = optuna.create_study(direction='minimize', pruner = optuna.pruners.MedianPruner(n_startup_trials = 10))

        def callback(study: Any, trial: Any) -> None:
            if trial.number % 10 == 0:
                self.logger.info(f'Trial {trial.number}: Best value = {-study.best_value:.4f}')
        study.optimize(objective, n_trials = self.n_trials, callbacks=[callback])
        best_params = study.best_params
        total = sum(best_params.values())
        optimized_weights = {k: v / total for k, v in best_params.items()}
        self.logger.info(f'Optimization complete. Best performance: {-study.best_value:.4f}')
        self.logger.info(f'Optimized weights: {optimized_weights}')
        self.optimization_history.append({'timestamp': datetime.now(), 'method': 'optuna', 'best_performance': -study.best_value, 'weights': optimized_weights, 'n_trials': len(study.trials)})
        self.best_weights = optimized_weights
        return optimized_weights

    def _optimize_with_scipy(self, historical_data: pd.DataFrame, detected_sr_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> Dict[str, float]:
        """Optimize using SciPy differential evolution."""
        bounds = [self.weight_bounds[factor] for factor in self.default_weights.keys()]

        def objective(weights_array: List[Any]) -> None:
            weights_dict = dict(zip(self.default_weights.keys(), weights_array))
            total = sum(weights_dict.values())
            weights_norm = {k: v / total for k, v in weights_dict.items()}
            return -self._evaluate_weights(weights_norm, historical_data, detected_sr_levels, actual_outcomes)
        result = differential_evolution(objective, bounds, maxiter = self.n_trials // 10, popsize = 15, seed = 42)
        weights_array = result.x
        weights_dict = dict(zip(self.default_weights.keys(), weights_array))
        total = sum(weights_dict.values())
        optimized_weights = {k: v / total for k, v in weights_dict.items()}
        self.logger.info(f'Optimization complete. Best performance: {-result.fun:.4f}')
        self.logger.info(f'Optimized weights: {optimized_weights}')
        self.best_weights = optimized_weights
        return optimized_weights

    def _optimize_with_grid_search(self, historical_data: pd.DataFrame, detected_sr_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> Dict[str, float]:
        """Optimize using grid search (simple but thorough)."""
        grid_resolution = 5
        weight_options = np.linspace(0, 1, grid_resolution)
        best_performance = -np.inf
        best_weights = self.default_weights.copy()
        tested = 0
        for w1 in weight_options:
            for w2 in weight_options:
                for w3 in weight_options:
                    for w4 in weight_options:
                        w5 = 1 - (w1 + w2 + w3 + w4)
                        if w5 < 0 or w5 > 1:
                            continue
                        weights = {'return_magnitude': w1, 'touch_count': w2, 'recency': w3, 'volume_confirmation': w4, 'success_rate': w5}
                        if not self._check_weight_bounds(weights):
                            continue
                        performance = self._evaluate_weights(weights, historical_data, detected_sr_levels, actual_outcomes)
                        if performance > best_performance:
                            best_performance = performance
                            best_weights = weights.copy()
                        tested += 1
                        if tested % 100 == 0:
                            self.logger.info(f'Tested {tested} combinations...')
        self.logger.info(f'Grid search complete. Best performance: {best_performance:.4f}')
        self.logger.info(f'Optimized weights: {best_weights}')
        self.best_weights = best_weights
        return best_weights

    def _evaluate_weights(self, weights: Dict[str, float], historical_data: pd.DataFrame, detected_sr_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> float:
        """
        Evaluate a set of weights by scoring S/R levels and measuring prediction quality.
        """
        scored_levels = []
        for level in detected_sr_levels:
            score = self._calculate_weighted_score(level, weights)
            level_copy = level.copy()
            level_copy['relevance_score'] = score
            scored_levels.append(level_copy)
        scored_levels.sort(key = lambda x: x['relevance_score'], reverse = True)
        if self.validation_metric == 'accuracy':
            return self._evaluate_accuracy(scored_levels, actual_outcomes)
        elif self.validation_metric == 'profit':
            return self._evaluate_profit(scored_levels, actual_outcomes, historical_data)
        else:
            return self._evaluate_sharpe_ratio(scored_levels, actual_outcomes, historical_data)

    def _calculate_weighted_score(self, level: Dict[str, Any], weights: Dict[str, float]) -> float:
        """Calculate relevance score for a level using given weights."""
        scores = level.get('component_scores', {})
        weighted_score = weights['return_magnitude'] * scores.get('return_magnitude', 0) + weights['touch_count'] * scores.get('touch_count', 0) + weights['recency'] * scores.get('recency', 0) + weights['volume_confirmation'] * scores.get('volume_confirmation', 0) + weights['success_rate'] * scores.get('success_rate', 0)
        return weighted_score

    def _evaluate_accuracy(self, scored_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame) -> float:
        """Evaluate accuracy of high-relevance S/R levels."""
        n_top = max(1, len(scored_levels) // 5)
        top_levels = scored_levels[:n_top]
        if not top_levels:
            return 0.0
        correct_predictions = 0
        total_predictions = 0
        for level in top_levels:
            level_price = level['price']
            level_type = level['type']
            approaches = actual_outcomes[actual_outcomes['approach_price'].between(level_price * 0.995, level_price * 1.005)]
            for _, outcome in approaches.iterrows():
                total_predictions += 1
                if level_type == 'support' and outcome['bounced']:
                    correct_predictions += 1
                elif level_type == 'resistance' and (not outcome['broke_through']):
                    correct_predictions += 1
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return accuracy

    def _evaluate_profit(self, scored_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame, historical_data: pd.DataFrame) -> float:
        """Evaluate profit from trading high-relevance S/R levels."""
        n_top = max(1, len(scored_levels) // 5)
        top_levels = scored_levels[:n_top]
        total_return = 0.0
        n_trades = 0
        for level in top_levels:
            level_price = level['price']
            level_type = level['type']
            relevance = level['relevance_score']
            signals = self._generate_signals(level, historical_data)
            for signal in signals:
                position_size = relevance
                if level_type == 'support':
                    entry = signal['entry_price']
                    exit = signal['exit_price']
                    trade_return = (exit - entry) / entry * position_size
                else:
                    entry = signal['entry_price']
                    exit = signal['exit_price']
                    trade_return = (entry - exit) / entry * position_size
                total_return += trade_return
                n_trades += 1
        avg_return = total_return / n_trades if n_trades > 0 else 0.0
        return avg_return

    def _evaluate_sharpe_ratio(self, scored_levels: List[Dict[str, Any]], actual_outcomes: pd.DataFrame, historical_data: pd.DataFrame) -> float:
        """Evaluate Sharpe ratio of trading high-relevance S/R levels."""
        n_top = max(1, len(scored_levels) // 5)
        top_levels = scored_levels[:n_top]
        daily_returns = []
        for level in top_levels:
            signals = self._generate_signals(level, historical_data)
            for signal in signals:
                relevance = level['relevance_score']
                if level['type'] == 'support':
                    trade_return = (signal['exit_price'] - signal['entry_price']) / signal['entry_price']
                else:
                    trade_return = (signal['entry_price'] - signal['exit_price']) / signal['entry_price']
                weighted_return = trade_return * relevance
                daily_returns.append(weighted_return)
        if len(daily_returns) < 2:
            return 0.0
        returns_series = pd.Series(daily_returns)
        sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0.0
        return sharpe_ratio

    def _generate_signals(self, level: Dict[str, Any], historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate trading signals when price approaches S/R level."""
        signals = []
        level_price = level['price']
        tolerance = 0.002
        for i in range(1, len(historical_data) - 10):
            current_price = historical_data['close'].iloc[i]
            prev_price = historical_data['close'].iloc[i - 1]
            if abs(current_price - level_price) / level_price <= tolerance and abs(prev_price - level_price) / level_price > tolerance:
                entry_price = current_price
                future_prices = historical_data['close'].iloc[i:i + 10]
                if level['type'] == 'support':
                    exit_price = future_prices.max()
                else:
                    exit_price = future_prices.min()
                signals.append({'entry_price': entry_price, 'exit_price': exit_price, 'entry_index': i, 'level_price': level_price})
        return signals

    def _check_weight_bounds(self, weights: Dict[str, float]) -> bool:
        """Check if weights are within bounds."""
        for factor, weight in weights.items():
            min_bound, max_bound = self.weight_bounds[factor]
            if weight < min_bound or weight > max_bound:
                return False
        return True

    def dynamic_weight_adjustment(self, market_regime: str, volatility_percentile: float) -> Dict[str, float]:
        """
        Dynamically adjust weights based on market conditions.
        
        Args:
            market_regime: Current market regime (trending, ranging, volatile)
            volatility_percentile: Current volatility percentile (0-1)
            
        Returns:
            Adjusted weights
        """
        base_weights = self.best_weights.copy()
        if market_regime == 'trending':
            base_weights['recency'] *= 1.2
            base_weights['success_rate'] *= 1.3
            base_weights['touch_count'] *= 0.8
        elif market_regime == 'ranging':
            base_weights['touch_count'] *= 1.4
            base_weights['return_magnitude'] *= 1.2
            base_weights['recency'] *= 0.8
        elif market_regime == 'volatile':
            base_weights['volume_confirmation'] *= 1.5
            base_weights['return_magnitude'] *= 1.3
            base_weights['touch_count'] *= 0.7
        if volatility_percentile > 0.8:
            base_weights['volume_confirmation'] *= 1.2
            base_weights['return_magnitude'] *= 1.1
        elif volatility_percentile < 0.2:
            base_weights['touch_count'] *= 1.2
            base_weights['success_rate'] *= 1.1
        total = sum(base_weights.values())
        adjusted_weights = {k: v / total for k, v in base_weights.items()}
        return adjusted_weights

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get detailed optimization report."""
        if not self.optimization_history:
            return {'status': 'No optimization performed', 'best_weights': self.default_weights}
        latest = self.optimization_history[-1]
        performance_improvement = None
        if len(self.optimization_history) > 1:
            initial_performance = self.optimization_history[0].get('best_performance', 0)
            latest_performance = latest.get('best_performance', 0)
            if initial_performance > 0:
                performance_improvement = (latest_performance - initial_performance) / initial_performance
        report = {'timestamp': latest['timestamp'], 'method': latest['method'], 'n_trials': latest.get('n_trials', 0), 'best_performance': latest['best_performance'], 'performance_improvement': performance_improvement, 'optimized_weights': latest['weights'], 'default_weights': self.default_weights, 'weight_changes': {k: latest['weights'][k] - self.default_weights[k] for k in self.default_weights.keys()}, 'optimization_history': self.optimization_history}
        return report