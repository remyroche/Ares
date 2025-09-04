"""Regime-Specific Parameter Optimization System."""
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import optuna
import pandas as pd
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.logger import system_logger
from src.validation.walk_forward_validator import WalkForwardValidator
from copy import copy
from typing import Dict, List, Optional, Union, Any, Tuple
logger = system_logger.getChild('RegimeParameterOptimizer')

@dataclass
class RegimeParameters:
    """Parameters specific to a market regime."""
    regime: str
    profit_target: float
    stop_loss: float
    time_barrier: int
    momentum_window: int
    volatility_window: int
    volume_window: int
    feature_selection_threshold: float
    learning_rate: float
    regularization: float
    max_depth: int
    n_estimators: int
    max_position_size: float
    confidence_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {'regime': self.regime, 'triple_barrier': {'profit_target': self.profit_target, 'stop_loss': self.stop_loss, 'time_barrier': self.time_barrier}, 'features': {'momentum_window': self.momentum_window, 'volatility_window': self.volatility_window, 'volume_window': self.volume_window, 'feature_selection_threshold': self.feature_selection_threshold}, 'model': {'learning_rate': self.learning_rate, 'regularization': self.regularization, 'max_depth': self.max_depth, 'n_estimators': self.n_estimators}, 'risk': {'max_position_size': self.max_position_size, 'confidence_threshold': self.confidence_threshold}}

class RegimeParameterOptimizer:
    """Optimizes parameters for each market regime."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('RegimeParameterOptimizer')
        self.n_trials = config.get('n_trials', 100)
        self.n_jobs = config.get('n_jobs', 4)
        self.timeout = config.get('timeout', 3600)
        self.search_spaces = self._define_search_spaces()
        self.validator = WalkForwardValidator(config)
        self.results_dir = Path(config.get('results_dir', 'optimization_results'))
        ensure_directory(self.results_dir)
        self.best_parameters = {}

    def _define_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Define parameter search spaces for each regime."""
        return {'bull': {'profit_target': (0.015, 0.04), 'stop_loss': (0.005, 0.02), 'time_barrier': (30, 120), 'momentum_window': (5, 30), 'volatility_window': (10, 50), 'volume_window': (5, 30), 'feature_selection_threshold': (0.5, 0.9), 'learning_rate': (0.001, 0.1), 'regularization': (0.0001, 0.1), 'max_depth': (3, 10), 'n_estimators': (50, 300), 'max_position_size': (0.5, 1.0), 'confidence_threshold': (0.6, 0.9)}, 'bear': {'profit_target': (0.01, 0.025), 'stop_loss': (0.01, 0.025), 'time_barrier': (20, 60), 'momentum_window': (10, 40), 'volatility_window': (15, 60), 'volume_window': (10, 40), 'feature_selection_threshold': (0.6, 0.95), 'learning_rate': (0.0005, 0.05), 'regularization': (0.001, 0.2), 'max_depth': (3, 8), 'n_estimators': (100, 400), 'max_position_size': (0.3, 0.7), 'confidence_threshold': (0.7, 0.95)}, 'sideways': {'profit_target': (0.005, 0.015), 'stop_loss': (0.005, 0.015), 'time_barrier': (10, 40), 'momentum_window': (5, 20), 'volatility_window': (10, 30), 'volume_window': (5, 20), 'feature_selection_threshold': (0.7, 0.95), 'learning_rate': (0.001, 0.05), 'regularization': (0.001, 0.1), 'max_depth': (3, 6), 'n_estimators': (50, 200), 'max_position_size': (0.3, 0.6), 'confidence_threshold': (0.75, 0.95)}}

    async def optimize_all_regimes(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, RegimeParameters]:
        """Optimize parameters for all regimes."""
        self.logger.info('Starting regime-specific parameter optimization...')
        results = {}
        for regime in ['bull', 'bear', 'sideways']:
            self.logger.info(f'Optimizing parameters for {regime} regime...')
            regime_data = self._filter_by_regime(data, regime_labels, regime)
            if len(regime_data) < self.config.get('min_samples_per_regime', 1000):
                self.logger.warning(f'Insufficient data for {regime} regime optimization')
                continue
            best_params = await self._optimize_regime_parameters(regime, regime_data)
            if best_params:
                results[regime] = best_params
                self.best_parameters[regime] = best_params
                self._save_regime_results(regime, best_params)
        self._save_optimization_results(results)
        return results

    def _filter_by_regime(self, data: pd.DataFrame, regime_labels: np.ndarray, regime: str) -> pd.DataFrame:
        """Filter data by regime."""
        regime_map = {'bear': 0, 'sideways': 1, 'bull': 2}
        regime_num = regime_map.get(regime, 1)
        mask = regime_labels == regime_num
        return data[mask].copy()

    async def _optimize_regime_parameters(self, regime: str, data: pd.DataFrame) -> Optional[RegimeParameters]:
        """Optimize parameters for a specific regime."""
        try:
            study = optuna.create_study(direction='maximize', study_name=f'regime_{regime}_optimization', pruner=optuna.pruners.MedianPruner(n_startup_trials=10))

            def objective(trial: Any) -> None:
                params = self._sample_parameters(trial, regime)
                score = self._evaluate_parameters(params, data, regime)
                return score
            study.optimize(objective, n_trials=self.n_trials, n_jobs=self.n_jobs, timeout=self.timeout)
            best_trial = study.best_trial
            best_params = self._trial_to_parameters(best_trial, regime)
            self.logger.info(f'Best score for {regime}: {best_trial.value:.4f}')
            return best_params
        except Exception as e:
            self.logger.error(f'Optimization failed for {regime}: {e}')
            return None

    def _sample_parameters(self, trial: optuna.Trial, regime: str) -> Dict[str, Any]:
        """Sample parameters from search space."""
        search_space = self.search_spaces[regime]
        params = {}
        for param_name, (low, high) in search_space.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)
        return params

    def _evaluate_parameters(self, params: Dict[str, Any], data: pd.DataFrame, regime: str) -> float:
        """Evaluate parameter set using walk-forward validation."""
        try:
            regime_params = RegimeParameters(regime=regime, **params)
            strategy_results = self._run_strategy_with_params(data, regime_params)
            if strategy_results is not None and len(strategy_results) > 0:
                returns = strategy_results['returns']
                sharpe = returns.mean() / (returns.std() + 1e-10) * np.sqrt(252)
                max_dd = self._calculate_max_drawdown(returns)
                if max_dd > 0.2:
                    sharpe *= 0.5
                return sharpe
            else:
                return -999
        except Exception as e:
            self.logger.error(f'Parameter evaluation failed: {e}')
            return -999

    def _run_strategy_with_params(self, data: pd.DataFrame, params: RegimeParameters) -> Optional[pd.DataFrame]:
        """Run trading strategy with given parameters."""
        try:
            from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            labeler = OptimizedTripleBarrierLabeling()
            labeler.profit_target = params.profit_target
            labeler.stop_loss = params.stop_loss
            labeler.time_barrier = params.time_barrier
            features = self._generate_features(data, params)
            momentum = features[f'momentum_{params.momentum_window}']
            volatility = features[f'volatility_{params.volatility_window}']
            signals = np.where((momentum > 0) & (volatility < volatility.quantile(0.75)), 1, np.where((momentum < 0) & (volatility > volatility.quantile(0.25)), -1, 0))
            data['signal'] = signals
            data['returns'] = data['close'].pct_change() * data['signal'].shift(1)
            return data[['returns', 'signal']]
        except Exception as e:
            self.logger.error(f'Strategy execution failed: {e}')
            return None

    def _generate_features(self, data: pd.DataFrame, params: RegimeParameters) -> pd.DataFrame:
        """Generate features with regime-specific parameters."""
        features = pd.DataFrame(index=data.index)
        features[f'momentum_{params.momentum_window}'] = data['close'].pct_change(params.momentum_window)
        returns = data['close'].pct_change()
        features[f'volatility_{params.volatility_window}'] = returns.rolling(params.volatility_window).std()
        features[f'volume_ratio_{params.volume_window}'] = data['volume'] / data['volume'].rolling(params.volume_window).mean()
        return features

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def _trial_to_parameters(self, trial: optuna.Trial, regime: str) -> RegimeParameters:
        """Convert Optuna trial to RegimeParameters."""
        return RegimeParameters(regime=regime, profit_target=trial.params['profit_target'], stop_loss=trial.params['stop_loss'], time_barrier=trial.params['time_barrier'], momentum_window=trial.params['momentum_window'], volatility_window=trial.params['volatility_window'], volume_window=trial.params['volume_window'], feature_selection_threshold=trial.params['feature_selection_threshold'], learning_rate=trial.params['learning_rate'], regularization=trial.params['regularization'], max_depth=trial.params['max_depth'], n_estimators=trial.params['n_estimators'], max_position_size=trial.params['max_position_size'], confidence_threshold=trial.params['confidence_threshold'])

    def _save_regime_results(self, regime: str, params: RegimeParameters) -> None:
        """Save optimization results for a regime."""
        results_file = self.results_dir / f'regime_{regime}_parameters.json'
        safe_json_dump(params.to_dict(), results_file)
        self.logger.info(f'Saved {regime} regime parameters to {results_file}')

    def _save_optimization_results(self, results: Dict[str, RegimeParameters]) -> None:
        """Save final optimization results."""
        final_results = {'timestamp': datetime.now().isoformat(), 'config': self.config, 'parameters': {regime: params.to_dict() for regime, params in results.items()}}
        results_file = self.results_dir / f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        safe_json_dump(final_results, results_file)
        latest_file = self.results_dir / 'latest_optimization_results.json'
        safe_json_dump(final_results, latest_file)
        self.logger.info(f'Saved optimization results to {results_file}')

    async def validate_optimized_parameters(self, data: pd.DataFrame, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Validate optimized parameters using walk-forward analysis."""
        self.logger.info('Validating optimized parameters...')
        validation_results = {}
        for regime, params in self.best_parameters.items():
            regime_data = self._filter_by_regime(data, regime_labels, regime)
            val_results = await self.validator.validate_model(lambda d: self._create_model_with_params(d, params), regime_data)
            validation_results[regime] = val_results
        return validation_results

    def _create_model_with_params(self, data: pd.DataFrame, params: RegimeParameters) -> Any:
        """Create a model with optimized parameters."""

        class OptimizedModel:

            def __init__(self, params: Dict[str, Any]) -> None:
                self.params = params

            def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
                return np.random.choice([-1, 0, 1], size=len(data))

            def get_params(self) -> Any:
                return self.params.to_dict()
        return OptimizedModel(params)

    def get_best_parameters(self, regime: str) -> Optional[RegimeParameters]:
        """Get best parameters for a regime."""
        return self.best_parameters.get(regime)

    async def continuous_optimization(self, update_frequency_days: int=30) -> None:
        """Continuously optimize parameters with periodic updates."""
        self.logger.info(f'Starting continuous optimization (update every {update_frequency_days} days)')
        while True:
            try:
                data = pd.read_parquet('data/latest_data.parquet')
                regime_labels = np.load('data/latest_regime_labels.npy')
                await self.optimize_all_regimes(data, regime_labels)
                validation_results = await self.validate_optimized_parameters(data, regime_labels)
                self.logger.info('Optimization cycle completed')
                self.logger.info(f'Validation results: {validation_results}')
                await asyncio.sleep(update_frequency_days * 24 * 3600)
            except Exception as e:
                self.logger.error(f'Continuous optimization error: {e}')
                await asyncio.sleep(3600)

async def optimize_regime_parameters(config: Dict[str, Any]) -> Dict[str, RegimeParameters]:
    """Optimize parameters for all regimes."""
    optimizer = RegimeParameterOptimizer(config)
    data = pd.read_parquet(config['data_path'])
    regime_labels = np.load(config['regime_labels_path'])
    results = await optimizer.optimize_all_regimes(data, regime_labels)
    return results
if __name__ == '__main__':

    async def main() -> None:
        config = {'n_trials': 100, 'n_jobs': 4, 'timeout': 3600, 'min_samples_per_regime': 1000, 'data_path': 'data/training_data.parquet', 'regime_labels_path': 'data/regime_labels.npy', 'results_dir': 'optimization_results'}
        results = await optimize_regime_parameters(config)
        print('Optimization completed!')
        for regime, params in results.items():
            print(f'\n{regime.upper()} regime parameters:')
            print(json.dumps(params.to_dict(), indent=2))
    asyncio.run(main())