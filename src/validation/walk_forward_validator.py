"""Walk-Forward Validation System for preventing overfitting."""
import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import pandas as pd
from src.utils.common_operations import ensure_directory, safe_json_dump
from src.utils.logger import system_logger
from typing import Dict, List, Optional, Union, Any, Tuple
logger = system_logger.getChild('WalkForwardValidator')

@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {'window_id': self.window_id, 'train_start': self.train_start.isoformat(), 'train_end': self.train_end.isoformat(), 'test_start': self.test_start.isoformat(), 'test_end': self.test_end.isoformat(), 'train_days': (self.train_end - self.train_start).days, 'test_days': (self.test_end - self.test_start).days}

class WalkForwardValidator:
    """Implements walk-forward validation for trading strategies."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('WalkForwardValidator')
        self.train_period_days = config.get('train_period_days', 365)
        self.test_period_days = config.get('test_period_days', 30)
        self.step_days = config.get('step_days', 30)
        self.min_train_samples = config.get('min_train_samples', 1000)
        self.regime_aware = config.get('regime_aware', True)
        self.min_samples_per_regime = config.get('min_samples_per_regime', 500)
        self.adaptive_windows = config.get('adaptive_windows', True)
        self.volatility_threshold = config.get('volatility_threshold', 0.03)
        self.max_acceptable_degradation = config.get('max_acceptable_degradation', 0.3)
        self.min_out_sample_sharpe = config.get('min_out_sample_sharpe', 0.5)
        self.results_dir = Path(config.get('results_dir', 'validation_results'))
        ensure_directory(self.results_dir)

    def generate_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """Generate walk-forward validation windows."""
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp')
        start_date = data.index.min()
        end_date = data.index.max()
        windows = []
        window_id = 0
        train_end = start_date + timedelta(days=self.train_period_days)
        while train_end + timedelta(days=self.test_period_days) <= end_date:
            train_start = start_date
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_period_days)
            if self.adaptive_windows:
                window_params = self._adjust_window_for_volatility(data, train_start, train_end)
                if window_params:
                    train_start = window_params['train_start']
                    test_end = window_params['test_end']
            window = WalkForwardWindow(train_start=train_start, train_end=train_end, test_start=test_start, test_end=test_end, window_id=window_id)
            windows.append(window)
            train_end += timedelta(days=self.step_days)
            window_id += 1
        self.logger.info(f'Generated {len(windows)} walk-forward windows')
        return windows

    def _adjust_window_for_volatility(self, data: pd.DataFrame, train_start: datetime, train_end: datetime) -> Optional[Dict[str, datetime]]:
        """Adjust window size based on market volatility."""
        train_data = data[train_start:train_end]
        returns = train_data['close'].pct_change().dropna()
        volatility = returns.std()
        if volatility > self.volatility_threshold:
            new_train_days = int(self.train_period_days * 0.5)
            new_test_days = int(self.test_period_days * 0.5)
        else:
            return None
        return {'train_start': train_end - timedelta(days=new_train_days), 'test_end': train_end + timedelta(days=new_test_days)}

    async def validate_model(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: Optional[np.ndarray]=None) -> Dict[str, Any]:
        """Run walk-forward validation on a model."""
        self.logger.info('Starting walk-forward validation...')
        windows = self.generate_walk_forward_windows(data)
        if self.regime_aware and regime_labels is not None:
            results = await self._validate_regime_aware(model_trainer, data, regime_labels, windows)
        else:
            results = await self._validate_standard(model_trainer, data, windows)
        analysis = self._analyze_validation_results(results)
        self._save_validation_results(results, analysis)
        return {'windows': [w.to_dict() for w in windows], 'results': results, 'analysis': analysis}

    async def _validate_standard(self, model_trainer: Callable, data: pd.DataFrame, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Standard walk-forward validation."""
        results = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            for window in windows:
                future = executor.submit(self._validate_single_window, model_trainer, data, window)
                futures.append((window, future))
            for window, future in futures:
                try:
                    result = future.result(timeout=3600)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f'Window {window.window_id} failed: {e}')
                    results.append({'window': window.to_dict(), 'error': str(e), 'success': False})
        return results

    def _validate_single_window(self, model_trainer: Callable, data: pd.DataFrame, window: WalkForwardWindow) -> Dict[str, Any]:
        """Validate a single window."""
        train_data = data[window.train_start:window.train_end]
        test_data = data[window.test_start:window.test_end]
        if len(train_data) < self.min_train_samples:
            return {'window': window.to_dict(), 'error': 'Insufficient training samples', 'success': False}
        try:
            model = model_trainer(train_data)
            train_predictions = model.predict(train_data)
            test_predictions = model.predict(test_data)
            train_metrics = self._calculate_metrics(train_data, train_predictions, 'train')
            test_metrics = self._calculate_metrics(test_data, test_predictions, 'test')
            degradation = self._calculate_degradation(train_metrics, test_metrics)
            return {'window': window.to_dict(), 'train_metrics': train_metrics, 'test_metrics': test_metrics, 'degradation': degradation, 'model_params': model.get_params() if hasattr(model, 'get_params') else {}, 'success': True}
        except Exception as e:
            return {'window': window.to_dict(), 'error': str(e), 'success': False}

    async def _validate_regime_aware(self, model_trainer: Callable, data: pd.DataFrame, regime_labels: np.ndarray, windows: List[WalkForwardWindow]) -> List[Dict[str, Any]]:
        """Regime-aware walk-forward validation."""
        results = []
        for window in windows:
            window_results = {'window': window.to_dict(), 'regime_results': {}, 'success': True}
            train_mask = (data.index >= window.train_start) & (data.index <= window.train_end)
            test_mask = (data.index >= window.test_start) & (data.index <= window.test_end)
            train_data = data[train_mask]
            test_data = data[test_mask]
            train_regimes = regime_labels[train_mask]
            test_regimes = regime_labels[test_mask]
            for regime in ['bull', 'bear', 'sideways']:
                regime_result = await self._validate_regime_window(model_trainer, train_data, test_data, train_regimes, test_regimes, regime)
                window_results['regime_results'][regime] = regime_result
                if not regime_result['success']:
                    window_results['success'] = False
            results.append(window_results)
        return results

    async def _validate_regime_window(self, model_trainer: Callable, train_data: pd.DataFrame, test_data: pd.DataFrame, train_regimes: np.ndarray, test_regimes: np.ndarray, regime: str) -> Dict[str, Any]:
        """Validate a single regime within a window."""
        regime_map = {'bear': 0, 'sideways': 1, 'bull': 2}
        regime_num = regime_map.get(regime, 1)
        train_regime_data = train_data[train_regimes == regime_num]
        test_regime_data = test_data[test_regimes == regime_num]
        if len(train_regime_data) < self.min_samples_per_regime:
            return {'regime': regime, 'error': f'Insufficient {regime} training samples: {len(train_regime_data)}', 'success': False}
        if len(test_regime_data) < 10:
            return {'regime': regime, 'error': f'Insufficient {regime} test samples: {len(test_regime_data)}', 'success': False}
        try:
            model = model_trainer(train_regime_data, regime=regime)
            train_predictions = model.predict(train_regime_data)
            test_predictions = model.predict(test_regime_data)
            train_metrics = self._calculate_metrics(train_regime_data, train_predictions, f'train_{regime}')
            test_metrics = self._calculate_metrics(test_regime_data, test_predictions, f'test_{regime}')
            degradation = self._calculate_degradation(train_metrics, test_metrics)
            return {'regime': regime, 'train_samples': len(train_regime_data), 'test_samples': len(test_regime_data), 'train_metrics': train_metrics, 'test_metrics': test_metrics, 'degradation': degradation, 'success': True}
        except Exception as e:
            return {'regime': regime, 'error': str(e), 'success': False}

    def _calculate_metrics(self, data: pd.DataFrame, predictions: np.ndarray, prefix: str) -> Dict[str, float]:
        """Calculate performance metrics."""
        if 'returns' not in data.columns:
            data['returns'] = data['close'].pct_change()
        strategy_returns = data['returns'].values[1:] * predictions[:-1]
        metrics = {f'{prefix}_sharpe': self._calculate_sharpe(strategy_returns), f'{prefix}_sortino': self._calculate_sortino(strategy_returns), f'{prefix}_max_drawdown': self._calculate_max_drawdown(strategy_returns), f'{prefix}_win_rate': (strategy_returns > 0).mean(), f'{prefix}_total_return': strategy_returns.sum(), f'{prefix}_volatility': strategy_returns.std()}
        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = downside_returns.std()
        if downside_std == 0:
            return float('inf')
        return returns.mean() / downside_std * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_degradation(self, train_metrics: Dict[str, float], test_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate performance degradation from train to test."""
        degradation = {}
        train_sharpe = next((v for k, v in train_metrics.items() if 'sharpe' in k), 0)
        test_sharpe = next((v for k, v in test_metrics.items() if 'sharpe' in k), 0)
        if train_sharpe != 0:
            degradation['sharpe_degradation'] = (train_sharpe - test_sharpe) / abs(train_sharpe)
        else:
            degradation['sharpe_degradation'] = 0
        train_wr = next((v for k, v in train_metrics.items() if 'win_rate' in k), 0)
        test_wr = next((v for k, v in test_metrics.items() if 'win_rate' in k), 0)
        if train_wr != 0:
            degradation['win_rate_degradation'] = (train_wr - test_wr) / train_wr
        else:
            degradation['win_rate_degradation'] = 0
        degradation['overall'] = (degradation['sharpe_degradation'] + degradation['win_rate_degradation']) / 2
        degradation['potential_overfitting'] = degradation['overall'] > self.max_acceptable_degradation
        return degradation

    def _analyze_validation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward validation results."""
        analysis = {'total_windows': len(results), 'successful_windows': sum((1 for r in results if r.get('success', False))), 'average_degradation': 0, 'overfitting_windows': 0, 'regime_analysis': {} if self.regime_aware else None}
        degradations = []
        for result in results:
            if result.get('success', False):
                if 'degradation' in result:
                    degradations.append(result['degradation']['overall'])
                    if result['degradation']['potential_overfitting']:
                        analysis['overfitting_windows'] += 1
        if degradations:
            analysis['average_degradation'] = np.mean(degradations)
        if self.regime_aware:
            for regime in ['bull', 'bear', 'sideways']:
                regime_degradations = []
                for result in results:
                    if 'regime_results' in result:
                        regime_result = result['regime_results'].get(regime, {})
                        if regime_result.get('success', False):
                            regime_degradations.append(regime_result['degradation']['overall'])
                if regime_degradations:
                    analysis['regime_analysis'][regime] = {'avg_degradation': np.mean(regime_degradations), 'windows_analyzed': len(regime_degradations)}
        analysis['validation_passed'] = analysis['average_degradation'] <= self.max_acceptable_degradation and analysis['overfitting_windows'] / max(analysis['successful_windows'], 1) < 0.3
        return analysis

    def _save_validation_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
        """Save validation results to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.results_dir / f'validation_results_{timestamp}.json'
        safe_json_dump({'config': self.config, 'results': results, 'analysis': analysis, 'timestamp': timestamp}, results_path)
        summary_path = self.results_dir / f'validation_summary_{timestamp}.json'
        safe_json_dump(analysis, summary_path)
        self.logger.info(f'Saved validation results to {results_path}')
        self.logger.info(f'Saved validation summary to {summary_path}')

async def example_model_trainer(data: pd.DataFrame, regime: Optional[str]=None) -> Any:
    """Example model trainer for testing walk-forward validation."""

    class DummyModel:

        def __init__(self, regime: Any=None) -> None:
            self.regime = regime

        def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> None:
            return np.random.choice([-1, 0, 1], size=len(data))

        def get_params(self) -> Any:
            return {'regime': self.regime}
    return DummyModel(regime)
if __name__ == '__main__':

    async def main() -> None:
        config = {'train_period_days': 365, 'test_period_days': 30, 'step_days': 30, 'regime_aware': True, 'adaptive_windows': True}
        validator = WalkForwardValidator(config)
        print('Walk-forward validation system initialized')
    asyncio.run(main())