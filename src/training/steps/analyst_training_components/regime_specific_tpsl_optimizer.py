from src.core.decorators import handles_errors
from src.core.domain import handle_specific_errors
from typing import Dict, List, Optional, Union, Any, Tuple
'Regime-Specific SL/TP Optimizer.\n\nThis module provides regime-specific optimization of Stop Loss (SL) and Take Profit (TP)\nparameters based on the current market context identified by the meta-labeling system.\n\nThe optimizer uses meta-label intensities and activations to determine optimal SL/TP levels\nfor each label-driven regime, considering success proxies from backtest simulations.\n'
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import optuna
import pandas as pd
import asyncio
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, failed, initialization_error, warning

class RegimeSpecificTPSLOptimizer:
    """Optimizes Take Profit (TP) and Stop Loss (SL) parameters based on HMM market regimes."

    This optimizer uses HMM market regimes to identify the current market state
    and then applies regime-specific optimization based on backtest performance.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the regime-specific TP/SL optimizer.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.logger = system_logger.getChild('RegimeSpecificTPSLOptimizer')
        self.print = self.logger.info
        self.logger.info('ℹ️ Meta-labeling system removed - using only HMM market regimes for labeling')
        self.regime_parameters = {'hmm_cluster_0': {'target_pct': 0.5, 'stop_pct': 0.2, 'risk_reward_ratio': 2.5, 'avg_duration_minutes': 45.0, 'success_rate': 7.0, 'frequency_score': 100.0}, 'hmm_cluster_1': {'target_pct': 0.4, 'stop_pct': 0.15, 'risk_reward_ratio': 2.67, 'avg_duration_minutes': 35.0, 'success_rate': 6.5, 'frequency_score': 80.0}, 'hmm_cluster_2': {'target_pct': 0.3, 'stop_pct': 0.2, 'risk_reward_ratio': 1.5, 'avg_duration_minutes': 60.0, 'success_rate': 7.5, 'frequency_score': 100.0}, 'hmm_cluster_3': {'target_pct': 0.6, 'stop_pct': 0.15, 'risk_reward_ratio': 4.0, 'avg_duration_minutes': 30.0, 'success_rate': 6.0, 'frequency_score': 70.0}, 'hmm_cluster_4': {'target_pct': 0.35, 'stop_pct': 0.2, 'risk_reward_ratio': 1.75, 'avg_duration_minutes': 25.0, 'success_rate': 5.5, 'frequency_score': 60.0}, 'hmm_cluster_5': {'target_pct': 0.5, 'stop_pct': 0.15, 'risk_reward_ratio': 3.33, 'avg_duration_minutes': 20.0, 'success_rate': 5.5, 'frequency_score': 70.0}, 'hmm_cluster_6': {'target_pct': 0.25, 'stop_pct': 0.2, 'risk_reward_ratio': 1.25, 'avg_duration_minutes': 90.0, 'success_rate': 6.0, 'frequency_score': 90.0}, 'hmm_cluster_7': {'target_pct': 0.5, 'stop_pct': 0.25, 'risk_reward_ratio': 2.0, 'avg_duration_minutes': 35.0, 'success_rate': 5.8, 'frequency_score': 70.0}, 'VOLATILE': {'target_pct': 0.6, 'stop_pct': 0.4, 'risk_reward_ratio': 1.5, 'avg_duration_minutes': 45.0, 'success_rate': 6.0, 'frequency_score': 100.0}, 'SIDEWAYS_RANGE': {'target_pct': 0.5, 'stop_pct': 0.3, 'risk_reward_ratio': 1.67, 'avg_duration_minutes': 67.4, 'success_rate': 7.81, 'frequency_score': 100.0}, 'DEFAULT': {'target_pct': 0.4, 'stop_pct': 0.2, 'risk_reward_ratio': 2.0, 'avg_duration_minutes': 40.0, 'success_rate': 6.5, 'frequency_score': 100.0}}
        self.optimization_config = config.get('regime_specific_tpsl_optimizer', {})
        self.n_trials = self.optimization_config.get('n_trials', 100)
        self.min_trades = self.optimization_config.get('min_trades', 20)
        self.optimization_metric = self.optimization_config.get('optimization_metric', 'sharpe_ratio')
        self.candidate_labels: list[str] = self.optimization_config.get('candidate_labels', ['STRONG_TREND_CONTINUATION', 'EXHAUSTION_REVERSAL', 'RANGE_MEAN_REVERSION', 'BREAKOUT_SUCCESS', 'BREAKOUT_FAILURE', 'MOMENTUM_IGNITION', 'VOLATILITY_COMPRESSION', 'VOLATILITY_EXPANSION', 'SR_TOUCH', 'SR_BOUNCE', 'SR_BREAK', 'IGNITION_BAR'])
        self.analysis_timeframe: str = self.optimization_config.get('analysis_timeframe', '30m')
        self.model_dir = os.path.join(CONFIG['CHECKPOINT_DIR'], 'regime_tpsl_models')
        if 'SR_TOUCH' in self.regime_parameters:
            self.regime_parameters['SR_BOUNCE'] = self.regime_parameters['SR_TOUCH']
        os.makedirs(self.model_dir, exist_ok=True)
        self.optimization_results: dict[str, dict[str, Any]] = {}
        self.last_optimization_time: datetime | None = None

    @handle_specific_errors(error_handlers={ValueError: (False, 'Invalid regime-specific TP/SL optimization configuration'), AttributeError: (False, 'Missing required optimization parameters')}, default_return=False, context='regime-specific TP/SL optimizer initialization')
    async def initialize(self) -> bool:
        """Initialize the regime-specific TP/SL optimizer.

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            self.logger.info('Initializing Regime-Specific TP/SL Optimizer (Meta-Label)...')
            if not await self._initialize_meta_label_system():
                self.print(failed('Failed to initialize Meta-Labeling system'))
                return False
            await self._load_optimization_results()
            self.logger.info('✅ Regime-Specific TP/SL Optimizer initialized successfully')
            return True
        except Exception as e:
            self.logger.exception(f'❌ Failed to initialize Regime-Specific TP/SL Optimizer: {e}')
            return False

    async def _initialize_meta_label_system(self) -> bool:
        """Initialize the MetaLabelingSystem.

        Returns:
            bool: True if initialization successful, False otherwise

        """
        try:
            ok = await self.meta_labeling_system.initialize()
            if ok:
                self.logger.info('✅ Meta-Labeling system initialized for regime identification')
                return True
            self.logger.warning('Meta-Labeling system failed to initialize')
            return False
        except Exception as e:
            self.print(initialization_error(f'Error initializing Meta-Labeling system: {e}'))
            return False

    async def _load_optimization_results(self) -> None:
        """Load existing optimization results from disk."""
        try:
            results_file = os.path.join(self.model_dir, 'optimization_results.json')
            if os.path.exists(results_file):
                import json
                with open(results_file) as f:
                    self.optimization_results = json.load(f)
                    self.logger.info(f'✅ Loaded {len(self.optimization_results)} regime optimization results')
        except Exception as e:
            self.print(warning(f'Could not load optimization results: {e}'))

    async def _save_optimization_results(self) -> None:
        """Save optimization results to disk."""
        try:
            results_file = os.path.join(self.model_dir, 'optimization_results.json')
            import json
            with open(results_file, 'w') as f:
                json.dump(self.optimization_results, f, indent=2, default=str)
            self.logger.info('✅ Saved optimization results')
        except Exception as e:
            self.print(failed(f'Failed to save optimization results: {e}'))

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='regime identification')
    async def identify_current_regime(self, current_data: pd.DataFrame) -> tuple[str, float, dict[str, Any]]:
        """Identify the current dominant meta-label driven market regime.

        Args:
            current_data: Current market OHLCV data

        Returns: Tuple of (regime_label, confidence, additional_info)

        """
        try:
            if not getattr(self.meta_labeling_system, 'is_initialized', False):
                self.print(warning('Meta-Labeling system not initialized, using default regime'))
                return ('SIDEWAYS_RANGE', 0.5, {'method': 'default'})
            labels = await self.meta_labeling_system.generate_analyst_labels(price_data=current_data, volume_data=current_data, timeframe=self.analysis_timeframe)
            intensities: dict[str, float] = {}
            actives: dict[str, int] = {}
            for label in self.candidate_labels:
                intensities[label] = float(labels.get(f'intensity_{label}', 0.0))
                actives[label] = int(labels.get(f'active_{label}', labels.get(label, 0)))
            best_label = max(self.candidate_labels, key=lambda k: (intensities.get(k, 0.0), actives.get(k, 0)), default='SIDEWAYS_RANGE')
            confidence = float(intensities.get(best_label, 0.0))
            top3 = sorted(((k, intensities.get(k, 0.0)) for k in self.candidate_labels), key=lambda x: x[1], reverse=True)[:3]
            self.logger.info({'msg': 'Identified label-driven regime', 'regime': best_label, 'confidence': round(confidence, 3), 'top3': [(k, round(v, 3)) for k, v in top3], 'timeframe': self.analysis_timeframe})
            return (best_label, confidence, {'method': 'meta_labeling', 'timeframe': self.analysis_timeframe, 'top3': top3, 'actives': {k: actives.get(k, 0) for k in self.candidate_labels}})
        except Exception as e:
            self.print(error(f'Error identifying regime: {e}'))
            return ('SIDEWAYS_RANGE', 0.5, {'method': 'fallback', 'error': str(e)})

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='regime-specific TP/SL optimization')
    async def optimize_tpsl_for_regime(self, regime: str, historical_data: pd.DataFrame, current_data: pd.DataFrame) -> dict[str, Any]:
        """Optimize TP/SL parameters for a specific label-driven market regime.

        Args:
            regime: Regime/meta-label to optimize for
            historical_data: Historical data for optimization
            current_data: Current market data

        Returns:
            Dictionary with optimized TP/SL parameters

        """
        try:
            self.logger.info(f'🎯 Optimizing TP/SL for regime: {regime}')
            base_params = self.regime_parameters.get(regime, self.regime_parameters['SIDEWAYS_RANGE'])
            study = optuna.create_study(direction='maximize', study_name=f'tpsl_optimization_{regime}')

            def objective(trial: Any) -> None:
                return self._evaluate_tpsl_parameters(trial, regime, historical_data, base_params)
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
            best_params = study.best_params
            best_value = study.best_value
            optimized_params = {**base_params, **best_params, 'optimization_score': best_value, 'optimization_trials': self.n_trials, 'optimization_time': datetime.now().isoformat()}
            self.optimization_results[regime] = optimized_params
            await self._save_optimization_results()
            self.logger.info(f'✅ Optimized TP/SL for {regime}: {best_params}')
            return optimized_params
        except Exception as e:
            self.print(error(f'Error optimizing TP/SL for regime {regime}: {e}'))
            return self.regime_parameters.get(regime, self.regime_parameters['SIDEWAYS_RANGE'])

    def _evaluate_tpsl_parameters(self, trial: optuna.Trial, regime: str, historical_data: pd.DataFrame, base_params: dict[str, Any]) -> float:
        """Evaluate TP/SL parameters using backtesting simulation.

        Args:
            trial: Optuna trial object
            regime: Market regime
            historical_data: Historical data for backtesting
            base_params: Base parameters for the regime

        Returns:
            float: Optimization score (higher is better)

        """
        try:
            target_pct = trial.suggest_float('target_pct', base_params['target_pct'] * 0.5, base_params['target_pct'] * 1.5)
            stop_pct = trial.suggest_float('stop_pct', base_params['stop_pct'] * 0.5, base_params['stop_pct'] * 1.5)
            if target_pct <= stop_pct:
                return -1.0
            trades = self._simulate_trades(historical_data, target_pct, stop_pct, regime)
            if len(trades) < self.min_trades:
                return -1.0
            returns = [trade['return'] for trade in trades]
            total_return = sum(returns)
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-08)
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            if self.optimization_metric == 'sharpe_ratio':
                score = sharpe_ratio
            elif self.optimization_metric == 'total_return':
                score = total_return
            elif self.optimization_metric == 'win_rate':
                score = win_rate
            else:
                score = sharpe_ratio * 0.4 + total_return * 0.3 + win_rate * 0.3
            return score
        except Exception as e:
            self.print(error(f'Error in parameter evaluation: {e}'))
            return -1.0

    def _simulate_trades(self, data: pd.DataFrame, target_pct: float, stop_pct: float, regime: str) -> list[dict[str, Any]]:
        """Simulate trades using given TP/SL parameters.

        Args:
            data: Historical price data
            target_pct: Take profit percentage
            stop_pct: Stop loss percentage
            regime: Market regime

        Returns:
            List of trade dictionaries

        """
        trades = []
        position_open = False
        entry_price = 0.0
        entry_time = None
        for i in range(1, len(data)):
            current_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            if not position_open:
                if data.iloc[i]['close'] > data.iloc[i - 1]['close']:
                    position_open = True
                    entry_price = current_price
                    entry_time = data.index[i]
            elif high_price >= entry_price * (1 + target_pct):
                trades.append({'entry_time': entry_time, 'exit_time': data.index[i], 'entry_price': entry_price, 'exit_price': entry_price * (1 + target_pct), 'return': target_pct, 'type': 'TP'})
                position_open = False
            elif low_price <= entry_price * (1 - stop_pct):
                trades.append({'entry_time': entry_time, 'exit_time': data.index[i], 'entry_price': entry_price, 'exit_price': entry_price * (1 - stop_pct), 'return': -stop_pct, 'type': 'SL'})
                position_open = False
        return trades

    @handles_errors(exceptions=(ValueError, AttributeError), default_return=None, context='regime-specific TP/SL prediction')
    async def get_optimized_tpsl(self, current_data: pd.DataFrame, historical_data: pd.DataFrame, force_optimization: bool=False) -> dict[str, Any]:
        """Get optimized TP/SL parameters for the current label-driven market regime.

        Args:
            current_data: Current market data (OHLCV)
            historical_data: Historical data for optimization
            force_optimization: Force re-optimization even if cached

        Returns:
            Dictionary with optimized TP/SL parameters

        """
        try:
            regime, confidence, regime_info = await self.identify_current_regime(current_data)
            if not force_optimization and regime in self.optimization_results:
                cached_params = self.optimization_results[regime]
                self.logger.info(f'Using cached TP/SL parameters for {regime}')
                return {**cached_params, 'regime': regime, 'confidence': confidence, 'regime_info': regime_info}
            optimized_params = await self.optimize_tpsl_for_regime(regime, historical_data, current_data)
            return {**optimized_params, 'regime': regime, 'confidence': confidence, 'regime_info': regime_info}
        except Exception as e:
            self.print(error(f'Error getting optimized TP/SL: {e}'))
            return {**self.regime_parameters['SIDEWAYS_RANGE'], 'regime': 'SIDEWAYS_RANGE', 'confidence': 0.5, 'regime_info': {'method': 'fallback', 'error': str(e)}}

    def get_regime_statistics(self) -> dict[str, Any]:
        """Get statistics about regime-specific TP/SL optimization.

        Returns:
            Dictionary with optimization statistics

        """
        return {'optimized_regimes': list(self.optimization_results.keys()), 'total_optimizations': len(self.optimization_results), 'last_optimization_time': self.last_optimization_time, 'regime_parameters': self.regime_parameters}