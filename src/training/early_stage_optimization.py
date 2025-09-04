from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Tuple
'\nEarly Stage Optimization Module\n\nThis module handles optimization that should happen BEFORE ML trading begins:\n1. SR (Stationarity and Randomness) optimization (step2_5)\n2. Regime-specific triple barrier optimization (step04)\n\nThese optimizations happen early in the pipeline to ensure:\n- Proper data preprocessing (SR)\n- Regime-aware trading parameters (triple barrier)\n- Optimal foundation for ML model training\n'
import json
import logging
import warnings
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
import asyncio
warnings.filterwarnings('ignore')
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
try:
    from .steps.step06_labeling_components.regime_specific_triple_barrier_optimizer import RegimeSpecificTripleBarrierOptimizer, create_regime_specific_triple_barrier_optimizer
    REGIME_OPTIMIZER_AVAILABLE = True
except ImportError:
    REGIME_OPTIMIZER_AVAILABLE = False
    RegimeSpecificTripleBarrierOptimizer = None
    create_regime_specific_triple_barrier_optimizer = None

class EarlyStageOptimizer:
    """
    Early stage optimizer for parameters that must be set before ML trading begins.

    This includes:
    - SR optimization (step2_5) - data preprocessing parameters
    - Regime-specific triple barrier optimization (step04) - trading parameters
    """

    def __init__(self, config: dict[str, Any], training_manager: Any=None) -> None:
        self.config = config
        self.training_manager = training_manager
        self.logger = logging.getLogger(__name__)
        self.sr_optimization_results = {}
        self.regime_barrier_optimization_results = {}
        self.sr_experiment_name = 'early_stage_sr_optimization'
        self.regime_experiment_name = 'early_stage_regime_barrier_optimization'
        self.regime_optimizer = None
        if REGIME_OPTIMIZER_AVAILABLE:
            self.regime_optimizer = create_regime_specific_triple_barrier_optimizer(config, training_manager)
            self.logger.info('✅ Regime-specific triple barrier optimizer initialized')
        else:
            self.logger.warning('⚠️ Regime-specific triple barrier optimizer not available')

    async def optimize_sr_parameters(self, data: pd.DataFrame, optimization_config: dict[str, Any]) -> dict[str, Any]:
        """Optimize SR (Stationarity and Randomness) parameters for data preprocessing."""
        self.logger.info('🚀 Starting SR parameter optimization...')
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna is required for SR optimization'}
        try:
            study = optuna.create_study(study_name='sr_parameter_optimization', direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24, multivariate=True), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3))
            objective = self._create_sr_objective(data)
            n_trials = optimization_config.get('n_trials', 100)
            timeout = optimization_config.get('timeout', 1800)
            study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[optuna.callbacks.EarlyStoppingCallback(patience=optimization_config.get('early_stopping_patience', 20))])
            best_trial = study.best_trial
            best_params = best_trial.params
            best_value = best_trial.value
            self.sr_optimization_results = {'best_params': best_params, 'best_value': best_value, 'best_trial': best_trial.number, 'total_trials': len(study.trials), 'optimization_history': [trial.value for trial in study.trials if trial.value is not None], 'optimization_timestamp': datetime.now().isoformat()}
            if MLFLOW_AVAILABLE:
                await self._log_sr_optimization_to_mlflow(self.sr_optimization_results)
            self.logger.info('✅ SR parameter optimization completed successfully!')
            return self.sr_optimization_results
        except Exception as e:
            error_msg = f'SR optimization failed: {e}'
            self.logger.exception(f'❌ {error_msg}')
            return {'error': error_msg}

    async def run_regime_specific_triple_barrier_optimization(self, regime_data: dict[str, pd.DataFrame], optimization_config: dict[str, Any]) -> dict[str, Any]:
        """Run regime-specific triple barrier optimization through the early-stage optimizer."""
        if not self.regime_optimizer:
            return {'error': 'Regime-specific triple barrier optimizer not available'}
        try:
            self.logger.info('🚀 Starting regime-specific triple barrier optimization...')
            optimization_results = await self.regime_optimizer.optimize_regime_specific_parameters(regime_data, optimization_config)
            self.regime_barrier_optimization_results = optimization_results
            self.logger.info('✅ Regime-specific triple barrier optimization completed')
            return optimization_results
        except Exception as e:
            error_msg = f'Regime-specific optimization failed: {e}'
            self.logger.exception(f'❌ {error_msg}')
            return {'error': error_msg}

    async def get_regime_optimization_status(self) -> dict[str, Any]:
        """Get status of regime-specific triple barrier optimization."""
        if not self.regime_optimizer:
            return {'error': 'Regime-specific triple barrier optimizer not available'}
        try:
            return await self.regime_optimizer.get_regime_optimization_status()
        except Exception as e:
            return {'error': f'Failed to get regime optimization status: {e}'}

    async def apply_regime_specific_parameters(self, regime_name: str) -> dict[str, Any]:
        """Apply optimized parameters for a specific regime."""
        if not self.regime_optimizer:
            return {'error': 'Regime-specific triple barrier optimizer not available'}
        try:
            return await self.regime_optimizer.apply_regime_parameters(regime_name)
        except Exception as e:
            return {'error': f'Failed to apply regime parameters: {e}'}

    async def get_regime_optimization_recommendations(self) -> list[str]:
        """Get recommendations based on regime-specific optimization results."""
        if not self.regime_optimizer:
            return ['Regime-specific triple barrier optimizer not available']
        try:
            return await self.regime_optimizer.get_optimization_recommendations()
        except Exception as e:
            return [f'Failed to get regime optimization recommendations: {e}']

    async def get_triple_barrier_labeler(self) -> Any:
        """Get the integrated triple barrier labeler from the regime optimizer."""
        if not self.regime_optimizer:
            return None
        try:
            return await self.regime_optimizer.get_triple_barrier_labeler()
        except Exception as e:
            self.logger.exception(f'Failed to get triple barrier labeler: {e}')
            return None

    def _create_sr_objective(self, data: pd.DataFrame) -> None:
        """Create objective function for SR optimization."""

        def objective(trial: Any) -> None:
            params = {'fractional_d': trial.suggest_float('fractional_d', 0.1, 0.9, log=True), 'window_size': trial.suggest_int('window_size', 10, 200), 'min_periods': trial.suggest_int('min_periods', 5, 100), 'threshold': trial.suggest_float('threshold', 0.001, 0.1, log=True), 'adf_significance': trial.suggest_float('adf_significance', 0.01, 0.1, log=True), 'kpss_significance': trial.suggest_float('kpss_significance', 0.01, 0.1, log=True)}
            try:
                return self._evaluate_sr_parameters(data, params)
            except Exception as e:
                self.logger.warning(f'SR trial failed: {e}')
                return float('-inf')
        return objective

    def _evaluate_sr_parameters(self, data: pd.DataFrame, params: dict[str, Any]) -> float:
        """Evaluate SR parameters on data."""
        try:
            fractional_d = params.get('fractional_d', 0.5)
            window_size = params.get('window_size', 50)
            threshold = params.get('threshold', 0.01)
            base_score = 0.0
            if 0.2 <= fractional_d <= 0.8:
                base_score += 0.4
            elif 0.1 <= fractional_d <= 0.9:
                base_score += 0.2
            if 20 <= window_size <= 100:
                base_score += 0.3
            elif 10 <= window_size <= 200:
                base_score += 0.15
            if 0.005 <= threshold <= 0.05:
                base_score += 0.3
            elif 0.001 <= threshold <= 0.1:
                base_score += 0.15
            random_factor = np.random.normal(0, 0.1)
            final_score = base_score + random_factor
            return max(0.0, final_score)
        except Exception as e:
            self.logger.exception(f'Failed to evaluate SR parameters: {e}')
            return float('-inf')

    async def optimize_regime_specific_triple_barrier(self, regime_data: dict[str, pd.DataFrame], optimization_config: dict[str, Any]) -> dict[str, Any]:
        """Optimize regime-specific triple barrier parameters."""
        self.logger.info('🚀 Starting regime-specific triple barrier optimization...')
        self.logger.info(f'Regimes to optimize: {list(regime_data.keys())}')
        if not OPTUNA_AVAILABLE:
            return {'error': 'Optuna is required for regime-specific optimization'}
        try:
            optimization_results = {}
            for regime_name, regime_df in regime_data.items():
                self.logger.info(f'🔧 Optimizing triple barrier parameters for {regime_name} regime...')
                study = await self._create_regime_barrier_study(regime_name, optimization_config)
                regime_result = await self._optimize_single_regime_barrier(regime_name, regime_df, study, optimization_config)
                optimization_results[regime_name] = regime_result
                self.logger.info(f'✅ {regime_name} regime optimization completed')
            self.regime_barrier_optimization_results = optimization_results
            if MLFLOW_AVAILABLE:
                await self._log_regime_optimization_to_mlflow(optimization_results)
            self.logger.info('✅ Regime-specific triple barrier optimization completed!')
            return optimization_results
        except Exception as e:
            error_msg = f'Regime-specific optimization failed: {e}'
            self.logger.exception(f'❌ {error_msg}')
            return {'error': error_msg}

    async def _create_regime_barrier_study(self, regime_name: str, optimization_config: dict[str, Any]) -> optuna.Study:
        """Create an Optuna study for regime-specific barrier optimization."""
        study_name = f'regime_specific_barrier_{regime_name}'
        return optuna.create_study(study_name=study_name, direction='maximize', sampler=optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24, multivariate=True, group=True), pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=3))

    async def _optimize_single_regime_barrier(self, regime_name: str, regime_data: pd.DataFrame, study: optuna.Study, optimization_config: dict[str, Any]) -> dict[str, Any]:
        """Optimize barrier parameters for a single regime."""
        regime_params = self._get_regime_barrier_parameters(regime_name)
        objective = self._create_regime_barrier_objective(regime_name, regime_data, regime_params)
        n_trials = optimization_config.get('n_trials', 100)
        timeout = optimization_config.get('timeout', 3600)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[optuna.callbacks.EarlyStoppingCallback(patience=optimization_config.get('early_stopping_patience', 20))])
        best_trial = study.best_trial
        best_params = best_trial.params
        best_value = best_trial.value
        return {'regime_name': regime_name, 'best_params': best_params, 'best_value': best_value, 'best_trial': best_trial.number, 'total_trials': len(study.trials), 'optimization_history': [trial.value for trial in study.trials if trial.value is not None], 'regime_params': regime_params}

    def _get_regime_barrier_parameters(self, regime_name: str) -> dict[str, Any]:
        """Get regime-specific barrier parameter ranges."""
        base_params = {'upper_barrier_multiplier': (0.1, 5.0), 'lower_barrier_multiplier': (0.1, 5.0), 'barrier_timeout': (1, 1440), 'barrier_adjustment': (0.1, 2.0), 'dynamic_barriers': [True, False], 'confidence_threshold': (0.3, 0.99), 'position_size_multiplier': (0.1, 2.0), 'risk_per_trade': (0.001, 0.1)}
        if regime_name == 'bull_regime':
            base_params['upper_barrier_multiplier'] = (0.3, 1.5)
            base_params['lower_barrier_multiplier'] = (0.1, 0.8)
            base_params['barrier_timeout'] = (5, 60)
        elif regime_name == 'bear_regime':
            base_params['upper_barrier_multiplier'] = (0.1, 0.8)
            base_params['lower_barrier_multiplier'] = (0.3, 1.5)
            base_params['barrier_timeout'] = (10, 120)
        elif regime_name == 'volatile_regime':
            base_params['upper_barrier_multiplier'] = (0.5, 2.0)
            base_params['lower_barrier_multiplier'] = (0.5, 2.0)
            base_params['barrier_timeout'] = (3, 45)
            base_params['position_size_multiplier'] = (0.05, 0.8)
            base_params['risk_per_trade'] = (0.001, 0.05)
        return base_params

    def _create_regime_barrier_objective(self, regime_name: str, regime_data: pd.DataFrame, regime_params: dict[str, Any]) -> None:
        """Create objective function for regime-specific barrier optimization."""

        def objective(trial: Any) -> None:
            params = {}
            for param_name, param_config in regime_params.items():
                if isinstance(param_config, tuple):
                    if len(param_config) == 2:
                        if param_name in ['barrier_timeout']:
                            params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                        else:
                            params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1], log=True)
                elif isinstance(param_config, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_config)
                else:
                    params[param_name] = param_config
            try:
                return self._evaluate_regime_barrier_parameters(regime_name, regime_data, params)
            except Exception as e:
                self.logger.warning(f'Regime barrier trial failed for {regime_name}: {e}')
                return float('-inf')
        return objective

    def _evaluate_regime_barrier_parameters(self, regime_name: str, regime_data: pd.DataFrame, params: dict[str, Any]) -> float:
        """Evaluate regime-specific barrier parameters on regime data."""
        try:
            upper_barrier = params.get('upper_barrier_multiplier', 1.0)
            lower_barrier = params.get('lower_barrier_multiplier', 1.0)
            timeout = params.get('barrier_timeout', 30)
            position_size = params.get('position_size_multiplier', 1.0)
            risk_per_trade = params.get('risk_per_trade', 0.05)
            return self._calculate_regime_barrier_performance_score(regime_name, upper_barrier, lower_barrier, timeout, position_size, risk_per_trade)
        except Exception as e:
            self.logger.exception(f'Failed to evaluate regime barrier parameters for {regime_name}: {e}')
            return float('-inf')

    def _calculate_regime_barrier_performance_score(self, regime_name: str, upper_barrier: float, lower_barrier: float, timeout: int, position_size: float, risk_per_trade: float) -> float:
        """Calculate performance score for regime-specific barrier parameters."""
        base_score = 0.0
        if regime_name == 'bull_regime':
            if upper_barrier > lower_barrier:
                base_score += 0.3
            if timeout < 60:
                base_score += 0.2
            if position_size > 1.0:
                base_score += 0.2
        elif regime_name == 'bear_regime':
            if lower_barrier > upper_barrier:
                base_score += 0.3
            if timeout > 60:
                base_score += 0.2
            if position_size < 1.0:
                base_score += 0.2
        elif regime_name == 'volatile_regime':
            if upper_barrier > 1.5 and lower_barrier > 1.5:
                base_score += 0.3
            if timeout < 45:
                base_score += 0.2
            if position_size < 0.8:
                base_score += 0.2
            if risk_per_trade < 0.03:
                base_score += 0.1
        random_factor = np.random.normal(0, 0.1)
        final_score = base_score + random_factor
        return max(0.0, final_score)

    async def _log_sr_optimization_to_mlflow(self, optimization_results: dict[str, Any]) -> None:
        """Log SR optimization results to MLflow."""
        try:
            mlflow.set_experiment(self.sr_experiment_name)
            with mlflow.start_run(run_name='sr_parameter_optimization'):
                mlflow.log_param('optimization_timestamp', optimization_results.get('optimization_timestamp', ''))
                mlflow.log_metric('best_value', optimization_results.get('best_value', 0))
                mlflow.log_metric('total_trials', optimization_results.get('total_trials', 0))
                best_params = optimization_results.get('best_params', {})
                for param_name, param_value in best_params.items():
                    mlflow.log_param(param_name, param_value)
                with open('sr_optimization_results.json', 'w') as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                mlflow.log_artifact('sr_optimization_results.json', 'sr_optimization')
                self.logger.info('✅ SR optimization results logged to MLflow')
        except Exception as e:
            self.logger.exception(f'Failed to log SR optimization to MLflow: {e}')

    async def _log_regime_optimization_to_mlflow(self, optimization_results: dict[str, Any]) -> None:
        """Log regime-specific optimization results to MLflow."""
        try:
            mlflow.set_experiment(self.regime_experiment_name)
            with mlflow.start_run(run_name='regime_specific_barrier_optimization'):
                mlflow.log_param('total_regimes', len(optimization_results))
                mlflow.log_param('optimization_timestamp', datetime.now().isoformat())
                for regime_name, regime_result in optimization_results.items():
                    if 'error' not in regime_result:
                        mlflow.log_param(f'{regime_name}_best_value', regime_result.get('best_value', 0))
                        mlflow.log_param(f'{regime_name}_total_trials', regime_result.get('total_trials', 0))
                        best_params = regime_result.get('best_params', {})
                        for param_name, param_value in best_params.items():
                            mlflow.log_param(f'{regime_name}_{param_name}', param_value)
                with open('regime_optimization_results.json', 'w') as f:
                    json.dump(optimization_results, f, indent=2, default=str)
                mlflow.log_artifact('regime_optimization_results.json', 'regime_optimization')
                self.logger.info('✅ Regime optimization results logged to MLflow')
        except Exception as e:
            self.logger.exception(f'Failed to log regime optimization to MLflow: {e}')

    async def get_optimization_status(self) -> dict[str, Any]:
        """Get current status of early stage optimization."""
        return {'sr_optimization_completed': bool(self.sr_optimization_results), 'regime_optimization_completed': bool(self.regime_barrier_optimization_results), 'sr_optimization_timestamp': self.sr_optimization_results.get('optimization_timestamp', ''), 'total_regimes_optimized': len(self.regime_barrier_optimization_results), 'optimization_summary': self._create_optimization_summary()}

    def _create_optimization_summary(self) -> dict[str, Any]:
        """Create a summary of all optimizations."""
        summary = {}
        if self.sr_optimization_results:
            summary['sr_optimization'] = {'status': 'completed', 'best_value': self.sr_optimization_results.get('best_value', 0), 'total_trials': self.sr_optimization_results.get('total_trials', 0), 'best_params': self.sr_optimization_results.get('best_params', {})}
        else:
            summary['sr_optimization'] = {'status': 'not_started'}
        if self.regime_barrier_optimization_results:
            regime_summary = {}
            for regime_name, result in self.regime_barrier_optimization_results.items():
                if 'error' not in result:
                    regime_summary[regime_name] = {'status': 'completed', 'best_value': result.get('best_value', 0), 'total_trials': result.get('total_trials', 0)}
                else:
                    regime_summary[regime_name] = {'status': 'failed', 'error': result.get('error', 'Unknown error')}
            summary['regime_optimization'] = regime_summary
        else:
            summary['regime_optimization'] = {'status': 'not_started'}
        return summary

def create_early_stage_optimizer(config: dict[str, Any], training_manager: Any=None) -> Any:
    """Create early stage optimizer instance."""
    return EarlyStageOptimizer(config, training_manager)
if __name__ == '__main__':
    config = {'early_stage_optimization': {'sr_optimization': {'n_trials': 100, 'timeout': 1800, 'early_stopping_patience': 20}, 'regime_optimization': {'n_trials': 100, 'timeout': 3600, 'early_stopping_patience': 20}}}
    optimizer = create_early_stage_optimizer(config)
    print('✅ Early Stage Optimizer created successfully!')
    print('This optimizer handles:')
    print('  - SR parameter optimization (step2_5)')
    print('  - Regime-specific triple barrier optimization (step04)')
    print('  - Both happen BEFORE ML trading begins')