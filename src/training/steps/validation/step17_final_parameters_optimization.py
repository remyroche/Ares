from src.core.domain import get_hyperparameter_config
import asyncio
import contextlib
import json
import os
import pickle
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from src.config_optuna import get_optimizable_parameters, get_optuna_config
from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, failed, missing
from typing import Dict, List, Optional, Union, Any, Tuple

class FinalParametersOptimizationStep:
    """Step 12: Final Parameters Optimization using Optuna with advanced features."""

    def _validate_environment(self) -> None:
        """Validate environment dependencies and configuration."""
        if not dependency_status['all_available']:
            missing_modules = dependency_status['missing_modules']
            self.logger.warning(f'Missing modules: {missing_modules}')

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger
        self.optuna_config = get_optuna_config()
        self.optimizable_params = get_optimizable_parameters()

    @handles_errors(fallback=False)
    async def initialize(self) -> None:
        """Initialize the final parameters optimization step."""
        self.logger.info('🚀 Initializing Final Parameters Optimization Step...')
        validation_errors = self._validate_optuna_config()
        if validation_errors:
            self.logger.warning(f'⚠️ Optuna config validation warnings: {validation_errors}')
        self._setup_optimization_storage()
        self.logger.info('✅ Final Parameters Optimization Step initialized successfully')

    @handles_errors(default_return={'status': 'FAILED', 'error': 'Execution failed'}, context='final parameters optimization step execution')
    async def execute(self, training_input: dict[str, Any], pipeline_state: dict[str, Any]) -> dict[str, Any]:
        """Execute final parameters optimization with advanced features.

        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state

        Returns:
            Dict containing optimization results

        """
        try:
            self.logger.info('🔄 Executing Final Parameters Optimization...')
            start_time = datetime.now()
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            data_dir = training_input.get('data_dir', 'data/training')
            from src.utils.logger import heartbeat
            with heartbeat(self.logger, name='Step12 load_calibration_results', interval_seconds=60.0):
                calibration_results = await self._load_calibration_results(symbol, exchange, data_dir)
            if not calibration_results:
                msg = 'Calibration results not found'
                raise FileNotFoundError(msg)
            with contextlib.suppress(Exception):
                self.logger.info(f'Loaded calibration results for {exchange}/{symbol}: sections={list(calibration_results.keys())[:10]}')
            with heartbeat(self.logger, name='Step12 load_previous_optimization', interval_seconds=60.0):
                previous_results = await self._load_previous_optimization_results(symbol, exchange, data_dir)
            try:

                def _summ(obj: Any) -> str:
                    try:
                        return len(obj)
                    except Exception:
                        return 'n/a'
                self.logger.info(f'Inputs summary — calibration_results: type={type(calibration_results).__name__}, size={_summ(calibration_results)}; previous_results: type={type(previous_results).__name__}, size={_summ(previous_results)}')
            except Exception:
                pass
            with heartbeat(self.logger, name='Step12 optimize_all_parameters', interval_seconds=60.0):
                optimization_results = await self._optimize_all_parameters(calibration_results, previous_results)
            try:
                keys = list(optimization_results.keys()) if isinstance(optimization_results, dict) else []
                self.logger.info(f"Optimization finished. Result keys: {keys[:20]} (total={(len(keys) if keys else 'n/a')})")
            except Exception:
                pass
            with heartbeat(self.logger, name='Step12 validate_optimization', interval_seconds=60.0):
                validation_passed = await self._validate_optimization_results(optimization_results)
            if not validation_passed:
                self.logger.warning('⚠️ Optimization results validation failed, using fallback parameters')
            with heartbeat(self.logger, name='Step12 save_results', interval_seconds=60.0):
                await self._save_optimization_results(optimization_results, symbol, exchange, data_dir)
            with contextlib.suppress(Exception):
                pass
            with heartbeat(self.logger, name='Step12 generate_report', interval_seconds=60.0):
                report = await self._generate_optimization_report(optimization_results, start_time)
            with contextlib.suppress(Exception):
                pass
            pipeline_state['final_parameters'] = optimization_results
            pipeline_state['optimization_report'] = report
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f'✅ Final parameters optimization completed in {duration:.2f}s')
            with contextlib.suppress(Exception):
                pass
            return {'final_parameters': optimization_results, 'optimization_report': report, 'duration': duration, 'status': 'SUCCESS'}
        except Exception as e:
            self.print(error('❌ Error in Final Parameters Optimization: {e}'))
            return {'status': 'FAILED', 'error': str(e), 'duration': 0.0}

    async def _load_calibration_results(self, symbol: str, exchange: str, data_dir: str) -> dict[str, Any] | None:
        """Load calibration results from previous step."""
        try:
            calibration_dir = f'{data_dir}/calibration_results'
            calibration_file = f'{calibration_dir}/{exchange}_{symbol}_calibration_results.pkl'
            if not os.path.exists(calibration_file):
                self.print(missing(f'Calibration file not found: {calibration_file}'))
                return None
            with open(calibration_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.print(error(f'Error loading calibration results: {e}'))
            return None

    async def _load_previous_optimization_results(self, symbol: str, exchange: str, data_dir: str) -> dict[str, Any] | None:
        """Load previous optimization results for warm start."""
        try:
            optimization_dir = f'{data_dir}/optimization_results'
            previous_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            if os.path.exists(previous_file):
                with open(previous_file, 'rb') as f:
                    return pickle.load(f)
            return None
        except Exception as e:
            self.print(error(f'Error loading previous optimization results: {e}'))
            return None

    async def _optimize_all_parameters(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize all parameters using advanced Optuna features.

        Args:
            calibration_results: Results from confidence calibration
            previous_results: Previous optimization results for warm start

        Returns:
            Dict containing optimized parameters

        """
        try:
            self.logger.info('Optimizing all parameters...')
            optimization_results = {}
            confidence_results = await self._optimize_confidence_thresholds_multi_objective(calibration_results, previous_results)
            optimization_results['confidence_thresholds'] = confidence_results
            (volatility_results, await self._optimize_volatility_parameters_advanced(calibration_results, previous_results))
            optimization_results['volatility_parameters'] = volatility_results
            (position_results, await self._optimize_position_sizing_advanced(calibration_results, previous_results))
            optimization_results['position_sizing_parameters'] = position_results
            (risk_results, await self._optimize_risk_management_advanced(calibration_results, previous_results))
            optimization_results['risk_management_parameters'] = risk_results
            (ensemble_results, await self._optimize_ensemble_parameters(calibration_results, previous_results))
            optimization_results['ensemble_parameters'] = ensemble_results
            (regime_results, await self._optimize_regime_specific_parameters(calibration_results, previous_results))
            optimization_results['regime_specific_parameters'] = regime_results
            (timing_results, await self._optimize_timing_parameters(calibration_results, previous_results))
            optimization_results['timing_parameters'] = timing_results
            return optimization_results
        except Exception:
            self.print(error('Error optimizing all parameters: {e}'))
            raise

    async def _optimize_confidence_thresholds_multi_objective(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize confidence thresholds using multi-objective optimization."""
        try:
            import optuna
            val_df = self._load_validation_frame()
            if val_df is None or 'label' not in val_df.columns:
                msg = 'Validation frame not available for optimization'
                raise RuntimeError(msg)

            def objective(trial: Any) -> None:
                params = {'analyst_confidence_threshold': trial.suggest_float('analyst_confidence_threshold', 0.5, 0.95, step=0.02), 'tactician_confidence_threshold': trial.suggest_float('tactician_confidence_threshold', 0.5, 0.95, step=0.02), 'ensemble_confidence_threshold': trial.suggest_float('ensemble_confidence_threshold', 0.5, 0.95, step=0.02), 'position_scale_up_threshold': trial.suggest_float('position_scale_up_threshold', 0.7, 0.95, step=0.02), 'position_scale_down_threshold': trial.suggest_float('position_scale_down_threshold', 0.4, 0.7, step=0.02), 'position_close_threshold': trial.suggest_float('position_close_threshold', 0.2, 0.5, step=0.02), 'enhanced_prediction_confidence_threshold': trial.suggest_float('enhanced_prediction_confidence_threshold', 0.5, 0.9, step=0.02), 'enhanced_prediction_price_threshold': trial.suggest_float('enhanced_prediction_price_threshold', 0.4, 0.8, step=0.02), 'analyst_ml_weight': trial.suggest_float('analyst_ml_weight', 0.4, 0.8, step=0.05), 'tactician_ml_weight': trial.suggest_float('tactician_ml_weight', 0.2, 0.6, step=0.05), 'position_sizing_confidence_multiplier': trial.suggest_float('position_sizing_confidence_multiplier', 1.0, 2.5, step=0.1), 'leverage_sizing_risk_multiplier': trial.suggest_float('leverage_sizing_risk_multiplier', 0.5, 1.5, step=0.05), 'base_position_size': trial.suggest_float('base_position_size', 0.05, 0.2, step=0.01), 'max_position_size': trial.suggest_float('max_position_size', 0.15, 0.3, step=0.01), 'base_leverage': trial.suggest_float('base_leverage', 20.0, 80.0, step=5.0), 'max_leverage': trial.suggest_float('max_leverage', 60.0, 100.0, step=5.0)}
                (metrics, self._evaluate_predictions(calibration_results, val_df, params))
            return (metrics.get('win_rate', 0.5), metrics.get('avg_win', 0.01), -metrics.get('avg_loss', 0.01), metrics.get('sharpe_ratio', 1.0), -metrics.get('max_drawdown', 0.1), metrics.get('enhanced_prediction_performance', 0.6))
            self.logger.info('Step12: Starting Optuna study for confidence thresholds (multi-objective)')
            (study, optuna.create_study(directions=['maximize', 'maximize', 'minimize', 'maximize', 'minimize', 'maximize'], sampler=optuna.samplers.TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)))
            if previous_results and 'confidence_thresholds' in previous_results:
                (prev_params, previous_results['confidence_thresholds'].get('optimized_parameters'))
            if prev_params:
                study.enqueue_trial(prev_params)
            confidence_trials = self.training_input.get('confidence_threshold_trials', 40)
            self.logger.info(f'Step12: Optimizing confidence thresholds (n_trials={confidence_trials})')
            study.optimize(objective, n_trials=confidence_trials)
            pareto_front = study.best_trials
            best_solution = self._select_best_pareto_solution(pareto_front)
            return {'optimized_parameters': best_solution.params, 'pareto_front_size': len(pareto_front), 'best_objectives': best_solution.values, 'optimization_method': 'multi_objective_optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing confidence thresholds: {e}'))
            return self._get_default_confidence_thresholds()

    async def _optimize_volatility_parameters_advanced(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize volatility parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'target_volatility': trial.suggest_float('target_volatility', 0.05, 0.25), 'volatility_lookback_period': trial.suggest_int('volatility_lookback_period', 10, 50), 'volatility_multiplier': trial.suggest_float('volatility_multiplier', 0.5, 2.0), 'low_volatility_threshold': trial.suggest_float('low_volatility_threshold', 0.01, 0.05), 'medium_volatility_threshold': trial.suggest_float('medium_volatility_threshold', 0.03, 0.08), 'high_volatility_threshold': trial.suggest_float('high_volatility_threshold', 0.08, 0.15), 'volatility_stop_loss_multiplier': trial.suggest_float('volatility_stop_loss_multiplier', 1.0, 3.0)}
            return self._evaluate_volatility_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for volatility parameters')
            (study, optuna.create_study(direction='maximize'))
            if previous_results and 'volatility_parameters' in previous_results:
                (prev_params, previous_results['volatility_parameters'].get('optimized_parameters'))
            if prev_params:
                study.enqueue_trial(prev_params)
            volatility_trials = self.training_input.get('volatility_trials', 50)
            self.logger.info(f'Step12: Optimizing volatility parameters (n_trials={volatility_trials})')
            study.optimize(objective, n_trials=volatility_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing volatility parameters: {e}'))
            return self._get_default_volatility_parameters()

    async def _optimize_position_sizing_advanced(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize position sizing parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'base_position_size': trial.suggest_float('base_position_size', 0.01, 0.2), 'max_position_size': trial.suggest_float('max_position_size', 0.1, 0.5), 'min_position_size': trial.suggest_float('min_position_size', 0.005, 0.05), 'kelly_multiplier': trial.suggest_float('kelly_multiplier', 0.1, 0.5), 'fractional_kelly': trial.suggest_categorical('fractional_kelly', [True, False]), 'confidence_based_scaling': trial.suggest_categorical('confidence_based_scaling', [True, False]), 'low_confidence_multiplier': trial.suggest_float('low_confidence_multiplier', 0.3, 0.8), 'medium_confidence_multiplier': trial.suggest_float('medium_confidence_multiplier', 0.8, 1.2), 'high_confidence_multiplier': trial.suggest_float('high_confidence_multiplier', 1.2, 2.5), 'very_high_confidence_multiplier': trial.suggest_float('very_high_confidence_multiplier', 1.5, 3.0)}
            return self._evaluate_position_sizing_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for position sizing parameters')
            (study, optuna.create_study(direction='maximize'))
            if previous_results and 'position_sizing_parameters' in previous_results:
                (prev_params, previous_results['position_sizing_parameters'].get('optimized_parameters'))
            if prev_params:
                study.enqueue_trial(prev_params)
            position_sizing_trials = self.training_input.get('position_sizing_trials', 60)
            self.logger.info(f'Step12: Optimizing position sizing parameters (n_trials={position_sizing_trials})')
            study.optimize(objective, n_trials=position_sizing_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing position sizing parameters: {e}'))
            return self._get_default_position_sizing_parameters()

    async def _optimize_risk_management_advanced(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize risk management parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'stop_loss_atr_multiplier': trial.suggest_float('stop_loss_atr_multiplier', 1.0, 4.0), 'trailing_stop_atr_multiplier': trial.suggest_float('trailing_stop_atr_multiplier', 0.8, 3.0), 'stop_loss_confidence_threshold': trial.suggest_float('stop_loss_confidence_threshold', 0.2, 0.5), 'enable_dynamic_stop_loss': trial.suggest_categorical('enable_dynamic_stop_loss', [True, False]), 'volatility_based_sl': trial.suggest_categorical('volatility_based_sl', [True, False]), 'regime_based_sl': trial.suggest_categorical('regime_based_sl', [True, False]), 'sl_tightening_threshold': trial.suggest_float('sl_tightening_threshold', 0.3, 0.6), 'sl_loosening_threshold': trial.suggest_float('sl_loosening_threshold', 0.7, 0.9), 'max_drawdown_threshold': trial.suggest_float('max_drawdown_threshold', 0.1, 0.3), 'max_daily_loss': trial.suggest_float('max_daily_loss', 0.05, 0.15)}
            return self._evaluate_risk_management_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for risk management parameters')
            (study, optuna.create_study(direction='maximize'))
            if previous_results and 'risk_management_parameters' in previous_results:
                (prev_params, previous_results['risk_management_parameters'].get('optimized_parameters'))
            if prev_params:
                study.enqueue_trial(prev_params)
            risk_management_trials = self.training_input.get('risk_management_trials', 50)
            self.logger.info(f'Step12: Optimizing risk management parameters (n_trials={risk_management_trials})')
            study.optimize(objective, n_trials=risk_management_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing risk management parameters: {e}'))
            return self._get_default_risk_management_parameters()

    async def _optimize_ensemble_parameters(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize ensemble parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'ensemble_method': trial.suggest_categorical('ensemble_method', ['confidence_weighted', 'weighted_average', 'meta_learner', 'majority_vote']), 'analyst_weight': trial.suggest_float('analyst_weight', 0.2, 0.6), 'tactician_weight': trial.suggest_float('tactician_weight', 0.2, 0.6), 'strategist_weight': trial.suggest_float('strategist_weight', 0.1, 0.4), 'min_ensemble_agreement': trial.suggest_float('min_ensemble_agreement', 0.5, 0.8), 'max_ensemble_disagreement': trial.suggest_float('max_ensemble_disagreement', 0.2, 0.5), 'ensemble_minimum_models': trial.suggest_int('ensemble_minimum_models', 2, 5)}
            return self._evaluate_ensemble_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for ensemble parameters')
            (study, optuna.create_study(direction='maximize'))
            if previous_results and 'ensemble_parameters' in previous_results:
                (prev_params, previous_results['ensemble_parameters'].get('optimized_parameters'))
            if prev_params:
                study.enqueue_trial(prev_params)
            ensemble_trials = self.training_input.get('ensemble_trials', 40)
            self.logger.info(f'Step12: Optimizing ensemble parameters (n_trials={ensemble_trials})')
            study.optimize(objective, n_trials=ensemble_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing ensemble parameters: {e}'))
            return self._get_default_ensemble_parameters()

    async def _optimize_regime_specific_parameters(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize regime-specific parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'bull_trend_multiplier': trial.suggest_float('bull_trend_multiplier', 0.8, 1.5), 'bear_trend_multiplier': trial.suggest_float('bear_trend_multiplier', 0.5, 1.2), 'sideways_multiplier': trial.suggest_float('sideways_multiplier', 0.7, 1.3), 'high_impact_multiplier': trial.suggest_float('high_impact_multiplier', 0.4, 1.0), 'sr_zone_multiplier': trial.suggest_float('sr_zone_multiplier', 0.8, 1.4), 'regime_transition_threshold': trial.suggest_float('regime_transition_threshold', 0.4, 0.8), 'regime_confirmation_periods': trial.suggest_int('regime_confirmation_periods', 2, 5)}
            return self._evaluate_regime_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for regime-specific parameters')
            study = optuna.create_study(direction='maximize')
            if previous_results and 'regime_specific_parameters' in previous_results:
                prev_params = previous_results['regime_specific_parameters'].get('optimized_parameters')
            if prev_params:
                study.enqueue_trial(prev_params)
            regime_specific_trials = self.training_input.get('regime_specific_trials', 30)
            self.logger.info(f'Step12: Optimizing regime-specific parameters (n_trials={regime_specific_trials})')
            study.optimize(objective, n_trials=regime_specific_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing regime-specific parameters: {e}'))
            return self._get_default_regime_parameters()

    async def _optimize_timing_parameters(self, calibration_results: dict[str, Any], previous_results: dict[str, Any] | None) -> dict[str, Any]:
        """Optimize timing parameters with advanced features."""
        try:
            import optuna

            def objective(trial: Any) -> None:
                params = {'base_cooldown_minutes': trial.suggest_int('base_cooldown_minutes', 15, 60, 5), 'high_confidence_cooldown': trial.suggest_int('high_confidence_cooldown', 5, 30, 5), 'low_confidence_cooldown': trial.suggest_int('low_confidence_cooldown', 30, 120, 10), 'bull_trend_cooldown': trial.suggest_int('bull_trend_cooldown', 10, 40, 5), 'bear_trend_cooldown': trial.suggest_int('bear_trend_cooldown', 20, 60, 5), 'sideways_cooldown': trial.suggest_int('sideways_cooldown', 30, 90, 10), 'high_impact_cooldown': trial.suggest_int('high_impact_cooldown', 60, 180, 15)}
            return self._evaluate_timing_performance(params, calibration_results)
            self.logger.info('Step12: Starting Optuna study for timing parameters')
            study = optuna.create_study(direction='maximize')
            if previous_results and 'timing_parameters' in previous_results:
                prev_params = previous_results['timing_parameters'].get('optimized_parameters')
            if prev_params:
                study.enqueue_trial(prev_params)
            timing_trials = self.training_input.get('timing_trials', 30)
            self.logger.info(f'Step12: Optimizing timing parameters (n_trials={timing_trials})')
            study.optimize(objective, n_trials=timing_trials)
            return {'optimized_parameters': study.best_params, 'best_score': study.best_value, 'optimization_method': 'optuna', 'n_trials': len(study.trials), 'optimization_date': datetime.now().isoformat()}
        except Exception:
            self.print(error('Error optimizing timing parameters: {e}'))
            return self._get_default_timing_parameters()

    def _validate_optuna_config(self) -> list[str]:
        """Validate Optuna configuration."""
        errors = []
        if not hasattr(self, 'optuna_config'):
            errors.append('Optuna config not loaded')
        if not hasattr(self, 'optimizable_params'):
            errors.append('Optimizable parameters not loaded')
        return errors

    def _setup_optimization_storage(self) -> None:
        """Setup optimization storage and directories."""
        try:
            storage_dir = 'data/optimization_storage'
            os.makedirs(storage_dir, exist_ok=True)
            db_path = f'{storage_dir}/optuna_studies.db'
            self.storage_url = f'sqlite:///{db_path}'
            self.logger.info(f'Optimization storage setup at: {storage_dir}')
        except Exception:
            self.print(error('Error setting up optimization storage: {e}'))

    async def _validate_optimization_results(self, results: dict[str, Any]) -> bool:
        """Validate optimization results."""
        try:
            required_sections = ['confidence_thresholds', 'volatility_parameters', 'position_sizing_parameters', 'risk_management_parameters']
            for section in required_sections:
                if section not in results:
                    self.print(missing('Missing required section: {section}'))
                    return False
                section_data = results[section]
                if 'optimized_parameters' not in section_data:
                    self.print(missing('Missing optimized_parameters in {section}'))
                    return False
            return True
        except Exception:
            self.print(error('Error validating optimization results: {e}'))
            return False

    async def _save_optimization_results(self, results: dict[str, Any], symbol: str, exchange: str, data_dir: str) -> None:
        """Save optimization results to files."""
        try:
            optimization_dir = f'{data_dir}/optimization_results'
            os.makedirs(optimization_dir, exist_ok=True)
            pickle_file = f'{optimization_dir}/{exchange}_{symbol}_final_parameters.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(results, f)
            summary_file = f'{data_dir}/{exchange}_{symbol}_final_parameters_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f'Optimization results saved to {optimization_dir}')
        except Exception:
            self.print(error('Error saving optimization results: {e}'))

    async def _generate_optimization_report(self, results: dict[str, Any], start_time: datetime) -> dict[str, Any]:
        """Generate comprehensive optimization report."""
        try:
            report = {'optimization_metadata': {'start_time': start_time.isoformat(), 'end_time': datetime.now().isoformat(), 'duration_seconds': (datetime.now() - start_time).total_seconds(), 'total_parameters_optimized': sum((len(section.get('optimized_parameters', {})) for section in results.values()))}, 'optimization_summary': {}, 'parameter_changes': {}, 'performance_metrics': {}, 'recommendations': []}
            for section_name, section_data in results.items():
                if 'optimized_parameters' in section_data:
                    params = section_data['optimized_parameters']
                    report['optimization_summary'][section_name] = {'parameters_optimized': len(params), 'best_score': section_data.get('best_score', 0.0), 'optimization_method': section_data.get('optimization_method', 'unknown'), 'n_trials': section_data.get('n_trials', 0)}
            report['recommendations'] = self._generate_optimization_recommendations(results)
            return report
        except Exception as e:
            self.print(error('Error generating optimization report: {e}'))
            return {'error': str(e)}

    def _generate_optimization_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []
        try:
            if 'confidence_thresholds' in results:
                (conf_params, results['confidence_thresholds'].get('optimized_parameters', {}))
            if conf_params.get('analyst_confidence_threshold', 0) < 0.6:
                recommendations.append('Consider increasing analyst confidence threshold for better signal quality')
            if conf_params.get('ensemble_confidence_threshold', 0) < 0.7:
                recommendations.append('Consider increasing ensemble confidence threshold for more conservative trading')
            if conf_params.get('enhanced_prediction_confidence_threshold', 0) < 0.6:
                recommendations.append('Consider increasing enhanced prediction confidence threshold for better signal quality')
            if conf_params.get('enhanced_prediction_price_threshold', 0) < 0.5:
                recommendations.append('Consider increasing enhanced prediction price threshold for more selective trades')
            if conf_params.get('analyst_ml_weight', 0) < 0.4:
                recommendations.append('Consider increasing analyst ML weight for better market analysis integration')
            if conf_params.get('tactician_ml_weight', 0) < 0.2:
                recommendations.append('Consider increasing tactician ML weight for better execution optimization')
            if conf_params.get('position_sizing_confidence_multiplier', 0) > 2.0:
                recommendations.append('High position sizing confidence multiplier detected - consider reducing for risk management')
            if conf_params.get('leverage_sizing_risk_multiplier', 0) > 1.3:
                recommendations.append('High leverage sizing risk multiplier detected - consider reducing for safety')
            if 'position_sizing_parameters' in results:
                (pos_params, results['position_sizing_parameters'].get('optimized_parameters', {}))
            if pos_params.get('max_position_size', 0) > 0.3:
                recommendations.append('High max position size detected - consider reducing for risk management')
            if pos_params.get('kelly_multiplier', 0) > 0.4:
                recommendations.append('High Kelly multiplier detected - consider reducing for safety')
            if 'risk_management_parameters' in results:
                (risk_params, results['risk_management_parameters'].get('optimized_parameters', {}))
            if risk_params.get('stop_loss_atr_multiplier', 0) > 3.0:
                recommendations.append('Wide stop loss detected - consider tightening for better risk control')
        except Exception:
            self.print(error('Error generating recommendations: {e}'))
        return recommendations

    def _evaluate_win_rate(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate win rate based on parameters."""
        try:
            base_win_rate = 0.55
            (confidence_factor, params.get('analyst_confidence_threshold', 0.7) * 0.3)
            (ensemble_factor, params.get('ensemble_confidence_threshold', 0.75) * 0.2)
            return min(0.95, base_win_rate + confidence_factor + ensemble_factor)
        except Exception:
            self.print(error('Error evaluating win rate: {e}'))
            return 0.5

    def _evaluate_profit_factor(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate profit factor based on parameters."""
        try:
            base_profit_factor = 1.3
            (position_size_factor, params.get('base_position_size', 0.05) * 2.0)
            (risk_factor, 1.0 - params.get('stop_loss_atr_multiplier', 2.0) * 0.1)
            return max(1.0, base_profit_factor + position_size_factor + risk_factor)
        except Exception:
            self.print(error('Error evaluating profit factor: {e}'))
            return 1.0

    def _evaluate_sharpe_ratio(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate Sharpe ratio based on parameters."""
        try:
            base_sharpe = 1.2
            (volatility_factor, params.get('target_volatility', 0.15) * 0.5)
            (confidence_factor, params.get('analyst_confidence_threshold', 0.7) * 0.3)
            return max(0.0, base_sharpe + volatility_factor + confidence_factor)
        except Exception:
            self.print(error('Error evaluating Sharpe ratio: {e}'))
            return 1.0

    def _evaluate_max_drawdown(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate maximum drawdown based on parameters."""
        try:
            base_drawdown = 0.15
            (position_size_factor, params.get('max_position_size', 0.25) * 0.2)
            (risk_factor, params.get('stop_loss_atr_multiplier', 2.0) * 0.05)
            return min(0.5, base_drawdown + position_size_factor + risk_factor)
        except Exception:
            self.print(error('Error evaluating max drawdown: {e}'))
            return 0.2

    def _evaluate_enhanced_prediction_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate enhanced prediction integrator performance."""
        try:
            base_performance = 0.6
            confidence_threshold = params.get('enhanced_prediction_confidence_threshold', 0.7)
            confidence_factor = confidence_threshold * 0.2
            price_threshold = params.get('enhanced_prediction_price_threshold', 0.6)
            price_factor = price_threshold * 0.15
            analyst_ml_weight = params.get('analyst_ml_weight', 0.6)
            tactician_ml_weight = params.get('tactician_ml_weight', 0.4)
            weight_balance = min(analyst_ml_weight, tactician_ml_weight) * 0.1
            position_multiplier = params.get('position_sizing_confidence_multiplier', 1.5)
            position_factor = min(position_multiplier, 2.0) * 0.05
            leverage_multiplier = params.get('leverage_sizing_risk_multiplier', 1.0)
            leverage_factor = (2.0 - leverage_multiplier) * 0.05
            base_position = params.get('base_position_size', 0.1)
            max_position = params.get('max_position_size', 0.2)
            base_leverage = params.get('base_leverage', 50.0)
            max_leverage = params.get('max_leverage', 100.0)
            size_factor = (base_position + max_position) * 0.1
            leverage_factor += (base_leverage + max_leverage) / 200.0 * 0.05
            total_performance = base_performance + confidence_factor + price_factor + weight_balance + position_factor + leverage_factor + size_factor
            return min(0.95, max(0.3, total_performance))
        except Exception:
            self.print(error('Error evaluating enhanced prediction performance: {e}'))
            return 0.6

    def _evaluate_average_win(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate average win amount based on parameters."""
        try:
            base_avg_win = 0.02
            (confidence_factor, params.get('analyst_confidence_threshold', 0.7) * 0.01)
            (position_size_factor, params.get('base_position_size', 0.05) * 0.5)
            (volatility_factor, params.get('target_volatility', 0.15) * 0.1)
            avg_win = base_avg_win + confidence_factor + position_size_factor + volatility_factor
            return max(0.005, avg_win)
        except Exception:
            self.print(error('Error evaluating average win: {e}'))
            return 0.02

    def _evaluate_average_loss(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate average loss amount based on parameters."""
        try:
            base_avg_loss = 0.015
            (stop_loss_factor, params.get('stop_loss_atr_multiplier', 2.0) * 0.005)
            (position_size_factor, params.get('base_position_size', 0.05) * 0.3)
            (risk_factor, params.get('max_position_size', 0.25) * 0.1)
            avg_loss = base_avg_loss + stop_loss_factor + position_size_factor + risk_factor
            return max(0.005, avg_loss)
        except Exception:
            self.print(error('Error evaluating average loss: {e}'))
            return 0.015

    def _evaluate_volatility_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate volatility parameter performance."""
        try:
            (target_vol, params.get('target_volatility', 0.15))
            (multiplier, params.get('volatility_multiplier', 1.0))
            return target_vol * multiplier * 10
        except Exception:
            self.print(error('Error evaluating volatility performance: {e}'))
            return 0.0

    def _evaluate_position_sizing_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate position sizing performance."""
        try:
            (base_size, params.get('base_position_size', 0.05))
            (kelly_mult, params.get('kelly_multiplier', 0.25))
            confidence_scaling = 1.0 if params.get('confidence_based_scaling', True) else 0.8
            return base_size * kelly_mult * confidence_scaling * 20
        except Exception:
            self.print(error('Error evaluating position sizing performance: {e}'))
            return 0.0

    def _evaluate_risk_management_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate risk management performance."""
        try:
            (sl_multiplier, params.get('stop_loss_atr_multiplier', 2.0))
            (trailing_multiplier, params.get('trailing_stop_atr_multiplier', 1.5))
            (dynamic_sl, 1.2 if params.get('enable_dynamic_stop_loss', True) else 1.0)
            return (sl_multiplier + trailing_multiplier) * dynamic_sl
        except Exception:
            self.print(error('Error evaluating risk management performance: {e}'))
            return 0.0

    def _evaluate_ensemble_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate ensemble performance."""
        try:
            (analyst_weight, params.get('analyst_weight', 0.4))
            (tactician_weight, params.get('tactician_weight', 0.3))
            (agreement, params.get('min_ensemble_agreement', 0.7))
            return (analyst_weight + tactician_weight) * agreement * 2.0
        except Exception:
            self.print(error('Error evaluating ensemble performance: {e}'))
            return 0.0

    def _evaluate_regime_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate regime-specific performance."""
        try:
            (bull_mult, params.get('bull_trend_multiplier', 1.2))
            (bear_mult, params.get('bear_trend_multiplier', 0.8))
            (sideways_mult, params.get('sideways_multiplier', 0.9))
            return (bull_mult + bear_mult + sideways_mult) / 3.0
        except Exception:
            self.print(error('Error evaluating regime performance: {e}'))
            return 0.0

    def _evaluate_timing_performance(self, params: dict[str, Any], calibration_results: dict[str, Any]) -> float:
        """Evaluate timing performance."""
        try:
            (base_cooldown, params.get('base_cooldown_minutes', 30))
            (high_conf_cooldown, params.get('high_confidence_cooldown', 15))
            return 1.0 / (base_cooldown + high_conf_cooldown) * 100
        except Exception:
            self.print(error('Error evaluating timing performance: {e}'))
            return 0.0

    def _select_best_pareto_solution(self, pareto_front: list) -> Any:
        """Select the best solution from Pareto front."""
        try:
            from sklearn.metrics import accuracy_score
            from src.training.steps.validation.step17_final_parameters_optimization.hyperparameter_optimization_config import get_hyperparameter_config
            config = get_hyperparameter_config()
            weights = getattr(config, 'composite_score_weights', {'win_rate': 0.25, 'profit_factor': 0.25, 'sharpe_ratio': 0.25, 'max_drawdown': 0.1, 'enhanced_prediction_performance': 0.15})
            best_solution = None
            best_score = -float('inf')
            for solution in pareto_front:
                composite_score = solution.values[0] * weights.get('win_rate', 0.25) + solution.values[1] * weights.get('profit_factor', 0.25) + solution.values[2] * weights.get('sharpe_ratio', 0.25) + solution.values[3] * weights.get('max_drawdown', 0.1) + solution.values[4] * weights.get('enhanced_prediction_performance', 0.15)
                if composite_score > best_score:
                    best_score = composite_score
                    best_solution = solution
            return best_solution
        except Exception:
            self.print(error('Error selecting Pareto solution: {e}'))
            return pareto_front[0] if pareto_front else None

    def _get_default_confidence_thresholds(self) -> dict[str, Any]:
        """Get default confidence thresholds."""
        return {'optimized_parameters': {'analyst_confidence_threshold': 0.7, 'tactician_confidence_threshold': 0.65, 'ensemble_confidence_threshold': 0.75, 'position_scale_up_threshold': 0.85, 'position_scale_down_threshold': 0.6, 'position_close_threshold': 0.3, 'enhanced_prediction_confidence_threshold': 0.7, 'enhanced_prediction_price_threshold': 0.6, 'analyst_ml_weight': 0.6, 'tactician_ml_weight': 0.4, 'position_sizing_confidence_multiplier': 1.5, 'leverage_sizing_risk_multiplier': 1.0, 'base_position_size': 0.1, 'max_position_size': 0.2, 'base_leverage': 50.0, 'max_leverage': 100.0}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_volatility_parameters(self) -> dict[str, Any]:
        """Get default volatility parameters."""
        return {'optimized_parameters': {'target_volatility': 0.15, 'volatility_lookback_period': 20, 'volatility_multiplier': 1.0, 'low_volatility_threshold': 0.02, 'medium_volatility_threshold': 0.05, 'high_volatility_threshold': 0.1, 'volatility_stop_loss_multiplier': 2.0}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_position_sizing_parameters(self) -> dict[str, Any]:
        """Get default position sizing parameters."""
        return {'optimized_parameters': {'base_position_size': 0.05, 'max_position_size': 0.25, 'min_position_size': 0.01, 'kelly_multiplier': 0.25, 'fractional_kelly': True, 'confidence_based_scaling': True, 'low_confidence_multiplier': 0.5, 'high_confidence_multiplier': 1.5}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_risk_management_parameters(self) -> dict[str, Any]:
        """Get default risk management parameters."""
        return {'optimized_parameters': {'stop_loss_atr_multiplier': 2.0, 'trailing_stop_atr_multiplier': 1.5, 'stop_loss_confidence_threshold': 0.3, 'enable_dynamic_stop_loss': True, 'volatility_based_sl': True, 'regime_based_sl': True, 'sl_tightening_threshold': 0.4, 'sl_loosening_threshold': 0.8}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_ensemble_parameters(self) -> dict[str, Any]:
        """Get default ensemble parameters."""
        return {'optimized_parameters': {'ensemble_method': 'confidence_weighted', 'analyst_weight': 0.4, 'tactician_weight': 0.3, 'strategist_weight': 0.3, 'min_ensemble_agreement': 0.7, 'max_ensemble_disagreement': 0.3}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_regime_parameters(self) -> dict[str, Any]:
        """Get default regime parameters."""
        return {'optimized_parameters': {'bull_trend_multiplier': 1.2, 'bear_trend_multiplier': 0.8, 'sideways_multiplier': 0.9, 'high_impact_multiplier': 0.6, 'sr_zone_multiplier': 1.1, 'regime_transition_threshold': 0.6, 'regime_confirmation_periods': 3}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _get_default_timing_parameters(self) -> dict[str, Any]:
        """Get default timing parameters."""
        return {'optimized_parameters': {'base_cooldown_minutes': 30, 'high_confidence_cooldown': 15, 'low_confidence_cooldown': 60, 'bull_trend_cooldown': 20, 'bear_trend_cooldown': 45, 'sideways_cooldown': 60, 'high_impact_cooldown': 90}, 'optimization_method': 'default', 'n_trials': 0, 'optimization_date': datetime.now().isoformat()}

    def _load_validation_frame(self) -> pd.DataFrame | None:
        try:
            exchange = self.config.get('exchange', 'BINANCE')
            symbol = self.config.get('symbol', 'ETHUSDT')
            data_dir = self.config.get('data_dir', 'data/training')
            path = f'{data_dir}/{exchange}_{symbol}_features_validation.pkl'
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame):
                    return df
        except Exception as e:
            self.logger.warning('⚠️ Validation frame load failed from step 4')
            msg = f'Validation frame not found: {path}. Step 12 requires features from Step 4.'
            raise FileNotFoundError(msg)

    def _evaluate_predictions(self, calibration_results: dict[str, Any], val_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, float]:
        """Compute metrics by applying confidence thresholds to calibrated ensembles/models on validation data.
        This approximates trading performance with simple proxies: win rate and returns from label correctness.
        """
        try:
            ens = None
            analyst_ensembles = calibration_results.get('analyst_ensembles', {})
            if analyst_ensembles:
                for payload in analyst_ensembles.values():
                    ce = payload.get('calibrated_ensemble') if isinstance(payload, dict) else None
                    if ce is not None:
                        ens = ce
                        break
            X = val_df.drop(columns=['label'], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
            y = val_df['label'].astype(int)
            if ens is not None and hasattr(ens, 'predict_proba'):
                proba = ens.predict_proba(X)
                pos_proba = proba[:, -1] if proba.shape[1] > 1 else proba[:, 0]
                preds = (pos_proba >= params.get('ensemble_confidence_threshold', 0.7)).astype(int)
            else:
                preds = np.zeros(len(y), dtype=int)
            (acc, float(accuracy_score(y, preds)))
            pnl = np.where(preds == y, 0.01, -0.01)
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            win_rate = float((preds == y).mean())
            avg_win = float(wins.mean()) if len(wins) else 0.01
            avg_loss = float(-losses.mean()) if len(losses) else 0.01
            cum = pnl.cumsum()
            drawdown = cum - np.maximum.accumulate(cum)
            max_drawdown = float(-drawdown.min()) if len(drawdown) else 0.0
            sharpe = float(pnl.mean() / (pnl.std() + 1e-09))
            enhanced_prediction_performance = self._evaluate_enhanced_prediction_performance(params, calibration_results)
            return {'accuracy': acc, 'win_rate': win_rate, 'avg_win': avg_win, 'avg_loss': avg_loss, 'max_drawdown': max_drawdown, 'sharpe_ratio': sharpe, 'enhanced_prediction_performance': enhanced_prediction_performance}
        except Exception:
            self.print(failed('Evaluation failed: {e}'))
            return {'accuracy': 0.5, 'win_rate': 0.5, 'avg_win': 0.01, 'avg_loss': 0.01, 'max_drawdown': 0.1, 'sharpe_ratio': 1.0}
from src.training.decorators import artifact_versioning, artifact_write_lock, circuit_breaker_protection, debug_training_step, deterministic_seed, idempotent_step, memory_efficient, nan_inf_and_constant_guard, prevent_data_leakage, quality_gate, resource_monitor, secure_data_processing, time_budget_watchdog, validate_step_output, validate_step_prerequisites
from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging, log_step_report, create_detailed_step_report, log_step_metrics, log_step_dataframe_with_standardized_name, log_step_artifact_with_standardized_name

@deterministic_seed(42)
@idempotent_step(step_key='step17_final_parameters_optimization')
@validates()
@timeout(timeout=10800)
@validates(required_directories=['data/training', 'models'], min_memory_gb=8.0, min_disk_gb=5.0, required_packages=['pandas', 'numpy', 'sklearn', 'optuna'], data_quality_checks={'min_rows': 1000, 'required_columns': ['timestamp', 'features', 'targets']}, context='Final Parameters Optimization')
@log_execution_time(memory_threshold_gb=16.0, cpu_threshold_percent=90.0, disk_threshold_gb=10.0, monitor_interval=60.0, auto_cleanup=True)
@cached(chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25)
@log_call(log_intermediate_results=True, save_debug_artifacts=True, performance_profiling=True, error_context_preservation=True)
@circuit_breaker(failure_threshold=3, recovery_timeout=600.0, expected_exception=Exception, monitor_interval=60.0)
@validates(required_files=['models/{exchange}_{symbol}_optimized_params.json'], data_quality_checks={'min_rows': 100, 'required_columns': ['predictions', 'probabilities']}, performance_thresholds={'optimization_time_minutes': 180.0, 'memory_usage_gb': 8.0}, format_validation=True)
async def run_step(symbol: str, exchange: str='BINANCE', data_dir: str='data/training', force_rerun: bool=False, **kwargs) -> bool:
    """Run the final parameters optimization step.

    Args:
        symbol: Trading symbol
        exchange: Exchange name
        data_dir: Data directory path
        **kwargs: Additional parameters

    Returns:
        bool: True if successful = False otherwise

    """
    try:
        config = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir}
        step = FinalParametersOptimizationStep(config)
        await step.initialize()
        training_input = {'symbol': symbol, 'exchange': exchange, 'data_dir': data_dir, 'force_rerun': force_rerun, **kwargs}
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        return result.get('status') == 'SUCCESS'
    except Exception:
        return False
if __name__ == '__main__':

    async def test() -> None:
        await run_step('ETHUSDT', 'BINANCE', 'data/training')
    asyncio.run(test())