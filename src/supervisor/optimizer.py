# src/supervisor/optimizer.py
import pandas as pd
import numpy as np
import copy
import json
import uuid
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from functools import partial

from src.config import CONFIG
from src.utils.logger import system_logger
from backtesting.ares_data_preparer import load_raw_data, get_sr_levels, calculate_and_label_regimes
from backtesting.ares_backtester import run_backtest
from backtesting.ares_deep_analyzer import calculate_detailed_metrics

class Optimizer:
    def __init__(self, config=CONFIG, firestore_manager=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('Optimizer')
        self.initial_equity = self.global_config['INITIAL_EQUITY']
        
        self.optimized_params_csv = self.config.get("optimized_params_csv", "reports/optimized_params_history.csv")

        # Attributes to hold data for the duration of an optimization run
        self._optimization_klines_df = None
        self._optimization_agg_trades_df = None
        self._optimization_futures_df = None
        self._optimization_sr_levels = None
        self.optimization_param_names = []
        self.optimization_dimensions = []


    def _evaluate_params_with_backtest(self, params: dict) -> dict:
        """
        Evaluates a given set of parameters by running a backtest.
        Returns a dictionary of performance metrics.
        """
        self.logger.debug("  Evaluating parameter set with backtest...")
        
        if self._optimization_klines_df is None or self._optimization_klines_df.empty:
            self.logger.error("  Optimization data not loaded. Cannot run backtest evaluation.")
            return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                    'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

        try:
            prepared_df = calculate_and_label_regimes(
                self._optimization_klines_df.copy(), 
                self._optimization_agg_trades_df.copy(), 
                self._optimization_futures_df.copy(), 
                params, 
                self._optimization_sr_levels,
                params.get('trend_strength_threshold', 25)
            )
            
            if prepared_df.empty:
                self.logger.warning("    Prepared data is empty for this parameter set. Returning low scores.")
                return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                        'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

            portfolio = run_backtest(prepared_df, params)
            num_days = (prepared_df.index.max() - prepared_df.index.min()).days if not prepared_df.empty else 0
            detailed_metrics = calculate_detailed_metrics(portfolio, num_days)
            
            self.logger.debug(f"    Backtest finished. Final Equity: ${detailed_metrics['Final Equity']:,.2f}")
            return detailed_metrics
        except Exception as e:
            self.logger.error(f"  Error during backtest evaluation: {e}", exc_info=True)
            return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                    'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

    def _get_param_value_from_path(self, base_dict, path_parts):
        current = base_dict
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _set_param_value_at_path(self, base_dict, path_parts, value):
        current = base_dict
        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

    def _create_full_params_dict(self, opt_params: dict) -> dict:
        full_params = copy.deepcopy(self.global_config['BEST_PARAMS']) 
        for param_path, value in opt_params.items():
            parts = param_path.split('.')
            self._set_param_value_at_path(full_params, parts, value)

        for param_path in self.global_config['OPTIMIZATION_CONFIG']['INTEGER_PARAMS']:
            parts = param_path.split('.')
            current_value = self._get_param_value_from_path(full_params, parts)
            if current_value is not None:
                self._set_param_value_at_path(full_params, parts, int(round(current_value)))

        for group_path, keys in self.global_config['OPTIMIZATION_CONFIG']['WEIGHT_PARAMS_GROUPS']:
            parts = group_path.split('.')
            weights_dict = self._get_param_value_from_path(full_params, parts)
            if weights_dict and isinstance(weights_dict, dict):
                total_weight = sum(weights_dict.get(k, 0) for k in keys)
                if total_weight > 0:
                    for k in keys:
                        self._set_param_value_at_path(weights_dict, [k], weights_dict.get(k, 0) / total_weight)
        return full_params

    def _define_optimization_dimensions(self):
        dimensions, param_names = [], []
        for param_path, values in self.global_config['OPTIMIZATION_CONFIG']['COARSE_GRID_RANGES'].items():
            if param_path in self.global_config['OPTIMIZATION_CONFIG']['INTEGER_PARAMS']:
                dimensions.append(Integer(min(values), max(values), name=param_path))
            else:
                dimensions.append(Real(min(values), max(values), name=param_path))
            param_names.append(param_path)
        self.optimization_dimensions = dimensions
        self.optimization_param_names = param_names

    def _objective_function(self, params_list, strategy_breakdown_data_ref):
        candidate_opt_params = dict(zip(self.optimization_param_names, params_list))
        full_candidate_params = self._create_full_params_dict(candidate_opt_params)

        self.logger.info(f"  Evaluating candidate: {full_candidate_params}")
        metrics = self._evaluate_params_with_backtest(full_candidate_params)

        sharpe = metrics.get('Sharpe Ratio', -1e9)
        profit_factor = metrics.get('Profit Factor', -1e9)
        final_equity = metrics.get('Final Equity', self.initial_equity)
        max_drawdown = metrics.get('Max Drawdown (%)', 1e9)
        win_rate = metrics.get('Win Rate (%)', 0.0)

        w_sharpe, w_profit_factor, w_equity, w_drawdown, w_win_rate = 0.5, 0.2, 0.2, 0.1, 0.1
        
        normalized_profit_factor = np.log1p(profit_factor) if profit_factor > 0 else 0
        
        composite_score = (
            w_sharpe * sharpe +
            w_profit_factor * normalized_profit_factor +
            w_equity * (final_equity / self.initial_equity - 1) * 100 +
            w_win_rate * win_rate -
            w_drawdown * max_drawdown
        )
        
        return -composite_score

    async def implement_global_system_optimization(self, historical_pnl_data: pd.DataFrame, strategy_breakdown_data: dict):
        self.logger.info("\nRunning Global System Optimization (Bayesian Optimization)...")
        
        self.logger.info("  Loading raw data for optimization...")
        self._optimization_klines_df, self._optimization_agg_trades_df, self._optimization_futures_df = load_raw_data()
        if self._optimization_klines_df is None or self._optimization_klines_df.empty:
            self.logger.error("  Failed to load raw data. Skipping optimization.")
            return

        daily_df = self._optimization_klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        self._optimization_sr_levels = get_sr_levels(daily_df)
        
        self._define_optimization_dimensions()
        
        initial_params_values = [self._get_param_value_from_path(self.global_config['BEST_PARAMS'], name.split('.')) for name in self.optimization_param_names]
        
        objective_with_feedback = partial(self._objective_function, strategy_breakdown_data_ref=strategy_breakdown_data)

        n_calls = self.config.get("bayesian_opt_n_calls", 20)
        n_initial_points = self.config.get("bayesian_opt_n_initial_points", 5)

        self.logger.info(f"  Starting Bayesian Optimization with {n_calls} calls...")

        res = gp_minimize(
            func=objective_with_feedback,
            dimensions=self.optimization_dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            x0=[initial_params_values],
            random_state=self.config.get("bayesian_opt_random_state", 42),
            verbose=True,
            acq_func="gp_hedge"
        )

        best_opt_params = dict(zip(self.optimization_param_names, res.x))
        best_overall_params = self._create_full_params_dict(best_opt_params)
        best_overall_performance = self._evaluate_params_with_backtest(best_overall_params)

        self.logger.info(f"\n--- Optimization Complete. Final Best Performance: {best_overall_performance} ---")
        self.logger.info(f"  Final Optimized Parameters: {best_overall_params}")

        self._deep_update_dict(self.global_config['BEST_PARAMS'], best_overall_params)
        self.logger.info("  CONFIG.BEST_PARAMS updated with optimized values.")

        optimization_run_id = str(uuid.uuid4())
        date_applied = datetime.datetime.now().isoformat()

        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            params_doc = {
                "timestamp": date_applied, "optimization_run_id": optimization_run_id,
                "performance_metrics": best_overall_performance, "date_applied": date_applied,
                "params": best_overall_params
            }
            await self.firestore_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'],
                doc_id=optimization_run_id, data=params_doc, is_public=True
            )
            await self.firestore_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'],
                doc_id='latest', data=params_doc, is_public=True
            )
            self.logger.info("Optimized parameters saved to Firestore.")

        try:
            performance_metrics_str = json.dumps(best_overall_performance)
            with open(self.optimized_params_csv, 'a') as f:
                f.write(f"{date_applied},{optimization_run_id},{performance_metrics_str},{date_applied},{json.dumps(best_overall_params)}\n")
            self.logger.info("Optimized parameters exported to CSV.")
        except Exception as e:
            self.logger.error(f"Error exporting optimized parameters to CSV: {e}")

    def _deep_update_dict(self, target_dict, source_dict):
        """Recursively updates a dictionary."""
        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                self._deep_update_dict(target_dict[key], value)
            else:
                target_dict[key] = value
