# src/supervisor/optimizer.py
import pandas as pd
import numpy as np
import copy
import json
import uuid
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import dump, load # Import dump and load for checkpointing
from functools import partial
import os # Import os for file existence checks
from typing import Optional, Union, Any, Dict

from backtesting.ares_data_preparer import load_raw_data, calculate_and_label_regimes, get_sr_levels
from backtesting.ares_backtester import run_backtest
from backtesting.ares_deep_analyzer import calculate_detailed_metrics

from src.config import CONFIG
from src.utils.logger import system_logger as logger
# Import both managers, but use the one passed in __init__
from src.database.firestore_manager import FirestoreManager
from src.database.sqlite_manager import SQLiteManager


class Optimizer:
    def __init__(self, config=CONFIG, db_manager: Union[FirestoreManager, SQLiteManager, None] = None): # Fixed: Accept generic db_manager
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.db_manager = db_manager # Use the passed db_manager
        self.logger = logger.getChild('Optimizer')
        self.initial_equity = self.global_config['INITIAL_EQUITY']
        
        self.optimized_params_csv = self.config.get("optimized_params_csv", "reports/optimized_params_history.csv")

        # Attributes to hold data for the duration of an optimization run
        self._optimization_klines_df: Optional[pd.DataFrame] = None
        self._optimization_agg_trades_df: Optional[pd.DataFrame] = None
        self._optimization_futures_df: Optional[pd.DataFrame] = None
        self._optimization_sr_levels: Optional[list] = None
        self.optimization_param_names: list[str] = []
        self.optimization_dimensions: list[Union[Real, Integer]] = []


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
            # Ensure the correct trend_strength_threshold is passed if it's in params
            trend_strength_threshold = params.get('analyst', {}).get('market_regime_classifier', {}).get('trend_threshold', 25)

            # Fixed: Ensure _optimization_sr_levels is not None before passing
            if self._optimization_sr_levels is None:
                self.logger.warning("  S/R levels not available for optimization. Proceeding with empty list.")
                sr_levels_to_pass = []
            else:
                sr_levels_to_pass = self._optimization_sr_levels

            prepared_df = calculate_and_label_regimes(
                self._optimization_klines_df.copy(), 
                self._optimization_agg_trades_df.copy(), 
                self._optimization_futures_df.copy(), 
                params, 
                sr_levels_to_pass, # Pass the (possibly empty) S/R levels
                trend_strength_threshold 
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

        # Ensure integer parameters are cast correctly
        # Fixed: Use self.global_config['backtesting']['optimization']['params'] for iteration
        for param_path_key, param_def in self.global_config['backtesting']['optimization']['params'].items():
            if param_def.get('type') == 'int':
                parts = param_path_key.split('.')
                current_value = self._get_param_value_from_path(full_params, parts)
                if current_value is not None:
                    self._set_param_value_at_path(full_params, parts, int(round(current_value)))


        # Normalize weight parameters groups
        # Fixed: Iterate through the defined WEIGHT_PARAMS_GROUPS
        for group_path, keys in self.global_config['OPTIMIZATION_CONFIG'].get('WEIGHT_PARAMS_GROUPS', []):
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
        # Use the new structure for optimization parameters from CONFIG
        opt_params_config = self.global_config['backtesting']['optimization']['params']

        def add_dimensions_from_dict(d: dict, parent_key: str = ''): # Fixed: Added type hints
            for k, v in d.items():
                full_path = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict) and 'type' in v and 'low' in v and 'high' in v:
                    if v['type'] == 'int':
                        dimensions.append(Integer(v['low'], v['high'], name=full_path))
                    elif v['type'] == 'float':
                        dimensions.append(Real(v['low'], v['high'], name=full_path))
                    param_names.append(full_path)
                elif isinstance(v, dict):
                    add_dimensions_from_dict(v, full_path)

        add_dimensions_from_dict(opt_params_config)
        
        self.optimization_dimensions = dimensions
        self.optimization_param_names = param_names

    def _objective_function(self, params_list: list, strategy_breakdown_data_ref: Any): # Fixed: Type hint for strategy_breakdown_data_ref
        """
        The core function for Bayesian optimization. It runs a backtest with a given
        set of parameters and returns a composite performance score to be minimized.
        """
        candidate_opt_params = dict(zip(self.optimization_param_names, params_list))
        full_candidate_params = self._create_full_params_dict(candidate_opt_params)

        self.logger.info(f"  Evaluating candidate: {full_candidate_params}")
        
        metrics = self._evaluate_params_with_backtest(full_candidate_params)

        sharpe = metrics.get('Sharpe Ratio', -1e9)
        profit_factor = metrics.get('Profit Factor', -1e9)
        final_equity = metrics.get('Final Equity', self.initial_equity)
        max_drawdown = metrics.get('Max Drawdown (%)', 1e9)
        win_rate = metrics.get('Win Rate (%)', 0.0)

        # Weights for the different components of the score
        w_sharpe, w_profit_factor, w_equity, w_drawdown, w_win_rate = 0.5, 0.2, 0.2, 0.1, 0.1
        
        # Normalize profit factor to prevent extreme values from dominating
        normalized_profit_factor = np.log1p(profit_factor) if profit_factor > 0 else 0
        
        # Calculate the composite score
        composite_score = (
            w_sharpe * sharpe +
            w_profit_factor * normalized_profit_factor +
            w_equity * (final_equity / self.initial_equity - 1) * 100 +
            w_win_rate * win_rate -
            w_drawdown * max_drawdown
        )
        
        self.logger.info(f"Candidate score: {composite_score:.4f}")

        # gp_minimize seeks to minimize the function, so we return the negative of our composite score.
        return -composite_score

    async def implement_global_system_optimization(self, historical_pnl_data: pd.DataFrame, strategy_breakdown_data: dict, checkpoint_file_path: Optional[str] = None): # Fixed: Optional for checkpoint_file_path
        self.logger.info("\nRunning Global System Optimization (Bayesian Optimization)...")
        
        self.logger.info("  Loading raw data for optimization...")
        self._optimization_klines_df, self._optimization_agg_trades_df, self._optimization_futures_df = load_raw_data()
        if self._optimization_klines_df is None or self._optimization_klines_df.empty:
            self.logger.error("  Failed to load raw data. Skipping optimization.")
            return

        daily_df = self._optimization_klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        self._optimization_sr_levels = get_sr_levels(daily_df)
        
        self._define_optimization_dimensions()
        
        x0 = None
        y0 = None
        n_calls = self.global_config['backtesting']['optimization'].get("bayesian_opt_n_calls", 20)
        n_initial_points = self.global_config['backtesting']['optimization'].get("bayesian_opt_n_initial_points", 5)

        # --- Check for existing optimization checkpoint ---
        res = None # Initialize res to None
        if checkpoint_file_path and os.path.exists(checkpoint_file_path):
            try:
                res_loaded = load(checkpoint_file_path)
                x0 = res_loaded.x_iters
                y0 = res_loaded.func_vals
                n_calls_remaining = n_calls - len(x0)
                if n_calls_remaining > 0:
                    self.logger.info(f"  Resuming Bayesian Optimization from checkpoint. {len(x0)} points already evaluated. {n_calls_remaining} calls remaining.")
                    n_calls = n_calls_remaining
                else:
                    self.logger.info("  Bayesian Optimization already completed from checkpoint. Skipping new calls.")
                    res = res_loaded # Use the loaded result directly
            except Exception as e:
                self.logger.warning(f"  Failed to load optimization checkpoint: {e}. Starting optimization from scratch.")
                x0 = None
                y0 = None
        
        if res is None: # Only run gp_minimize if res is not already loaded from checkpoint
            if x0 is None: # If no checkpoint or failed to load, start fresh
                initial_params_values = [self._get_param_value_from_path(self.global_config['BEST_PARAMS'], name.split('.')) for name in self.optimization_param_names]
                x0 = [initial_params_values] # Provide initial point if starting fresh
                self.logger.info(f"  Starting Bayesian Optimization with {n_calls} calls (fresh run).")
            
            # Only run gp_minimize if there are calls remaining
            if n_calls > 0:
                objective_with_feedback = partial(self._objective_function, strategy_breakdown_data_ref=strategy_breakdown_data)

                res = gp_minimize(
                    func=objective_with_feedback,
                    dimensions=self.optimization_dimensions,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points if x0 is None else 0, # Only use initial points if starting fresh
                    x0=x0,
                    y0=y0,
                    random_state=self.global_config['backtesting']['optimization'].get("bayesian_opt_random_state", 42),
                    verbose=True,
                    acq_func="gp_hedge"
                )

                # Save checkpoint after optimization completes
                if checkpoint_file_path:
                    try:
                        dump(res, checkpoint_file_path, store_objective=False) # store_objective=False to save space
                        self.logger.info(f"  Bayesian Optimization state saved to checkpoint: {checkpoint_file_path}")
                    except Exception as e:
                        self.logger.error(f"  Error saving optimization checkpoint: {e}", exc_info=True)
            else:
                self.logger.info("  No new optimization calls needed. Using loaded result.")
                # If n_calls was 0 from the start and res was not loaded, this path might be problematic.
                # Ensure 'res' is set if n_calls was 0 and it was supposed to be loaded.
                # This case is handled by `if res is None:` block.

        if res is None: # Final check in case res was not set by any path
            self.logger.error("Optimization result (res) is None after all attempts. Cannot proceed.")
            return # Or raise an exception

        best_opt_params = dict(zip(self.optimization_param_names, res.x))
        best_overall_params = self._create_full_params_dict(best_opt_params)
        best_overall_performance = self._evaluate_params_with_backtest(best_overall_params)

        self.logger.info(f"\n--- Optimization Complete. Final Best Performance: {best_overall_performance} ---")
        self.logger.info(f"  Final Optimized Parameters: {best_overall_params}")

        # Update the global CONFIG with the new best parameters
        self._deep_update_dict(self.global_config['best_params'], best_overall_params)
        self.logger.info("  CONFIG.best_params updated with optimized values.")

        optimization_run_id = str(uuid.uuid4())
        date_applied = datetime.datetime.now().isoformat()

        # Fixed: Use the generic db_manager for storing optimized params
        if self.db_manager:
            params_doc = {
                "timestamp": date_applied, "optimization_run_id": optimization_run_id,
                "performance_metrics": best_overall_performance, "date_applied": date_applied,
                "params": best_overall_params
            }
            await self.db_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'], # Table name will be this string
                doc_id=optimization_run_id, data=params_doc, is_public=True
            )
            await self.db_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'],
                doc_id='latest', data=params_doc, is_public=True
            )
            self.logger.info("Optimized parameters saved to DB.")
        else:
            self.logger.warning("DB Manager is None. Cannot save optimized parameters to DB.")

        try:
            performance_metrics_str = json.dumps(best_overall_performance)
            with open(self.optimized_params_csv, 'a') as f:
                f.write(f"{date_applied},{optimization_run_id},{performance_metrics_str},{date_applied},{json.dumps(best_overall_params)}\n")
            self.logger.info("Optimized parameters exported to CSV.")
        except Exception as e:
            self.logger.error(f"Error exporting optimized parameters to CSV: {e}")

    def _deep_update_dict(self, target_dict: Dict[str, Any], source_dict: Dict[str, Any]): # Fixed: Type hints
        """Recursively updates a dictionary."""
        for key, value in source_dict.items():
            if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
                self._deep_update_dict(target_dict[key], value)
            else:
                target_dict[key] = value

