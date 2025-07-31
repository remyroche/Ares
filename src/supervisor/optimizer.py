# src/supervisor/optimizer.py
import pandas as pd
import numpy as np
import copy
import uuid
import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import dump, load  # Import dump and load for checkpointing
from functools import partial
import os  # Import os for file existence checks
from typing import Optional, Union, Any, Dict

from backtesting.ares_data_preparer import (
    calculate_and_label_regimes,
    get_sr_levels,
)
from backtesting.ares_backtester import run_backtest
from backtesting.ares_deep_analyzer import calculate_detailed_metrics

from src.config import CONFIG
from src.utils.logger import system_logger

# Import both managers, but use the one passed in __init__
from src.database.sqlite_manager import SQLiteManager


class Optimizer:
    def __init__(
        self, config=CONFIG, db_manager: Union[SQLiteManager, None] = None
    ):  # Fixed: Accept generic db_manager
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.db_manager = db_manager  # Use the passed db_manager
        self.logger = system_logger.getChild("Optimizer")
        self.initial_equity = self.global_config["INITIAL_EQUITY"]

        self.optimized_params_csv = self.config.get(
            "optimized_params_csv", "reports/optimized_params_history.csv"
        )

        # Attributes to hold data for the duration of an optimization run
        # These will be passed from TrainingManager
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
            self.logger.error(
                "  Optimization data not loaded. Cannot run backtest evaluation."
            )
            return {
                "Final Equity": -np.inf,
                "Sharpe Ratio": -np.inf,
                "Max Drawdown (%)": np.inf,
                "Profit Factor": -np.inf,
                "Win Rate (%)": -np.inf,
            }

        try:
            # Ensure the correct trend_strength_threshold is passed if it's in params
            # trend_strength_threshold = (
            #     params.get("analyst", {})
            #     .get("market_regime_classifier", {})
            #     .get("trend_strength_threshold", 0.5)
            # )

            # Run optimization

            # Fixed: Ensure _optimization_sr_levels is not None before passing
            if self._optimization_sr_levels is None:
                self.logger.warning(
                    "  S/R levels not available for optimization. Proceeding with empty list."
                )
                sr_levels_to_pass = []
            else:
                sr_levels_to_pass = self._optimization_sr_levels

            prepared_df = calculate_and_label_regimes(
                self._optimization_klines_df.copy(),
                self._optimization_agg_trades_df.copy(),
                self._optimization_futures_df.copy(),
                params,
                sr_levels_to_pass,  # Pass the (possibly empty) S/R levels
                # trend_strength_threshold # This is now handled within calculate_and_label_regimes using params
            )

            if prepared_df.empty:
                self.logger.warning(
                    "    Prepared data is empty for this parameter set. Returning low scores."
                )
                return {
                    "Final Equity": -np.inf,
                    "Sharpe Ratio": -np.inf,
                    "Max Drawdown (%)": np.inf,
                    "Profit Factor": -np.inf,
                    "Win Rate (%)": -np.inf,
                }

            portfolio = run_backtest(prepared_df, params)
            num_days = (
                (prepared_df.index.max() - prepared_df.index.min()).days
                if not prepared_df.empty
                else 0
            )
            detailed_metrics = calculate_detailed_metrics(portfolio, num_days)

            self.logger.debug(
                f"    Backtest finished. Final Equity: ${detailed_metrics['Final Equity']:,.2f}"
            )
            return detailed_metrics
        except Exception as e:
            self.logger.error(f"  Error during backtest evaluation: {e}", exc_info=True)
            return {
                "Final Equity": -np.inf,
                "Sharpe Ratio": -np.inf,
                "Max Drawdown (%)": np.inf,
                "Profit Factor": -np.inf,
                "Win Rate (%)": -np.inf,
            }

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
        full_params = copy.deepcopy(self.global_config["best_params"])
        for param_path, value in opt_params.items():
            parts = param_path.split(".")
            self._set_param_value_at_path(full_params, parts, value)

        # Ensure integer parameters are cast correctly
        # Iterate through the optimization parameters defined in CONFIG
        opt_params_config = self.global_config["backtesting"]["optimization"]["params"]
        for param_path_key, param_def in opt_params_config.items():
            if param_def.get("type") == "int":
                parts = param_path_key.split(".")
                current_value = self._get_param_value_from_path(full_params, parts)
                if current_value is not None:
                    self._set_param_value_at_path(
                        full_params, parts, int(round(current_value))
                    )

        # Normalize weight parameters groups
        for group_path, keys in self.global_config["OPTIMIZATION_CONFIG"].get(
            "WEIGHT_PARAMS_GROUPS", []
        ):
            parts = group_path.split(".")
            weights_dict = self._get_param_value_from_path(full_params, parts)
            if weights_dict and isinstance(weights_dict, dict):
                total_weight = sum(weights_dict.get(k, 0) for k in keys)
                if total_weight > 0:
                    for k in keys:
                        self._set_param_value_at_path(
                            weights_dict, [k], weights_dict.get(k, 0) / total_weight
                        )
        return full_params

    def _define_optimization_dimensions(
        self, hpo_ranges: Optional[Dict[str, Any]] = None
    ):
        """
        Defines the search space for the Bayesian optimization.
        If hpo_ranges are provided, it uses them; otherwise, it falls back to the global config.
        """
        dimensions, param_names = [], []

        if hpo_ranges:
            self.logger.info(
                "Defining optimization dimensions from narrowed HPO ranges."
            )
            # Use the narrowed ranges from the coarse search
            for param_path, definition in hpo_ranges.items():
                if definition["type"] == "int":
                    dimensions.append(
                        Integer(definition["low"], definition["high"], name=param_path)
                    )
                elif definition["type"] == "float":
                    dimensions.append(
                        Real(definition["low"], definition["high"], name=param_path)
                    )
                param_names.append(param_path)
        else:
            self.logger.warning(
                "No HPO ranges provided. Defining dimensions from global config."
            )
            # Fallback to the original method of reading from the config
            opt_params_config = self.global_config["backtesting"]["optimization"][
                "params"
            ]

            # Iterate through the structured config to define dimensions
            def flatten_and_add_dims(d, parent_key=""):
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if (
                        isinstance(v, dict)
                        and "type" in v
                        and "low" in v
                        and "high" in v
                    ):
                        if v["type"] == "int":
                            dimensions.append(
                                Integer(v["low"], v["high"], name=new_key)
                            )
                        elif v["type"] == "float":
                            dimensions.append(Real(v["low"], v["high"], name=new_key))
                        param_names.append(new_key)
                    elif isinstance(v, dict):
                        flatten_and_add_dims(v, new_key)

            flatten_and_add_dims(opt_params_config)

        self.optimization_dimensions = dimensions
        self.optimization_param_names = param_names

    def _objective_function(self, params_list: list, strategy_breakdown_data_ref: Any):
        """
        The core function for Bayesian optimization. It runs a backtest with a given
        set of parameters and returns a composite performance score to be minimized.
        """
        candidate_opt_params = dict(zip(self.optimization_param_names, params_list))
        full_candidate_params = self._create_full_params_dict(candidate_opt_params)

        self.logger.info(f"  Evaluating candidate: {full_candidate_params}")

        metrics = self._evaluate_params_with_backtest(full_candidate_params)

        sharpe = metrics.get("Sharpe Ratio", -1e9)
        profit_factor = metrics.get("Profit Factor", -1e9)
        final_equity = metrics.get("Final Equity", self.initial_equity)
        max_drawdown = metrics.get("Max Drawdown (%)", 1e9)
        win_rate = metrics.get("Win Rate (%)", 0.0)

        # Weights for the different components of the score
        w_sharpe, w_profit_factor, w_equity, w_drawdown, w_win_rate = (
            0.5,
            0.2,
            0.2,
            0.1,
            0.1,
        )

        # Normalize profit factor to prevent extreme values from dominating
        normalized_profit_factor = np.log1p(profit_factor) if profit_factor > 0 else 0

        # Calculate the composite score
        composite_score = (
            w_sharpe * sharpe
            + w_profit_factor * normalized_profit_factor
            + w_equity * (final_equity / self.initial_equity - 1) * 100
            + w_win_rate * win_rate
            - w_drawdown * max_drawdown
        )

        self.logger.info(f"Candidate score: {composite_score:.4f}")

        # gp_minimize seeks to minimize the function, so we return the negative of our composite score.
        return -composite_score

    async def implement_global_system_optimization(
        self,
        historical_pnl_data: pd.DataFrame,
        strategy_breakdown_data: dict,
        checkpoint_file_path: Optional[str] = None,
        hpo_ranges: Optional[Dict[str, Any]] = None,  # Accept the narrowed ranges
        klines_df: Optional[pd.DataFrame] = None,  # New: Pass klines_df
        agg_trades_df: Optional[pd.DataFrame] = None,  # New: Pass agg_trades_df
        futures_df: Optional[pd.DataFrame] = None,  # New: Pass futures_df
    ):
        self.logger.info("\nRunning Final Fine-Tuned System Optimization (Stage 3b)...")

        # Store the passed dataframes for use in _evaluate_params_with_backtest
        self._optimization_klines_df = klines_df
        self._optimization_agg_trades_df = agg_trades_df
        self._optimization_futures_df = futures_df

        if self._optimization_klines_df is None or self._optimization_klines_df.empty:
            self.logger.error(
                "  No klines data provided for optimization. Skipping optimization."
            )
            return

        daily_df = self._optimization_klines_df.resample("D").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        daily_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        self._optimization_sr_levels = get_sr_levels(daily_df)

        # --- KEY CHANGE: Use hpo_ranges to define the search space ---
        self._define_optimization_dimensions(hpo_ranges=hpo_ranges)

        # --- The rest of the logic remains the same, but now operates on the new dimensions ---
        x0, y0, res = None, None, None
        n_calls = self.global_config["backtesting"]["optimization"].get(
            "bayesian_opt_n_calls", 20
        )
        n_initial_points = self.global_config["backtesting"]["optimization"].get(
            "bayesian_opt_n_initial_points", 5
        )

        if checkpoint_file_path and os.path.exists(checkpoint_file_path):
            try:
                res_loaded = load(checkpoint_file_path)
                x0, y0 = res_loaded.x_iters, res_loaded.func_vals
                n_calls_remaining = n_calls - len(x0)
                if n_calls_remaining > 0:
                    self.logger.info(
                        f"  Resuming Bayesian Optimization from checkpoint. {n_calls_remaining} calls remaining."
                    )
                    n_calls = n_calls_remaining
                else:
                    self.logger.info(
                        "  Bayesian Optimization already completed from checkpoint."
                    )
                    res = res_loaded
            except Exception as e:
                self.logger.warning(
                    f"  Failed to load optimization checkpoint: {e}. Starting fresh."
                )
                x0, y0 = None, None

        if res is None:
            if n_calls > 0:
                objective_with_feedback = partial(
                    self._objective_function,
                    strategy_breakdown_data_ref=strategy_breakdown_data,
                )
                res = gp_minimize(
                    func=objective_with_feedback,
                    dimensions=self.optimization_dimensions,
                    n_calls=n_calls,
                    n_initial_points=n_initial_points if x0 is None else 0,
                    x0=x0,
                    y0=y0,
                    random_state=42,
                    verbose=True,
                    acq_func="gp_hedge",
                )
                if checkpoint_file_path:
                    try:
                        dump(res, checkpoint_file_path, store_objective=False)
                        self.logger.info(
                            f"  Bayesian Optimization state saved to checkpoint: {checkpoint_file_path}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"  Error saving optimization checkpoint: {e}",
                            exc_info=True,
                        )
            else:
                self.logger.info("  No new optimization calls needed.")

        if res is None:
            self.logger.error("Optimization result is None. Cannot proceed.")
            return

        best_opt_params = dict(zip(self.optimization_param_names, res.x))
        best_overall_params = self._create_full_params_dict(best_opt_params)
        best_overall_performance = self._evaluate_params_with_backtest(
            best_overall_params
        )

        self.logger.info(
            f"\n--- Optimization Complete. Final Best Performance: {best_overall_performance} ---"
        )
        self.logger.info(f"  Final Optimized Parameters: {best_overall_params}")

        self._deep_update_dict(self.global_config["best_params"], best_overall_params)
        self.logger.info("  CONFIG.best_params updated with optimized values.")

        optimization_run_id = str(uuid.uuid4())
        date_applied = datetime.datetime.now().isoformat()

        if self.db_manager:
            params_doc = {
                "timestamp": date_applied,
                "optimization_run_id": optimization_run_id,
                "performance_metrics": best_overall_performance,
                "params": best_overall_params,
            }
            await self.db_manager.set_document(
                self.global_config["sqlite"][
                    "optimized_params_collection"
                ],  # Use sqlite collection name
                doc_id="latest",
                data=params_doc,
                is_public=True,
            )
            self.logger.info("Optimized parameters saved to DB.")

        return best_overall_params

    def _deep_update_dict(
        self, target_dict: Dict[str, Any], source_dict: Dict[str, Any]
    ):
        """Recursively updates a dictionary."""
        for key, value in source_dict.items():
            if (
                isinstance(value, dict)
                and key in target_dict
                and isinstance(target_dict[key], dict)
            ):
                self._deep_update_dict(target_dict[key], value)
            else:
                target_dict[key] = value
