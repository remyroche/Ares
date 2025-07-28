# src/supervisor/supervisor.py
import pandas as pd
import numpy as np
import os
import datetime
import json # For handling nested dicts in logs if needed
import asyncio # For async Firestore operations
import uuid # For generating unique IDs for optimization runs
import itertools # For generating parameter combinations in optimization

# Import scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer # For defining search space dimensions
from skopt.utils import use_named_args # For using named arguments in objective function

# Import the main CONFIG dictionary
from config import CONFIG
from utils.logger import system_logger
from database.firestore_manager import FirestoreManager # New import

# Import backtesting components
from backtesting.ares_data_preparer import load_raw_data, get_sr_levels, calculate_and_label_regimes
from backtesting.ares_backtester import run_backtest, PortfolioManager
from backtesting.ares_deep_analyzer import calculate_detailed_metrics # Import for detailed metrics


class Supervisor:
    """
    The Supervisor module (Meta-Learning Governor) optimizes the entire trading strategy
    and manages capital allocation over long time horizons. It also handles enhanced
    performance reporting.
    """
    def __init__(self, config=CONFIG):
        self.config = config.get("supervisor", {})
        self.global_config = config # Store global config to access BEST_PARAMS etc.
        self.initial_equity = self.global_config['INITIAL_EQUITY'] # Access from CONFIG
        self.firestore_manager = firestore_manager
        self.logger = system_logger.getChild('Supervisor') # Child logger for Supervisor
        
        # Current capital allocation multiplier, adjusted dynamically
        self.allocated_capital_multiplier = self.config.get("initial_allocated_capital_multiplier", 1.0)

        self.daily_summary_log_filename = self.config.get("daily_summary_log_filename", "reports/daily_summary_log.csv")
        self.strategy_performance_log_filename = self.config.get("strategy_performance_log_filename", "reports/strategy_performance_log.csv")
        self.optimized_params_csv = self.config.get("optimized_params_csv", "reports/optimized_params_history.csv")
        self.model_metadata_csv = self.config.get("model_metadata_csv", "reports/model_metadata_history.csv")

        # Ensure reports directory exists
        os.makedirs(os.path.dirname(self.daily_summary_log_filename), exist_ok=True)
        
        # Initialize CSV headers if files don't exist
        self._initialize_daily_summary_csv()
        self._initialize_strategy_performance_csv()
        self._initialize_optimized_params_csv()
        self._initialize_model_metadata_csv()

        # Store historical data for optimization, loaded once
        self._optimization_klines_df = None
        self._optimization_agg_trades_df = None
        self._optimization_futures_df = None
        self._optimization_sr_levels = None

    def _initialize_daily_summary_csv(self):
        """Ensures the daily summary CSV file exists with correct headers."""
        if not os.path.exists(self.daily_summary_log_filename):
            with open(self.daily_summary_log_filename, 'w') as f:
                f.write("Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n")
            self.logger.info(f"Created daily summary log: {self.daily_summary_log_filename}")

    def _initialize_strategy_performance_csv(self):
        """Ensures the strategy performance CSV file exists with correct headers."""
        if not os.path.exists(self.strategy_performance_log_filename):
            with open(self.strategy_performance_log_filename, 'w') as f:
                f.write("Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n")
            self.logger.info(f"Created strategy performance log: {self.strategy_performance_log_filename}")

    def _initialize_optimized_params_csv(self):
        """Ensures the optimized parameters CSV file exists with correct headers."""
        if not os.path.exists(self.optimized_params_csv):
            with open(self.optimized_params_csv, 'w') as f:
                f.write("Timestamp,OptimizationRunID,PerformanceMetric,DateApplied,Parameters\n")
            self.logger.info(f"Created optimized parameters log: {self.optimized_params_csv}")

    def _initialize_model_metadata_csv(self):
        """Ensures the model metadata CSV file exists with correct headers."""
        if not os.path.exists(self.model_metadata_csv):
            with open(self.model_metadata_csv, 'w') as f:
                f.write("ModelName,Version,TrainingDate,PerformanceMetrics,FilePathReference,ConfigSnapshot\n")
            self.logger.info(f"Created model metadata log: {self.model_metadata_csv}")

    def _evaluate_params_with_backtest(self, params: dict) -> dict:
        """
        Evaluates a given set of parameters by running a backtest.
        Returns a dictionary of performance metrics including Final Equity, Sharpe Ratio,
        Max Drawdown, Profit Factor, and Win Rate.
        """
        self.logger.debug("  Evaluating parameter set with backtest...")
        
        # Use pre-loaded optimization data
        klines_df = self._optimization_klines_df
        agg_trades_df = self._optimization_agg_trades_df
        futures_df = self._optimization_futures_df
        sr_levels = self._optimization_sr_levels

        if klines_df is None or klines_df.empty:
            self.logger.error("  Optimization data not loaded. Cannot run backtest evaluation.")
            return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                    'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

        try:
            # Prepare data with the current parameter set
            prepared_df = calculate_and_label_regimes(
                klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), params, sr_levels,
                params.get('trend_strength_threshold', 25) # Fallback if not explicitly in params
            )
            
            if prepared_df.empty:
                self.logger.warning("    Prepared data is empty for this parameter set. Returning low scores.")
                return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                        'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

            # Run backtest with the current parameter set
            portfolio = run_backtest(prepared_df, params)
            
            # Calculate detailed metrics
            num_days = (prepared_df.index.max() - prepared_df.index.min()).days if not prepared_df.empty else 0
            detailed_metrics = calculate_detailed_metrics(portfolio, num_days)
            
            self.logger.debug(f"    Backtest finished. Final Equity: ${detailed_metrics['Final Equity']:,.2f}, "
                             f"Sharpe: {detailed_metrics['Sharpe Ratio']:.2f}, "
                             f"Max Drawdown: {detailed_metrics['Max Drawdown (%)']:.2f}%, "
                             f"Profit Factor: {detailed_metrics['Profit Factor']:.2f}, "
                             f"Win Rate: {detailed_metrics['Win Rate (%)']:.2f}%")
            return detailed_metrics
        except Exception as e:
            self.logger.error(f"  Error during backtest evaluation for params {params}: {e}", exc_info=True)
            return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf, 
                    'Profit Factor': -np.inf, 'Win Rate (%)': -np.inf}

    def _get_param_value_from_path(self, base_dict, path_parts):
        """Helper to get a nested parameter value from a dictionary using a list of path parts."""
        current = base_dict
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None # Path not found
        return current

    def _set_param_value_at_path(self, base_dict, path_parts, value):
        """Helper to set a nested parameter value in a dictionary using a list of path parts."""
        current = base_dict
        for i, part in enumerate(path_parts):
            if i == len(path_parts) - 1:
                current[part] = value
            else:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]

    def _create_flat_params_dict(self, nested_params: dict) -> dict:
        """
        Converts a nested parameter dictionary (from optimization) into a flat dictionary
        matching the structure expected by `calculate_and_label_regimes` and `run_backtest`.
        Also handles weight normalization and integer conversions.
        """
        # Start with a deep copy of the current BEST_PARAMS from global_config
        flat_params = copy.deepcopy(self.global_config['BEST_PARAMS']) 

        # The 'nested_params' here is actually a flat dict of param_path:value from skopt.
        # We need to apply these values to a copy of BEST_PARAMS.
        for param_path, value in nested_params.items():
            parts = param_path.split('.')
            self._set_param_value_at_path(flat_params, parts, value)

        # Handle INTEGER_PARAMS
        for param_path in self.global_config['OPTIMIZATION_CONFIG']['INTEGER_PARAMS']:
            parts = param_path.split('.')
            current_value = self._get_param_value_from_path(flat_params, parts)
            if current_value is not None:
                self._set_param_value_at_path(flat_params, parts, int(round(current_value)))

        # Handle WEIGHT_PARAMS_GROUPS normalization
        for group_path, keys in self.global_config['OPTIMIZATION_CONFIG']['WEIGHT_PARAMS_GROUPS']:
            parts = group_path.split('.')
            weights_dict = self._get_param_value_from_path(flat_params, parts)
            
            if weights_dict and isinstance(weights_dict, dict):
                total_weight = sum(weights_dict.get(k, 0) for k in keys)
                if total_weight > 0:
                    for k in keys:
                        self._set_param_value_at_path(weights_dict, [k], weights_dict.get(k, 0) / total_weight)
                else: # If all weights are zero, distribute evenly
                    num_keys = len(keys)
                    if num_keys > 0:
                        for k in keys:
                            self._set_param_value_at_path(weights_dict, [k], 1.0 / num_keys)

        return flat_params

    def _define_optimization_dimensions(self):
        """
        Defines the search space dimensions for Bayesian Optimization based on OPTIMIZATION_CONFIG.
        Returns a list of skopt.space Dimension objects and a list of corresponding parameter names.
        """
        dimensions = []
        param_names = []

        # Use COARSE_GRID_RANGES as the primary source for defining dimensions
        for param_path, values in self.global_config['OPTIMIZATION_CONFIG']['COARSE_GRID_RANGES'].items():
            if param_path in self.global_config['OPTIMIZATION_CONFIG']['INTEGER_PARAMS']:
                dimensions.append(Integer(min(values), max(values), name=param_path))
            else:
                dimensions.append(Real(min(values), max(values), name=param_path))
            param_names.append(param_path)
        
        return dimensions, param_names

    def _objective_function(self, params_list, strategy_breakdown_data_ref):
        """
        Objective function for Bayesian Optimization.
        Takes a list of parameter values, converts to a dict, runs backtest,
        and returns a scalar score to be minimized (so, negative of performance).
        """
        # Map flat list of params back to a dictionary with named arguments
        # This requires param_names to be accessible. We'll pass it via partial or closure.
        # For simplicity with use_named_args, we assume params_list order matches dimensions.
        
        # Convert the flat list of parameters from skopt into a dictionary
        # This assumes the order of params_list corresponds to self.optimization_param_names
        candidate_nested_params = {}
        for i, param_name in enumerate(self.optimization_param_names):
            self._set_param_value_at_path(candidate_nested_params, param_name.split('.'), params_list[i])

        # Convert to the flat structure expected by backtesting
        flat_candidate_params = self._create_flat_params_dict(candidate_nested_params)

        self.logger.info(f"  Evaluating candidate: {flat_candidate_params}")
        metrics = self._evaluate_params_with_backtest(flat_candidate_params)

        # Define a composite score to maximize (so return negative for minimization)
        # Prioritize Sharpe, then Profit Factor, then Equity, then minimize Drawdown, maximize Win Rate
        
        # Handle cases where metrics might be -inf or inf
        sharpe = metrics['Sharpe Ratio'] if np.isfinite(metrics['Sharpe Ratio']) else -1e9
        profit_factor = metrics['Profit Factor'] if np.isfinite(metrics['Profit Factor']) else -1e9
        final_equity = metrics['Final Equity'] if np.isfinite(metrics['Final Equity']) else self.initial_equity # Use self.initial_equity
        max_drawdown = metrics['Max Drawdown (%)'] if np.isfinite(metrics['Max Drawdown (%)']) else 1e9
        win_rate = metrics['Win Rate (%)'] if np.isfinite(metrics['Win Rate (%)']) else 0.0

        # Weights for composite score (can be optimized or configured)
        w_sharpe = 0.5
        w_profit_factor = 0.2
        w_equity = 0.2
        w_drawdown = 0.1 # Negative weight as we want to minimize drawdown
        w_win_rate = 0.1

        # Normalize metrics to a similar scale if necessary, or use their raw values carefully
        # For simplicity, let's use raw values and adjust weights.
        
        # Ensure profit_factor is not negative for log scaling if used
        normalized_profit_factor = np.log1p(profit_factor) if profit_factor > 0 else 0 # log1p(x) = log(1+x)
        
        # Composite score (to be maximized)
        composite_score = (
            w_sharpe * sharpe +
            w_profit_factor * normalized_profit_factor + # Use normalized profit factor
            w_equity * (final_equity / self.initial_equity - 1) * 100 + # % return on initial equity, use self.initial_equity
            w_win_rate * win_rate - # Win rate directly
            w_drawdown * max_drawdown # Penalize drawdown
        )

        # --- Explicit Regime-Specific Feedback ---
        # Identify underperforming regimes from the previous period's data
        underperforming_regimes = [
            regime for regime, perf in strategy_breakdown_data_ref.items() 
            if perf.get('NetPnL', 0) < 0 and perf.get('TotalTrades', 0) > 0 # Negative PnL and at least one trade
        ]

        if underperforming_regimes:
            # This part is conceptual as we don't have per-regime backtest results for the *current* candidate.
            # A true implementation would re-run backtests for specific regimes or have a model
            # that predicts regime-specific performance for the candidate parameters.
            
            # For this implementation, we apply a general penalty if the overall Sharpe is low
            # AND there were underperforming regimes in the *previous* period.
            # This encourages the optimizer to find more robust parameters across all regimes.
            
            # Heuristic: If overall Sharpe is poor (e.g., < 0.5) AND there were underperforming regimes,
            # apply a small additional penalty to encourage finding more stable parameters.
            if sharpe < 0.5: # Example threshold for poor Sharpe
                penalty_magnitude = 0.1 # Small penalty
                composite_score -= penalty_magnitude * len(underperforming_regimes) # Larger penalty for more bad regimes
                self.logger.info(f"  Penalty applied due to underperforming regimes: {underperforming_regimes}")

        # Bayesian optimization minimizes, so return the negative of the score
        return -composite_score


    async def _implement_global_system_optimization(self, historical_pnl_data: pd.DataFrame, strategy_breakdown_data: dict):
        """
        Implements Global System Optimization (Meta-Learning) using Bayesian Optimization.
        It integrates with the backtesting pipeline to tune parameters.
        """
        self.logger.info("\nSupervisor: Running Global System Optimization (Meta-Learning) - Bayesian Optimization...")
        
        # Load raw data once for the optimization process and store it
        self.logger.info("  Loading raw data for optimization backtests...")
        self._optimization_klines_df, self._optimization_agg_trades_df, self._optimization_futures_df = load_raw_data()
        if self._optimization_klines_df is None or self._optimization_klines_df.empty:
            self.logger.error("  Failed to load raw data for optimization. Skipping optimization.")
            return

        daily_df = self._optimization_klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        self._optimization_sr_levels = get_sr_levels(daily_df)
        
        # Define search space dimensions
        self.optimization_dimensions, self.optimization_param_names = self._define_optimization_dimensions()
        
        # Initial evaluation of current BEST_PARAMS
        # Access BEST_PARAMS from global_config
        initial_params_values = [self._get_param_value_from_path(self.global_config['BEST_PARAMS'], name.split('.')) 
                                 for name in self.optimization_param_names]
        
        # Ensure initial_params_values are within the defined dimensions (e.g., integer rounding)
        for i, dim in enumerate(self.optimization_dimensions):
            if isinstance(dim, Integer):
                initial_params_values[i] = int(round(initial_params_values[i]))
            else: # Real
                initial_params_values[i] = float(initial_params_values[i])
            # Clip to bounds if necessary (skopt handles this internally but good for consistency)
            initial_params_values[i] = max(dim.low, min(dim.high, initial_params_values[i]))

        # Pass strategy_breakdown_data to the objective function
        # We need to wrap the objective function to pass additional arguments
        from functools import partial
        objective_with_feedback = partial(self._objective_function, strategy_breakdown_data_ref=strategy_breakdown_data)

        # Run Bayesian Optimization
        n_calls = self.config.get("bayesian_opt_n_calls", 20) # Number of optimization iterations
        n_initial_points = self.config.get("bayesian_opt_n_initial_points", 5) # Number of random initial points

        self.logger.info(f"  Starting Bayesian Optimization with {n_calls} calls ({n_initial_points} initial random points).")
        
        # gp_minimize returns the result of the optimization
        # `res.x` contains the best parameters found (as a list)
        # `res.fun` contains the objective function value at `res.x` (minimized value)
        res = gp_minimize(
            func=objective_with_feedback,
            dimensions=self.optimization_dimensions,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            x0=[initial_params_values], # Provide initial point (current BEST_PARAMS)
            random_state=self.config.get("bayesian_opt_random_state", 42),
            verbose=True, # Show optimization progress
            acq_func="gp_hedge" # Acquisition function
        )

        # Extract best parameters found by the optimizer
        best_params_list = res.x
        
        # Convert the best parameters list back to a nested dictionary
        best_candidate_nested_params = {}
        for i, param_name in enumerate(self.optimization_param_names):
            self._set_param_value_at_path(best_candidate_nested_params, param_name.split('.'), best_params_list[i])

        # Create the final flat params dictionary for updating CONFIG.BEST_PARAMS
        best_overall_params = self._create_flat_params_dict(best_candidate_nested_params)
        
        # Re-evaluate the best parameters to get their full metrics
        best_overall_performance = self._evaluate_params_with_backtest(best_overall_params)

        self.logger.info(f"\n--- Optimization Complete. Final Best Performance: {best_overall_performance} ---")
        self.logger.info(f"  Final Optimized Parameters: {best_overall_params}")

        # Update CONFIG.BEST_PARAMS with the newly found optimal parameters
        self._deep_update_dict(self.global_config['BEST_PARAMS'], best_overall_params)
        self.logger.info("  CONFIG.BEST_PARAMS updated with optimized values.")

        optimization_run_id = str(uuid.uuid4())
        date_applied = datetime.datetime.now().isoformat()

        # Save to Firestore
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            params_doc = {
                "timestamp": date_applied,
                "optimization_run_id": optimization_run_id,
                "performance_metrics": best_overall_performance, # Store the dictionary of metrics
                "date_applied": date_applied,
                "params": best_overall_params # Store the actual parameters
            }
            # Save to a document with a unique ID and also update a 'latest' document
            await self.firestore_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'],
                doc_id=optimization_run_id,
                data=params_doc,
                is_public=True
            )
            await self.firestore_manager.set_document(
                self.global_config['firestore']['optimized_params_collection'],
                doc_id='latest', # Update a special 'latest' document
                data=params_doc,
                is_public=True
            )
            self.logger.info("Optimized parameters saved to Firestore.")

        # Export to CSV
        try:
            # Prepare performance metrics for CSV: convert dict to JSON string
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

    async def _calculate_dynamic_capital_allocation(self, historical_pnl_data: pd.DataFrame):
        """
        Adjusts the capital allocation multiplier based on recent performance.
        :param historical_pnl_data: DataFrame with daily P&L (at least 'Date' and 'NetPnL' columns).
        """
        self.logger.info("Supervisor: Calculating Dynamic Capital Allocation...")
        lookback_days = self.config.get("risk_allocation_lookback_days", 30)
        max_increase_pct = self.config.get("max_capital_allocation_increase_pct", 1.0)
        max_decrease_pct = self.config.get("max_capital_allocation_decrease_pct", 0.75)

        if historical_pnl_data.empty or len(historical_pnl_data) < lookback_days:
            self.logger.warning(f"Insufficient historical P&L data ({len(historical_pnl_data)} days) for dynamic allocation. Need at least {lookback_days} days. Keeping current allocation.")
            return # Keep current allocation

        recent_pnl = historical_pnl_data['NetPnL'].tail(lookback_days)
        total_pnl_over_lookback = recent_pnl.sum()
        
        # Simple performance metric: average daily P&L percentage
        # Normalize P&L by initial equity to get a comparable performance metric
        # Use current allocated capital for normalization to reflect actual capital deployed
        current_effective_capital = self.initial_equity * self.allocated_capital_multiplier
        if current_effective_capital == 0:
            self.logger.warning("Current effective capital is zero, cannot calculate avg daily P&L pct for allocation.")
            return

        avg_daily_pnl_pct = total_pnl_over_lookback / (lookback_days * current_effective_capital)

        # Adjust multiplier based on performance
        adjustment_factor = 0.1 # This can be a config parameter too
        
        change_in_multiplier = avg_daily_pnl_pct * adjustment_factor * lookback_days # Scale by lookback days

        new_multiplier = self.allocated_capital_multiplier + change_in_multiplier

        # Apply bounds: -75% to +100% relative to the *initial* allocated capital multiplier (1.0)
        min_allowed_multiplier = 1.0 - max_decrease_pct
        max_allowed_multiplier = 1.0 + max_increase_pct

        self.allocated_capital_multiplier = np.clip(new_multiplier, min_allowed_multiplier, max_allowed_multiplier)
        
        self.logger.info(f"Dynamic Capital Allocation: Total P&L over {lookback_days} days: ${total_pnl_over_lookback:,.2f}")
        self.logger.info(f"New Allocated Capital Multiplier: {self.allocated_capital_multiplier:.2f}x (Effective Capital: ${self.get_current_allocated_capital():,.2f})")

    def get_current_allocated_capital(self):
        """Returns the current dynamically allocated capital."""
        return self.initial_equity * self.allocated_capital_multiplier

    async def generate_performance_report(self, trade_logs: list, current_date: datetime.date):
        """
        Generates a detailed performance report for the given day.
        :param trade_logs: List of dictionaries, each representing a completed trade.
        :param current_date: The date for which the report is being generated.
        :return: Dictionary containing daily summary and strategy breakdown.
        """
        self.logger.info(f"Supervisor: Generating Performance Report for {current_date}...")
        
        if not trade_logs:
            self.logger.info("No trades recorded for this period. Generating empty report.")
            daily_summary = {
                "Date": current_date.strftime('%Y-%m-%d'),
                "TotalTrades": 0, "WinRate": 0.0, "NetPnL": 0.0,
                "MaxDrawdown": 0.0, "EndingCapital": self.get_current_allocated_capital(),
                "AllocatedCapitalMultiplier": self.allocated_capital_multiplier
            }
            strategy_breakdown = {}
        else:
            df_trades = pd.DataFrame(trade_logs)
            
            # Ensure PnL is numeric
            if 'realized_pnl_usd' in df_trades.columns:
                df_trades['realized_pnl_usd'] = pd.to_numeric(df_trades['realized_pnl_usd'], errors='coerce').fillna(0)
            else:
                df_trades['realized_pnl_usd'] = 0.0 # Default if column missing

            # Daily Summary Metrics
            total_trades = len(df_trades)
            wins = df_trades[df_trades['realized_pnl_usd'] > 0]
            win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
            net_pnl = df_trades['realized_pnl_usd'].sum()

            # Calculate Max Drawdown for the day's trades
            # This requires building an equity curve from the trades
            equity_curve_values = [self.get_current_allocated_capital()] # Start with allocated capital
            for pnl in df_trades['realized_pnl_usd']:
                equity_curve_values.append(equity_curve_values[-1] + pnl)
            
            equity_series = pd.Series(equity_curve_values)
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = -drawdown.min() * 100 if not drawdown.empty else 0.0 # Convert to positive percentage

            ending_capital = self.get_current_allocated_capital() + net_pnl # Very simplified for demo

            daily_summary = {
                "Date": current_date.strftime('%Y-%m-%d'),
                "TotalTrades": total_trades,
                "WinRate": round(win_rate, 2),
                "NetPnL": round(net_pnl, 2),
                "MaxDrawdown": round(max_drawdown, 2),
                "EndingCapital": round(ending_capital, 2),
                "AllocatedCapitalMultiplier": round(self.allocated_capital_multiplier, 2)
            }

            # Strategy Performance Breakdown
            strategy_breakdown = {}
            if 'market_state_at_entry' in df_trades.columns:
                for regime in df_trades['market_state_at_entry'].unique():
                    regime_trades = df_trades[df_trades['market_state_at_entry'] == regime]
                    regime_total_trades = len(regime_trades)
                    regime_wins = regime_trades[regime_trades['realized_pnl_usd'] > 0]
                    regime_win_rate = (len(regime_wins) / regime_total_trades * 100) if regime_total_trades > 0 else 0.0
                    regime_net_pnl = regime_trades['realized_pnl_usd'].sum()
                    regime_avg_pnl_per_trade = regime_net_pnl / regime_total_trades if regime_total_trades > 0 else 0.0
                    
                    # TradeDuration remains a placeholder as entry_timestamp is not available in trade_logs
                    regime_trade_duration = 0.0 

                    strategy_breakdown[regime] = {
                        "TotalTrades": regime_total_trades,
                        "WinRate": round(regime_win_rate, 2),
                        "NetPnL": round(regime_net_pnl, 2),
                        "AvgPnLPerTrade": round(regime_avg_pnl_per_trade, 2),
                        "TradeDuration": round(regime_trade_duration, 2)
                    }
        
        self.logger.info("Performance Report Generated.")
        return {"daily_summary": daily_summary, "strategy_breakdown": strategy_breakdown}

    async def _update_daily_summary_csv_and_firestore(self, daily_summary: dict):
        """Appends the daily summary to the CSV log and saves to Firestore."""
        try:
            # CSV
            row = [
                daily_summary["Date"],
                daily_summary["TotalTrades"],
                daily_summary["WinRate"],
                daily_summary["NetPnL"],
                daily_summary["MaxDrawdown"],
                daily_summary["EndingCapital"],
                daily_summary["AllocatedCapitalMultiplier"]
            ]
            with open(self.daily_summary_log_filename, 'a') as f:
                f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended daily summary for {daily_summary['Date']} to CSV.")

            # Firestore
            if self.firestore_manager and self.firestore_manager.firestore_enabled:
                await self.firestore_manager.set_document(
                    self.global_config['firestore']['daily_summary_log_filename'].split('/')[-1].replace('.csv', ''), # Collection name
                    doc_id=daily_summary["Date"], # Use date as document ID
                    data=daily_summary,
                    is_public=False # Private per user
                )
                self.logger.info(f"Saved daily summary for {daily_summary['Date']} to Firestore.")

        except Exception as e:
            self.logger.error(f"Error updating daily summary (CSV/Firestore): {e}", exc_info=True)

    async def _update_strategy_performance_log_and_firestore(self, current_date: datetime.date, strategy_breakdown: dict):
        """Appends strategy performance breakdown to its CSV log and saves to Firestore."""
        try:
            # CSV
            with open(self.strategy_performance_log_filename, 'a') as f:
                for regime, metrics in strategy_breakdown.items():
                    row = [
                        current_date.strftime('%Y-%m-%d'),
                        regime,
                        metrics["TotalTrades"],
                        metrics["WinRate"],
                        metrics["NetPnL"],
                        metrics["AvgPnLPerTrade"],
                        metrics["TradeDuration"]
                    ]
                    f.write(",".join(map(str, row)) + "\n")
            self.logger.info(f"Appended strategy performance for {current_date} to CSV.")

            # Firestore
            if self.firestore_manager and self.firestore_manager.firestore_enabled:
                for regime, metrics in strategy_breakdown.items():
                    doc_data = {
                        "date": current_date.isoformat(),
                        "regime": regime,
                        **metrics
                    }
                    await self.firestore_manager.add_document(
                        self.global_config['firestore']['strategy_performance_log_filename'].split('/')[-1].replace('.csv', ''), # Collection name
                        data=doc_data,
                        is_public=False # Private per user
                    )
                self.logger.info(f"Saved strategy performance for {current_date} to Firestore.")

        except Exception as e:
            self.logger.error(f"Error updating strategy performance log (CSV/Firestore): {e}", exc_info=True)

    async def orchestrate_supervision(self, current_date: datetime.date, total_equity: float, 
                                daily_trade_logs: list, daily_pnl_per_regime: dict,
                                historical_daily_pnl_data: pd.DataFrame):
        """
        Main orchestration method for the Supervisor, called periodically (e.g., daily/weekly).
        :param current_date: The current date of the supervision cycle.
        :param total_equity: The current total equity of the trading system.
        :param daily_trade_logs: List of dictionaries, each representing a completed trade.
        :param daily_pnl_per_regime: Dictionary of P&L attributed to each regime for the current day.
        :param historical_daily_pnl_data: DataFrame of historical daily P&L for dynamic allocation.
        """
        self.logger.info(f"\n--- Supervisor: Starting Orchestration for {current_date} ---")

        # 1. Dynamic Risk Allocation
        # Update historical_daily_pnl_data with today's P&L before passing
        today_net_pnl = sum(t.get('realized_pnl_usd', 0) for t in daily_trade_logs)
        new_daily_pnl_row = pd.DataFrame([{
            'Date': current_date,
            'NetPnL': today_net_pnl
        }])
        if 'Date' in historical_daily_pnl_data.columns:
            historical_daily_pnl_data['Date'] = pd.to_datetime(historical_daily_pnl_data['Date'])
        
        updated_historical_pnl = pd.concat([historical_daily_pnl_data, new_daily_pnl_row]).drop_duplicates(subset=['Date']).sort_values('Date')
        
        await self._calculate_dynamic_capital_allocation(updated_historical_pnl)

        # 2. Performance Reporting
        report = await self.generate_performance_report(daily_trade_logs, current_date)
        await self._update_daily_summary_csv_and_firestore(report["daily_summary"])
        await self._update_strategy_performance_log_and_firestore(current_date, report["strategy_breakdown"])

        # 3. Global System Optimization (Meta-Learning) - Run periodically
        if current_date.day % self.config.get("meta_learning_frequency_days", 7) == 0:
            # Pass the detailed regime-specific performance for conceptual feedback
            await self._implement_global_system_optimization(updated_historical_pnl, report["strategy_breakdown"])
        
        self.logger.info(f"--- Supervisor: Orchestration Complete for {current_date} ---")
        return {
            "allocated_capital": self.get_current_allocated_capital(),
            "daily_summary": report["daily_summary"],
            "strategy_breakdown": report["strategy_breakdown"]
        }

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Supervisor Module Demonstration...")

    # Create dummy directories if they don't exist
    os.makedirs("reports", exist_ok=True)
    os.makedirs("logs", exist_ok=True) # For logger
    os.makedirs("data_cache", exist_ok=True) # For backtesting data
    os.makedirs("models/analyst", exist_ok=True) # For analyst models

    # Initialize logger for demo
    from importlib import reload
    import utils.logger
    reload(utils.logger)
    system_logger = utils.logger.system_logger

    # Initialize FirestoreManager for demo (will be disabled if not configured)
    # In a real Canvas environment, __app_id, __firebase_config, __initial_auth_token are provided
    # For local testing, you'd need to set these env vars or provide dummy values.
    # Set firestore enabled to False for local demo if no firebase setup
    CONFIG['firestore']['enabled'] = False 
    
    firestore_manager_demo = FirestoreManager(
        config=CONFIG,
        app_id="demo-app",
        firebase_config_str="{}", # Dummy config
        initial_auth_token=None
    )
    
    supervisor = Supervisor(config=CONFIG, firestore_manager=firestore_manager_demo)

    # Simulate historical daily P&L for dynamic allocation
    historical_pnl = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2024-06-01', periods=40, freq='D')),
        'NetPnL': np.random.randn(40) * 500 # Simulate some daily P&L
    })
    historical_pnl.loc[historical_pnl.index[:20], 'NetPnL'] = np.random.rand(20) * 1000 - 200 # Mixed
    historical_pnl.loc[historical_pnl.index[20:], 'NetPnL'] = np.random.rand(20) * 1500 + 100 # More profitable recently

    # Simulate daily run over a few days
    start_date = datetime.date(2024, 7, 25)
    num_days_to_simulate = 5

    current_total_equity = CONFIG['INITIAL_EQUITY'] # Start with initial equity from CONFIG

    # Create dummy data files for backtesting
    # Access filenames from CONFIG
    klines_filename = CONFIG['KLINES_FILENAME']
    agg_trades_filename = CONFIG['AGG_TRADES_FILENAME']
    futures_filename = CONFIG['FUTURES_FILENAME']
    from analyst.data_utils import create_dummy_data
    create_dummy_data(klines_filename, 'klines')
    create_dummy_data(agg_trades_filename, 'agg_trades')
    create_dummy_data(futures_filename, 'futures')

    async def run_demo():
        nonlocal current_total_equity
        nonlocal historical_pnl
        for i in range(num_days_to_simulate):
            sim_date = start_date + datetime.timedelta(days=i)
            
            print(f"\n--- Simulating Day: {sim_date} ---")

            # Simulate trades for the day
            num_trades_today = np.random.randint(5, 20)
            daily_trades = []
            daily_pnl_by_regime = {}

            for _ in range(num_trades_today):
                pnl = np.random.randn() * 100 # Simulate P&L for each trade
                regime = np.random.choice(["BULL_TREND", "BEAR_TREND", "SIDEWAYS_RANGE", "SR_ZONE_ACTION"])
                trade_log = {
                    "trade_id": f"T{sim_date.strftime('%Y%m%d')}-{_}",
                    "timestamp": sim_date.isoformat(),
                    "asset": CONFIG['SYMBOL'], # Access SYMBOL from CONFIG
                    "direction": np.random.choice(["LONG", "SHORT"]),
                    "market_state_at_entry": regime,
                    "entry_price": 2000 + np.random.rand() * 100,
                    "exit_price": 2000 + np.random.rand() * 100,
                    "position_size": np.random.rand() * 0.5 + 0.1,
                    "leverage_used": np.random.randint(25, 75),
                    "confidence_score_at_entry": np.random.rand(),
                    "lss_at_entry": np.random.rand() * 100,
                    "fees_paid": abs(pnl) * 0.001,
                    "funding_rate_pnl": np.random.rand() * 0.0001 - 0.00005,
                    "realized_pnl_usd": pnl,
                    "exit_reason": np.random.choice(["Take Profit", "Stop Loss", "Manual Close"])
                }
                daily_trades.append(trade_log)

                if regime not in daily_pnl_by_regime:
                    daily_pnl_by_regime[regime] = 0.0
                daily_pnl_by_regime[regime] += pnl

            current_total_equity += sum(t.get('realized_pnl_usd', 0) for t in daily_trades)

            supervisor_output = await supervisor.orchestrate_supervision(
                current_date=sim_date,
                total_equity=current_total_equity,
                daily_trade_logs=daily_trades,
                daily_pnl_per_regime=daily_pnl_by_regime,
                historical_daily_pnl_data=historical_pnl.copy()
            )
            print(f"Day {sim_date} Summary: Allocated Capital: ${supervisor_output['allocated_capital']:,.2f}, Net P&L: ${supervisor_output['daily_summary']['NetPnL']:,.2f}")
            
            historical_pnl = pd.concat([historical_pnl, pd.DataFrame([{'Date': sim_date, 'NetPnL': supervisor_output['daily_summary']['NetPnL']}])]).drop_duplicates(subset=['Date']).sort_values('Date')


        print("\nSupervisor Module Demonstration Complete.")
        print(f"Check '{supervisor.daily_summary_log_filename}', '{supervisor.strategy_performance_log_filename}', '{supervisor.optimized_params_csv}', '{supervisor.model_metadata_csv}' for logs.")

    asyncio.run(run_demo())
