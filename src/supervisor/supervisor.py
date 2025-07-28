# src/supervisor/supervisor.py
import pandas as pd
import numpy as np
import os
import datetime
import json # For handling nested dicts in logs if needed
import asyncio # For async Firestore operations
import uuid # For generating unique IDs for optimization runs
import itertools # For generating parameter combinations in optimization

# Assume these are available in the same package or through sys.path
from config import CONFIG, INITIAL_EQUITY, BEST_PARAMS # Import BEST_PARAMS
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
    def __init__(self, config=CONFIG, firestore_manager: FirestoreManager = None):
        self.config = config.get("supervisor", {})
        self.global_config = config # Store global config to access BEST_PARAMS etc.
        self.initial_equity = INITIAL_EQUITY # Total capital available to the system
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

    def _evaluate_params_with_backtest(self, params: dict, klines_df, agg_trades_df, futures_df, sr_levels) -> dict:
        """
        Evaluates a given set of parameters by running a backtest.
        Returns a dictionary of performance metrics including Final Equity, Sharpe Ratio, and Max Drawdown.
        """
        self.logger.info("  Evaluating parameter set with backtest...")
        try:
            # Prepare data with the current parameter set
            prepared_df = calculate_and_label_regimes(
                klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), params, sr_levels,
                params.get('trend_strength_threshold', 25) # Fallback if not explicitly in params
            )
            
            if prepared_df.empty:
                self.logger.warning("    Prepared data is empty for this parameter set. Returning low scores.")
                return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf}

            # Run backtest with the current parameter set
            portfolio = run_backtest(prepared_df, params)
            
            # Calculate detailed metrics
            num_days = (prepared_df.index.max() - prepared_df.index.min()).days if not prepared_df.empty else 0
            detailed_metrics = calculate_detailed_metrics(portfolio, num_days)
            
            self.logger.info(f"    Backtest finished. Final Equity: ${detailed_metrics['Final Equity']:,.2f}, "
                             f"Sharpe: {detailed_metrics['Sharpe Ratio']:.2f}, "
                             f"Max Drawdown: {detailed_metrics['Max Drawdown (%)']:.2f}%")
            return detailed_metrics
        except Exception as e:
            self.logger.error(f"  Error during backtest evaluation for params {params}: {e}", exc_info=True)
            return {'Final Equity': -np.inf, 'Sharpe Ratio': -np.inf, 'Max Drawdown (%)': np.inf}

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
        flat_params = self.global_config['BEST_PARAMS'].copy() # Start with a copy of default BEST_PARAMS

        # Iterate through all parameters in OPTIMIZATION_CONFIG to ensure they are set
        # This handles both coarse and fine-tuning parameters
        all_opt_params = {}
        all_opt_params.update(self.global_config['OPTIMIZATION_CONFIG']['COARSE_GRID_RANGES'])
        # If fine-tuning, ranges are dynamically created, so we rely on nested_params having them.

        for param_path in all_opt_params.keys():
            parts = param_path.split('.')
            value = self._get_param_value_from_path(nested_params, parts)
            
            if value is not None:
                self._set_param_value_at_path(flat_params, parts, value)
            else:
                # If a parameter path from OPTIMIZATION_CONFIG is not in nested_params,
                # ensure it defaults to its value in BEST_PARAMS
                default_value = self._get_param_value_from_path(self.global_config['BEST_PARAMS'], parts)
                if default_value is not None:
                    self._set_param_value_at_path(flat_params, parts, default_value)


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


    async def _implement_global_system_optimization(self, historical_pnl_data: pd.DataFrame, strategy_breakdown_data: dict):
        """
        Implements Global System Optimization (Meta-Learning).
        This method integrates with the backtesting pipeline to tune parameters
        using a two-stage approach: Coarse Grid Search followed by Fine-Tuning (Random Search).
        """
        self.logger.info("\nSupervisor: Running Global System Optimization (Meta-Learning) - Two-Stage Optimization...")
        
        # Load raw data once for the optimization process
        self.logger.info("  Loading raw data for optimization backtests...")
        klines_df, agg_trades_df, futures_df = load_raw_data()
        if klines_df is None or klines_df.empty:
            self.logger.error("  Failed to load raw data for optimization. Skipping optimization.")
            return

        daily_df = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        sr_levels = get_sr_levels(daily_df)
        
        best_overall_params = self.global_config['BEST_PARAMS'].copy()
        # Initialize best performance with a dictionary of metrics
        best_overall_performance = self._evaluate_params_with_backtest(best_overall_params, klines_df, agg_trades_df, futures_df, sr_levels)
        self.logger.info(f"  Initial BEST_PARAMS performance: {best_overall_performance}")

        # --- Stage 1: Coarse Grid Search ---
        self.logger.info("\n--- Stage 1: Coarse Grid Search ---")
        coarse_grid_ranges = self.global_config['OPTIMIZATION_CONFIG']['COARSE_GRID_RANGES']
        
        # Generate all combinations for coarse grid
        keys = coarse_grid_ranges.keys()
        values = coarse_grid_ranges.values()
        
        coarse_param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        self.logger.info(f"  Evaluating {len(coarse_param_combinations)} coarse parameter combinations.")

        for i, coarse_candidate_nested in enumerate(coarse_param_combinations):
            # Transform nested candidate to flat params for backtesting
            flat_candidate_params = self._create_flat_params_dict(coarse_candidate_nested)
            
            self.logger.info(f"  [Coarse {i+1}/{len(coarse_param_combinations)}] Testing: {flat_candidate_params}")
            candidate_performance = self._evaluate_params_with_backtest(flat_candidate_params, klines_df, agg_trades_df, futures_df, sr_levels)
            
            # Compare performance: Prioritize Sharpe Ratio, then Final Equity, then minimize Max Drawdown
            if (candidate_performance['Sharpe Ratio'] > best_overall_performance['Sharpe Ratio']) or \
               (candidate_performance['Sharpe Ratio'] == best_overall_performance['Sharpe Ratio'] and \
                candidate_performance['Final Equity'] > best_overall_performance['Final Equity']) or \
               (candidate_performance['Sharpe Ratio'] == best_overall_performance['Sharpe Ratio'] and \
                candidate_performance['Final Equity'] == best_overall_performance['Final Equity'] and \
                candidate_performance['Max Drawdown (%)'] < best_overall_performance['Max Drawdown (%)']):
                
                best_overall_performance = candidate_performance.copy() # Store the dictionary of metrics
                best_overall_params = flat_candidate_params.copy() # Store the flat best params
                self.logger.info(f"    New best coarse found! Performance: {best_overall_performance}")
        
        self.logger.info(f"\n--- Stage 1 Complete. Best Coarse Performance: {best_overall_performance} ---")
        self.logger.info(f"  Best Coarse Params: {best_overall_params}")

        # --- Stage 2: Fine-Tuning (Random Search around best coarse) ---
        self.logger.info("\n--- Stage 2: Fine-Tuning (Random Search) ---")
        fine_tune_multiplier = self.global_config['OPTIMIZATION_CONFIG']['FINE_TUNE_RANGES_MULTIPLIER']
        num_fine_tune_samples = self.global_config['OPTIMIZATION_CONFIG']['FINE_TUNE_NUM_POINTS']

        # Only proceed to fine-tuning if coarse search yielded profitable results
        if best_overall_performance['Final Equity'] <= INITIAL_EQUITY:
            self.logger.warning("  Best coarse performance not profitable. Skipping fine-tuning.")
        else:
            self.logger.info(f"  Running {num_fine_tune_samples} random samples for fine-tuning around best coarse params.")
            
            for i in range(num_fine_tune_samples):
                fine_tune_candidate_nested = {}
                
                # Perturb each parameter that was in the coarse grid
                for param_path, coarse_values in coarse_grid_ranges.items():
                    current_best_val = self._get_param_value_from_path(best_overall_params, param_path.split('.'))
                    
                    if current_best_val is None: # Fallback if param not found in best_overall_params
                        current_best_val = np.mean(coarse_values)

                    # Define fine-tune range around current_best_val
                    if isinstance(current_best_val, (int, float)):
                        variation = abs(current_best_val * fine_tune_multiplier)
                        low_bound = current_best_val - variation
                        high_bound = current_best_val + variation
                        
                        # Ensure bounds are reasonable
                        if len(coarse_values) > 0:
                            low_bound = max(low_bound, min(coarse_values))
                            high_bound = min(high_bound, max(coarse_values))

                        if low_bound == high_bound: # Avoid division by zero in uniform
                            fine_tuned_val = low_bound
                        else:
                            fine_tuned_val = np.random.uniform(low_bound, high_bound)
                        
                        if param_path in self.global_config['OPTIMIZATION_CONFIG']['INTEGER_PARAMS']:
                            fine_tuned_val = int(round(fine_tuned_val))
                        else:
                            fine_tuned_val = round(fine_tuned_val, 4) # Round floats for consistency

                        self._set_param_value_at_path(fine_tune_candidate_nested, param_path.split('.'), fine_tuned_val)
                    # Handle weight groups here if they were part of coarse grid and need specific perturbation
                    # For now, assuming simple numerical parameters.

                # Transform nested candidate to flat params for backtesting
                flat_candidate_params = self._create_flat_params_dict(fine_tune_candidate_nested)
                
                self.logger.info(f"  [Fine-Tune {i+1}/{num_fine_tune_samples}] Testing: {flat_candidate_params}")
                candidate_performance = self._evaluate_params_with_backtest(flat_candidate_params, klines_df, agg_trades_df, futures_df, sr_levels)
                
                # Compare performance: Prioritize Sharpe Ratio, then Final Equity, then minimize Max Drawdown
                if (candidate_performance['Sharpe Ratio'] > best_overall_performance['Sharpe Ratio']) or \
                   (candidate_performance['Sharpe Ratio'] == best_overall_performance['Sharpe Ratio'] and \
                    candidate_performance['Final Equity'] > best_overall_performance['Final Equity']) or \
                   (candidate_performance['Sharpe Ratio'] == best_overall_performance['Sharpe Ratio'] and \
                    candidate_performance['Final Equity'] == best_overall_performance['Final Equity'] and \
                    candidate_performance['Max Drawdown (%)'] < best_overall_performance['Max Drawdown (%)']):
                    
                    best_overall_performance = candidate_performance.copy()
                    best_overall_params = flat_candidate_params.copy()
                    self.logger.info(f"    New best fine-tuned found! Performance: {best_overall_performance}")

        self.logger.info(f"\n--- Optimization Complete. Final Best Performance: {best_overall_performance} ---")
        self.logger.info(f"  Final Optimized Parameters: {best_overall_params}")

        # --- Feedback Loop from Regime-Specific Performance (Conceptual) ---
        # A real meta-learning algorithm would use `strategy_breakdown_data` to inform
        # its search. For example:
        # - If 'BULL_TREND' regime performance is consistently poor, the optimizer might
        #   focus on tuning parameters related to trend-following indicators or entry/exit
        #   thresholds within that regime.
        # - If 'SIDEWAYS_RANGE' performance is excellent, it might try to make the system
        #   more aggressive in that regime by adjusting relevant parameters.
        # This feedback would typically be integrated into the objective function or the
        # proposal mechanism of the optimization algorithm.
        self.logger.info(f"  (Conceptual) Meta-learning informed by regime performance: {strategy_breakdown_data}")

        # Update CONFIG.BEST_PARAMS with the newly found optimal parameters
        # This is a critical step for the pipeline to use the optimized parameters.
        # Deep update ensures nested dictionaries are handled.
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

    current_total_equity = INITIAL_EQUITY # Start with initial equity

    # Create dummy data files for backtesting
    from config import KLINES_FILENAME, AGG_TRADES_FILENAME, FUTURES_FILENAME
    from analyst.data_utils import create_dummy_data
    create_dummy_data(KLINES_FILENAME, 'klines')
    create_dummy_data(AGG_TRADES_FILENAME, 'agg_trades')
    create_dummy_data(FUTURES_FILENAME, 'futures')

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
                    "asset": "ETHUSDT",
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
