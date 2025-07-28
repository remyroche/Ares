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

# Import backtesting components (conceptual import for direct use in optimization)
# In a real system, you'd likely have a dedicated backtesting service/module
# that can be invoked with specific parameters.
# For this demo, we'll simulate calling these functions.
from backtesting.ares_data_preparer import load_raw_data, get_sr_levels, calculate_and_label_regimes
from backtesting.ares_backtester import run_backtest, PortfolioManager


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

    def _evaluate_params_with_backtest(self, params: dict, klines_df, agg_trades_df, futures_df, sr_levels) -> float:
        """
        Evaluates a given set of parameters by running a backtest.
        Returns a performance metric (e.g., Sharpe Ratio or final equity).
        """
        self.logger.info("  Evaluating parameter set with backtest...")
        try:
            # Prepare data with the current parameter set
            # Ensure 'trend_strength_threshold' is in params or handled by calculate_and_label_regimes
            prepared_df = calculate_and_label_regimes(
                klines_df.copy(), agg_trades_df.copy(), futures_df.copy(), params, sr_levels,
                params.get('trend_strength_threshold', 25) # Fallback if not explicitly in params
            )
            
            if prepared_df.empty:
                self.logger.warning("    Prepared data is empty for this parameter set. Returning low score.")
                return -np.inf # Return a very low score if no data

            # Run backtest with the current parameter set
            portfolio = run_backtest(prepared_df, params)
            
            # Use final equity as the performance metric for simplicity in this demo
            # In a real system, you'd use Sharpe, Sortino, Calmar, etc.
            performance_metric = portfolio.equity
            self.logger.info(f"    Backtest finished. Final Equity: ${performance_metric:,.2f}")
            return performance_metric
        except Exception as e:
            self.logger.error(f"  Error during backtest evaluation for params {params}: {e}", exc_info=True)
            return -np.inf # Return a very low score on error

    async def _implement_global_system_optimization(self, historical_pnl_data: pd.DataFrame, strategy_breakdown_data: dict):
        """
        Implements Global System Optimization (Meta-Learning) using a random search approach.
        It integrates with the backtesting pipeline to tune parameters.
        """
        self.logger.info("\nSupervisor: Running Global System Optimization (Meta-Learning) using Random Search...")
        
        # Load raw data once for the optimization process
        self.logger.info("  Loading raw data for optimization backtests...")
        klines_df, agg_trades_df, futures_df = load_raw_data()
        if klines_df is None or klines_df.empty:
            self.logger.error("  Failed to load raw data for optimization. Skipping optimization.")
            return

        daily_df = klines_df.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        sr_levels = get_sr_levels(daily_df)
        
        # Define a concrete search space for key parameters
        # This is a subset of parameters from CONFIG.BEST_PARAMS for demonstration
        # In a real system, this would be more extensive and configurable.
        search_space = {
            "trade_entry_threshold": [0.5, 0.6, 0.7, 0.8],
            "sl_atr_multiplier": [1.0, 1.5, 2.0, 2.5],
            "take_profit_rr": [1.5, 2.0, 2.5, 3.0],
            "adx_period": [10, 14, 20, 25],
            "initial_leverage": [20, 25, 30, 40], # Corresponds to tactician.laddering.initial_leverage
            "min_lss_for_ladder": [60, 65, 70, 75], # Corresponds to tactician.laddering.min_lss_for_ladder
            "ladder_step_leverage_increase": [3, 5, 7, 10], # Corresponds to tactician.laddering.ladder_step_leverage_increase
            "trend_strength_threshold": [20, 25, 30, 35] # For regime classifier
        }

        # Create a list of all possible parameter combinations
        keys = search_space.keys()
        values = search_space.values()
        
        # Generate all combinations (for small search spaces) or sample randomly (for large spaces)
        # For this implementation, we'll iterate through a fixed number of random samples
        num_random_samples = 20 # Number of parameter sets to test in each optimization run

        current_best_params = self.global_config['BEST_PARAMS'].copy()
        best_performance = self._evaluate_params_with_backtest(current_best_params, klines_df, agg_trades_df, futures_df, sr_levels)
        self.logger.info(f"  Initial BEST_PARAMS performance: ${best_performance:,.2f}")

        # Store the best parameters found during this optimization run
        optimized_params_found = current_best_params.copy()
        
        self.logger.info(f"  Running {num_random_samples} random samples...")

        for i in range(num_random_samples):
            # Construct a random candidate parameter set
            candidate_params = self.global_config['BEST_PARAMS'].copy() # Start with current best as base
            
            # Randomly select a few parameters to change for this candidate
            params_to_change = np.random.choice(list(keys), size=min(len(keys), 3), replace=False) # Change 3 random params

            for param_key in params_to_change:
                # Get a random value from the defined range for this parameter
                random_value = np.random.choice(search_space[param_key])
                
                # Update the nested parameter path in the candidate_params dictionary
                parts = param_key.split('.')
                temp_dict = candidate_params
                for part in parts[:-1]:
                    temp_dict = temp_dict.setdefault(part, {})
                temp_dict[parts[-1]] = random_value

            self.logger.info(f"  Evaluating candidate {i+1}/{num_random_samples}: {candidate_params}")
            candidate_performance = self._evaluate_params_with_backtest(candidate_params, klines_df, agg_trades_df, futures_df, sr_levels)
            
            if candidate_performance > best_performance:
                best_performance = candidate_performance
                optimized_params_found = candidate_params.copy() # Update the best found
                self.logger.info(f"  New best found! Performance: ${best_performance:,.2f}")

        self.logger.info(f"  Optimization complete. Final best performance from random search: ${best_performance:,.2f}")
        self.logger.info(f"  Final optimized parameters for this run: {optimized_params_found}")

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
        self._deep_update_dict(self.global_config['BEST_PARAMS'], optimized_params_found)
        self.logger.info("  CONFIG.BEST_PARAMS updated with optimized values.")

        optimization_run_id = str(uuid.uuid4())
        date_applied = datetime.datetime.now().isoformat()

        # Save to Firestore
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            params_doc = {
                "timestamp": date_applied,
                "optimization_run_id": optimization_run_id,
                "performance_metric": best_performance,
                "date_applied": date_applied,
                "params": optimized_params_found # Store the actual parameters
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
            with open(self.optimized_params_csv, 'a') as f:
                f.write(f"{date_applied},{optimization_run_id},{best_performance},{date_applied},{json.dumps(optimized_params_found)}\n")
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

            # Max Drawdown (simplified for daily, would need equity curve for full)
            # For a single day, if we only have daily net PnL, max drawdown is tricky.
            max_drawdown = 0.0 # Placeholder for now, requires intraday equity curve
            
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
                    
                    regime_trade_duration = 0.0 # Placeholder, requires entry/exit timestamps

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
        :param daily_trade_logs: List of all trades completed on the current day.
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
