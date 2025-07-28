# src/supervisor/supervisor.py
import pandas as pd
import numpy as np
import os
import datetime
import json # For handling nested dicts in logs if needed
import asyncio # For async Firestore operations

# Assume these are available in the same package or through sys.path
from config import CONFIG, INITIAL_EQUITY, BEST_PARAMS # Import BEST_PARAMS
from utils.logger import system_logger
from database.firestore_manager import FirestoreManager # New import

class Supervisor:
    """
    The Supervisor module (Meta-Learning Governor) optimizes the entire trading strategy
    and manages capital allocation over long time horizons. It also handles enhanced
    performance reporting.
    """
    def __init__(self, config=CONFIG, firestore_manager: FirestoreManager = None):
        self.config = config.get("supervisor", {})
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

    async def _implement_global_system_optimization(self):
        """
        Placeholder for Global System Optimization (Meta-Learning).
        This would involve:
        1. Defining a comprehensive search space for parameters across Analyst, Tactician, Strategist.
        2. Setting up an objective function (e.g., Sharpe Ratio, Calmar Ratio from backtesting).
        3. Running Bayesian Optimization or Genetic Algorithms on this search space.
        4. Updating the `config.py` or an internal parameter store with the new optimal parameters.
        5. Saving these new parameters to Firestore and CSV.
        """
        self.logger.info("\nSupervisor: Running Global System Optimization (Meta-Learning) - Placeholder.")
        # Simulate finding new best parameters
        simulated_new_best_params = {
            "atr.stop_loss_multiplier": round(np.random.uniform(1.0, 2.0), 2),
            "analyst.market_regime_classifier.adx_period": np.random.randint(10, 20),
            "tactician.laddering.initial_leverage": np.random.randint(20, 30),
            "analyst.regime_predictive_ensembles.ensemble_weights": {
                "lstm": round(np.random.uniform(0.2, 0.4), 2),
                "transformer": round(np.random.uniform(0.2, 0.4), 2),
                "statistical": round(np.random.uniform(0.1, 0.3), 2),
                "volume": round(np.random.uniform(0.1, 0.3), 2)
            }
        }
        # Ensure weights sum to 1 (simple normalization for demo)
        total_weight = sum(simulated_new_best_params["analyst.regime_predictive_ensembles.ensemble_weights"].values())
        if total_weight > 0:
            for k in simulated_new_best_params["analyst.regime_predictive_ensembles.ensemble_weights"]:
                simulated_new_best_params["analyst.regime_predictive_ensembles.ensemble_weights"][k] /= total_weight


        optimization_run_id = str(uuid.uuid4())
        performance_metric = round(np.random.uniform(0.5, 2.0), 2) # Simulated Sharpe Ratio
        date_applied = datetime.datetime.now().isoformat()

        # Update CONFIG.BEST_PARAMS (in a real system, this would be more robust)
        # For this demo, we'll just log it.
        self.logger.info(f"Simulated new best parameters found: {simulated_new_best_params}")
        self.logger.info(f"Simulated performance metric: {performance_metric}")

        # Save to Firestore
        if self.firestore_manager and self.firestore_manager.firestore_enabled:
            params_doc = {
                "timestamp": date_applied,
                "optimization_run_id": optimization_run_id,
                "performance_metric": performance_metric,
                "date_applied": date_applied,
                "params": simulated_new_best_params # Store the actual parameters
            }
            # Save to a document with a unique ID and also update a 'latest' document
            await self.firestore_manager.set_document(
                self.config['firestore']['optimized_params_collection'],
                doc_id=optimization_run_id,
                data=params_doc,
                is_public=True
            )
            await self.firestore_manager.set_document(
                self.config['firestore']['optimized_params_collection'],
                doc_id='latest', # Update a special 'latest' document
                data=params_doc,
                is_public=True
            )
            self.logger.info("Optimized parameters saved to Firestore.")

        # Export to CSV
        try:
            with open(self.optimized_params_csv, 'a') as f:
                f.write(f"{date_applied},{optimization_run_id},{performance_metric},{date_applied},{json.dumps(simulated_new_best_params)}\n")
            self.logger.info("Optimized parameters exported to CSV.")
        except Exception as e:
            self.logger.error(f"Error exporting optimized parameters to CSV: {e}")

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
                    self.config['firestore']['daily_summary_log_filename'].split('/')[-1].replace('.csv', ''), # Collection name
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
                        self.config['firestore']['strategy_performance_log_filename'].split('/')[-1].replace('.csv', ''), # Collection name
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
            await self._implement_global_system_optimization()
        
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

