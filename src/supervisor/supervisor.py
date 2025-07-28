# src/supervisor/supervisor.py
import pandas as pd
import numpy as np
import os
import datetime
import json # For handling nested dicts in logs if needed

# Assume these are available in the same package or through sys.path
from config import CONFIG, INITIAL_EQUITY

class Supervisor:
    """
    The Supervisor module (Meta-Learning Governor) optimizes the entire trading strategy
    and manages capital allocation over long time horizons. It also handles enhanced
    performance reporting.
    """
    def __init__(self, config=CONFIG):
        self.config = config.get("supervisor", {})
        self.initial_equity = INITIAL_EQUITY # Total capital available to the system
        
        # Current capital allocation multiplier, adjusted dynamically
        self.allocated_capital_multiplier = self.config.get("initial_allocated_capital_multiplier", 1.0)

        self.daily_summary_log_filename = self.config.get("daily_summary_log_filename", "reports/daily_summary_log.csv")
        self.strategy_performance_log_filename = self.config.get("strategy_performance_log_filename", "reports/strategy_performance_log.csv")

        # Ensure reports directory exists
        os.makedirs(os.path.dirname(self.daily_summary_log_filename), exist_ok=True)
        
        # Initialize CSV headers if files don't exist
        self._initialize_daily_summary_csv()
        self._initialize_strategy_performance_csv()

    def _initialize_daily_summary_csv(self):
        """Ensures the daily summary CSV file exists with correct headers."""
        if not os.path.exists(self.daily_summary_log_filename):
            with open(self.daily_summary_log_filename, 'w') as f:
                f.write("Date,TotalTrades,WinRate,NetPnL,MaxDrawdown,EndingCapital,AllocatedCapitalMultiplier\n")
            print(f"Created daily summary log: {self.daily_summary_log_filename}")

    def _initialize_strategy_performance_csv(self):
        """Ensures the strategy performance CSV file exists with correct headers."""
        if not os.path.exists(self.strategy_performance_log_filename):
            with open(self.strategy_performance_log_filename, 'w') as f:
                f.write("Date,Regime,TotalTrades,WinRate,NetPnL,AvgPnLPerTrade,TradeDuration\n")
            print(f"Created strategy performance log: {self.strategy_performance_log_filename}")

    def _implement_global_system_optimization(self):
        """
        Placeholder for Global System Optimization (Meta-Learning).
        This would involve:
        1. Defining a comprehensive search space for parameters across Analyst, Tactician, Strategist.
        2. Setting up an objective function (e.g., Sharpe Ratio, Calmar Ratio from backtesting).
        3. Running Bayesian Optimization or Genetic Algorithms on this search space.
        4. Updating the `config.py` or an internal parameter store with the new optimal parameters.
        """
        print("\nSupervisor: Running Global System Optimization (Meta-Learning) - Placeholder.")
        print("This process would involve extensive backtesting simulations and advanced optimization algorithms.")
        # Example:
        # new_optimal_params = run_bayesian_optimization(self.current_config, historical_data)
        # self.update_system_parameters(new_optimal_params)
        pass

    def _calculate_dynamic_capital_allocation(self, historical_pnl_data: pd.DataFrame):
        """
        Adjusts the capital allocation multiplier based on recent performance.
        :param historical_pnl_data: DataFrame with daily P&L (at least 'Date' and 'NetPnL' columns).
        """
        print("\nSupervisor: Calculating Dynamic Capital Allocation...")
        lookback_days = self.config.get("risk_allocation_lookback_days", 30)
        max_increase_pct = self.config.get("max_capital_allocation_increase_pct", 1.0)
        max_decrease_pct = self.config.get("max_capital_allocation_decrease_pct", 0.75)

        if historical_pnl_data.empty or len(historical_pnl_data) < lookback_days:
            print(f"Insufficient historical P&L data ({len(historical_pnl_data)} days) for dynamic allocation. Need at least {lookback_days} days.")
            return # Keep current allocation

        recent_pnl = historical_pnl_data['NetPnL'].tail(lookback_days)
        total_pnl_over_lookback = recent_pnl.sum()
        
        # Simple performance metric: average daily P&L percentage
        # Normalize P&L by initial equity to get a comparable performance metric
        avg_daily_pnl_pct = total_pnl_over_lookback / (lookback_days * self.initial_equity * self.allocated_capital_multiplier)

        # Adjust multiplier based on performance
        # Positive performance increases allocation, negative decreases
        # Scale the adjustment based on a sensitivity factor (e.g., 0.1 for modest changes)
        adjustment_factor = 0.1 # This can be a config parameter too
        
        # Calculate proposed change
        change_in_multiplier = avg_daily_pnl_pct * adjustment_factor * lookback_days # Scale by lookback days

        new_multiplier = self.allocated_capital_multiplier + change_in_multiplier

        # Apply bounds: -75% to +100% relative to the *initial* allocated capital multiplier (1.0)
        min_allowed_multiplier = 1.0 - max_decrease_pct
        max_allowed_multiplier = 1.0 + max_increase_pct

        self.allocated_capital_multiplier = np.clip(new_multiplier, min_allowed_multiplier, max_allowed_multiplier)
        
        print(f"Dynamic Capital Allocation: Total P&L over {lookback_days} days: ${total_pnl_over_lookback:,.2f}")
        print(f"New Allocated Capital Multiplier: {self.allocated_capital_multiplier:.2f}x (Effective Capital: ${self.get_current_allocated_capital():,.2f})")

    def get_current_allocated_capital(self):
        """Returns the current dynamically allocated capital."""
        return self.initial_equity * self.allocated_capital_multiplier

    def generate_performance_report(self, trade_logs: list, current_date: datetime.date):
        """
        Generates a detailed performance report for the given day.
        :param trade_logs: List of dictionaries, each representing a completed trade.
        :param current_date: The date for which the report is being generated.
        :return: Dictionary containing daily summary and strategy breakdown.
        """
        print(f"\nSupervisor: Generating Performance Report for {current_date}...")
        
        if not trade_logs:
            print("No trades recorded for this period. Generating empty report.")
            return {
                "daily_summary": {
                    "Date": current_date.strftime('%Y-%m-%d'),
                    "TotalTrades": 0, "WinRate": 0.0, "NetPnL": 0.0,
                    "MaxDrawdown": 0.0, "EndingCapital": self.get_current_allocated_capital(),
                    "AllocatedCapitalMultiplier": self.allocated_capital_multiplier
                },
                "strategy_breakdown": {}
            }

        df_trades = pd.DataFrame(trade_logs)
        
        # Ensure PnL is numeric
        if 'Realized P&L ($)' in df_trades.columns:
            df_trades['Realized P&L ($)'] = pd.to_numeric(df_trades['Realized P&L ($)'], errors='coerce').fillna(0)
        else:
            df_trades['Realized P&L ($)'] = 0.0 # Default if column missing

        # Daily Summary Metrics
        total_trades = len(df_trades)
        wins = df_trades[df_trades['Realized P&L ($)'] > 0]
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        net_pnl = df_trades['Realized P&L ($)'].sum()

        # Calculate Max Drawdown (simplified for daily, would need equity curve for full)
        # For a single day, if we only have daily net PnL, max drawdown is tricky.
        # Assuming we track equity over the day, or just use daily PnL relative to start.
        # For this demo, we'll use a placeholder or a simple interpretation.
        max_drawdown = 0.0 # Placeholder for now, requires intraday equity curve
        
        # Ending capital for the day (assuming it's tracked by the main orchestrator)
        # For this demo, we'll just use the current allocated capital
        ending_capital = self.get_current_allocated_capital() + net_pnl # Very simplified

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
        if 'Market State at Entry' in df_trades.columns:
            for regime in df_trades['Market State at Entry'].unique():
                regime_trades = df_trades[df_trades['Market State at Entry'] == regime]
                regime_total_trades = len(regime_trades)
                regime_wins = regime_trades[regime_trades['Realized P&L ($)'] > 0]
                regime_win_rate = (len(regime_wins) / regime_total_trades * 100) if regime_total_trades > 0 else 0.0
                regime_net_pnl = regime_trades['Realized P&L ($)'].sum()
                regime_avg_pnl_per_trade = regime_net_pnl / regime_total_trades if regime_total_trades > 0 else 0.0
                
                # Trade duration requires entry/exit timestamps
                # For this demo, we'll use a placeholder
                regime_trade_duration = 0.0 # Placeholder

                strategy_breakdown[regime] = {
                    "TotalTrades": regime_total_trades,
                    "WinRate": round(regime_win_rate, 2),
                    "NetPnL": round(regime_net_pnl, 2),
                    "AvgPnLPerTrade": round(regime_avg_pnl_per_trade, 2),
                    "TradeDuration": round(regime_trade_duration, 2)
                }
        
        print("Performance Report Generated.")
        return {"daily_summary": daily_summary, "strategy_breakdown": strategy_breakdown}

    def _update_daily_summary_csv(self, daily_summary: dict):
        """Appends the daily summary to the CSV log."""
        try:
            # Convert dictionary to a list of values in the correct order for CSV
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
            print(f"Appended daily summary for {daily_summary['Date']} to CSV.")
        except Exception as e:
            print(f"Error updating daily summary CSV: {e}")

    def _update_strategy_performance_log(self, current_date: datetime.date, strategy_breakdown: dict):
        """Appends strategy performance breakdown to its CSV log."""
        try:
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
            print(f"Appended strategy performance for {current_date} to CSV.")
        except Exception as e:
            print(f"Error updating strategy performance log CSV: {e}")

    def orchestrate_supervision(self, current_date: datetime.date, total_equity: float, 
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
        print(f"\n--- Supervisor: Starting Orchestration for {current_date} ---")

        # 1. Dynamic Risk Allocation
        # Update historical_daily_pnl_data with today's P&L before passing
        today_net_pnl = sum(t.get('Realized P&L ($)', 0) for t in daily_trade_logs)
        new_daily_pnl_row = pd.DataFrame([{
            'Date': current_date,
            'NetPnL': today_net_pnl
        }])
        # Ensure 'Date' column is datetime type in historical_daily_pnl_data
        if 'Date' in historical_daily_pnl_data.columns:
            historical_daily_pnl_data['Date'] = pd.to_datetime(historical_daily_pnl_data['Date'])
        
        updated_historical_pnl = pd.concat([historical_daily_pnl_data, new_daily_pnl_row]).drop_duplicates(subset=['Date']).sort_values('Date')
        
        self._calculate_dynamic_capital_allocation(updated_historical_pnl)

        # 2. Performance Reporting
        report = self.generate_performance_report(daily_trade_logs, current_date)
        self._update_daily_summary_csv(report["daily_summary"])
        self._update_strategy_performance_log(current_date, report["strategy_breakdown"])

        # 3. Global System Optimization (Meta-Learning) - Run periodically
        # This would typically be triggered less frequently, e.g., weekly or monthly
        if current_date.day % self.config.get("meta_learning_frequency_days", 7) == 0:
            self._implement_global_system_optimization()
        
        print(f"--- Supervisor: Orchestration Complete for {current_date} ---")
        return {
            "allocated_capital": self.get_current_allocated_capital(),
            "daily_summary": report["daily_summary"],
            "strategy_breakdown": report["strategy_breakdown"]
        }

# --- Example Usage (Main execution block for demonstration) ---
if __name__ == "__main__":
    print("Running Supervisor Module Demonstration...")

    # Create a dummy reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    supervisor = Supervisor()

    # Simulate historical daily P&L for dynamic allocation
    # In a real system, this would be loaded from daily_summary_log.csv
    historical_pnl = pd.DataFrame({
        'Date': pd.to_datetime(pd.date_range(start='2024-06-01', periods=40, freq='D')),
        'NetPnL': np.random.randn(40) * 500 # Simulate some daily P&L
    })
    # Make some days profitable, some losing
    historical_pnl.loc[historical_pnl.index[:20], 'NetPnL'] = np.random.rand(20) * 1000 - 200 # Mixed
    historical_pnl.loc[historical_pnl.index[20:], 'NetPnL'] = np.random.rand(20) * 1500 + 100 # More profitable recently

    # Simulate daily run over a few days
    start_date = datetime.date(2024, 7, 25)
    num_days_to_simulate = 5

    current_total_equity = INITIAL_EQUITY # Start with initial equity

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
                "Trade ID": f"T{sim_date.strftime('%Y%m%d')}-{_}",
                "Entry/Exit Timestamps": sim_date.isoformat(),
                "Asset": "ETHUSDT",
                "Direction": np.random.choice(["LONG", "SHORT"]),
                "Market State at Entry": regime,
                "Entry Price": 2000 + np.random.rand() * 100,
                "Exit Price": 2000 + np.random.rand() * 100,
                "Position Size": np.random.rand() * 0.5 + 0.1,
                "Leverage Used": np.random.randint(25, 75),
                "Confidence Score & LSS at Entry": {"conf": np.random.rand(), "lss": np.random.rand() * 100},
                "Fees Paid": abs(pnl) * 0.001,
                "Funding Rate Paid/Received": np.random.rand() * 0.0001 - 0.00005,
                "Realized P&L ($)": pnl,
                "Exit Reason": np.random.choice(["Take Profit", "Stop Loss", "Manual Close"])
            }
            daily_trades.append(trade_log)

            # Aggregate P&L per regime for the day
            if regime not in daily_pnl_by_regime:
                daily_pnl_by_regime[regime] = 0.0
            daily_pnl_by_regime[regime] += pnl

        # Update total equity (very simplified for demo)
        current_total_equity += sum(t.get('Realized P&L ($)', 0) for t in daily_trades)

        # Orchestrate supervision for the day
        supervisor_output = supervisor.orchestrate_supervision(
            current_date=sim_date,
            total_equity=current_total_equity,
            daily_trade_logs=daily_trades,
            daily_pnl_per_regime=daily_pnl_by_regime,
            historical_daily_pnl_data=historical_pnl.copy() # Pass a copy to avoid modifying original
        )
        print(f"Day {sim_date} Summary: Allocated Capital: ${supervisor_output['allocated_capital']:,.2f}, Net P&L: ${supervisor_output['daily_summary']['NetPnL']:,.2f}")
        
        # Add today's P&L to historical_pnl for next iteration
        historical_pnl = pd.concat([historical_pnl, pd.DataFrame([{'Date': sim_date, 'NetPnL': supervisor_output['daily_summary']['NetPnL']}])]).drop_duplicates(subset=['Date']).sort_values('Date')


    print("\nSupervisor Module Demonstration Complete.")
    print(f"Check '{supervisor.daily_summary_log_filename}' and '{supervisor.strategy_performance_log_filename}' for logs.")

