# backtesting/training_pipeline.py
import pandas as pd
import os
import sys
import datetime
import json
import asyncio

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG
from src.supervisor.main import Supervisor
from src.analyst.analyst import Analyst
from backtesting.ares_data_preparer import load_raw_data
from backtesting.ares_deep_analyzer import run_walk_forward_analysis, run_monte_carlo_simulation, plot_results
from src.utils.logger import system_logger

class TrainingPipeline:
    """
    Orchestrates the entire offline training, optimization, and validation process.
    """
    def __init__(self):
        self.logger = system_logger.getChild('TrainingPipeline')
        # We need both Analyst and Supervisor instances for their respective roles
        self.analyst = Analyst(CONFIG)
        self.supervisor = Supervisor(CONFIG)

    def get_training_and_walkforward_data(self, klines, agg_trades, futures):
        """
        Splits the data into a main training set and a final walk-forward validation set.
        - Training/Optimization Set: All data except the last 3 months.
        - Walk-Forward Set: The last 3 months of data for final validation.
        """
        self.logger.info("Splitting data into training and walk-forward sets...")
        three_months_ago = klines.index.max() - pd.DateOffset(months=3)
        
        train_klines = klines[klines.index < three_months_ago]
        wf_klines = klines[klines.index >= three_months_ago]

        train_agg_trades = agg_trades[agg_trades.index < three_months_ago]
        wf_agg_trades = agg_trades[agg_trades.index >= three_months_ago]

        train_futures = futures[futures.index < three_months_ago]
        wf_futures = futures[futures.index >= three_months_ago]

        self.logger.info(f"Training set size: {len(train_klines)} candles.")
        self.logger.info(f"Walk-forward validation set size: {len(wf_klines)} candles.")
        
        return (train_klines, train_agg_trades, train_futures), (wf_klines, wf_agg_trades, wf_futures)

    async def run(self):
        """
        Executes the full training and validation pipeline.
        """
        self.logger.info("--- Starting Ares Training & Validation Pipeline ---")
        report_lines = ["Ares Training Pipeline Report"]
        separator = "="*80
        report_lines.append(separator)

        # --- STAGE 1: Data Loading & Preparation ---
        self.logger.info("STAGE 1: Loading all available historical data...")
        all_klines, all_agg_trades, all_futures = load_raw_data()
        if all_klines.empty:
            self.logger.error("Failed to load data. Halting pipeline.")
            return

        (train_klines, train_agg_trades, train_futures), \
        (wf_klines, wf_agg_trades, wf_futures) = self.get_training_and_walkforward_data(
            all_klines, all_agg_trades, all_futures
        )

        # --- STAGE 2: Retrain Analyst Models ---
        self.logger.info("STAGE 2: Retraining all Analyst models on the new training dataset...")
        self.analyst._historical_klines = train_klines
        self.analyst._historical_agg_trades = train_agg_trades
        self.analyst._historical_futures = train_futures
        await self.analyst.load_and_prepare_historical_data()
        self.logger.info("All Analyst models have been retrained.")
        report_lines.append("STAGE 2: Analyst models retrained successfully.")

        # --- STAGE 3: Hyperparameter Optimization (Governor) ---
        self.logger.info("STAGE 3: Running hyperparameter optimization on the new models...")
        # Use the optimizer from the supervisor instance to run the global optimization
        await self.supervisor.optimizer.implement_global_system_optimization(pd.DataFrame(), {})
        
        optimized_params = CONFIG['BEST_PARAMS']
        report_lines.append("STAGE 3: Hyperparameter optimization complete.")
        report_lines.append("Optimized Parameters:")
        report_lines.append(json.dumps(optimized_params, indent=2))

        # --- STAGE 4: Walk-Forward Validation ---
        self.logger.info("STAGE 4: Performing walk-forward validation on the last 3 months of data...")
        from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
        
        wf_daily_df = wf_klines.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
        wf_sr_levels = get_sr_levels(wf_daily_df)

        wf_prepared_df = calculate_and_label_regimes(
            wf_klines, wf_agg_trades, wf_futures, optimized_params, wf_sr_levels
        )

        wf_report = run_walk_forward_analysis(wf_prepared_df, optimized_params)
        report_lines.append("\n" + separator)
        report_lines.append("STAGE 4: Walk-Forward Validation Report")
        report_lines.append(wf_report)

        # --- STAGE 5: Monte Carlo Validation ---
        self.logger.info("STAGE 5: Performing Monte Carlo simulation on the walk-forward results...")
        mc_curves, base_portfolio, mc_report = run_monte_carlo_simulation(wf_prepared_df, optimized_params)
        
        report_lines.append("\n" + separator)
        report_lines.append("STAGE 5: Monte Carlo Validation Report")
        report_lines.append(mc_report)

        # --- STAGE 6: Final Reporting and Model Saving ---
        self.logger.info("STAGE 6: Generating final report and saving challenger model...")
        final_report = "\n".join(report_lines)
        print(final_report)

        report_filename = f"reports/training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_filename, 'w') as f:
            f.write(final_report)
        self.logger.info(f"Full training report saved to {report_filename}")

        challenger_model_dir = "models/challenger"
        os.makedirs(challenger_model_dir, exist_ok=True)
        with open(os.path.join(challenger_model_dir, "optimized_params.json"), 'w') as f:
            json.dump(optimized_params, f, indent=2)
        self.logger.info(f"Challenger model parameters saved to {challenger_model_dir}")
        
        plot_results(mc_curves, base_portfolio)

        self.logger.info("--- Ares Training & Validation Pipeline Finished ---")


def main():
    pipeline = TrainingPipeline()
    asyncio.run(pipeline.run())

    analyst_output = analyst.run_analysis(current_data)
    strategist = Strategist(long_threshold=0.7, short_threshold=0.7) # Can be optimized
    strategic_bias = strategist.decide_strategy(analyst_output)
    backtester.handle_signal(strategic_bias)

if __name__ == "__main__":
    main()
