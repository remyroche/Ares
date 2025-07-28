# backtesting/training_pipeline.py
import os
import sys
import datetime
import json
import asyncio
import pandas as pd # Import pandas for pd.DataFrame() for the optimizer call

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG, settings # Import settings to temporarily override trading_environment
from src.supervisor.main import Supervisor
from src.analyst.analyst import Analyst # Keep Analyst import for its specific use in this pipeline
from backtesting.ares_data_preparer import load_raw_data
from backtesting.ares_deep_analyzer import run_walk_forward_analysis, run_monte_carlo_simulation, plot_results
from src.utils.logger import system_logger
from src.exchange.binance import exchange # Import the global exchange instance for Supervisor init

class TrainingPipeline:
    """
    Orchestrates the entire offline training, optimization, and validation process.
    """
    def __init__(self):
        self.logger = system_logger.getChild('TrainingPipeline')
        # Analyst and Supervisor instances will be initialized in run() after setting environment
        self.analyst = None
        self.supervisor = None

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
        wf_futures = futures[wf_futures.index >= three_months_ago] # Corrected: Use wf_futures.index

        self.logger.info(f"Training set size: {len(train_klines)} candles.")
        self.logger.info(f"Walk-forward validation set size: {len(wf_klines)} candles.")
        
        return (train_klines, train_agg_trades, train_futures), (wf_klines, wf_agg_trades, wf_futures)

    async def run(self):
        """
        Executes the full training and validation pipeline.
        Automatically switches to TESTNET environment for the duration of the run.
        """
        self.logger.info("--- Starting Ares Training & Validation Pipeline ---")
        report_lines = ["Ares Training Pipeline Report"]
        separator = "="*80
        report_lines.append(separator)

        original_trading_environment = settings.trading_environment
        try:
            # Force TRADING_ENVIRONMENT to TESTNET for the duration of the training pipeline
            settings.trading_environment = "TESTNET"
            self.logger.info(f"Temporarily setting TRADING_ENVIRONMENT to '{settings.trading_environment}' for training.")

            # Initialize Supervisor *after* setting the environment, passing the exchange client
            # The Supervisor will now handle the instantiation of Analyst, Sentinel, etc., internally.
            self.supervisor = Supervisor(exchange_client=exchange) 
            self.analyst = self.supervisor.analyst # Get the Analyst instance from the Supervisor

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
            # The Analyst instance needs to be updated with the training data for its internal models
            # Ensure the Analyst has these properties or a method to set them.
            # Assuming Analyst has a method to load historical data for its internal components.
            # If Analyst's load_and_prepare_historical_data uses the exchange client, it will now use Testnet.
            await self.analyst.load_and_prepare_historical_data(
                historical_klines=train_klines,
                historical_agg_trades=train_agg_trades,
                historical_futures=train_futures
            )
            self.logger.info("All Analyst models have been retrained.")
            report_lines.append("STAGE 2: Analyst models retrained successfully.")

            # --- STAGE 3: Hyperparameter Optimization (Governor) ---
            self.logger.info("STAGE 3: Running hyperparameter optimization on the new models...")
            # Use the optimizer from the supervisor instance to run the global optimization
            # Note: historical_pnl_data and strategy_breakdown_data are passed as empty DataFrames/dicts
            # as they are typically generated by the live system, not during offline optimization.
            await self.supervisor.optimizer.implement_global_system_optimization(pd.DataFrame(), {})
            
            optimized_params = CONFIG['BEST_PARAMS']
            report_lines.append("STAGE 3: Hyperparameter optimization complete.")
            report_lines.append("Optimized Parameters:")
            report_lines.append(json.dumps(optimized_params, indent=2))

            # --- STAGE 4: Walk-Forward Validation ---
            self.logger.info("STAGE 4: Performing walk-forward validation on the last 3 months of data...")
            from backtesting.ares_data_preparer import calculate_and_label_regimes, get_sr_levels
            
            # *** NEW: Inject fee configuration into the parameters for the backtest ***
            params_with_fees = optimized_params.copy()
            params_with_fees['fees'] = CONFIG.get('fees', {'taker': 0.0004, 'maker': 0.0002}) # Add fees to params
            self.logger.info("Fee configuration included for backtesting.", extra=params_with_fees['fees'])

            wf_daily_df = wf_klines.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            wf_sr_levels = get_sr_levels(wf_daily_df)

            wf_prepared_df = calculate_and_label_regimes(
                wf_klines, wf_agg_trades, wf_futures, params_with_fees, wf_sr_levels
            )

            wf_report = run_walk_forward_analysis(wf_prepared_df, params_with_fees)
            report_lines.append("\n" + separator)
            report_lines.append("STAGE 4: Walk-Forward Validation Report")
            report_lines.append(wf_report)

            # --- STAGE 5: Monte Carlo Validation ---
            self.logger.info("STAGE 5: Performing Monte Carlo simulation on the walk-forward results...")
            mc_curves, base_portfolio, mc_report = run_monte_carlo_simulation(wf_prepared_df, params_with_fees)
            
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

        finally:
            # Restore the original trading environment
            settings.trading_environment = original_trading_environment
            self.logger.info(f"Restored TRADING_ENVIRONMENT to '{settings.trading_environment}'.")


def main():
    pipeline = TrainingPipeline()
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
