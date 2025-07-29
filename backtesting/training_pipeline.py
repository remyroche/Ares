import os
import sys
import datetime
import json
import asyncio
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import CONFIG, settings
from src.supervisor.main import Supervisor
from src.analyst.analyst import Analyst
from backtesting.ares_data_preparer import load_raw_data, calculate_and_label_regimes, get_sr_levels
from backtesting.ares_deep_analyzer import run_walk_forward_analysis, run_monte_carlo_simulation, plot_results
from src.utils.logger import system_logger
from src.exchange.binance import exchange

class TrainingPipeline:
    """
    Orchestrates the entire offline training, optimization, and validation process.
    """
    def __init__(self):
        self.logger = system_logger.getChild('TrainingPipeline')
        self.analyst = None
        self.supervisor = None

    async def run(self):
        """
        ## CHANGE: Re-architected the pipeline to use a robust walk-forward validation process.
        ## Instead of a single train/test split, this method now uses TimeSeriesSplit
        ## to create multiple chronological folds. In each fold, the models are retrained
        ## and re-optimized on past data and then validated on unseen future data,
        ## providing a much more accurate measure of performance.
        """
        self.logger.info("--- Starting Ares Training & Validation Pipeline ---")
        report_lines = ["Ares Training Pipeline Report"]
        separator = "="*80
        report_lines.append(separator)

        original_trading_environment = settings.trading_environment
        try:
            settings.trading_environment = "TESTNET"
            self.logger.info(f"Temporarily setting TRADING_ENVIRONMENT to '{settings.trading_environment}' for training.")

            self.supervisor = Supervisor(exchange_client=exchange)
            self.analyst = self.supervisor.analyst

            self.logger.info("STAGE 1: Loading all available historical data...")
            all_klines, all_agg_trades, all_futures = load_raw_data()
            if all_klines.empty:
                self.logger.error("Failed to load data. Halting pipeline.")
                return

            self.logger.info("STAGE 2: Preparing full dataset for walk-forward validation...")
            # Prepare the entire dataset once to ensure consistent feature calculation
            daily_df = all_klines.resample('D').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            sr_levels = get_sr_levels(daily_df)
            full_prepared_data = calculate_and_label_regimes(all_klines, all_agg_trades, all_futures, CONFIG['best_params'], sr_levels)

            # --- Walk-Forward Validation Loop ---
            n_splits = CONFIG.get("training_pipeline", {}).get("walk_forward", {}).get("n_splits", 5)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            all_fold_reports = []

            self.logger.info(f"STAGE 3: Starting {n_splits}-Fold Walk-Forward Validation...")
            for fold, (train_index, test_index) in enumerate(tscv.split(full_prepared_data)):
                self.logger.info(f"\n--- FOLD {fold + 1}/{n_splits} ---")
                
                train_data = full_prepared_data.iloc[train_index]
                test_data = full_prepared_data.iloc[test_index]

                # Filter raw data based on the training index for retraining
                train_klines = all_klines.loc[train_data.index]
                train_agg_trades = all_agg_trades.loc[train_data.index]
                train_futures = all_futures.loc[train_data.index]

                self.logger.info(f"Training window: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} rows)")
                self.logger.info(f"Validation window: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} rows)")

                # STAGE 3a: Retrain models on the current training slice
                await self.analyst.load_and_prepare_historical_data(
                    historical_klines=train_klines,
                    historical_agg_trades=train_agg_trades,
                    historical_futures=train_futures
                )
                self.logger.info("Analyst models retrained for this fold.")

                # STAGE 3b: Run hyperparameter optimization on the retrained models
                await self.supervisor.optimizer.implement_global_system_optimization(pd.DataFrame(), {})
                optimized_params = CONFIG['best_params']
                self.logger.info("Hyperparameters re-optimized for this fold.")

                # STAGE 3c: Perform walk-forward validation on the unseen test data
                params_with_fees = optimized_params.copy()
                params_with_fees['fees'] = CONFIG.get('fees', {'taker': 0.0004, 'maker': 0.0002})
                
                fold_report = run_walk_forward_analysis(test_data, params_with_fees)
                all_fold_reports.append(fold_report)
                self.logger.info(f"Fold {fold + 1} validation complete.")
                report_lines.append(f"\n--- FOLD {fold + 1} REPORT ---\n{fold_report}")

            # --- Final Stages ---
            self.logger.info("STAGE 4: Aggregating walk-forward results...")
            # Here you would add logic to average the metrics from all_fold_reports
            # For now, we just append all reports.
            report_lines.append("\n" + separator)
            report_lines.append("STAGE 4: Walk-Forward Validation Complete.")

            self.logger.info("STAGE 5: Training final model on ALL historical data...")
            await self.analyst.load_and_prepare_historical_data(all_klines, all_agg_trades, all_futures)
            await self.supervisor.optimizer.implement_global_system_optimization(pd.DataFrame(), {})
            final_optimized_params = CONFIG['best_params']
            self.logger.info("Final production model trained and optimized.")
            report_lines.append("STAGE 5: Final production model trained successfully.")
            report_lines.append("Final Optimized Parameters:")
            report_lines.append(json.dumps(final_optimized_params, indent=2))

            self.logger.info("STAGE 6: Generating final report and saving challenger model...")
            final_report = "\n".join(report_lines)
            print(final_report)

            report_filename = f"reports/training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            os.makedirs("reports", exist_ok=True)
            with open(report_filename, 'w') as f:
                f.write(final_report)
            self.logger.info(f"Full training report saved to {report_filename}")

            challenger_model_dir = "models/challenger"
            os.makedirs(challenger_model_dir, exist_ok=True)
            with open(os.path.join(challenger_model_dir, "optimized_params.json"), 'w') as f:
                json.dump(final_optimized_params, f, indent=2)
            self.logger.info(f"Challenger model parameters saved to {challenger_model_dir}")

            self.logger.info("--- Ares Training & Validation Pipeline Finished ---")

        finally:
            settings.trading_environment = original_trading_environment
            self.logger.info(f"Restored TRADING_ENVIRONMENT to '{settings.trading_environment}'.")

def main():
    pipeline = TrainingPipeline()
    asyncio.run(pipeline.run())

if __name__ == "__main__":
    main()
