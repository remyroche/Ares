import os
import sys
import datetime
import json
import asyncio
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Ensure the source directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import CONFIG, settings
from src.supervisor.supervisor import Supervisor
from backtesting.ares_data_preparer import (
    load_raw_data,
    calculate_and_label_regimes,
    get_sr_levels,
)
from backtesting.ares_deep_analyzer import (
    run_walk_forward_analysis,
)
from src.utils.logger import system_logger
from exchange.binance import BinanceExchange


class TrainingPipeline:
    """
    Orchestrates the entire offline training, optimization, and validation process.
    Now includes robust checkpointing to resume from previous failures.
    """

    def __init__(self):
        self.logger = system_logger.getChild("TrainingPipeline")
        self.analyst = None
        self.supervisor = None

        # Checkpointing paths
        self.checkpoint_dir = CONFIG["CHECKPOINT_DIR"]
        self.pipeline_progress_file = os.path.join(
            self.checkpoint_dir, CONFIG["PIPELINE_PROGRESS_FILE"]
        )
        self.prepared_data_checkpoint_file = os.path.join(
            self.checkpoint_dir, CONFIG["PREPARED_DATA_CHECKPOINT_FILE"]
        )
        self.walk_forward_reports_file = os.path.join(
            self.checkpoint_dir, CONFIG["WALK_FORWARD_REPORTS_FILE"]
        )
        self.optimizer_checkpoint_file = os.path.join(
            self.checkpoint_dir, CONFIG["OPTIMIZER_CHECKPOINT_FILE"]
        )

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    async def run(self):
        """
        Runs the training pipeline with checkpointing.
        """
        self.logger.info("--- Starting Ares Training & Validation Pipeline ---")
        report_lines = ["Ares Training Pipeline Report"]
        separator = "=" * 80
        report_lines.append(separator)

        original_trading_environment = settings.trading_environment
        try:
            settings.trading_environment = "TESTNET"
            self.logger.info(
                f"Temporarily setting TRADING_ENVIRONMENT to '{settings.trading_environment}' for training."
            )

            exchange_client = BinanceExchange(
                api_key=settings.binance_api_key,
                api_secret=settings.binance_api_secret,
                trade_symbol=CONFIG.get("trading_symbol", "ETHUSDT"),
            )
            self.supervisor = Supervisor(exchange_client=exchange_client)
            # The Analyst instance within the Supervisor is the one we'll use for training
            self.analyst = self.supervisor.analyst

            # --- Load Pipeline Progress ---
            pipeline_progress = self._load_pipeline_progress()
            last_completed_fold = pipeline_progress.get("last_completed_fold", -1)
            all_fold_reports = pipeline_progress.get("all_fold_reports", [])
            self.logger.info(
                f"Resuming from last completed fold: {last_completed_fold}"
            )

            # --- STAGE 1 & 2: Data Loading and Preparation (with checkpoint) ---
            self.logger.info(
                "STAGE 1 & 2: Loading and preparing historical data (with checkpoint)..."
            )
            full_prepared_data = pd.DataFrame()
            if os.path.exists(self.prepared_data_checkpoint_file):
                self.logger.info(
                    f"Loading prepared data from checkpoint: {self.prepared_data_checkpoint_file}"
                )
                full_prepared_data = pd.read_parquet(self.prepared_data_checkpoint_file)

            if full_prepared_data.empty:
                self.logger.info(
                    "Prepared data checkpoint not found or empty. Loading raw data and preparing from scratch."
                )
                all_klines, all_agg_trades, all_futures = load_raw_data()
                if all_klines.empty:
                    self.logger.error("Failed to load raw data. Halting pipeline.")
                    return

                daily_df = all_klines.resample("D").agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                sr_levels = get_sr_levels(daily_df)

                # Pass the CONFIG['best_params'] to ensure consistent feature calculation
                full_prepared_data = calculate_and_label_regimes(
                    all_klines,
                    all_agg_trades,
                    all_futures,
                    CONFIG["best_params"],
                    sr_levels,
                )

                if full_prepared_data.empty:
                    self.logger.error(
                        "Failed to prepare full dataset. Halting pipeline."
                    )
                    return

                self.logger.info(
                    f"Saving prepared data to checkpoint: {self.prepared_data_checkpoint_file}"
                )
                full_prepared_data.to_parquet(self.prepared_data_checkpoint_file)
            else:
                self.logger.info(
                    f"Loaded {len(full_prepared_data)} rows of prepared data from checkpoint."
                )

            # --- Walk-Forward Validation Loop ---
            n_splits = (
                CONFIG.get("training_pipeline", {})
                .get("walk_forward", {})
                .get("n_splits", 5)
            )
            tscv = TimeSeriesSplit(n_splits=n_splits)

            self.logger.info(
                f"STAGE 3: Starting {n_splits}-Fold Walk-Forward Validation..."
            )

            # Get the actual train/test indices from TimeSeriesSplit once
            all_splits = list(tscv.split(full_prepared_data))

            for fold, (train_index, test_index) in enumerate(all_splits):
                if fold <= last_completed_fold:
                    self.logger.info(
                        f"Skipping fold {fold + 1} as it was already completed."
                    )
                    continue

                self.logger.info(f"\n--- FOLD {fold + 1}/{n_splits} ---")

                train_data = full_prepared_data.iloc[train_index]
                test_data = full_prepared_data.iloc[test_index]

                # Filter raw data based on the training index for retraining
                # We need to re-load raw data for each fold to ensure correct time alignment
                # This is a simplification; in a very large dataset, you'd pre-filter raw data too.
                # For now, assume `load_raw_data` provides the full range, and we filter.
                all_klines, all_agg_trades, all_futures = (
                    load_raw_data()
                )  # Re-load full raw data

                train_klines = all_klines.loc[
                    train_data.index.min() : train_data.index.max()
                ]
                train_agg_trades = all_agg_trades.loc[
                    train_data.index.min() : train_data.index.max()
                ]
                train_futures = all_futures.loc[
                    train_data.index.min() : train_data.index.max()
                ]

                self.logger.info(
                    f"Training window: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} rows)"
                )
                self.logger.info(
                    f"Validation window: {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} rows)"
                )

                # STAGE 3a: Retrain models on the current training slice (with fold-specific checkpointing)
                self.logger.info("Analyst models retraining/loading for this fold.")
                await self.analyst.load_and_prepare_historical_data(
                    historical_klines=train_klines,
                    historical_agg_trades=train_agg_trades,
                    historical_futures=train_futures,
                    fold_id=fold,  # Pass fold_id for model saving/loading
                )
                self.logger.info("Analyst models ready for this fold.")

                # STAGE 3b: Run hyperparameter optimization on the retrained models (with checkpoint)
                self.logger.info(
                    "Hyperparameters re-optimizing/resuming for this fold."
                )
                await self.supervisor.optimizer.implement_global_system_optimization(
                    historical_pnl_data=pd.DataFrame(),  # Not used in current optimizer logic
                    strategy_breakdown_data={},  # Not used in current optimizer logic
                    checkpoint_file_path=self.optimizer_checkpoint_file,  # Pass checkpoint path
                )
                # After optimization, CONFIG['best_params'] is updated by the optimizer
                optimized_params = CONFIG["best_params"]
                self.logger.info("Hyperparameters re-optimized for this fold.")

                # STAGE 3c: Perform walk-forward validation on the unseen test data
                params_with_fees = optimized_params.copy()
                params_with_fees["fees"] = CONFIG.get(
                    "fees", {"taker": 0.0004, "maker": 0.0002}
                )

                fold_report = run_walk_forward_analysis(test_data, params_with_fees)
                all_fold_reports.append(fold_report)
                self.logger.info(f"Fold {fold + 1} validation complete.")
                report_lines.append(f"\n--- FOLD {fold + 1} REPORT ---\n{fold_report}")

                # Save pipeline progress after each successful fold
                self._save_pipeline_progress(fold, all_fold_reports)

            # --- Final Stages ---
            self.logger.info("STAGE 4: Aggregating walk-forward results...")
            report_lines.append("\n" + separator)
            report_lines.append("STAGE 4: Walk-Forward Validation Complete.")

            self.logger.info("STAGE 5: Training final model on ALL historical data...")
            # Re-load full raw data for final training
            all_klines, all_agg_trades, all_futures = load_raw_data()
            await self.analyst.load_and_prepare_historical_data(
                historical_klines=all_klines,
                historical_agg_trades=all_agg_trades,
                historical_futures=all_futures,
                fold_id="final",  # Use a distinct ID for the final model
            )
            self.logger.info("Final Analyst models retrained/loaded.")

            # Run final optimization on all data
            await self.supervisor.optimizer.implement_global_system_optimization(
                historical_pnl_data=pd.DataFrame(),
                strategy_breakdown_data={},
                checkpoint_file_path=self.optimizer_checkpoint_file,  # Use the same checkpoint file for final opt
            )
            final_optimized_params = CONFIG["best_params"]
            self.logger.info("Final production model trained and optimized.")
            report_lines.append("STAGE 5: Final production model trained successfully.")
            report_lines.append("Final Optimized Parameters:")
            report_lines.append(json.dumps(final_optimized_params, indent=2))

            self.logger.info(
                "STAGE 6: Generating final report and saving challenger model..."
            )
            final_report = "\n".join(report_lines)
            print(final_report)

            report_filename = f"reports/training_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            os.makedirs("reports", exist_ok=True)
            with open(report_filename, "w") as f:
                f.write(final_report)
            self.logger.info(f"Full training report saved to {report_filename}")

            challenger_model_dir = "models/challenger"
            os.makedirs(challenger_model_dir, exist_ok=True)
            with open(
                os.path.join(challenger_model_dir, "optimized_params.json"), "w"
            ) as f:
                json.dump(final_optimized_params, f, indent=2)
            self.logger.info(
                f"Challenger model parameters saved to {challenger_model_dir}"
            )

            self.logger.info("--- Ares Training & Validation Pipeline Finished ---")

        except Exception as e:
            self.logger.error(
                f"An error occurred during the pipeline: {e}", exc_info=True
            )
            raise  # Re-raise the exception to be caught by the orchestrator if needed

        finally:
            settings.trading_environment = original_trading_environment
            self.logger.info(
                f"Restored TRADING_ENVIRONMENT to '{settings.trading_environment}'."
            )

    def _load_pipeline_progress(self):
        """Loads the last saved pipeline progress."""
        if os.path.exists(self.pipeline_progress_file):
            try:
                with open(self.pipeline_progress_file, "r") as f:
                    progress = json.load(f)
                    self.logger.info(f"Loaded pipeline progress: {progress}")
                    return progress
            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"Error reading pipeline progress file: {e}. Starting fresh."
                )
        return {"last_completed_fold": -1, "all_fold_reports": []}

    def _save_pipeline_progress(self, last_completed_fold: int, all_fold_reports: list):
        """Saves the current pipeline progress."""
        progress = {
            "last_completed_fold": last_completed_fold,
            "all_fold_reports": all_fold_reports,  # Save reports for resume
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(self.pipeline_progress_file, "w") as f:
                json.dump(progress, f, indent=2)
            self.logger.info(
                f"Pipeline progress saved: Completed fold {last_completed_fold}"
            )
        except Exception as e:
            self.logger.error(f"Error saving pipeline progress: {e}")


def main():
    pipeline = TrainingPipeline()
    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
