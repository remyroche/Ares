#!/usr/bin/env python3
"""
Training Command Line Interface for Ares Trading Bot

This script provides a command-line interface for training operations:
1. Full training for a specific token
2. Model retraining
3. Model import from database
4. Training status and history
5. Full Test Run: Train -> Backtest -> Instruct for Paper Trading
6. Regularization Management: Show and Validate policy (now in src/training/regularization.py)

Usage:
    python scripts/training_cli.py train <symbol> [exchange]
    python scripts/training_cli.py retrain <symbol> [exchange]
    python scripts/training_cli.py import <model_path> <symbol>
    python scripts/training_cli.py status <symbol>
    python scripts/training_cli.py list-tokens
    python scripts/training_cli.py list-models
    python scripts/training_cli.py full-test-run <symbol> [exchange]

If <symbol> is omitted for train, retrain, or full-test-run, the command
will run for ALL supported tokens defined in the configuration.
    # Regularization commands are now in src/training/regularization.py
    # python scripts/training_cli.py show-regularization
    # python scripts/training_cli.py validate-regularization
"""

import asyncio
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from backtesting.ares_backtester import run_backtest
from backtesting.ares_data_preparer import (
    calculate_and_label_regimes,
    get_sr_levels,
    load_raw_data,
)
from backtesting.ares_deep_analyzer import calculate_detailed_metrics

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.training.enhanced_training_manager import EnhancedTrainingManager
from src.utils.logger import setup_logging, system_logger
from src.utils.warning_symbols import error as error_src_utils_warning_symbols, execution_error, failed, warning

# Add the project root to the Python path
project_root=Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TrainingCLI:
    """Command-line interface for training operations."""

    def __init__(self) -> None:
        self.logger=system_logger.getChild("TrainingCLI")
        self.logger.info("🔧 Initializing TrainingCLI...")

        # Initialize SQLiteManager for CLI scope, passed to TrainingManager
        self.db_manager=SQLiteManager({})
        self.training_manager=EnhancedTrainingManager(self.db_manager)

        self.logger.info("✅ TrainingCLI initialized successfully")

    async def initialize(self) -> None:
        """
        Initializes the database manager for CLI operations.
        """
        self.logger.info("🔧 Initializing database manager...")
        await self.db_manager.initialize()
        self.logger.info("✅ Database manager initialized successfully")

    async def run_full_training(self, symbol: str, exchange_name: str="BINANCE") -> bool:
        """Runs the full training pipeline and tags the resulting model as a candidate."""
        start_time=time.time()
        self.logger.info("=" * 80)
        self.logger.info("🚀 FULL TRAINING PIPELINE START")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Exchange: {exchange_name}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            await self.initialize()  # Ensure DB is initialized
            self.logger.info("✅ Database initialization completed")

            print(f"🚀 Starting full training for {symbol} on {exchange_name}...")
            print("=" * 60)

            # The training manager now handles its own data loading and preparation.
            # It returns the MLflow run_id upon completion.
            self.logger.info("🔧 Starting training manager execution...")
            training_start_time=time.time()

            run_id=await self.training_manager.run_full_training(
                symbol=symbol,
                exchange=exchange_name,
            )

            training_duration=time.time() - training_start_time
            self.logger.info(
                f"⏱️  Training manager execution completed in {training_duration:.2f} seconds",
            )

            if run_id:
                self.logger.info(
                    f"✅ Training completed successfully. MLflow Run ID: {run_id}",
                )
                print(f"✅ Training completed successfully. MLflow Run ID: {run_id}")

                # Tag the model in MLflow so the supervisor can find it
                self.logger.info("🔧 Tagging model as 'candidate' in MLflow...")
                client=mlflow.tracking.MlflowClient()
                client.set_tag(run_id, "model_status", "candidate")
                self.logger.info(
                    "✅ Model tagged as 'candidate' for production review.",
                )
                print("✅ Model tagged as 'candidate' for production review.")

                total_duration=time.time() - start_time
                self.logger.info("📊 Full training summary:")
                self.logger.info(f"   Symbol: {symbol}")
                self.logger.info(f"   Exchange: {exchange_name}")
                self.logger.info(f"   MLflow Run ID: {run_id}")
                self.logger.info(
                    f"   Training duration: {training_duration:.2f} seconds",
                )
                self.logger.info(f"   Total duration: {total_duration:.2f} seconds")
                self.logger.info("   Status: SUCCESS")

                return True

            # No run_id returned
            self.logger.error(
                f"❌ Full training failed for {symbol}. No MLflow run ID was returned.",
            )
            print(
                f"❌ Full training failed for {symbol}. No MLflow run ID was returned.",
            )

            total_duration=time.time() - start_time
            self.logger.error("📊 Full training summary:")
            self.logger.error(f"   Symbol: {symbol}")
            self.logger.error(f"   Exchange: {exchange_name}")
            self.logger.error(f"   Training duration: {training_duration:.2f} seconds")
            self.logger.error(f"   Total duration: {total_duration:.2f} seconds")
            self.logger.error("   Status: FAILED")

            return False

        except Exception as e:  # noqa: BLE001
            total_duration=time.time() - start_time
            print(failed(f"💥 Full training failed: {e}"))
            print(error(f"Error type: {type(e).__name__}"))
            print(error("Full traceback:"))
            self.logger.exception(traceback.format_exc())

            print(error("📊 Error context:"))
            print(error(f"   Symbol: {symbol}"))
            print(error(f"   Exchange: {exchange_name}"))
            print(error(f"   Duration: {total_duration:.2f} seconds"))
            print(error(f"   Error: {str(e)}"))

            print(warning(f"Training error: {e}"))
            return False
        finally:
            self.logger.info("🔧 Closing database connection...")
            await self.db_manager.close()  # Close DB connection after operation
            self.logger.info("✅ Database connection closed")

            total_duration=time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info("🏁 FULL TRAINING PIPELINE END")
            self.logger.info("=" * 80)
            self.logger.info(
                f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            )
            self.logger.info(f"Total duration: {total_duration:.2f} seconds")
            self.logger.info(
                f"Status: {'SUCCESS' if 'run_id' in locals() and run_id else 'ERROR'}",
            )

    async def retrain_models(self, symbol: str, exchange_name: str="BINANCE") -> bool:
        """Retrains models. In the new workflow, this is an alias for a full training run."""
        self.logger.info("🔄 Starting model retraining (alias for full training)...")
        print("🔄 Retraining is now an alias for the full training pipeline.")
        try:
            return await self.run_full_training(symbol, exchange_name)
        except Exception as e:  # noqa: BLE001
            print(failed(f"💥 Model retraining failed: {e}"))
            print(error(f"Error type: {type(e).__name__}"))
            print(error("Full traceback:"))
            self.logger.exception(traceback.format_exc())
            print(warning(f"Retraining error: {e}"))
            return False

    async def run_full_test_run(self, symbol: str, exchange_name: str="BINANCE") -> None:
        """
        Executes a full test run: Training -> Backtesting -> Instructions for Paper Trading.
        """
        start_time=time.time()
        self.logger.info("=" * 80)
        self.logger.info("🚀 FULL TEST RUN START")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Exchange: {exchange_name}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n========================================================")
        print(f"🚀 Initiating FULL TEST RUN for {symbol} on {exchange_name}")
        print("========================================================\n")

        # --- Step 1: Run Full Training ---
        self.logger.info("📋 STEP 1/2: Running Full Training Pipeline")
        print("\n--- STEP 1/2: Running Full Training Pipeline ---")
        training_start_time=time.time()

        training_success=await self.run_full_training(symbol, exchange_name)
        training_duration=time.time() - training_start_time

        self.logger.info(
            f"⏱️  Training step completed in {training_duration:.2f} seconds",
        )

        if not training_success:
            print(failed("💥 Full training failed. Aborting full test run."))
            return

        self.logger.info("✅ STEP 1/2 Complete: Training Successful!")
        print("\n--- STEP 1/2 Complete: Training Successful! ---")

        # --- Step 2: Run Backtesting with Newly Optimized Parameters ---
        self.logger.info(
            "📋 STEP 2/2: Running Backtesting with Newly Optimized Parameters",
        )
        print("\n--- STEP 2/2: Running Backtesting with Newly Optimized Parameters ---")
        backtest_start_time=time.time()

        try:
            # Re-initialize DB connection as it was closed after training_manager.run_full_training
            self.logger.info(
                "🔧 Re-initializing database connection for backtesting...",
            )
            await self.db_manager.initialize()
            self.logger.info("✅ Database connection re-initialized")

            self.logger.info("📊 Loading raw data for backtesting...")
            print("Loading raw data for backtesting...")
            # Pass symbol to data loader
            klines_df=load_raw_data(symbol=symbol, exchange=exchange_name)

            if klines_df.empty:  # type: ignore[attr-defined]
                self.logger.error(
                    "💥 Failed to load raw data for backtesting. Aborting.",
                )
                return

            self.logger.info("✅ Raw data loaded successfully:")
            if hasattr(klines_df, "shape"):
                self.logger.info(f"   Klines shape: {klines_df.shape}")

            daily_df=(
                klines_df.resample("D")  # type: ignore[attr-defined]
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    },
                )
                .dropna()
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
            sr_levels=get_sr_levels(daily_df)
            self.logger.info("✅ Daily data prepared and SR levels calculated")

            # CONFIG['BEST_PARAMS'] should now contain the parameters optimized by the training run
            current_best_params=CONFIG.get("best_params", {})
            self.logger.info(
                f"📊 Using optimized parameters for backtest: {current_best_params}",
            )
            print(f"Using optimized parameters for backtest: {current_best_params}")

            self.logger.info("📊 Preparing data for backtesting...")
            print("Preparing data for backtesting...")
            prepared_df=calculate_and_label_regimes(
                klines_df.copy(),
                current_best_params=current_best_params,
                sr_levels=sr_levels,
            )

            if prepared_df.empty:  # type: ignore[attr-defined]
                self.logger.error(
                    "💥 Prepared data for backtesting is empty. Aborting.",
                )
                return

            self.logger.info(
                f"✅ Data prepared for backtesting. Shape: {prepared_df.shape}",
            )

            self.logger.info("📊 Running backtest...")
            print("Running backtest...")
            portfolio=run_backtest(prepared_df, current_best_params)
            self.logger.info("✅ Backtest completed successfully")

            report_lines: list[str] = []
            separator="=" * 80

            report_lines.append("\n" + separator)
            report_lines.append("BACKTESTING RESULTS WITH NEWLY TRAINED MODEL")
            report_lines.append(separator)
            if hasattr(portfolio, "equity"):
                report_lines.append(f"\nFinal Equity: ${portfolio.equity:,.2f}")
            if hasattr(portfolio, "trades"):
                report_lines.append(f"Total Trades: {len(portfolio.trades)}\n")

            # Calculate detailed metrics for a comprehensive report
            num_days_in_backtest=(
                prepared_df.index.max() - prepared_df.index.min()  # type: ignore[attr-defined]
            ).days
            detailed_metrics=calculate_detailed_metrics(
                portfolio=portfolio,
                num_days=num_days_in_backtest,
            )

            report_lines.append("Detailed Metrics:")
            for key, value in detailed_metrics.items():
                report_lines.append(f"  {key:<20}: {value:.2f}")

            report_string="\n".join(report_lines)
            print(report_string)
            self.logger.info(f"📊 Backtesting complete. Results:\n{report_string}")

        except Exception as e:  # noqa: BLE001
            backtest_duration=time.time() - backtest_start_time
            print(failed(f"💥 Backtesting failed during full test run: {e}"))
            print(error(f"Error type: {type(e).__name__}"))
            print(error("Full traceback:"))
            self.logger.exception(traceback.format_exc())
            print(error("📊 Backtest error context:"))
            print(error(f"   Duration: {backtest_duration:.2f} seconds"))
            print(error(f"   Error: {str(e)}"))
            print(failed(f"Backtesting failed: {e}"))
            return
        finally:
            backtest_duration=time.time() - backtest_start_time
            self.logger.info(
                f"⏱️  Backtesting step completed in {backtest_duration:.2f} seconds",
            )
            self.logger.info("🔧 Closing database connection...")
            await self.db_manager.close()  # Close DB connection after backtesting
            self.logger.info("✅ Database connection closed")

        self.logger.info("✅ STEP 2/2 Complete: Backtesting Successful!")
        print("\n--- STEP 2/2 Complete: Backtesting Successful! ---")

        # --- Step 3: Instruct for Paper Trading ---
        total_duration=time.time() - start_time
        self.logger.info("=" * 80)
        self.logger.info("🎉 FULL TEST RUN COMPLETE")
        self.logger.info("=" * 80)
        self.logger.info(f"Symbol: {symbol}")
        self.logger.info(f"Exchange: {exchange_name}")
        self.logger.info(f"Total duration: {total_duration:.2f} seconds")
        self.logger.info(f"Training duration: {training_duration:.2f} seconds")
        self.logger.info(f"Backtesting duration: {backtest_duration:.2f} seconds")

        print("\n========================================================")
        print(f"🎉 FULL TEST RUN COMPLETE for {symbol} on {exchange_name}!")
        print("========================================================\n")
        print("Your models have been trained, optimized, and backtested.")
        print(
            "The new 'candidate' model is ready for live evaluation in paper trading mode.",
        )
        print(
            "\nTo start paper trading with the new model, ensure 'TRADING_ENVIRONMENT' in your .env file is set to 'PAPER', then run:",
        )
        print(f"\n    python scripts/paper_trader_launcher.py {symbol} {exchange_name}")
        print(
            "\nMonitor the logs. The Supervisor component will automatically detect and promote the new model for paper trading.",
        )
        print("It may take a few minutes for the Supervisor to pick up the new model.")
        print("========================================================\n")

    # Removed show_regularization_config and validate_regularization_policy from here
    # as they are now in src/training/regularization.py

    def list_supported_tokens(self) -> None:
        """List all supported tokens and exchanges."""
        self.logger.info("📋 Listing supported tokens and exchanges...")
        print("🪙 Supported Tokens and Exchanges (100x Leverage):")
        print("=" * 60)

        supported_tokens: dict[str, list[str]] = CONFIG.get("SUPPORTED_TOKENS", {})
        self.logger.info(
            f"📊 Found {len(supported_tokens)} exchanges with supported tokens",
        )

        for exchange_name, tokens in supported_tokens.items():
            self.logger.info(f"📈 Exchange {exchange_name}: {len(tokens)} tokens")
            print(f"\n📈 {exchange_name}:")
            for token in tokens:
                print(f"   • {token}")

        self.logger.info("✅ Token listing completed")
        print("\n💡 Usage: python scripts/training_cli.py train <SYMBOL> <EXCHANGE>")
        print("   Example: python scripts/training_cli.py train BTCUSDT BINANCE")
        print("   All tokens support 100x leverage for high-frequency trading")

    def list_model_types(self) -> None:
        """List available model types for training."""
        self.logger.info("📋 Listing available model types...")
        print("🤖 Available Model Types:")
        print("=" * 60)

        model_configs: dict[str, dict[str, Any]] = CONFIG.get("MODEL_TRAINING", {}).get("model_types", {})
        self.logger.info(f"📊 Found {len(model_configs)} model types in configuration")

        for model_name, config in model_configs.items():
            enabled="✅" if config.get("enabled", False) else "❌"
            self.logger.info(
                f"📊 Model {model_name}: {'enabled' if config.get('enabled', False) else 'disabled'}",
            )
            print(f"{enabled} {model_name.upper()}")

            if model_name== "lightgbm":
                print("   - Gradient boosting with LightGBM")
                print("   - Fast training and good performance")
            elif model_name== "xgboost":
                print("   - Extreme gradient boosting")
                print("   - Excellent for structured data")
            elif model_name== "neural_network":
                print("   - Multi-layer perceptron")
                print("   - Good for complex patterns")
            elif model_name== "random_forest":
                print("   - Ensemble of decision trees")
                print("   - Robust and interpretable")

            print()

        self.logger.info("✅ Model type listing completed")

    def show_training_config(self) -> None:
        """Show current training configuration."""
        self.logger.info("📋 Showing current training configuration...")
        print("⚙️ Training Configuration:")
        print("=" * 60)

        config: dict[str, Any] = CONFIG.get("MODEL_TRAINING", {})
        self.logger.info("📊 Training configuration loaded from CONFIG")

        print(f"📊 Data retention: {config.get('data_retention_days', 'N/A')} days")
        print(f"📈 Min data points: {config.get('min_data_points', 'N/A')}")
        print(f"🔀 Train/test split: {config.get('train_test_split', 'N/A')}")
        print(f"✅ Validation split: {config.get('validation_split', 'N/A')}")
        print(f"🚶 Forward walk days: {config.get('forward_walk_days', 'N/A')}")
        print(
            f"🎲 Monte Carlo simulations: {config.get('monte_carlo_simulations', 'N/A')}",
        )
        print(
            f"🔄 A/B test duration: {config.get('ab_test_duration_days', 'N/A')} days",
        )

        print("\n🔧 Regularization:")
        reg_config: dict[str, Any] = config.get("regularization", {})
        print(f"   - L1 alpha: {reg_config.get('l1_alpha', 'N/A')}")
        print(f"   - L2 alpha: {reg_config.get('l2_alpha', 'N/A')}")
        print(f"   - Dropout rate: {reg_config.get('dropout_rate', 'N/A')}")

        print("\n🎯 Hyperparameter tuning:")
        hp_config: dict[str, Any] = config.get("hyperparameter_tuning", {})
        print(f"   - Enabled: {hp_config.get('enabled', 'N/A')}")
        print(f"   - Max trials: {hp_config.get('max_trials', 'N/A')}")
        print(
            f"   - Optimization metric: {hp_config.get('optimization_metric', 'N/A')}",
        )

        self.logger.info("✅ Training configuration display completed")


def print_usage() -> None:
    """Print usage information."""
    print(__doc__)
    print("\nExamples:")
    print("  # Full training for BTCUSDT")
    print("  python scripts/training_cli.py train BTCUSDT BINANCE")
    print()
    print("  # Retrain models for ETHUSDT")
    print("  python scripts/training_cli.py retrain ETHUSDT BINANCE")
    print()
    print("  # Import model from file")
    print("  python scripts/training_cli.py import models/btc_model.pkl BTCUSDT")
    print()
    print("  # Check training status")
    print("  python scripts/training_cli.py status BTCUSDT")
    print()
    print("  # List supported tokens")
    print("  python scripts/training_cli.py list-tokens")
    print()
    print("  # List model types")
    print("  python scripts/training_cli.py list-models")
    print()
    print("  # Show training configuration")
    print("  python scripts/training_cli.py config")
    print()
    print("  # Run full test cycle (Train -> Backtest -> Instruct for Paper Trading)")
    print("  python scripts/training_cli.py full-test-run BTCUSDT BINANCE")
    print("  # Run full test cycle for ALL supported tokens")
    print("  python scripts/training_cli.py full-test-run")
    print()
    print(
        "  # Show the current regularization configuration (now in src/training/regularization.py)",
    )
    print("  python src/training/regularization.py show")
    print()
    print(
        "  # Validate the regularization policy (now in src/training/regularization.py)",
    )
    print("  python src/training/regularization.py validate")


def get_symbols_to_process(argv: list[str]) -> list[tuple[str, str]]:
    """
    Determines which symbols and exchanges to process based on command-line arguments.
    If a symbol is provided, it processes that one. Otherwise, it processes all
    supported tokens from the configuration.
    Returns a list of (symbol, exchange) tuples.
    """
    if len(argv) > 2:  # A specific symbol is provided
        symbol=argv[2].upper()
        exchange=argv[3].upper() if len(argv) > 3 else "BINANCE"
        system_logger.info(f"Processing specific symbol: {symbol} on {exchange}")
        return [(symbol, exchange)]
    # No symbol provided, use all from config
    system_logger.info(
        "No symbol provided. Processing all supported tokens from config.",
    )
    symbols_list: list[tuple[str, str]] = []
    supported_tokens: dict[str, list[str]] = CONFIG.get("SUPPORTED_TOKENS", {})
    for exchange, tokens in supported_tokens.items():
        for token in tokens:
            symbols_list.append((token, exchange))
    if not symbols_list:
        print(warning("No supported tokens found in configuration."))
    return symbols_list


async def main() -> None:
    """Main function."""
    start_time=time.time()

    # Setup logging
    setup_logging()
    logger=system_logger.getChild("TrainingCLIMain")

    logger.info("=" * 80)
    logger.info("🚀 TRAINING CLI START")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Command line arguments: {sys.argv}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {Path.cwd()}")

    if len(sys.argv) < 2:
        print(warning("❌ No command provided, showing usage"))
        print_usage()
        sys.exit(1)

    command=sys.argv[1]
    logger.info(f"📋 Executing command: {command}")

    cli=TrainingCLI()

    try:
        if command in ["train", "retrain", "full-test-run"]:
            symbols_to_process=get_symbols_to_process(sys.argv)
            if not symbols_to_process:
                print(error("No symbols to process. Exiting."))
                sys.exit(1)

            if len(sys.argv) < 3:
                print(
                    f"Running '{command}' for all {len(symbols_to_process)} supported tokens...",
                )

            overall_success=True
            for symbol, exchange in symbols_to_process:
                logger.info(
                    f"--- Processing {symbol} on {exchange} for command '{command}' ---",
                )
                print(
                    f"\n--- Processing {symbol} on {exchange} for command '{command}' ---",
                )

                success=False
                if command == "train":
                    success = await cli.run_full_training(symbol, exchange)
                elif command== "retrain":
                    success = await cli.retrain_models(symbol, exchange)
                elif command== "full-test-run":
                    await cli.run_full_test_run(symbol, exchange)
                    success=True  # Assume success if no exception

                if not success:
                    overall_success = False
                    logger.error(
                        f"--- Command '{command}' failed for {symbol} on {exchange} ---",
                    )

            sys.exit(0 if overall_success else 1)

        elif command== "list-tokens":
            logger.info("📋 Listing supported tokens")
            cli.list_supported_tokens()
            sys.exit(0)

        elif command== "list-models":
            logger.info("📋 Listing model types")
            cli.list_model_types()
            sys.exit(0)

        elif command== "config":
            logger.info("📋 Showing training configuration")
            cli.show_training_config()
            sys.exit(0)

        # Removed handling for show-regularization and validate-regularization here
        # as they are now handled by src/training/regularization.py

        else:
            print(error(f"❌ Unknown command: {command}"))
            print(warning(f"Unknown command: {command}"))
            print_usage()
            sys.exit(1)

    except Exception as e:  # noqa: BLE001
        total_duration=time.time() - start_time
        print(execution_error(f"💥 CRITICAL ERROR in main execution: {e}"))
        print(error(f"Error type: {type(e).__name__}"))
        print(error("Full traceback:"))
        logger.critical(traceback.format_exc())
        print(error("📊 Error context:"))
        print(error(f"   Command: {command}"))
        print(error(f"   Arguments: {sys.argv}"))
        print(error(f"   Duration: {total_duration:.2f} seconds"))
        print(error(f"   Error: {str(e)}"))

        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
    finally:
        total_duration=time.time() - start_time
        logger.info("=" * 80)
        logger.info("🏁 TRAINING CLI END")
        logger.info("=" * 80)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {total_duration:.2f} seconds")
        logger.info(f"Command: {command}")
        logger.info(f"Status: {'SUCCESS' if 'success' in locals() else 'ERROR'}")


if __name__== "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Training CLI interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"\n💥 Unexpected error in main execution: {e}")
        sys.exit(1)
