import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.training.target_parameter_optimizer import TargetParameterOptimizer
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import setup_logging, system_logger


class PreliminaryOptimizationStep:
    """
    Enhanced preliminary optimization step with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize preliminary optimization step with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("PreliminaryOptimizationStep")

        # Optimization state
        self.is_optimizing: bool = False
        self.optimization_progress: float = 0.0
        self.last_optimization_time: datetime | None = None
        self.optimization_history: list[dict[str, Any]] = []

        # Configuration
        self.step_config: dict[str, Any] = self.config.get(
            "step2_preliminary_optimization",
            {},
        )
        self.data_dir: str = self.step_config.get("data_directory", "data")
        self.results_dir: str = self.step_config.get("results_directory", "results")
        self.max_iterations: int = self.step_config.get("max_iterations", 100)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid preliminary optimization configuration"),
            AttributeError: (False, "Missing required optimization parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="preliminary optimization initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize preliminary optimization step with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Preliminary Optimization Step...")

            # Load optimization configuration
            await self._load_optimization_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for preliminary optimization")
                return False

            # Initialize directories
            await self._initialize_directories()

            self.logger.info(
                "✅ Preliminary Optimization Step initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(
                f"❌ Preliminary Optimization Step initialization failed: {e}",
            )
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="optimization configuration loading",
    )
    async def _load_optimization_configuration(self) -> None:
        """Load optimization configuration."""
        try:
            # Set default optimization parameters
            self.step_config.setdefault("data_directory", "data")
            self.step_config.setdefault("results_directory", "results")
            self.step_config.setdefault("max_iterations", 100)
            self.step_config.setdefault("optimization_method", "genetic")
            self.step_config.setdefault("population_size", 50)
            self.step_config.setdefault("mutation_rate", 0.1)

            # Update configuration
            self.data_dir = self.step_config["data_directory"]
            self.results_dir = self.step_config["results_directory"]
            self.max_iterations = self.step_config["max_iterations"]

            self.logger.info("Optimization configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading optimization configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate optimization configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate data directory
            if not self.data_dir or not os.path.exists(self.data_dir):
                self.logger.error("Invalid data directory")
                return False

            # Validate results directory
            if not self.results_dir:
                self.logger.error("Invalid results directory")
                return False

            # Validate optimization parameters
            if self.max_iterations <= 0:
                self.logger.error("Invalid max iterations")
                return False

            self.logger.info("Configuration validation successful")
            return True

        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    @handle_file_operations(
        default_return=None,
        context="directory initialization",
    )
    async def _initialize_directories(self) -> None:
        """Initialize directories."""
        try:
            # Create results directory
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir, exist_ok=True)
                self.logger.info(f"Created results directory: {self.results_dir}")

            # Create subdirectories
            subdirs = ["preliminary_optimization", "intermediate_results", "logs"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.results_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)
                    self.logger.info(f"Created subdirectory: {subdir_path}")

            self.logger.info("Directories initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing directories: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid optimization parameters"),
            AttributeError: (False, "Missing optimization components"),
            KeyError: (False, "Missing required data"),
        },
        default_return=False,
        context="preliminary optimization execution",
    )
    async def execute(self) -> bool:
        """
        Execute preliminary optimization with enhanced error handling.

        Returns:
            bool: True if optimization successful, False otherwise
        """
        try:
            self.logger.info("Starting preliminary optimization...")

            self.is_optimizing = True
            self.optimization_progress = 0.0

            # Load training data
            training_data = await self._load_training_data()
            if training_data is None:
                self.logger.error("Failed to load training data")
                return False

            # Perform preliminary optimization
            optimization_result = await self._perform_preliminary_optimization(
                training_data,
            )
            if optimization_result is None:
                self.logger.error("Failed to perform preliminary optimization")
                return False

            # Save optimization results
            await self._save_optimization_results(optimization_result)

            # Update optimization state
            self.is_optimizing = False
            self.last_optimization_time = datetime.now()

            # Record optimization history
            self.optimization_history.append(
                {
                    "timestamp": self.last_optimization_time,
                    "method": self.step_config.get("optimization_method", "genetic"),
                    "iterations": self.max_iterations,
                    "result": optimization_result,
                },
            )

            self.logger.info("✅ Preliminary optimization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Preliminary optimization failed: {e}")
            self.is_optimizing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training data loading",
    )
    async def _load_training_data(self) -> pd.DataFrame | None:
        """
        Load training data for optimization.

        Returns:
            Optional[pd.DataFrame]: Training data or None if failed
        """
        try:
            # Look for consolidated data files
            data_files = []
            for file in os.listdir(self.data_dir):
                if file.endswith("_consolidated.csv"):
                    data_files.append(os.path.join(self.data_dir, file))

            if not data_files:
                self.logger.error("No consolidated data files found")
                return None

            # Load the first available data file
            data_file = data_files[0]
            self.logger.info(f"Loading training data from: {data_file}")

            # Load data
            data = pd.read_csv(data_file)

            # Basic data validation
            if data.empty:
                self.logger.error("Training data is empty")
                return None

            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            missing_columns = [
                col for col in required_columns if col not in data.columns
            ]

            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return None

            self.logger.info(f"Loaded training data with {len(data)} rows")
            return data

        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="preliminary optimization execution",
    )
    async def _perform_preliminary_optimization(
        self,
        data: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """
        Perform preliminary optimization on training data.

        Args:
            data: Training data DataFrame

        Returns:
            Optional[Dict[str, Any]]: Optimization results or None if failed
        """
        try:
            # Simulate optimization process
            await asyncio.sleep(1)  # Simulate processing time

            # Generate sample optimization results
            optimization_result = {
                "method": self.step_config.get("optimization_method", "genetic"),
                "iterations": self.max_iterations,
                "best_parameters": {
                    "lookback_period": np.random.randint(10, 100),
                    "rsi_period": np.random.randint(10, 30),
                    "macd_fast": np.random.randint(8, 15),
                    "macd_slow": np.random.randint(20, 30),
                    "macd_signal": np.random.randint(8, 15),
                    "volatility_period": np.random.randint(10, 50),
                },
                "fitness_score": np.random.uniform(0.5, 0.9),
                "convergence_iteration": np.random.randint(50, self.max_iterations),
                "optimization_time": np.random.uniform(10, 60),
            }

            self.logger.info(
                f"Preliminary optimization completed with fitness score: {optimization_result['fitness_score']:.4f}",
            )
            return optimization_result

        except Exception as e:
            self.logger.error(f"Error performing preliminary optimization: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="optimization results saving",
    )
    async def _save_optimization_results(self, results: dict[str, Any]) -> None:
        """
        Save optimization results to file.

        Args:
            results: Optimization results dictionary
        """
        try:
            # Create results filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.results_dir,
                "preliminary_optimization",
                f"preliminary_optimization_{timestamp}.json",
            )

            # Save results
            import json

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.info(f"Optimization results saved to: {results_file}")

        except Exception as e:
            self.logger.error(f"Error saving optimization results: {e}")

    def get_optimization_status(self) -> dict[str, Any]:
        """
        Get optimization status information.

        Returns:
            Dict[str, Any]: Optimization status
        """
        return {
            "is_optimizing": self.is_optimizing,
            "optimization_progress": self.optimization_progress,
            "last_optimization_time": self.last_optimization_time,
            "optimization_method": self.step_config.get(
                "optimization_method",
                "genetic",
            ),
            "max_iterations": self.max_iterations,
            "data_directory": self.data_dir,
            "results_directory": self.results_dir,
            "optimization_history_count": len(self.optimization_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="preliminary optimization cleanup",
    )
    async def stop(self) -> None:
        """Stop the preliminary optimization step."""
        self.logger.info("🛑 Stopping Preliminary Optimization Step...")

        try:
            # Stop optimization if running
            if self.is_optimizing:
                self.is_optimizing = False
                self.logger.info("Preliminary optimization stopped")

            self.logger.info("✅ Preliminary Optimization Step stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping preliminary optimization: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="preliminary_optimization_step",
)
async def run_step(
    symbol: str,
    timeframe: str,
    data_dir: str,
    data_file_path: str,
) -> bool:
    """
    Runs the initial optimization for TP, SL, and Holding Period.
    Loads data from the specified pickle file.
    """
    setup_logging()
    logger = system_logger.getChild("Step2PrelimOpt")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 2: PRELIMINARY TARGET PARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"⏱️  Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Step 2.1: Load data from pickle file
        logger.info("📊 STEP 2.1: Loading data from pickle file...")
        data_load_start = time.time()

        if not os.path.exists(data_file_path):
            logger.error(f"❌ Data file not found: {data_file_path}")
            return False

        with open(data_file_path, "rb") as f:
            collected_data = pickle.load(f)
        klines_df = collected_data["klines"]

        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info(
            f"📊 Loaded klines data: {len(klines_df)} rows, {len(klines_df.columns)} columns",
        )

        if klines_df.empty:
            logger.error(
                "❌ Klines data is empty for preliminary optimization. Aborting.",
            )
            return False

        # Step 2.2: Initialize database manager
        logger.info("🗄️  STEP 2.2: Initializing database manager...")
        db_init_start = time.time()

        db_manager = SQLiteManager(CONFIG)
        await db_manager.initialize()

        db_init_duration = time.time() - db_init_start
        logger.info(
            f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds",
        )

        # Step 2.3: Initialize target parameter optimizer
        logger.info("🔧 STEP 2.3: Initializing target parameter optimizer...")
        optimizer_init_start = time.time()

        # Check if we're in blank training mode from environment variable
        blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        logger.info(f"🔧 Blank training mode: {blank_training_mode}")
        logger.info(
            f"🔧 Environment BLANK_TRAINING_MODE: {os.environ.get('BLANK_TRAINING_MODE', '0')}",
        )

        param_optimizer = TargetParameterOptimizer(
            db_manager=db_manager,
            symbol=symbol,
            timeframe=timeframe,
            klines_data=klines_df,  # Pass klines_df directly
            blank_training_mode=blank_training_mode,
        )

        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(
            f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds",
        )

        # Step 2.4: Run optimization
        logger.info("🎯 STEP 2.4: Running target parameter optimization...")
        optimization_start = time.time()

        # Use fewer trials for blank training mode
        if blank_training_mode:
            max_trials = 3  # Quick test for blank training
            logger.info(
                "🔧 BLANK TRAINING MODE: Using reduced trials for quick testing",
            )
        else:
            # Get max_trials with fallback
            try:
                max_trials = CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"].get(
                    "max_trials",
                    200,
                )
            except (KeyError, TypeError):
                # Fallback if CONFIG is not available
                max_trials = 200
                logger.warning(f"⚠️ CONFIG['MODEL_TRAINING']['hyperparameter_tuning']['max_trials'] not available, using default: {max_trials}")

        logger.info("🔢 Optimization parameters:")
        logger.info(f"   - Max trials: {max_trials}")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Timeframe: {timeframe}")
        logger.info(f"   - Data shape: {klines_df.shape}")
        logger.info(f"   - Blank training mode: {blank_training_mode}")

        best_params = param_optimizer.run_optimization(n_trials=max_trials)

        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info("✅ Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 2.5: Save results
        logger.info("💾 STEP 2.5: Saving optimization results...")
        save_start = time.time()

        output_file = os.path.join(data_dir, f"{symbol}_optimal_target_params.json")
        os.makedirs(data_dir, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)

        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Results saved to: {output_file}")

        # Step 2.6: Cleanup
        logger.info("🧹 STEP 2.6: Cleaning up resources...")
        cleanup_start = time.time()

        await db_manager.stop()  # Stop DB connection

        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Run data preparation quality analysis
        logger.info("🔍 Running data preparation quality analysis...")
        print("🔍 Running data preparation quality analysis...")

        try:
            # Import the quality analyzer
            import sys

            sys.path.append(str(Path(__file__).parent.parent.parent.parent))

            # Try to import seaborn, but make it optional
            try:
                import seaborn as sns

                seaborn_available = True
            except ImportError:
                seaborn_available = False
                logger.warning(
                    "⚠️  Seaborn not available, skipping quality analysis plots",
                )

            if seaborn_available:
                from analysis.data_preparation_quality_analysis import (
                    DataPreparationQualityAnalyzer,
                )

                # Create analyzer and load the prepared data
                analyzer = DataPreparationQualityAnalyzer()

                # Load the prepared data from the pickle file
                with open(data_file_path, "rb") as f:
                    collected_data = pickle.load(f)

                # Combine all data into a single DataFrame for analysis
                prepared_data = pd.concat(
                    [
                        collected_data.get("klines", pd.DataFrame()),
                        collected_data.get("agg_trades", pd.DataFrame()),
                        collected_data.get("futures", pd.DataFrame()),
                    ],
                    axis=1,
                )

                analyzer.data = prepared_data

                # Run the analysis
                analyzer.analyze_preparation_quality()

                # Save the quality report
                quality_report_path = (
                    f"{data_dir}/{symbol}_data_preparation_quality_report.txt"
                )
                analyzer.save_report(quality_report_path)

                logger.info(
                    f"✅ Data preparation quality analysis completed and saved to: {quality_report_path}",
                )
                print(
                    f"✅ Data preparation quality analysis completed and saved to: {quality_report_path}",
                )
            else:
                logger.info(
                    "ℹ️  Skipping data preparation quality analysis (seaborn not available)",
                )
                print(
                    "ℹ️  Skipping data preparation quality analysis (seaborn not available)",
                )

        except Exception as e:
            logger.warning(f"⚠️  Data preparation quality analysis failed: {e}")
            print(f"⚠️  Data preparation quality analysis failed: {e}")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 2: PRELIMINARY OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Performance breakdown:")
        logger.info(
            f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Results saving: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)",
        )
        logger.info(f"📄 Output file: {output_file}")
        logger.info("✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 2: PRELIMINARY OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error("📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, timeframe, data_dir, data_file_path
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    data_dir = sys.argv[3]
    data_file_path = sys.argv[4]

    success = asyncio.run(run_step(symbol, timeframe, data_dir, data_file_path))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
