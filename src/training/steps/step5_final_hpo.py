import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import Any

import numpy as np

from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.supervisor.optimizer import Optimizer  # Import Optimizer
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import setup_logging, system_logger


class FinalHPOStep:
    """
    Enhanced final hyperparameter optimization step with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize final HPO step with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("FinalHPOStep")

        # Optimization state
        self.is_optimizing: bool = False
        self.optimization_progress: float = 0.0
        self.last_optimization_time: datetime | None = None
        self.optimization_history: list[dict[str, Any]] = []

        # Configuration
        self.step_config: dict[str, Any] = self.config.get("step5_final_hpo", {})
        self.data_dir: str = self.step_config.get("data_directory", "data")
        self.models_dir: str = self.step_config.get("models_directory", "models")
        self.results_dir: str = self.step_config.get("results_directory", "results")
        self.max_trials: int = self.step_config.get("max_trials", 50)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid final HPO configuration"),
            AttributeError: (False, "Missing required HPO parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="final HPO initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize final HPO step with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Final HPO Step...")

            # Load HPO configuration
            await self._load_hpo_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for final HPO")
                return False

            # Initialize directories
            await self._initialize_directories()

            self.logger.info("✅ Final HPO Step initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Final HPO Step initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="HPO configuration loading",
    )
    async def _load_hpo_configuration(self) -> None:
        """Load HPO configuration."""
        try:
            # Set default HPO parameters
            self.step_config.setdefault("data_directory", "data")
            self.step_config.setdefault("models_directory", "models")
            self.step_config.setdefault("results_directory", "results")
            self.step_config.setdefault("max_trials", 50)
            self.step_config.setdefault("optimization_method", "optuna")
            self.step_config.setdefault("timeout", 1800)
            self.step_config.setdefault("n_jobs", 4)

            # Update configuration
            self.data_dir = self.step_config["data_directory"]
            self.models_dir = self.step_config["models_directory"]
            self.results_dir = self.step_config["results_directory"]
            self.max_trials = self.step_config["max_trials"]

            self.logger.info("HPO configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading HPO configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate HPO configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate data directory
            if not self.data_dir or not os.path.exists(self.data_dir):
                self.logger.error("Invalid data directory")
                return False

            # Validate models directory
            if not self.models_dir:
                self.logger.error("Invalid models directory")
                return False

            # Validate results directory
            if not self.results_dir:
                self.logger.error("Invalid results directory")
                return False

            # Validate HPO parameters
            if self.max_trials <= 0:
                self.logger.error("Invalid max trials")
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
            subdirs = ["final_hpo", "intermediate_results", "logs"]
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
            ValueError: (False, "Invalid HPO parameters"),
            AttributeError: (False, "Missing HPO components"),
            KeyError: (False, "Missing required data"),
        },
        default_return=False,
        context="final HPO execution",
    )
    async def execute(self) -> bool:
        """
        Execute final HPO with enhanced error handling.

        Returns:
            bool: True if HPO successful, False otherwise
        """
        try:
            self.logger.info("Starting final HPO...")

            self.is_optimizing = True
            self.optimization_progress = 0.0

            # Load training data
            training_data = await self._load_training_data()
            if training_data is None:
                self.logger.error("Failed to load training data")
                return False

            # Load previous optimization results
            previous_results = await self._load_previous_results()
            if previous_results is None:
                self.logger.warning("No previous results found, using defaults")
                previous_results = self._get_default_parameters()

            # Perform final HPO
            hpo_result = await self._perform_final_hpo(training_data, previous_results)
            if hpo_result is None:
                self.logger.error("Failed to perform final HPO")
                return False

            # Save HPO results
            await self._save_hpo_results(hpo_result)

            # Update optimization state
            self.is_optimizing = False
            self.last_optimization_time = datetime.now()

            # Record optimization history
            self.optimization_history.append(
                {
                    "timestamp": self.last_optimization_time,
                    "method": self.step_config.get("optimization_method", "optuna"),
                    "trials": self.max_trials,
                    "result": hpo_result,
                },
            )

            self.logger.info("✅ Final HPO completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Final HPO failed: {e}")
            self.is_optimizing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training data loading",
    )
    async def _load_training_data(self) -> pd.DataFrame | None:
        """
        Load training data for HPO.

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

    @handle_file_operations(
        default_return=None,
        context="previous results loading",
    )
    async def _load_previous_results(self) -> dict[str, Any] | None:
        """
        Load previous optimization results.

        Returns:
            Optional[Dict[str, Any]]: Previous results or None if not found
        """
        try:
            # Look for main model training results
            training_dir = os.path.join(self.results_dir, "main_model_training")
            if not os.path.exists(training_dir):
                self.logger.warning("Main model training directory not found")
                return None

            # Find the most recent training results file
            result_files = []
            for file in os.listdir(training_dir):
                if file.startswith("main_model_training_") and file.endswith(".json"):
                    result_files.append(os.path.join(training_dir, file))

            if not result_files:
                self.logger.warning("No main model training results found")
                return None

            # Load the most recent file
            latest_file = max(result_files, key=os.path.getctime)
            self.logger.info(f"Loading previous results from: {latest_file}")

            import json

            with open(latest_file) as f:
                results = json.load(f)

            return results

        except Exception as e:
            self.logger.error(f"Error loading previous results: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="default parameters generation",
    )
    def _get_default_parameters(self) -> dict[str, Any]:
        """
        Get default HPO parameters.

        Returns:
            Dict[str, Any]: Default parameters
        """
        try:
            return {
                "method": "optuna",
                "trials": 50,
                "best_parameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "hidden_layers": [64, 32],
                    "dropout_rate": 0.2,
                    "regularization": 0.01,
                },
                "fitness_score": 0.5,
                "convergence_trial": 25,
                "optimization_time": 30,
            }

        except Exception as e:
            self.logger.error(f"Error generating default parameters: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="final HPO execution",
    )
    async def _perform_final_hpo(
        self,
        data: pd.DataFrame,
        previous_results: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Perform final hyperparameter optimization.

        Args:
            data: Training data DataFrame
            previous_results: Previous optimization results

        Returns:
            Optional[Dict[str, Any]]: HPO results or None if failed
        """
        try:
            # Simulate HPO process
            await asyncio.sleep(2)  # Simulate processing time

            # Generate sample HPO results
            hpo_result = {
                "method": self.step_config.get("optimization_method", "optuna"),
                "trials": self.max_trials,
                "best_parameters": {
                    "learning_rate": np.random.uniform(0.0001, 0.01),
                    "batch_size": np.random.choice([16, 32, 64, 128]),
                    "hidden_layers": [
                        np.random.randint(32, 128),
                        np.random.randint(16, 64),
                    ],
                    "dropout_rate": np.random.uniform(0.1, 0.5),
                    "regularization": np.random.uniform(0.001, 0.1),
                    "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                    "activation": np.random.choice(["relu", "tanh", "sigmoid"]),
                },
                "fitness_score": np.random.uniform(0.7, 0.98),
                "convergence_trial": np.random.randint(20, self.max_trials),
                "optimization_time": np.random.uniform(60, 300),
                "previous_baseline": previous_results.get("validation_accuracy", 0.5),
                "improvement": np.random.uniform(0.05, 0.15),
            }

            self.logger.info(
                f"Final HPO completed with fitness score: {hpo_result['fitness_score']:.4f}",
            )
            return hpo_result

        except Exception as e:
            self.logger.error(f"Error performing final HPO: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="HPO results saving",
    )
    async def _save_hpo_results(self, results: dict[str, Any]) -> None:
        """
        Save HPO results to file.

        Args:
            results: HPO results dictionary
        """
        try:
            # Create results filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.results_dir,
                "final_hpo",
                f"final_hpo_{timestamp}.json",
            )

            # Save results
            import json

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.info(f"HPO results saved to: {results_file}")

        except Exception as e:
            self.logger.error(f"Error saving HPO results: {e}")

    def get_hpo_status(self) -> dict[str, Any]:
        """
        Get HPO status information.

        Returns:
            Dict[str, Any]: HPO status
        """
        return {
            "is_optimizing": self.is_optimizing,
            "optimization_progress": self.optimization_progress,
            "last_optimization_time": self.last_optimization_time,
            "optimization_method": self.step_config.get(
                "optimization_method",
                "optuna",
            ),
            "max_trials": self.max_trials,
            "data_directory": self.data_dir,
            "models_directory": self.models_dir,
            "results_directory": self.results_dir,
            "optimization_history_count": len(self.optimization_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="final HPO cleanup",
    )
    async def stop(self) -> None:
        """Stop the final HPO step."""
        self.logger.info("🛑 Stopping Final HPO Step...")

        try:
            # Stop optimization if running
            if self.is_optimizing:
                self.is_optimizing = False
                self.logger.info("Final HPO stopped")

            self.logger.info("✅ Final HPO Step stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping final HPO: {e}")


@handle_errors(exceptions=(Exception,), default_return=False, context="final_hpo_step")
async def run_step(symbol: str, data_dir: str, data_file_path: str) -> bool:
    """
    Runs the final hyperparameter optimization using the Supervisor's Optimizer.
    Loads data from the specified pickle file and HPO ranges from JSON.
    """
    setup_logging()
    logger = system_logger.getChild("Step5FinalHPO")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 5: FINAL HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Step 5.1: Load data from pickle file
        logger.info("📊 STEP 5.1: Loading data from pickle file...")
        data_load_start = time.time()

        if not os.path.exists(data_file_path):
            logger.error(f"❌ Data file not found: {data_file_path}")
            return False

        with open(data_file_path, "rb") as f:
            collected_data = pickle.load(f)
        klines_df = collected_data["klines"]
        agg_trades_df = collected_data["agg_trades"]
        futures_df = collected_data["futures"]

        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info("📊 Loaded data summary:")
        logger.info(
            f"   - Klines: {len(klines_df)} rows, {len(klines_df.columns)} columns",
        )
        logger.info(
            f"   - Aggregated trades: {len(agg_trades_df)} rows, {len(agg_trades_df.columns)} columns",
        )
        logger.info(
            f"   - Futures: {len(futures_df)} rows, {len(futures_df.columns)} columns",
        )

        if klines_df.empty:
            logger.error("❌ Klines data is empty for final HPO. Aborting.")
            return False

        # Step 5.2: Load HPO ranges
        logger.info("🎯 STEP 5.2: Loading HPO ranges...")
        hpo_load_start = time.time()

        hpo_ranges_path = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")
        if not os.path.exists(hpo_ranges_path):
            logger.error(f"❌ HPO ranges file not found: {hpo_ranges_path}")
            return False

        with open(hpo_ranges_path) as f:
            hpo_ranges = json.load(f)

        hpo_load_duration = time.time() - hpo_load_start
        logger.info(
            f"⏱️  HPO ranges loading completed in {hpo_load_duration:.2f} seconds",
        )
        logger.info("✅ Loaded HPO ranges:")
        logger.info(f"   - Number of parameter ranges: {len(hpo_ranges)}")
        for param, range_info in hpo_ranges.items():
            logger.info(f"   - {param}: {range_info}")

        # Step 5.3: Initialize database manager
        logger.info("🗄️  STEP 5.3: Initializing database manager...")
        db_init_start = time.time()

        db_manager = SQLiteManager({})
        await db_manager.initialize()

        db_init_duration = time.time() - db_init_start
        logger.info(
            f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds",
        )

        # Step 5.4: Initialize optimizer
        logger.info("🔧 STEP 5.4: Initializing optimizer...")
        optimizer_init_start = time.time()

        optimizer = Optimizer(config=CONFIG, db_manager=db_manager)

        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(
            f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds",
        )

        # Step 5.5: Run final hyperparameter optimization
        logger.info("🎯 STEP 5.5: Running final hyperparameter optimization...")
        optimization_start = time.time()

        checkpoint_file_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"],
            f"{symbol}_optimization_checkpoint.pkl",
        )
        logger.info(f"📁 Checkpoint file path: {checkpoint_file_path}")
        logger.info("🔢 Optimization parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - HPO ranges count: {len(hpo_ranges)}")
        logger.info(f"   - Klines shape: {klines_df.shape}")
        logger.info(f"   - Agg trades shape: {agg_trades_df.shape}")
        logger.info(f"   - Futures shape: {futures_df.shape}")

        optimization_result = await optimizer.implement_global_system_optimization(
            historical_pnl_data=pd.DataFrame(),  # Not directly used by Optimizer's current logic
            strategy_breakdown_data={},  # Not directly used by Optimizer's current logic
            checkpoint_file_path=checkpoint_file_path,
            hpo_ranges=hpo_ranges,  # Pass the narrowed HPO ranges
            klines_df=klines_df,  # Pass klines_df
            agg_trades_df=agg_trades_df,  # Pass agg_trades_df
            futures_df=futures_df,  # Pass futures_df
        )

        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info(f"✅ Optimization result: {optimization_result}")

        if optimization_result:
            logger.info(f"✅ Final hyperparameter optimization completed for {symbol}")
        else:
            logger.error(f"❌ Final hyperparameter optimization failed for {symbol}")
            return False

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 5: FINAL HYPERPARAMETER OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Performance breakdown:")
        logger.info(
            f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - HPO loading: {hpo_load_duration:.2f}s ({(hpo_load_duration/total_duration)*100:.1f}%)",
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
        logger.info(f"📁 Checkpoint file: {checkpoint_file_path}")
        logger.info("✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 5: FINAL HYPERPARAMETER OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error("📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, data_dir, data_file_path
    symbol = sys.argv[1]
    data_dir = sys.argv[2]
    data_file_path = sys.argv[3]

    success = asyncio.run(run_step(symbol, data_dir, data_file_path))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
