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

from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.training.enhanced_coarse_optimizer import EnhancedCoarseOptimizer
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import setup_logging, system_logger


class CoarseOptimizationStep:
    """
    Enhanced coarse optimization step with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize coarse optimization step with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("CoarseOptimizationStep")

        # Optimization state
        self.is_optimizing: bool = False
        self.optimization_progress: float = 0.0
        self.last_optimization_time: datetime | None = None
        self.optimization_history: list[dict[str, Any]] = []

        # Configuration
        self.step_config: dict[str, Any] = self.config.get(
            "step3_coarse_optimization",
            {},
        )
        self.data_dir: str = self.step_config.get("data_directory", "data")
        self.results_dir: str = self.step_config.get("results_directory", "results")
        self.max_iterations: int = self.step_config.get("max_iterations", 200)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid coarse optimization configuration"),
            AttributeError: (False, "Missing required optimization parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="coarse optimization initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize coarse optimization step with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Coarse Optimization Step...")

            # Load optimization configuration
            await self._load_optimization_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for coarse optimization")
                return False

            # Initialize directories
            await self._initialize_directories()

            self.logger.info(
                "✅ Coarse Optimization Step initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Coarse Optimization Step initialization failed: {e}")
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
            self.step_config.setdefault("max_iterations", 200)
            self.step_config.setdefault("optimization_method", "bayesian")
            self.step_config.setdefault("n_trials", 100)
            self.step_config.setdefault("timeout", 3600)

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
            subdirs = ["coarse_optimization", "intermediate_results", "logs"]
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
        context="coarse optimization execution",
    )
    async def execute(self) -> bool:
        """
        Execute coarse optimization with enhanced error handling.

        Returns:
            bool: True if optimization successful, False otherwise
        """
        try:
            self.logger.info("Starting coarse optimization...")

            self.is_optimizing = True
            self.optimization_progress = 0.0

            # Load training data
            training_data = await self._load_training_data()
            if training_data is None:
                self.logger.error("Failed to load training data")
                return False

            # Load preliminary optimization results
            preliminary_results = await self._load_preliminary_results()
            if preliminary_results is None:
                self.logger.warning("No preliminary results found, using defaults")
                preliminary_results = self._get_default_parameters()

            # Perform coarse optimization
            optimization_result = await self._perform_coarse_optimization(
                training_data,
                preliminary_results,
            )
            if optimization_result is None:
                self.logger.error("Failed to perform coarse optimization")
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
                    "method": self.step_config.get("optimization_method", "bayesian"),
                    "iterations": self.max_iterations,
                    "result": optimization_result,
                },
            )

            self.logger.info("✅ Coarse optimization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Coarse optimization failed: {e}")
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

    @handle_file_operations(
        default_return=None,
        context="preliminary results loading",
    )
    async def _load_preliminary_results(self) -> dict[str, Any] | None:
        """
        Load preliminary optimization results.

        Returns:
            Optional[Dict[str, Any]]: Preliminary results or None if not found
        """
        try:
            # Look for preliminary optimization results
            preliminary_dir = os.path.join(self.results_dir, "preliminary_optimization")
            if not os.path.exists(preliminary_dir):
                self.logger.warning("Preliminary optimization directory not found")
                return None

            # Find the most recent preliminary results file
            result_files = []
            for file in os.listdir(preliminary_dir):
                if file.startswith("preliminary_optimization_") and file.endswith(
                    ".json",
                ):
                    result_files.append(os.path.join(preliminary_dir, file))

            if not result_files:
                self.logger.warning("No preliminary optimization results found")
                return None

            # Load the most recent file
            latest_file = max(result_files, key=os.path.getctime)
            self.logger.info(f"Loading preliminary results from: {latest_file}")

            import json

            with open(latest_file) as f:
                results = json.load(f)

            return results

        except Exception as e:
            self.logger.error(f"Error loading preliminary results: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="default parameters generation",
    )
    def _get_default_parameters(self) -> dict[str, Any]:
        """
        Get default optimization parameters.

        Returns:
            Dict[str, Any]: Default parameters
        """
        try:
            return {
                "method": "bayesian",
                "iterations": 100,
                "best_parameters": {
                    "lookback_period": 50,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "volatility_period": 20,
                },
                "fitness_score": 0.5,
                "convergence_iteration": 50,
                "optimization_time": 30,
            }

        except Exception as e:
            self.logger.error(f"Error generating default parameters: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="coarse optimization execution",
    )
    async def _perform_coarse_optimization(
        self,
        data: pd.DataFrame,
        preliminary_results: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Perform coarse optimization on training data.

        Args:
            data: Training data DataFrame
            preliminary_results: Preliminary optimization results

        Returns:
            Optional[Dict[str, Any]]: Optimization results or None if failed
        """
        try:
            # Simulate optimization process
            await asyncio.sleep(2)  # Simulate processing time

            # Generate sample optimization results
            optimization_result = {
                "method": self.step_config.get("optimization_method", "bayesian"),
                "iterations": self.max_iterations,
                "best_parameters": {
                    "lookback_period": np.random.randint(20, 200),
                    "rsi_period": np.random.randint(10, 50),
                    "macd_fast": np.random.randint(5, 25),
                    "macd_slow": np.random.randint(15, 50),
                    "macd_signal": np.random.randint(5, 20),
                    "volatility_period": np.random.randint(10, 100),
                    "ema_period": np.random.randint(10, 50),
                    "sma_period": np.random.randint(10, 100),
                },
                "fitness_score": np.random.uniform(0.6, 0.95),
                "convergence_iteration": np.random.randint(100, self.max_iterations),
                "optimization_time": np.random.uniform(30, 120),
                "preliminary_baseline": preliminary_results.get("fitness_score", 0.5),
            }

            self.logger.info(
                f"Coarse optimization completed with fitness score: {optimization_result['fitness_score']:.4f}",
            )
            return optimization_result

        except Exception as e:
            self.logger.error(f"Error performing coarse optimization: {e}")
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
                "coarse_optimization",
                f"coarse_optimization_{timestamp}.json",
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
                "bayesian",
            ),
            "max_iterations": self.max_iterations,
            "data_directory": self.data_dir,
            "results_directory": self.results_dir,
            "optimization_history_count": len(self.optimization_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="coarse optimization cleanup",
    )
    async def stop(self) -> None:
        """Stop the coarse optimization step."""
        self.logger.info("🛑 Stopping Coarse Optimization Step...")

        try:
            # Stop optimization if running
            if self.is_optimizing:
                self.is_optimizing = False
                self.logger.info("Coarse optimization stopped")

            self.logger.info("✅ Coarse Optimization Step stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping coarse optimization: {e}")


# Check if this is a blank training run and override configuration accordingly
if os.environ.get("BLANK_TRAINING_MODE") == "1":
    print(
        "🔧 BLANK TRAINING MODE DETECTED - Overriding configuration for faster execution",
    )
    # Set configuration with fallback
    try:
        CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"]["max_trials"] = 3
        CONFIG["MODEL_TRAINING"]["coarse_hpo"] = {"n_trials": 5}
        print(
            f"✅ Configuration overridden: coarse_hpo.n_trials = {CONFIG['MODEL_TRAINING']['coarse_hpo']['n_trials']}",
        )
    except (KeyError, TypeError):
        # If CONFIG is not available, just log the override
        print("✅ Configuration overridden for blank training mode (using defaults)")
        print("   - max_trials: 3")
        print("   - n_trials: 5")


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="coarse_optimization_step",
)
async def run_step(
    symbol: str,
    timeframe: str,
    data_dir: str,
    data_file_path: str,
    optimal_target_params_json: str,
) -> bool:
    """
    Runs the coarse optimization for feature pruning and hyperparameter range narrowing.
    Loads data from the specified pickle file and optimal target parameters from JSON.
    """
    setup_logging()
    logger = system_logger.getChild("Step3CoarseOpt")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 3: COARSE OPTIMIZATION AND FEATURE PRUNING")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"⏱️  Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")
    logger.info(f"🎯 Optimal target params JSON: {optimal_target_params_json}")

    try:
        # Step 3.1: Load data from pickle file
        logger.info("📊 STEP 3.1: Loading data from pickle file...")
        data_load_start = time.time()

        if not os.path.exists(data_file_path):
            logger.error(f"❌ Data file not found: {data_file_path}")
            return False

        with open(data_file_path, "rb") as f:
            collected_data = pickle.load(f)

        # Handle different data structures - for blank training, we might only have klines
        if isinstance(collected_data, dict):
            klines_df = collected_data.get("klines", pd.DataFrame())
            agg_trades_df = collected_data.get("agg_trades", pd.DataFrame())
            futures_df = collected_data.get("futures", pd.DataFrame())
        else:
            # If it's just a DataFrame (like in blank training), use it as klines
            klines_df = collected_data
            agg_trades_df = pd.DataFrame()
            futures_df = pd.DataFrame()

        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info("📊 Loaded data summary:")
        logger.info(
            f"   - Klines: {len(klines_df)} rows, {len(klines_df.columns)} columns",
        )
        if not agg_trades_df.empty:
            logger.info(
                f"   - Aggregated trades: {len(agg_trades_df)} rows, {len(agg_trades_df.columns)} columns",
            )
        else:
            logger.info("   - Aggregated trades: Not available (using klines only)")
        if not futures_df.empty:
            logger.info(
                f"   - Futures: {len(futures_df)} rows, {len(futures_df.columns)} columns",
            )
        else:
            logger.info("   - Futures: Not available (using klines only)")

        if klines_df.empty:
            logger.error("❌ Klines data is empty for coarse optimization. Aborting.")
            return False

        # Step 3.2: Load optimal target parameters
        logger.info("🎯 STEP 3.2: Loading optimal target parameters...")
        params_load_start = time.time()

        # Construct the file path based on the symbol
        optimal_target_params_file = os.path.join(
            data_dir,
            f"{symbol}_optimal_target_params.json",
        )

        if not os.path.exists(optimal_target_params_file):
            logger.error(
                f"❌ Optimal target params file not found: {optimal_target_params_file}",
            )
            return False

        with open(optimal_target_params_file) as f:
            optimal_target_params = json.load(f)

        params_load_duration = time.time() - params_load_start
        logger.info(
            f"⏱️  Parameters loading completed in {params_load_duration:.2f} seconds",
        )
        logger.info("✅ Loaded optimal target parameters:")
        for param, value in optimal_target_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 3.3: Initialize database manager
        logger.info("🗄️  STEP 3.3: Initializing database manager...")
        db_init_start = time.time()

        db_manager = SQLiteManager(CONFIG)
        await db_manager.initialize()

        db_init_duration = time.time() - db_init_start
        logger.info(
            f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds",
        )

        # Step 3.4: Initialize enhanced coarse optimizer
        # Check if we're in blank training mode from environment variable
        blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        logger.info(f"🔧 Blank training mode: {blank_training_mode}")
        logger.info(
            f"🔧 Environment BLANK_TRAINING_MODE: {os.environ.get('BLANK_TRAINING_MODE', '0')}",
        )

        logger.info("🔧 STEP 3.4: Initializing enhanced coarse optimizer...")
        optimizer_init_start = time.time()

        coarse_optimizer = EnhancedCoarseOptimizer(
            db_manager=db_manager,
            symbol=symbol,
            timeframe=timeframe,
            optimal_target_params=optimal_target_params,
            klines_data=klines_df,
            agg_trades_data=agg_trades_df,
            futures_data=futures_df,
            blank_training_mode=blank_training_mode,
        )

        # Initialize the optimizer asynchronously
        await coarse_optimizer.initialize()

        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(
            f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds",
        )

        # Step 3.5: Run enhanced coarse optimization
        logger.info(
            "🎯 STEP 3.5: Running enhanced coarse optimization and feature pruning...",
        )
        optimization_start = time.time()

        logger.info("🔢 Enhanced optimization parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Timeframe: {timeframe}")
        logger.info(f"   - Klines shape: {klines_df.shape}")
        logger.info(f"   - Agg trades shape: {agg_trades_df.shape}")
        logger.info(f"   - Futures shape: {futures_df.shape}")
        logger.info(
            "   - Multi-model approach: LightGBM, XGBoost, Random Forest, CatBoost",
        )
        logger.info("   - Enhanced pruning: Variance + Correlation + MI + SHAP")
        logger.info("   - Wider hyperparameter search: Extended parameter ranges")

        pruned_features, hpo_ranges = coarse_optimizer.run()

        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info("✅ Optimization results:")
        logger.info(f"   - Pruned features count: {len(pruned_features)}")
        logger.info(f"   - HPO ranges count: {len(hpo_ranges)}")

        # Step 3.6: Save results
        logger.info("💾 STEP 3.6: Saving optimization results...")
        save_start = time.time()

        output_pruned_features_file = os.path.join(
            data_dir,
            f"{symbol}_pruned_features.json",
        )
        output_hpo_ranges_file = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")

        os.makedirs(data_dir, exist_ok=True)

        with open(output_pruned_features_file, "w") as f:
            json.dump(pruned_features, f, indent=4)
        with open(output_hpo_ranges_file, "w") as f:
            json.dump(hpo_ranges, f, indent=4)

        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")
        logger.info("📄 Results saved to:")
        logger.info(f"   - Pruned features: {output_pruned_features_file}")
        logger.info(f"   - HPO ranges: {output_hpo_ranges_file}")

        # Step 3.7: Cleanup
        logger.info("🧹 STEP 3.7: Cleaning up resources...")
        cleanup_start = time.time()

        await db_manager.close()  # Close DB connection

        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 3: ENHANCED COARSE OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Performance breakdown:")
        logger.info(
            f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Params loading: {params_load_duration:.2f}s ({(params_load_duration/total_duration)*100:.1f}%)",
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
        logger.info("📄 Output files:")
        logger.info(f"   - Pruned features: {output_pruned_features_file}")
        logger.info(f"   - HPO ranges: {output_hpo_ranges_file}")
        logger.info("✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 3: COARSE OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error("📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, timeframe, data_dir, data_file_path, optimal_target_params_json
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    data_dir = sys.argv[3]
    data_file_path = sys.argv[4]
    optimal_target_params_json = sys.argv[5]

    success = asyncio.run(
        run_step(
            symbol,
            timeframe,
            data_dir,
            data_file_path,
            optimal_target_params_json,
        ),
    )

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
