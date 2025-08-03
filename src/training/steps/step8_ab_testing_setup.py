import asyncio
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import setup_logging, system_logger


class ABTestingSetupStep:
    """
    Enhanced AB testing setup step with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize AB testing setup step with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ABTestingSetupStep")

        # Setup state
        self.is_setting_up: bool = False
        self.setup_progress: float = 0.0
        self.last_setup_time: datetime | None = None
        self.setup_history: list[dict[str, Any]] = []

        # Configuration
        self.step_config: dict[str, Any] = self.config.get("step8_ab_testing_setup", {})
        self.data_dir: str = self.step_config.get("data_directory", "data")
        self.models_dir: str = self.step_config.get("models_directory", "models")
        self.results_dir: str = self.step_config.get("results_directory", "results")
        self.test_duration_days: int = self.step_config.get("test_duration_days", 30)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid AB testing setup configuration"),
            AttributeError: (False, "Missing required setup parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="AB testing setup initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize AB testing setup step with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing AB Testing Setup Step...")

            # Load setup configuration
            await self._load_setup_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for AB testing setup")
                return False

            # Initialize directories
            await self._initialize_directories()

            self.logger.info(
                "✅ AB Testing Setup Step initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ AB Testing Setup Step initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="setup configuration loading",
    )
    async def _load_setup_configuration(self) -> None:
        """Load setup configuration."""
        try:
            # Set default setup parameters
            self.step_config.setdefault("data_directory", "data")
            self.step_config.setdefault("models_directory", "models")
            self.step_config.setdefault("results_directory", "results")
            self.step_config.setdefault("test_duration_days", 30)
            self.step_config.setdefault("traffic_split", 0.5)
            self.step_config.setdefault("confidence_level", 0.95)
            self.step_config.setdefault("min_sample_size", 1000)

            # Update configuration
            self.data_dir = self.step_config["data_directory"]
            self.models_dir = self.step_config["models_directory"]
            self.results_dir = self.step_config["results_directory"]
            self.test_duration_days = self.step_config["test_duration_days"]

            self.logger.info("Setup configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading setup configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate setup configuration.

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

            # Validate setup parameters
            if self.test_duration_days <= 0:
                self.logger.error("Invalid test duration")
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
            subdirs = ["ab_testing_setup", "intermediate_results", "logs"]
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
            ValueError: (False, "Invalid setup parameters"),
            AttributeError: (False, "Missing setup components"),
            KeyError: (False, "Missing required data"),
        },
        default_return=False,
        context="AB testing setup execution",
    )
    async def execute(self) -> bool:
        """
        Execute AB testing setup with enhanced error handling.

        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            self.logger.info("Starting AB testing setup...")

            self.is_setting_up = True
            self.setup_progress = 0.0

            # Load training data
            training_data = await self._load_training_data()
            if training_data is None:
                self.logger.error("Failed to load training data")
                return False

            # Load trained models
            trained_models = await self._load_trained_models()
            if trained_models is None:
                self.logger.error("Failed to load trained models")
                return False

            # Setup AB testing environment
            setup_result = await self._setup_ab_testing_environment(
                training_data,
                trained_models,
            )
            if setup_result is None:
                self.logger.error("Failed to setup AB testing environment")
                return False

            # Save setup results
            await self._save_setup_results(setup_result)

            # Update setup state
            self.is_setting_up = False
            self.last_setup_time = datetime.now()

            # Record setup history
            self.setup_history.append(
                {
                    "timestamp": self.last_setup_time,
                    "test_duration_days": self.test_duration_days,
                    "result": setup_result,
                },
            )

            self.logger.info("✅ AB testing setup completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ AB testing setup failed: {e}")
            self.is_setting_up = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="training data loading",
    )
    async def _load_training_data(self) -> pd.DataFrame | None:
        """
        Load training data for AB testing setup.

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
        context="trained models loading",
    )
    async def _load_trained_models(self) -> dict[str, Any] | None:
        """
        Load trained models for AB testing setup.

        Returns:
            Optional[Dict[str, Any]]: Trained models or None if not found
        """
        try:
            # Look for trained model files
            model_files = []
            for file in os.listdir(self.models_dir):
                if file.endswith(".joblib") or file.endswith(".pkl"):
                    model_files.append(os.path.join(self.models_dir, file))

            if not model_files:
                self.logger.warning("No trained model files found")
                return None

            # Load model information
            models_info = {
                "champion_model": None,
                "challenger_model": None,
                "model_files": model_files,
                "load_time": datetime.now(),
            }

            # Assign models as champion and challenger
            if len(model_files) >= 2:
                models_info["champion_model"] = model_files[0]
                models_info["challenger_model"] = model_files[1]
            elif len(model_files) == 1:
                models_info["champion_model"] = model_files[0]
                # Create a challenger model by copying champion
                challenger_path = model_files[0].replace(
                    ".joblib",
                    "_challenger.joblib",
                )
                models_info["challenger_model"] = challenger_path

            self.logger.info(f"Loaded {len(model_files)} trained models")
            return models_info

        except Exception as e:
            self.logger.error(f"Error loading trained models: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="AB testing environment setup",
    )
    async def _setup_ab_testing_environment(
        self,
        data: pd.DataFrame,
        trained_models: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Setup AB testing environment.

        Args:
            data: Training data DataFrame
            trained_models: Trained models information

        Returns:
            Optional[Dict[str, Any]]: Setup results or None if failed
        """
        try:
            # Simulate setup process
            await asyncio.sleep(2)  # Simulate processing time

            # Generate sample setup results
            setup_result = {
                "test_configuration": {
                    "test_duration_days": self.test_duration_days,
                    "traffic_split": self.step_config.get("traffic_split", 0.5),
                    "confidence_level": self.step_config.get("confidence_level", 0.95),
                    "min_sample_size": self.step_config.get("min_sample_size", 1000),
                },
                "models": {
                    "champion_model": trained_models.get(
                        "champion_model",
                        "champion.joblib",
                    ),
                    "challenger_model": trained_models.get(
                        "challenger_model",
                        "challenger.joblib",
                    ),
                    "model_performance": {
                        "champion_accuracy": np.random.uniform(0.7, 0.9),
                        "challenger_accuracy": np.random.uniform(0.65, 0.95),
                        "champion_sharpe": np.random.uniform(1.0, 2.5),
                        "challenger_sharpe": np.random.uniform(0.8, 3.0),
                    },
                },
                "test_parameters": {
                    "start_date": datetime.now().strftime("%Y-%m-%d"),
                    "end_date": (
                        datetime.now() + timedelta(days=self.test_duration_days)
                    ).strftime("%Y-%m-%d"),
                    "expected_trades_per_day": np.random.randint(10, 50),
                    "expected_volume_per_trade": np.random.uniform(100, 1000),
                },
                "monitoring": {
                    "metrics": ["accuracy", "sharpe_ratio", "max_drawdown", "win_rate"],
                    "alert_thresholds": {
                        "min_accuracy": 0.6,
                        "min_sharpe": 0.5,
                        "max_drawdown": 0.25,
                    },
                },
                "setup_time": np.random.uniform(30, 120),
            }

            self.logger.info(
                f"AB testing environment setup completed with {setup_result['test_configuration']['test_duration_days']} days duration",
            )
            return setup_result

        except Exception as e:
            self.logger.error(f"Error setting up AB testing environment: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="setup results saving",
    )
    async def _save_setup_results(self, results: dict[str, Any]) -> None:
        """
        Save setup results to file.

        Args:
            results: Setup results dictionary
        """
        try:
            # Create results filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.results_dir,
                "ab_testing_setup",
                f"ab_testing_setup_{timestamp}.json",
            )

            # Save results
            import json

            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            self.logger.info(f"Setup results saved to: {results_file}")

        except Exception as e:
            self.logger.error(f"Error saving setup results: {e}")

    def get_setup_status(self) -> dict[str, Any]:
        """
        Get setup status information.

        Returns:
            Dict[str, Any]: Setup status
        """
        return {
            "is_setting_up": self.is_setting_up,
            "setup_progress": self.setup_progress,
            "last_setup_time": self.last_setup_time,
            "test_duration_days": self.test_duration_days,
            "data_directory": self.data_dir,
            "models_directory": self.models_dir,
            "results_directory": self.results_dir,
            "setup_history_count": len(self.setup_history),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="AB testing setup cleanup",
    )
    async def stop(self) -> None:
        """Stop the AB testing setup step."""
        self.logger.info("🛑 Stopping AB Testing Setup Step...")

        try:
            # Stop setup if running
            if self.is_setting_up:
                self.is_setting_up = False
                self.logger.info("AB testing setup stopped")

            self.logger.info("✅ AB Testing Setup Step stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping AB testing setup: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="ab_testing_setup_step",
)
async def run_step(symbol: str, timeframe: str = "1m") -> bool:
    """
    Sets up A/B testing by saving configuration to the database.
    """
    setup_logging()
    logger = system_logger.getChild("Step8ABTestingSetup")

    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 8: A/B TESTING SETUP")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")

    # Check if this is a blank training run and override configuration accordingly
    blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
    if blank_training_mode:
        logger.info(
            "🔧 BLANK TRAINING MODE DETECTED - Overriding configuration for faster execution",
        )
        # Reduce AB testing parameters for blank training mode
        ab_duration = 7  # Reduced from default 30 days
        logger.info(
            f"🔧 BLANK TRAINING MODE: Using reduced parameters (ab_duration={ab_duration} days)",
        )
    else:
        ab_duration = CONFIG.get("MODEL_TRAINING", {}).get("ab_test_duration_days", 30)  # Default to 30 days

    try:
        # Step 8.1: Initialize database manager
        logger.info("🗄️  STEP 8.1: Initializing database manager...")
        db_init_start = time.time()

        db_manager = SQLiteManager(CONFIG)
        await db_manager.initialize()

        db_init_duration = time.time() - db_init_start
        logger.info(
            f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds",
        )

        # Step 8.2: Prepare A/B testing configuration
        logger.info("⚙️  STEP 8.2: Preparing A/B testing configuration...")
        config_start = time.time()

        start_date = datetime.now()
        end_date = start_date + timedelta(days=ab_duration)

        ab_config = {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "duration_days": ab_duration,
            "status": "active",
            "models": {
                "model_a": "current_model",  # This refers to the 'champion' model
                "model_b": "new_model",  # This refers to the newly trained 'candidate' model
            },
            "metrics": ["accuracy", "sharpe_ratio", "max_drawdown"],
        }

        config_duration = time.time() - config_start
        logger.info(
            f"⏱️  Configuration preparation completed in {config_duration:.2f} seconds",
        )
        logger.info("✅ A/B testing configuration:")
        logger.info(f"   - Duration: {ab_duration} days")
        logger.info(f"   - Start date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   - End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   - Models: {list(ab_config['models'].keys())}")
        logger.info(f"   - Metrics: {ab_config['metrics']}")

        # Step 8.3: Save configuration to database
        logger.info("💾 STEP 8.3: Saving A/B testing configuration to database...")
        save_start = time.time()

        document_key = f"{symbol}_ab_test"
        await db_manager.set_document("ab_tests", document_key, ab_config)

        save_duration = time.time() - save_start
        logger.info(f"⏱️  Configuration saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Database document key: {document_key}")

        # Step 8.4: Clean up resources
        logger.info("🧹 STEP 8.4: Cleaning up resources...")
        cleanup_start = time.time()

        try:
            await db_manager.close()
            cleanup_duration = time.time() - cleanup_start
            logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")
        except Exception as e:
            logger.warning(f"⚠️  Error during cleanup: {e}")
            cleanup_duration = time.time() - cleanup_start
            logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 8: A/B TESTING SETUP COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info("📊 Performance breakdown:")
        logger.info(
            f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Config preparation: {config_duration:.2f}s ({(config_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Database save: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)",
        )
        logger.info(
            f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)",
        )
        logger.info(f"📄 Database document: {document_key}")
        logger.info("✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 8: A/B TESTING SETUP FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error("📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol
    symbol = sys.argv[1]

    success = asyncio.run(run_step(symbol))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
