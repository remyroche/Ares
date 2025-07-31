# src/utils/model_manager.py
import os
import json
import copy
from typing import Optional, Dict, Any, TYPE_CHECKING
from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.async_utils import async_file_manager
from src.analyst.analyst import Analyst
from src.tactician.tactician import Tactician
from src.strategist.strategist import Strategist
import datetime

if TYPE_CHECKING:
    from src.supervisor.performance_reporter import PerformanceReporter


class ModelManager:
    """
    Manages the loading, serving, and hot-swapping of trading models, parameters, and their versions.
    This allows for updating the strategy without restarting the bot, with full version tracking.
    Now uses async operations for better performance.
    """

    def __init__(
        self,
        database_manager=None,
        performance_reporter: Optional["PerformanceReporter"] = None,
    ):
        self.logger = system_logger.getChild("ModelManager")
        self.database_manager = database_manager
        self.performance_reporter = performance_reporter

        self.analyst: Optional[Analyst] = None
        self.tactician: Optional[Tactician] = None
        self.strategist: Optional[Strategist] = None
        self.current_params: Optional[Dict[str, Any]] = None
        self.current_model_version: Optional[Dict[str, Any]] = None

        # Load the initial 'champion' models on startup
        # Note: This is now async, so it should be called with await
        self._load_task = None

    async def initialize(self):
        """Initialize the model manager and load initial models"""
        self.logger.info("Initializing ModelManager...")
        await self.load_models(
            model_version="champion", performance_reporter=self.performance_reporter
        )

    async def load_models(
        self,
        model_version="champion",
        performance_reporter: Optional["PerformanceReporter"] = None,
    ) -> bool:
        """
        Loads a specific version of the models (e.g., 'champion' or 'challenger').
        Instantiates the core logic modules with the appropriate models and parameters.
        Also loads and logs model version metadata.
        """
        self.logger.info(f"Attempting to load '{model_version}' model set...")

        # Define paths for the given model version
        model_dir = os.path.join("models", model_version)
        params_path = os.path.join(model_dir, "optimized_params.json")
        version_path = os.path.join(model_dir, "version.json")

        # Load parameters asynchronously
        if not os.path.exists(params_path):
            if model_version == "champion" and "best_params" in CONFIG:
                params = CONFIG["best_params"]
                self.logger.info(
                    "Using fallback CONFIG['best_params'] as champion parameters."
                )
            else:
                self.logger.error(
                    f"Parameters file not found at {params_path}. Cannot load '{model_version}'."
                )
                return False
        else:
            try:
                params_content = await async_file_manager.read_file(params_path)
                if params_content:
                    params = json.loads(params_content)
                    self.logger.info(
                        f"Loaded '{model_version}' parameters from {params_path}"
                    )
                else:
                    self.logger.error(f"Failed to read parameters from {params_path}")
                    return False
            except Exception as e:
                self.logger.error(f"Error loading parameters from {params_path}: {e}")
                return False

        # Load version info asynchronously
        if not os.path.exists(version_path):
            self.logger.warning(
                f"Version info file not found at {version_path}. Model metadata will be unavailable."
            )
            version_info = {"version": "unknown", "timestamp_utc": "N/A"}
        else:
            try:
                version_content = await async_file_manager.read_file(version_path)
                if version_content:
                    version_info = json.loads(version_content)
                    self.logger.info(
                        f"Loaded '{model_version}' version info: {version_info}"
                    )
                else:
                    self.logger.warning(
                        f"Failed to read version info from {version_path}"
                    )
                    version_info = {"version": "unknown", "timestamp_utc": "N/A"}
            except Exception as e:
                self.logger.error(
                    f"Error loading version info from {version_path}: {e}"
                )
                version_info = {"version": "unknown", "timestamp_utc": "N/A"}

        self.current_params = params
        self.current_model_version = version_info

        # Update CONFIG['best_params'] with the currently loaded parameters for backward compatibility
        CONFIG["best_params"] = copy.deepcopy(self.current_params)

        # Instantiate the core modules
        self.analyst = Analyst(exchange_client=None, state_manager=None)
        self.strategist = Strategist(exchange_client=None, state_manager=None)
        self.tactician = Tactician(
            exchange_client=None,
            state_manager=None,
            performance_reporter=performance_reporter,
        )

        self.logger.info(
            f"Successfully loaded '{model_version}' model set (Version: {version_info.get('version', 'unknown')}). Modules are now active."
        )
        return True

    async def promote_challenger_to_champion(self) -> bool:
        """
        Performs the hot-swap. First, it copies the challenger model files to the champion directory.
        Then, it loads the newly promoted 'champion' models and replaces the live instances.
        """
        self.logger.critical("--- HOT-SWAP: Promoting Challenger model to Champion ---")

        challenger_dir = "models/challenger"
        champion_dir = "models/champion"

        challenger_params_path = os.path.join(challenger_dir, "optimized_params.json")
        challenger_version_path = os.path.join(challenger_dir, "version.json")

        if not os.path.exists(challenger_params_path) or not os.path.exists(
            challenger_version_path
        ):
            self.logger.error(
                "Challenger model files (params or version) are missing. Promotion aborted."
            )
            return False

        try:
            # Create champion directory if it doesn't exist
            os.makedirs(champion_dir, exist_ok=True)

            # Copy challenger files to champion directory asynchronously
            champion_params_path = os.path.join(champion_dir, "optimized_params.json")
            champion_version_path = os.path.join(champion_dir, "version.json")

            # Read challenger files
            challenger_params_content = await async_file_manager.read_file(
                challenger_params_path
            )
            challenger_version_content = await async_file_manager.read_file(
                challenger_version_path
            )

            if not challenger_params_content or not challenger_version_content:
                self.logger.error("Failed to read challenger model files")
                return False

            # Write to champion directory
            await async_file_manager.write_file(
                champion_params_path, challenger_params_content
            )
            await async_file_manager.write_file(
                champion_version_path, challenger_version_content
            )

            self.logger.info(
                "Copied challenger model and version files to champion directory."
            )

            # Now, load the newly promoted champion model
            if await self.load_models(
                model_version="champion", performance_reporter=self.performance_reporter
            ):
                promote_flag_file = CONFIG.get("PROMOTE_CHALLENGER_FLAG_FILE")
                if promote_flag_file and os.path.exists(promote_flag_file):
                    os.remove(promote_flag_file)
                    self.logger.info(
                        f"Removed promotion flag file: {promote_flag_file}"
                    )

                self.logger.critical(
                    "--- HOT-SWAP COMPLETE: System is now running on the newly promoted model. ---"
                )
                return True
            else:
                self.logger.error(
                    "Failed to load the newly promoted champion models. Promotion failed. System continues on old model."
                )
                # NOTE: In a real-world scenario, you might want to handle this failure more gracefully,
                # perhaps by rolling back the file copy or reloading the previous champion.
                return False
        except Exception as e:
            self.logger.error(
                f"An unexpected error occurred during challenger promotion: {e}",
                exc_info=True,
            )
            return False

    def get_analyst(self) -> Optional[Analyst]:
        return self.analyst

    def get_strategist(self) -> Optional[Strategist]:
        return self.strategist

    def get_tactician(
        self, performance_reporter: Optional["PerformanceReporter"] = None
    ) -> Optional[Tactician]:
        if performance_reporter and self.tactician:
            self.tactician.performance_reporter = performance_reporter
        return self.tactician

    def get_current_model_version(self) -> Optional[Dict[str, Any]]:
        """Returns the metadata of the currently loaded model version."""
        return self.current_model_version

    async def save_model_checkpoint(
        self,
        model_name: str,
        checkpoint_data: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> bool:
        """Save a model checkpoint asynchronously"""
        try:
            checkpoint_dir = os.path.join("models", "checkpoints", model_name)
            os.makedirs(checkpoint_dir, exist_ok=True)

            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_checkpoint.json"
            )
            checkpoint_info = {
                "model_name": model_name,
                "checkpoint_data": checkpoint_data,
                "performance_metrics": performance_metrics,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            await async_file_manager.write_json(checkpoint_path, checkpoint_info)
            self.logger.info(f"Saved model checkpoint for {model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model checkpoint for {model_name}: {e}")
            return False

    async def load_model_checkpoint(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load a model checkpoint asynchronously"""
        try:
            checkpoint_dir = os.path.join("models", "checkpoints", model_name)
            checkpoint_path = os.path.join(
                checkpoint_dir, f"{model_name}_checkpoint.json"
            )

            if os.path.exists(checkpoint_path):
                checkpoint_data = await async_file_manager.read_json(checkpoint_path)
                self.logger.info(f"Loaded model checkpoint for {model_name}")
                return checkpoint_data
            else:
                self.logger.warning(f"No checkpoint found for {model_name}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load model checkpoint for {model_name}: {e}")
            return None
