"""
Model manager for loading, serving, and hot-swapping trading models.

This module manages the loading, serving, and hot-swapping of trading models,
parameters, and their versions. This allows for updating the strategy without
restarting the bot, with full version tracking. Now uses async operations for
better performance.
"""

import json
import os
from datetime import datetime
from typing import Any

from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class ModelManager:
    """
    Enhanced model manager with comprehensive error handling and type safety.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize model manager with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("ModelManager")

        # Model management
        self.models: dict[str, Any] = {}
        self.model_metadata: dict[str, Any] = {}
        self.active_model: str | None = None

        # Configuration
        self.model_config: dict[str, Any] = self.config.get("model_manager", {})
        self.models_dir: str = self.model_config.get("models_directory", "models")
        self.metadata_file: str = self.model_config.get(
            "metadata_file",
            "model_metadata.json",
        )
        self.auto_backup: bool = self.model_config.get("auto_backup", True)
        self.max_models: int = self.model_config.get("max_models", 10)

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model manager configuration"),
            AttributeError: (False, "Missing required model parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="model manager initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize model manager with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Model Manager...")

            # Load model configuration
            await self._load_model_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for model manager")
                return False

            # Initialize directories
            await self._initialize_directories()

            # Load existing models
            await self._load_existing_models()

            self.logger.info("✅ Model Manager initialization completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Model Manager initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model configuration loading",
    )
    async def _load_model_configuration(self) -> None:
        """Load model configuration."""
        try:
            # Set default model parameters
            self.model_config.setdefault("models_directory", "models")
            self.model_config.setdefault("metadata_file", "model_metadata.json")
            self.model_config.setdefault("auto_backup", True)
            self.model_config.setdefault("max_models", 10)
            self.model_config.setdefault(
                "supported_formats",
                [".joblib", ".pkl", ".h5"],
            )
            self.model_config.setdefault("compression_enabled", True)

            # Update configuration
            self.models_dir = self.model_config["models_directory"]
            self.metadata_file = self.model_config["metadata_file"]
            self.auto_backup = self.model_config["auto_backup"]
            self.max_models = self.model_config["max_models"]

            self.logger.info("Model configuration loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model configuration: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """
        Validate model configuration.

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Validate models directory
            if not self.models_dir:
                self.logger.error("Invalid models directory")
                return False

            # Validate metadata file
            if not self.metadata_file:
                self.logger.error("Invalid metadata file")
                return False

            # Validate max models
            if self.max_models <= 0:
                self.logger.error("Invalid max models")
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
            # Create models directory
            if not os.path.exists(self.models_dir):
                os.makedirs(self.models_dir, exist_ok=True)
                self.logger.info(f"Created models directory: {self.models_dir}")

            # Create subdirectories
            subdirs = ["champion", "challenger", "backups", "archives"]
            for subdir in subdirs:
                subdir_path = os.path.join(self.models_dir, subdir)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)
                    self.logger.info(f"Created subdirectory: {subdir_path}")

            self.logger.info("Directories initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing directories: {e}")

    @handle_file_operations(
        default_return=None,
        context="existing models loading",
    )
    async def _load_existing_models(self) -> None:
        """Load existing models and metadata."""
        try:
            # Load metadata if exists
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    self.model_metadata = json.load(f)
                self.logger.info(f"Loaded model metadata from: {metadata_path}")
            else:
                self.model_metadata = {
                    "models": {},
                    "active_model": None,
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0.0",
                }
                self.logger.info("Created new model metadata")

            # Load existing model files
            supported_formats = self.model_config.get(
                "supported_formats",
                [".joblib", ".pkl", ".h5"],
            )
            for file in os.listdir(self.models_dir):
                if any(file.endswith(fmt) for fmt in supported_formats):
                    model_name = os.path.splitext(file)[0]
                    model_path = os.path.join(self.models_dir, file)

                    # Get file info
                    stat = os.stat(model_path)
                    self.models[model_name] = {
                        "path": model_path,
                        "size": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    }

            # Set active model
            self.active_model = self.model_metadata.get("active_model")

            self.logger.info(f"Loaded {len(self.models)} existing models")

        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model parameters"),
            AttributeError: (False, "Missing model components"),
            KeyError: (False, "Missing required model data"),
        },
        default_return=False,
        context="model registration",
    )
    async def register_model(
        self,
        model_name: str,
        model_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Register a new model.

        Args:
            model_name: Name of the model
            model_path: Path to the model file
            metadata: Optional model metadata

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not model_name or not model_path:
                self.logger.error("Invalid model name or path")
                return False

            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return False

            # Check if model already exists
            if model_name in self.models:
                self.logger.warning(f"Model {model_name} already exists, overwriting")

            # Get file info
            stat = os.stat(model_path)

            # Register model
            self.models[model_name] = {
                "path": model_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "registered": datetime.now().isoformat(),
            }

            # Add metadata
            if metadata:
                self.model_metadata["models"][model_name] = metadata
            else:
                self.model_metadata["models"][model_name] = {
                    "description": f"Model {model_name}",
                    "version": "1.0.0",
                    "created": datetime.now().isoformat(),
                }

            # Update metadata
            self.model_metadata["last_updated"] = datetime.now().isoformat()

            # Save metadata
            await self._save_metadata()

            self.logger.info(f"Model {model_name} registered successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model loading",
    )
    async def load_model(self, model_name: str) -> Any | None:
        """
        Load a model.

        Args:
            model_name: Name of the model to load

        Returns:
            Optional[Any]: Loaded model or None if failed
        """
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return None

            model_path = self.models[model_name]["path"]

            # Load model based on file extension
            if model_path.endswith(".joblib"):
                import joblib

                model = joblib.load(model_path)
            elif model_path.endswith(".pkl"):
                import pickle

                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            elif model_path.endswith(".h5"):
                import h5py

                model = h5py.File(model_path, "r")
            else:
                self.logger.error(f"Unsupported model format: {model_path}")
                return None

            self.logger.info(f"Model {model_name} loaded successfully")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="model saving",
    )
    async def save_model(
        self,
        model: Any,
        model_name: str,
        format: str = "joblib",
    ) -> bool:
        """
        Save a model.

        Args:
            model: Model to save
            model_name: Name for the model
            format: Model format (joblib, pickle, h5)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not model_name:
                self.logger.error("Invalid model name")
                return False

            # Determine file extension
            if format == "joblib":
                extension = ".joblib"
            elif format == "pickle":
                extension = ".pkl"
            elif format == "h5":
                extension = ".h5"
            else:
                self.logger.error(f"Unsupported format: {format}")
                return False

            # Create model path
            model_path = os.path.join(self.models_dir, f"{model_name}{extension}")

            # Save model
            if format == "joblib":
                import joblib

                joblib.dump(model, model_path)
            elif format == "pickle":
                import pickle

                with open(model_path, "wb") as f:
                    pickle.dump(model, f)
            elif format == "h5":
                import h5py

                with h5py.File(model_path, "w") as f:
                    # This is a simplified example - actual implementation depends on model type
                    f.create_dataset("model", data=str(model))

            # Register model
            await self.register_model(model_name, model_path)

            self.logger.info(f"Model {model_name} saved successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="active model setting",
    )
    async def set_active_model(self, model_name: str) -> bool:
        """
        Set the active model.

        Args:
            model_name: Name of the model to set as active

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return False

            self.active_model = model_name
            self.model_metadata["active_model"] = model_name
            self.model_metadata["last_updated"] = datetime.now().isoformat()

            # Save metadata
            await self._save_metadata()

            self.logger.info(f"Active model set to: {model_name}")
            return True

        except Exception as e:
            self.logger.error(f"Error setting active model: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="active model getting",
    )
    async def get_active_model(self) -> str | None:
        """
        Get the active model name.

        Returns:
            Optional[str]: Active model name or None
        """
        try:
            return self.active_model

        except Exception as e:
            self.logger.error(f"Error getting active model: {e}")
            return None

    @handle_file_operations(
        default_return=None,
        context="metadata saving",
    )
    async def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            metadata_path = os.path.join(self.models_dir, self.metadata_file)
            with open(metadata_path, "w") as f:
                json.dump(self.model_metadata, f, indent=2, default=str)

            self.logger.info(f"Model metadata saved to: {metadata_path}")

        except Exception as e:
            self.logger.error(f"Error saving model metadata: {e}")

    @handle_file_operations(
        default_return=None,
        context="model backup creation",
    )
    async def create_backup(self, model_name: str) -> None:
        """
        Create backup of a model.

        Args:
            model_name: Name of the model to backup
        """
        try:
            if model_name not in self.models:
                self.logger.error(f"Model {model_name} not found")
                return

            model_path = self.models[model_name]["path"]
            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                return

            # Create backup directory
            backup_dir = os.path.join(self.models_dir, "backups")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(
                backup_dir,
                f"{model_name}_backup_{timestamp}{os.path.splitext(model_path)[1]}",
            )

            # Copy model file
            import shutil

            shutil.copy2(model_path, backup_path)

            self.logger.info(f"Model backup created: {backup_path}")

        except Exception as e:
            self.logger.error(f"Error creating model backup: {e}")

    def get_model_status(self) -> dict[str, Any]:
        """
        Get model manager status information.

        Returns:
            Dict[str, Any]: Model manager status
        """
        return {
            "total_models": len(self.models),
            "active_model": self.active_model,
            "models_directory": self.models_dir,
            "auto_backup": self.auto_backup,
            "max_models": self.max_models,
            "model_names": list(self.models.keys()),
            "last_updated": self.model_metadata.get("last_updated"),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the model manager."""
        self.logger.info("🛑 Stopping Model Manager...")

        try:
            # Save final metadata
            await self._save_metadata()

            self.logger.info("✅ Model Manager stopped successfully")

        except Exception as e:
            self.logger.error(f"Error stopping model manager: {e}")


# Global model manager instance
model_manager: ModelManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="model manager setup",
)
async def setup_model_manager(
    config: dict[str, Any] | None = None,
) -> ModelManager | None:
    """
    Setup global model manager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Optional[ModelManager]: Global model manager instance
    """
    try:
        global model_manager

        if config is None:
            config = {
                "model_manager": {
                    "models_directory": "models",
                    "metadata_file": "model_metadata.json",
                    "auto_backup": True,
                    "max_models": 10,
                    "supported_formats": [".joblib", ".pkl", ".h5"],
                    "compression_enabled": True,
                },
            }

        # Create model manager
        model_manager = ModelManager(config)

        # Initialize model manager
        success = await model_manager.initialize()
        if success:
            return model_manager
        return None

    except Exception as e:
        print(f"Error setting up model manager: {e}")
        return None
