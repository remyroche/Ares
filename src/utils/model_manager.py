"""
Model manager for loading, serving, and hot-swapping trading models.

This module manages the loading, serving, and hot-swapping of trading models, parameters,
and their versions. This allows for updating the strategy without restarting the bot,
with full version tracking. Now uses async operations for better performance.
"""

import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from src.utils.logger import system_logger

from src.utils.warning_symbols import warning as warn_symbol, _warn_symbol as _warn_symbol
import h5py
import joblib
import pickle
import shutil
from src.utils.error_handler import (
    handle_errors,
    handle_file_operations,
    handle_specific_errors,
    error,
    failed,
    initialization_error,
    invalid,
    missing,
    warning as eh_warning,
)

# --- Compatibility shim for NumPy RNG unpickling across versions ---
_NUMPY_RNG_UNPICKLE_PATCHED = False
_NP_ORIGINAL_BITGEN_CTOR = None  # type: ignore[var-annotated]



def _enable_numpy_rng_unpickle_compat(logger=None) -> None:
    """Enable compatibility for unpickling NumPy RNG BitGenerators (idempotent)."""
    global _NUMPY_RNG_UNPICKLE_PATCHED, _NP_ORIGINAL_BITGEN_CTOR
    if _NUMPY_RNG_UNPICKLE_PATCHED:
        return
    try:
        import numpy.random._pickle as np_random_pickle  # type: ignore[attr-defined]

        original_ctor = getattr(np_random_pickle, "__bit_generator_ctor", None)
        if original_ctor is None:
        # Fallback implementation for original_ctor
            _NUMPY_RNG_UNPICKLE_PATCHED = True
            return

        _NP_ORIGINAL_BITGEN_CTOR = original_ctor
        np_random_pickle.__bit_generator_ctor = _normalized_numpy_bitgen_ctor  # type: ignore[attr-defined]
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            logger.info("Applied NumPy RNG unpickle compatibility shim (ModelManager)")
    except Exception as _shim_exc:  # noqa: BLE001
        _NUMPY_RNG_UNPICKLE_PATCHED = True
        if logger is not None:
            try:
                logger.warning(
                    _warn_symbol(
                        f"NumPy RNG unpickle shim not applied (ModelManager): {_shim_exc}",
                    ),
                )
            except Exception:
                logger.warning(
                    f"NumPy RNG unpickle shim not applied (ModelManager): {_shim_exc}",
                )


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
        self.models: dict[str, dict[str, Any]] = {}
        self.model_metadata: dict[str, Any] = {}
        self.active_model: str | None = None

        # Configuration
        self.model_config: dict[str, Any] = self.config.get("model_manager", {})
        self.models_dir: str = self.model_config.get("models_directory", "models")
        self.metadata_file: str = self.model_config.get(
            "metadata_file",
            "model_metadata.json",
        )
        self.auto_backup: bool = bool(self.model_config.get("auto_backup", True))
        self.max_models: int = int(self.model_config.get("max_models", 10))

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid model manager configuration"),
            AttributeError: (False, "Missing required model parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="model manager initialization",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model configuration loading",
    )
    async def _load_model_configuration(self) -> None:
        """Load model configuration."""
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
        self.models_dir = str(self.model_config["models_directory"])
        self.metadata_file = str(self.model_config["metadata_file"])
        self.auto_backup = bool(self.model_config["auto_backup"])
        self.max_models = int(self.model_config["max_models"])

        self.logger.info("Model configuration loaded successfully")

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
        # Validate models directory
        if not self.models_dir:
            self.logger.error(invalid("Invalid models directory"))
            return False

        # Validate metadata file
        if not self.metadata_file:
            self.logger.error(invalid("Invalid metadata file"))
            return False

        # Validate max models
        if self.max_models <= 0:
            self.logger.error(invalid("Invalid max models"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_file_operations(
        default_return=None,
        context="directory initialization",
    )
    @handle_file_operations(
        default_return=None,
        context="existing models loading",
    )
    async def _load_existing_models(self) -> None:
        """Load existing models and metadata."""
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
        supported_formats: list[str] = self.model_config.get(
            "supported_formats",
            [".joblib", ".pkl", ".h5"],
        )
        if os.path.isdir(self.models_dir):
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
        if not model_name or not model_path:
            self.logger.error(invalid("Invalid model name or path"))
            return False

        if not os.path.exists(model_path):
            self.logger.error(missing(f"Model file not found: {model_path}"))
            return False

        # Check if model already exists
        if model_name in self.models:
            self.logger.warning(warn_symbol(f"Model {model_name} already exists - overwriting"))

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
            self.model_metadata.setdefault("models", {})[model_name] = metadata
        else:
            self.model_metadata.setdefault("models", {})[model_name] = {
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

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="model loading",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="model saving",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="active model setting",
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="active model getting",
    )
    @handle_file_operations(
        default_return=None,
        context="metadata saving",
    )
    async def _save_metadata(self) -> None:
        """Save model metadata to file."""
        metadata_path = os.path.join(self.models_dir, self.metadata_file)
        os.makedirs(self.models_dir, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(self.model_metadata, f, indent=2, default=str)

        self.logger.info(f"Model metadata saved to: {metadata_path}")

    @handle_file_operations(
        default_return=None,
        context="model backup creation",
    )
    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model manager cleanup",
    )

# Global model manager instance
model_manager: ModelManager | None = None


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="model manager setup",
)