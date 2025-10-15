from ..core.decorators import handles_errors
from .logger import system_logger
"""
Model manager for loading, serving, and hot-swapping trading models.

This module manages the loading, serving, and hot-swapping of trading models, parameters,
and their versions. This allows for updating the strategy without restarting the bot,
with full version tracking. Now uses async operations for better performance.
"""
from src.utils.warning_symbols import (
    error,
    failed,
    invalid,
    missing,
    warning as eh_warning
)

import json
import os
import pickle
import shutil
from datetime import datetime
from typing import Any

import h5py
import joblib
import numpy.random._pickle as np_random_pickle
from .common_operations import ensure_directory, format_datetime, get_current_datetime

# File operations functions - using built-in alternatives

from src.utils.warning_symbols import warning_symbol as _warn_symbol
from src.utils.warning_symbols import warning as warn_symbol

import numpy as np
import logging
import time

# --- Compatibility shim for NumPy RNG unpickling across versions ---
_NUMPY_RNG_UNPICKLE_PATCHED = False
_NP_ORIGINAL_BITGEN_CTOR = None  # type: ignore[var-annotated]

# type: ignore[override]
def _normalized_numpy_bitgen_ctor(bit_generator_name: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Normalized ctor to keep picklable; avoids closures.

    Attempts to resolve the bit generator by name/class and call the original constructor
    with a possibly adjusted signature for cross-version compatibility.
    """
    global _NP_ORIGINAL_BITGEN_CTOR  # noqa: F824
    name_candidate: Any = bit_generator_name
    try:
        if hasattr(name_candidate, "__name__"):
            name_candidate = name_candidate.__name__
        elif isinstance(name_candidate, str) and name_candidate.startswith("<class "):
            name_candidate = name_candidate.split(".")[-1].split("'>")[0]
    except Exception as e:
        # Log the exception for debugging but don't fail the operation
        import logging
        logging.getLogger(__name__).warning(f"Failed to parse name_candidate: {e}")

    effective_state = kwargs.get("state", state)
    try:
        # Newer numpy expects (name, state)
        return _NP_ORIGINAL_BITGEN_CTOR(name_candidate, effective_state)  # type: ignore[misc]
    except (TypeError, ValueError):
        # Some versions expect only name
        return _NP_ORIGINAL_BITGEN_CTOR(name_candidate)  # type: ignore[misc]
    except Exception:
        # Last resort: try resolving class directly
        bitgen_cls = getattr(np.random, str(name_candidate), None)
        if bitgen_cls is None and str(name_candidate) == "MT19937":
            bitgen_cls = getattr(np.random, "MT19937", None)
        if bitgen_cls is not None:
            return bitgen_cls()
        raise

def _enable_numpy_rng_unpickle_compat(logger=None) -> None:
    """Enable compatibility for unpickling NumPy RNG BitGenerators (idempotent)."""
    global _NUMPY_RNG_UNPICKLE_PATCHED, _NP_ORIGINAL_BITGEN_CTOR
    if _NUMPY_RNG_UNPICKLE_PATCHED:
        return
    try:
        
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
    def __init__(self, config: dict[str, Any] | None = None, save_path: str | None = None, save_format: str = "joblib", database_manager: Any | None = None, performance_reporter: Any | None = None) -> None:
        """
        Initialize model manager with flexible configuration.

        Args:
            config: Optional configuration dictionary
            save_path: Optional base directory for training-oriented save/load
            save_format: Default model persistence format ('joblib', 'pickle', 'h5')
            database_manager: Optional dependency for backward compatibility
            performance_reporter: Optional dependency for backward compatibility
        """
        self.config: dict[str, Any] = config or {}
        self.logger = system_logger.getChild("ModelManager")
        self.database_manager = database_manager
        self.performance_reporter = performance_reporter

        # Model management
        self.models: dict[str, dict[str, Any]] = {}
        self.model_metadata: dict[str, Any] = {}
        self.active_model: str | None = None

        # Configuration
        self.model_config: dict[str, Any] = self.config.get("model_manager", {}) if isinstance(self.config, dict) else {}
        # Apply defaults immediately so the manager is usable without initialize()
        self.model_config.setdefault("models_directory", "models")
        self.model_config.setdefault("metadata_file", "model_metadata.json")
        self.model_config.setdefault("auto_backup", True)
        self.model_config.setdefault("max_models", 10)
        self.model_config.setdefault("supported_formats", [".joblib", ".pkl", ".h5"])
        self.models_dir: str = save_path or str(self.model_config.get("models_directory"))
        self.metadata_file: str = str(self.model_config.get("metadata_file"))
        self.auto_backup: bool = bool(self.model_config.get("auto_backup"))
        self.max_models: int = int(self.model_config.get("max_models"))
        self.save_format: str = save_format or "joblib"
        # Base path used by training-style persistence APIs
        try:
            os.makedirs(self.models_dir, exist_ok=True)
        except Exception as e:
            # Log the exception for debugging but don't fail the operation
            self.logger.warning(f"Failed to create models directory: {e}")
        self._save_base_path = self.models_dir

    @handles_errors(
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
        self.logger.info("Initializing Model Manager...")

        # Load model configuration
        await self._load_model_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for model manager"))
            return False

        # Initialize directories
        await self._initialize_directories()

        # Load existing models
        await self._load_existing_models()

        self.logger.info("✅ Model Manager initialization completed successfully")
        return True

    @handles_errors(
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

    @handles_errors(
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

    @handles_errors(
        default_return=None,
        context="directory initialization",
    )
    async def _initialize_directories(self) -> None:
        """Initialize directories."""
        # Create models directory
        if not os.path.exists(self.models_dir):
            ensure_directory(self.models_dir)
            self.logger.info(f"Created models directory: {self.models_dir}")

        # Create subdirectories
        subdirs = ["champion", "challenger", "backups", "archives"]
        for subdir in subdirs:
            subdir_path = os.path.join(self.models_dir, subdir)
            if not os.path.exists(subdir_path):
                ensure_directory(subdir_path)
                self.logger.info(f"Created subdirectory: {subdir_path}")

        self.logger.info("Directories initialized successfully")

    @handles_errors(
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
                "last_updated": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
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

    @handles_errors(
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
            "registered": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
        }

        # Add metadata
        if metadata:
            self.model_metadata.setdefault("models", {})[model_name] = metadata
        else:
            self.model_metadata.setdefault("models", {})[model_name] = {
                "description": f"Model {model_name}",
                "version": "1.0.0",
                "created": format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S"),
            }

        # Update metadata
        self.model_metadata["last_updated"] = format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S")

        # Save metadata
        await self._save_metadata()

        self.logger.info(f"Model {model_name} registered successfully")
        return True

    @handles_errors(
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
        # Ensure NumPy RNG pickles created under different versions can be loaded
        _enable_numpy_rng_unpickle_compat(self.logger)
        # Try from registry first
        if model_name in self.models:
            model_path = self.models[model_name]["path"]
        else:
            # Fallback: resolve from models directory by scanning supported formats
            model_path = None
            supported_formats: list[str] = self.model_config.get(
                "supported_formats",
                [".joblib", ".pkl", ".h5"],
            )
            try:
                if os.path.isdir(self.models_dir):
                    for file in os.listdir(self.models_dir):
                        name, ext = os.path.splitext(file)
                        if name == model_name and ext in supported_formats:
                            model_path = os.path.join(self.models_dir, file)
                            break
            except Exception as e:
                self.logger.warning(f"Could not scan models directory: {e}")
            if not model_path:
                self.logger.error(missing(f"Model {model_name} not found"))
                return None

        # Load model based on file extension
        model: Any
        if model_path.endswith(".joblib"):
            model = joblib.load(model_path)
        elif model_path.endswith(".pkl"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        elif model_path.endswith(".h5"):
            model = h5py.File(model_path, "r")
        else:
            self.logger.error(error(f"Unsupported model format: {model_path}"))
            return None

        self.logger.info(f"Model {model_name} loaded successfully")
        return model

    async def list_available_models(self) -> list[str]:
        """List available models by scanning the models directory for supported formats."""
        try:
            supported_formats: list[str] = self.model_config.get(
                "supported_formats",
                [".joblib", ".pkl", ".h5"],
            )
            if not os.path.isdir(self.models_dir):
                return []
            model_names: list[str] = []
            for file in os.listdir(self.models_dir):
                if any(file.endswith(fmt) for fmt in supported_formats):
                    model_names.append(os.path.splitext(file)[0])
            return sorted(set(model_names))
        except Exception as e:
            self.logger.error(error(f"Failed to list available models: {e}"))
            return []

    async def get_prediction(self, model: Any, data: Any) -> dict[str, Any]:
        """Run prediction on a loaded model with best-effort handling across common types."""
        try:
            # scikit-learn style
            if hasattr(model, "predict"):
                y_pred = model.predict(data)
                # Some models provide predict_proba for classification
                proba = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(data)
                    except Exception:
                        proba = None
                return {"predictions": y_pred, "probabilities": proba}

            # XGBoost Booster
            try:
                import xgboost as xgb  # type: ignore
                if isinstance(model, xgb.Booster):
                    dmatrix = xgb.DMatrix(data)
                    y_pred = model.predict(dmatrix)
                    return {"predictions": y_pred}
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                self.logger.warning(f"Failed to predict with XGBoost model: {e}")

            # LightGBM Booster
            try:
                import lightgbm as lgb  # type: ignore
                if isinstance(model, lgb.Booster):
                    y_pred = model.predict(data)
                    return {"predictions": y_pred}
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                self.logger.warning(f"Failed to predict with LightGBM model: {e}")

            # PyTorch models require a wrapper; we cannot infer here
            self.logger.warning(warn_symbol("Model type not directly supported for prediction"))
            return {"error": "unsupported_model_type"}
        except Exception as e:
            self.logger.error(error(f"Prediction failed: {e}"))
            return {"error": str(e)}

    @handles_errors(
        exceptions=(Exception,),
        default_return=[],
        context="available models listing",
    )
    async def list_available_models(self) -> list[str]:
        """List available models in the models directory by supported formats."""
        supported_formats: list[str] = self.model_config.get(
            "supported_formats",
            [".joblib", ".pkl", ".h5"],
        )
        names: set[str] = set()
        if os.path.isdir(self.models_dir):
            for file in os.listdir(self.models_dir):
                name, ext = os.path.splitext(file)
                if ext in supported_formats:
                    names.add(name)
        return sorted(names)

    @handles_errors(
        exceptions=(Exception,),
        default_return={"status": "error"},
        context="generic prediction",
    )
    async def get_prediction(self, model: Any, data: Any) -> dict[str, Any]:
        """
        Generic prediction helper that attempts to call the model's predict method.
        Returns a standardized dictionary structure.
        """
        prediction_result: Any = None
        if model is None:
            return {"status": "error", "error": "model_is_none"}
        # Try common predict methods
        if hasattr(model, "predict"):
            try:
                prediction_result = model.predict(data)
            except Exception as e:
                self.logger.warning(f"Model predict failed: {e}")
        elif hasattr(model, "__call__"):
            try:
                prediction_result = model(data)
            except Exception as e:
                self.logger.warning(f"Model call failed: {e}")
        # Normalize output
        try:
            if hasattr(prediction_result, "tolist"):
                normalized = prediction_result.tolist()
            else:
                normalized = prediction_result
        except Exception:
            normalized = None
        return {
            "status": "ok" if prediction_result is not None else "unknown",
            "predictions": normalized,
            "timestamp": datetime.now().isoformat(),
        }

    # ===== Training-oriented persistence utilities (ML Common compatibility) =====

    def save_models(
        self,
        models: dict[str, Any],
        model_type: str,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
        regime: int | None = None,
    ) -> list[str]:
        """Save multiple models under a structured directory layout."""
        base_dir = self._save_base_path
        model_type = str(model_type)
        # Determine directory
        model_dir = os.path.join(base_dir, model_type, f"regime_{regime}" if regime is not None else "")
        os.makedirs(model_dir, exist_ok=True)

        saved_paths: list[str] = []
        for model_name, model in models.items():
            parts = [str(model_type), str(model_name)]
            if symbol:
                parts.append(str(symbol))
            if exchange:
                parts.append(str(exchange))
            if timeframe:
                parts.append(str(timeframe))
            filename = "_".join(parts) + f".{self.save_format}"
            model_path = os.path.join(model_dir, filename)
            try:
                if self.save_format == "joblib":
                    joblib.dump(model, model_path)
                elif self.save_format == "pickle":
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)
                elif self.save_format == "h5":
                    if hasattr(model, "save"):
                        model.save(model_path)
                    else:
                        # Fallback
                        joblib.dump(model, os.path.splitext(model_path)[0] + ".joblib")
                else:
                    raise ValueError(f"Unsupported save format: {self.save_format}")
                saved_paths.append(model_path)
                self.logger.debug(f"Saved {model_name} to {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to save {model_name}: {e}")
        if saved_paths:
            self.logger.info(f"Saved {len(saved_paths)} models to {model_dir}")
        return saved_paths

    def load_models(
        self,
        model_type: str,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
        regime: int | None = None,
    ) -> dict[str, Any]:
        """Load models for a given type and optional regime."""
        base_dir = self._save_base_path
        model_type = str(model_type)
        model_dir = os.path.join(base_dir, model_type, f"regime_{regime}" if regime is not None else "")
        if not os.path.isdir(model_dir):
            self.logger.warning(f"Model directory not found: {model_dir}")
            return {}
        loaded: dict[str, Any] = {}
        for file in os.listdir(model_dir):
            if not file.endswith(self.save_format):
                continue
            model_path = os.path.join(model_dir, file)
            try:
                if self.save_format == "joblib":
                    model = joblib.load(model_path)
                elif self.save_format == "pickle":
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                elif self.save_format == "h5":
                    model = h5py.File(model_path, "r")
                else:
                    raise ValueError(f"Unsupported save format: {self.save_format}")
                loaded[os.path.splitext(file)[0]] = model
                self.logger.debug(f"Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load {file}: {e}")
        self.logger.info(f"Loaded {len(loaded)} models from {model_dir}")
        return loaded

    def save_metadata(
        self,
        metadata: dict[str, Any],
        model_type: str,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
        regime: int | None = None,
    ) -> str:
        """Save metadata JSON next to models for a given type and regime."""
        base_dir = self._save_base_path
        model_type = str(model_type)
        model_dir = os.path.join(base_dir, model_type, f"regime_{regime}" if regime is not None else "")
        os.makedirs(model_dir, exist_ok=True)
        parts = [model_type, "metadata"]
        if symbol:
            parts.append(symbol)
        if exchange:
            parts.append(exchange)
        if timeframe:
            parts.append(timeframe)
        metadata_path = os.path.join(model_dir, "_".join(parts) + ".json")
        payload = dict(metadata or {})
        payload["saved_at"] = datetime.now().isoformat()
        with open(metadata_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        self.logger.info(f"Saved metadata to {metadata_path}")
        return metadata_path

    def load_metadata(
        self,
        model_type: str,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
        regime: int | None = None,
    ) -> dict[str, Any] | None:
        """Load metadata JSON for a given type and regime if present."""
        base_dir = self._save_base_path
        model_type = str(model_type)
        model_dir = os.path.join(base_dir, model_type, f"regime_{regime}" if regime is not None else "")
        parts = [model_type, "metadata"]
        if symbol:
            parts.append(symbol)
        if exchange:
            parts.append(exchange)
        if timeframe:
            parts.append(timeframe)
        metadata_path = os.path.join(model_dir, "_".join(parts) + ".json")
        if not os.path.exists(metadata_path):
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None
        try:
            with open(metadata_path) as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

    def get_model_metadata(
        self,
        model: Any,
        model_name: str,
        training_time: float = 0.0,
        optimization_time: float = 0.0,
        samples: int = 0,
        features: int = 0,
    ) -> dict[str, Any]:
        """Extract common metadata fields from a model instance."""
        metadata: dict[str, Any] = {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "training_time": float(training_time),
            "optimization_time": float(optimization_time),
            "samples": int(samples),
            "features": int(features),
            "created_at": datetime.now().isoformat(),
        }
        if hasattr(model, "get_params"):
            try:
                metadata["model_params"] = model.get_params()
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                self.logger.warning(f"Failed to get model parameters: {e}")
        if hasattr(model, "feature_importances_"):
            try:
                metadata["feature_importances"] = getattr(model, "feature_importances_").tolist()  # type: ignore[no-any-return]
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                self.logger.warning(f"Failed to get feature importances: {e}")
        if hasattr(model, "n_features_in_"):
            try:
                metadata["n_features_in"] = int(getattr(model, "n_features_in_"))
            except Exception as e:
                # Log the exception for debugging but don't fail the operation
                self.logger.warning(f"Failed to get n_features_in: {e}")
        return metadata

    def cleanup_old_models(
        self,
        model_type: str,
        keep_latest: int = 5,
        symbol: str | None = None,
        exchange: str | None = None,
        timeframe: str | None = None,
    ) -> int:
        """Delete older model files, keeping only the latest modified ones."""
        base_dir = self._save_base_path
        model_dir = os.path.join(base_dir, model_type)
        if not os.path.isdir(model_dir):
            return 0
        # Build pattern components
        def match(file: str) -> bool:
            if not file.endswith(self.save_format):
                return False
            if symbol and symbol not in file:
                return False
            if exchange and exchange not in file:
                return False
            if timeframe and timeframe not in file:
                return False
            return True
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if match(f)]
        if len(files) <= keep_latest:
            return 0
        files.sort(key=lambda p: os.stat(p).st_mtime, reverse=True)
        deleted = 0
        for path in files[keep_latest:]:
            try:
                os.unlink(path)
                deleted += 1
                self.logger.debug(f"Deleted old model: {path}")
            except Exception as e:
                self.logger.warning(f"Failed to delete {path}: {e}")
        if deleted:
            self.logger.info(f"Cleaned up {deleted} old model files")
        return deleted

    @handles_errors(
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
        if not model_name:
            self.logger.error(invalid("Invalid model name"))
            return False

        # Determine file extension
        if format == "joblib":
            extension = ".joblib"
        elif format == "pickle":
            extension = ".pkl"
        elif format == "h5":
            extension = ".h5"
        else:
            self.logger.error(error(f"Unsupported format: {format}"))
            return False

        # Create model path
        model_path = os.path.join(self.models_dir, f"{model_name}{extension}")
        os.makedirs(self.models_dir, exist_ok=True)

        # Save model
        if format == "joblib":
            joblib.dump(model, model_path)
        elif format == "pickle":
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        elif format == "h5":
            with h5py.File(model_path, "w") as f:
                # This is a simplified example - actual implementation depends on model type
                f.create_dataset("model", data=str(model))

        # Register model
        await self.register_model(model_name, model_path)

        self.logger.info(f"Model {model_name} saved successfully")
        return True

    @handles_errors(
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
        if model_name not in self.models:
            self.logger.error(missing(f"Model {model_name} not found"))
            return False

        self.active_model = model_name
        self.model_metadata["active_model"] = model_name
        self.model_metadata["last_updated"] = format_datetime(get_current_datetime(), "%Y-%m-%dT%H:%M:%S")

        # Save metadata
        await self._save_metadata()

        self.logger.info(f"Active model set to: {model_name}")
        return True

    @handles_errors(
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
        return self.active_model

    @handles_errors(
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

    @handles_errors(
        default_return=None,
        context="model backup creation",
    )
    async def create_backup(self, model_name: str) -> None:
        """
        Create backup of a model.

        Args:
            model_name: Name of the model to backup
        """
        if model_name not in self.models:
            self.logger.error(missing(f"Model {model_name} not found"))
            return

        model_path = self.models[model_name]["path"]
        if not os.path.exists(model_path):
            self.logger.error(missing(f"Model file not found: {model_path}"))
            return

        # Create backup directory
        backup_dir = os.path.join(self.models_dir, "backups")
        ensure_directory(backup_dir)
        timestamp = format_datetime(get_current_datetime(), "%Y%m%d_%H%M%S")
        backup_path = os.path.join(
            backup_dir,
            f"{model_name}_backup_{timestamp}{os.path.splitext(model_path)[1]}"
        )

        # Copy model file
        shutil.copy2(model_path, backup_path)

        self.logger.info(f"Model backup created: {backup_path}")

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

    @handles_errors(
        exceptions=(Exception,),
        default_return=None,
        context="model manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the model manager."""
        self.logger.info("🛑 Stopping Model Manager...")

        # Save final metadata
        await self._save_metadata()

        self.logger.info("✅ Model Manager stopped successfully")

# Global model manager instance
model_manager: ModelManager | None = None
