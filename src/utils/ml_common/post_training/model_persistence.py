"""
Model Persistence Component

This module provides comprehensive model persistence capabilities including
model saving, loading, versioning, and metadata tracking.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import gc
import psutil
from pathlib import Path
import joblib
import pickle
import json

# Common utilities
from src.utils.common_operations import (
    safe_json_dump, safe_json_load, safe_file_exists, ensure_directory,
    safe_mean, safe_std, safe_float, safe_int, get_current_datetime,
    safe_append, safe_extend, safe_dict_get, safe_lower, safe_upper,
    format_datetime, validate_file_path, get_file_size, check_disk_space
)
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
    validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils, ParquetUtils
from src.core.decorators import (
    handles_errors, validates, traced, log_execution_time,
    timeout, error_boundary, compose, validate_data_quality,
    monitor_step_execution, ensure_data_integrity, validate_pipeline_step
)
from src.utils.intensity_scaler import (
    get_intensity_from_environment, get_scaled_hpo_trials,
    get_scaled_hpo_timeout, log_intensity_info
)
from src.core.errors import (
    ValidationError, DataIntegrityError, FileOperationError,
    ConfigurationError, ModelTrainingError
)
from src.utils.logger import system_logger

@dataclass
class ModelMetadata:
    """Metadata for a saved model."""

    # Model identification
    model_name: str
    model_type: str
    version: str

    # Training information
    training_timestamp: str
    training_duration: float
    training_data_size: int
    feature_count: int

    # Performance metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    r2_score: Optional[float] = None
    sharpe_ratio: Optional[float] = None

    # Model configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)

    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    author: str = ""

    # File information
    file_size: Optional[int] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None

@dataclass
class PersistenceConfig:
    """Configuration for model persistence."""

    # Storage settings
    base_model_dir: str = "models"
    enable_versioning: bool = True
    max_versions: int = 10

    # Serialization settings
    serialization_format: str = "joblib"  # "joblib", "pickle", "json"
    compression: bool = True
    compression_level: int = 3

    # Metadata settings
    save_metadata: bool = True
    metadata_format: str = "json"  # "json", "yaml"

    # Backup settings
    enable_backup: bool = True
    backup_count: int = 3

    # Validation settings
    validate_on_save: bool = True
    validate_on_load: bool = True

    # Output settings
    save_persistence_log: bool = True
    persistence_log_path: Optional[str] = None

@dataclass
class PersistenceResult:
    """Result of model persistence operation."""

    # Operation status
    success: bool = False
    operation: str = ""  # "save", "load", "delete"

    # File information
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None

    # Model information
    model_name: str = ""
    model_type: str = ""
    version: str = ""

    # Metadata
    metadata: Optional[ModelMetadata] = None

    # Operation details
    operation_time: float = 0.0
    operation_timestamp: str = ""
    error_message: Optional[str] = None

class ModelPersistence:
    """Comprehensive model persistence manager with versioning and metadata tracking."""

    def __init__(self, config: PersistenceConfig):
        """Initialize the model persistence manager.

        Args:
            config: Persistence configuration
        """
        self.config = config
        self.logger = system_logger.getChild('ModelPersistence')

        # Ensure base directory exists
        ensure_directory(Path(self.config.base_model_dir))

        # Apply intensity scaling
        intensity_pct = get_intensity_from_environment()
        if intensity_pct < 1.0:
            self.config = self._apply_intensity_scaling(intensity_pct)
            self.logger.info(f"🔧 Applied intensity scaling ({intensity_pct*100:.0f}%) to persistence config")

    def _apply_intensity_scaling(self, intensity_pct: float) -> PersistenceConfig:
        """Apply intensity scaling to the configuration."""
        return PersistenceConfig(
            base_model_dir=self.config.base_model_dir,
            enable_versioning=self.config.enable_versioning,
            max_versions=max(3, int(self.config.max_versions * intensity_pct)),
            serialization_format=self.config.serialization_format,
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            save_metadata=self.config.save_metadata,
            metadata_format=self.config.metadata_format,
            enable_backup=self.config.enable_backup and intensity_pct > 0.5,
            backup_count=max(1, int(self.config.backup_count * intensity_pct)),
            validate_on_save=self.config.validate_on_save,
            validate_on_load=self.config.validate_on_load,
            save_persistence_log=self.config.save_persistence_log,
            persistence_log_path=self.config.persistence_log_path
        )

    @handles_errors(default_return=PersistenceResult(success=False), context='Model saving')
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def save_model(self, model: Any, model_name: str, model_type: str,
                        metadata: Optional[ModelMetadata] = None,
                        version: Optional[str] = None) -> PersistenceResult:
        """Save a trained model with metadata and versioning.

        Args:
            model: Trained model to save
            model_name: Name of the model
            model_type: Type of the model
            metadata: Model metadata
            version: Model version (auto-generated if None)

        Returns:
            PersistenceResult with save operation details
        """
        try:
            self.logger.info(f"💾 Saving model: {model_name}")
            start_time = time.time()

            # Generate version if not provided
            if version is None:
                version = self._generate_version(model_name)

            # Create model directory
            model_dir = Path(self.config.base_model_dir) / model_name / version
            ensure_directory(model_dir)

            # Prepare metadata
            if metadata is None:
                metadata = ModelMetadata(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    training_timestamp=get_current_datetime(),
                    training_duration=0.0,
                    training_data_size=0,
                    feature_count=0
                )
            else:
                metadata.model_name = model_name
                metadata.model_type = model_type
                metadata.version = version

            # Save model
            model_path = model_dir / f"{model_name}_{version}.{self._get_file_extension()}"

            if self.config.serialization_format == "joblib":
                if self.config.compression:
                    joblib.dump(model, model_path, compress=self.config.compression_level)
                else:
                    joblib.dump(model, model_path)
            elif self.config.serialization_format == "pickle":
                with open(model_path, 'wb') as f:
                    if self.config.compression:
                        import gzip
                        with gzip.GzipFile(fileobj=f, compresslevel=self.config.compression_level) as gz:
                            pickle.dump(model, gz)
                    else:
                        pickle.dump(model, f)
            else:
                raise ValueError(f"Unsupported serialization format: {self.config.serialization_format}")

            # Calculate file information
            file_size = get_file_size(str(model_path))
            checksum = self._calculate_checksum(str(model_path))

            # Update metadata
            metadata.file_path = str(model_path)
            metadata.file_size = file_size
            metadata.checksum = checksum

            # Save metadata
            if self.config.save_metadata:
                await self._save_metadata(metadata, model_dir)

            # Validate saved model
            if self.config.validate_on_save:
                validation_result = await self._validate_saved_model(str(model_path))
                if not validation_result:
                    self.logger.warning("⚠️ Model validation failed after saving")

            # Create backup if enabled
            if self.config.enable_backup:
                await self._create_backup(str(model_path))

            # Clean up old versions
            if self.config.enable_versioning:
                await self._cleanup_old_versions(model_name)

            # Create result
            result = PersistenceResult(
                success=True,
                operation="save",
                file_path=str(model_path),
                file_size=file_size,
                checksum=checksum,
                model_name=model_name,
                model_type=model_type,
                version=version,
                metadata=metadata,
                operation_time=time.time() - start_time,
                operation_timestamp=get_current_datetime()
            )

            # Log operation
            if self.config.save_persistence_log:
                await self._log_persistence_operation(result)

            self.logger.info(f"✅ Model saved successfully: {model_path}")
            return result

        except Exception as e:
            self.logger.exception(f"💥 Error saving model: {e}")
            return PersistenceResult(
                success=False,
                operation="save",
                model_name=model_name,
                model_type=model_type,
                operation_time=time.time() - start_time,
                operation_timestamp=get_current_datetime(),
                error_message=str(e)
            )

    @handles_errors(default_return=PersistenceResult(success=False), context='Model loading')
    # @log_execution_time  # Temporarily disabled due to import conflicts
    async def load_model(self, model_name: str, version: Optional[str] = None) -> PersistenceResult:
        """Load a saved model with metadata.

        Args:
            model_name: Name of the model to load
            version: Model version (loads latest if None)

        Returns:
            PersistenceResult with loaded model and metadata
        """
        try:
            self.logger.info(f"📂 Loading model: {model_name}")
            start_time = time.time()

            # Find model file
            if version is None:
                version = self._get_latest_version(model_name)

            if version is None:
                raise FileNotFoundError(f"No versions found for model: {model_name}")

            model_dir = Path(self.config.base_model_dir) / model_name / version
            model_path = model_dir / f"{model_name}_{version}.{self._get_file_extension()}"

            if not safe_file_exists(str(model_path)):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model
            if self.config.serialization_format == "joblib":
                model = joblib.load(model_path)
            elif self.config.serialization_format == "pickle":
                with open(model_path, 'rb') as f:
                    if self.config.compression:
                        with gzip.GzipFile(fileobj=f) as gz:
                            model = pickle.load(gz)
                    else:
                        model = pickle.load(f)
            else:
                raise ValueError(f"Unsupported serialization format: {self.config.serialization_format}")

            # Load metadata
            metadata = None
            if self.config.save_metadata:
                metadata = await self._load_metadata(model_dir)

            # Validate loaded model
            if self.config.validate_on_load:
                validation_result = await self._validate_loaded_model(model)
                if not validation_result:
                    self.logger.warning("⚠️ Model validation failed after loading")

            # Calculate file information
            file_size = get_file_size(str(model_path))
            checksum = self._calculate_checksum(str(model_path))

            # Create result
            result = PersistenceResult(
                success=True,
                operation="load",
                file_path=str(model_path),
                file_size=file_size,
                checksum=checksum,
                model_name=model_name,
                model_type=metadata.model_type if metadata else "unknown",
                version=version,
                metadata=metadata,
                operation_time=time.time() - start_time,
                operation_timestamp=get_current_datetime()
            )

            # Log operation
            if self.config.save_persistence_log:
                await self._log_persistence_operation(result)

            self.logger.info(f"✅ Model loaded successfully: {model_path}")
            return result

        except Exception as e:
            self.logger.exception(f"💥 Error loading model: {e}")
            return PersistenceResult(
                success=False,
                operation="load",
                model_name=model_name,
                version=version or "unknown",
                operation_time=time.time() - start_time,
                operation_timestamp=get_current_datetime(),
                error_message=str(e)
            )

    def _generate_version(self, model_name: str) -> str:
        """Generate a new version for the model."""
        try:
            if not self.config.enable_versioning:
                return "v1.0.0"

            # Get existing versions
            model_dir = Path(self.config.base_model_dir) / model_name
            if not model_dir.exists():
                return "v1.0.0"

            versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]

            if not versions:
                return "v1.0.0"

            # Parse versions and increment
            version_numbers = []
            for version in versions:
                try:
                    # Extract version number (e.g., "v1.2.3" -> [1, 2, 3])
                    version_parts = version[1:].split('.')
                    if len(version_parts) == 3:
                        version_numbers.append([int(x) for x in version_parts])
                except:
                    continue

            if not version_numbers:
                return "v1.0.0"

            # Find the latest version and increment
            latest_version = max(version_numbers)
            latest_version[2] += 1  # Increment patch version

            return f"v{latest_version[0]}.{latest_version[1]}.{latest_version[2]}"

        except Exception as e:
            self.logger.warning(f"⚠️ Error generating version: {e}")
            return f"v1.0.0_{int(time.time())}"

    def _get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        try:
            model_dir = Path(self.config.base_model_dir) / model_name
            if not model_dir.exists():
                return None

            versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]

            if not versions:
                return None

            # Parse versions and find the latest
            version_numbers = []
            for version in versions:
                try:
                    version_parts = version[1:].split('.')
                    if len(version_parts) == 3:
                        version_numbers.append((version, [int(x) for x in version_parts]))
                except:
                    continue

            if not version_numbers:
                return versions[0]  # Return first version if parsing fails

            # Return the latest version
            latest_version = max(version_numbers, key=lambda x: x[1])
            return latest_version[0]

        except Exception as e:
            self.logger.warning(f"⚠️ Error getting latest version: {e}")
            return None

    def _get_file_extension(self) -> str:
        """Get file extension based on serialization format."""
        if self.config.serialization_format == "joblib":
            return "joblib"
        elif self.config.serialization_format == "pickle":
            return "pkl"
        else:
            return "bin"

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate checksum for a file."""
        try:
            import hashlib

            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating checksum: {e}")
            return ""

    @handles_errors(default_return=None, context='Metadata saving')
    async def _save_metadata(self, metadata: ModelMetadata, model_dir: Path):
        """Save model metadata."""
        try:
            metadata_path = model_dir / f"metadata.{self.config.metadata_format}"

            if self.config.metadata_format == "json":
                safe_json_dump(metadata.__dict__, str(metadata_path))
            else:
                # Default to JSON
                safe_json_dump(metadata.__dict__, str(metadata_path))

            self.logger.info(f"💾 Metadata saved: {metadata_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error saving metadata: {e}")

    @handles_errors(default_return=None, context='Metadata loading')
    async def _load_metadata(self, model_dir: Path) -> Optional[ModelMetadata]:
        """Load model metadata."""
        try:
            metadata_path = model_dir / f"metadata.{self.config.metadata_format}"

            if not safe_file_exists(str(metadata_path)):
                return None

            metadata_dict = safe_json_load(str(metadata_path))
            if metadata_dict:
                return ModelMetadata(**metadata_dict)

            return None

        except Exception as e:
            self.logger.exception(f"💥 Error loading metadata: {e}")
            return None

    @handles_errors(default_return=False, context='Model validation')
    async def _validate_saved_model(self, model_path: str) -> bool:
        """Validate a saved model."""
        try:
            # Try to load the model
            if self.config.serialization_format == "joblib":
                model = joblib.load(model_path)
            elif self.config.serialization_format == "pickle":
                with open(model_path, 'rb') as f:
                    if self.config.compression:
                        with gzip.GzipFile(fileobj=f) as gz:
                            model = pickle.load(gz)
                    else:
                        model = pickle.load(f)
            else:
                return False

            # Basic validation - check if model has required methods
            if hasattr(model, 'predict'):
                return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ Model validation failed: {e}")
            return False

    @handles_errors(default_return=False, context='Model validation')
    async def _validate_loaded_model(self, model: Any) -> bool:
        """Validate a loaded model."""
        try:
            # Basic validation - check if model has required methods
            if hasattr(model, 'predict'):
                return True

            return False

        except Exception as e:
            self.logger.warning(f"⚠️ Model validation failed: {e}")
            return False

    @handles_errors(default_return=None, context='Backup creation')
    async def _create_backup(self, model_path: str):
        """Create backup of the model."""
        try:
            import shutil

            backup_dir = Path(model_path).parent / "backups"
            ensure_directory(backup_dir)

            timestamp = get_current_datetime().replace(":", "-").replace(" ", "_")
            backup_path = backup_dir / f"{Path(model_path).stem}_backup_{timestamp}{Path(model_path).suffix}"

            shutil.copy2(model_path, backup_path)

            # Clean up old backups
            backups = list(backup_dir.glob("*_backup_*"))
            if len(backups) > self.config.backup_count:
                backups.sort(key=lambda x: x.stat().st_mtime)
                for old_backup in backups[:-self.config.backup_count]:
                    old_backup.unlink()

            self.logger.info(f"💾 Backup created: {backup_path}")

        except Exception as e:
            self.logger.exception(f"💥 Error creating backup: {e}")

    @handles_errors(default_return=None, context='Version cleanup')
    async def _cleanup_old_versions(self, model_name: str):
        """Clean up old versions of the model."""
        try:
            model_dir = Path(self.config.base_model_dir) / model_name
            if not model_dir.exists():
                return

            versions = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]

            if len(versions) <= self.config.max_versions:
                return

            # Sort versions by modification time
            versions.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove old versions
            for old_version in versions[self.config.max_versions:]:
                shutil.rmtree(old_version)
                self.logger.info(f"🗑️ Removed old version: {old_version}")

        except Exception as e:
            self.logger.exception(f"💥 Error cleaning up old versions: {e}")

    @handles_errors(default_return=None, context='Persistence logging')
    async def _log_persistence_operation(self, result: PersistenceResult):
        """Log persistence operation."""
        try:
            log_path = self.config.persistence_log_path or f"data_cache/persistence_log_{get_current_datetime()}.json"
            ensure_directory(Path(log_path).parent)

            log_entry = {
                'timestamp': result.operation_timestamp,
                'operation': result.operation,
                'model_name': result.model_name,
                'model_type': result.model_type,
                'version': result.version,
                'success': result.success,
                'operation_time': result.operation_time,
                'file_path': result.file_path,
                'file_size': result.file_size,
                'checksum': result.checksum,
                'error_message': result.error_message
            }

            # Load existing log or create new one
            if safe_file_exists(log_path):
                log_data = safe_json_load(log_path) or []
            else:
                log_data = []

            log_data.append(log_entry)

            # Keep only recent entries (last 1000)
            if len(log_data) > 1000:
                log_data = log_data[-1000:]

            safe_json_dump(log_data, log_path)

        except Exception as e:
            self.logger.exception(f"💥 Error logging persistence operation: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        try:
            models = []
            base_dir = Path(self.config.base_model_dir)

            if not base_dir.exists():
                return models

            for model_dir in base_dir.iterdir():
                if model_dir.is_dir():
                    model_info = {
                        'model_name': model_dir.name,
                        'versions': [],
                        'latest_version': None
                    }

                    # Get versions
                    versions = [d.name for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
                    model_info['versions'] = versions

                    # Get latest version
                    if versions:
                        model_info['latest_version'] = self._get_latest_version(model_dir.name)

                    models.append(model_info)

            return models

        except Exception as e:
            self.logger.exception(f"💥 Error listing models: {e}")
            return []

    async def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        try:
            if version is None:
                version = self._get_latest_version(model_name)

            if version is None:
                return None

            model_dir = Path(self.config.base_model_dir) / model_name / version

            # Load metadata
            metadata = None
            if self.config.save_metadata:
                metadata = await self._load_metadata(model_dir)

            # Get file information
            model_path = model_dir / f"{model_name}_{version}.{self._get_file_extension()}"
            file_size = get_file_size(str(model_path)) if safe_file_exists(str(model_path)) else None

            return {
                'model_name': model_name,
                'version': version,
                'model_dir': str(model_dir),
                'model_path': str(model_path),
                'file_size': file_size,
                'metadata': metadata.__dict__ if metadata else None,
                'exists': safe_file_exists(str(model_path))
            }

        except Exception as e:
            self.logger.exception(f"💥 Error getting model info: {e}")
            return None
