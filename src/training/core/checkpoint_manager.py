"""Checkpoint manager for the modular training pipeline.

This module provides checkpointing functionality for pipeline stages, allowing for resuming from failures and maintaining state across
pipeline executions.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.error_handler import (
    handle_errors, handle_specific_errors,
)
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    error, execution_error, failed,
    initialization_error, invalid, validation_error
)


class CheckpointManager:
    """Checkpoint manager with comprehensive error handling and type safety."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the checkpoint manager."""
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("CheckpointManager")

        # Checkpoint manager state
        self.is_managing: bool = False
        self.checkpoint_results: dict[str, Any] = {}
        self.checkpoint_history: list[dict[str, Any]] = []

        # Configuration
        self.checkpoint_config: dict[str, Any] = self.config.get(
            "checkpoint_manager",
            {},
        )
        self.checkpoint_interval: int = self.checkpoint_config.get(
            "checkpoint_interval",
            3600
        )
        self.max_checkpoint_history: int = self.checkpoint_config.get(
            "max_checkpoint_history", 100,
        )
        self.enable_checkpoint_saving: bool = self.checkpoint_config.get(
            "enable_checkpoint_saving",
            True
        )
        self.enable_checkpoint_loading: bool = self.checkpoint_config.get(
            "enable_checkpoint_loading", True,
        )

        # Initialize checkpoint storage
        self.checkpoint_dir: str = self.checkpoint_config.get(
            "checkpoint_directory",
            "./checkpoints"
        )
        self._ensure_checkpoint_directory()

    def _ensure_checkpoint_directory(self) -> None:
        """Ensure checkpoint directory exists."""
        try:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Checkpoint directory ensured: {self.checkpoint_dir}")
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint directory: {e}")
            raise

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid checkpoint manager configuration"),
            AttributeError: (False, "Missing required checkpoint manager parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False, context="checkpoint manager initialization"
    )
    async def initialize(self) -> bool:
        """Initialize the checkpoint manager."""
        self.logger.info("Initializing Checkpoint Manager...")

        # Load checkpoint configuration
        await self._load_checkpoint_configuration()

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for checkpoint manager"))
            return False

        # Initialize checkpoint modules
        await self._initialize_checkpoint_modules()

        self.logger.info(
            "✅ Checkpoint Manager initialization completed successfully",
        )
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint configuration loading",
    )
    async def _load_checkpoint_configuration(self) -> None:
        """Load checkpoint configuration."""
        # Set default checkpoint parameters
        self.checkpoint_config.setdefault("checkpoint_interval", 3600)
        self.checkpoint_config.setdefault("max_checkpoint_history", 100)
        self.checkpoint_config.setdefault("enable_checkpoint_saving", True)
        self.checkpoint_config.setdefault("enable_checkpoint_loading", True)
        self.checkpoint_config.setdefault("enable_checkpoint_validation", True)
        self.checkpoint_config.setdefault("enable_checkpoint_cleanup", True)
        self.checkpoint_config.setdefault("checkpoint_directory", "./checkpoints")

        # Update configuration
        self.checkpoint_interval = self.checkpoint_config["checkpoint_interval"]
        self.max_checkpoint_history = self.checkpoint_config["max_checkpoint_history"]
        self.enable_checkpoint_saving = self.checkpoint_config["enable_checkpoint_saving"]
        self.enable_checkpoint_loading = self.checkpoint_config["enable_checkpoint_loading"]

        self.logger.info("Checkpoint configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate checkpoint manager configuration."""
        # Validate checkpoint interval
        if self.checkpoint_interval <= 0:
            self.logger.error(invalid("Invalid checkpoint interval"))
            return False

        # Validate max checkpoint history
        if self.max_checkpoint_history <= 0:
            self.logger.error(invalid("Invalid max checkpoint history"))
            return False

        # Validate that at least one checkpoint type is enabled
        if not any([
            self.enable_checkpoint_saving,
            self.enable_checkpoint_loading,
            self.checkpoint_config.get("enable_checkpoint_validation", True),
            self.checkpoint_config.get("enable_checkpoint_cleanup", True),
        ]):
            self.logger.error(error("At least one checkpoint type must be enabled"))
            return False

        self.logger.info("Configuration validation successful")
        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint modules initialization",
    )
    async def _initialize_checkpoint_modules(self) -> None:
        """Initialize checkpoint modules."""
        # Initialize checkpoint saving module
        if self.enable_checkpoint_saving:
            await self._initialize_checkpoint_saving()

        # Initialize checkpoint loading module
        if self.enable_checkpoint_loading:
            await self._initialize_checkpoint_loading()

        # Initialize checkpoint validation module
        if self.checkpoint_config.get("enable_checkpoint_validation", True):
            await self._initialize_checkpoint_validation()

        # Initialize checkpoint cleanup module
        if self.checkpoint_config.get("enable_checkpoint_cleanup", True):
            await self._initialize_checkpoint_cleanup()

        self.logger.info("Checkpoint modules initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint saving initialization",
    )
    async def _initialize_checkpoint_saving(self) -> None:
        """Initialize checkpoint saving module."""
        # Initialize checkpoint saving components
        self.checkpoint_saving_components = {
            "checkpoint_creation": True,
            "checkpoint_serialization": True,
            "checkpoint_storage": True,
            "checkpoint_metadata": True,
        }

        self.logger.info("Checkpoint saving module initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint loading initialization",
    )
    async def _initialize_checkpoint_loading(self) -> None:
        """Initialize checkpoint loading module."""
        # Initialize checkpoint loading components
        self.checkpoint_loading_components = {
            "checkpoint_discovery": True,
            "checkpoint_deserialization": True,
            "checkpoint_restoration": True,
            "checkpoint_validation": True,
        }

        self.logger.info("Checkpoint loading module initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint validation initialization",
    )
    async def _initialize_checkpoint_validation(self) -> None:
        """Initialize checkpoint validation module."""
        # Initialize checkpoint validation components
        self.checkpoint_validation_components = {
            "integrity_validation": True,
            "format_validation": True,
            "metadata_validation": True,
            "compatibility_validation": True,
        }

        self.logger.info("Checkpoint validation module initialized")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint cleanup initialization",
    )
    async def _initialize_checkpoint_cleanup(self) -> None:
        """Initialize checkpoint cleanup module."""
        # Initialize checkpoint cleanup components
        self.checkpoint_cleanup_components = {
            "cleanup_scheduling": True,
            "cleanup_execution": True,
            "cleanup_verification": True,
            "cleanup_reporting": True,
        }

        self.logger.info("Checkpoint cleanup module initialized")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid checkpoint parameters"),
            AttributeError: (False, "Missing checkpoint components"),
            KeyError: (False, "Missing required checkpoint data"),
        },
        default_return=False, context="checkpoint execution",
    )
    async def execute_checkpoint(self, checkpoint_input: dict[str, Any]) -> bool:
        """Execute checkpointing operations."""
        if not self._validate_checkpoint_inputs(checkpoint_input):
            return False

        self.is_managing = True
        try:
            self.logger.info("🔄 Starting checkpoint execution...")

            # Perform checkpoint saving
            if self.enable_checkpoint_saving:
                saving_results = await self._perform_checkpoint_saving(checkpoint_input)
                self.checkpoint_results["checkpoint_saving"] = saving_results

            # Perform checkpoint loading
            if self.enable_checkpoint_loading:
                loading_results = await self._perform_checkpoint_loading(checkpoint_input)
                self.checkpoint_results["checkpoint_loading"] = loading_results

            # Perform checkpoint validation
            if self.checkpoint_config.get("enable_checkpoint_validation", True):
                validation_results = await self._perform_checkpoint_validation(checkpoint_input)
                self.checkpoint_results["checkpoint_validation"] = validation_results

            # Perform checkpoint cleanup
            if self.checkpoint_config.get("enable_checkpoint_cleanup", True):
                cleanup_results = await self._perform_checkpoint_cleanup(checkpoint_input)
                self.checkpoint_results["checkpoint_cleanup"] = cleanup_results

            # Store checkpoint results
            await self._store_checkpoint_results()

            self.logger.info("✅ Checkpoint execution completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Critical error during checkpoint execution: {e}")
            raise
        finally:
            self.is_managing = False

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=False,
        context="checkpoint inputs validation",
    )
    def _validate_checkpoint_inputs(self, checkpoint_input: dict[str, Any]) -> bool:
        """Validate checkpoint input parameters."""
        # Check required checkpoint input fields
        required_fields = ["checkpoint_type", "checkpoint_name", "timestamp"]
        for field in required_fields:
            if field not in checkpoint_input:
                self.logger.error(
                    f"Missing required checkpoint input field: {field}",
                )
                return False

        # Validate data types
        if not isinstance(checkpoint_input["checkpoint_type"], str):
            self.logger.error(invalid("Invalid checkpoint type"))
            return False

        if not isinstance(checkpoint_input["checkpoint_name"], str):
            self.logger.error(invalid("Invalid checkpoint name"))
            return False

        return True

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint saving",
    )
    async def _perform_checkpoint_saving(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Perform checkpoint saving operations."""
        results: dict[str, Any] = {}

        # Perform checkpoint creation
        if self.checkpoint_saving_components.get("checkpoint_creation", False):
            results["checkpoint_creation"] = await self._perform_checkpoint_creation(checkpoint_input)

        # Perform checkpoint serialization
        if self.checkpoint_saving_components.get("checkpoint_serialization", False):
            results["checkpoint_serialization"] = await self._perform_checkpoint_serialization(checkpoint_input)

        # Perform checkpoint storage
        if self.checkpoint_saving_components.get("checkpoint_storage", False):
            results["checkpoint_storage"] = await self._perform_checkpoint_storage(checkpoint_input)

        # Perform checkpoint metadata
        if self.checkpoint_saving_components.get("checkpoint_metadata", False):
            results["checkpoint_metadata"] = await self._perform_checkpoint_metadata(checkpoint_input)

        self.logger.info("Checkpoint saving completed")
        return results

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint loading",
    )
    async def _perform_checkpoint_loading(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Perform checkpoint loading operations."""
        results: dict[str, Any] = {}

        # Perform checkpoint discovery
        if self.checkpoint_loading_components.get("checkpoint_discovery", False):
            results["checkpoint_discovery"] = await self._perform_checkpoint_discovery(checkpoint_input)

        # Perform checkpoint deserialization
        if self.checkpoint_loading_components.get("checkpoint_deserialization", False):
            results["checkpoint_deserialization"] = await self._perform_checkpoint_deserialization(checkpoint_input)

        # Perform checkpoint restoration
        if self.checkpoint_loading_components.get("checkpoint_restoration", False):
            results["checkpoint_restoration"] = await self._perform_checkpoint_restoration(checkpoint_input)

        # Perform checkpoint validation
        if self.checkpoint_loading_components.get("checkpoint_validation", False):
            results["checkpoint_validation"] = await self._perform_checkpoint_validation_core(checkpoint_input)

        self.logger.info("Checkpoint loading completed")
        return results

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint validation",
    )
    async def _perform_checkpoint_validation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Perform checkpoint validation operations."""
        results: dict[str, Any] = {}

        # Perform integrity validation
        if self.checkpoint_validation_components.get("integrity_validation", False):
            results["integrity_validation"] = await self._perform_integrity_validation(checkpoint_input)

        # Perform format validation
        if self.checkpoint_validation_components.get("format_validation", False):
            results["format_validation"] = await self._perform_format_validation(checkpoint_input)

        # Perform metadata validation
        if self.checkpoint_validation_components.get("metadata_validation", False):
            results["metadata_validation"] = await self._perform_metadata_validation(checkpoint_input)

        # Perform compatibility validation
        if self.checkpoint_validation_components.get("compatibility_validation", False):
            results["compatibility_validation"] = await self._perform_compatibility_validation(checkpoint_input)

        self.logger.info("Checkpoint validation completed")
        return results

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint cleanup",
    )
    async def _perform_checkpoint_cleanup(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Perform checkpoint cleanup operations."""
        results: dict[str, Any] = {}

        # Perform cleanup scheduling
        if self.checkpoint_cleanup_components.get("cleanup_scheduling", False):
            results["cleanup_scheduling"] = await self._perform_cleanup_scheduling(checkpoint_input)

        # Perform cleanup execution
        if self.checkpoint_cleanup_components.get("cleanup_execution", False):
            results["cleanup_execution"] = await self._perform_cleanup_execution(checkpoint_input)

        # Perform cleanup verification
        if self.checkpoint_cleanup_components.get("cleanup_verification", False):
            results["cleanup_verification"] = await self._perform_cleanup_verification(checkpoint_input)

        # Perform cleanup reporting
        if self.checkpoint_cleanup_components.get("cleanup_reporting", False):
            results["cleanup_reporting"] = await self._perform_cleanup_reporting(checkpoint_input)

        self.logger.info("Checkpoint cleanup completed")
        return results

    # Checkpoint saving methods
    async def _perform_checkpoint_creation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Create a new checkpoint."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            # Create checkpoint data structure
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_input["checkpoint_type"],
                "checkpoint_name": checkpoint_input["checkpoint_name"],
                "timestamp": checkpoint_input["timestamp"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data": checkpoint_input.get("data", {}),
                "metadata": checkpoint_input.get("metadata", {})
            }
            
            return {
                "checkpoint_creation_completed": True,
                "checkpoint_id": checkpoint_id,
                "checkpoint_path": checkpoint_path,
                "checkpoint_data": checkpoint_data,
                "creation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint creation: {e}")
            raise

    async def _perform_checkpoint_serialization(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Serialize checkpoint data."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            # Serialize using pickle for complex objects
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_input["checkpoint_type"],
                "checkpoint_name": checkpoint_input["checkpoint_name"],
                "timestamp": checkpoint_input["timestamp"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data": checkpoint_input.get("data", {}),
                "metadata": checkpoint_input.get("metadata", {})
            }
            
            serialized_data = pickle.dumps(checkpoint_data)
            serialization_size = len(serialized_data)
            
            return {
                "checkpoint_serialization_completed": True,
                "serialization_format": "pickle",
                "serialization_size": f"{serialization_size} bytes",
                "serialization_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint serialization: {e}")
            raise

    async def _perform_checkpoint_storage(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Store checkpoint to disk."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            # Create checkpoint data
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_input["checkpoint_type"],
                "checkpoint_name": checkpoint_input["checkpoint_name"],
                "timestamp": checkpoint_input["timestamp"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "data": checkpoint_input.get("data", {}),
                "metadata": checkpoint_input.get("metadata", {})
            }
            
            # Serialize and store
            serialized_data = pickle.dumps(checkpoint_data)
            with open(checkpoint_path, 'wb') as f:
                f.write(serialized_data)
            
            # Calculate file size
            file_size = os.path.getsize(checkpoint_path)
            
            return {
                "checkpoint_storage_completed": True,
                "storage_location": checkpoint_path,
                "storage_method": "pickle",
                "file_size": f"{file_size} bytes",
                "storage_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint storage: {e}")
            raise

    async def _perform_checkpoint_metadata(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Create and store checkpoint metadata."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.meta")
            
            # Create metadata
            metadata = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_type": checkpoint_input["checkpoint_type"],
                "checkpoint_name": checkpoint_input["checkpoint_name"],
                "timestamp": checkpoint_input["timestamp"],
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "checksum": hashlib.md5(str(checkpoint_input).encode()).hexdigest(),
                "size": len(str(checkpoint_input)),
                "tags": checkpoint_input.get("tags", []),
                "description": checkpoint_input.get("description", ""),
                "author": checkpoint_input.get("author", "system"),
                "dependencies": checkpoint_input.get("dependencies", [])
            }
            
            # Store metadata as JSON
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "checkpoint_metadata_completed": True,
                "metadata_entries": len(metadata),
                "metadata_format": "json",
                "metadata_path": metadata_path,
                "creation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint metadata creation: {e}")
            raise

    # Checkpoint loading methods
    async def _perform_checkpoint_discovery(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Discover available checkpoints."""
        try:
            checkpoint_pattern = f"{checkpoint_input['checkpoint_name']}_*.ckpt"
            checkpoint_files = list(Path(self.checkpoint_dir).glob(checkpoint_pattern))
            
            discovered_checkpoints = []
            for checkpoint_file in checkpoint_files:
                try:
                    # Extract checkpoint info from filename
                    filename = checkpoint_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        checkpoint_name = parts[0]
                        timestamp = '_'.join(parts[1:])
                        discovered_checkpoints.append({
                            "filename": filename,
                            "checkpoint_name": checkpoint_name,
                            "timestamp": timestamp,
                            "file_path": str(checkpoint_file),
                            "file_size": checkpoint_file.stat().st_size,
                            "modified_time": datetime.fromtimestamp(checkpoint_file.stat().st_mtime).isoformat()
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing checkpoint file {checkpoint_file}: {e}")
                    continue
            
            return {
                "checkpoint_discovery_completed": True,
                "checkpoints_found": len(discovered_checkpoints),
                "discovery_method": "pattern_matching",
                "discovered_checkpoints": discovered_checkpoints,
                "discovery_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint discovery: {e}")
            raise

    async def _perform_checkpoint_deserialization(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Deserialize checkpoint data."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Read and deserialize
            with open(checkpoint_path, 'rb') as f:
                serialized_data = f.read()
            
            checkpoint_data = pickle.loads(serialized_data)
            deserialization_size = len(serialized_data)
            
            return {
                "checkpoint_deserialization_completed": True,
                "deserialization_format": "pickle",
                "deserialization_size": f"{deserialization_size} bytes",
                "checkpoint_data": checkpoint_data,
                "deserialization_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint deserialization: {e}")
            raise

    async def _perform_checkpoint_restoration(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Restore checkpoint data."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Read and deserialize
            with open(checkpoint_path, 'rb') as f:
                serialized_data = f.read()
            
            checkpoint_data = pickle.loads(serialized_data)
            
            # Validate restored data
            if not isinstance(checkpoint_data, dict):
                raise ValueError("Invalid checkpoint data format")
            
            required_fields = ["checkpoint_id", "checkpoint_type", "checkpoint_name", "timestamp"]
            for field in required_fields:
                if field not in checkpoint_data:
                    raise ValueError(f"Missing required field in checkpoint data: {field}")
            
            return {
                "checkpoint_restoration_completed": True,
                "restoration_success": True,
                "restored_data": checkpoint_data,
                "restoration_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint restoration: {e}")
            raise

    async def _perform_checkpoint_validation_core(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Core checkpoint validation."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Basic file validation
            file_size = os.path.getsize(checkpoint_path)
            if file_size == 0:
                raise ValueError("Checkpoint file is empty")
            
            # Try to deserialize to validate format
            with open(checkpoint_path, 'rb') as f:
                serialized_data = f.read()
            
            checkpoint_data = pickle.loads(serialized_data)
            
            # Validate data structure
            if not isinstance(checkpoint_data, dict):
                raise ValueError("Invalid checkpoint data format")
            
            return {
                "checkpoint_validation_completed": True,
                "validation_score": 1.0,
                "validation_method": "structure_check",
                "file_size": file_size,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in checkpoint validation: {e}")
            raise

    # Checkpoint validation methods
    async def _perform_integrity_validation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Validate checkpoint integrity."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Calculate checksum
            with open(checkpoint_path, 'rb') as f:
                file_content = f.read()
            
            calculated_checksum = hashlib.md5(file_content).hexdigest()
            
            # Check if metadata file exists for checksum comparison
            metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.meta")
            stored_checksum = None
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    stored_checksum = metadata.get("checksum")
                except Exception as e:
                    self.logger.warning(f"Could not read metadata for checksum validation: {e}")
            
            integrity_score = 1.0 if stored_checksum == calculated_checksum else 0.5
            
            return {
                "integrity_validation_completed": True,
                "integrity_score": integrity_score,
                "validation_method": "checksum_check",
                "calculated_checksum": calculated_checksum,
                "stored_checksum": stored_checksum,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in integrity validation: {e}")
            raise

    async def _perform_format_validation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Validate checkpoint format."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Try to deserialize to validate pickle format
            with open(checkpoint_path, 'rb') as f:
                serialized_data = f.read()
            
            try:
                checkpoint_data = pickle.loads(serialized_data)
                format_valid = True
                format_score = 1.0
            except (pickle.UnpicklingError, EOFError, ValueError) as e:
                format_valid = False
                format_score = 0.0
                self.logger.warning(f"Format validation failed: {e}")
            
            return {
                "format_validation_completed": True,
                "format_score": format_score,
                "validation_method": "pickle_validation",
                "format_valid": format_valid,
                "validation_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in format validation: {e}")
            raise

    async def _perform_metadata_validation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Validate checkpoint metadata."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            metadata_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.meta")
            
            if not os.path.exists(metadata_path):
                return {
                    "metadata_validation_completed": True,
                    "metadata_score": 0.0,
                    "validation_method": "file_existence",
                    "metadata_exists": False,
                    "validation_time": datetime.now().isoformat(),
                }
            
            # Validate metadata format
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check required metadata fields
                required_metadata_fields = ["checkpoint_id", "checkpoint_type", "checkpoint_name", "timestamp"]
                missing_fields = [field for field in required_metadata_fields if field not in metadata]
                
                if missing_fields:
                    metadata_score = 0.5
                else:
                    metadata_score = 1.0
                
                return {
                    "metadata_validation_completed": True,
                    "metadata_score": metadata_score,
                    "validation_method": "json_validation",
                    "metadata_exists": True,
                    "missing_fields": missing_fields,
                    "validation_time": datetime.now().isoformat(),
                }
            except json.JSONDecodeError as e:
                return {
                    "metadata_validation_completed": True,
                    "metadata_score": 0.0,
                    "validation_method": "json_validation",
                    "metadata_exists": True,
                    "json_error": str(e),
                    "validation_time": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(f"Critical error in metadata validation: {e}")
            raise

    async def _perform_compatibility_validation(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Validate checkpoint compatibility."""
        try:
            checkpoint_id = f"{checkpoint_input['checkpoint_name']}_{checkpoint_input['timestamp']}"
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
            
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            # Try to deserialize to check compatibility
            with open(checkpoint_path, 'rb') as f:
                serialized_data = f.read()
            
            try:
                checkpoint_data = pickle.loads(serialized_data)
                
                # Check version compatibility
                version = checkpoint_data.get("version", "unknown")
                current_version = "1.0"
                
                if version == current_version:
                    compatibility_score = 1.0
                else:
                    compatibility_score = 0.7  # Partial compatibility for different versions
                
                return {
                    "compatibility_validation_completed": True,
                    "compatibility_score": compatibility_score,
                    "validation_method": "version_check",
                    "checkpoint_version": version,
                    "current_version": current_version,
                    "validation_time": datetime.now().isoformat(),
                }
            except Exception as e:
                return {
                    "compatibility_validation_completed": True,
                    "compatibility_score": 0.0,
                    "validation_method": "deserialization_check",
                    "compatibility_error": str(e),
                    "validation_time": datetime.now().isoformat(),
                }
        except Exception as e:
            self.logger.error(f"Critical error in compatibility validation: {e}")
            raise

    # Checkpoint cleanup methods
    async def _perform_cleanup_scheduling(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Schedule checkpoint cleanup."""
        try:
            # Find old checkpoints based on age
            current_time = datetime.now()
            max_age_days = self.checkpoint_config.get("max_checkpoint_age_days", 30)
            
            checkpoint_files = list(Path(self.checkpoint_dir).glob("*.ckpt"))
            scheduled_for_cleanup = []
            
            for checkpoint_file in checkpoint_files:
                try:
                    file_age = current_time - datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                    if file_age.days > max_age_days:
                        scheduled_for_cleanup.append({
                            "file_path": str(checkpoint_file),
                            "file_age_days": file_age.days,
                            "reason": "age_based"
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing file {checkpoint_file} for cleanup: {e}")
                    continue
            
            return {
                "cleanup_scheduling_completed": True,
                "scheduled_cleanups": len(scheduled_for_cleanup),
                "scheduling_method": "age_based",
                "max_age_days": max_age_days,
                "scheduled_files": scheduled_for_cleanup,
                "scheduling_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in cleanup scheduling: {e}")
            raise

    async def _perform_cleanup_execution(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Execute checkpoint cleanup."""
        try:
            # Get scheduled cleanups
            cleanup_schedule = await self._perform_cleanup_scheduling(checkpoint_input)
            scheduled_files = cleanup_schedule.get("scheduled_files", [])
            
            cleaned_files = []
            for file_info in scheduled_files:
                try:
                    file_path = file_info["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        
                        # Also remove metadata file if it exists
                        metadata_path = file_path.replace('.ckpt', '.meta')
                        if os.path.exists(metadata_path):
                            os.remove(metadata_path)
                        
                        cleaned_files.append({
                            "file_path": file_path,
                            "cleanup_success": True,
                            "cleanup_time": datetime.now().isoformat()
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to clean up file {file_path}: {e}")
                    cleaned_files.append({
                        "file_path": file_path,
                        "cleanup_success": False,
                        "error": str(e),
                        "cleanup_time": datetime.now().isoformat()
                    })
            
            return {
                "cleanup_execution_completed": True,
                "cleanups_executed": len(cleaned_files),
                "execution_method": "batch",
                "cleaned_files": cleaned_files,
                "execution_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in cleanup execution: {e}")
            raise

    async def _perform_cleanup_verification(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Verify cleanup execution."""
        try:
            # Verify that scheduled files were actually removed
            cleanup_schedule = await self._perform_cleanup_scheduling(checkpoint_input)
            scheduled_files = cleanup_schedule.get("scheduled_files", [])
            
            verification_results = []
            for file_info in scheduled_files:
                file_path = file_info["file_path"]
                file_exists = os.path.exists(file_path)
                
                verification_results.append({
                    "file_path": file_path,
                    "still_exists": file_exists,
                    "verification_success": not file_exists
                })
            
            # Calculate verification score
            total_files = len(verification_results)
            if total_files == 0:
                verification_score = 1.0
            else:
                successful_verifications = sum(1 for r in verification_results if r["verification_success"])
                verification_score = successful_verifications / total_files
            
            return {
                "cleanup_verification_completed": True,
                "verification_score": verification_score,
                "verification_method": "file_existence_check",
                "total_files": total_files,
                "successful_verifications": sum(1 for r in verification_results if r["verification_success"]),
                "verification_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in cleanup verification: {e}")
            raise

    async def _perform_cleanup_reporting(self, checkpoint_input: dict[str, Any]) -> dict[str, Any]:
        """Generate cleanup report."""
        try:
            # Generate comprehensive cleanup report
            cleanup_schedule = await self._perform_cleanup_scheduling(checkpoint_input)
            cleanup_execution = await self._perform_cleanup_execution(checkpoint_input)
            cleanup_verification = await self._perform_cleanup_verification(checkpoint_input)
            
            # Create report
            report = {
                "cleanup_report": {
                    "timestamp": datetime.now().isoformat(),
                    "scheduling": cleanup_schedule,
                    "execution": cleanup_execution,
                    "verification": cleanup_verification,
                    "summary": {
                        "total_scheduled": cleanup_schedule.get("scheduled_cleanups", 0),
                        "total_executed": cleanup_execution.get("cleanups_executed", 0),
                        "verification_score": cleanup_verification.get("verification_score", 0.0)
                    }
                }
            }
            
            # Save report to file
            report_path = os.path.join(self.checkpoint_dir, f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return {
                "cleanup_reporting_completed": True,
                "report_format": "json",
                "report_location": report_path,
                "report_size": len(json.dumps(report)),
                "reporting_time": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Critical error in cleanup reporting: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint results storage",
    )
    async def _store_checkpoint_results(self) -> None:
        """Store checkpoint results."""
        # Add timestamp
        self.checkpoint_results["timestamp"] = datetime.now().isoformat()

        # Add to history
        self.checkpoint_history.append(self.checkpoint_results.copy())

        # Limit history size
        if len(self.checkpoint_history) > self.max_checkpoint_history:
            self.checkpoint_history.pop(0)

        self.logger.info("Checkpoint results stored successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint results getting",
    )
    def get_checkpoint_results(self, checkpoint_type: str | None = None) -> dict[str, Any]:
        """Get checkpoint results."""
        if checkpoint_type:
            return self.checkpoint_results.get(checkpoint_type, {})
        return self.checkpoint_results.copy()

    @handle_errors(
        exceptions=(ValueError, AttributeError), default_return=None,
        context="checkpoint history getting",
    )
    def get_checkpoint_history(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Get checkpoint history."""
        history = self.checkpoint_history.copy()
        if limit:
            history = history[-limit:]
        return history

    def get_checkpoint_status(self) -> dict[str, Any]:
        """Get checkpoint manager status."""
        return {
            "is_managing": self.is_managing,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoint_history": self.max_checkpoint_history,
            "enable_checkpoint_saving": self.enable_checkpoint_saving,
            "enable_checkpoint_loading": self.enable_checkpoint_loading,
            "enable_checkpoint_validation": self.checkpoint_config.get("enable_checkpoint_validation"),
            "enable_checkpoint_cleanup": self.checkpoint_config.get("enable_checkpoint_cleanup"),
            "checkpoint_history_count": len(self.checkpoint_history),
            "checkpoint_directory": self.checkpoint_dir,
        }

    @handle_errors(
        exceptions=(Exception, ), default_return=None,
        context="checkpoint manager cleanup",
    )
    async def stop(self) -> None:
        """Stop the checkpoint manager."""
        self.logger.info("🛑 Stopping Checkpoint Manager...")

        # Stop managing
        self.is_managing = False

        # Clear results
        self.checkpoint_results.clear()

        # Clear history
        self.checkpoint_history.clear()

        self.logger.info("✅ Checkpoint Manager stopped successfully")


# Global checkpoint manager instance
checkpoint_manager: CheckpointManager | None = None


@handle_errors(
    exceptions=(Exception, ), default_return=None,
    context="checkpoint manager setup",
)
async def setup_checkpoint_manager(config: dict[str, Any] | None = None) -> CheckpointManager | None:
    """Setup the global checkpoint manager."""
    try:
        global checkpoint_manager

        if config is None:
            config = {
                "checkpoint_manager": {
                    "checkpoint_interval": 3600,
                    "max_checkpoint_history": 100,
                    "enable_checkpoint_saving": True,
                    "enable_checkpoint_loading": True,
                    "enable_checkpoint_validation": True,
                    "enable_checkpoint_cleanup": True,
                    "checkpoint_directory": "./checkpoints",
                    "max_checkpoint_age_days": 30,
                },
            }

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(config)

        # Initialize checkpoint manager
        success = await checkpoint_manager.initialize()
        if success:
            return checkpoint_manager
        return None

    except Exception as e:
        return None


def _validate_data_quality(data):
    """Validate data quality."""
    try:
        if data is None or data.empty:
            return type('ValidationResult', (), {'is_valid': False, 'errors': ['Empty data']})()
        
        errors = []
        if data.isnull().sum().sum() > 0:
            errors.append('Missing values detected')
        
        if len(data) < 10:
            errors.append('Insufficient data')
        
        is_valid = len(errors) == 0
        return type('ValidationResult', (), {'is_valid': is_valid, 'errors': errors})()
    except Exception as e:
        # Log error but don't fail validation
        return type('ValidationResult', (), {'is_valid': False, 'errors': [str(e)]})()

