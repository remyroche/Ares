"""
Centralized Artifact Manager for Market Analysis Pipeline.

This module provides centralized artifact management with consistent naming,
timestamps, and failure handling.

Enhanced with memory optimization and computational efficiency features.
"""

import gc
import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Handle optional dependencies gracefully
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

from src.training.common.artifact_persistence import SaveReport, persist_artifacts
from src.utils.logger import system_logger

class ArtifactManager:
    """
    Centralized artifact manager for market analysis pipeline.

    Ensures all artifacts are saved in the same folder with timestamps
    and proper failure handling.
    """

    def __init__(self, base_dir: str = "artifacts", symbol: str = "BTCUSDT", exchange: str = "binance", timeframe: str = "30m"):
        """
        Initialize the artifact manager.

        Args:
            base_dir: Base directory for all artifacts
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
        """
        self.logger = system_logger.getChild('ArtifactManager')
        self.symbol = symbol
        self.exchange = exchange
        self.timeframe = timeframe

        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Store base directory and artifact directory path (lazy creation)
        self.base_dir = Path(base_dir)
        self.artifact_dir = self.base_dir / f"{symbol}_{exchange}_{timeframe}_{self.session_timestamp}"

        # Memory optimization settings
        self.max_cache_size_mb = 256
        self.enable_compression = True
        self.compression_threshold_mb = 0.5
        self.enable_data_type_optimization = True
        self.enable_aggressive_cleanup = True
        self.cleanup_interval_seconds = 300
        
        # Initialize memory optimization components
        self._cache = OrderedDict()  # LRU cache
        self._cache_size_bytes = 0
        self._max_cache_size_bytes = self.max_cache_size_mb * 1024 * 1024
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_savings_mb': 0.0,
            'optimization_savings_mb': 0.0
        }

        self.logger.info(f"Artifact directory path prepared: {self.artifact_dir}")

    def get_artifact_path(self, component_name: str, artifact_type: str, extension: str = "json") -> Path:
        """
        Get standardized artifact path.

        Args:
            component_name: Name of the component
            artifact_type: Type of artifact
            extension: File extension

        Returns:
            Path to the artifact file
        """
        # Ensure artifact directory exists only when we actually need to save something
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{component_name}_{artifact_type}_{self.session_timestamp}.{extension}"
        return self.artifact_dir / filename

    async def save_artifacts(
        self,
        component_name: str,
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> SaveReport:
        """
        Save artifacts with proper error handling and timestamps.

        Args:
            component_name: Name of the component
            artifacts: Dictionary of artifacts to save
            metadata: Optional metadata to include

        Returns:
            Dictionary mapping artifact names to file paths

        Raises:
            Exception: If artifact saving fails
        """
        try:
            # Add metadata to artifacts
            if metadata is None:
                metadata = {}

            # Add session metadata
            session_metadata = {
                "session_timestamp": self.session_timestamp,
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "component_name": component_name,
                "save_timestamp": datetime.now().isoformat()
            }

            # Merge metadata
            full_metadata = {**session_metadata, **metadata}

            report = persist_artifacts(
                component_name=component_name,
                artifacts=artifacts,
                metadata=full_metadata,
                base_dir=self.artifact_dir,
                logger=self.logger,
                json_serializer=self._json_serializer,
            )

            self.logger.info(
                f"✅ All artifacts saved for {component_name} (correlation_id={report.correlation_id})"
            )
            return report

        except Exception as e:
            self.logger.error(f"❌ Failed to save artifacts for {component_name}: {e}")
            raise Exception(f"Artifact saving failed for {component_name}: {e}")

    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for complex objects.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        if NUMPY_AVAILABLE and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif NUMPY_AVAILABLE and isinstance(obj, np.integer):
            return int(obj)
        elif NUMPY_AVAILABLE and isinstance(obj, np.floating):
            return float(obj)
        elif PANDAS_AVAILABLE and isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Exception):
            return {
                "type": obj.__class__.__name__,
                "message": str(obj)
            }
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, dict):
            # Handle dictionaries with numpy int64 keys
            if NUMPY_AVAILABLE:
                new_dict = {}
                for key, value in obj.items():
                    # Convert numpy int64 keys to regular Python int
                    if isinstance(key, np.integer):
                        key = int(key)
                    elif isinstance(key, (str, int, float, bool)):
                        pass  # Keep as-is
                    else:
                        # Convert other non-JSON-serializable keys to string
                        key = str(key)

                    # Recursively handle nested structures
                    if isinstance(value, (dict, list, tuple)):
                        value = self._json_serializer(value)
                    new_dict[key] = value
                return new_dict
            return obj
        elif isinstance(obj, (list, tuple)):
            # Handle lists/tuples with numpy types
            if NUMPY_AVAILABLE:
                return [self._json_serializer(item) for item in obj]
            return list(obj)
        else:
            return str(obj)

    def get_artifact_summary(self) -> Dict[str, Any]:
        """
        Get summary of all artifacts in the session.

        Returns:
            Dictionary with artifact summary
        """
        try:
            summary = {
                "session_timestamp": self.session_timestamp,
                "artifact_directory": str(self.artifact_dir),
                "symbol": self.symbol,
                "exchange": self.exchange,
                "timeframe": self.timeframe,
                "total_files": 0,
                "components": {},
                "file_sizes": {},
                "directory_exists": self.artifact_dir.exists()
            }

            # Only count files if directory exists
            if self.artifact_dir.exists():
                # Count files and get sizes
                for file_path in self.artifact_dir.glob("*"):
                    if file_path.is_file():
                        summary["total_files"] += 1
                        summary["file_sizes"][file_path.name] = file_path.stat().st_size

                        # Group by component
                        component_name = file_path.name.split('_')[0]
                        if component_name not in summary["components"]:
                            summary["components"][component_name] = []
                        summary["components"][component_name].append(file_path.name)

            return summary

        except Exception as e:
            self.logger.error(f"Failed to get artifact summary: {e}")
            return {"error": str(e)}

    def cleanup_failed_artifacts(self, component_name: str) -> None:
        """
        Clean up artifacts from a failed component.

        Args:
            component_name: Name of the component that failed
        """
        try:
            # Only cleanup if directory exists
            if not self.artifact_dir.exists():
                self.logger.info(f"No artifacts to cleanup for {component_name} - directory doesn't exist")
                return

            pattern = f"{component_name}_*_{self.session_timestamp}.*"
            for file_path in self.artifact_dir.glob(pattern):
                file_path.unlink()
                self.logger.info(f"🗑️ Cleaned up failed artifact: {file_path.name}")

        except Exception as e:
            self.logger.error(f"Failed to cleanup artifacts for {component_name}: {e}")

    def validate_artifacts(self, component_name: str, required_artifacts: List[str]) -> bool:
        """
        Validate that all required artifacts exist and are non-empty.

        Args:
            component_name: Name of the component
            required_artifacts: List of required artifact names

        Returns:
            True if all artifacts are valid
        """
        try:
            # If directory doesn't exist, no artifacts to validate
            if not self.artifact_dir.exists():
                self.logger.error(f"Artifact directory doesn't exist: {self.artifact_dir}")
                return False

            for artifact_name in required_artifacts:
                # Check for any file with this artifact name
                pattern = f"{component_name}_{artifact_name}_*"
                matching_files = list(self.artifact_dir.glob(pattern))

                if not matching_files:
                    self.logger.error(f"Missing required artifact: {artifact_name}")
                    return False

                # Check file size
                for file_path in matching_files:
                    if file_path.stat().st_size == 0:
                        self.logger.error(f"Empty artifact file: {file_path.name}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Failed to validate artifacts for {component_name}: {e}")
            return False

    async def load_artifacts_from_previous_stage(self, previous_component_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """
        Load artifacts from a previous pipeline stage.

        Args:
            previous_component_name: Name of the previous component
            artifact_names: List of artifact names to load

        Returns:
            Dictionary of loaded artifacts
        """
        loaded_artifacts = {}

        try:
            # If directory doesn't exist, return empty dict
            if not self.artifact_dir.exists():
                self.logger.warning(f"Artifact directory doesn't exist: {self.artifact_dir}")
                return loaded_artifacts

            for artifact_name in artifact_names:
                try:
                    # Look for the most recent artifact file
                    pattern = f"{previous_component_name}_{artifact_name}_*"
                    matching_files = list(self.artifact_dir.glob(pattern))

                    if not matching_files:
                        self.logger.warning(f"No artifact found for {previous_component_name}_{artifact_name}")
                        continue

                    # Get the most recent file
                    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

                    # Load the artifact based on file extension
                    artifact_data = await self._load_single_artifact(latest_file)
                    loaded_artifacts[artifact_name] = artifact_data

                    self.logger.info(f"✅ Loaded artifact {artifact_name} from {latest_file.name}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to load artifact {artifact_name}: {e}")
                    continue

            return loaded_artifacts

        except Exception as e:
            self.logger.error(f"Failed to load artifacts from previous stage {previous_component_name}: {e}")
            return loaded_artifacts

    async def _load_single_artifact(self, file_path: Path) -> Any:
        """
        Load a single artifact from file.

        Args:
            file_path: Path to the artifact file

        Returns:
            Loaded artifact data
        """
        try:
            file_extension = file_path.suffix.lower()

            if file_extension == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # If it's a simple value wrapper, extract the actual value
                    if isinstance(data, dict) and 'value' in data and len(data) == 2:
                        return data['value']
                    return data

            elif file_extension == '.parquet' and PANDAS_AVAILABLE:
                return pd.read_parquet(file_path)

            elif file_extension == '.npy' and NUMPY_AVAILABLE:
                return np.load(file_path)

            else:
                # Try to load as JSON as fallback
                with open(file_path, 'r') as f:
                    return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load artifact from {file_path}: {e}")
            raise

    def find_latest_artifact_session(self, symbol: str, exchange: str, timeframe: str) -> Optional[str]:
        """
        Find the most recent artifact session for the given parameters.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe

        Returns:
            Session timestamp if found, None otherwise
        """
        try:
            if not self.base_dir.exists():
                return None

            # Look for directories matching the pattern
            pattern = f"{symbol}_{exchange}_{timeframe}_*"
            matching_dirs = [d for d in self.base_dir.glob(pattern) if d.is_dir()]

            if not matching_dirs:
                return None

            # Sort by creation time and get the latest
            latest_dir = max(matching_dirs, key=lambda d: d.stat().st_ctime)

            # Extract timestamp from directory name
            timestamp_part = latest_dir.name.split('_')[-2:]  # Get last two parts (date and time)
            return '_'.join(timestamp_part)

        except Exception as e:
            self.logger.error(f"Failed to find latest artifact session: {e}")
            return None

    def load_artifacts_from_latest_session(self, component_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """
        Load artifacts from the most recent session.

        Args:
            component_name: Name of the component
            artifact_names: List of artifact names to load

        Returns:
            Dictionary of loaded artifacts
        """
        loaded_artifacts = {}

        try:
            # Find the latest session
            latest_session = self.find_latest_artifact_session(
                self.symbol, self.exchange, self.timeframe
            )

            if not latest_session:
                self.logger.warning("No previous artifact session found")
                return loaded_artifacts

            # Create artifact directory path for the latest session
            latest_artifact_dir = self.base_dir / f"{self.symbol}_{self.exchange}_{self.timeframe}_{latest_session}"

            if not latest_artifact_dir.exists():
                self.logger.warning(f"Latest artifact directory doesn't exist: {latest_artifact_dir}")
                return loaded_artifacts

            # Load artifacts from the latest session
            for artifact_name in artifact_names:
                try:
                    pattern = f"{component_name}_{artifact_name}_*"
                    matching_files = list(latest_artifact_dir.glob(pattern))

                    if not matching_files:
                        self.logger.warning(f"No artifact found for {component_name}_{artifact_name}")
                        continue

                    # Get the most recent file
                    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

                    # Load the artifact
                    artifact_data = self._load_single_artifact_sync(latest_file)
                    loaded_artifacts[artifact_name] = artifact_data

                    self.logger.info(f"✅ Loaded artifact {artifact_name} from latest session: {latest_file.name}")

                except Exception as e:
                    self.logger.error(f"❌ Failed to load artifact {artifact_name}: {e}")
                    continue

            return loaded_artifacts

        except Exception as e:
            self.logger.error(f"Failed to load artifacts from latest session: {e}")
            return loaded_artifacts

    def _load_single_artifact_sync(self, file_path: Path) -> Any:
        """
        Synchronous version of _load_single_artifact for use in non-async contexts.

        Args:
            file_path: Path to the artifact file

        Returns:
            Loaded artifact data
        """
        try:
            file_extension = file_path.suffix.lower()

            if file_extension == '.json':
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # If it's a simple value wrapper, extract the actual value
                    if isinstance(data, dict) and 'value' in data and len(data) == 2:
                        return data['value']
                    return data

            elif file_extension == '.parquet' and PANDAS_AVAILABLE:
                return pd.read_parquet(file_path)

            elif file_extension == '.npy' and NUMPY_AVAILABLE:
                return np.load(file_path)

            else:
                # Try to load as JSON as fallback
                with open(file_path, 'r') as f:
                    return json.load(f)

        except Exception as e:
            self.logger.error(f"Failed to load artifact from {file_path}: {e}")
            raise
