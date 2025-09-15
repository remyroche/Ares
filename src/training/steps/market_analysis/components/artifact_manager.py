"""
Centralized Artifact Manager for Market Analysis Pipeline.

This module provides centralized artifact management with consistent naming,
timestamps, and failure handling.
"""

import json
import os
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
        
        # Create base artifact directory
        self.base_dir = Path(base_dir)
        self.artifact_dir = self.base_dir / f"{symbol}_{exchange}_{timeframe}_{self.session_timestamp}"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Artifact directory created: {self.artifact_dir}")
    
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
        filename = f"{component_name}_{artifact_type}_{self.session_timestamp}.{extension}"
        return self.artifact_dir / filename
    
    async def save_artifacts(
        self, 
        component_name: str, 
        artifacts: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
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
        saved_files = {}
        
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
            
            # Save each artifact
            for artifact_name, artifact_data in artifacts.items():
                try:
                    file_path = await self._save_single_artifact(
                        component_name, artifact_name, artifact_data, full_metadata
                    )
                    saved_files[artifact_name] = str(file_path)
                    self.logger.info(f"✅ Saved {artifact_name} to {file_path}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to save artifact {artifact_name}: {e}")
                    raise Exception(f"Failed to save artifact {artifact_name}: {e}")
            
            # Save consolidated metadata file
            metadata_path = self.get_artifact_path(component_name, "metadata", "json")
            with open(metadata_path, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=self._json_serializer)
            saved_files["metadata"] = str(metadata_path)
            
            self.logger.info(f"✅ All artifacts saved for {component_name}")
            return saved_files
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save artifacts for {component_name}: {e}")
            raise Exception(f"Artifact saving failed for {component_name}: {e}")
    
    async def _save_single_artifact(
        self, 
        component_name: str, 
        artifact_name: str, 
        artifact_data: Any,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save a single artifact with appropriate format.
        
        Args:
            component_name: Name of the component
            artifact_name: Name of the artifact
            artifact_data: Data to save
            metadata: Metadata to include
            
        Returns:
            Path to the saved file
        """
        # Determine file format based on data type
        if PANDAS_AVAILABLE and isinstance(artifact_data, pd.DataFrame):
            file_path = self.get_artifact_path(component_name, artifact_name, "parquet")
            artifact_data.to_parquet(file_path)
            
        elif isinstance(artifact_data, (list, dict)) and len(str(artifact_data)) > 1000:
            # Large data structures - save as JSON
            file_path = self.get_artifact_path(component_name, artifact_name, "json")
            with open(file_path, 'w') as f:
                json.dump(artifact_data, f, indent=2, default=self._json_serializer)
                
        elif NUMPY_AVAILABLE and isinstance(artifact_data, np.ndarray):
            file_path = self.get_artifact_path(component_name, artifact_name, "npy")
            np.save(file_path, artifact_data)
            
        elif isinstance(artifact_data, (str, int, float, bool)):
            # Simple values - save as JSON
            file_path = self.get_artifact_path(component_name, artifact_name, "json")
            with open(file_path, 'w') as f:
                json.dump({
                    "value": artifact_data,
                    "metadata": metadata
                }, f, indent=2, default=self._json_serializer)
                
        else:
            # Default to JSON
            file_path = self.get_artifact_path(component_name, artifact_name, "json")
            with open(file_path, 'w') as f:
                json.dump(artifact_data, f, indent=2, default=self._json_serializer)
        
        return file_path
    
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
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
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
                "file_sizes": {}
            }
            
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