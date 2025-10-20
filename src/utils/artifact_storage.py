"""Artifact Storage Module.

Handles file I/O operations for artifacts with support for multiple formats.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

# Optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .logger import system_logger


class ArtifactStorage:
    """Handles file I/O operations for artifacts."""
    
    def __init__(self, base_dir: Path):
        """Initialize artifact storage.
        
        Args:
            base_dir: Base directory for artifact storage
        """
        self.base_dir = base_dir
        self.logger = system_logger.getChild("ArtifactStorage")
        
    def save_artifact(self, data: Any, file_path: Path, 
                     artifact_type: str = "data",
                     metadata: Optional[Dict] = None) -> bool:
        """Save an artifact to file.
        
        Args:
            data: Data to save
            file_path: Path where to save the artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save based on file extension
            if file_path.suffix == '.parquet' and PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                data.to_parquet(file_path, compression='snappy')
            elif file_path.suffix == '.csv' and PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=True)
            elif file_path.suffix == '.json':
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            elif file_path.suffix == '.npy' and NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                np.save(file_path, data)
            else:
                # Default to pickle
                with open(file_path.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                file_path = file_path.with_suffix('.pkl')
            
            # Save metadata if provided
            if metadata:
                metadata_path = file_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            self.logger.debug(f"Saved artifact to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save artifact to {file_path}: {e}")
            return False
    
    def load_artifact(self, file_path: Path) -> Optional[Any]:
        """Load an artifact from file.
        
        Args:
            file_path: Path to the artifact file
            
        Returns:
            Loaded data or None if failed
        """
        try:
            if not file_path.exists():
                self.logger.warning(f"Artifact file not found: {file_path}")
                return None
            
            # Load based on file extension
            if file_path.suffix == '.parquet' and PANDAS_AVAILABLE:
                return pd.read_parquet(file_path)
            elif file_path.suffix == '.csv' and PANDAS_AVAILABLE:
                return pd.read_csv(file_path, index_col=0)
            elif file_path.suffix == '.json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            elif file_path.suffix == '.npy' and NUMPY_AVAILABLE:
                return np.load(file_path)
            else:
                # Try to load as pickle as fallback
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to load artifact from {file_path}: {e}")
            return None
    
    def load_metadata(self, file_path: Path) -> Optional[Dict]:
        """Load metadata for an artifact.
        
        Args:
            file_path: Path to the artifact file
            
        Returns:
            Metadata dictionary or None if not found
        """
        try:
            metadata_path = file_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
            return None
    
    def delete_artifact(self, file_path: Path) -> bool:
        """Delete an artifact file.
        
        Args:
            file_path: Path to the artifact file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if file_path.exists():
                file_path.unlink()
                
            # Also delete metadata file
            metadata_path = file_path.with_suffix('.metadata.json')
            if metadata_path.exists():
                metadata_path.unlink()
                
            self.logger.debug(f"Deleted artifact {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete artifact {file_path}: {e}")
            return False
    
    def list_artifacts(self, pattern: str = "*") -> list[Path]:
        """List artifacts matching a pattern.
        
        Args:
            pattern: Glob pattern to match
            
        Returns:
            List of matching artifact paths
        """
        try:
            return list(self.base_dir.glob(pattern))
        except Exception as e:
            self.logger.error(f"Failed to list artifacts with pattern {pattern}: {e}")
            return []