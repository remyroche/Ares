"""
Artifact Manager for Pre-Training Steps

This module provides a simplified artifact management system specifically designed
for the pre_training pipeline steps, with proper artifact creation and consumption
patterns.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.utils.enhanced_artifact_manager import EnhancedArtifactManager, ArtifactConfig, StorageBackend, CompressionType

# Setup logging
logger = logging.getLogger(__name__)

class PreTrainingArtifactManager:
    """Artifact manager specifically for pre-training steps."""
    
    def __init__(self, base_path: str = "artifacts/pre_training"):
        """Initialize the pre-training artifact manager."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize enhanced artifact manager
        config = ArtifactConfig(
            backend=StorageBackend.FILESYSTEM,
            base_path=str(self.base_path),
            compression_type=CompressionType.GZIP,
            max_cache_size_mb=512,
            enable_metrics=True
        )
        self.enhanced_manager = EnhancedArtifactManager(config)
        
        # Context for current execution
        self.context = {}
        
        logger.info(f"PreTrainingArtifactManager initialized at {self.base_path}")
    
    def set_context(self, symbol: str, exchange: str, timeframe: str, 
                   direction: str = "long", model: str = "Analyst", 
                   information: str = "pre_training"):
        """Set the execution context for artifact naming."""
        self.context = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'direction': direction,
            'model': model,
            'information': information,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        logger.info(f"Context set: {self.context}")
    
    def _build_key(self, step_name: str, artifact_name: str) -> str:
        """Build a unique key for an artifact."""
        if not self.context:
            raise ValueError("Context not set. Call set_context() first.")
        
        return f"{self.context['information']}_{step_name}_{artifact_name}_{self.context['symbol']}_{self.context['exchange']}_{self.context['timeframe']}_{self.context['timestamp']}"
    
    def save_artifact(self, step_name: str, artifact_name: str, data: Any, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save an artifact with proper naming and metadata."""
        try:
            key = self._build_key(step_name, artifact_name)
            
            # Prepare metadata
            artifact_metadata = {
                'step_name': step_name,
                'artifact_name': artifact_name,
                'data_type': type(data).__name__,
                'created_at': datetime.now().isoformat(),
                'context': self.context.copy(),
                **(metadata or {})
            }
            
            # Add data-specific metadata
            if isinstance(data, pd.DataFrame):
                artifact_metadata.update({
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'dtypes': data.dtypes.to_dict()
                })
            elif isinstance(data, pd.Series):
                artifact_metadata.update({
                    'shape': data.shape,
                    'name': data.name,
                    'dtype': str(data.dtype)
                })
            elif isinstance(data, np.ndarray):
                artifact_metadata.update({
                    'shape': data.shape,
                    'dtype': str(data.dtype)
                })
            
            # Store using enhanced manager
            result = self.enhanced_manager.store(key, data, artifact_metadata)
            
            if result.success:
                logger.info(f"Saved artifact: {step_name}/{artifact_name} ({artifact_metadata['data_type']})")
                return True
            else:
                logger.error(f"Failed to save artifact: {step_name}/{artifact_name} - {result.error}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving artifact {step_name}/{artifact_name}: {e}")
            return False
    
    def load_artifact(self, step_name: str, artifact_name: str, 
                     version: Optional[str] = None) -> Optional[Any]:
        """Load an artifact by step name and artifact name."""
        try:
            if version:
                # Load specific version
                key = f"{self.context['information']}_{step_name}_{artifact_name}_{self.context['symbol']}_{self.context['exchange']}_{self.context['timeframe']}_{version}"
            else:
                # Load latest version
                key = self._build_key(step_name, artifact_name)
            
            data = self.enhanced_manager.retrieve(key)
            
            if data is not None:
                logger.info(f"Loaded artifact: {step_name}/{artifact_name}")
                return data
            else:
                logger.warning(f"Artifact not found: {step_name}/{artifact_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading artifact {step_name}/{artifact_name}: {e}")
            return None
    
    def list_artifacts(self, step_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available artifacts, optionally filtered by step name."""
        try:
            all_artifacts = self.enhanced_manager.list_artifacts()
            
            artifacts = []
            for key in all_artifacts:
                # Parse key to extract information
                parts = key.split('_')
                if len(parts) >= 6:
                    info = parts[0]
                    step = parts[1]
                    artifact = parts[2]
                    
                    if step_name is None or step == step_name:
                        metadata = self.enhanced_manager.get_metadata(key)
                        artifacts.append({
                            'step_name': step,
                            'artifact_name': artifact,
                            'key': key,
                            'metadata': metadata
                        })
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Error listing artifacts: {e}")
            return []
    
    def get_artifact_metadata(self, step_name: str, artifact_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific artifact."""
        try:
            key = self._build_key(step_name, artifact_name)
            metadata = self.enhanced_manager.get_metadata(key)
            return metadata.__dict__ if metadata else None
        except Exception as e:
            logger.error(f"Error getting metadata for {step_name}/{artifact_name}: {e}")
            return None
    
    def save_step_artifacts(self, step_name: str, artifacts: Dict[str, Any], 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Save multiple artifacts for a step."""
        success = True
        for artifact_name, data in artifacts.items():
            if not self.save_artifact(step_name, artifact_name, data, metadata):
                success = False
        return success
    
    def load_step_artifacts(self, step_name: str, artifact_names: List[str]) -> Dict[str, Any]:
        """Load multiple artifacts for a step."""
        artifacts = {}
        for artifact_name in artifact_names:
            data = self.load_artifact(step_name, artifact_name)
            if data is not None:
                artifacts[artifact_name] = data
        return artifacts
    
    def cleanup_old_artifacts(self, days: int = 7) -> int:
        """Clean up artifacts older than specified days."""
        return self.enhanced_manager.cleanup_old_artifacts()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.enhanced_manager.get_metrics()
    
    def close(self):
        """Close the artifact manager."""
        self.enhanced_manager.close()

# Global instance
_artifact_manager = None

def get_pretraining_artifact_manager() -> PreTrainingArtifactManager:
    """Get the global pre-training artifact manager instance."""
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = PreTrainingArtifactManager()
    return _artifact_manager

def reset_artifact_manager():
    """Reset the global artifact manager (useful for testing)."""
    global _artifact_manager
    if _artifact_manager:
        _artifact_manager.close()
    _artifact_manager = None