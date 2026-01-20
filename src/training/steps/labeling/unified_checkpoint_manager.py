"""
Unified Checkpoint Manager for Layer 2.5, Layer 3, and Layer 4

Provides a unified interface for managing checkpoints across all layers,
with automatic layer detection and appropriate checkpoint manager instantiation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

from .layer2_checkpoint_manager import Layer2CheckpointManager
from .layer25_checkpoint_manager import Layer25CheckpointManager  
from .layer3_checkpoint_manager import Layer3CheckpointManager
from .layer4_checkpoint_manager import Layer4CheckpointManager

logger = logging.getLogger(__name__)

class UnifiedCheckpointManager:
    """
    Unified checkpoint manager that automatically detects the layer and provides
    the appropriate checkpoint manager instance.
    
    Supports:
    - Layer 2: Full meta-labeling pipeline
    - Layer 2.5: Chaser residual learning models
    - Layer 3: Multi-horizon meta-models
    - Layer 4: Final gate models
    """
    
    LAYER_MAPPING = {
        'layer2': Layer2CheckpointManager,
        'layer2.5': Layer25CheckpointManager,
        'layer25': Layer25CheckpointManager,
        'layer3': Layer3CheckpointManager,
        'layer4': Layer4CheckpointManager
    }
    
    def __init__(self, layer: str, symbol: str, checkpoint_dir: Optional[Path] = None):
        """
        Initialize unified checkpoint manager for a specific layer.
        
        Args:
            layer: Layer name ('layer2', 'layer2.5', 'layer25', 'layer3', 'layer4')
            symbol: Trading symbol
            checkpoint_dir: Optional custom checkpoint directory
        """
        self.layer = self._normalize_layer_name(layer)
        self.symbol = symbol.upper()
        self.checkpoint_dir = checkpoint_dir
        
        # Get appropriate checkpoint manager
        manager_class = self.LAYER_MAPPING.get(self.layer)
        if manager_class is None:
            raise ValueError(f"Unsupported layer: {layer}. Supported layers: {list(self.LAYER_MAPPING.keys())}")
        
        self.manager = manager_class(checkpoint_dir)
        logger.info(f"🔧 Initialized unified checkpoint manager for {self.layer} ({self.symbol})")
    
    def _normalize_layer_name(self, layer: str) -> str:
        """Normalize layer name to standard format."""
        layer = layer.lower().replace(' ', '').replace('_', '')
        if layer == 'layer25':
            return 'layer25'
        return layer
    
    def save_checkpoint(self, step: str, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save a checkpoint for the current layer.
        
        Args:
            step: Name of the sub-step
            data: Dictionary of data to checkpoint
            config: Optional pipeline configuration
            
        Returns:
            Path to the saved checkpoint
        """
        return self.manager.save_checkpoint(step, data, self.symbol, config)
    
    def load_checkpoint(self, step: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for the current layer.
        
        Args:
            step: Name of the sub-step
            
        Returns:
            Dictionary of checkpoint data, or None if not found
        """
        return self.manager.load_checkpoint(step, self.symbol)
    
    def delete_checkpoints_from(self, step: str) -> int:
        """
        Delete checkpoints from a specific step onwards.
        
        Args:
            step: Starting step to delete from (inclusive)
            
        Returns:
            Number of checkpoints deleted
        """
        return self.manager.delete_checkpoints_from(step, self.symbol)
    
    def get_latest_checkpoint(self) -> Optional[Tuple[str, Any]]:
        """
        Get the latest checkpoint for the current layer.
        
        Returns:
            Tuple of (step_name, metadata) or None if no checkpoints exist
        """
        return self.manager.get_latest_checkpoint(self.symbol)
    
    def list_checkpoints(self) -> List[Tuple[str, Any]]:
        """
        List all available checkpoints for the current layer.
        
        Returns:
            List of (step_name, metadata) tuples, ordered by step index
        """
        return self.manager.list_checkpoints(self.symbol)
    
    def get_auto_resume_step(self) -> str:
        """
        Automatically determine the best step to resume execution from.
        
        Returns:
            Name of the step to start/resume execution from
        """
        return self.manager.get_auto_resume_step(self.symbol)
    
    def get_available_steps(self) -> List[str]:
        """
        Get the list of available sub-steps for the current layer.
        
        Returns:
            List of step names in execution order
        """
        return self.manager.SUBSTEPS
    
    def get_step_index(self, step: str) -> int:
        """
        Get the index of a step in the pipeline.
        
        Args:
            step: Name of the sub-step
            
        Returns:
            Step index in the pipeline
        """
        return self.manager._get_step_index(step)
    
    def validate_checkpoint_data(self, step: str, data: Dict[str, Any]) -> None:
        """
        Validate checkpoint data before saving.
        
        Args:
            step: Name of the sub-step
            data: Dictionary of data to checkpoint
        """
        return self.manager.validate_checkpoint_data(step, data)

def get_checkpoint_manager(layer: str, symbol: str, checkpoint_dir: Optional[Path] = None) -> UnifiedCheckpointManager:
    """
    Get a unified checkpoint manager instance for any layer.
    
    Args:
        layer: Layer name ('layer2', 'layer2.5', 'layer25', 'layer3', 'layer4')
        symbol: Trading symbol
        checkpoint_dir: Optional custom checkpoint directory
        
    Returns:
        UnifiedCheckpointManager instance
    """
    return UnifiedCheckpointManager(layer, symbol, checkpoint_dir)

def get_all_checkpoint_managers(symbol: str, checkpoint_dir: Optional[Path] = None) -> Dict[str, UnifiedCheckpointManager]:
    """
    Get checkpoint managers for all layers.
    
    Args:
        symbol: Trading symbol
        checkpoint_dir: Optional custom checkpoint directory
        
    Returns:
        Dictionary of layer_name -> UnifiedCheckpointManager
    """
    managers = {}
    for layer_name in ['layer2', 'layer25', 'layer3', 'layer4']:
        try:
            managers[layer_name] = get_checkpoint_manager(layer_name, symbol, checkpoint_dir)
        except Exception as e:
            logger.warning(f"⚠️ Failed to create checkpoint manager for {layer_name}: {e}")
    
    return managers

def auto_resume_pipeline(layer: str, symbol: str, checkpoint_dir: Optional[Path] = None) -> Tuple[str, UnifiedCheckpointManager]:
    """
    Auto-resume a pipeline from the latest checkpoint.
    
    Args:
        layer: Layer name
        symbol: Trading symbol
        checkpoint_dir: Optional custom checkpoint directory
        
    Returns:
        Tuple of (resume_step, checkpoint_manager)
    """
    manager = get_checkpoint_manager(layer, symbol, checkpoint_dir)
    resume_step = manager.get_auto_resume_step()
    
    logger.info(f"🔄 Auto-resuming {layer} for {symbol} from step: {resume_step}")
    return resume_step, manager

# Add class method to UnifiedCheckpointManager
UnifiedCheckpointManager.auto_resume_pipeline = staticmethod(auto_resume_pipeline)

# Convenience functions for each layer
def get_layer2_checkpoint_manager(symbol: str, checkpoint_dir: Optional[Path] = None) -> Layer2CheckpointManager:
    """Get Layer 2 checkpoint manager."""
    return Layer2CheckpointManager(checkpoint_dir)

def get_layer25_checkpoint_manager(symbol: str, checkpoint_dir: Optional[Path] = None) -> Layer25CheckpointManager:
    """Get Layer 2.5 checkpoint manager."""
    return Layer25CheckpointManager(checkpoint_dir)

def get_layer3_checkpoint_manager(symbol: str, checkpoint_dir: Optional[Path] = None) -> Layer3CheckpointManager:
    """Get Layer 3 checkpoint manager."""
    return Layer3CheckpointManager(checkpoint_dir)

def get_layer4_checkpoint_manager(symbol: str, checkpoint_dir: Optional[Path] = None) -> Layer4CheckpointManager:
    """Get Layer 4 checkpoint manager."""
    return Layer4CheckpointManager(checkpoint_dir)
