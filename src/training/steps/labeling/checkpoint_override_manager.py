"""
Checkpoint Override Manager

Allows users to override checkpoint usage by restarting from any specific stage
and automatically replaces existing checkpoints from that point onwards.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .unified_checkpoint_manager import get_checkpoint_manager
from .checkpoint_aware_runner import CheckpointAwareRunner

logger = logging.getLogger(__name__)

@dataclass
class OverrideConfig:
    """Configuration for checkpoint override."""
    layer: str
    symbol: str
    override_step: str
    force_restart: bool = False
    keep_earlier_checkpoints: bool = False
    checkpoint_dir: Optional[Path] = None

class CheckpointOverrideManager:
    """
    Manages checkpoint override functionality.
    
    Allows users to:
    1. Restart from any specific stage
    2. Force restart from beginning
    3. Replace checkpoints from override point onwards
    4. Optionally preserve earlier checkpoints
    """
    
    def __init__(self, layer: str, symbol: str, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint override manager.
        
        Args:
            layer: Layer name ('layer2', 'layer25', 'layer3', 'layer4')
            symbol: Trading symbol
            checkpoint_dir: Optional custom checkpoint directory
        """
        self.layer = layer.lower()
        self.symbol = symbol.upper()
        self.checkpoint_dir = checkpoint_dir
        
        # Get checkpoint manager
        self.manager = get_checkpoint_manager(self.layer, self.symbol, checkpoint_dir)
        
        logger.info(f"🔧 Initialized checkpoint override manager for {self.layer} ({self.symbol})")
    
    def get_available_override_steps(self) -> List[str]:
        """
        Get list of available steps that can be used as override points.
        
        Returns:
            List of step names in execution order
        """
        return self.manager.get_available_steps()
    
    def validate_override_step(self, override_step: str) -> bool:
        """
        Validate that the override step is valid for this layer.
        
        Args:
            override_step: Step name to override from
            
        Returns:
            True if valid, False otherwise
        """
        available_steps = self.get_available_override_steps()
        is_valid = override_step in available_steps
        
        if not is_valid:
            logger.error(f"❌ Invalid override step '{override_step}'. Valid steps: {available_steps}")
        else:
            logger.info(f"✅ Valid override step: {override_step}")
        
        return is_valid
    
    def get_override_plan(self, override_config: OverrideConfig) -> Dict[str, Any]:
        """
        Get detailed plan for checkpoint override.
        
        Args:
            override_config: Override configuration
            
        Returns:
            Dictionary with override plan details
        """
        # Validate override step
        if not self.validate_override_step(override_config.override_step):
            raise ValueError(f"Invalid override step: {override_config.override_step}")
        
        # Get current checkpoint status
        current_checkpoints = self.manager.list_checkpoints()
        available_steps = self.get_available_override_steps()
        
        # Determine override index
        override_idx = available_steps.index(override_config.override_step)
        
        # Determine what will be deleted
        steps_to_delete = []
        if override_config.force_restart:
            # Delete all checkpoints
            steps_to_delete = available_steps
        else:
            # Delete from override step onwards
            steps_to_delete = available_steps[override_idx:]
        
        # Determine what will be kept
        steps_to_keep = []
        if not override_config.force_restart and override_config.keep_earlier_checkpoints:
            # Keep steps before override point
            steps_to_keep = available_steps[:override_idx]
        
        # Check which checkpoints actually exist
        existing_checkpoints = {step: metadata for step, metadata in current_checkpoints}
        existing_to_delete = [step for step in steps_to_delete if step in existing_checkpoints]
        existing_to_keep = [step for step in steps_to_keep if step in existing_checkpoints]
        
        plan = {
            'layer': self.layer,
            'symbol': self.symbol,
            'override_step': override_config.override_step,
            'override_index': override_idx,
            'force_restart': override_config.force_restart,
            'keep_earlier_checkpoints': override_config.keep_earlier_checkpoints,
            'total_steps': len(available_steps),
            'steps_to_execute': available_steps[override_idx:],
            'steps_to_delete': steps_to_delete,
            'steps_to_keep': steps_to_keep,
            'existing_checkpoints': len(existing_checkpoints),
            'existing_to_delete': existing_to_delete,
            'existing_to_keep': existing_to_keep,
            'checkpoints_to_be_removed': len(existing_to_delete),
            'checkpoints_to_be_preserved': len(existing_to_keep)
        }
        
        return plan
    
    def execute_override(self, override_config: OverrideConfig) -> Dict[str, Any]:
        """
        Execute checkpoint override.
        
        Args:
            override_config: Override configuration
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"🔄 Executing checkpoint override for {self.layer} ({self.symbol})")
        
        # Get override plan
        plan = self.get_override_plan(override_config)
        
        logger.info(f"📋 Override plan:")
        logger.info(f"   Override step: {plan['override_step']} (index {plan['override_index']})")
        logger.info(f"   Force restart: {plan['force_restart']}")
        logger.info(f"   Keep earlier checkpoints: {plan['keep_earlier_checkpoints']}")
        logger.info(f"   Steps to execute: {len(plan['steps_to_execute'])}")
        logger.info(f"   Checkpoints to remove: {plan['checkpoints_to_be_removed']}")
        logger.info(f"   Checkpoints to preserve: {plan['checkpoints_to_be_preserved']}")
        
        # Execute checkpoint deletion
        deleted_count = 0
        if plan['checkpoints_to_be_removed'] > 0:
            if override_config.force_restart:
                # Delete all checkpoints
                logger.info(f"🗑️ Deleting all checkpoints (force restart)")
                deleted_count = self.manager.delete_checkpoints_from(plan['steps_to_delete'][0])
            else:
                # Delete from override step onwards
                logger.info(f"🗑️ Deleting checkpoints from '{plan['override_step']}' onwards")
                deleted_count = self.manager.delete_checkpoints_from(plan['override_step'])
        
        # Create checkpoint-aware runner with override
        runner = CheckpointAwareRunner(
            self.layer, 
            self.symbol, 
            self.checkpoint_dir
        )
        
        # Force the runner to start from the override step
        runner.execution_plan.resume_step = override_config.override_step
        runner.execution_plan.start_from_beginning = override_config.force_restart
        
        # Rebuild execution order from override step
        available_steps = runner.get_available_steps()
        override_idx = available_steps.index(override_config.override_step)
        runner.execution_plan.execution_order = available_steps[override_idx:]
        
        results = {
            'plan': plan,
            'deleted_checkpoints': deleted_count,
            'runner': runner,
            'execution_plan': runner.execution_plan,
            'status': 'override_executed'
        }
        
        logger.info(f"✅ Checkpoint override executed successfully")
        logger.info(f"   Deleted {deleted_count} checkpoints")
        logger.info(f"   Ready to execute from '{override_config.override_step}'")
        
        return results
    
    def create_override_runner(
        self, 
        override_step: str,
        force_restart: bool = False,
        keep_earlier_checkpoints: bool = False
    ) -> CheckpointAwareRunner:
        """
        Create a checkpoint-aware runner with override configuration.
        
        Args:
            override_step: Step to override from
            force_restart: Force restart from beginning
            keep_earlier_checkpoints: Keep checkpoints before override step
            
        Returns:
            Configured CheckpointAwareRunner
        """
        override_config = OverrideConfig(
            layer=self.layer,
            symbol=self.symbol,
            override_step=override_step,
            force_restart=force_restart,
            keep_earlier_checkpoints=keep_earlier_checkpoints,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Execute override
        results = self.execute_override(override_config)
        
        return results['runner']

def create_checkpoint_override(
    layer: str,
    symbol: str,
    override_step: str,
    force_restart: bool = False,
    keep_earlier_checkpoints: bool = False,
    checkpoint_dir: Optional[Path] = None
) -> CheckpointAwareRunner:
    """
    Convenience function to create checkpoint override.
    
    Args:
        layer: Layer name ('layer2', 'layer25', 'layer3', 'layer4')
        symbol: Trading symbol
        override_step: Step to override from
        force_restart: Force restart from beginning
        keep_earlier_checkpoints: Keep checkpoints before override step
        checkpoint_dir: Optional checkpoint directory
        
    Returns:
        Configured CheckpointAwareRunner ready for execution
    """
    override_manager = CheckpointOverrideManager(layer, symbol, checkpoint_dir)
    return override_manager.create_override_runner(
        override_step=override_step,
        force_restart=force_restart,
        keep_earlier_checkpoints=keep_earlier_checkpoints
    )

def list_override_options(layer: str, symbol: str, checkpoint_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    List available override options for a layer and symbol.
    
    Args:
        layer: Layer name
        symbol: Trading symbol
        checkpoint_dir: Optional checkpoint directory
        
    Returns:
        Dictionary with override options
    """
    override_manager = CheckpointOverrideManager(layer, symbol, checkpoint_dir)
    
    available_steps = override_manager.get_available_override_steps()
    current_checkpoints = override_manager.manager.list_checkpoints()
    
    options = {
        'layer': layer,
        'symbol': symbol,
        'available_steps': available_steps,
        'current_checkpoints': {step: metadata.timestamp for step, metadata in current_checkpoints},
        'total_steps': len(available_steps),
        'completed_steps': len(current_checkpoints),
        'completion_percentage': (len(current_checkpoints) / len(available_steps)) * 100 if available_steps else 0
    }
    
    return options

# Decorator for checkpoint override
def checkpoint_override_from(step: str, force_restart: bool = False, keep_earlier: bool = False):
    """
    Decorator to apply checkpoint override to a function.
    
    Usage:
    @checkpoint_override_from('dual_head_training', force_restart=True)
    def run_layer3_training(...):
        # Training logic here
        pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract layer and symbol from kwargs or function
            layer = kwargs.get('layer', 'layer3')
            symbol = kwargs.get('symbol', 'ETHUSDT')
            
            # Create override runner
            runner = create_checkpoint_override(
                layer=layer,
                symbol=symbol,
                override_step=step,
                force_restart=force_restart,
                keep_earlier_checkpoints=keep_earlier
            )
            
            # Add runner to kwargs
            kwargs['checkpoint_runner'] = runner
            
            # Execute function
            return func(*args, **kwargs)
        return wrapper
    return decorator
